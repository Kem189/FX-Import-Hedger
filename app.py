import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

st.set_page_config(
    page_title="QuantTrade FX & Freight Risk Engine",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Stochastic FX, Freight & FEC Hedging Simulator")
st.markdown(
    "Coupled **Merton Jump-Diffusion** currency modeling, **Log-Normal port delay/shutdown** physics, "
    "and **Mean-Variance Optimal Forward Exchange Contract (FEC)** solver."
)

PRESETS = {
    "South Africa (Durban / USD-ZAR)": {
        "ticker": "USDZAR=X",
        "currency": "ZAR",
        "r_dom": 8.00,
        "r_for": 4.50,
        "selling_price": 2450000.0,
        "duty_pct": 15.0,
        "exp_days": 35,
        "free_days": 40,
    },
    "Brazil (Santos / USD-BRL)": {
        "ticker": "USDBRL=X",
        "currency": "BRL",
        "r_dom": 11.25,
        "r_for": 4.50,
        "selling_price": 820000.0,
        "duty_pct": 18.0,
        "exp_days": 32,
        "free_days": 38,
    },
    "Mexico (Manzanillo / USD-MXN)": {
        "ticker": "USDMXN=X",
        "currency": "MXN",
        "r_dom": 10.00,
        "r_for": 4.50,
        "selling_price": 2850000.0,
        "duty_pct": 10.0,
        "exp_days": 24,
        "free_days": 30,
    },
    "Colombia (Cartagena / USD-COP)": {
        "ticker": "USDCOP=X",
        "currency": "COP",
        "r_dom": 9.75,
        "r_for": 4.50,
        "selling_price": 590000000.0,
        "duty_pct": 12.0,
        "exp_days": 28,
        "free_days": 35,
    },
}

@st.cache_data(ttl=3600)
def fetch_fx_stats(ticker):
    try:
        data = yf.download(ticker, period="2y", progress=False)
        close = data["Close"].squeeze().dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        vol = float(log_ret.std() * np.sqrt(252))
        spot = float(close.iloc[-1])
        return spot, vol, True
    except Exception:
        fallbacks = {
            "USDZAR=X": 18.20,
            "USDBRL=X": 5.75,
            "USDMXN=X": 19.80,
            "USDCOP=X": 4350.0,
        }
        return fallbacks.get(ticker, 18.0), 0.14, False

st.sidebar.header("1. Corridor & Currency Preset")
region_name = st.sidebar.selectbox("Select Trade Corridor", list(PRESETS.keys()))
cfg = PRESETS[region_name]
ccy = cfg["currency"]

live_spot, live_vol, fetched_ok = fetch_fx_stats(cfg["ticker"])
if not fetched_ok:
    st.sidebar.warning("Using fallback spot/volatility (live feed unavailable).")

spot_rate = st.sidebar.number_input(f"Current Spot Rate (USD/{ccy})", value=round(live_spot, 4), format="%.4f")
annual_vol = st.sidebar.slider("Annualized FX Diffusion Volatility (%)", 5.0, 40.0, round(live_vol * 100, 1)) / 100.0
r_domestic = st.sidebar.number_input(f"Domestic Risk-Free Rate ({ccy} %)", value=cfg["r_dom"], step=0.25) / 100.0
r_foreign = st.sidebar.number_input("Foreign Risk-Free Rate (USD %)", value=cfg["r_for"], step=0.25) / 100.0

st.sidebar.header("2. Shipment & Commercials")
invoice_usd = st.sidebar.number_input("Supplier Invoice (USD)", value=100000.0, step=5000.0)
freight_usd = st.sidebar.number_input("Base Ocean Freight (USD)", value=8000.0, step=500.0)
import_duty_pct = st.sidebar.slider("Import Duty & Tariffs (%)", 0.0, 45.0, cfg["duty_pct"]) / 100.0
local_selling_price = st.sidebar.number_input(f"Contracted Local Revenue ({ccy})", value=cfg["selling_price"], step=10000.0)

st.sidebar.header("3. Port & Crisis Stress-Test")
expected_days = st.sidebar.slider("Expected Transit + Customs (Days)", 10, 90, cfg["exp_days"])
delay_std_days = st.sidebar.slider("Normal Delay Std Dev (Days)", 2, 25, 8)
free_port_days = st.sidebar.slider("Port Demurrage Grace Period (Days)", 15, 90, cfg["free_days"])
demurrage_per_day_usd = st.sidebar.number_input("Daily Demurrage Penalty (USD/day)", value=250.0, step=50.0)

st.sidebar.subheader("Emergency Shock Parameters")
jump_lambda = st.sidebar.slider("FX Jump Frequency (Shocks / Year)", 0.0, 6.0, 1.5, step=0.5)
jump_mean = st.sidebar.slider("Avg FX Jump Size (% Depreciation)", 0.0, 20.0, 6.0, step=1.0) / 100.0
shutdown_prob = st.sidebar.slider("Port Shutdown / Strike Probability (%)", 0, 80, 20, step=5) / 100.0
shutdown_delay_days = st.sidebar.slider("Extra Delay if Shutdown Hits (Days)", 5, 60, 25, step=5)
risk_aversion_gamma = st.sidebar.slider("CFO Risk Aversion (Gamma)", 0.02, 0.50, 0.15, step=0.01)

def run_full_model(n_sims=20000):
    np.random.seed(42)
    jump_std = 0.04
    emergency_reroute_usd = 4500.0

    phi = np.sqrt(delay_std_days**2 + expected_days**2)
    mu_log = np.log(expected_days**2 / phi)
    sigma_log = np.sqrt(np.log(phi**2 / expected_days**2))
    base_days = np.random.lognormal(mu_log, sigma_log, n_sims)

    shutdown_triggered = np.random.binomial(1, shutdown_prob, n_sims)
    simulated_days = base_days + (shutdown_triggered * shutdown_delay_days)
    T = simulated_days / 365.0

    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    drift = (r_domestic - r_foreign - jump_lambda * kappa - 0.5 * annual_vol**2) * T
    diffusion = annual_vol * np.sqrt(T) * np.random.normal(0, 1, n_sims)
    num_jumps = np.random.poisson(jump_lambda * T)
    jump_component = np.random.normal(num_jumps * jump_mean, np.sqrt(num_jumps) * jump_std)

    simulated_fx = spot_rate * np.exp(drift + diffusion + jump_component)

    T_0 = expected_days / 365.0
    forward_rate = spot_rate * np.exp((r_domestic - r_foreign) * T_0)

    base_cif_usd = invoice_usd + freight_usd
    overdue_days = np.maximum(0, simulated_days - free_port_days)
    variable_usd = (overdue_days * demurrage_per_day_usd) + (shutdown_triggered * emergency_reroute_usd)

    def margins_for_hedge(h):
        hedged_local = (h * base_cif_usd) * forward_rate
        unhedged_local = ((1.0 - h) * base_cif_usd + variable_usd) * simulated_fx
        total_landed = (hedged_local + unhedged_local) * (1.0 + import_duty_pct)
        return ((local_selling_price - total_landed) / local_selling_price) * 100.0

    res = minimize_scalar(
        lambda h: -(np.mean(margins_for_hedge(h)) - 0.5 * risk_aversion_gamma * np.var(margins_for_hedge(h))),
        bounds=(0.0, 1.0),
        method="bounded",
    )
    opt_h = float(res.x)

    h_grid = np.linspace(0, 1, 40)
    exp_m, cvar_95, loss_p = [], [], []
    for h_val in h_grid:
        m = margins_for_hedge(h_val)
        v95 = np.percentile(m, 5)
        exp_m.append(np.mean(m))
        cvar_95.append(np.mean(m[m <= v95]))
        loss_p.append(np.mean(m < 0) * 100.0)

    return {
        "forward_rate": forward_rate,
        "opt_h": opt_h,
        "m_unhedged": margins_for_hedge(0.0),
        "m_optimal": margins_for_hedge(opt_h),
        "m_full": margins_for_hedge(1.0),
        "h_grid": h_grid * 100.0,
        "exp_m": exp_m,
        "cvar_95": cvar_95,
        "loss_p": loss_p,
    }

out = run_full_model()

def calc_stats(arr):
    v95 = np.percentile(arr, 5)
    return np.mean(arr), v95, np.mean(arr[arr <= v95]), np.mean(arr < 0) * 100.0

u_exp, u_var, u_cvar, u_loss = calc_stats(out["m_unhedged"])
o_exp, o_var, o_cvar, o_loss = calc_stats(out["m_optimal"])
f_exp, f_var, f_cvar, f_loss = calc_stats(out["m_full"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bank Forward Rate", f"{out['forward_rate']:.4f} {ccy}", f"+{(out['forward_rate']/spot_rate - 1)*100:.2f}% Carry")
col2.metric("Optimal FEC Hedge (h*)", f"{out['opt_h']*100:.1f}%", f"Lock ${out['opt_h']*(invoice_usd+freight_usd):,.0f}")
col3.metric("Unhedged Loss Probability", f"{u_loss:.1f}%", f"Tail CVaR: {u_cvar:.1f}%", delta_color="inverse")
col4.metric("Optimal Hedge Loss Prob", f"{o_loss:.1f}%", f"Tail CVaR: {o_cvar:.1f}%", delta_color="normal")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.hist(out["m_unhedged"], bins=60, alpha=0.5, color="#d62728", density=True, label="0% Hedged (Spot Exposure)")
ax1.hist(out["m_optimal"], bins=60, alpha=0.65, color="#1f77b4", density=True, label=f"Optimal FEC ({out['opt_h']*100:.0f}%)")
ax1.axvline(0, color="black", linestyle="--", linewidth=2, label="Break-Even (0%)")
ax1.set_title("Net Margin Distribution (20,000 Crisis Simulations)", fontweight="bold")
ax1.set_xlabel("Net Profit Margin (%)")
ax1.set_ylabel("Probability Density")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(out["h_grid"], out["exp_m"], color="green", linewidth=2.5, label="Expected Net Margin (%)")
ax2.plot(out["h_grid"], out["cvar_95"], color="darkorange", linewidth=2.5, label="95% Tail CVaR (Worst-Case Avg %)")
ax2.axvline(out["opt_h"] * 100, color="#1f77b4", linestyle="--", linewidth=2, label=f"Optimal h* = {out['opt_h']*100:.0f}%")
ax2.axhline(0, color="black", linestyle=":", linewidth=1.5)
ax2.set_title("FEC Efficient Frontier: Carry Cost vs. Tail Protection", fontweight="bold")
ax2.set_xlabel("Invoice Hedged via FEC (%)")
ax2.set_ylabel("Net Profit Margin (%)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.subheader("CFO Strategy Comparison Table")
summary_df = pd.DataFrame({
    "Hedging Strategy": [
        "0% Hedged (Full Spot Exposure)",
        f"{out['opt_h']*100:.1f}% Hedged (Quant Optimal FEC)",
        "100% Hedged (Full Forward Lock)"
    ],
    "USD Locked at Forward": [
        "$0",
        f"${out['opt_h']*(invoice_usd+freight_usd):,.0f}",
        f"${(invoice_usd+freight_usd):,.0f}"
    ],
    "Expected Net Margin": [f"{u_exp:.2f}%", f"{o_exp:.2f}%", f"{f_exp:.2f}%"],
    "95% Worst-Case VaR": [f"{u_var:.2f}%", f"{o_var:.2f}%", f"{f_var:.2f}%"],
    "95% Tail CVaR (Expected Shortfall)": [f"{u_cvar:.2f}%", f"{o_cvar:.2f}%", f"{f_cvar:.2f}%"],
    "Probability of Net Loss": [f"{u_loss:.1f}%", f"{o_loss:.1f}%", f"{f_loss:.1f}%"],
})
st.table(summary_df)
