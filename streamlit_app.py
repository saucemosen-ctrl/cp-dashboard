import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Set page config for a professional corporate dashboard look
st.set_page_config(
    page_title="ASCPS - Australian Safeguard Carbon Price Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Executive Header
st.title("📈 Australian Safeguard Carbon Price Simulator (ASCPS)")
st.markdown("""
This simulator forecasts carbon clearing prices ($P_{ACCU}$), physical abatement behavior, 
and General Equilibrium capital flight dynamics resulting from Australia's **Safeguard Mechanism**.
*Adjust the policy levers in the sidebar to stress-test the model.*
""")

# =====================================================================
# 1. SIDEBAR CONFIGURATOR (THE POLICY DIALS)
# =====================================================================
st.sidebar.header("🎛️ Policy & Market Controls")

threshold = st.sidebar.slider(
    "1. Safeguard Threshold (tCO2-e)",
    min_value=0,
    max_value=200000,
    value=100000,
    step=5000,
    help="Facilities emitting above this threshold are caught. Standard threshold is 100,000 tonnes."
)

tightness = st.sidebar.slider(
    "2. Policy Tightness Factor",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Multiplier on decline rates (1.0 = Standard statutory decline rate)."
)

stockpile_input = st.sidebar.slider(
    "3. Private ACCU Stockpile (Mt)",
    min_value=0.0,
    max_value=150.0,
    value=60.7,
    step=5.0,
    help="Total volume of ACCUs held in private registries. Default is 60.7 Mt."
)

tech_speed = st.sidebar.slider(
    "4. Tech Maturity Speed (Wright's Law)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Time dilation factor for technology learning. Low speed stretches cost curves out."
)

tier2_input = st.sidebar.slider(
    "5. Tier 2 Supply Potential",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Scales emergency mid-cost supply capacity (e.g. Carbon Farming/Engineering)."
)

remove_cap = st.sidebar.checkbox(
    "6. Remove Cost Containment Cap?",
    value=False,
    help="If selected, prices can float past the legal government ceiling."
)

# =====================================================================
# 2. MODEL ENGINE (ENCAPSULATED FOR STREAMLIT RUNTIME)
# =====================================================================

# Static physical limits and elasticities
TOTAL_SECTOR_EMISSIONS = {
    'MIN': {'total_mt': 95.0, 'e_min': 25000, 'alpha': 1.5},
    'MAN': {'total_mt': 52.0, 'e_min': 20000, 'alpha': 1.5},
    'ELC': {'total_mt': 115.0, 'e_min': 0, 'alpha': 1.0} 
}

MAX_ABATEMENT_LIMITS = {
    'MIN_C': 0.80, 
    'MAN_C': 0.70, 
    'ELC':   0.98  
}

SIGMA_CAPITAL = {
    'MIN_C': 0.3, 
    'MAN_C': 0.5,
    'ELC':   1.5  
}

# Dynamic Base setup
market_params = {
    'annual_issuance': 21.7,
    'private_stockpile': stockpile_input,
    'cost_containment_cap': 82.68
}

policy_state = {
    'remove_cap': remove_cap,
    'tier2_factor': tier2_input
}

# Calculate baseline coverage partition based on threshold slider
emissions_state = {}
for sector, data in TOTAL_SECTOR_EMISSIONS.items():
    if sector == 'ELC':
        coverage_share = 1.0
    else:
        if threshold <= data['e_min']:
            coverage_share = 1.0
        else:
            coverage_share = (data['e_min'] / threshold) ** (data['alpha'] - 1)
        coverage_share = min(max(coverage_share, 0.0), 1.0)
    
    e_actual_covered = data['total_mt'] * coverage_share
    baseline_assigned = e_actual_covered * (1 - 0.049)
    
    key = f"{sector}_C" if sector != 'ELC' else 'ELC'
    emissions_state[key] = {'e_actual': e_actual_covered, 'baseline': baseline_assigned, 'coverage_share': coverage_share}

# Local helper functions referencing active state
def get_abatement(p_accu, sector, active_emissions, active_maccs):
    e_actual = active_emissions[sector]['e_actual']
    baseline = active_emissions[sector]['baseline']
    liability_gap = max(0.0, e_actual - baseline)
    
    if liability_gap == 0:
        return 0.0
        
    macc_t = active_maccs[sector]
    lam = -np.log(0.5) / macc_t
    max_abatable_tonnes = liability_gap * MAX_ABATEMENT_LIMITS[sector]
    
    abatement_mt = max_abatable_tonnes * (1 - np.exp(-lam * p_accu))
    return abatement_mt

def get_compliance_cost(p_accu, sector, abatement_mt, active_emissions):
    e_actual = active_emissions[sector]['e_actual']
    baseline = active_emissions[sector]['baseline']
    liability_gap = max(0.0, e_actual - baseline)
    
    liability_mt = max(0.0, liability_gap - abatement_mt)
    accu_cost = liability_mt * p_accu
    abatement_cost = (abatement_mt * p_accu) * 0.45 
    
    return accu_cost + abatement_cost, liability_mt

def get_accu_supply(p_accu, local_market, local_policy):
    # TIER 1
    base_wts = 1 / (1 + np.exp(-0.4 * (p_accu - 36.25)))
    tier1_supply = local_market['annual_issuance'] * base_wts
    
    # TIER 2
    tier2_wts = 1 / (1 + np.exp(-0.15 * (p_accu - 75.0)))
    tier2_max_supply = 30.0 * local_policy['tier2_factor']
    tier2_supply = tier2_max_supply * tier2_wts
    
    # TIER 3
    tier3_wts = 1 / (1 + np.exp(-0.02 * (p_accu - 400.0)))
    tier3_max_supply = 40.0
    tier3_supply = tier3_max_supply * tier3_wts
    
    # PRIVATE STOCKPILE
    release_rate = 1 / (1 + np.exp(-0.2 * (p_accu - 55.0)))
    released_stockpile = local_market['private_stockpile'] * release_rate
    
    return tier1_supply + tier2_supply + tier3_supply + released_stockpile

def get_market_clearing(p_accu, active_emissions, active_maccs, local_market, local_policy):
    total_accu_demand = 5.0 
    
    for sector in active_emissions.keys():
        abatement = get_abatement(p_accu, sector, active_emissions, active_maccs)
        _, liability = get_compliance_cost(p_accu, sector, abatement, active_emissions)
        total_accu_demand += liability
        
    total_accu_supply = get_accu_supply(p_accu, local_market, local_policy)
    return total_accu_demand - total_accu_supply

def run_solver(year, active_emissions, active_maccs, local_market, local_policy):
    cap = local_market['cost_containment_cap']
    upper_bracket = 2000.0 if local_policy['remove_cap'] else cap
    
    def target_fn(p):
        return get_market_clearing(p, active_emissions, active_maccs, local_market, local_policy)
    
    try:
        res = root_scalar(target_fn, bracket=[0.1, upper_bracket], method='brentq')
        if res.converged:
            return res.root, "Cleared"
        else:
            return upper_bracket, "Saturated (Cap Hit)"
    except ValueError:
        return upper_bracket, "Saturated (Cap Hit)"

# =====================================================================
# 3. BASELINE (2026) ANALYSIS DISPLAY
# =====================================================================
st.subheader("📊 2026 Baseline Solve")

# Sector baseline info cards
base_maccs = {'MIN_C': 45.0, 'MAN_C': 35.0, 'ELC': 25.0}
p_2026, status = run_solver(2026, emissions_state, base_maccs, market_params, policy_state)

col_metric1, col_metric2, col_metric3 = st.columns(3)
with col_metric1:
    st.metric("Equilibrium ACCU Price (2026)", f"${p_2026:.2f}", help=f"Market status: {status}")
with col_metric2:
    cov_pct = emissions_state['MIN_C']['coverage_share'] * 100
    st.metric("Mining Covered Emissions", f"{emissions_state['MIN_C']['e_actual']:.2f} Mt", f"{cov_pct:.1f}% Covered")
with col_metric3:
    cov_pct_man = emissions_state['MAN_C']['coverage_share'] * 100
    st.metric("Manufacturing Covered Emissions", f"{emissions_state['MAN_C']['e_actual']:.2f} Mt", f"{cov_pct_man:.1f}% Covered")

st.markdown("#### Macroeconomic Ripple Effects (Capital Flight Proxy)")
cols_ge = st.columns(2)
for idx, sector in enumerate(['MIN_C', 'MAN_C']):
    abatement = get_abatement(p_2026, sector, emissions_state, base_maccs)
    total_cost, liability = get_compliance_cost(p_2026, sector, abatement, emissions_state)
    capital_flight_m = total_cost * SIGMA_CAPITAL[sector]
    
    with cols_ge[idx]:
        st.info(f"**Sector: {sector}**\n"
                f"*   Abated Internally: **{abatement:.2f} Mt**\n"
                f"*   Residual Offset Demand: **{liability:.2f} Mt**\n"
                f"*   Total Compliance Burden: **${total_cost:.2f} Million**\n"
                f"⚠️ **Structural Capital Flight: ${capital_flight_m:.2f} Million**")

# =====================================================================
# 4. MULTI-PERIOD DYNAMIC SIMULATION (2026-2060)
# =====================================================================
st.subheader("🔮 Multi-Period Dynamic Forecast (2026 - 2060)")

sim_years = [2026, 2027, 2028, 2029, 2030, 2035, 2040, 2050, 2060]
decline_rate = 0.049 * tightness
history = {'years': [], 'accu': [], 'min': [], 'man': [], 'elc': [], 'status': []}

for target_year in sim_years:
    years_elapsed = target_year - 2026
    
    # Apply baseline reductions over time
    temp_emissions = {}
    for k, v in emissions_state.items():
        shrunk_baseline = v['baseline'] * ((1 - decline_rate) ** years_elapsed)
        temp_emissions[k] = {'e_actual': v['e_actual'], 'baseline': shrunk_baseline}
        
    # Hill Equation Cost decay trajectories (Wright's Law)
    base_inflection = 12.0
    temp_maccs = {}
    for sector, base_cost in base_maccs.items():
        floor = 15.0 if sector == 'MIN_C' else (20.0 if sector == 'MAN_C' else 5.0)
        effective_years = years_elapsed * tech_speed
        time_factor = (effective_years / base_inflection) ** 3
        cost_t = floor + (base_cost - floor) / (1 + time_factor)
        temp_maccs[sector] = cost_t
        
    # Scale Cost containment cap over time by 2.5% inflation
    temp_market = market_params.copy()
    if not policy_state['remove_cap']:
        temp_market['cost_containment_cap'] = 82.68 * ((1 + 0.025) ** years_elapsed)
        
    # Solve
    p_eq, yr_status = run_solver(target_year, temp_emissions, temp_maccs, temp_market, policy_state)
    
    history['years'].append(target_year)
    history['accu'].append(p_eq)
    history['min'].append(temp_maccs['MIN_C'])
    history['man'].append(temp_maccs['MAN_C'])
    history['elc'].append(temp_maccs['ELC'])
    history['status'].append(yr_status)

# Format & Output results table
results_df = pd.DataFrame({
    "Year": history['years'],
    "ACCU Price": [f"${x:.2f}" for x in history['accu']],
    "Mining MACC": [f"${x:.2f}" for x in history['min']],
    "Mfg MACC": [f"${x:.2f}" for x in history['man']],
    "Electricity MACC": [f"${x:.2f}" for x in history['elc']],
    "Market Status": history['status']
})

col_table, col_chart = st.columns([2, 3])

with col_table:
    st.markdown("**Simulation Output Ledger**")
    st.dataframe(results_df, use_container_width=True)

with col_chart:
    # 5. DYNAMIC ACCU VS MACC TRAJECTORY CHART
    fig_traj, ax_traj = plt.subplots(figsize=(8, 5))
    ax_traj.plot(history['years'], history['min'], label="Mining MACC (MIN_C)", color='#d95f02', linestyle='--', linewidth=1.5)
    ax_traj.plot(history['years'], history['man'], label="Manufacturing MACC (MAN_C)", color='#7570b3', linestyle='--', linewidth=1.5)
    ax_traj.plot(history['years'], history['elc'], label="Electricity MACC (ELC)", color='#1b9e77', linestyle='--', linewidth=1.5)
    ax_traj.plot(history['years'], history['accu'], label="ACCU Clearing Price", color='black', linewidth=3, marker='o', markersize=6)
    
    ax_traj.set_title("ACCU Price vs. Technology Learning Curves", fontsize=12, fontweight='bold')
    ax_traj.set_xlabel("Simulation Year", fontsize=10)
    ax_traj.set_ylabel("Price (AUD / tCO2-e)", fontsize=10)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_traj)

# =====================================================================
# 5. UNDER THE HOOD MECHANICS PLOTS
# =====================================================================
with st.expander("🛠️ View Model Structural Mechanics"):
    st.markdown("These charts plot the static mathematical distributions calibrating the solver's logic.")
    
    fig_mech, axs = plt.subplots(2, 2, figsize=(14, 9))
    
    # 1. Pareto Threshold Distribution
    ax1 = axs[0, 0]
    t_vals = np.linspace(15000, 200000, 200)
    min_cov = np.clip((25000 / t_vals) ** 0.5, 0, 1) * 100
    man_cov = np.clip((20000 / t_vals) ** 0.5, 0, 1) * 100
    ax1.plot(t_vals, min_cov, label="Mining (e_min=25k)", color='#d95f02', linewidth=2)
    ax1.plot(t_vals, man_cov, label="Manufacturing (e_min=20k)", color='#7570b3', linewidth=2)
    ax1.set_title("1. Pareto Coverage Distribution")
    ax1.set_xlabel("Policy Threshold (Tonnes CO2-e)")
    ax1.set_ylabel("Sector Emissions Captured (%)")
    ax1.axvline(x=threshold, color='r', linestyle='--', label=f"Selected: {threshold:,.0f}t")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Exponential Abatement Response
    ax2 = axs[0, 1]
    p_vals = np.linspace(0, 150, 200)
    lam_min = -np.log(0.5) / 45.0
    lam_man = -np.log(0.5) / 35.0
    lam_elc = -np.log(0.5) / 25.0
    ax2.plot(p_vals, 80.0 * (1 - np.exp(-lam_min * p_vals)), label="MIN_C (Max 80%)", color='#d95f02', linewidth=2)
    ax2.plot(p_vals, 70.0 * (1 - np.exp(-lam_man * p_vals)), label="MAN_C (Max 70%)", color='#7570b3', linewidth=2)
    ax2.plot(p_vals, 98.0 * (1 - np.exp(-lam_elc * p_vals)), label="ELC (Max 98%)", color='#1b9e77', linewidth=2)
    ax2.set_title("2. Abatement Response & Physical Limits")
    ax2.set_xlabel("ACCU Price (AUD)")
    ax2.set_ylabel("Abatement Achieved (% of Liability Gap)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Three-Tiered Logistic Supply Curve
    ax3 = axs[1, 0]
    p_supply = np.linspace(0, 500, 500)
    tier1 = 21.7 * (1 / (1 + np.exp(-0.4 * (p_supply - 36.25))))
    tier2 = (30.0 * tier2_input) * (1 / (1 + np.exp(-0.15 * (p_supply - 75.0))))
    tier3 = 40.0 * (1 / (1 + np.exp(-0.02 * (p_supply - 400.0))))
    stock = stockpile_input * (1 / (1 + np.exp(-0.2 * (p_supply - 55.0))))
    total_supply = tier1 + tier2 + tier3 + stock
    
    ax3.plot(p_supply, tier1, label="Tier 1 (Base/Land)", linestyle='--')
    ax3.plot(p_supply, tier2, label="Tier 2 (Eng/Carbon Farm)", linestyle='--')
    ax3.plot(p_supply, tier3, label="Tier 3 (DAC Backstop)", linestyle='--')
    ax3.plot(p_supply, stock, label="Private Stockpile Release", linestyle='--')
    ax3.plot(p_supply, total_supply, label="Total Market Supply", color='black', linewidth=2)
    ax3.set_title("3. Multi-Tiered Logistic Supply Curve")
    ax3.set_xlabel("ACCU Price (AUD)")
    ax3.set_ylabel("Supply Volume (Mt)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Hill Equation (Wright's Law)
    ax4 = axs[1, 1]
    years = np.arange(2026, 2061)
    base_inflection = 12.0
    min_costs, man_costs, elc_costs = [], [], []
    
    for y in years:
        elapsed = y - 2026
        eff_years = elapsed * tech_speed
        time_fac = (eff_years / base_inflection) ** 3
        min_costs.append(15.0 + (45.0 - 15.0) / (1 + time_fac))
        man_costs.append(20.0 + (35.0 - 20.0) / (1 + time_fac))
        elc_costs.append(5.0 + (25.0 - 5.0) / (1 + time_fac))
        
    ax4.plot(years, min_costs, label="MIN_C (Floor $15)", color='#d95f02', linewidth=2)
    ax4.plot(years, man_costs, label="MAN_C (Floor $20)", color='#7570b3', linewidth=2)
    ax4.plot(years, elc_costs, label="ELC (Floor $5)", color='#1b9e77', linewidth=2)
    ax4.set_title(f"4. Technology Learning Curve (Speed={tech_speed:.1f})")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Marginal Abatement Cost (AUD)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_mech)
