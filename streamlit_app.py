import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

# Set page config for a professional corporate dashboard look
st.set_page_config(
    page_title="ASCPS - Australian Safeguard Carbon Price Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Executive Header
st.title("Australian Safeguard Carbon Price Simulator (ASCPS)")
st.markdown("""
This simulator forecasts carbon clearing prices ($P_{ACCU}$), physical abatement behavior, 
and General Equilibrium capital flight dynamics resulting from Australia's **Safeguard Mechanism**.
*Adjust the policy levers in the sidebar to stress-test the model.*
""")

# =====================================================================
# 1. SIDEBAR CONFIGURATOR (THE POLICY DIALS)
# =====================================================================
st.sidebar.header("Policy & Market Controls")

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
st.subheader("2026 Baseline Solve")

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
st.subheader("Multi-Period Dynamic Forecast (2026 - 2060)")

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

st.markdown("**Simulation Output Ledger**")
st.dataframe(results_df, use_container_width=True)

# Note: All matplotlib visualization code and the 'Model Structural Mechanics' expander have been successfully removed as requested.
```
# eof

### Summary of Changes:
1. **Removed `matplotlib` Import:** Cleaned up the top of the file to prevent unnecessary library loads.
2. **Simplified Layout:** Eliminated the columns split in Section 4; the simulation results table now elegantly spans the full container width for maximum readability.
3. **Removed Mechanics Expander:** Completely excised the 2x2 structural mechanics charts from the bottom of the dashboard.
