import io
import time
import math
import random
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


# ---------------------------------------------------
# LOAD ASSIGNMENT HELPER FUNCTION
# ---------------------------------------------------
def load_run_sim():
    possible_files = [
        "Assignment3_Functions_Solution.py",
        "Assignment3_Functions_Solutions.py",
        "Assignment3_Functions_Solutions-1.py",
    ]

    for file_name in possible_files:
        file_path = Path(__file__).parent / file_name

        if file_path.exists():
            spec = importlib.util.spec_from_file_location(
                "assignment3_functions",
                file_path,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.run_sim

    raise FileNotFoundError(
        "Simulation helper file not found. Make sure your Assignment3_Functions file is in the same folder."
    )


run_sim = load_run_sim()


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Epidemiological Decision Dashboard",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 Epidemiological Decision Dashboard")
st.caption("Interactive disease spread analytics for SIR / SIRD / SIRDV models")
st.caption("Built by Omar Rulida Abdul-Rahman | MPH Candidate")


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
session_defaults = {
    "simulation_df": None,
    "simulation_params": None,
    "compare_results": None,
    "sensitivity_results": None,
    "timing_results": None,
    "heatmap_results": None,
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------
# CORE MODEL FUNCTIONS
# ---------------------------------------------------
def run_model(model_choice, pop, infected, recovered, beta, gamma, mu, vac, days):
    susceptible = pop - infected - recovered

    sim_s, sim_i, sim_r, sim_d, sim_v = run_sim(
        S_0=susceptible,
        I_0=infected,
        R_0=recovered,
        beta=beta,
        gamma=gamma,
        mu=mu,
        vac_rate=vac,
        days=int(days),
        model_choice=model_choice,
    )

    return pd.DataFrame(
        {
            "Day": range(int(days)),
            "Susceptible": sim_s,
            "Infected": sim_i,
            "Recovered": sim_r,
            "Dead": sim_d,
            "Vaccinated": sim_v,
        }
    )


def run_model_with_intervention(
    model_choice,
    pop,
    infected,
    recovered,
    beta,
    gamma,
    mu,
    vac,
    days,
    intervention_enabled,
    intervention_day,
    reduced_beta,
    increased_vac,
):
    if not intervention_enabled or intervention_day <= 1 or intervention_day >= days:
        return run_model(
            model_choice,
            pop,
            infected,
            recovered,
            beta,
            gamma,
            mu,
            vac,
            days,
        )

    first_df = run_model(
        model_choice,
        pop,
        infected,
        recovered,
        beta,
        gamma,
        mu,
        vac,
        intervention_day,
    )

    last_row = first_df.iloc[-1]

    sim_s, sim_i, sim_r, sim_d, sim_v = run_sim(
        S_0=float(last_row["Susceptible"]),
        I_0=float(last_row["Infected"]),
        R_0=float(last_row["Recovered"]),
        beta=reduced_beta,
        gamma=gamma,
        mu=mu,
        vac_rate=increased_vac,
        days=int(days - intervention_day),
        model_choice=model_choice,
    )

    second_df = pd.DataFrame(
        {
            "Day": range(intervention_day, days),
            "Susceptible": sim_s,
            "Infected": sim_i,
            "Recovered": sim_r,
            "Dead": sim_d,
            "Vaccinated": sim_v,
        }
    )

    return pd.concat([first_df, second_df], ignore_index=True)


def get_columns(model):
    if model == "SIR":
        return ["Susceptible", "Infected", "Recovered"]

    if model == "SIRD":
        return ["Susceptible", "Infected", "Recovered", "Dead"]

    return ["Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"]


def convert_to_percent(df, columns, population):
    df_percent = df.copy()

    for col in columns:
        df_percent[col] = (df_percent[col] / population) * 100

    return df_percent


# ---------------------------------------------------
# LABEL MODE
# ---------------------------------------------------
def label_text(label, real_world_mode):
    if not real_world_mode:
        return label

    label_map = {
        "Beta (Infection Rate)": "Transmission Rate",
        "Gamma (Recovery Rate)": "Recovery Speed",
        "Mu (Death Rate)": "Mortality Pressure",
        "Vaccination Rate": "Vaccination Speed",
    }

    return label_map.get(label, label)


# ---------------------------------------------------
# METRICS AND INSIGHTS
# ---------------------------------------------------
def metrics_from_df(df, beta, gamma, population, hospitalization_rate):
    peak_infected = float(df["Infected"].max())
    day_of_peak = int(df["Infected"].idxmax())
    final_recovered = float(df["Recovered"].iloc[-1])
    final_deaths = float(df["Dead"].iloc[-1])
    final_vaccinated = float(df["Vaccinated"].iloc[-1])
    remaining_susceptible = float(df["Susceptible"].iloc[-1])

    r0 = beta / gamma if gamma > 0 else 0
    total_impacted = population - remaining_susceptible
    attack_rate = ((final_recovered + final_deaths) / population) * 100
    estimated_peak_hospitalizations = peak_infected * hospitalization_rate
    cumulative_cases = final_recovered + final_deaths

    return {
        "Peak Infected": peak_infected,
        "Day of Peak": day_of_peak,
        "Recovered": final_recovered,
        "Deaths": final_deaths,
        "Vaccinated": final_vaccinated,
        "Susceptible": remaining_susceptible,
        "R0": r0,
        "Total Impacted": total_impacted,
        "Attack Rate": attack_rate,
        "Estimated Peak Hospitalizations": estimated_peak_hospitalizations,
        "Cumulative Cases": cumulative_cases,
    }


def outbreak_phase(df):
    current_infected = float(df["Infected"].iloc[-1])
    peak_infected = float(df["Infected"].max())
    peak_day = int(df["Infected"].idxmax())
    final_day = int(df["Day"].iloc[-1])

    if current_infected <= 1:
        return "Controlled"

    if final_day < peak_day:
        return "Acceleration"

    if abs(final_day - peak_day) <= 5:
        return "Peak"

    if current_infected < peak_infected * 0.25:
        return "Decline"

    if final_day < 15:
        return "Emergence"

    return "Post-Peak Monitoring"


def first_icu_crossing_day(df, hospitalization_rate, icu_capacity):
    estimated_hospitalizations = df["Infected"] * hospitalization_rate
    crossed = df.loc[estimated_hospitalizations > icu_capacity]

    if crossed.empty:
        return None

    return int(crossed["Day"].iloc[0])


def policy_recommendation(metrics, icu_capacity):
    if metrics["Estimated Peak Hospitalizations"] > icu_capacity:
        return "Urgent intervention recommended: estimated peak hospitalizations exceed ICU capacity."

    if metrics["R0"] > 2:
        return "Strong intervention recommended: reduce transmission through vaccination, distancing, or other controls."

    if metrics["Attack Rate"] > 50:
        return "High population impact expected: vaccination and transmission reduction should be prioritized."

    if metrics["R0"] <= 1:
        return "Current parameters suggest the outbreak may be controlled."

    return "Moderate risk: continue monitoring and consider preventive interventions."


def risk_level(metrics, icu_capacity):
    if metrics["Estimated Peak Hospitalizations"] > icu_capacity or metrics["R0"] > 3:
        return "High Risk", "error"

    if metrics["R0"] > 1 or metrics["Attack Rate"] > 25:
        return "Moderate Risk", "warning"

    return "Controlled / Lower Risk", "success"


def readiness_grade(metrics, icu_capacity, vac):
    score = 100

    if metrics["R0"] > 3:
        score -= 30
    elif metrics["R0"] > 2:
        score -= 20
    elif metrics["R0"] > 1:
        score -= 10

    if metrics["Attack Rate"] > 50:
        score -= 25
    elif metrics["Attack Rate"] > 25:
        score -= 15
    elif metrics["Attack Rate"] > 10:
        score -= 5

    if metrics["Estimated Peak Hospitalizations"] > icu_capacity:
        score -= 25

    if vac >= 0.05:
        score += 5

    score = max(min(score, 100), 0)

    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 60:
        return "D"

    return "F"


def policy_impact_score(current_metrics, baseline_metrics, icu_capacity):
    baseline_peak = baseline_metrics["Peak Infected"]
    current_peak = current_metrics["Peak Infected"]

    baseline_cases = baseline_metrics["Cumulative Cases"]
    current_cases = current_metrics["Cumulative Cases"]

    peak_reduction_pct = 0
    case_reduction_pct = 0

    if baseline_peak > 0:
        peak_reduction_pct = ((baseline_peak - current_peak) / baseline_peak) * 100

    if baseline_cases > 0:
        case_reduction_pct = ((baseline_cases - current_cases) / baseline_cases) * 100

    icu_bonus = 20 if current_metrics["Estimated Peak Hospitalizations"] <= icu_capacity else 0

    score = 40 + (peak_reduction_pct * 0.3) + (case_reduction_pct * 0.3) + icu_bonus
    score = max(min(score, 100), 0)

    return round(score, 1)


def generate_insights(df, beta, gamma, mu, vac, population, hospitalization_rate):
    metrics = metrics_from_df(
        df,
        beta,
        gamma,
        population,
        hospitalization_rate,
    )

    insights = []

    if metrics["R0"] > 3:
        insights.append(
            "High transmission pressure: R₀ is well above 1, suggesting rapid outbreak growth."
        )
    elif metrics["R0"] > 1:
        insights.append(
            "Moderate outbreak potential: R₀ is above 1, so infections may continue growing."
        )
    else:
        insights.append(
            "Controlled transmission: R₀ is at or below 1, suggesting the outbreak may shrink over time."
        )

    if metrics["Day of Peak"] <= 30:
        insights.append("Peak infection occurs early, indicating fast disease spread.")
    elif metrics["Day of Peak"] >= 90:
        insights.append("Peak infection occurs later, suggesting a slower-moving outbreak curve.")
    else:
        insights.append("Peak infection occurs in the middle of the simulation period.")

    if gamma > beta:
        insights.append("Recovery is stronger than transmission, which helps flatten the curve.")

    if vac >= 0.05:
        insights.append("Vaccination is meaningfully reducing the susceptible population over time.")

    if mu >= 0.03:
        insights.append("Death rate is elevated, making this a higher-severity scenario.")

    if metrics["Attack Rate"] >= 50:
        insights.append(
            "A large share of the population experiences infection or death by the end of the simulation."
        )
    elif metrics["Attack Rate"] <= 10:
        insights.append("Final infection burden remains relatively low.")

    if hospitalization_rate >= 0.10:
        insights.append("Hospital burden may become significant because the hospitalization assumption is high.")

    return insights


def executive_brief(metrics, phase, recommendation, grade):
    return (
        f"This simulation indicates a {phase.lower()} outbreak phase with peak infection "
        f"on day {metrics['Day of Peak']} and an estimated attack rate of "
        f"{metrics['Attack Rate']:.1f}%. The estimated R₀ is {metrics['R0']:.2f}, "
        f"and the public health readiness grade is {grade}. {recommendation}"
    )


# ---------------------------------------------------
# CHARTING
# ---------------------------------------------------
def make_plot(
    df,
    model,
    y_axis_title,
    title,
    template,
    show_cumulative,
    hospitalization_rate=None,
    icu_capacity=None,
    intervention_day=None,
):
    colors = {
        "Susceptible": "#1f77b4",
        "Infected": "#d62728",
        "Recovered": "#2ca02c",
        "Dead": "#7f7f7f",
        "Vaccinated": "#9467bd",
        "Cumulative Cases": "#ff7f0e",
    }

    fig = go.Figure()

    for col in get_columns(model):
        line_width = 4 if col == "Infected" else 3

        fig.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=line_width, color=colors[col]),
                hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
            )
        )

    if show_cumulative:
        fig.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df["Recovered"] + df["Dead"],
                mode="lines",
                name="Cumulative Cases",
                line=dict(width=3, dash="dot", color=colors["Cumulative Cases"]),
            )
        )

    peak_idx = int(df["Infected"].idxmax())
    peak_val = float(df["Infected"].max())

    label_bg = "#ffffff" if template == "plotly_white" else "#111827"
    label_font = "#111827" if template == "plotly_white" else "#ffffff"

    fig.add_trace(
        go.Scatter(
            x=[peak_idx],
            y=[peak_val],
            mode="markers",
            name="Peak Marker",
            marker=dict(size=12, color="#d62728", line=dict(width=2, color="white")),
            showlegend=False,
            hovertemplate="Peak Infection: %{y:.0f}<extra></extra>",
        )
    )

    fig.add_annotation(
        x=peak_idx,
        y=peak_val,
        text="<b>Peak</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#d62728",
        ax=0,
        ay=-45,
        font=dict(size=13, color=label_font),
        bgcolor=label_bg,
        bordercolor="#d62728",
        borderwidth=2,
        borderpad=4,
        opacity=0.98,
    )

    if icu_capacity is not None and y_axis_title == "Population":
        fig.add_hline(
            y=icu_capacity,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="ICU Capacity",
            annotation_position="top left",
        )

        if hospitalization_rate is not None:
            crossing_day = first_icu_crossing_day(df, hospitalization_rate, icu_capacity)

            if crossing_day is not None:
                fig.add_vline(
                    x=crossing_day,
                    line_dash="dot",
                    line_color="#f59e0b",
                    annotation_text="ICU crossed",
                    annotation_position="top right",
                )

    if intervention_day is not None:
        fig.add_vline(
            x=intervention_day,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text="Intervention begins",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Day",
        yaxis_title=y_axis_title,
        template=template,
        hovermode="x unified",
        height=650,
        margin=dict(l=30, r=30, t=70, b=40),
        legend_title="Compartments",
    )

    return fig


def make_healthcare_timeline(df, hospitalization_rate, icu_capacity, template):
    df_health = df.copy()
    df_health["Estimated Hospitalizations"] = df_health["Infected"] * hospitalization_rate

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_health["Day"],
            y=df_health["Estimated Hospitalizations"],
            mode="lines",
            name="Estimated Hospitalizations",
            line=dict(width=3),
        )
    )

    fig.add_hline(
        y=icu_capacity,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="ICU Capacity",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Healthcare Capacity Timeline",
        xaxis_title="Day",
        yaxis_title="Estimated Hospitalizations",
        template=template,
        hovermode="x unified",
        height=420,
    )

    return fig


# ---------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------
def create_parameter_summary(params):
    return f"""Epidemiological Decision Dashboard Parameter Summary

Model: {params["model_choice"]}
Population: {params["population"]}
Initial Infected: {params["infected"]}
Initial Recovered: {params["recovered"]}
Beta: {params["beta"]}
Gamma: {params["gamma"]}
Mu: {params["mu"]}
Vaccination Rate: {params["vac"]}
Hospitalization Rate: {params["hospitalization_rate"]}
ICU Capacity: {params["icu_capacity"]}
Days: {params["days"]}

Intervention Enabled: {params["intervention_enabled"]}
Intervention Day: {params["intervention_day"]}
Post-Intervention Transmission: {params["reduced_beta"]}
Post-Intervention Vaccination: {params["increased_vac"]}
"""


def create_full_report(params, metrics, insights, recommendation, phase, grade, brief, impact_score=None):
    return f"""Epidemiological Decision Dashboard Report

Executive Summary:
{brief}

Model:
{params["model_choice"]}

Inputs:
Population: {params["population"]}
Initial Infected: {params["infected"]}
Initial Recovered: {params["recovered"]}
Beta: {params["beta"]}
Gamma: {params["gamma"]}
Mu: {params["mu"]}
Vaccination Rate: {params["vac"]}
Hospitalization Rate: {params["hospitalization_rate"]}
ICU Capacity: {params["icu_capacity"]}
Days: {params["days"]}

Intervention Settings:
Intervention Enabled: {params["intervention_enabled"]}
Intervention Day: {params["intervention_day"]}
Post-Intervention Transmission: {params["reduced_beta"]}
Post-Intervention Vaccination: {params["increased_vac"]}

Key Results:
Peak Infected: {metrics["Peak Infected"]:,.0f}
Day of Peak: {metrics["Day of Peak"]}
R0: {metrics["R0"]:.2f}
Attack Rate: {metrics["Attack Rate"]:.2f}%
Recovered: {metrics["Recovered"]:,.0f}
Deaths: {metrics["Deaths"]:,.0f}
Vaccinated: {metrics["Vaccinated"]:,.0f}
Susceptible Remaining: {metrics["Susceptible"]:,.0f}
Estimated Peak Hospitalizations: {metrics["Estimated Peak Hospitalizations"]:,.0f}
Outbreak Phase: {phase}
Public Health Readiness Grade: {grade}
Policy Impact Score: {impact_score if impact_score is not None else "N/A"}

Automated Interpretation:
{chr(10).join("- " + insight for insight in insights)}

Policy Recommendation:
{recommendation}

Methods and Assumptions:
This dashboard uses simplified deterministic compartmental modeling.
Results are educational estimates and should not be used as real-world forecasts.
"""


def create_scenario_summary(metrics, phase, grade, recommendation):
    return f"""Scenario Summary

Risk Phase: {phase}
Readiness Grade: {grade}
Peak Infected: {metrics["Peak Infected"]:,.0f}
Day of Peak: {metrics["Day of Peak"]}
R0: {metrics["R0"]:.2f}
Attack Rate: {metrics["Attack Rate"]:.2f}%
Estimated Peak Hospitalizations: {metrics["Estimated Peak Hospitalizations"]:,.0f}

Recommendation:
{recommendation}
"""


def plot_download_buttons(fig, model_choice):
    html_buffer = io.StringIO()
    fig.write_html(html_buffer, include_plotlyjs="cdn")

    st.download_button(
        "Download Interactive Plot as HTML",
        data=html_buffer.getvalue(),
        file_name=f"{model_choice.lower()}_interactive_plot.html",
        mime="text/html",
    )

    try:
        png_bytes = fig.to_image(format="png", scale=3)

        st.download_button(
            "Download Plot as PNG",
            data=png_bytes,
            file_name=f"{model_choice.lower()}_simulation_plot.png",
            mime="image/png",
        )
    except Exception:
        st.info("PNG export needs `kaleido` in requirements.txt.")


# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------
st.sidebar.header("Simulation Controls")

if st.sidebar.button("Reset Dashboard"):
    for key in session_defaults:
        st.session_state[key] = None
    st.rerun()

mode = st.sidebar.radio(
    "Interface Mode",
    ["Academic Mode", "Real-World Mode"],
)

real_world_mode = mode == "Real-World Mode"

guided_start = st.sidebar.checkbox(
    "Start with a realistic baseline scenario",
    value=False,
)

preset = st.sidebar.selectbox(
    "Preset Scenario",
    [
        "Custom",
        "COVID-like",
        "Seasonal Flu",
        "High Vaccination",
        "Severe Outbreak",
    ],
)

preset_values = {
    "Custom": {
        "population": 1000,
        "infected": 5,
        "recovered": 0,
        "beta": 0.40,
        "gamma": 0.05,
        "mu": 0.01,
        "vac": 0.03,
        "days": 120,
    },
    "COVID-like": {
        "population": 1000,
        "infected": 5,
        "recovered": 0,
        "beta": 0.45,
        "gamma": 0.08,
        "mu": 0.01,
        "vac": 0.02,
        "days": 160,
    },
    "Seasonal Flu": {
        "population": 1000,
        "infected": 5,
        "recovered": 0,
        "beta": 0.28,
        "gamma": 0.12,
        "mu": 0.002,
        "vac": 0.01,
        "days": 100,
    },
    "High Vaccination": {
        "population": 1000,
        "infected": 5,
        "recovered": 0,
        "beta": 0.40,
        "gamma": 0.06,
        "mu": 0.01,
        "vac": 0.08,
        "days": 120,
    },
    "Severe Outbreak": {
        "population": 1000,
        "infected": 10,
        "recovered": 0,
        "beta": 0.80,
        "gamma": 0.03,
        "mu": 0.04,
        "vac": 0.00,
        "days": 140,
    },
}

defaults = preset_values[preset]

if guided_start:
    defaults = {
        "population": 1000,
        "infected": 5,
        "recovered": 0,
        "beta": 0.36,
        "gamma": 0.10,
        "mu": 0.008,
        "vac": 0.04,
        "days": 120,
    }

model_choice = st.sidebar.selectbox("Choose Model", ["SIR", "SIRD", "SIRDV"])

population = st.sidebar.number_input(
    "Total Population",
    min_value=1,
    value=defaults["population"],
)

infected = st.sidebar.number_input(
    "Initial Infected",
    min_value=0,
    value=defaults["infected"],
)

recovered = st.sidebar.number_input(
    "Initial Recovered",
    min_value=0,
    value=defaults["recovered"],
)

beta = st.sidebar.slider(
    label_text("Beta (Infection Rate)", real_world_mode),
    0.0,
    1.0,
    defaults["beta"],
    0.01,
    help="Higher transmission reflects more contact, lower masking, or faster disease spread.",
)

gamma = st.sidebar.slider(
    label_text("Gamma (Recovery Rate)", real_world_mode),
    0.0,
    1.0,
    defaults["gamma"],
    0.01,
    help="Higher recovery speed means infected individuals recover faster.",
)

mu = 0.0
vac = 0.0

if model_choice in ["SIRD", "SIRDV"]:
    mu = st.sidebar.slider(
        label_text("Mu (Death Rate)", real_world_mode),
        0.0,
        1.0,
        defaults["mu"],
        0.01,
        help="Higher mortality pressure means more severe disease outcomes.",
    )

if model_choice == "SIRDV":
    vac = st.sidebar.slider(
        label_text("Vaccination Rate", real_world_mode),
        0.0,
        1.0,
        defaults["vac"],
        0.01,
        help="Higher vaccination speed reduces the susceptible population faster.",
    )

days = st.sidebar.slider("Days", 10, 365, defaults["days"])

hospitalization_rate = st.sidebar.slider(
    "Estimated Hospitalization Rate",
    0.0,
    1.0,
    0.05,
    0.01,
)

icu_capacity = st.sidebar.number_input(
    "ICU Capacity",
    min_value=0,
    value=100,
    step=10,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Intervention Timeline")

intervention_enabled = st.sidebar.toggle("Enable intervention", value=False)

intervention_day = st.sidebar.slider(
    "Intervention Start Day",
    1,
    days - 1,
    min(30, days - 1),
)

reduced_beta = st.sidebar.slider(
    "Post-Intervention Transmission",
    0.0,
    1.0,
    max(beta * 0.6, 0.0),
    0.01,
)

increased_vac = st.sidebar.slider(
    "Post-Intervention Vaccination",
    0.0,
    1.0,
    max(vac, 0.05),
    0.01,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")

theme = st.sidebar.radio("Chart Theme", ["Light", "Dark"])
template = "plotly_dark" if theme == "Dark" else "plotly_white"

show_percent = st.sidebar.toggle("Show chart as percentage", value=False)
show_total_check = st.sidebar.toggle("Show population conservation check", value=True)
show_cumulative = st.sidebar.toggle("Show cumulative cases curve", value=True)


# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------
errors = []

if infected > population:
    errors.append("Initial infected cannot exceed total population.")

if infected + recovered > population:
    errors.append("Initial infected + recovered cannot exceed total population.")

if gamma == 0:
    st.sidebar.warning("Gamma is 0, so R₀ will be shown as 0 to avoid division by zero.")

for error in errors:
    st.sidebar.error(error)

run_button = st.sidebar.button(
    "Run Simulation",
    type="primary",
    disabled=bool(errors),
)


# ---------------------------------------------------
# LANDING SECTION
# ---------------------------------------------------
st.markdown(
    """
### Public Health Scenario Simulator

Use this dashboard to explore how transmission, recovery, mortality, vaccination, healthcare capacity, intervention timing, and geographic risk assumptions shape infectious disease outcomes.
"""
)

with st.expander("About This Dashboard"):
    st.write(
        "This dashboard is an educational public health simulation tool. It combines compartmental disease models, "
        "interactive visualization, scenario comparison, healthcare capacity analysis, intervention timing, sensitivity analysis, "
        "location-based risk visualization, and downloadable reports."
    )


# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs(
    [
        "Simulation Results",
        "Data Table",
        "Model Explanation",
        "Compare Scenarios",
        "Sensitivity Analysis",
        "Day-by-Day View",
        "Intervention Timing",
        "Risk Heat Map",
        "Methods & Assumptions",
    ]
)


# ---------------------------------------------------
# RUN MAIN SIMULATION
# ---------------------------------------------------
if run_button:
    with st.spinner("Running simulation..."):
        time.sleep(0.4)

        df = run_model_with_intervention(
            model_choice,
            population,
            infected,
            recovered,
            beta,
            gamma,
            mu,
            vac,
            days,
            intervention_enabled,
            intervention_day,
            reduced_beta,
            increased_vac,
        )

        st.session_state.simulation_df = df

        st.session_state.simulation_params = {
            "model_choice": model_choice,
            "population": population,
            "infected": infected,
            "recovered": recovered,
            "beta": beta,
            "gamma": gamma,
            "mu": mu,
            "vac": vac,
            "hospitalization_rate": hospitalization_rate,
            "icu_capacity": icu_capacity,
            "days": days,
            "intervention_enabled": intervention_enabled,
            "intervention_day": intervention_day,
            "reduced_beta": reduced_beta,
            "increased_vac": increased_vac,
        }


# ---------------------------------------------------
# TAB 1: SIMULATION RESULTS
# ---------------------------------------------------
with tabs[0]:
    if st.session_state.simulation_df is None:
        st.info("Choose parameters in the sidebar and click Run Simulation.")

    else:
        df = st.session_state.simulation_df
        params = st.session_state.simulation_params

        plot_df = df.copy()
        y_axis_title = "Population"
        icu_line = params["icu_capacity"]

        if show_percent:
            plot_df = convert_to_percent(
                df,
                get_columns(params["model_choice"]),
                params["population"],
            )
            y_axis_title = "Population (%)"
            icu_line = None

        fig = make_plot(
            plot_df,
            params["model_choice"],
            y_axis_title,
            f"{params['model_choice']} Simulation Results",
            template,
            show_cumulative,
            hospitalization_rate=params["hospitalization_rate"],
            icu_capacity=icu_line,
            intervention_day=params["intervention_day"] if params["intervention_enabled"] else None,
        )

        metrics = metrics_from_df(
            df,
            params["beta"],
            params["gamma"],
            params["population"],
            params["hospitalization_rate"],
        )

        insights = generate_insights(
            df,
            params["beta"],
            params["gamma"],
            params["mu"],
            params["vac"],
            params["population"],
            params["hospitalization_rate"],
        )

        recommendation = policy_recommendation(metrics, params["icu_capacity"])
        risk, risk_style = risk_level(metrics, params["icu_capacity"])
        phase = outbreak_phase(df)
        grade = readiness_grade(metrics, params["icu_capacity"], params["vac"])

        baseline_df = run_model(
            params["model_choice"],
            params["population"],
            params["infected"],
            params["recovered"],
            params["beta"],
            params["gamma"],
            params["mu"],
            params["vac"],
            params["days"],
        )

        baseline_metrics = metrics_from_df(
            baseline_df,
            params["beta"],
            params["gamma"],
            params["population"],
            params["hospitalization_rate"],
        )

        impact_score = policy_impact_score(metrics, baseline_metrics, params["icu_capacity"])
        brief = executive_brief(metrics, phase, recommendation, grade)

        st.subheader("Executive Summary")

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("Risk Level", risk)
        k2.metric("R₀", f"{metrics['R0']:.2f}")
        k3.metric("Peak Day", metrics["Day of Peak"])
        k4.metric(
            "ICU Status",
            "Exceeded"
            if metrics["Estimated Peak Hospitalizations"] > params["icu_capacity"]
            else "Within Capacity",
        )
        k5.metric("Readiness Grade", grade)

        if risk_style == "error":
            st.error(brief)
        elif risk_style == "warning":
            st.warning(brief)
        else:
            st.success(brief)

        st.subheader("Simulation Dashboard")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)

        c1.metric("Peak Infected", f"{metrics['Peak Infected']:,.0f}")
        c2.metric("Day of Peak", f"{metrics['Day of Peak']}")
        c3.metric("Attack Rate", f"{metrics['Attack Rate']:.1f}%")
        c4.metric("Outbreak Phase", phase)

        c5.metric("Recovered", f"{metrics['Recovered']:,.0f}")
        c6.metric("Deaths", f"{metrics['Deaths']:,.0f}")
        c7.metric("Vaccinated", f"{metrics['Vaccinated']:,.0f}")
        c8.metric("Policy Impact Score", f"{impact_score:.1f} / 100")

        st.subheader("Healthcare Capacity Timeline")
        fig_health = make_healthcare_timeline(
            df,
            params["hospitalization_rate"],
            params["icu_capacity"],
            template,
        )
        st.plotly_chart(fig_health, use_container_width=True)

        if params["intervention_enabled"]:
            infections_prevented = baseline_metrics["Cumulative Cases"] - metrics["Cumulative Cases"]
            peak_reduction = baseline_metrics["Peak Infected"] - metrics["Peak Infected"]

            st.subheader("Intervention Impact Score")

            i1, i2, i3 = st.columns(3)
            i1.metric("Estimated Infections Prevented", f"{infections_prevented:,.0f}")
            i2.metric("Peak Reduction", f"{peak_reduction:,.0f}")
            i3.metric("Policy Effectiveness", f"{impact_score:.1f} / 100")

        st.subheader("Policy Recommendation")

        if "Urgent" in recommendation:
            st.error(recommendation)
        elif "Strong" in recommendation or "High" in recommendation:
            st.warning(recommendation)
        else:
            st.success(recommendation)

        st.subheader("Automated Interpretation")

        for insight in insights:
            st.info(insight)

        if show_total_check:
            st.subheader("Population Conservation Check")

            df_check = df.copy()
            df_check["Total"] = df_check[
                ["Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"]
            ].sum(axis=1)

            max_drift = abs(df_check["Total"] - params["population"]).max()

            check1, check2 = st.columns(2)
            check1.metric("Expected Population", f"{params['population']:,.0f}")
            check2.metric("Maximum Population Drift", f"{max_drift:,.4f}")

        st.subheader("Download Center")

        dl1, dl2, dl3, dl4 = st.columns(4)

        with dl1:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode(),
                file_name=f"{params['model_choice'].lower()}_simulation_results.csv",
                mime="text/csv",
            )

        with dl2:
            st.download_button(
                "Download Parameters",
                create_parameter_summary(params),
                file_name=f"{params['model_choice'].lower()}_parameters.txt",
                mime="text/plain",
            )

        with dl3:
            st.download_button(
                "Download Full Report",
                create_full_report(
                    params,
                    metrics,
                    insights,
                    recommendation,
                    phase,
                    grade,
                    brief,
                    impact_score,
                ),
                file_name=f"{params['model_choice'].lower()}_simulation_report.txt",
                mime="text/plain",
            )

            st.download_button(
                "Download Scenario Summary",
                create_scenario_summary(metrics, phase, grade, recommendation),
                file_name=f"{params['model_choice'].lower()}_scenario_summary.txt",
                mime="text/plain",
            )

        with dl4:
            plot_download_buttons(fig, params["model_choice"])


# ---------------------------------------------------
# TAB 2: DATA TABLE
# ---------------------------------------------------
with tabs[1]:
    if st.session_state.simulation_df is None:
        st.info("Run a simulation first to view the data table.")

    else:
        st.subheader("Simulation Data")
        st.dataframe(st.session_state.simulation_df, use_container_width=True)

        st.subheader("Quick Summary")
        st.dataframe(
            st.session_state.simulation_df.describe().round(2),
            use_container_width=True,
        )


# ---------------------------------------------------
# TAB 3: MODEL EXPLANATION
# ---------------------------------------------------
with tabs[2]:
    st.subheader("Model Explanation")

    st.markdown(
        """
## SIR Model
**Susceptible → Infected → Recovered**

The SIR model is useful for diseases where individuals become immune after recovery.

## SIRD Model
**Susceptible → Infected → Recovered / Dead**

The SIRD model adds mortality and is useful when deaths are an important outcome.

## SIRDV Model
**Susceptible → Infected → Recovered / Dead / Vaccinated**

The SIRDV model adds vaccination, allowing users to explore how vaccination changes outbreak dynamics.

## Dashboard Features

| Feature | Purpose |
|---|---|
| R₀ | Estimates outbreak growth potential |
| Attack Rate | Shows share infected or dead |
| ICU Capacity | Tests healthcare strain |
| Outbreak Phase | Labels epidemic stage |
| Readiness Grade | Summarizes public health preparedness |
| Policy Impact Score | Estimates effectiveness of interventions |
| Timeline Callouts | Highlights peak, intervention start, and ICU crossing |
| Real-World Mode | Makes labels easier for non-technical users |
| Sensitivity Analysis | Tests parameter influence |
| Location Risk Heat Map | Shows simulated local risk intensity |

## Real-World Uses

- COVID-19 outbreak analysis
- Seasonal flu forecasting
- Vaccine planning
- Hospital capacity planning
- Public health education
- Scenario-based policy discussion
"""
    )


# ---------------------------------------------------
# TAB 4: COMPARE SCENARIOS
# ---------------------------------------------------
with tabs[3]:
    st.subheader("Compare Two Scenarios")
    st.caption("Compare custom Scenario A and Scenario B, plus automatic best/current/worst cases.")

    scenario_presets = {
        "Custom": {},
        "Early Intervention": {
            "beta_multiplier": 0.60,
            "gamma_multiplier": 1.10,
            "vac_boost": 0.04,
        },
        "Late Response": {
            "beta_multiplier": 1.05,
            "gamma_multiplier": 1.00,
            "vac_boost": 0.00,
        },
        "No Vaccination": {
            "beta_multiplier": 1.00,
            "gamma_multiplier": 1.00,
            "vac_override": 0.00,
        },
        "Aggressive Vaccination": {
            "beta_multiplier": 0.80,
            "gamma_multiplier": 1.00,
            "vac_boost": 0.10,
        },
    }

    preset_a = st.selectbox("Scenario A Preset", list(scenario_presets.keys()), index=0)
    preset_b = st.selectbox("Scenario B Preset", list(scenario_presets.keys()), index=1)

    def apply_policy_preset(base_beta, base_gamma, base_mu, base_vac, preset_name):
        preset_config = scenario_presets[preset_name]

        new_beta = base_beta * preset_config.get("beta_multiplier", 1.0)
        new_gamma = base_gamma * preset_config.get("gamma_multiplier", 1.0)
        new_mu = base_mu
        new_vac = base_vac + preset_config.get("vac_boost", 0.0)

        if "vac_override" in preset_config:
            new_vac = preset_config["vac_override"]

        return (
            min(max(new_beta, 0.0), 1.0),
            min(max(new_gamma, 0.0), 1.0),
            min(max(new_mu, 0.0), 1.0),
            min(max(new_vac, 0.0), 1.0),
        )

    default_beta_a, default_gamma_a, default_mu_a, default_vac_a = apply_policy_preset(
        beta,
        gamma,
        mu,
        vac,
        preset_a,
    )

    default_beta_b, default_gamma_b, default_mu_b, default_vac_b = apply_policy_preset(
        beta,
        gamma,
        mu,
        vac,
        preset_b,
    )

    with st.form("comparison_form"):
        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Scenario A")

            beta_a = st.slider("Beta A", 0.0, 1.0, default_beta_a, 0.01)
            gamma_a = st.slider("Gamma A", 0.0, 1.0, default_gamma_a, 0.01)

            mu_a = default_mu_a
            vac_a = default_vac_a

            if model_choice in ["SIRD", "SIRDV"]:
                mu_a = st.slider("Mu A", 0.0, 1.0, default_mu_a, 0.01)

            if model_choice == "SIRDV":
                vac_a = st.slider("Vaccination A", 0.0, 1.0, default_vac_a, 0.01)

        with colB:
            st.markdown("### Scenario B")

            beta_b = st.slider("Beta B", 0.0, 1.0, default_beta_b, 0.01)
            gamma_b = st.slider("Gamma B", 0.0, 1.0, default_gamma_b, 0.01)

            mu_b = default_mu_b
            vac_b = default_vac_b

            if model_choice in ["SIRD", "SIRDV"]:
                mu_b = st.slider("Mu B", 0.0, 1.0, default_mu_b, 0.01)

            if model_choice == "SIRDV":
                vac_b = st.slider("Vaccination B", 0.0, 1.0, default_vac_b, 0.01)

        compare_clicked = st.form_submit_button(
            "Run Comparison",
            type="primary",
            disabled=bool(errors),
        )

    if compare_clicked:
        with st.spinner("Comparing scenarios..."):
            df_a = run_model(model_choice, population, infected, recovered, beta_a, gamma_a, mu_a, vac_a, days)
            df_b = run_model(model_choice, population, infected, recovered, beta_b, gamma_b, mu_b, vac_b, days)

            best_df = run_model(
                model_choice,
                population,
                infected,
                recovered,
                max(beta * 0.65, 0),
                min(gamma * 1.25, 1),
                max(mu * 0.75, 0),
                min(vac + 0.05, 1),
                days,
            )

            current_df = run_model(
                model_choice,
                population,
                infected,
                recovered,
                beta,
                gamma,
                mu,
                vac,
                days,
            )

            worst_df = run_model(
                model_choice,
                population,
                infected,
                recovered,
                min(beta * 1.35, 1),
                max(gamma * 0.75, 0),
                min(mu * 1.25, 1),
                max(vac * 0.5, 0),
                days,
            )

            st.session_state.compare_results = {
                "df_a": df_a,
                "df_b": df_b,
                "best_df": best_df,
                "current_df": current_df,
                "worst_df": worst_df,
                "beta_a": beta_a,
                "gamma_a": gamma_a,
                "beta_b": beta_b,
                "gamma_b": gamma_b,
            }

    if st.session_state.compare_results is not None:
        result = st.session_state.compare_results

        fig_compare = go.Figure()

        compare_series = [
            ("Scenario A", result["df_a"], "solid"),
            ("Scenario B", result["df_b"], "dash"),
            ("Best Case", result["best_df"], "dot"),
            ("Current Case", result["current_df"], "solid"),
            ("Worst Case", result["worst_df"], "longdash"),
        ]

        for name, scenario_df, dash_style in compare_series:
            fig_compare.add_trace(
                go.Scatter(
                    x=scenario_df["Day"],
                    y=scenario_df["Infected"],
                    mode="lines",
                    name=f"{name} Infected",
                    line=dict(width=3, dash=dash_style),
                )
            )

        fig_compare.update_layout(
            title="Scenario Comparison",
            xaxis_title="Day",
            yaxis_title="Infected Population",
            template=template,
            hovermode="x unified",
            height=650,
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        m1 = metrics_from_df(result["df_a"], result["beta_a"], result["gamma_a"], population, hospitalization_rate)
        m2 = metrics_from_df(result["df_b"], result["beta_b"], result["gamma_b"], population, hospitalization_rate)

        comparison_table = pd.DataFrame(
            {
                "Metric": ["Peak Infected", "Day of Peak", "R₀", "Attack Rate", "ICU Overflow"],
                "Scenario A": [
                    round(m1["Peak Infected"], 2),
                    m1["Day of Peak"],
                    round(m1["R0"], 2),
                    round(m1["Attack Rate"], 2),
                    "Yes" if m1["Estimated Peak Hospitalizations"] > icu_capacity else "No",
                ],
                "Scenario B": [
                    round(m2["Peak Infected"], 2),
                    m2["Day of Peak"],
                    round(m2["R0"], 2),
                    round(m2["Attack Rate"], 2),
                    "Yes" if m2["Estimated Peak Hospitalizations"] > icu_capacity else "No",
                ],
            }
        )

        st.subheader("Comparison Table")
        st.dataframe(comparison_table, use_container_width=True)

        st.download_button(
            "Download Scenario Comparison",
            comparison_table.to_csv(index=False).encode(),
            file_name="scenario_comparison.csv",
            mime="text/csv",
        )


# ---------------------------------------------------
# TAB 5: SENSITIVITY ANALYSIS
# ---------------------------------------------------
with tabs[4]:
    st.subheader("Sensitivity Analysis")
    st.caption("Test how changing one parameter affects infection curves.")

    parameter_choice = st.selectbox(
        "Parameter to vary",
        ["Beta", "Gamma", "Mu", "Vaccination Rate"],
    )

    low_value, high_value = st.slider(
        "Parameter Range",
        0.0,
        1.0,
        (0.20, 0.80),
        0.01,
    )

    num_runs = st.slider("Number of Scenarios", 3, 9, 5, 1)

    if st.button("Run Sensitivity Analysis", type="primary", disabled=bool(errors)):
        values = [
            low_value + i * ((high_value - low_value) / (num_runs - 1))
            for i in range(num_runs)
        ]

        sensitivity_data = []

        with st.spinner("Running sensitivity analysis..."):
            for value in values:
                beta_s = beta
                gamma_s = gamma
                mu_s = mu
                vac_s = vac

                if parameter_choice == "Beta":
                    beta_s = value
                elif parameter_choice == "Gamma":
                    gamma_s = value
                elif parameter_choice == "Mu":
                    mu_s = value
                elif parameter_choice == "Vaccination Rate":
                    vac_s = value

                df_s = run_model(
                    model_choice,
                    population,
                    infected,
                    recovered,
                    beta_s,
                    gamma_s,
                    mu_s,
                    vac_s,
                    days,
                )

                m = metrics_from_df(df_s, beta_s, gamma_s, population, hospitalization_rate)

                sensitivity_data.append(
                    {
                        "value": value,
                        "df": df_s,
                        "metrics": m,
                    }
                )

        st.session_state.sensitivity_results = {
            "parameter": parameter_choice,
            "data": sensitivity_data,
        }

    if st.session_state.sensitivity_results is not None:
        result = st.session_state.sensitivity_results

        fig_sens = go.Figure()

        infected_matrix = pd.DataFrame(
            {
                f"{result['parameter']}={item['value']:.2f}": item["df"]["Infected"]
                for item in result["data"]
            }
        )

        lower_band = infected_matrix.min(axis=1)
        upper_band = infected_matrix.max(axis=1)

        fig_sens.add_trace(
            go.Scatter(
                x=result["data"][0]["df"]["Day"],
                y=upper_band,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig_sens.add_trace(
            go.Scatter(
                x=result["data"][0]["df"]["Day"],
                y=lower_band,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(214,39,40,0.18)",
                line=dict(width=0),
                name="Uncertainty Band",
                hoverinfo="skip",
            )
        )

        for item in result["data"]:
            fig_sens.add_trace(
                go.Scatter(
                    x=item["df"]["Day"],
                    y=item["df"]["Infected"],
                    mode="lines",
                    name=f"{result['parameter']}={item['value']:.2f}",
                    line=dict(width=3),
                )
            )

        if hospitalization_rate > 0:
            fig_sens.add_hline(
                y=icu_capacity / hospitalization_rate,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="ICU-equivalent infected threshold",
                annotation_position="top left",
            )

        fig_sens.update_layout(
            title=f"Sensitivity Analysis: {result['parameter']}",
            xaxis_title="Day",
            yaxis_title="Infected Population",
            template=template,
            hovermode="x unified",
            height=650,
        )

        st.plotly_chart(fig_sens, use_container_width=True)

        sens_table = pd.DataFrame(
            [
                {
                    result["parameter"]: round(item["value"], 3),
                    "Peak Infected": round(item["metrics"]["Peak Infected"], 2),
                    "Day of Peak": item["metrics"]["Day of Peak"],
                    "R0": round(item["metrics"]["R0"], 2),
                    "Attack Rate (%)": round(item["metrics"]["Attack Rate"], 2),
                    "ICU Overflow": "Yes"
                    if item["metrics"]["Estimated Peak Hospitalizations"] > icu_capacity
                    else "No",
                }
                for item in result["data"]
            ]
        )

        st.subheader("Sensitivity Summary Table")
        st.dataframe(sens_table, use_container_width=True)

        if sens_table["ICU Overflow"].eq("Yes").any():
            st.warning("Higher-risk sensitivity scenarios exceed ICU capacity.")
        else:
            st.success("None of the sensitivity scenarios exceed ICU capacity.")

        st.download_button(
            "Download Sensitivity Results",
            sens_table.to_csv(index=False).encode(),
            file_name="sensitivity_analysis_results.csv",
            mime="text/csv",
        )


# ---------------------------------------------------
# TAB 6: DAY-BY-DAY VIEW
# ---------------------------------------------------
with tabs[5]:
    st.subheader("Day-by-Day Outbreak View")

    if st.session_state.simulation_df is None:
        st.info("Run a simulation first to use the day-by-day view.")

    else:
        df = st.session_state.simulation_df

        day = st.slider("Select Day", 0, len(df) - 1, 0)
        row = df.iloc[day]

        st.markdown(f"### Day {day}")

        d1, d2, d3, d4, d5 = st.columns(5)

        d1.metric("Susceptible", f"{row['Susceptible']:,.0f}")
        d2.metric("Infected", f"{row['Infected']:,.0f}")
        d3.metric("Recovered", f"{row['Recovered']:,.0f}")
        d4.metric("Dead", f"{row['Dead']:,.0f}")
        d5.metric("Vaccinated", f"{row['Vaccinated']:,.0f}")

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=["Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"],
                    y=[
                        row["Susceptible"],
                        row["Infected"],
                        row["Recovered"],
                        row["Dead"],
                        row["Vaccinated"],
                    ],
                )
            ]
        )

        fig_bar.update_layout(
            title=f"Compartment Counts on Day {day}",
            yaxis_title="Population",
            template=template,
            height=500,
        )

        st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------
# TAB 7: INTERVENTION TIMING ANALYSIS
# ---------------------------------------------------
with tabs[6]:
    st.subheader("Intervention Timing Analysis")
    st.caption("Test how delaying intervention changes outbreak outcomes.")

    timing_days = st.multiselect(
        "Intervention Days to Test",
        [10, 20, 30, 40, 50, 60, 75, 90],
        default=[10, 20, 30, 40, 50],
    )

    if st.button("Run Intervention Timing Analysis", type="primary", disabled=bool(errors)):
        timing_data = []

        with st.spinner("Testing intervention timing..."):
            for day_value in timing_days:
                if day_value < days:
                    df_t = run_model_with_intervention(
                        model_choice,
                        population,
                        infected,
                        recovered,
                        beta,
                        gamma,
                        mu,
                        vac,
                        days,
                        True,
                        day_value,
                        reduced_beta,
                        increased_vac,
                    )

                    m = metrics_from_df(df_t, beta, gamma, population, hospitalization_rate)

                    timing_data.append(
                        {
                            "Intervention Day": day_value,
                            "df": df_t,
                            "metrics": m,
                        }
                    )

        st.session_state.timing_results = timing_data

    if st.session_state.timing_results is not None:
        fig_timing = go.Figure()

        for item in st.session_state.timing_results:
            fig_timing.add_trace(
                go.Scatter(
                    x=item["df"]["Day"],
                    y=item["df"]["Infected"],
                    mode="lines",
                    name=f"Day {item['Intervention Day']}",
                    line=dict(width=3),
                )
            )

        fig_timing.update_layout(
            title="Effect of Intervention Timing",
            xaxis_title="Day",
            yaxis_title="Infected Population",
            template=template,
            hovermode="x unified",
            height=650,
        )

        st.plotly_chart(fig_timing, use_container_width=True)

        timing_table = pd.DataFrame(
            [
                {
                    "Intervention Day": item["Intervention Day"],
                    "Peak Infected": round(item["metrics"]["Peak Infected"], 2),
                    "Day of Peak": item["metrics"]["Day of Peak"],
                    "Attack Rate (%)": round(item["metrics"]["Attack Rate"], 2),
                    "ICU Overflow": "Yes"
                    if item["metrics"]["Estimated Peak Hospitalizations"] > icu_capacity
                    else "No",
                }
                for item in st.session_state.timing_results
            ]
        )

        st.subheader("Timing Summary Table")
        st.dataframe(timing_table, use_container_width=True)


# ---------------------------------------------------
# TAB 8: LOCATION-BASED RISK HEAT MAP
# ---------------------------------------------------
with tabs[7]:
    st.subheader("Location-Based Risk Heat Map")
    st.caption("Simulated local risk intensity based on the selected location and current model assumptions.")

    locations = {
        "New York, NY": (40.7128, -74.0060),
        "Chicago, IL": (41.8781, -87.6298),
        "Atlanta, GA": (33.7490, -84.3880),
        "Miami, FL": (25.7617, -80.1918),
        "Los Angeles, CA": (34.0522, -118.2437),
        "Houston, TX": (29.7604, -95.3698),
        "Philadelphia, PA": (39.9526, -75.1652),
        "Boston, MA": (42.3601, -71.0589),
        "Washington, DC": (38.9072, -77.0369),
        "Seattle, WA": (47.6062, -122.3321),
    }

    selected_location = st.selectbox("Choose Location", list(locations.keys()))

    heat_metric = st.selectbox(
        "Risk Metric",
        ["Peak Infected", "Attack Rate", "Estimated Peak Hospitalizations"],
    )

    num_heat_points = st.slider("Heat Map Smoothness", 150, 700, 400, 50)
    spread = st.slider("Geographic Spread", 0.03, 0.15, 0.08, 0.01)

    if st.button("Generate Location Risk Heat Map", type="primary", disabled=bool(errors)):
        with st.spinner("Generating simulated local risk map..."):
            base_df = run_model(
                model_choice,
                population,
                infected,
                recovered,
                beta,
                gamma,
                mu,
                vac,
                days,
            )

            base_metrics = metrics_from_df(
                base_df,
                beta,
                gamma,
                population,
                hospitalization_rate,
            )

            center_lat, center_lon = locations[selected_location]
            risk_base = base_metrics[heat_metric]

            heat_points = []

            for _ in range(num_heat_points):
                lat_offset = random.gauss(0, spread / 3)
                lon_offset = random.gauss(0, spread / 3)

                distance = math.sqrt(lat_offset**2 + lon_offset**2)
                decay = max(0.15, 1 - (distance / spread))

                local_noise = random.uniform(0.7, 1.3)

                intensity = risk_base * decay * local_noise

                lat = center_lat + lat_offset
                lon = center_lon + lon_offset

                heat_points.append([lat, lon, intensity])

            st.session_state.heatmap_results = {
                "location": selected_location,
                "center": (center_lat, center_lon),
                "points": heat_points,
                "metric": heat_metric,
                "risk_base": risk_base,
            }

    if st.session_state.heatmap_results is not None:
        result = st.session_state.heatmap_results

        st.metric(
            f"Base Simulated Risk: {result['metric']}",
            f"{result['risk_base']:,.2f}",
        )

        if FOLIUM_AVAILABLE:
            center_lat, center_lon = result["center"]

            fmap = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles="CartoDB positron",
            )

            HeatMap(
                result["points"],
                radius=22,
                blur=20,
                max_zoom=13,
            ).add_to(fmap)

            folium.Marker(
                [center_lat, center_lon],
                tooltip=result["location"],
                popup=f"{result['location']}<br>{result['metric']}: {result['risk_base']:,.2f}",
            ).add_to(fmap)

            st_folium(fmap, width=None, height=600)

        else:
            st.info(
                "For interactive map tiles, add `folium` and `streamlit-folium` to requirements.txt. Showing Plotly fallback map."
            )

            heat_df = pd.DataFrame(result["points"], columns=["lat", "lon", "risk"])

            fig_map = go.Figure(
                go.Densitymapbox(
                    lat=heat_df["lat"],
                    lon=heat_df["lon"],
                    z=heat_df["risk"],
                    radius=18,
                    colorscale="Reds",
                    name="Risk Intensity",
                )
            )

            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={
                    "lat": result["center"][0],
                    "lon": result["center"][1],
                },
                mapbox_zoom=10,
                height=650,
                margin=dict(l=0, r=0, t=40, b=0),
                title=f"Simulated Risk Heat Map: {result['location']}",
            )

            st.plotly_chart(fig_map, use_container_width=True)

        heat_df_download = pd.DataFrame(
            result["points"],
            columns=["Latitude", "Longitude", "Risk Intensity"],
        )

        st.download_button(
            "Download Heat Map Risk Points",
            heat_df_download.to_csv(index=False).encode(),
            file_name="location_risk_heatmap_points.csv",
            mime="text/csv",
        )


# ---------------------------------------------------
# TAB 9: METHODS & ASSUMPTIONS
# ---------------------------------------------------
with tabs[8]:
    st.subheader("Methods & Assumptions")

    st.markdown(
        """
## Modeling Approach

This dashboard uses simplified compartmental infectious disease models:

- **SIR:** Susceptible, Infected, Recovered
- **SIRD:** Susceptible, Infected, Recovered, Dead
- **SIRDV:** Susceptible, Infected, Recovered, Dead, Vaccinated

## Added Analytics

- Scenario comparison
- Policy impact scoring
- Timeline callouts
- Real-world mode labels
- Sensitivity uncertainty band
- Intervention timing analysis
- Healthcare capacity tracking
- Simulated location-based risk heat map

## Assumptions

- The population is treated as a closed system.
- Parameters remain constant unless an intervention is enabled.
- The model is deterministic and does not include random uncertainty.
- Hospitalizations are estimated from peak infected counts using a user-defined rate.
- ICU capacity is used as a simplified healthcare strain threshold.
- The location heat map is simulated and not based on live surveillance data.

## Limitations

- This is not a calibrated real-world forecast.
- It does not account for age, geography, behavior change, reinfection, testing, or reporting delays.
- The location heat map is for visualization and educational purposes only.

## Why This Matters

This tool demonstrates how small differences in transmission, recovery, vaccination, and intervention timing can produce large differences in outbreak severity and healthcare strain.
"""
    )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built by Omar Rulida Abdul-Rahman | MPH Candidate | Python + Streamlit")
