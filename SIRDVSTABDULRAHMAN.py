import io
import time
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------
# LOAD ASSIGNMENT HELPER FUNCTION
# ---------------------------------------------------
def load_run_sim():
    """Load run_sim() from the provided assignment helper file."""
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
                file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.run_sim

    raise FileNotFoundError(
        "Simulation helper file not found. Make sure Assignment3_Functions_Solution.py is in the same folder."
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
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------
# RESET BUTTON
# ---------------------------------------------------
if st.sidebar.button("Reset Dashboard"):
    for key in session_defaults:
        st.session_state[key] = None
    st.rerun()


# ---------------------------------------------------
# CORE MODEL FUNCTIONS
# ---------------------------------------------------
def run_model(model_choice, pop, infected, recovered, beta, gamma, mu, vac, days):
    """Run one SIR/SIRD/SIRDV simulation using the assignment helper function."""
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
    """Run simulation with optional intervention starting on a chosen day."""
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
    """Return visible model compartments."""
    if model == "SIR":
        return ["Susceptible", "Infected", "Recovered"]

    if model == "SIRD":
        return ["Susceptible", "Infected", "Recovered", "Dead"]

    return ["Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"]


def convert_to_percent(df, columns, population):
    """Convert selected columns from counts to percentages."""
    df_percent = df.copy()

    for col in columns:
        df_percent[col] = (df_percent[col] / population) * 100

    return df_percent


# ---------------------------------------------------
# METRICS, RISK, INSIGHTS
# ---------------------------------------------------
def metrics_from_df(df, beta, gamma, population, hospitalization_rate):
    """Calculate summary metrics from simulation output."""
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
    """Detect current outbreak phase from infection curve."""
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


def policy_recommendation(metrics, icu_capacity):
    """Generate public-health style recommendation."""
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
    """Classify overall risk."""
    if metrics["Estimated Peak Hospitalizations"] > icu_capacity or metrics["R0"] > 3:
        return "High Risk", "error"

    if metrics["R0"] > 1 or metrics["Attack Rate"] > 25:
        return "Moderate Risk", "warning"

    return "Controlled / Lower Risk", "success"


def readiness_grade(metrics, icu_capacity, vac):
    """Create a simple public health readiness grade."""
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


def generate_insights(df, beta, gamma, mu, vac, population, hospitalization_rate):
    """Generate plain-English interpretation."""
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
    """Create a polished executive brief."""
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
    icu_capacity=None,
    intervention_day=None,
):
    """Create main Plotly time-series chart."""
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
        fig.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=3, color=colors[col]),
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
            line_color="red",
            annotation_text="ICU Capacity",
            annotation_position="top left",
        )

    if intervention_day is not None:
        fig.add_vline(
            x=intervention_day,
            line_dash="dash",
            line_color="green",
            annotation_text="Intervention",
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
    """Create estimated hospitalizations over time."""
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
        line_color="red",
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
    """Create parameter summary download."""
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
Post-Intervention Beta: {params["reduced_beta"]}
Post-Intervention Vaccination Rate: {params["increased_vac"]}
"""


def create_full_report(params, metrics, insights, recommendation, phase, grade, brief):
    """Create a formal downloadable text report."""
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
Post-Intervention Beta: {params["reduced_beta"]}
Post-Intervention Vaccination Rate: {params["increased_vac"]}

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

Automated Interpretation:
{chr(10).join("- " + insight for insight in insights)}

Policy Recommendation:
{recommendation}

Methods and Assumptions:
This dashboard uses simplified deterministic compartmental modeling.
Results are educational estimates and should not be used as real-world forecasts.
"""


def plot_download_buttons(fig, model_choice):
    """Create plot download buttons."""
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
    "Beta (Infection Rate)",
    0.0,
    1.0,
    defaults["beta"],
    0.01,
)

gamma = st.sidebar.slider(
    "Gamma (Recovery Rate)",
    0.0,
    1.0,
    defaults["gamma"],
    0.01,
)

mu = 0.0
vac = 0.0

if model_choice in ["SIRD", "SIRDV"]:
    mu = st.sidebar.slider(
        "Mu (Death Rate)",
        0.0,
        1.0,
        defaults["mu"],
        0.01,
    )

if model_choice == "SIRDV":
    vac = st.sidebar.slider(
        "Vaccination Rate",
        0.0,
        1.0,
        defaults["vac"],
        0.01,
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
    "Post-Intervention Beta",
    0.0,
    1.0,
    max(beta * 0.6, 0.0),
    0.01,
)

increased_vac = st.sidebar.slider(
    "Post-Intervention Vaccination Rate",
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

Use this dashboard to explore how transmission, recovery, mortality, vaccination, healthcare capacity, and intervention timing shape infectious disease outcomes.
"""
)

with st.expander("About This Dashboard"):
    st.write(
        "This dashboard is an educational public health simulation tool. It combines compartmental disease models, "
        "interactive visualization, scenario comparison, healthcare capacity analysis, intervention timing, and downloadable reports."
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
        brief = executive_brief(metrics, phase, recommendation, grade)

        st.subheader("Executive Summary")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Risk Level", risk)
        k2.metric("R₀", f"{metrics['R0']:.2f}")
        k3.metric("Peak Day", metrics["Day of Peak"])
        k4.metric("ICU Status", "Exceeded" if metrics["Estimated Peak Hospitalizations"] > params["icu_capacity"] else "Within Capacity")
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
        c8.metric("Peak Hospitalizations", f"{metrics['Estimated Peak Hospitalizations']:,.0f}")

        st.subheader("Healthcare Capacity Timeline")
        fig_health = make_healthcare_timeline(
            df,
            params["hospitalization_rate"],
            params["icu_capacity"],
            template,
        )
        st.plotly_chart(fig_health, use_container_width=True)

        if params["intervention_enabled"]:
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

            infections_prevented = baseline_metrics["Cumulative Cases"] - metrics["Cumulative Cases"]
            peak_reduction = baseline_metrics["Peak Infected"] - metrics["Peak Infected"]

            st.subheader("Intervention Impact Score")

            i1, i2 = st.columns(2)
            i1.metric("Estimated Infections Prevented", f"{infections_prevented:,.0f}")
            i2.metric("Peak Reduction", f"{peak_reduction:,.0f}")

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
                ),
                file_name=f"{params['model_choice'].lower()}_simulation_report.txt",
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
| Intervention Timeline | Models delayed policy response |
| Intervention Impact Score | Estimates prevented infections |
| Sensitivity Analysis | Tests parameter influence |
| Day-by-Day View | Shows outbreak state each day |
| Healthcare Timeline | Tracks estimated hospital burden |

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

    with st.form("comparison_form"):
        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Scenario A")

            beta_a = st.slider("Beta A", 0.0, 1.0, beta, 0.01)
            gamma_a = st.slider("Gamma A", 0.0, 1.0, gamma, 0.01)

            mu_a = mu
            vac_a = vac

            if model_choice in ["SIRD", "SIRDV"]:
                mu_a = st.slider("Mu A", 0.0, 1.0, mu, 0.01)

            if model_choice == "SIRDV":
                vac_a = st.slider("Vaccination A", 0.0, 1.0, vac, 0.01)

        with colB:
            st.markdown("### Scenario B")

            beta_b = st.slider("Beta B", 0.0, 1.0, min(beta + 0.2, 1.0), 0.01)
            gamma_b = st.slider("Gamma B", 0.0, 1.0, gamma, 0.01)

            mu_b = mu
            vac_b = vac

            if model_choice in ["SIRD", "SIRDV"]:
                mu_b = st.slider("Mu B", 0.0, 1.0, mu, 0.01)

            if model_choice == "SIRDV":
                vac_b = st.slider("Vaccination B", 0.0, 1.0, vac, 0.01)

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

        fig_compare.add_trace(
            go.Scatter(
                x=result["df_a"]["Day"],
                y=result["df_a"]["Infected"],
                mode="lines",
                name="Scenario A Infected",
                line=dict(width=3),
            )
        )

        fig_compare.add_trace(
            go.Scatter(
                x=result["df_b"]["Day"],
                y=result["df_b"]["Infected"],
                mode="lines",
                name="Scenario B Infected",
                line=dict(width=3, dash="dash"),
            )
        )

        fig_compare.add_trace(
            go.Scatter(
                x=result["best_df"]["Day"],
                y=result["best_df"]["Infected"],
                mode="lines",
                name="Best Case",
                line=dict(width=2, dash="dot"),
            )
        )

        fig_compare.add_trace(
            go.Scatter(
                x=result["current_df"]["Day"],
                y=result["current_df"]["Infected"],
                mode="lines",
                name="Current Case",
                line=dict(width=2),
            )
        )

        fig_compare.add_trace(
            go.Scatter(
                x=result["worst_df"]["Day"],
                y=result["worst_df"]["Infected"],
                mode="lines",
                name="Worst Case",
                line=dict(width=2, dash="longdash"),
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
                "Metric": ["Peak Infected", "Day of Peak", "R₀", "Attack Rate"],
                "Scenario A": [
                    round(m1["Peak Infected"], 2),
                    m1["Day of Peak"],
                    round(m1["R0"], 2),
                    round(m1["Attack Rate"], 2),
                ],
                "Scenario B": [
                    round(m2["Peak Infected"], 2),
                    m2["Day of Peak"],
                    round(m2["R0"], 2),
                    round(m2["Attack Rate"], 2),
                ],
            }
        )

        st.subheader("Comparison Table")
        st.dataframe(comparison_table, use_container_width=True)


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

    num_runs = st.slider("Number of Scenarios", 3, 7, 5, 1)

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
                }
                for item in result["data"]
            ]
        )

        st.subheader("Sensitivity Summary Table")
        st.dataframe(sens_table, use_container_width=True)

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
                }
                for item in st.session_state.timing_results
            ]
        )

        st.subheader("Timing Summary Table")
        st.dataframe(timing_table, use_container_width=True)


# ---------------------------------------------------
# TAB 8: METHODS & ASSUMPTIONS
# ---------------------------------------------------
with tabs[7]:
    st.subheader("Methods & Assumptions")

    st.markdown(
        """
## Modeling Approach

This dashboard uses simplified compartmental infectious disease models:

- **SIR:** Susceptible, Infected, Recovered
- **SIRD:** Susceptible, Infected, Recovered, Dead
- **SIRDV:** Susceptible, Infected, Recovered, Dead, Vaccinated

## Assumptions

- The population is treated as a closed system.
- Parameters remain constant unless an intervention is enabled.
- The model is deterministic and does not include random uncertainty.
- Hospitalizations are estimated from peak infected counts using a user-defined rate.
- ICU capacity is used as a simplified healthcare strain threshold.

## Limitations

- This is not a calibrated real-world forecast.
- It does not account for age, geography, behavior change, reinfection, testing, or reporting delays.
- It is intended for education, coursework, and scenario exploration.

## Suggested Interpretation

Use the dashboard to compare relative scenarios rather than predict exact real-world outcomes.
"""
    )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built by Omar Rulida Abdul-Rahman | MPH Candidate | Python + Streamlit")
