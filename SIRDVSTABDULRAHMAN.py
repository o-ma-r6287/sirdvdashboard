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
    layout="wide"
)

st.title("🦠 Epidemiological Decision Dashboard")
st.caption("Interactive disease spread analytics for SIR / SIRD / SIRDV models")
st.caption("Built by Omar Rulida Abdul-Rahman | MPH Candidate")


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "simulation_df" not in st.session_state:
    st.session_state.simulation_df = None

if "simulation_params" not in st.session_state:
    st.session_state.simulation_params = None

if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

if "sensitivity_results" not in st.session_state:
    st.session_state.sensitivity_results = None


# ---------------------------------------------------
# MODEL FUNCTIONS
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
        model_choice=model_choice
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
    if not intervention_enabled:
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

    if intervention_day <= 1 or intervention_day >= days:
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
    """Return the relevant compartments for each model."""
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
# METRICS AND INSIGHTS
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


def generate_insights(df, beta, gamma, mu, vac, population, hospitalization_rate):
    """Generate plain-English interpretation of the simulation."""
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
        insights.append(
            "Peak infection occurs early, indicating fast disease spread."
        )
    elif metrics["Day of Peak"] >= 90:
        insights.append(
            "Peak infection occurs later, suggesting a slower-moving outbreak curve."
        )
    else:
        insights.append(
            "Peak infection occurs in the middle of the simulation period."
        )

    if gamma > beta:
        insights.append(
            "Recovery is stronger than transmission, which helps flatten the curve."
        )

    if vac >= 0.05:
        insights.append(
            "Vaccination is meaningfully reducing the susceptible population over time."
        )

    if mu >= 0.03:
        insights.append(
            "Death rate is elevated, making this a higher-severity scenario."
        )

    if metrics["Attack Rate"] >= 50:
        insights.append(
            "A large share of the population experiences infection or death by the end of the simulation."
        )
    elif metrics["Attack Rate"] <= 10:
        insights.append(
            "Final infection burden remains relatively low."
        )

    if hospitalization_rate >= 0.10:
        insights.append(
            "Hospital burden may become significant because the hospitalization assumption is high."
        )

    return insights


def policy_recommendation(metrics, icu_capacity):
    """Generate a simple public-health style recommendation."""
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
    """Classify the overall scenario risk level."""
    if metrics["Estimated Peak Hospitalizations"] > icu_capacity or metrics["R0"] > 3:
        return "High Risk", "error"

    if metrics["R0"] > 1 or metrics["Attack Rate"] > 25:
        return "Moderate Risk", "warning"

    return "Controlled / Lower Risk", "success"


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
    """Create the main Plotly time-series chart."""
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
                line=dict(
                    width=3,
                    dash="dot",
                    color=colors["Cumulative Cases"],
                ),
            )
        )

    peak_idx = int(df["Infected"].idxmax())
    peak_val = float(df["Infected"].max())

    fig.add_annotation(
        x=peak_idx,
        y=peak_val,
        text="Peak Infection",
        showarrow=True,
        arrowhead=2,
        ax=35,
        ay=-45,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d62728",
        borderwidth=1,
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


# ---------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------
def create_parameter_summary(params):
    """Create a downloadable parameter summary."""
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

Notes:
Beta controls infection pressure.
Gamma controls recovery.
Mu controls mortality.
Vaccination rate moves susceptible people into the vaccinated compartment.
Hospitalization rate estimates peak healthcare burden.
ICU capacity is used as a healthcare strain benchmark.
"""


def create_full_report(params, metrics, insights, recommendation):
    """Create a downloadable text report."""
    return f"""Epidemiological Decision Dashboard Report

Model: {params["model_choice"]}

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

Automated Interpretation:
{chr(10).join("- " + insight for insight in insights)}

Policy Recommendation:
{recommendation}

Disclaimer:
This dashboard is intended for educational and exploratory purposes only.
"""


def plot_download_buttons(fig, model_choice):
    """Create chart download buttons."""
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
        st.info(
            "PNG export needs `kaleido`. Add `kaleido` to requirements.txt to enable PNG downloads."
        )


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

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["SIR", "SIRD", "SIRDV"],
)

population = st.sidebar.number_input(
    "Total Population",
    min_value=1,
    value=defaults["population"],
    help="Total number of people in the modeled population.",
)

infected = st.sidebar.number_input(
    "Initial Infected",
    min_value=0,
    value=defaults["infected"],
    help="Number of people infected at the start.",
)

recovered = st.sidebar.number_input(
    "Initial Recovered",
    min_value=0,
    value=defaults["recovered"],
    help="Number of people already recovered at the start.",
)

beta = st.sidebar.slider(
    "Beta (Infection Rate)",
    0.0,
    1.0,
    defaults["beta"],
    0.01,
    help="Higher beta means faster transmission.",
)

gamma = st.sidebar.slider(
    "Gamma (Recovery Rate)",
    0.0,
    1.0,
    defaults["gamma"],
    0.01,
    help="Higher gamma means faster recovery.",
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
        help="Rate at which infected individuals move into the death compartment.",
    )

if model_choice == "SIRDV":
    vac = st.sidebar.slider(
        "Vaccination Rate",
        0.0,
        1.0,
        defaults["vac"],
        0.01,
        help="Rate at which susceptible individuals become vaccinated.",
    )

days = st.sidebar.slider(
    "Days",
    10,
    365,
    defaults["days"],
    help="Number of days to simulate.",
)

hospitalization_rate = st.sidebar.slider(
    "Estimated Hospitalization Rate",
    0.0,
    1.0,
    0.05,
    0.01,
    help="Estimated proportion of peak infected individuals requiring hospitalization.",
)

icu_capacity = st.sidebar.number_input(
    "ICU Capacity",
    min_value=0,
    value=100,
    step=10,
    help="Healthcare capacity benchmark for peak hospitalization strain.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Intervention Timeline")

intervention_enabled = st.sidebar.toggle(
    "Enable intervention",
    value=False,
)

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

theme = st.sidebar.radio(
    "Chart Theme",
    ["Light", "Dark"],
)

template = "plotly_dark" if theme == "Dark" else "plotly_white"

show_percent = st.sidebar.toggle(
    "Show chart as percentage",
    value=False,
)

show_total_check = st.sidebar.toggle(
    "Show population conservation check",
    value=True,
)

show_cumulative = st.sidebar.toggle(
    "Show cumulative cases curve",
    value=True,
)


# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------
errors = []

if infected > population:
    errors.append("Initial infected cannot exceed total population.")

if infected + recovered > population:
    errors.append("Initial infected + recovered cannot exceed total population.")

if gamma == 0:
    st.sidebar.warning(
        "Gamma is 0, so R₀ will be shown as 0 to avoid division by zero."
    )

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
            intervention_day=params["intervention_day"]
            if params["intervention_enabled"]
            else None,
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

        recommendation = policy_recommendation(
            metrics,
            params["icu_capacity"],
        )

        risk, risk_style = risk_level(
            metrics,
            params["icu_capacity"],
        )

        st.subheader("Executive Summary")

        if risk_style == "error":
            st.error(f"Overall Assessment: {risk}")
        elif risk_style == "warning":
            st.warning(f"Overall Assessment: {risk}")
        else:
            st.success(f"Overall Assessment: {risk}")

        st.subheader("Simulation Dashboard")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)

        c1.metric("Peak Infected", f"{metrics['Peak Infected']:,.0f}")
        c2.metric("Day of Peak", f"{metrics['Day of Peak']}")
        c3.metric("R₀", f"{metrics['R0']:.2f}")
        c4.metric("Attack Rate", f"{metrics['Attack Rate']:.1f}%")

        c5.metric("Recovered", f"{metrics['Recovered']:,.0f}")
        c6.metric("Deaths", f"{metrics['Deaths']:,.0f}")
        c7.metric("Vaccinated", f"{metrics['Vaccinated']:,.0f}")
        c8.metric(
            "Peak Hospitalizations",
            f"{metrics['Estimated Peak Hospitalizations']:,.0f}",
        )

        st.success(
            f"Key takeaway: this scenario peaks on day {metrics['Day of Peak']} "
            f"with {metrics['Peak Infected']:,.0f} infected individuals."
        )

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
                [
                    "Susceptible",
                    "Infected",
                    "Recovered",
                    "Dead",
                    "Vaccinated",
                ]
            ].sum(axis=1)

            max_drift = abs(df_check["Total"] - params["population"]).max()

            col_check1, col_check2 = st.columns(2)
            col_check1.metric(
                "Expected Population",
                f"{params['population']:,.0f}",
            )
            col_check2.metric(
                "Maximum Population Drift",
                f"{max_drift:,.4f}",
            )

            fig_total = go.Figure()

            fig_total.add_trace(
                go.Scatter(
                    x=df_check["Day"],
                    y=df_check["Total"],
                    mode="lines",
                    name="Total Population",
                    line=dict(width=3),
                )
            )

            fig_total.update_layout(
                title="Total Population Over Time",
                xaxis_title="Day",
                yaxis_title="Total Population",
                template=template,
                height=350,
            )

            st.plotly_chart(fig_total, use_container_width=True)

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
                ),
                file_name=f"{params['model_choice'].lower()}_simulation_report.txt",
                mime="text/plain",
            )

        with dl4:
            plot_download_buttons(
                fig,
                params["model_choice"],
            )


# ---------------------------------------------------
# TAB 2: DATA TABLE
# ---------------------------------------------------
with tabs[1]:
    if st.session_state.simulation_df is None:
        st.info("Run a simulation first to view the data table.")

    else:
        st.subheader("Simulation Data")
        st.dataframe(
            st.session_state.simulation_df,
            use_container_width=True,
        )

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

## Added Dashboard Features

| Feature | Purpose |
|---|---|
| R₀ | Estimates outbreak growth potential |
| Attack Rate | Shows share infected or dead |
| ICU Capacity | Tests healthcare strain |
| Intervention Timeline | Models delayed policy response |
| Sensitivity Analysis | Tests how parameter changes affect outcomes |
| Day-by-Day View | Shows outbreak state at each time point |

## Real-World Uses

- COVID-19 outbreak analysis
- Seasonal flu forecasting
- Vaccine planning
- Hospital capacity planning
- Public health education
- Scenario-based policy discussion

## Important Note

This dashboard is intended for education and exploration, not real-world forecasting.
"""
    )


# ---------------------------------------------------
# TAB 4: COMPARE SCENARIOS
# ---------------------------------------------------
with tabs[3]:
    st.subheader("Compare Two Scenarios")
    st.caption("Compare how different parameter choices affect outbreak outcomes.")

    with st.form("comparison_form"):
        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Scenario A")

            beta_a = st.slider(
                "Beta A",
                0.0,
                1.0,
                beta,
                0.01,
            )

            gamma_a = st.slider(
                "Gamma A",
                0.0,
                1.0,
                gamma,
                0.01,
            )

            mu_a = mu
            vac_a = vac

            if model_choice in ["SIRD", "SIRDV"]:
                mu_a = st.slider(
                    "Mu A",
                    0.0,
                    1.0,
                    mu,
                    0.01,
                )

            if model_choice == "SIRDV":
                vac_a = st.slider(
                    "Vaccination A",
                    0.0,
                    1.0,
                    vac,
                    0.01,
                )

        with colB:
            st.markdown("### Scenario B")

            beta_b = st.slider(
                "Beta B",
                0.0,
                1.0,
                min(beta + 0.2, 1.0),
                0.01,
            )

            gamma_b = st.slider(
                "Gamma B",
                0.0,
                1.0,
                gamma,
                0.01,
            )

            mu_b = mu
            vac_b = vac

            if model_choice in ["SIRD", "SIRDV"]:
                mu_b = st.slider(
                    "Mu B",
                    0.0,
                    1.0,
                    mu,
                    0.01,
                )

            if model_choice == "SIRDV":
                vac_b = st.slider(
                    "Vaccination B",
                    0.0,
                    1.0,
                    vac,
                    0.01,
                )

        compare_clicked = st.form_submit_button(
            "Run Comparison",
            type="primary",
            disabled=bool(errors),
        )

    if compare_clicked:
        with st.spinner("Comparing scenarios..."):
            df_a = run_model(
                model_choice,
                population,
                infected,
                recovered,
                beta_a,
                gamma_a,
                mu_a,
                vac_a,
                days,
            )

            df_b = run_model(
                model_choice,
                population,
                infected,
                recovered,
                beta_b,
                gamma_b,
                mu_b,
                vac_b,
                days,
            )

            st.session_state.compare_results = {
                "df_a": df_a,
                "df_b": df_b,
                "beta_a": beta_a,
                "gamma_a": gamma_a,
                "beta_b": beta_b,
                "gamma_b": gamma_b,
            }

    if st.session_state.compare_results is not None:
        result = st.session_state.compare_results
        df_a = result["df_a"]
        df_b = result["df_b"]

        fig_compare = go.Figure()

        compare_cols = ["Infected", "Recovered"]

        if model_choice in ["SIRD", "SIRDV"]:
            compare_cols.append("Dead")

        if model_choice == "SIRDV":
            compare_cols.append("Vaccinated")

        for col in compare_cols:
            fig_compare.add_trace(
                go.Scatter(
                    x=df_a["Day"],
                    y=df_a[col],
                    mode="lines",
                    name=f"Scenario A {col}",
                    line=dict(width=3),
                )
            )

            fig_compare.add_trace(
                go.Scatter(
                    x=df_b["Day"],
                    y=df_b[col],
                    mode="lines",
                    name=f"Scenario B {col}",
                    line=dict(width=3, dash="dash"),
                )
            )

        fig_compare.update_layout(
            title="Scenario Comparison",
            xaxis_title="Day",
            yaxis_title="Population",
            template=template,
            hovermode="x unified",
            height=650,
            legend_title="Scenario / Compartment",
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        m1 = metrics_from_df(
            df_a,
            result["beta_a"],
            result["gamma_a"],
            population,
            hospitalization_rate,
        )

        m2 = metrics_from_df(
            df_b,
            result["beta_b"],
            result["gamma_b"],
            population,
            hospitalization_rate,
        )

        comparison_table = pd.DataFrame(
            {
                "Metric": [
                    "Peak Infected",
                    "Day of Peak",
                    "R₀",
                    "Attack Rate",
                ],
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
        st.dataframe(
            comparison_table,
            use_container_width=True,
        )

        peak_difference = m2["Peak Infected"] - m1["Peak Infected"]

        if peak_difference > 0:
            st.warning(
                f"Scenario B has {peak_difference:,.0f} more peak infections than Scenario A."
            )
        elif peak_difference < 0:
            st.success(
                f"Scenario B has {abs(peak_difference):,.0f} fewer peak infections than Scenario A."
            )
        else:
            st.info("Both scenarios have the same peak infection level.")


# ---------------------------------------------------
# TAB 5: SENSITIVITY ANALYSIS
# ---------------------------------------------------
with tabs[4]:
    st.subheader("Sensitivity Analysis")
    st.caption("Test how changing one parameter affects infection curves.")

    parameter_choice = st.selectbox(
        "Parameter to vary",
        [
            "Beta",
            "Gamma",
            "Mu",
            "Vaccination Rate",
        ],
    )

    low_value, high_value = st.slider(
        "Parameter Range",
        0.0,
        1.0,
        (0.20, 0.80),
        0.01,
    )

    num_runs = st.slider(
        "Number of Scenarios",
        3,
        7,
        5,
        1,
    )

    if st.button(
        "Run Sensitivity Analysis",
        type="primary",
        disabled=bool(errors),
    ):
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

                m = metrics_from_df(
                    df_s,
                    beta_s,
                    gamma_s,
                    population,
                    hospitalization_rate,
                )

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
                    "Peak Infected": round(
                        item["metrics"]["Peak Infected"],
                        2,
                    ),
                    "Day of Peak": item["metrics"]["Day of Peak"],
                    "R0": round(
                        item["metrics"]["R0"],
                        2,
                    ),
                    "Attack Rate (%)": round(
                        item["metrics"]["Attack Rate"],
                        2,
                    ),
                }
                for item in result["data"]
            ]
        )

        st.subheader("Sensitivity Summary Table")
        st.dataframe(
            sens_table,
            use_container_width=True,
        )

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

        day = st.slider(
            "Select Day",
            0,
            len(df) - 1,
            0,
        )

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
                    x=[
                        "Susceptible",
                        "Infected",
                        "Recovered",
                        "Dead",
                        "Vaccinated",
                    ],
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

        st.plotly_chart(
            fig_bar,
            use_container_width=True,
        )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built by Omar Rulida Abdul-Rahman | MPH Candidate | Python + Streamlit")
