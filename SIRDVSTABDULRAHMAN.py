import io
import time
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------
# LOAD HELPER FUNCTION
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
                file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.run_sim

    raise FileNotFoundError("Simulation helper file not found.")


run_sim = load_run_sim()


# ---------------------------------------------------
# PAGE CONFIG
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


# ---------------------------------------------------
# HELPER FUNCTIONS
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
        model_choice=model_choice
    )

    return pd.DataFrame(
        {
            "Day": range(int(days)),
            "Susceptible": sim_s,
            "Infected": sim_i,
            "Recovered": sim_r,
            "Dead": sim_d,
            "Vaccinated": sim_v
        }
    )


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


def make_plot(df, model, y_axis_title="Population", title=None):
    colors = {
        "Susceptible": "#1f77b4",
        "Infected": "#d62728",
        "Recovered": "#2ca02c",
        "Dead": "#7f7f7f",
        "Vaccinated": "#9467bd"
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
                hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>"
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
        borderwidth=1
    )

    fig.update_layout(
        title=title or f"{model} Simulation Results",
        xaxis_title="Day",
        yaxis_title=y_axis_title,
        template="plotly_white",
        hovermode="x unified",
        height=620,
        margin=dict(l=30, r=30, t=70, b=40),
        legend_title="Compartments"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")

    return fig


def metrics_from_df(df, beta, gamma, population):
    peak_infected = float(df["Infected"].max())
    day_of_peak = int(df["Infected"].idxmax())
    final_recovered = float(df["Recovered"].iloc[-1])
    final_deaths = float(df["Dead"].iloc[-1])
    final_vaccinated = float(df["Vaccinated"].iloc[-1])
    remaining_susceptible = float(df["Susceptible"].iloc[-1])
    r0 = beta / gamma if gamma > 0 else 0
    total_impacted = population - remaining_susceptible
    attack_rate = ((final_recovered + final_deaths) / population) * 100

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
    }


def generate_insights(df, beta, gamma, mu, vac, population):
    metrics = metrics_from_df(df, beta, gamma, population)
    insights = []

    if metrics["R0"] > 3:
        insights.append("High transmission pressure: R₀ is well above 1, suggesting rapid outbreak growth.")
    elif metrics["R0"] > 1:
        insights.append("Moderate outbreak potential: R₀ is above 1, so infections may continue growing.")
    else:
        insights.append("Controlled transmission: R₀ is at or below 1, suggesting the outbreak may shrink over time.")

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
        insights.append("A large share of the population experiences infection or death by the end of the simulation.")
    elif metrics["Attack Rate"] <= 10:
        insights.append("Final infection burden remains relatively low.")

    return insights


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
Days: {params["days"]}

Notes:
Beta controls infection pressure.
Gamma controls recovery.
Mu controls mortality.
Vaccination rate moves susceptible people into the vaccinated compartment.
"""


def plot_download_buttons(fig, model_choice):
    html_buffer = io.StringIO()
    fig.write_html(html_buffer, include_plotlyjs="cdn")

    st.download_button(
        "Download Interactive Plot as HTML",
        data=html_buffer.getvalue(),
        file_name=f"{model_choice.lower()}_interactive_plot.html",
        mime="text/html"
    )

    try:
        png_bytes = fig.to_image(format="png", scale=3)

        st.download_button(
            "Download Plot as PNG",
            data=png_bytes,
            file_name=f"{model_choice.lower()}_simulation_plot.png",
            mime="image/png"
        )
    except Exception:
        st.info("PNG export needs the package `kaleido`. Add `kaleido` to requirements.txt to enable PNG downloads.")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Simulation Controls")

preset = st.sidebar.selectbox(
    "Preset Scenario",
    ["Custom", "COVID-like", "Seasonal Flu", "High Vaccination", "Severe Outbreak"]
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
    ["SIR", "SIRD", "SIRDV"]
)

population = st.sidebar.number_input(
    "Total Population",
    min_value=1,
    value=defaults["population"],
    help="Total number of people in the modeled population."
)

infected = st.sidebar.number_input(
    "Initial Infected",
    min_value=0,
    value=defaults["infected"],
    help="Number of people infected at the start of the simulation."
)

recovered = st.sidebar.number_input(
    "Initial Recovered",
    min_value=0,
    value=defaults["recovered"],
    help="Number of people already recovered at the start."
)

beta = st.sidebar.slider(
    "Beta (Infection Rate)",
    0.0,
    1.0,
    defaults["beta"],
    0.01,
    help="Higher beta means faster transmission from infected to susceptible people."
)

gamma = st.sidebar.slider(
    "Gamma (Recovery Rate)",
    0.0,
    1.0,
    defaults["gamma"],
    0.01,
    help="Higher gamma means infected people recover faster."
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
        help="Rate at which infected individuals move into the death compartment."
    )

if model_choice == "SIRDV":
    vac = st.sidebar.slider(
        "Vaccination Rate",
        0.0,
        1.0,
        defaults["vac"],
        0.01,
        help="Rate at which susceptible individuals move into the vaccinated compartment."
    )

days = st.sidebar.slider(
    "Days",
    10,
    365,
    defaults["days"],
    help="Number of days to simulate."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")

show_percent = st.sidebar.toggle(
    "Show chart as percentage",
    value=False
)

show_total_check = st.sidebar.toggle(
    "Show population conservation check",
    value=True
)

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
    disabled=bool(errors)
)


# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs(
    [
        "Simulation Results",
        "Data Table",
        "Model Explanation",
        "Compare Scenarios"
    ]
)


# ---------------------------------------------------
# RUN MAIN SIMULATION
# ---------------------------------------------------
if run_button:
    with st.spinner("Running simulation..."):
        time.sleep(0.4)

        df = run_model(
            model_choice,
            population,
            infected,
            recovered,
            beta,
            gamma,
            mu,
            vac,
            days
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
            "days": days,
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

        if show_percent:
            plot_df = convert_to_percent(
                df,
                get_columns(params["model_choice"]),
                params["population"]
            )
            y_axis_title = "Population (%)"

        fig = make_plot(
            plot_df,
            params["model_choice"],
            y_axis_title=y_axis_title,
            title=f"{params['model_choice']} Simulation Results"
        )

        metrics = metrics_from_df(
            df,
            params["beta"],
            params["gamma"],
            params["population"]
        )

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
        c8.metric("Susceptible Left", f"{metrics['Susceptible']:,.0f}")

        st.subheader("Automated Interpretation")

        for insight in generate_insights(
            df,
            params["beta"],
            params["gamma"],
            params["mu"],
            params["vac"],
            params["population"]
        ):
            st.info(insight)

        if show_total_check:
            st.subheader("Population Conservation Check")

            df_check = df.copy()
            df_check["Total"] = df_check[
                ["Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"]
            ].sum(axis=1)

            max_drift = abs(df_check["Total"] - params["population"]).max()

            col_check1, col_check2 = st.columns(2)

            col_check1.metric("Expected Population", f"{params['population']:,.0f}")
            col_check2.metric("Maximum Population Drift", f"{max_drift:,.4f}")

            fig_total = go.Figure()
            fig_total.add_trace(
                go.Scatter(
                    x=df_check["Day"],
                    y=df_check["Total"],
                    mode="lines",
                    name="Total Population",
                    line=dict(width=3)
                )
            )

            fig_total.update_layout(
                title="Total Population Over Time",
                xaxis_title="Day",
                yaxis_title="Total Population",
                template="plotly_white",
                height=350
            )

            st.plotly_chart(fig_total, use_container_width=True)

        st.subheader("Download Center")

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            csv = df.to_csv(index=False).encode()

            st.download_button(
                "Download CSV",
                csv,
                file_name=f"{params['model_choice'].lower()}_simulation_results.csv",
                mime="text/csv"
            )

        with dl2:
            parameter_summary = create_parameter_summary(params)

            st.download_button(
                "Download Parameters",
                parameter_summary,
                file_name=f"{params['model_choice'].lower()}_parameters.txt",
                mime="text/plain"
            )

        with dl3:
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

        summary_df = st.session_state.simulation_df.describe().round(2)
        st.dataframe(summary_df, use_container_width=True)


# ---------------------------------------------------
# TAB 3: MODEL EXPLANATION
# ---------------------------------------------------
with tabs[2]:
    st.subheader("Model Explanation")

    st.markdown("""
## SIR Model
**Susceptible → Infected → Recovered**

The SIR model is useful for diseases where individuals become immune after recovery.

## SIRD Model
**Susceptible → Infected → Recovered / Dead**

The SIRD model adds mortality and is useful when deaths are an important outcome.

## SIRDV Model
**Susceptible → Infected → Recovered / Dead / Vaccinated**

The SIRDV model adds vaccination, allowing users to explore how vaccination changes outbreak dynamics.

## Parameter Guide

| Parameter | Meaning | Interpretation |
|---|---|---|
| Beta | Infection rate | Higher beta means faster spread |
| Gamma | Recovery rate | Higher gamma means faster recovery |
| Mu | Death rate | Higher mu means greater mortality |
| Vaccination rate | Movement from susceptible to vaccinated | Higher vaccination reduces the susceptible pool |
| R₀ | Basic reproduction number | Values above 1 suggest outbreak growth |

## Real-World Uses

- COVID-19 outbreak analysis
- Seasonal flu forecasting
- Vaccine planning
- Hospital capacity planning
- Public health education
- Scenario-based policy discussion

## Important Note

This dashboard is intended for educational and exploratory use. Real-world epidemiological forecasting requires calibrated data, uncertainty modeling, and expert interpretation.
""")


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
            disabled=bool(errors)
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
                days
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
                days
            )

            st.session_state.compare_results = {
                "df_a": df_a,
                "df_b": df_b,
                "beta_a": beta_a,
                "gamma_a": gamma_a,
                "mu_a": mu_a,
                "vac_a": vac_a,
                "beta_b": beta_b,
                "gamma_b": gamma_b,
                "mu_b": mu_b,
                "vac_b": vac_b,
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
                    line=dict(width=3)
                )
            )

            fig_compare.add_trace(
                go.Scatter(
                    x=df_b["Day"],
                    y=df_b[col],
                    mode="lines",
                    name=f"Scenario B {col}",
                    line=dict(width=3, dash="dash")
                )
            )

        fig_compare.update_layout(
            title="Scenario Comparison",
            xaxis_title="Day",
            yaxis_title="Population",
            template="plotly_white",
            hovermode="x unified",
            height=650,
            legend_title="Scenario / Compartment"
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        m1 = metrics_from_df(df_a, result["beta_a"], result["gamma_a"], population)
        m2 = metrics_from_df(df_b, result["beta_b"], result["gamma_b"], population)

        st.subheader("Outcome Comparison")

        ca, cb = st.columns(2)

        with ca:
            st.markdown("### Scenario A")
            st.metric("Peak Infected", f"{m1['Peak Infected']:,.0f}")
            st.metric("Day of Peak", f"{m1['Day of Peak']}")
            st.metric("R₀", f"{m1['R0']:.2f}")
            st.metric("Attack Rate", f"{m1['Attack Rate']:.1f}%")

        with cb:
            st.markdown("### Scenario B")
            st.metric("Peak Infected", f"{m2['Peak Infected']:,.0f}")
            st.metric("Day of Peak", f"{m2['Day of Peak']}")
            st.metric("R₀", f"{m2['R0']:.2f}")
            st.metric("Attack Rate", f"{m2['Attack Rate']:.1f}%")

        peak_difference = m2["Peak Infected"] - m1["Peak Infected"]

        if peak_difference > 0:
            st.warning(f"Scenario B has {peak_difference:,.0f} more peak infections than Scenario A.")
        elif peak_difference < 0:
            st.success(f"Scenario B has {abs(peak_difference):,.0f} fewer peak infections than Scenario A.")
        else:
            st.info("Both scenarios have the same peak infection level.")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit | Epidemiological Decision Dashboard")
