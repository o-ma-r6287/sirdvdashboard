# Complete Streamlit SIR / SIRD / SIRDV Epidemiological Decision Dashboard
# Version 1.0
# Built by Omar Abdul-Rahman

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
    page_title="SIRDV Epidemiological Decision Dashboard",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 SIRDV Epidemiological Decision Dashboard")
st.caption("Version 1.0 | Interactive public health simulation for SIR / SIRD / SIRDV models")
st.caption("Built by Omar Abdul-Rahman | Python + Streamlit")

st.info(
    "Start by selecting a preset scenario or adjust parameters in the sidebar, "
    "then click **Run Simulation** to explore outcomes."
)

st.warning(
    "For educational scenario exploration only. This dashboard is not an official forecasting tool "
    "and should not be used for real-world public health decision-making without validated data and expert review."
)


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
    "auto_baseline_loaded": False,
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
        return run_model(model_choice, pop, infected, recovered, beta, gamma, mu, vac, days)

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
    day_of_peak = int(df.loc[df["Infected"].idxmax(), "Day"])
    final_recovered = float(df["Recovered"].iloc[-1])
    final_deaths = float(df["Dead"].iloc[-1])
    final_vaccinated = float(df["Vaccinated"].iloc[-1])
    remaining_susceptible = float(df["Susceptible"].iloc[-1])

    r0 = beta / gamma if gamma > 0 else 0
    attack_rate = ((final_recovered + final_deaths) / population) * 100
    estimated_peak_hospitalizations = peak_infected * hospitalization_rate
    cumulative_cases = final_recovered + final_deaths
    peak_percent = (peak_infected / population) * 100

    return {
        "Peak Infected": peak_infected,
        "Peak Infected %": peak_percent,
        "Day of Peak": day_of_peak,
        "Recovered": final_recovered,
        "Deaths": final_deaths,
        "Vaccinated": final_vaccinated,
        "Susceptible": remaining_susceptible,
        "R0": r0,
        "Attack Rate": attack_rate,
        "Estimated Peak Hospitalizations": estimated_peak_hospitalizations,
        "Cumulative Cases": cumulative_cases,
    }


def first_icu_crossing_day(df, hospitalization_rate, icu_capacity):
    estimated_hospitalizations = df["Infected"] * hospitalization_rate
    crossed = df.loc[estimated_hospitalizations > icu_capacity]

    if crossed.empty:
        return None

    return int(crossed["Day"].iloc[0])


def icu_status_text(df, hospitalization_rate, icu_capacity):
    breach_day = first_icu_crossing_day(df, hospitalization_rate, icu_capacity)
    if breach_day is None:
        return "Within Capacity", None
    return f"Exceeded Capacity (Day {breach_day})", breach_day


def risk_level(metrics, icu_capacity):
    icu_exceeded = metrics["Estimated Peak Hospitalizations"] > icu_capacity

    if icu_exceeded or metrics["R0"] >= 3.0 or metrics["Attack Rate"] >= 70:
        return "⚫ Critical", "Critical Overload", "error"

    if metrics["R0"] >= 2.0 or metrics["Attack Rate"] >= 40:
        return "🔴 High", "Uncontrolled Spread", "warning"

    if metrics["R0"] > 1.0 or metrics["Attack Rate"] >= 15:
        return "🟡 Moderate", "Elevated Transmission", "warning"

    return "🟢 Low", "Contained / Lower Risk", "success"


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


def policy_recommendation(metrics, icu_capacity):
    if metrics["Estimated Peak Hospitalizations"] > icu_capacity:
        return "Strong intervention is recommended: reduce transmission through earlier intervention, vaccination, distancing, or other control measures."

    if metrics["R0"] > 2:
        return "Strong intervention is recommended: reduce transmission through earlier intervention, vaccination, distancing, or other control measures."

    if metrics["Attack Rate"] > 50:
        return "High population impact is expected. Increase vaccination, reduce transmission, and monitor healthcare capacity closely."

    if metrics["R0"] <= 1:
        return "Current parameters suggest lower transmission pressure. Continue monitoring and maintain prevention measures."

    return "Moderate risk is present. Consider earlier intervention, vaccination, and continued ICU monitoring."


def public_health_decision_brief(metrics, risk_label, risk_description, grade, recommendation):
    if risk_label.startswith("🟢"):
        return (
            f"This simulation indicates a lower-risk outbreak scenario with peak infection around day "
            f"{metrics['Day of Peak']}, an estimated attack rate of {metrics['Attack Rate']:.1f}%, "
            f"an estimated R₀ of {metrics['R0']:.2f}, and a public health readiness grade of {grade}. "
            f"Continue monitoring, maintain prevention measures, and use scenario comparison to test whether conditions worsen."
        )

    return (
        f"This simulation indicates an uncontrolled outbreak scenario with rapid transmission, "
        f"a peak infection around day {metrics['Day of Peak']}, and an estimated attack rate of "
        f"{metrics['Attack Rate']:.1f}%. The estimated R₀ is {metrics['R0']:.2f}, and the public health "
        f"readiness grade is {grade}. {recommendation}"
    )


def bottom_line(metrics, icu_capacity):
    if metrics["Estimated Peak Hospitalizations"] > icu_capacity:
        return "Bottom Line: Without stronger intervention, the healthcare system may be overwhelmed early in the outbreak."

    if metrics["R0"] > 1:
        return "Bottom Line: Transmission remains above the replacement threshold, so earlier intervention may reduce future healthcare strain."

    return "Bottom Line: Current assumptions suggest the outbreak may remain manageable, but continued monitoring is still needed."


def policy_impact_score(current_metrics, baseline_metrics, icu_capacity):
    baseline_peak = baseline_metrics["Peak Infected"]
    current_peak = current_metrics["Peak Infected"]

    baseline_cases = baseline_metrics["Cumulative Cases"]
    current_cases = current_metrics["Cumulative Cases"]

    peak_reduction_pct = ((baseline_peak - current_peak) / baseline_peak) * 100 if baseline_peak > 0 else 0
    case_reduction_pct = ((baseline_cases - current_cases) / baseline_cases) * 100 if baseline_cases > 0 else 0

    icu_bonus = 20 if current_metrics["Estimated Peak Hospitalizations"] <= icu_capacity else 0

    score = 40 + (peak_reduction_pct * 0.3) + (case_reduction_pct * 0.3) + icu_bonus
    return round(max(min(score, 100), 0), 1)


def generate_insights(df, beta, gamma, mu, vac, population, hospitalization_rate):
    metrics = metrics_from_df(df, beta, gamma, population, hospitalization_rate)
    insights = []

    if metrics["R0"] > 3:
        insights.append("High transmission pressure: R₀ is well above 1, suggesting rapid outbreak growth.")
    elif metrics["R0"] > 1:
        insights.append("Moderate outbreak potential: R₀ is above 1, so infections may continue growing.")
    else:
        insights.append("Lower transmission pressure: R₀ is at or below 1, suggesting infections may decline over time.")

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
        insights.append("Mortality pressure is elevated, making this a higher-severity scenario.")

    if metrics["Attack Rate"] >= 50:
        insights.append("A large share of the population experiences infection or death by the end of the simulation.")
    elif metrics["Attack Rate"] <= 10:
        insights.append("Final infection burden remains relatively low.")

    if hospitalization_rate >= 0.10:
        insights.append("Hospital burden may become significant because the hospitalization assumption is high.")

    return insights


def build_comparison_metrics(df, beta, gamma, population, hospitalization_rate, icu_capacity):
    metrics = metrics_from_df(df, beta, gamma, population, hospitalization_rate)
    icu_cross_day = first_icu_crossing_day(df, hospitalization_rate, icu_capacity)
    icu_overflow = icu_cross_day is not None

    return {
        "Peak infected": metrics["Peak Infected"],
        "Peak day": metrics["Day of Peak"],
        "Total infected / attack rate": metrics["Attack Rate"],
        "Final deaths": metrics["Deaths"],
        "ICU overflow: Yes/No": "Yes" if icu_overflow else "No",
        "ICU breach day": icu_cross_day if icu_cross_day is not None else "N/A",
    }


def calculate_policy_effectiveness_score(metrics_a, metrics_b):
    score = 50

    if metrics_b["Peak infected"] < metrics_a["Peak infected"]:
        score += 15

    if metrics_b["Total infected / attack rate"] < metrics_a["Total infected / attack rate"]:
        score += 15

    if metrics_a["ICU overflow: Yes/No"] == "Yes" and metrics_b["ICU overflow: Yes/No"] == "No":
        score += 20
    elif metrics_b["ICU overflow: Yes/No"] == "Yes":
        score -= 10

    if metrics_b["Final deaths"] < metrics_a["Final deaths"]:
        score += 15

    return max(0, min(100, score))


def generate_comparison_insight(metrics_a, metrics_b):
    peak_difference = metrics_a["Peak infected"] - metrics_b["Peak infected"]
    peak_percent_change = (peak_difference / metrics_a["Peak infected"]) * 100 if metrics_a["Peak infected"] > 0 else 0
    peak_day_change = metrics_b["Peak day"] - metrics_a["Peak day"]

    if peak_difference > 0 and metrics_a["ICU overflow: Yes/No"] == "Yes" and metrics_b["ICU overflow: Yes/No"] == "No":
        return f"Scenario B reduced peak infections by {peak_percent_change:.1f}% and prevented ICU overflow."

    if peak_difference > 0 and metrics_b["ICU overflow: Yes/No"] == "Yes":
        return f"Scenario B reduced peak infections by {peak_percent_change:.1f}%, but did not prevent ICU overflow."

    if peak_day_change > 0 and metrics_b["ICU overflow: Yes/No"] == "Yes":
        return f"Scenario B delayed the peak by {peak_day_change} days but did not prevent ICU overflow."

    if peak_day_change > 0:
        return f"Scenario B delayed the peak by {peak_day_change} days."

    if peak_difference < 0:
        return "Scenario B produced a higher infection peak than Scenario A. This may indicate a weaker or delayed intervention strategy."

    return "Both scenarios produced similar peak infection outcomes."


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
        "Susceptible": "#2563eb",
        "Infected": "#dc2626",
        "Recovered": "#16a34a",
        "Dead": "#6b7280",
        "Vaccinated": "#7c3aed",
        "Cumulative Cases": "#f59e0b",
    }

    fig = go.Figure()

    for col in get_columns(model):
        fig.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=4 if col == "Infected" else 3, color=colors[col]),
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
    peak_day = int(df.loc[peak_idx, "Day"])
    peak_val = float(df["Infected"].max())

    fig.add_trace(
        go.Scatter(
            x=[peak_day],
            y=[peak_val],
            mode="markers",
            name="Peak Infection",
            marker=dict(size=12, color="#dc2626", line=dict(width=2, color="white")),
            hovertemplate="Peak Infection: %{y:.0f}<extra></extra>",
        )
    )

    fig.add_annotation(
        x=peak_day,
        y=peak_val,
        text="<b>Peak</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-45,
        bgcolor="#ffffff" if template == "plotly_white" else "#111827",
        font=dict(color="#111827" if template == "plotly_white" else "#ffffff"),
        bordercolor="#dc2626",
        borderwidth=2,
        borderpad=4,
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


def make_overlay_comparison_plot(
    df_a,
    df_b,
    template,
    hospitalization_rate,
    icu_capacity,
    intervention_day_a=None,
    intervention_day_b=None,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_a["Day"],
            y=df_a["Infected"],
            mode="lines",
            name="Scenario A Infected",
            line=dict(width=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_b["Day"],
            y=df_b["Infected"],
            mode="lines",
            name="Scenario B Infected",
            line=dict(width=4, dash="dash"),
        )
    )

    peak_idx_a = int(df_a["Infected"].idxmax())
    peak_day_a = int(df_a.loc[peak_idx_a, "Day"])
    peak_value_a = float(df_a["Infected"].max())

    peak_idx_b = int(df_b["Infected"].idxmax())
    peak_day_b = int(df_b.loc[peak_idx_b, "Day"])
    peak_value_b = float(df_b["Infected"].max())

    fig.add_trace(
        go.Scatter(
            x=[peak_day_a],
            y=[peak_value_a],
            mode="markers",
            name="Scenario A Peak",
            marker=dict(size=12),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[peak_day_b],
            y=[peak_value_b],
            mode="markers",
            name="Scenario B Peak",
            marker=dict(size=12),
        )
    )

    if hospitalization_rate > 0:
        fig.add_hline(
            y=icu_capacity / hospitalization_rate,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="ICU Threshold",
            annotation_position="top left",
        )

    if intervention_day_a is not None:
        fig.add_vline(
            x=intervention_day_a,
            line_dash="dot",
            line_color="#22c55e",
            annotation_text="Scenario A Intervention",
            annotation_position="top left",
        )

    if intervention_day_b is not None:
        fig.add_vline(
            x=intervention_day_b,
            line_dash="dash",
            line_color="#16a34a",
            annotation_text="Scenario B Intervention",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Overlay Comparison Plot: Scenario A vs Scenario B",
        xaxis_title="Day",
        yaxis_title="Infected Population",
        template=template,
        hovermode="x unified",
        height=650,
    )

    return fig


# ---------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------
def create_parameter_summary(params):
    return f"""SIRDV Epidemiological Decision Dashboard Parameter Summary

Version: 1.0
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

Disclaimer:
For educational scenario exploration only. This dashboard is not an official forecasting tool and should not be used for real-world public health decision-making without validated data and expert review.
"""


def create_full_report(params, metrics, insights, recommendation, grade, brief, risk_label, icu_status, impact_score=None):
    return f"""SIRDV Epidemiological Decision Dashboard Report
Version 1.0

Public Health Decision Brief:
{brief}

Risk Level:
{risk_label}

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
Peak Infected %: {metrics["Peak Infected %"]:.2f}%
Day of Peak: {metrics["Day of Peak"]}
R0: {metrics["R0"]:.2f}
Attack Rate: {metrics["Attack Rate"]:.2f}%
Recovered: {metrics["Recovered"]:,.0f}
Deaths: {metrics["Deaths"]:,.0f}
Vaccinated: {metrics["Vaccinated"]:,.0f}
Susceptible Remaining: {metrics["Susceptible"]:,.0f}
Estimated Peak Hospitalizations: {metrics["Estimated Peak Hospitalizations"]:,.0f}
ICU Status: {icu_status}
Public Health Readiness Grade: {grade}
Policy Impact Score: {impact_score if impact_score is not None else "N/A"}

Automated Interpretation:
{chr(10).join("- " + insight for insight in insights)}

Policy Recommendation:
{recommendation}

Disclaimer:
For educational scenario exploration only. This dashboard is not an official forecasting tool and should not be used for real-world public health decision-making without validated data and expert review.
"""


def plot_download_buttons(fig, file_prefix):
    html_buffer = io.StringIO()
    fig.write_html(html_buffer, include_plotlyjs="cdn")

    st.download_button(
        "Download Interactive Plot as HTML",
        data=html_buffer.getvalue(),
        file_name=f"{file_prefix}_interactive_plot.html",
        mime="text/html",
    )

    try:
        png_bytes = fig.to_image(format="png", scale=3)
        st.download_button(
            "Download Plot as PNG",
            data=png_bytes,
            file_name=f"{file_prefix}_plot.png",
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

mode = st.sidebar.radio("Interface Mode", ["Academic Mode", "Real-World Mode"])
real_world_mode = mode == "Real-World Mode"

guided_start = st.sidebar.checkbox("Start with a realistic baseline scenario", value=True)

preset = st.sidebar.selectbox(
    "Preset Scenario",
    ["Custom", "COVID-like", "Seasonal Flu", "High Vaccination", "Severe Outbreak"],
    index=1 if guided_start else 0,
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

model_choice = st.sidebar.selectbox("Model Selection", ["SIR", "SIRD", "SIRDV"], index=2)

population = st.sidebar.number_input("Population", min_value=1, value=defaults["population"])
infected = st.sidebar.number_input("Initial Infected", min_value=0, value=defaults["infected"])
recovered = st.sidebar.number_input("Initial Recovered", min_value=0, value=defaults["recovered"])

beta = st.sidebar.slider(
    label_text("Beta (Infection Rate)", real_world_mode),
    0.0,
    1.0,
    defaults["beta"],
    0.01,
)

gamma = st.sidebar.slider(
    label_text("Gamma (Recovery Rate)", real_world_mode),
    0.0,
    1.0,
    defaults["gamma"],
    0.01,
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
    )

if model_choice == "SIRDV":
    vac = st.sidebar.slider(
        label_text("Vaccination Rate", real_world_mode),
        0.0,
        1.0,
        defaults["vac"],
        0.01,
    )

days = st.sidebar.slider("Days", 10, 365, defaults["days"])

hospitalization_rate = st.sidebar.slider("Hospitalization Rate", 0.0, 1.0, 0.05, 0.01)
icu_capacity = st.sidebar.number_input("ICU Capacity", min_value=0, value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Intervention Settings")

intervention_enabled = st.sidebar.toggle("Enable Intervention", value=False)

intervention_day = st.sidebar.slider("Intervention Start Day", 1, days - 1, min(30, days - 1))

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

run_button = st.sidebar.button("Run Simulation", type="primary", disabled=bool(errors))


# ---------------------------------------------------
# AUTO BASELINE ON FIRST LOAD
# ---------------------------------------------------
if (
    guided_start
    and not st.session_state.auto_baseline_loaded
    and st.session_state.simulation_df is None
    and not errors
):
    baseline_df = run_model_with_intervention(
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

    st.session_state.simulation_df = baseline_df
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
    st.session_state.auto_baseline_loaded = True


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

st.markdown(
    """
### How to Interpret This Dashboard

This tool is designed to compare **relative outcomes between scenarios**, not to predict exact real-world case counts. 
It is best used to understand how changes in transmission, intervention timing, and vaccination affect trends like peak infections and healthcare strain.
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
        time.sleep(0.3)

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
        st.info(
            "Select a preset scenario or customize parameters in the sidebar, then click "
            "**Run Simulation** to explore how interventions affect outbreak outcomes."
        )
    else:
        df = st.session_state.simulation_df
        params = st.session_state.simulation_params

        st.markdown("")

        plot_df = df.copy()
        y_axis_title = "Population"
        icu_line = params["icu_capacity"]

        if show_percent:
            plot_df = convert_to_percent(df, get_columns(params["model_choice"]), params["population"])
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

        risk_label, risk_description, risk_style = risk_level(metrics, params["icu_capacity"])
        grade = readiness_grade(metrics, params["icu_capacity"], params["vac"])
        recommendation = policy_recommendation(metrics, params["icu_capacity"])
        brief = public_health_decision_brief(metrics, risk_label, risk_description, grade, recommendation)
        icu_status, breach_day = icu_status_text(df, params["hospitalization_rate"], params["icu_capacity"])

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

        st.subheader("Public Health Decision Brief")

        # Row 1
        k1 = st.columns(1)[0]
        k1.metric("Risk Level", risk_label)

        # Row 2 (3 metrics in one row)
        k2, k3, k4 = st.columns(3)
        k2.metric("R₀", f"{metrics['R0']:.2f}")
        k3.metric("Peak Day", metrics["Day of Peak"])
        k4.metric("ICU Status", icu_status)

        # Row 3
        k5 = st.columns(1)[0]
        k5.metric("Readiness Grade", grade)

        if risk_style == "error":
            st.error(brief)
        elif risk_style == "warning":
            st.warning(brief)
        else:
            st.success(brief)

        st.markdown(f"**{bottom_line(metrics, params['icu_capacity'])}**")

        st.markdown("---")

        st.subheader("Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak % Infected", f"{metrics['Peak Infected %']:.1f}%")
        c2.metric("Total Infected / Attack Rate", f"{metrics['Attack Rate']:.1f}%")
        c3.metric("Final Deaths", f"{metrics['Deaths']:,.0f}")
        c4.metric("ICU Breach Day", breach_day if breach_day is not None else "N/A")

        st.markdown("")

        st.subheader("Simulation Dashboard")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("")

        with st.container(border=True):
            st.markdown("### Structured Insight Box")

            s1, s2, s3 = st.columns(3)
            s1.markdown(f"**Peak Infection Day:** {metrics['Day of Peak']}")
            s1.markdown(f"**Peak Infected %:** {metrics['Peak Infected %']:.1f}%")
            s2.markdown(f"**Estimated Attack Rate %:** {metrics['Attack Rate']:.1f}%")
            s2.markdown(f"**R₀:** {metrics['R0']:.2f}")
            s3.markdown(f"**ICU Status:** {icu_status}")
            s3.markdown(f"**Readiness Grade:** {grade}")

            st.markdown("#### Recommendation")
            st.markdown(
                """
- Reduce transmission through earlier intervention
- Increase vaccination rate
- Monitor ICU capacity
- Use scenario comparison to evaluate policy timing
"""
            )

        st.markdown("")

        st.subheader("Healthcare Capacity Timeline")
        fig_health = make_healthcare_timeline(df, params["hospitalization_rate"], params["icu_capacity"], template)
        st.plotly_chart(fig_health, use_container_width=True)

        if params["intervention_enabled"]:
            infections_prevented = baseline_metrics["Cumulative Cases"] - metrics["Cumulative Cases"]
            peak_reduction = baseline_metrics["Peak Infected"] - metrics["Peak Infected"]

            st.subheader("Intervention Impact Score")
            i1, i2, i3 = st.columns(3)
            i1.metric("Estimated Infections Prevented", f"{infections_prevented:,.0f}")
            i2.metric("Peak Reduction", f"{peak_reduction:,.0f}")
            i3.metric("Policy Effectiveness", f"{impact_score:.1f} / 100")

        st.markdown("---")

        st.subheader("Policy Recommendation")
        if "Strong" in recommendation:
            st.warning(recommendation)
        else:
            st.success(recommendation)

        st.subheader("Automated Interpretation")
        for insight in insights:
            st.info(insight)

        with st.expander("Method Notes"):
            st.markdown(
                """
- **Peak infected** = highest infected count during simulation.
- **Peak day** = day when infected count is highest.
- **Attack rate** = final recovered + final deaths divided by total population.
- **ICU strain** = infected count × hospitalization rate compared with ICU capacity.
- **R₀** = beta / gamma.
"""
            )

        with st.expander("Parameter Guide"):
            st.markdown(
                """
- **Beta** = transmission rate.
- **Gamma** = recovery rate.
- **Mu** = mortality rate.
- **Vaccination rate** = rate susceptible people move into vaccinated group.
- **Intervention start day** = when reduced transmission or increased vaccination begins.
- **ICU capacity** = available critical care threshold.
"""
            )

        st.subheader("Example Scenario Walkthrough")
        st.info(
            "Try comparing Late Response vs Early Intervention. Notice how earlier intervention can reduce peak infections, delay the peak, and lower ICU strain."
        )

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

        st.markdown("---")

        st.subheader("Download Center")
        dl1, dl2, dl3, dl4 = st.columns(4)

        with dl1:
            st.download_button(
                "Download Simulation CSV",
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
                "Download Text Report",
                create_full_report(
                    params,
                    metrics,
                    insights,
                    recommendation,
                    grade,
                    brief,
                    f"{risk_label} ({risk_description})",
                    icu_status,
                    impact_score,
                ),
                file_name=f"{params['model_choice'].lower()}_simulation_report.txt",
                mime="text/plain",
            )

        with dl4:
            plot_download_buttons(fig, f"{params['model_choice'].lower()}_simulation")


# ---------------------------------------------------
# TAB 2: DATA TABLE
# ---------------------------------------------------
with tabs[1]:
    if st.session_state.simulation_df is None:
        st.info(
            "Select a preset scenario or customize parameters in the sidebar, then click "
            "**Run Simulation** to explore how interventions affect outbreak outcomes."
        )
    else:
        st.subheader("Simulation Data")
        st.dataframe(st.session_state.simulation_df, use_container_width=True)

        st.download_button(
            "Download Data Table CSV",
            st.session_state.simulation_df.to_csv(index=False).encode(),
            file_name="simulation_data_table.csv",
            mime="text/csv",
        )

        st.subheader("Quick Summary")
        st.dataframe(st.session_state.simulation_df.describe().round(2), use_container_width=True)


# ---------------------------------------------------
# TAB 3: MODEL EXPLANATION
# ---------------------------------------------------
with tabs[2]:
    st.subheader("Model Explanation")

    st.markdown(
        """
## SIRDV Model Overview

The model divides a population into five compartments:

- **S = Susceptible:** People who can still become infected.
- **I = Infected:** People currently infected and able to transmit disease.
- **R = Recovered:** People who recovered and are no longer infectious.
- **D = Deceased:** People who died from the disease.
- **V = Vaccinated:** People moved out of the susceptible group through vaccination.

## Parameter Interpretation

- **Beta / Transmission Rate:** Controls how quickly susceptible people become infected.
- **Gamma / Recovery Rate:** Controls how quickly infected people recover.
- **Mu / Mortality Rate:** Controls how quickly infected people move into the deceased compartment.
- **Vaccination Rate:** Controls how quickly susceptible people become vaccinated.

## How to Interpret This Dashboard

This tool is designed to compare **relative outcomes between scenarios**, not to predict exact real-world case counts. 
It is best used to understand how changes in transmission, intervention timing, and vaccination affect trends like peak infections and healthcare strain.
"""
    )


# ---------------------------------------------------
# TAB 4: COMPARE SCENARIOS
# ---------------------------------------------------
with tabs[3]:
    st.subheader("Compare Scenarios")
    st.caption("Use one-click presets or manually adjust Scenario A and Scenario B.")

    demo_choice = st.selectbox(
        "Comparison Quick Start",
        ["Custom Comparison", "Early vs Late Intervention (Demo)"],
    )

    scenario_presets = {
        "Baseline": {
            "beta_multiplier": 1.00,
            "gamma_multiplier": 1.00,
            "mu_multiplier": 1.00,
            "vac_boost": 0.00,
            "intervention_enabled": False,
            "intervention_day": intervention_day,
            "post_beta_multiplier": 1.00,
            "post_vac_boost": 0.00,
        },
        "Early Intervention": {
            "beta_multiplier": 1.00,
            "gamma_multiplier": 1.00,
            "mu_multiplier": 1.00,
            "vac_boost": 0.02,
            "intervention_enabled": True,
            "intervention_day": min(15, days - 1),
            "post_beta_multiplier": 0.45,
            "post_vac_boost": 0.08,
        },
        "Late Response": {
            "beta_multiplier": 1.05,
            "gamma_multiplier": 1.00,
            "mu_multiplier": 1.00,
            "vac_boost": 0.00,
            "intervention_enabled": True,
            "intervention_day": min(60, days - 1),
            "post_beta_multiplier": 0.80,
            "post_vac_boost": 0.02,
        },
        "No Vaccination": {
            "beta_multiplier": 1.00,
            "gamma_multiplier": 1.00,
            "mu_multiplier": 1.00,
            "vac_override": 0.00,
            "intervention_enabled": False,
            "intervention_day": intervention_day,
            "post_beta_multiplier": 1.00,
            "post_vac_boost": 0.00,
        },
        "Aggressive Vaccination": {
            "beta_multiplier": 0.85,
            "gamma_multiplier": 1.00,
            "mu_multiplier": 1.00,
            "vac_boost": 0.10,
            "intervention_enabled": True,
            "intervention_day": min(10, days - 1),
            "post_beta_multiplier": 0.50,
            "post_vac_boost": 0.12,
        },
    }

    def apply_comparison_preset(base_beta, base_gamma, base_mu, base_vac, preset_name):
        config = scenario_presets[preset_name]

        new_beta = base_beta * config.get("beta_multiplier", 1.0)
        new_gamma = base_gamma * config.get("gamma_multiplier", 1.0)
        new_mu = base_mu * config.get("mu_multiplier", 1.0)
        new_vac = base_vac + config.get("vac_boost", 0.0)

        if "vac_override" in config:
            new_vac = config["vac_override"]

        return {
            "beta": min(max(new_beta, 0.0), 1.0),
            "gamma": min(max(new_gamma, 0.0), 1.0),
            "mu": min(max(new_mu, 0.0), 1.0),
            "vac": min(max(new_vac, 0.0), 1.0),
            "intervention_enabled": config.get("intervention_enabled", False),
            "intervention_day": config.get("intervention_day", intervention_day),
            "reduced_beta": min(max(base_beta * config.get("post_beta_multiplier", 1.0), 0.0), 1.0),
            "increased_vac": min(max(base_vac + config.get("post_vac_boost", 0.0), 0.0), 1.0),
        }

    default_preset_a = "Early Intervention" if demo_choice == "Early vs Late Intervention (Demo)" else "Baseline"
    default_preset_b = "Late Response" if demo_choice == "Early vs Late Intervention (Demo)" else "Early Intervention"

    col_preset_a, col_preset_b = st.columns(2)
    with col_preset_a:
        preset_a = st.selectbox(
            "Scenario A Preset",
            list(scenario_presets.keys()),
            index=list(scenario_presets.keys()).index(default_preset_a),
        )
    with col_preset_b:
        preset_b = st.selectbox(
            "Scenario B Preset",
            list(scenario_presets.keys()),
            index=list(scenario_presets.keys()).index(default_preset_b),
        )

    defaults_a = apply_comparison_preset(beta, gamma, mu, vac, preset_a)
    defaults_b = apply_comparison_preset(beta, gamma, mu, vac, preset_b)

    with st.form("comparison_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Scenario A")
            beta_a = st.slider("Transmission Rate A", 0.0, 1.0, defaults_a["beta"], 0.01)
            gamma_a = st.slider("Recovery Rate A", 0.0, 1.0, defaults_a["gamma"], 0.01)

            mu_a = defaults_a["mu"]
            vac_a = defaults_a["vac"]

            if model_choice in ["SIRD", "SIRDV"]:
                mu_a = st.slider("Mortality Rate A", 0.0, 1.0, defaults_a["mu"], 0.01)

            if model_choice == "SIRDV":
                vac_a = st.slider("Vaccination Rate A", 0.0, 1.0, defaults_a["vac"], 0.01)

            intervention_a = st.checkbox("Enable Intervention A", value=defaults_a["intervention_enabled"])
            intervention_day_a = st.slider("Intervention Day A", 1, days - 1, defaults_a["intervention_day"])
            reduced_beta_a = st.slider("Post-Intervention Transmission A", 0.0, 1.0, defaults_a["reduced_beta"], 0.01)
            increased_vac_a = st.slider("Post-Intervention Vaccination A", 0.0, 1.0, defaults_a["increased_vac"], 0.01)

        with col_b:
            st.markdown("### Scenario B")
            beta_b = st.slider("Transmission Rate B", 0.0, 1.0, defaults_b["beta"], 0.01)
            gamma_b = st.slider("Recovery Rate B", 0.0, 1.0, defaults_b["gamma"], 0.01)

            mu_b = defaults_b["mu"]
            vac_b = defaults_b["vac"]

            if model_choice in ["SIRD", "SIRDV"]:
                mu_b = st.slider("Mortality Rate B", 0.0, 1.0, defaults_b["mu"], 0.01)

            if model_choice == "SIRDV":
                vac_b = st.slider("Vaccination Rate B", 0.0, 1.0, defaults_b["vac"], 0.01)

            intervention_b = st.checkbox("Enable Intervention B", value=defaults_b["intervention_enabled"])
            intervention_day_b = st.slider("Intervention Day B", 1, days - 1, defaults_b["intervention_day"])
            reduced_beta_b = st.slider("Post-Intervention Transmission B", 0.0, 1.0, defaults_b["reduced_beta"], 0.01)
            increased_vac_b = st.slider("Post-Intervention Vaccination B", 0.0, 1.0, defaults_b["increased_vac"], 0.01)

        compare_clicked = st.form_submit_button("Run Comparison", type="primary", disabled=bool(errors))

    if compare_clicked:
        with st.spinner("Comparing scenarios..."):
            df_a = run_model_with_intervention(
                model_choice,
                population,
                infected,
                recovered,
                beta_a,
                gamma_a,
                mu_a,
                vac_a,
                days,
                intervention_a,
                intervention_day_a,
                reduced_beta_a,
                increased_vac_a,
            )

            df_b = run_model_with_intervention(
                model_choice,
                population,
                infected,
                recovered,
                beta_b,
                gamma_b,
                mu_b,
                vac_b,
                days,
                intervention_b,
                intervention_day_b,
                reduced_beta_b,
                increased_vac_b,
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
                "intervention_a": intervention_a,
                "intervention_b": intervention_b,
                "intervention_day_a": intervention_day_a,
                "intervention_day_b": intervention_day_b,
            }

    if st.session_state.compare_results is not None:
        result = st.session_state.compare_results
        df_a = result["df_a"]
        df_b = result["df_b"]

        metrics_a = build_comparison_metrics(
            df_a,
            result["beta_a"],
            result["gamma_a"],
            population,
            hospitalization_rate,
            icu_capacity,
        )

        metrics_b = build_comparison_metrics(
            df_b,
            result["beta_b"],
            result["gamma_b"],
            population,
            hospitalization_rate,
            icu_capacity,
        )

        comparison_rows = []

        for metric_name in metrics_a:
            value_a = metrics_a[metric_name]
            value_b = metrics_b[metric_name]

            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                change = value_b - value_a
            else:
                change = "Changed" if value_a != value_b else "No Change"

            comparison_rows.append(
                {
                    "Metric": metric_name,
                    "Scenario A": value_a,
                    "Scenario B": value_b,
                    "Change": change,
                }
            )

        comparison_table = pd.DataFrame(comparison_rows)

        st.markdown("---")

        st.subheader("Comparison Results Panel")
        st.dataframe(comparison_table, use_container_width=True)

        st.subheader("Primary Difference Summary")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Transmission Rate Difference", f"{result['beta_b'] - result['beta_a']:.3f}")
        d2.metric("Intervention Timing Difference", f"{result['intervention_day_b'] - result['intervention_day_a']} days")
        d3.metric("Vaccination Rate Difference", f"{result['vac_b'] - result['vac_a']:.3f}")
        d4.metric(
            "Mortality / Recovery Difference",
            f"μ {result['mu_b'] - result['mu_a']:.3f} | γ {result['gamma_b'] - result['gamma_a']:.3f}",
        )

        score = calculate_policy_effectiveness_score(metrics_a, metrics_b)
        st.metric("Policy Effectiveness Score", f"{score} / 100")

        st.subheader("Auto Insight Generator")
        st.info(generate_comparison_insight(metrics_a, metrics_b))

        fig_compare = make_overlay_comparison_plot(
            df_a,
            df_b,
            template,
            hospitalization_rate,
            icu_capacity,
            intervention_day_a=result["intervention_day_a"] if result["intervention_a"] else None,
            intervention_day_b=result["intervention_day_b"] if result["intervention_b"] else None,
        )

        st.subheader("Overlay Comparison Plot")
        st.plotly_chart(fig_compare, use_container_width=True)

        csv_bytes = comparison_table.to_csv(index=False).encode()

        st.download_button(
            "Download Comparison Results CSV",
            csv_bytes,
            file_name="scenario_comparison_results.csv",
            mime="text/csv",
        )

        plot_download_buttons(fig_compare, "scenario_comparison")


# ---------------------------------------------------
# TAB 5: SENSITIVITY ANALYSIS
# ---------------------------------------------------
with tabs[4]:
    st.subheader("Sensitivity Analysis")
    st.caption("Test how changing one parameter affects infection curves.")

    parameter_choice = st.selectbox("Parameter to vary", ["Beta", "Gamma", "Mu", "Vaccination Rate"])

    low_value, high_value = st.slider("Parameter Range", 0.0, 1.0, (0.20, 0.80), 0.01)
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

                df_s = run_model(model_choice, population, infected, recovered, beta_s, gamma_s, mu_s, vac_s, days)
                m = metrics_from_df(df_s, beta_s, gamma_s, population, hospitalization_rate)

                sensitivity_data.append({"value": value, "df": df_s, "metrics": m})

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
                    "Peak Day": item["metrics"]["Day of Peak"],
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

        st.download_button(
            "Download Sensitivity Results CSV",
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
        st.info(
            "Select a preset scenario or customize parameters in the sidebar, then click "
            "**Run Simulation** to explore how interventions affect outbreak outcomes."
        )
    else:
        df = st.session_state.simulation_df

        day = st.slider("Select Day", 0, len(df) - 1, 0)
        row = df.iloc[day]

        st.markdown(f"### Day {int(row['Day'])}")

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
            title=f"Compartment Counts on Day {int(row['Day'])}",
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
                    "Peak Day": item["metrics"]["Day of Peak"],
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

        st.download_button(
            "Download Timing Results CSV",
            timing_table.to_csv(index=False).encode(),
            file_name="intervention_timing_results.csv",
            mime="text/csv",
        )


# ---------------------------------------------------
# TAB 8: RISK HEAT MAP
# ---------------------------------------------------
with tabs[7]:
    st.subheader("Risk Heat Map")
    st.caption("Simulated local risk intensity based on the selected location and current model assumptions.")

    st.info("Higher intensity indicates higher simulated infection pressure/risk.")

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
            base_df = run_model(model_choice, population, infected, recovered, beta, gamma, mu, vac, days)
            base_metrics = metrics_from_df(base_df, beta, gamma, population, hospitalization_rate)

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

                heat_points.append(
                    [
                        center_lat + lat_offset,
                        center_lon + lon_offset,
                        intensity,
                    ]
                )

            st.session_state.heatmap_results = {
                "location": selected_location,
                "center": (center_lat, center_lon),
                "points": heat_points,
                "metric": heat_metric,
                "risk_base": risk_base,
            }

    if st.session_state.heatmap_results is not None:
        result = st.session_state.heatmap_results

        st.metric(f"Base Simulated Risk: {result['metric']}", f"{result['risk_base']:,.2f}")

        if FOLIUM_AVAILABLE:
            center_lat, center_lon = result["center"]

            fmap = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles="CartoDB positron",
            )

            HeatMap(result["points"], radius=22, blur=20, max_zoom=13).add_to(fmap)

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
                mapbox_center={"lat": result["center"][0], "lon": result["center"][1]},
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
            "Download Heat Map Risk Points CSV",
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
## How to Interpret This Dashboard

This tool is designed to compare **relative outcomes between scenarios**, not to predict exact real-world case counts. 
It is best used to understand how changes in transmission, intervention timing, and vaccination affect trends like peak infections and healthcare strain.
"""
    )

    with st.expander("Model Assumptions"):
        st.markdown(
            """
- **Closed population**
- **Homogeneous mixing**
- **Deterministic model**
- **Simplified vaccination and intervention effects**
"""
        )

    with st.expander("Limitations"):
        st.markdown(
            """
- No age structure or demographic variation.
- No true geographic spread modeling.
- No stochastic uncertainty or random outbreak variation.
- Simplified vaccination, mortality, and hospitalization assumptions.
- Educational scenario exploration only, not official forecasting.
"""
        )

    with st.expander("Method Notes"):
        st.markdown(
            """
- **Peak infected** = highest infected count during simulation.
- **Peak day** = day when infected count is highest.
- **Attack rate** = final recovered + final deaths divided by total population.
- **ICU strain** = infected count × hospitalization rate compared with ICU capacity.
- **R₀** = beta / gamma.
"""
        )

    with st.expander("Parameter Guide"):
        st.markdown(
            """
- **Beta** = transmission rate.
- **Gamma** = recovery rate.
- **Mu** = mortality rate.
- **Vaccination rate** = rate susceptible people move into vaccinated group.
- **Intervention start day** = when reduced transmission or increased vaccination begins.
- **ICU capacity** = available critical care threshold.
"""
        )

    with st.expander("SIRDV Model Overview"):
        st.markdown(
            """
- **S = Susceptible:** Individuals who can still become infected.
- **I = Infected:** Individuals currently infected.
- **R = Recovered:** Individuals who have recovered.
- **D = Deceased:** Individuals who have died.
- **V = Vaccinated:** Individuals protected through vaccination.

### Parameter Effects

- **Beta / Transmission Rate:** Higher values increase new infections.
- **Gamma / Recovery Rate:** Higher values move infected individuals into recovery faster.
- **Mu / Mortality Rate:** Higher values increase deaths.
- **Vaccination Rate:** Higher values reduce the susceptible population faster.
"""
        )

    st.markdown(
        """
## Educational References

- Kermack, W. O., & McKendrick, A. G. (1927). *A Contribution to the Mathematical Theory of Epidemics.*
- Hethcote, H. W. (2000). *The Mathematics of Infectious Diseases.*
- Anderson, R. M., & May, R. M. (1991). *Infectious Diseases of Humans: Dynamics and Control.*
- Centers for Disease Control and Prevention. *Principles of Epidemiology in Public Health Practice.*
"""
    )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    """
### SIRDV Epidemiological Decision Dashboard

**Project title:** SIRDV Epidemiological Decision Dashboard  
**Built using:** Python and Streamlit  
**Name:** Omar Abdul-Rahman  

GitHub: `https://github.com/o-ma-r6287`  
LinkedIn: `https://www.linkedin.com/in/omar-abdul-rahman-19729323b/`
"""
)

st.caption(
    "Version 1.0 | For educational scenario exploration only. Not an official forecasting tool."
)
