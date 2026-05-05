# 🦠 SIRDV Epidemiological Decision Dashboard

Interactive epidemiological simulation dashboard built with **Python** and **Streamlit** for exploring infectious disease dynamics, comparing intervention strategies, and analyzing healthcare system impact.

---

## 📊 Overview

This dashboard allows users to explore how key epidemiological parameters influence disease spread over time, including:

- Transmission rate (β)
- Recovery rate (γ)
- Mortality rate (μ)
- Vaccination rate
- Population size
- Healthcare capacity (ICU)

It is designed for:
- Public health education
- Epidemiology coursework
- Scenario-based decision analysis
- Portfolio demonstration of data science + health analytics

---

## 🧪 Models Included

### **SIR Model**
Susceptible → Infected → Recovered  

### **SIRD Model**
Susceptible → Infected → Recovered / Deceased  

### **SIRDV Model**
Susceptible → Infected → Recovered / Deceased / Vaccinated  

---

## ⚙️ Features

- SIR, SIRD, and SIRDV model selection  
- Adjustable simulation parameters via interactive sidebar  
- Public Health Decision Brief with risk classification:
  - 🟢 Low  
  - 🟡 Moderate  
  - 🔴 High  
  - ⚫ Critical  
- ICU capacity monitoring and overflow detection  
- Key metrics: peak infection, attack rate, and final outcomes  
- Scenario comparison with side-by-side analysis  
- Early vs Late Intervention demo scenario  
- Policy Effectiveness Score  
- Sensitivity analysis  
- Intervention timing analysis  
- Day-by-day outbreak view  
- Simulated geographic risk heat map  
- Interactive Plotly time-series visualizations  
- Downloadable CSV reports and text summaries  
- Optional PNG chart export (Kaleido)  
- Model explanation, assumptions, and limitations section  

---

## 🧭 Dashboard Tabs

1. **Simulation Results** – Main outbreak curves and metrics  
2. **Data Table** – Full simulation dataset  
3. **Model Explanation** – SIRDV framework and parameter interpretation  
4. **Compare Scenarios** – Side-by-side policy comparison  
5. **Sensitivity Analysis** – Parameter impact testing  
6. **Day-by-Day View** – Snapshot of outbreak at a selected day  
7. **Intervention Timing** – Impact of delayed response  
8. **Risk Heat Map** – Simulated geographic risk visualization  
9. **Methods & Assumptions** – Model transparency and limitations  

---

## 🚀 How to Use

1. Start with a preset scenario or adjust parameters in the sidebar  
2. Click **Run Simulation**  
3. Review the **Public Health Decision Brief** and key metrics  
4. Use **Compare Scenarios** to test intervention strategies  
5. Explore additional tabs for deeper analysis  

A realistic baseline scenario is enabled by default for immediate visualization.

---

## 🧠 Key Metrics Explained

- **Peak Infected**: Maximum number of active infections  
- **Peak Day**: When infections reach their highest point  
- **Attack Rate**: % of population infected over time  
- **R₀ (Basic Reproduction Number)**: β / γ  
- **ICU Status**: Whether healthcare capacity is exceeded  

---

## ⚠️ Model Assumptions

- Closed population (no migration or births)
- Homogeneous mixing (equal contact probability)
- Deterministic model (no randomness)
- Simplified vaccination and intervention effects
- Simplified hospitalization and ICU modeling

---

## 🚧 Limitations

- No age or demographic structure  
- No real geographic spread modeling  
- No stochastic uncertainty  
- Not calibrated to real-world surveillance data  
- Designed for **education and scenario comparison**, not forecasting  

---

## ⚠️ Disclaimer

**For educational scenario exploration only.**  
This dashboard is not an official forecasting tool and should not be used for real-world public health decision-making without validated data and expert review.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/o-ma-r6287/sirdvdashboard.git
cd sirdvdashboard
python -m venv venv

# Activate environment
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```
