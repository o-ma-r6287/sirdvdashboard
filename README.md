# 🦠 SIRDV Epidemiological Decision Dashboard

An interactive Streamlit dashboard for modeling infectious disease spread using SIR, SIRD, and SIRDV compartmental models.

## Overview

This app allows users to explore how infection rate, recovery rate, death rate, vaccination rate, and population size affect disease spread over time.

It is designed for public health education, epidemiology coursework, and scenario-based decision analysis.

## Models Included

### SIR Model

Susceptible → Infected → Recovered

### SIRD Model

Susceptible → Infected → Recovered / Dead

### SIRDV Model

Susceptible → Infected → Recovered / Dead / Vaccinated

## Features

* Interactive Streamlit dashboard
* SIR, SIRD, and SIRDV model selection
* Adjustable simulation parameters
* Plotly time-series visualizations
* Summary metrics for peak infection and final outcomes
* Side-by-side scenario comparison
* Downloadable simulation results as CSV
* Educational model explanation tab

## Tech Stack

* Python
* Streamlit
* Pandas
* Plotly

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run SIRDVST.py
```

## Required Files

```text
SIRDVST.py
Assignment3_Functions_Solution.py
requirements.txt
```

## Example Use Cases

* Compare mild vs. severe outbreak scenarios
* Explore vaccination effects on disease spread
* Estimate peak infection timing
* Demonstrate compartmental disease modeling concepts

## Disclaimer

This dashboard is for educational and demonstration purposes only. It uses simplified model assumptions and should not be used for real-world public health forecasting.

## Author

Omar Rulida Abdul-Rahman
MPH Candidate

---

Built with Streamlit | Epidemiological Decision Dashboard
