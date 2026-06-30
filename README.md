# ⚽ FIFA World Cup 2026 Analysis & Monte Carlo Simulator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning analytical dashboard and simulation engine designed to predict the **FIFA World Cup 2026**. The system uses historical international match data, dynamic Elo ratings, linear-decay form curves, Poisson goal estimators, and a Random Forest match outcome classifier to model 10,000 tournament iterations. The standings, probabilities, and H2H predictions are aligned with elite sports analytics benchmarks.

---

## 🚀 Features

### 🏆 1. Monte Carlo Tournament Simulator
*   **10,000-Run Simulation**: Runs a high-performance simulation loop across all 12 groups (A to L) and the new Round of 32 knockout bracket.
*   **Opta-Aligned Win Probabilities**: Calibrated to match real-world supercomputer models (e.g., Spain, France, and Argentina as frontrunners).
*   **Dynamic Group Finish Ordering**: Automatically sorts the group stage draw view in the predicted finishing order (1st to 4th) based on simulated averages.

### 📊 2. Historical Insights & EDA
*   **International Goal Trends**: Analyzes goal scoring patterns over time.
*   **H2H Top 20 Matrices**: Interactive matrices showing historical performance, goals, and results among top international teams.
*   **Feature Importance Plots**: Visualizes features (Elo difference, recent goal forms, H2H statistics) that drive prediction accuracy.

### 🔮 3. Interactive H2H Match Simulator
*   **Custom Matchups**: Select any two international teams to estimate expected goals (xG), win/draw/loss probabilities, and view detailed historic head-to-head performance.
*   **Poisson Scoreline Generator**: Generates realistic scorelines using random Poisson draws modeled from teams' offensive/defensive form and Elo ratings.



---

## 🛠️ Tech Stack
*   **Frontend & Dashboard**: [Streamlit](https://streamlit.io/) (utilizing custom responsive CSS grids)
*   **Scientific Computing**: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (PoissonRegressor, RandomForestClassifier, GridSearchCV)
*   **Interactive Visualizations**: [Plotly](https://plotly.com/)
*   **Notebook Generation**: Automated via [create_notebook.py](create_notebook.py)

---

## 🗂️ Project Directory Structure

```text
├── Dataset/
│   └── results.csv
├── Image/
│   └── Frame 1.png
├── models/
│   └── dashboard_data.pkl
├── app.py
├── prepare_data.py
├── create_notebook.py
├── requirements.txt
└── EDA.ipynb
```

---

## ⚙️ Mathematical & Modeling Methodology

### 📈 1. ELO Rating System
Historical Elo ratings are calculated match-by-match since 1872 using the standard update formula:
$$R_{\text{new}} = R_{\text{old}} + K \times (S - E)$$
*   **Elo Scaling**: Elo features are scaled (divided by 400.0) before fitting to prevent exponential overflow in the Poisson log-link function.

### 📉 2. Linear-Decay Form Curves
A team's recent form (points accumulated, goals scored, and goals conceded) is computed using a linear decay average over their last 10 matches, giving higher weight to the most recent games:
$$\text{Form} = \frac{\sum_{i=1}^{n} (n - i + 1) \times \text{Metric}_i}{\sum_{i=1}^{n} (n - i + 1)}$$

### ⚽ 3. Goal Modeling (Poisson Regression)
Expected goals (xG) are modeled using a Poisson generalized linear model (GLM) with a log-link function:
$$\log(\lambda_1) = \beta_0 + \beta_1 \frac{\text{Elo}_1}{400} + \beta_2 \frac{\text{Elo}_2}{400} + \beta_3 \frac{\text{Elo}_1 - \text{Elo}_2}{400} + \beta_4 \text{GF\_Form}_1 + \beta_5 \text{GA\_Form}_2$$
Goals scored are then sampled from:
$$\text{Goals}_1 \sim \text{Poisson}(\lambda_1)$$

---

## 📥 Local Installation & Run Guide

### 1. Clone the Repository
```bash
git clone https://github.com/aryannverse/Fifa-2026-WC-Prediction-.git
cd Football-Analysis
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training & Preprocessing
To re-run the dataset processing, train models via GridSearchCV, and execute the 10,000-run simulation:
```bash
python prepare_data.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

