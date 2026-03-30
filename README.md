# Customer Churn Predictor

A data analysis and machine learning project built to demonstrate product thinking, business framing, and technical execution — the combination that matters in PM and Technical PM roles.

---

## Problem Statement

Acquiring a new customer costs 5–7× more than retaining an existing one. For a telecom company with thousands of customers, even a 1% reduction in monthly churn translates to meaningful revenue recovery. Yet most companies treat churn as a lagging indicator — they notice it after the customer has already left.

**The goal of this project is to shift churn from a lagging metric to a leading one.** By identifying at-risk customers before they cancel, retention teams can intervene with targeted offers, proactive outreach, or product improvements — at a fraction of the cost of re-acquisition.

The business question isn't *"can we predict churn?"* — it's **"which customers should we call tomorrow, and what should we say?"**

---

## What I Built

### 1. SQL-first business analysis (`churn_analysis.ipynb`)
Before touching any ML, I loaded the dataset into a SQLite database and used real SQL queries to answer the questions a PM would ask in a strategy meeting:

- What is our overall churn rate? *(26.5% — significant)*
- Which customer segments churn most? *(Month-to-month: 42% vs. two-year: 3%)*
- How does churn vary by product? *(Fiber optic: 41% despite being the premium tier)*
- What is the financial scale of the problem? *(~$2.8M annualised revenue at risk)*

SQL first, model second. Aggregates before algorithms.

### 2. Exploratory data analysis with business commentary
Every cleaning decision and chart is annotated with what it means for the business, not just what it shows technically. The EDA is written to be readable by a PM or analyst who doesn't write Python.

### 3. Logistic regression classifier
I chose logistic regression over gradient boosting deliberately:

- **Interpretability:** Coefficients are human-readable. A PM can explain to a non-technical stakeholder exactly why a customer was flagged.
- **Calibrated probabilities:** The output is a true probability, not a relative score. "73% churn risk" is actionable in a way that "score: 0.83" is not.
- **Honest baseline:** If logistic regression performs well, we understand what signal exists in the data before adding complexity.

Model is trained with `class_weight='balanced'` because missing a churner (false negative) is more costly than falsely flagging a retained customer (false positive).

### 4. Rigorous evaluation
Beyond accuracy (which is misleading on imbalanced classes), I report:
- Precision, Recall, F1 per class
- ROC-AUC (~0.84)
- Confusion matrix with explicit discussion of which errors matter more

### 5. SHAP explainability
Global feature importance tells you what drives churn across all customers. SHAP waterfall plots tell you *why this specific customer* is flagged — which is what a CSM actually needs to open a productive conversation.

### 6. Streamlit app (`app.py`)
A clean, simple interface where a retention analyst can enter a customer's details and instantly see their churn probability and the top factors driving it, in plain English.

---

## Key Findings

1. **Contract type is the strongest churn signal.** Month-to-month customers churn at ~42% vs. 11% (one-year) and 3% (two-year). Incentivising contract upgrades is the single highest-leverage retention lever.

2. **The first 12 months are the highest-risk window.** Churned customers average ~18 months tenure vs. ~38 months for retained customers. Onboarding experience and early product value are disproportionately important.

3. **Fiber optic customers churn at 41% — despite paying the most.** This is a product-market fit signal, not just a price sensitivity issue. High-spend customers churning suggests unmet expectations, not just affordability concerns.

4. **Electronic check payers churn significantly more than auto-pay customers.** Manual payment may be a proxy for low engagement or lower switching cost. A nudge to auto-pay could be both a retention signal and a genuine convenience improvement.

5. **Protective factors exist.** Customers with tech support, online security, and online backup subscriptions churn at lower rates. These services create stickiness — bundling them into retention offers may be more effective than price discounts.

---

## Business Recommendations

| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Launch a "commit and save" campaign converting M2M customers to annual plans | 42% → 3% churn rate difference; even 10% conversion materially moves the needle |
| High | Build a 90-day early-lifecycle check-in programme | Churn spikes in months 1–12; proactive contact at this stage is cheapest |
| Medium | Commission a fiber product satisfaction audit | 41% churn on the premium tier is a signal the product isn't delivering on its promise |
| Medium | Bundle tech support and security in retention offers | These services correlate with retention — position them as value, not upsell |
| Low | Experiment with auto-pay incentives | Lower churn among auto-pay users; causality unclear but worth testing |

---

## Model Limitations & Honest Caveats

- **Correlation ≠ causation.** Some features (e.g. no tech support) may be proxies for low engagement rather than levers to pull. A/B testing would be needed to validate causal claims.
- **Static snapshot.** This model scores customers at a point in time. A production system would score customers monthly and track score trajectories.
- **No outcome measurement.** We don't know whether customers who received retention interventions actually stayed longer — that would require a holdout experiment.
- **Class imbalance.** ~26% churn is manageable but means the model catches ~79% of churners at the cost of some false positives. The acceptable trade-off depends on retention offer economics.

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| **Python** | All data processing, ML training, and app development |
| **SQL (SQLite)** | Business insight queries in the notebook — churn rates, revenue at risk, segment analysis |
| **pandas** | Data loading, cleaning, feature engineering |
| **scikit-learn** | Train/test split, StandardScaler, LogisticRegression, evaluation metrics |
| **SHAP** | Global and per-customer explainability (LinearExplainer, summary_plot, waterfall) |
| **Streamlit** | Interactive prediction app with SHAP visualisation |
| **Jupyter Notebook** | End-to-end narrative analysis combining code, output, and business commentary |
| **Product thinking** | Business framing throughout — ROI estimates, segment prioritisation, PM-style recommendations |

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Run the notebook
Open `churn_analysis.ipynb` in Jupyter and run all cells. This will:
- Download the dataset automatically (no Kaggle login needed)
- Run SQL analysis, EDA, model training, and SHAP evaluation
- Save `model_artefacts.pkl` to the project directory

```bash
jupyter notebook churn_analysis.ipynb
```

### Step 2 — Launch the Streamlit app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 20 features.
Source: [IBM on GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d) — fetched directly via public URL, no login required.

---

*Built as a portfolio project to demonstrate the intersection of data analysis, machine learning, and product thinking.*
