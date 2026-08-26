## __Data Science: Capstone Final Deliverable__

## Overview

Your capstone is the culmination of work from Lightning Talk #1 (topic selection), Lightning Talk #2 (cleaning, in-depth EDA, and modeling plan), and the modeling phase. You will deliver a complete, reproducible analysis on **one** approved topic: build and compare **three models**, document your process, and present your findings to an audience.

This deliverable demonstrates that you can frame a real-world problem, prepare data, explore it rigorously, engineer features, train and evaluate multiple models, and recommend the best approach with clear justification.

__Goal__: Submit a complete Jupyter notebook (your technical report), and a presentation slide deck that document your end-to-end capstone project.


### REQUIREMENTS

1. Continue on the **one topic** you selected and developed in Lightning Talk #2.

2. Build **at least three basic models**, and spend time  to **improve each of** them using **Gridsearch or other model improvement technique**.    
    - Compare your best versions of the model for your topic. Document the process and your steps for improvement.

3. Use appropriate evaluation metrics for your problem type (regression vs classification vs clustering).

4. Submit **the deliverables** on **Google Classroom** by **1st of September 2026 (Tuesday) @ 11:59 PM**.

### Note

- ___Your notebook (technical report) and slides should tell a consistent story — same problem, data, models, and conclusions.___
- Code in your Jupyter notebook should be **well-commented** so reviewers can follow your reasoning without guessing.

## Deliverable Format & Submission

Submit the following on **Google Classroom** by **1st of September 2026 (Tuesday) @ 11:59 PM**:

1. **Complete Jupyter notebook (Technical report)** (with markdowns and comments)
2. **Presentation slide deck**

---

### 1. Jupyter Notebook (Complete, with Comments)

Your notebook should be fully executed (or runnable end-to-end) and include clear comments explaining **what** each section does and **why** you made key decisions.

Your notebook should align with the technical report and include:

- Capstone project title, your name, and topic name and cohort (DSB >>**Cohort Ref.**<<)
- Introduction (brief overview of your data and context)
- Problem statement + aim
- Objectives (key questions guiding analysis and modeling)
- Data dictionary
- Data overview (source, format, `.head()`, `.info()`, `.describe()`, etc.)
- Data cleaning (missing values, outliers, inconsistencies)
- Exploratory Data Analysis (EDA)
  - Trends, relationships, anomalies
  - Outlier and missing-value treatment
  - Univariate, bivariate, or multivariate analysis
  - Histograms, box plots, bar charts as appropriate
  
- **Insights** (bullet-point summary of useful findings)
- **Preprocessing for modeling**
    - Data preprocessing & feature engineering (implemented, not just outlined)
- **Feature Selection**
    - Correlation matrix for continuous columns (for continuous columns) 
    - Chi square analysis (for categorical columns) 
- **Model building** — three models, best version of each
  - Modeling steps for each model
  - Performance metrics for each model
- **Conclusion**
  - Model comparison table
  - Best model and justification
  - Discussion (business implications, limitations, future improvements)
- References & appendix (tools, libraries, sources)


| Section | What to include |
| -------- | ---------------- |
| **Title block** | Capstone project title, your full name, topic name, and cohort (DSB PT3) |
| **Introduction** | Brief overview of your data |
| **Problem Statement + Aim** | Real-world problem your model aims to solve |
| **Objectives** | Key questions guiding analysis and modeling |
| **Data Inspection** | |
| → Data Dictionary | Columns in table format with descriptions |
| → Data Overview | Source, format, inspection outputs |
| → Data Cleaning | How you handled missing values, outliers, inconsistencies. |
| **Exploratory Data Analysis (EDA)** | Trends, relationships, anomalies; treatments; univariate/bivariate/multivariate analysis; correlation matrix (if applicable) |
| → Insights | Bullet-point summary of discoveries |
| → EDA for Modeling | Analysis focused on modeling readiness |
| **Data Preprocessing & Feature Engineering** | Transforming, encoding, and preparing features |
| **Model Building** | Three models — document **best version only** per model; appropriate metrics per model type |
| → Model 1 | Include basic versions and how you improved each basic version |
| → Model 2 | Include basic versions and how you improved each basic version |
| → Model 3 | Include basic versions and how you improved each basic version |
| **Conclusion** | |
| → Model Comparison | Table comparing performance; which model is best; why |
| → Discussion | Business implications; limitations and future improvements |
| **References & Appendix** | Cite tools, libraries, and data sources |

---

### 2. Slide Presentation

Your slide deck should follow **Capstone Presentation Guidelines** below. Present only the **best version** of each model and how it improved from the Base model. However, you can compare the performance of different versions of the model to show how it improved, only if you think it is necessary.

**Introduction**
- Introduction on your topic

**Problem framing**
- Problem statement + aim

**Data summary**
- What data are you using and what do the records represent? (bullet points)

**Exploratory Data Analysis (EDA)**
- Key findings: useful insights discovered during analysis

**Modeling approach**
- Brief overview of how you approached modeling (features, preprocessing, model types)

**Model 1** (best version only)
- Best model version (what you did to it Ex: GridSearch, Reguralization)
- Key features & preprocessing
- Performance Metrics and how it compared to the Base Model. Optional: Comparing with other model versions, if relevant.
- Include the features that influenced the model

**Model 2** (best version only)
- Best model version (what you did to it Ex: GridSearch, Reguralization)
- Key features & preprocessing
- Performance Metrics and how it compared to the Base Model. Optional: Comparing with other model versions, if relevant.
- Include the features that influenced the model

**Model 3** (best version only)
- Best model version (what you did to it Ex: GridSearch, Reguralization)
- Key features & preprocessing
- Performance Metrics and how it compared to the Base Model. Optional: Comparing with other model versions, if relevant.
- Include the features that influenced the model

**Conclusion / Model comparison**
- Summary table comparing performance across all three models
- Which model is the best? and Why?
- What are the business implications of your findings?
- What are the limitations of your study?
- What are the potential improvements for your study?

**Each student is allocated a maximum of 15 minutes for presentation**

---

## Important Deadlines

- **Capstone submission (notebook + technical report + slide deck)**: 1st of September 2026 (Tuesday) @ 11:59 PM — **In Google Classroom**
- **Presentations**: 2nd - 3rd September 2026 (Wednesday - Thursday) @ 9:00 AM to 5:00 PM **[Presentation Schedule TBA on 1st September 2026]**
