## __Data Science: Capstone Part 2 (Lightning Talk #2): EDA, Cleaning & Modeling Plan__

## Overview

In Lightning Talk #1, you pitched **at least three** potential capstone topics and performed basic EDA on each. For Lightning Talk #2, you will **choose one topic** from those proposals and go deeper: refine your problem framing, clean the data, perform in-depth EDA, and outline how you will preprocess features and build models.

This deliverable moves you from exploration to a focused, modeling-ready dataset and a clear plan for your capstone.

__Goal__: Present a refined problem statement and progress update on **one** selected topic, supported by a cleaned dataset, in-depth EDA (including EDA for modeling), and an outlined preprocessing and modeling approach.

### REQUIREMENTS

1. **Choose one topic** from your Lightning Talk #1 proposals. Complete EDA and data cleaning on **that topic only**.

2. On your selected topic, you must:
   - Refine your problem statement ([Reference](https://github.com/ga-bahrain/dsb-pt3-bahrain-2026/blob/main/modules/supervised-machine-learning/0.How%20to%20Frame%20a%20Data%20Science%20Problem/ML-Framing-a-Data%20Science-Problem.pdf))
   - Create or compile a data dictionary
   - Clean the data and perform in-depth EDA (insights + EDA for modeling)
   - Outline proposed preprocessing/feature engineering and models

3. Your dataset must still meet the requirements from Lightning Talk #1 (at least **15k+** rows and **10+** columns, IA-approved, no NLP/text-only datasets).

### Note

- ___If you cannot finalise on your which data to select, reach out to your IAs.___
- For **image-based** topics, your EDA should cover: classes (target), settings (transformation), and data limitations.

## Deliverable Format & Submission

1. **Slide presentation**
2. **Jupyter notebook** 

Submit both deliverables on **Google Classroom** by **19 September 2026 (Wednesday)**.

**Jupyter Notebook**

Your Jupyter Notebook should contain the following elements for your **one selected topic**:

- Refined problem statement (Not mandatory but recommended to add it here)
- Objectives (key questions guiding your analysis and modeling)
- Data dictionary (mandatory)
- Data overview (source, format, `.head()`, `.info()`, `.describe()`, etc.)
- Data cleaning (missing values, outliers, inconsistencies)
- Exploratory Data Analysis (EDA)
  - Trends, relationships, anomalies
  - Outlier and missing-value treatment (document what you did)
  - Univariate, bivariate, or multivariate analysis
  - Histograms, box plots, bar charts as appropriate
  - Correlation matrix for continuous columns (required if applicable)
  - Chi Square for categorical columns (if applicable)
- Data preprocessing & feature engineering — **outline only** (no need to execute)
- Modelling approach — **outline only** (no need to execute)

For reference, use [Sample Jupyter Notebook](./lightning-talk-2.ipynb).

**Slide Presentation**

Your slide presentation should cover (keep data sections concise):

- Refined problem statement (including **success metrics**)
- Potential audience
- Goals & objectives
  - Type of model(s) (e.g., regression / classification)
  - Number of models you plan to build
  - Evaluation metrics
- About the data (data dictionary and EDA for modeling — brief)
- Progress report (challenges & next steps)

__You are allocated 5 minutes per person to present.__

## Suggested Ways to Get Started

Begin by asking:

- What did cleaning reveal about data quality, bias, or limitations?
- Which features are most promising for modeling, and which need transformation or encoding?
- What baseline models and metrics will define success?
- What blockers remain before I can train models in the next phase?

Other tips:

- Document every cleaning decision (what you changed and why) so you can defend it in your talk.
- Separate **EDA for insights** (stakeholder story) from **EDA for modeling** (distributions, correlations, leakage risks).
- Keep preprocessing and modeling sections as a clear plan — you do not need to implement them yet.

## Useful Resources

- [How to give a good lightning talk.](https://www.semrush.com/blog/16-ways-to-prepare-for-a-lightning-talk/)
- [How to Frame a Data Science Problem (PDF)]() **<--- To be covered on Tuesday**

## Important Deadlines

- **EDA Jupyter notebook and presentation slide submission**: 19 August 2026 (Wednesday) — **In Google Classroom**
- **Presentations**: 1-To-1 on 19 August 2026 (Wednesday) — **In Google Classroom**
