# ReadmitRisk

Hospital Readmission Risk and Resource Allocation Platform

A data analytics project that predicts which diabetic patients are at high risk of being readmitted to the hospital within 30 days, and helps hospitals allocate limited follow-up care resources to the patients who need them most.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Business Value](#business-value)
- [Dataset](#dataset)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Hypotheses](#hypotheses)
- [Modeling Approach](#modeling-approach)
- [Dashboard](#dashboard)
- [Team](#team)

## Overview

Thirty day hospital readmission among diabetic patients is common, costly, and often preventable. Most hospitals do not have a reliable way to know which patients need follow up care after they are discharged. This project builds a system that scores each patient's readmission risk before discharge, groups patients into risk tiers, and simulates how a limited follow up care budget can be allocated to prevent the most readmissions.

## Problem Statement

Follow up care resources such as case managers, home visits, and phone check ins are limited. Hospitals usually assign these resources on a first come basis or apply them to everyone equally, instead of targeting the patients who are actually at risk. The result is that high risk patients are missed, low risk patients receive resources they do not need, and both readmission rates and financial penalties stay high.

## Business Value

- **Targeted intervention**: Flags the specific patients most likely to be readmitted before they are discharged.
- **Smarter budget use**: Directs a fixed follow up care budget to where it prevents the most readmissions.
- **Fewer penalties**: Lower 30 day readmission rates reduce CMS style readmission penalties.
- **Better patient outcomes**: At risk patients receive proactive care instead of falling through the cracks.

## Dataset

**Source**: Diabetes 130-US Hospitals (1999-2008), available on the UCI Machine Learning Repository and on Kaggle as `brandao/diabetes`.

- 101,766 patient encounters (rows)
- 50 raw features
- Approximately 11 percent of encounters were readmitted within 30 days

### Key Feature Groups

| Group | Example Features |
| --- | --- |
| Demographics | race, gender, age (10 year bands) |
| Admission details | admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital |
| Clinical activity | num_lab_procedures, num_procedures, num_medications, number_diagnoses |
| Visit history | number_outpatient, number_emergency, number_inpatient |
| Diagnoses | diag_1, diag_2, diag_3 (ICD-9 codes) |
| Medications | 23 drug specific columns such as metformin and insulin, plus change and diabetesMed |
| Target | readmitted: NO, less than 30 days, greater than 30 days |

### Data Quality (Missingness)

Based on published rates for this dataset (Strack et al., 2014):

| Column | Missing |
| --- | --- |
| weight | 96.9% |
| medical_specialty | 49.1% |
| payer_code | 39.6% |
| race | 2.2% |

## Data Cleaning and Preprocessing

- Drop or bucket sparse columns (weight, payer_code, medical_specialty); impute the remainder as an "Unknown" category
- Remove encounters discharged to hospice or expired, since these are not eligible for readmission and would cause label leakage
- Collapse ICD-9 diagnosis codes (diag_1 to diag_3) into 9 standard clinical groups: circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other
- Encode categorical variables: one hot encoding for nominal fields, ordinal encoding for age bands
- Scale numeric features (time_in_hospital, num_medications, lab counts) for distance based models
- Address class imbalance (about 11 percent are readmitted within 30 days) using class weighting or SMOTE

## Exploratory Data Analysis

Planned analysis includes:

- Readmission rate by age group, admission type, and discharge disposition
- Medication count and insulin change compared against readmission outcome
- Primary diagnosis category compared against readmission risk
- Length of stay distribution split by readmission status

## Hypotheses

- **H1**: More prior inpatient visits are associated with higher 30 day readmission risk
- **H2**: Longer hospital stays are correlated with higher readmission risk
- **H3**: Recent diabetes medication changes shift readmission patterns compared to no change

## Modeling Approach

### Feature Selection

SHAP importance ranking on the trained classifier, combined with a correlation check across the 23 medication flag columns to remove redundancy. A baseline model using all 50 features is compared against a reduced model using the top 15 to 20 SHAP features, evaluated on accuracy, recall on the under 30 day class, and training time.

### Models

| Component | Model | Function | Role |
| --- | --- | --- | --- |
| Readmission Risk Classifier | XGBoost / Random Forest | Predicts 30 day readmission probability per encounter | Decision support |
| Sequential Risk Forecaster | LSTM / GRU | Learns per patient encounter history to forecast next visit risk | Forecasting |
| Follow-Up Budget Allocator | Optimization (knapsack / linear programming) on classifier scores | Allocates a fixed follow up care budget to maximize prevented readmissions | Value creation |
| Patient Risk-Tier Clustering | K-Means | Segments patients into interpretable risk tiers for care teams | Value creation |

## Dashboard

Planned framework: Streamlit or Dash (final choice to be decided), chosen for fast prototyping of live scoring with interactive controls.

Planned layout includes four tabs:

1. **EDA**: exploratory charts and summary statistics
2. **Risk Scorer**: enter patient details (age band, time in hospital, number of medications, prior visits, primary diagnosis) and get a live risk score with the top SHAP drivers behind it
3. **Budget Sim**: simulate how a fixed follow up care budget would be allocated across patients
4. **Risk Tiers**: view patient segments produced by the clustering model

## Team

Group 8, Data Analytics Laboratory, Summer 2026

- Nafiul Islam (ID: 0152410070)
- Onika Tahmim Subha (ID: 0152330134)
