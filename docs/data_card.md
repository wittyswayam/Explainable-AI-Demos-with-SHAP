# Data Card: Bank Customer Churn Prediction Dataset

**Version:** 1.0.0  
**Last Updated:** 2024-01-15  
**Dataset ID:** `bank-customer-churn-v1`  
**Maintainers:** XAI Platform Team

---

## Dataset Summary

A synthetic/anonymised tabular dataset representing 10,000 bank customers across France, Spain, and Germany, containing demographic, financial, and behavioural attributes alongside a binary churn label. Widely used in the ML community as a benchmark for binary classification and explainability research.

| Property | Value |
|---|---|
| **Rows** | 10,000 |
| **Features** | 14 raw → 10 model features (after dropping identifiers) |
| **Target** | `Exited` — binary churn indicator |
| **Positive rate** | ~20.4% (2,037 churned) |
| **Source** | Kaggle: shubhammeshram579/bank-customer-churn-prediction |
| **License** | CC0 1.0 Public Domain |
| **Format** | CSV |
| **Size** | ~600 KB |

---

## Dataset Description

### Motivation

Created to benchmark churn prediction models in a retail banking context. The dataset is intentionally synthetic/anonymised to avoid privacy concerns while preserving realistic statistical properties observed in European retail banking portfolios.

### Composition

#### Raw Features

| Column | Type | Range / Values | Missing | Description |
|---|---|---|---|---|
| `RowNumber` | Integer | 1–10,000 | 0% | Row index — dropped |
| `CustomerId` | Integer | ~15M range | 0% | Identifier — dropped |
| `Surname` | String | Various | 0% | Identifier — dropped |
| `CreditScore` | Integer | 350–850 | 0% | Customer creditworthiness score |
| `Geography` | Categorical | France, Spain, Germany | 0% | Country of residence |
| `Gender` | Categorical | Male, Female | 0% | Customer gender |
| `Age` | Integer | 18–92 | 0% | Customer age in years |
| `Tenure` | Integer | 0–10 | 0% | Years as bank customer |
| `Balance` | Float | 0–250,919 | 0% | Account balance in pounds |
| `NumOfProducts` | Integer | 1–4 | 0% | Number of bank products held |
| `HasCrCard` | Binary | 0, 1 | 0% | Has credit card |
| `IsActiveMember` | Binary | 0, 1 | 0% | Active member flag |
| `EstimatedSalary` | Float | 11–199,992 | 0% | Annual salary estimate |
| `Exited` | Binary | 0, 1 | 0% | **Target: 1=churned, 0=retained** |

#### Target Distribution

| Label | Count | Percentage |
|---|---|---|
| 0 (Retained) | 7,963 | 79.6% |
| 1 (Churned) | 2,037 | 20.4% |

**Class imbalance ratio:** ~3.9:1 (retained:churned). Mitigation: `class_weight="balanced"` in tree models; stratified CV splits.

### Statistical Summary

#### Numerical Features

| Feature | Mean | Std | Min | 25th | Median | 75th | Max |
|---|---|---|---|---|---|---|---|
| CreditScore | 650.5 | 96.7 | 350 | 584 | 652 | 718 | 850 |
| Age | 38.9 | 10.5 | 18 | 32 | 37 | 44 | 92 |
| Tenure | 5.0 | 2.9 | 0 | 3 | 5 | 7 | 10 |
| Balance | 76,485 | 62,397 | 0 | 0 | 97,199 | 127,644 | 250,899 |
| NumOfProducts | 1.53 | 0.58 | 1 | 1 | 2 | 2 | 4 |
| EstimatedSalary | 100,090 | 57,510 | 11 | 51,002 | 100,194 | 149,388 | 199,992 |

*Note: ~32% of customers have Balance=0, creating a bimodal balance distribution.*

#### Categorical Features

| Feature | Values | Distribution |
|---|---|---|
| Geography | France: 5,014 (50.1%), Spain: 2,477 (24.8%), Germany: 2,509 (25.1%) | |
| Gender | Male: 5,457 (54.6%), Female: 4,543 (45.4%) | |
| HasCrCard | Yes: 7,055 (70.6%), No: 2,945 (29.4%) | |
| IsActiveMember | Yes: 5,151 (51.5%), No: 4,849 (48.5%) | |

### Key Correlations with Target (Churn)

| Feature | Correlation (point-biserial or Cramér's V) | Direction |
|---|---|---|
| Age | 0.34 | Positive — older → more churn |
| Balance (=0 flag) | 0.29 | Positive — zero balance → more churn |
| NumOfProducts | 0.24 | Negative — more products → less churn |
| IsActiveMember | -0.16 | Negative — active → less churn |
| Geography (Germany) | 0.19 (V) | Germany has highest churn rate (32%) |
| Gender (Female) | 0.11 (V) | Female customers churn more (25% vs 16%) |
| CreditScore | -0.04 | Weak negative — higher score → slightly less churn |
| Tenure | 0.01 | Negligible — not a reliable churn signal |
| EstimatedSalary | 0.01 | Negligible |

---

## Collection Process

### Source and Method

The dataset was published on Kaggle as a bank churn benchmark. The original source is not fully documented by the publisher. The data appears to be:
- Synthetic (artificially generated to match realistic statistical properties)
- Or anonymised historical records from a European retail bank

**No collection methodology documentation is available.**

### Timeframe

No temporal information is included in the dataset. The churn label reflects a point-in-time snapshot rather than a specific observation window.

### Who Collected the Data

Unknown. The Kaggle dataset was published by user "shubhammeshram579". The original data generation or collection process is not described.

---

## Preprocessing

### Applied in This Platform

1. **Column normalisation:** All column names lowercased, spaces replaced with underscores
2. **Identifier removal:** `RowNumber`, `CustomerId`, `Surname` dropped (zero predictive value, privacy risk)
3. **Numerical scaling:** `MinMaxScaler` applied to all continuous features (range [0, 1])
4. **Categorical encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on `Geography` and `Gender`
5. **Output format:** `set_config(transform_output='pandas')` preserves column names through pipeline

### Rationale for MinMaxScaler vs StandardScaler

MinMaxScaler was chosen over StandardScaler because:
- SHAP feature value display is more interpretable (0–1 bounded)
- XGBoost/LightGBM are invariant to monotonic scaling — choice has no impact on model performance
- Several features (`Balance`, `EstimatedSalary`) have outlier distributions that StandardScaler handles poorly without further outlier treatment

### No Outlier Removal

Outliers in `Age` (max=92), `Balance` (bimodal with 32% zeros), and `EstimatedSalary` are retained as they represent genuine customer population heterogeneity rather than measurement error.

---

## Uses

### Appropriate Uses

- Binary classification benchmarking (churn, retention)
- Explainability research (SHAP, LIME, counterfactual explanations)
- MLOps pipeline development and testing
- Fairness analysis methodology development
- Feature importance comparison across algorithms

### Inappropriate Uses

- **Production deployment without validation:** This dataset's label quality and temporal scope are not documented. Any production deployment in a real banking context requires validation against actual labelled historical data.
- **Sensitive demographic inference:** Do not use `Gender` or `Geography` for downstream discrimination
- **Personal identification:** Customer identifiers have been removed; do not attempt re-identification

---

## Known Limitations

1. **Unknown provenance:** Original data source, collection method, and temporal scope not documented
2. **Binary gender:** Dataset only includes Male/Female — not representative of full gender diversity
3. **Limited geography:** Three European countries only; not generalisable globally without retraining
4. **Synthetic risk:** If synthetic, the statistical properties may not perfectly reflect real banking populations
5. **No temporal dimension:** Cannot model concept drift, seasonal patterns, or macroeconomic cycles
6. **Salary quality:** `EstimatedSalary` column has unknown estimation methodology; SHAP analysis confirms low predictive value

---

## Ethical Considerations

| Concern | Assessment | Mitigation |
|---|---|---|
| **Gender bias** | Female customers have genuinely higher churn rate (25% vs 16%) — model reflects this | Fairness audit per Model Card; DPD monitoring in production |
| **Geographic discrimination** | German customers flagged at higher rate — reflects real data pattern | Geography-stratified ROC-AUC monitoring; legal review before deployment |
| **Sensitive attributes** | Gender and geography included as features | Legal review required; consider excluding in jurisdictions with restrictions |
| **Privacy** | Identifiers removed; no re-identification risk from remaining features | Satisfactory for research use |

---

## Versioning and Updates

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2023-06-01 | Initial dataset integration |
| 1.0.1 | 2024-01-15 | Column normalisation pipeline; data card authored |

---

## Citation

```bibtex
@dataset{bank_churn_2023,
  title  = {Bank Customer Churn Prediction},
  author = {shubhammeshram579},
  year   = {2023},
  url    = {https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction},
  note   = {CC0 1.0 License}
}
```

---

*This data card follows the Datasheets for Datasets format (Gebru et al., Communications of the ACM, 2021).*
