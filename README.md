# 🧠 Explainable AI with SHAP  
**Understanding the “Why” behind Machine Learning Predictions**

This project explores the power of **Explainable Artificial Intelligence (XAI)** using the **SHAP (SHapley Additive exPlanations)** framework.  
It provides clear, beginner-friendly notebooks showing how SHAP helps explain both **classification** and **regression** machine learning models, turning black-box models into interpretable systems.

---

## 📘 Contents

| Folder | Description |
|---------|--------------|
| `intro/` | **Introduction to SHAP** — Covers the fundamental idea of Shapley values, global vs local interpretability, and different visualization types (summary, force, and dependence plots) using a small example dataset. |
| `classification/` | **Classification Case Study** — Demonstrates SHAP on a classification model (e.g., Decision Tree / Random Forest). Shows how each feature affects the prediction probability for each class and provides visual insights into feature importance and decision patterns. |
| `regression/` | **Regression Case Study** — Applies SHAP on a regression model (e.g., Linear Regression / XGBoost Regressor). Illustrates how continuous feature changes impact model output, exploring partial dependence and SHAP summary analysis. |

---

## 🚀 What You’ll Learn
- Understand what **SHAP values** are and how they explain model predictions  
- Visualize **feature importance** using SHAP summary, force, and dependence plots  
- Interpret how **classification and regression** models make decisions  
- Apply SHAP to any scikit-learn model with minimal setup  
- Learn why **Explainable AI (XAI)** is essential for fairness, transparency, and trust in ML models  

---

## 🧩 Tech Stack
- **Language:** Python 3.8+  
- **Libraries:** SHAP, scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter  

---

## ⚙️ How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/Explainable-AI-with-SHAP.git
   cd Explainable-AI-with-SHAP
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook  
   ```bash
   jupyter notebook
   ```

4. Open and explore the notebooks inside:
   - `intro/shap_intro.ipynb`
   - `classification/classification_shap.ipynb`
   - `regression/regression_shap.ipynb`

---

## 🌟 Example Visuals
- **Summary Plot:** Shows global feature importance across all samples  
- **Force Plot:** Explains how each feature pushes a single prediction higher or lower  
- **Dependence Plot:** Displays relationships between features and their SHAP impact  

---

## 💡 About the Project

This repository is designed for anyone exploring **Explainable AI** concepts using **SHAP**.  
It walks through multiple real-world examples that show how machine learning models arrive at their predictions — making them **transparent, interpretable, and trustworthy**.  
Each notebook builds upon the previous one:
- `intro/` helps you grasp the **theory and visualization basics**  
- `classification/` shows **how to interpret decision-based models**  
- `regression/` extends these ideas to **continuous prediction tasks**  
