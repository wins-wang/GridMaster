# 🏠 **Welcome to GridMaster**

Welcome to **GridMaster** — an advanced Python toolkit I built to automate hyperparameter tuning and model selection across multiple **classifiers**.

With just a few lines, GridMaster helps you:

✅ Automatically optimize key classifiers over your dataset  
✅ Narrow down from broad industry-recommended parameter grids  
✅ Fine-tune around best ranges using smart linear or logarithmic scaling  
✅ Run multiple models in parallel, selecting the top performer — all without manual, repetitive grid search loops.

---

## 🚀 Supported Models

GridMaster currently supports **classification models** only:  

✅ Logistic Regression  
✅ Random Forest  
✅ XGBoost  
✅ LightGBM  
✅ CatBoost  

> ⚠️ **Note**: Decision Trees are not included, as they are rarely used in industrial hyperparameter optimization workflows.

> 🏗️ **GridMaster** is built on top of `scikit-learn` and integrates seamlessly with popular libraries like `XGBoost`, `LightGBM`, and `CatBoost`, providing a familiar interface for model tuning and evaluation.

---

## 🔍 How It Works

1. **Coarse Search**  
Starts with broad, commonly recommended parameter grids for each classifier (e.g., C, max_depth, learning_rate), providing a robust initial exploration of the search space.

2. **Fine Search**  
Automatically refines parameter ranges around the best coarse result:  
For **linear-scale parameters**, narrows range by ±X% (default ±50%)  
For **log-scale parameters** (like C, learning_rate), adjusts intelligently on the log scale ensuring meaningful search coverage without wasting runs.

3. **Multi-Stage Search**  
Allows multiple fine-tuning rounds with custom precision, eliminating the need to manually loop grid searches for each model.

4. **Multi-Model Comparison**  
Trains and tunes all supported models in parallel, automatically identifies the top performer, and outputs detailed metrics and plots for interpretation.

---

## ✨ Why I Built GridMaster

I designed GridMaster to free myself (and others) from the repetitive burden of per-model grid search, offering a clean, unified, and automated workflow for classification tasks.  

> **_“Laziness fuels productivity.”_**

---

## 🛠️ Get Started

- Check the [Quickstart Guide](usage.md)  
- Run your first multi-model search pipeline  
- Visualize and compare model performances  
- Dive into [Essential Tools](api/core_api.md) and [Advanced Utilities](api/advanced_api.md)  

---

## 📇 About the Author

Hi, I’m Winston Wang — a data scientist passionate about making a meaningful contribution to the world, one data-driven solution at a time.

For feedback or suggestions, feel free to email me at:  
📧 **74311922+wins-wang@users.noreply.github.com**

For more about me, please visit [my personal website](https://winston-wang.com).
