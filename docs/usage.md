# 🚀 Quickstart Guide

Welcome to **GridMaster** — a powerful, user-friendly toolkit for hyperparameter tuning and model selection.

This guide will help you:

✅ Install the package  
✅ Prepare your data  
✅ Run your first grid search  
✅ Understand key outputs  
✅ Automate multi-stage search  
✅ Visualize results  

---

## 1. Installation

```bash
pip install gridmaster
```

---

## 2. Preparing Your Data

Ensure you have:
- A feature matrix `X` (as a pandas DataFrame or NumPy array)
- A target vector `y`

Example:
```python
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
```

---

## 3. Running Your First Grid Search

By default, you can run:

```python
from gridmaster import GridMaster

gm = GridMaster()
gm.coarse_search()
gm.fine_search()
```

This will:

✅ First perform a **broad search** across a coarse grid of hyperparameters,  
✅ Then perform a **narrower, fine-grained search** around the best coarse parameters,  
✅ Efficiently balance exploration and exploitation for better-tuned models.

> *By default, GridMaster balances system load and speed by using **half of available CPU** cores for parallel search; advanced users can adjust this with the* `n_jobs` *parameter (see* [*Advanced Settings*](api/advanced_api.md#advanced-setting---cpu-parallelism-n_jobs)*).*
>
> *By default, `fine_search` uses **smart mode**, automatically refining the top 2 impactful parameters based on coarse search performance variation. See [Modes](api/core_api.md#method-fine_search) for details.*
>
> *Advanced users can also pass custom **GPU-related estimator parameters** (e.g., `tree_method='gpu_hist'` for XGBoost) through the* `custom_estimator_params` *argument (see* [*Advanced Settings*](api/advanced_api.md#advanced-setting---custom-estimator-parameters-custom_estimator_params)*).*

---

#### 🛠️ Optional: Automate Multi-Stage Grid Search

If you want to automate the whole search pipeline with multiple refinement stages, use:

```python
gm.multi_stage_search(
    model_name='logistic',
    cv=5,
    scoring='accuracy',
    stages=[(0.5, 5), (0.2, 5)]
)
```

This will:

✅ Run an initial coarse search  
✅ Then automatically narrow the parameter grid by ±50% with 5 points  
✅ Then further narrow by ±20% with 5 points  

**Special note for Log Scale Parameters:**  
For hyperparameters that work on a log scale (like `C`, `learning_rate`), the fine grid will be generated intelligently in log space, ensuring the search focuses on meaningful ranges without wasting runs.

---

#### ⚠️ About Default Parameters

GridMaster uses **scikit-learn's default hyperparameter grids**  (e.g., for Logistic Regression, Random Forest, XGBoost),  which are designed for general-purpose datasets.

By default, the scoring metric is `'accuracy'`,  
but you can change it by setting the `scoring` argument directly:

```python
gm.coarse_search(scoring='recall')
gm.fine_search(scoring='f1')
gm.multi_stage_search(scoring='roc_auc')
```

For a full list of available parameters and options,  
see the [Essential Tools](api/core_api.md) section of the documentation.

---
#### 🔑  `multi_stage_search()` or `coarse()` + `fine()`?

By default, calling `multi_stage_search()` alone is equivalent to running `coarse_search()` followed by one `fine_search()` — it automatically performs a two-stage tuning.

However, if you want to perform multi-stage fine-tuning (i.e., multiple refinement rounds), you can pass a custom list of (scale, steps) stages to `multi_stage_search()`, enabling it to handle multi-level tuning in one go.

Alternatively, if you prefer manual control, you can run `coarse_search()` and `fine_search()` separately, allowing you to adjust parameters, scoring metrics, or grids between steps.


---

## 4. Checking Results

```python
summary = gm.get_best_model_summary()
print(summary)
```

Example output:
```json
{
  "model_name": "logistic",
  "best_estimator": "Pipeline(steps=[('clf', LogisticRegression(C=1.0))])",
  "best_params": {"clf__C": 1.0},
  "cv_best_score": 0.96,
  "test_scores": {"accuracy": 0.95, "f1": 0.94, "roc_auc": 0.97}
}
```

This tells you:

- **Which model** performed best

- **Which hyperparameters** were selected

- **Cross-validation score** during tuning

- **Test set performance metrics**

---

#### 📈 Visualizing Results

You can also visualize your model’s search results and performance.

For example, plot the cross-validation score curve:

```python
gm.plot_cv_score_curve(model_name='logistic', metric='mean_test_score')
```

Or visualize the confusion matrix on test data:

```python
gm.plot_confusion_matrix(model_name='logistic', X_test=X_test, y_test=y_test)
```

These plots help you:

✅ Understand how different parameter settings affect performance  
✅ Evaluate your model’s accuracy, recall, precision, and more  
✅ Identify which features or coefficients matter most (see `.plot_model_coefficients()` or `.plot_feature_importance()`)

For details, see [Essential Tools](api/core_api.md).

---

## 🚀 Next Steps

- Explore [Essential Tools](api/core_api.md)  
- Dive into [Advanced Utilities](api/advanced_api.md)  
- Check out [example notebooks](https://github.com/wins-wang/GridMaster/blob/main/demo_usage.ipynb)

---

**Happy modeling! 🎉**
