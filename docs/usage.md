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
pip install --upgrade gridmaster
```

> ⚠️ Make sure your Python version is >= 3.8,  
> and that you have compatible versions of `scikit-learn`, `xgboost`, `lightgbm`, and `catboost` installed.
>
>  ⚠️ I recommend **always installing the latest version** to benefit from new features, bug fixes, and improvements.

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

# Initialize with at least one model
gm = GridMaster(models=['logistic', 'random_forest'], X_train = X_train, y_train = y_train)

gm.coarse_search()
gm.fine_search()
```

This will:

✅ First perform a **broad search** across a coarse grid of hyperparameters,  
✅ Then perform a **narrower, fine-grained search** around the best coarse parameters,  
✅ Efficiently balance exploration and exploitation for better-tuned models.

> *By default, GridMaster balances system load and speed by using **half of available CPU** cores for parallel search; advanced users can adjust this with the* `n_jobs` *parameter (see* [*Advanced Settings*](/en/main/api/advanced_api/#advanced-setting-cpu-parallelism-n_jobs)*).*
>
> _By default, `GridMaster()` initialization uses **fast mode**, applying a **small, quick grid** of hyperparameters designed for rapid exploration or lightweight machines. For professional or production use, you can pass `mode='industrial'` to start with a larger industrial-grade coarse grid. Please check [`__init__()`](/en/main/api/core_api/#method-init) and [Tech Spaces](/en/main/api/parameters/#default-coarse-search-parameter-grids-by-mode) for details._
>
> *By default, `fine_search` and `multi_stage_search` uses **smart mode**, automatically refining the top 2 impactful parameters based on coarse search performance variation. See [Modes](api/core_api.md#method-fine_search) for details.*
>
> *Advanced users can also pass custom **GPU-related estimator parameters** (e.g., `tree_method='gpu_hist'` for XGBoost) through the* `custom_estimator_params` *argument (see* [*Advanced Settings*](/en/main/api/advanced_api/#advanced-setting-custom-estimator-parameters-custom_estimator_params)).

---

#### 🛠️ Optional: Automate Multi-Stage Grid Search

If you want to automate the whole search pipeline with multiple refinement stages, use:

```python
gm.multi_stage_search(
    model_name='logistic',
    cv=5,
    scoring='accuracy',
    stages=[(0.5, 5), (0.2, 5)],
    search_mode='smart'  # default is 'smart'
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

⚠️ **Note:**  
Unlike `coarse_search()` and `fine_search()`, which always operate on **all initialized models** by default,  `multi_stage_search()` allows you to optionally specify a subset of models to tune by passing the `model_name` argument.  If no `model_name` is provided, it will also run on all models.


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

Or you can check details by generating the report:

```pyth
gm.generate_search_report()
```

Example output:

```text
For Logistic model:
Scoring metric used: 'accuracy'
Stage 1: Coarse grid search:
- clf__C in [0.01, 0.1, 1, 10]
- clf__penalty in ['l1', 'l2']
Total of 8 parameter combinations.
Best parameters: {'clf__C': 0.1, 'clf__penalty': 'l2'}

Stage 2: Fine grid search:
- clf__C in [0.05, 0.075, 0.1, 0.125, 0.15]
Total of 5 parameter combinations.
Best parameters: {'clf__C': 0.125}

Stage 3: Multi-stage fine grid search (Round 1):
- clf__C in [0.05, 0.075, 0.1, 0.125, 0.15]
Total of 5 parameter combinations.
Best parameters: {'clf__C': 0.125}

Stage 4: Multi-stage fine grid search (Round 2):
- clf__C in [0.0875, 0.10625, 0.125, 0.14375, 0.1625]
Total of 5 parameter combinations.
Best parameters: {'clf__C': 0.125}

Stage 5: Multi-stage fine grid search (Round 3):
- clf__C in [0.1125, 0.11875, 0.125, 0.13125, 0.1375]
Total of 5 parameter combinations.
Best parameters: {'clf__C': 0.11875}

✅ Conclusion: Best model for Logistic is {'clf__C': 0.11875} with best 'accuracy' score of 0.9842
------------------------------------------------------------

For Random_forest model:
Scoring metric used: 'accuracy'
Stage 1: Coarse grid search:
- clf__n_estimators in [100, 200]
- clf__max_depth in [5, 10]
- clf__min_samples_split in [2, 5, 10]
Total of 12 parameter combinations.
Best parameters: {'clf__max_depth': 10, 'clf__min_samples_split': 2, 'clf__n_estimators': 100}

Stage 2: Fine grid search:
- clf__n_estimators in [100, 150, 200]
- clf__max_depth in [5, 10, 15]
Total of 9 parameter combinations.
Best parameters: {'clf__max_depth': 10, 'clf__n_estimators': 150}

✅ Conclusion: Best model for Random_forest is {'clf__max_depth': 10, 'clf__n_estimators': 150} with best 'accuracy' score of 0.9649
------------------------------------------------------------

🌟 Summary:
The ultimate best model is Logistic with parameters {'clf__C': 0.11875} and best 'accuracy' score of 0.9842
```



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
- Found a bug or have a feature request?  Please open an issue at [GitHub Issues](https://github.com/wins-wang/GridMaster/issues).

---

**Happy modeling! 🎉**
