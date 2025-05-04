# Class `GridMaster`
---
This class manages the end-to-end multi-model hyperparameter search, evaluation, and export pipeline.

---

## **Initialization & Setup**
---
### Method **`.init()`**

Initialize the GridMaster with specified models and training data.

This constructor sets up the internal model configuration for each model using the provided training dataset and optional custom hyperparameters.

---

#### **Args**

| Parameter        | Type                   | Description                                                                                                                                      | Default |
|------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `models`        | list                   | A list of model names (e.g., `'logistic'`, `'random_forest'`, `'xgboost'`).                                                                      | –       |
| `X_train`       | array-like or DataFrame | Training features.                                                                                                                               | –       |
| `y_train`       | array-like             | Training labels.                                                                                                                                 | –       |
| `custom_params` | dict, optional         | Dictionary of custom coarse-level hyperparameters for specific models. Format: `{model_name: param_dict}`.                     | `None`  |
| `custom_estimator_params` | dict, optional | Dictionary of custom estimator (model) initialization parameters. Format: `{model_name: param_dict}` Useful for enabling options like GPU. | `None` |
| `njobs` | int, optional | Number of parallel jobs for GridSearchCV. Defaults to **half of the total detected CPU cores** (based on system hardware). Use -1 to utilize all CPU cores. | **half of total CPU cores** |
| `verbose` | int, optional | Verbosity level for GridSearchCV. Controls how much logging is printed. <br />Recommendation: <br />- Beginner or clean run`0`  to suppress all messages.<br />- Medium-scale tasks where you want to monitor progress use `1` , especially for: Large grid searches where you want to know it’s still active, or Getting a rough idea of where the search is in its progress<br />- Advanced users or debugging use `2` or higher to see detailed cross-validation steps and logs from the estimators themselves. | `1` |
| `refit` | bool, optional | Whether to refit the best estimator on the entire dataset after search. | `True` |
| `return_train_score` | bool, optional | Whether to include training set scores in cv_results_. | `False` |

---

#### **Attributes**

| Attribute          | Type         | Description                                                        |
|--------------------|--------------|--------------------------------------------------------------------|
| `model_dict`       | dict         | Dictionary storing initialized models and their search spaces.      |
| `X_train`         | array-like   | Feature training set.                                              |
| `y_train`         | array-like   | Label training set.                                                |
| `results`         | dict         | Stores search results for each model.                              |
| `best_model_name` | str          | Name of the currently best-performing model.                       |
| `feature_names`   | list         | List of feature names for plotting and explanation.                |
| `njobs` | int | Number of parallel jobs for GridSearchCV. |
| `verbose` | int | Verbosity level for GridSearchCV. |
| `refit` | bool | Whether to refit the best estimator after grid search. |
| `return_train_score` | bool | Whether training scores are included in cv_results_. |

---

⚠️ **Warning:**  
This class sets up internal state; be cautious when modifying `results` or `model_dict` manually.

---
## **Coarse & Fine Hyperparameter Search**
---

### Method **`.coarse_search()`**

Perform coarse-level hyperparameter grid search across all models.

This method iterates through all configured models and performs GridSearchCV using their predefined coarse parameter grids.

---

#### **Args**

| Parameter | Type          | Description                                                                                                                                     | Default      |
| --------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `scoring` | str, optional | Evaluation metric to optimize. Must be one of `'accuracy'`, `'f1'`, `'roc_auc'`, `'precision'`, `'recall'`, or any valid sklearn scorer string. | `'accuracy'` |
| `cv`      | int           | Number of cross-validation folds.                                                                                                               | `5`          |

---

#### **Returns**

`None`: Updates the `results` dictionary with fitted GridSearchCV objects under the `'coarse'` key.

---

⚠️ **Warning:**  
This method modifies the **`results`** dictionary in-place.

---

#### Notes:

This method internally uses the following advanced GridSearchCV parameters, set during GridMaster initialization:

-  `n_jobs`: Number of parallel jobs. Defaults to None (single-threaded). Use `-1` for all CPU cores.
- `verbose`: Verbosity level. Controls logging detail.
- `refit`: Whether to refit the best model on the entire dataset after search.
- `return_train_score`: Whether to include training set scores in the results.

---

#### **Example**

<pre>
```python
gm = GridMaster()
gm.coarse_search(scoring='f1', cv=5)
```
</pre>

---

---
### Method **`.fine_search()`**

❗**Prerequisite:**  
_Before using `.fine_search()`, you **must first run** `.coarse_search()` to generate the required coarse search results._

Performs fine-level hyperparameter tuning based on coarse search results.

This method refines the hyperparameter grid by auto-generating a narrower search space around the best parameters from the coarse search and runs another GridSearchCV.  
It now supports **smart**, **expert**, or **custom** fine-tuning modes.

---

#### **Args**

| Parameter            | Type            | Description                                                  | Default      |
| -------------------- | --------------- | ------------------------------------------------------------ | ------------ |
| `scoring`            | str, optional   | Scoring metric to optimize. Must be one of `'accuracy'`, `'f1'`, `'roc_auc'`, `'precision'`, `'recall'`, or any valid sklearn scorer string. | `'accuracy'` |
| `cv`                 | int, optional   | Number of cross-validation folds.                            | 5            |
| `auto_scale`         | float, optional | Scaling factor for narrowing the search range (e.g., `0.5` = ±50% around the best value). Applies to both linear and log-scale parameters. | 0.5          |
| `auto_steps`         | int, optional   | Number of steps/grid points per parameter in fine grid.      | 5            |
| `search_mode`        | str, optional   | Fine-tuning mode. Choose from `'smart'` (auto-select most important params), `'expert'` (only adjusts learning rate, max depth), or `'custom'` (use provided `custom_fine_params`). | `'smart'`    |
| `custom_fine_params` | dict, optional  | Custom fine-tuning grid when `search_mode='custom'`. If provided, it overrides auto-generated grids. | None         |

---

#### **Modes (smart / expert / custom)**

##### **`smart` mode**

- **Default behavior**: Automatically selects the top-2 most impactful hyperparameters from coarse search results.

- **How it works**:

  1. Extracts GridSearchCV.cv_results_ and looks at how each hyperparameter’s different values affect mean_test_score.

  2. Calculates the performance variation range (max - min average score) for each parameter.

  3. Selects the top-N (default 2) parameters with the largest variation as the focus of fine-tuning.

- **Best for**: Users who want automated, data-driven refinement without needing to pre-select key parameters.

- **Caution**: If the coarse grid was too narrow or the data is insensitive to certain parameters, the generated fine grid may collapse to a single combination and skip fine-tuning.

##### **`expert` mode**

- **Default behavior**: Focuses on well-known, domain-recommended hyperparameters that typically have strong influence.
- **Current configuration**:
  - LogisticRegression: `'clf__C'`
  - RandomForest: `'clf__max_depth'`,` 'clf__min_samples_split'`
  - XGBoost, LightGBM, CatBoost: `'clf__learning_rate'`, `'clf__max_depth'`
- **Best for**: Users who trust established best practices and want to emphasize proven sensitive parameters.
- **Caution**: This mode is not data-adaptive; it may overlook dataset-specific influences that fall outside the “usual suspects.”

##### **`custom` mode**

- **Default behavior**: Uses user-supplied custom_fine_params to directly define the fine-tuning grid.
- **Best for**: Advanced users who want full control or have domain-specific knowledge about optimal parameter ranges.
- **Caution**: You are responsible for ensuring the parameter grid is meaningful; an overly narrow grid may result in only one combination and skip fine-tuning.

---

#### **Returns**

`None`: Updates the `results` dictionary with the fine-tuned GridSearchCV objects under the `'fine'` key.

---

⚠️ **Warning:**  
This method modifies the **`results`** dictionary in-place.

---

#### **Notes**

This method internally uses the following advanced GridSearchCV parameters, set during GridMaster initialization:

-  `n_jobs`: Number of parallel jobs. Defaults to half of CPU cores (detected at runtime). Use `-1` for all cores.
-  `verbose`: Verbosity level. Controls logging detail.
-  `refit`: Whether to refit the best model on the entire dataset after search.
-  `return_train_score`: Whether to include training set scores in the results.

---

#### **Example**

```python
# Step 1: Run coarse search first
gm.coarse_search(scoring='accuracy', cv=5)

# Step 2: Fine-tune using smart mode
gm.fine_search(
    scoring='roc_auc',
    cv=5,
    auto_scale=0.3,
    auto_steps=7,
    search_mode='smart'
)

# Or using expert mode:
gm.fine_search(
    scoring='accuracy',
    cv=3,
    search_mode='expert'
)

# Or using custom mode:
custom_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7]
}
gm.fine_search(
    scoring='f1',
    cv=4,
    search_mode='custom',
    custom_fine_params=custom_grid
)
```


---

---
### Method **`.multi_stage_search()`**

Perform a multi-stage grid search consisting of one coarse and multiple fine-tuning stages.

This method first performs a coarse search (if not already done), then iteratively refines the hyperparameter space using a list of `(scale, steps)` tuples.  
You can now choose **smart**, **expert**, or **custom** mode for each fine stage.

---

#### **Args**

| Parameter            | Type                    | Description                                                  | Default                |
| -------------------- | ----------------------- | ------------------------------------------------------------ | ---------------------- |
| `model_name`         | str                     | Name of the model to search (must be present in `model_dict`). Defaults to  to None, which means use all models from initialization. | `None`                 |
| `cv`                 | int, optional           | Number of cross-validation folds.                            | 5                      |
| `scoring`            | str, optional           | Scoring metric to optimize. Must be one of `'accuracy'`, `'f1'`, `'roc_auc'`, `'precision'`, `'recall'`, or any valid sklearn scorer string. | `'accuracy'`           |
| `stages`             | list of tuple, optional | List of `(scale, steps)` for each fine-tuning stage. Example: `[(0.5, 5), (0.2, 5)]` means two rounds: ±50% grid with 5 points, then ±20% with 5 points. | `[(0.5, 5), (0.2, 5)]` |
| `search_mode`        | str, optional           | Fine-tuning mode for all fine stages. Choose from `'smart'`, `'expert'`, or `'custom'`. | `'smart'`              |
| `custom_fine_params` | dict, optional          | Custom fine-tuning grid if `search_mode='custom'`.           | None                   |
| `verbose`            | bool, optional          | Whether to print progress messages.                          | True                   |

---

#### **Modes (smart / expert / custom)**

##### **`smart` mode**

- **Default behavior**: Automatically selects the top-2 most impactful hyperparameters from coarse search results.

- **How it works**:

  1. Extracts GridSearchCV.cv_results_ and looks at how each hyperparameter’s different values affect mean_test_score.

  2. Calculates the performance variation range (max - min average score) for each parameter.

  3. Selects the top-N (default 2) parameters with the largest variation as the focus of fine-tuning.

- **Best for**: Users who want automated, data-driven refinement without needing to pre-select key parameters.

- **Caution**: If the coarse grid was too narrow or the data is insensitive to certain parameters, the generated fine grid may collapse to a single combination and skip fine-tuning.

##### **`expert` mode**

- **Default behavior**: Focuses on well-known, domain-recommended hyperparameters that typically have strong influence.
- **Current configuration**:
  - LogisticRegression: `'clf__C'`
  - RandomForest: `'clf__max_depth'`,` 'clf__min_samples_split'`
  - XGBoost, LightGBM, CatBoost: `'clf__learning_rate'`, `'clf__max_depth'`
- **Best for**: Users who trust established best practices and want to emphasize proven sensitive parameters.
- **Caution**: This mode is not data-adaptive; it may overlook dataset-specific influences that fall outside the “usual suspects.”

##### **`custom` mode**

- **Default behavior**: Uses user-supplied custom_fine_params to directly define the fine-tuning grid.
- **Best for**: Advanced users who want full control or have domain-specific knowledge about optimal parameter ranges.
- **Caution**: You are responsible for ensuring the parameter grid is meaningful; an overly narrow grid may result in only one combination and skip fine-tuning.

---

#### **Returns**

`None`: Updates the `results` dictionary with intermediate GridSearchCV results for each stage.

---

⚠️ **Warning:**  
This method modifies the **`results`** dictionary in-place.

---

#### **Notes**

This method internally uses the following advanced GridSearchCV parameters, set during GridMaster initialization:

-  `n_jobs`: Number of parallel jobs. Defaults to half of CPU cores (detected at runtime). Use `-1` for all cores.
-  `verbose`: Verbosity level. Controls logging detail.
-  `refit`: Whether to refit the best model on the entire dataset after search.
-  `return_train_score`: Whether to include training set scores in the results.

---

#### **Example**

```python
gm.multi_stage_search(
    model_name='xgboost',
    scoring='accuracy',
    cv=3,
    stages=[(0.5, 5), (0.2, 5)],
    search_mode='smart'
)

# Or using expert mode:
gm.multi_stage_search(
    model_name='lightgbm',
    scoring='roc_auc',
    search_mode='expert'
)

# Or using custom mode:
custom_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7]
}
gm.multi_stage_search(
    model_name='catboost',
    scoring='f1',
    search_mode='custom',
    custom_fine_params=custom_grid
)
```




---

## **Model Evaluation & Summary**

---
### Method **`.compare_best_models()`**

Compare all trained models on test data using specified evaluation metrics.

This method selects the best estimator for each model, computes scores on the provided test set, and stores results for later access.

---

#### **Args**

| Parameter    | Type                | Description                                                                                                   | Default                         |
|--------------|---------------------|---------------------------------------------------------------------------------------------------------------|---------------------------------|
| `X_test`    | array-like          | Feature test set.                                                                                            | —                               |
| `y_test`    | array-like          | Ground truth labels for the test set.                                                                        | —                               |
| `metrics`   | list of str, optional| Evaluation metrics to compute. Valid values: `'accuracy'`, `'f1'`, `'roc_auc'`, `'precision'`, `'recall'`.    | `['accuracy', 'f1', 'roc_auc']`|
| `strategy`  | str, optional       | Placeholder for future ranking strategies (currently unused).                                                 | `'rank_sum'`                   |
| `weights`   | dict, optional      | Placeholder for future weighted metric strategies (currently unused).                                         | `None`                         |

---
#### **Returns**

`None`: Updates `results` with `'test_scores'` and `'best_model'` for each model.

---

 ⚠️ **Warning:**  
 This method modifies the **`results`** dictionary in-place.

#### **Example**

<pre>
```python
gm.compare_best_models(X_test, y_test, metrics=['accuracy', 'f1'])
```
</pre>

---

---
### Method **`.get_best_model_summary()`**

Retrieve a summary of the best model's configuration and performance.

This includes the best estimator, parameters, cross-validation score, and test set scores if available.

---

#### **Args**

| Parameter     | Type    | Description                                                                                                        | Default  |
|---------------|---------|--------------------------------------------------------------------------------------------------------------------|----------|
| `model_name` | str, optional | Name of the model to summarize. If `None`, uses the current `best_model_name` set by the user or internal logic. | `None`   |

---

#### **Returns**

`dict`: A dictionary with the following keys:  
- `'model_name'` (str): Name of the model.  
- `'best_estimator'` (sklearn.BaseEstimator): Best estimator object.  
- `'best_params'` (dict): Best hyperparameters from search.  
- `'cv_best_score'` (float): Best cross-validation score.  
- `'test_scores'` (dict): Optional test metrics if available.

---

#### **Example**

<pre>
```python
summary = gm.get_best_model_summary('logistic')
print(summary['best_params'])
```
</pre>

---

---
### Method `.generate_search_report()`

Generate a detailed multi-stage search report across all models, summarizing parameter grids, best parameter sets, and best metric scores.

---

#### Returns:
​            str: A formatted multi-line text report summarizing the entire search process.

---

**Example**

<pre>
```python
gm.generate_search_report() 
```
</pre>



---

---

### Method **`.get_cv_results()`**

Retrieve cross-validation results from GridSearchCV for a specific model.

This method allows you to extract detailed performance metrics from previous coarse or fine searches.

---

#### **Args**

| Parameter      | Type           | Description                                                                                              | Default |
| -------------- | -------------- | -------------------------------------------------------------------------------------------------------- | ------- |
| `model_name`   | str            | Name of the model to retrieve results for.                                                               | —       |
| `use_fine`     | bool, optional | Whether to retrieve results from fine-tuned search (`final`) or coarse-level search.                     | True    |
| `as_dataframe` | bool, optional | If `True`, returns results as a pandas DataFrame; if `False`, returns the raw `cv_results_` dictionary.  | True    |

---

#### **Returns**

`Union[pd.DataFrame, dict]`: Cross-validation results in the selected format.

---

#### **Example**

<pre>
```python
cv_results = gm.get_cv_results('xgboost', use_fine=True, as_dataframe=True)
print(cv_results.head())
```
</pre>

---
## **Visualization & Plotting**


---
### Method **`.plot_cv_score_curve()`**

Plot cross-validation scores for each parameter set tried during grid search.

This visualization helps analyze how performance varies across different hyperparameter combinations.

---

#### **Args**

| Parameter         | Type            | Description                                                                                                                                   | Default                 |
| ----------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `model_name`      | str             | Name of the model whose results will be plotted.                                                                                              | —                       |
| `metric`          | str, optional   | Score metric to plot. Must be one of `'mean_test_score'`, `'mean_train_score'`, `'std_test_score'`, `'std_train_score'`, `'rank_test_score'`. | `'mean_test_score'`     |
| `plot_train`      | bool, optional  | Whether to include training scores in the plot.                                                                                               | True                    |
| `figsize`         | tuple, optional | Size of the plot in inches (width, height).                                                                                                   | `(10, 5)`               |
| `show_best_point` | bool, optional  | Whether to mark the best score point on the plot.                                                                                             | True                    |
| `title`           | str, optional   | Custom plot title. If `None`, a default title will be used.                                                                                   | `None`                  |
| `xlabel`          | str, optional   | Label for x-axis.                                                                                                                             | `'Parameter Set Index'` |
| `ylabel`          | str, optional   | Label for y-axis. If `None`, uses the `metric`.                                                                                               | `None`                  |
| `save_path`       | str, optional   | If provided, saves the plot to the specified file path.                                                                                       | `None`                  |

---

#### **Returns**

`None`: Displays and optionally saves a matplotlib plot.

---

#### **Example**

<pre>
```python
gm.plot_cv_score_curve('xgboost', metric='mean_test_score')
```
</pre>

---

---
### Method **`.plot_confusion_matrix()`**

Plot the confusion matrix for a classification model on test data.

This method visualizes the true vs. predicted labels to evaluate model performance.

---
|Parameter|Type|Description|Default|
|---|---|---|---|
|`model_name`|str|Name of the model to use for prediction.|—|
|`X_test`|array-like|Test feature set.|—|
|`y_test`|array-like|True labels for the test set.|—|
|`labels`|list, optional|List of label names to display in the matrix.|`None`|
|`normalize`|str or None, optional|Normalization mode: `'true'` (row-wise), `'pred'` (column-wise), `'all'` (overall), or `None` (no normalization). See [sklearn confusion_matrix docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) for details.|`None`|
|`figsize`|tuple, optional|Size of the figure in inches.|`(6, 5)`|
|`cmap`|str or Colormap, optional|Colormap used for the plot. Must be a valid name or Colormap object from [Matplotlib colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html), e.g., `'Blues'`, `'viridis'`, `'plasma'`.|`'Blues'`|
|`title`|str, optional|Title of the plot. If `None`, a default title will be used.|`None`|
|`save_path`|str, optional|If specified, saves the figure to this file path instead of displaying it.|`None`|

---

#### **Returns**

`None`: Displays the confusion matrix plot.

---

#### **Example**

<pre>
```python
gm.plot_confusion_matrix('logistic', X_test, y_test, normalize='true')
```
</pre>

---

---
### Method **`.plot_model_coefficients()`**

Plot the top N coefficients of a linear model for interpretability.

This method visualizes the most important positive or negative coefficients from models like logistic regression.

---

#### **Args**

| Parameter         | Type               | Description                                                                                                                                                                   | Default      |
|-------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| `model_name`      | str                | Name of the model whose coefficients will be plotted.                                                                                                                       | —            |
| `top_n`          | int, optional      | Number of most important features to display.                                                                                                                                | 20           |
| `sort_descending`| bool, optional     | Whether to sort by absolute value in descending order.                                                                                                                       | True         |
| `figsize`        | tuple, optional    | Figure size in inches.                                                                                                                                                       | `(10, 5)`    |
| `color`          | str, optional      | Bar color. Must be a valid color name or hex code as accepted by [Matplotlib colors](https://matplotlib.org/stable/users/explain/colors/), e.g., `'teal'`, `'red'`, `'#1f77b4'`. | `'teal'`     |
| `title`          | str, optional      | Plot title. If `None`, uses default.                                                                                                                                        | `None`       |
| `xlabel`         | str, optional      | Label for x-axis.                                                                                                                                                           | `'Coefficient Value'` |
| `save_path`      | str, optional      | If provided, saves the plot to this file path.                                                                                                                               | `None`       |

---

#### **Returns**

`None`: Displays and optionally saves a matplotlib bar chart of model coefficients.

---

#### **Example**

<pre>
```python
gm.plot_model_coefficients('logistic', top_n=15, color='purple')
```
</pre>

---

---
### Method **`.plot_feature_importance()`**

Plot the top N feature importances from a tree-based model.

This method shows which features contributed most to predictions, based on models like random forest, XGBoost, or LightGBM.

---

#### **Args**

| Parameter         | Type               | Description                                                                                                                                                                   | Default        |
|-------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| `model_name`      | str                | Name of the model to visualize.                                                                                                                                              | —              |
| `top_n`          | int, optional      | Number of top features to display.                                                                                                                                           | 20             |
| `sort_descending`| bool, optional     | Whether to sort by importance in descending order.                                                                                                                           | True           |
| `figsize`        | tuple, optional    | Figure size in inches.                                                                                                                                                       | `(10, 5)`      |
| `color`          | str, optional      | Bar color. Must be a valid color name or hex code as accepted by [Matplotlib colors](https://matplotlib.org/stable/users/explain/colors/), e.g., `'darkgreen'`, `'red'`, `'#1f77b4'`. | `'darkgreen'`  |
| `title`          | str, optional      | Plot title. If `None`, a default will be used.                                                                                                                               | `None`         |
| `xlabel`         | str, optional      | X-axis label.                                                                                                                                                               | `'Feature Importance'` |
| `save_path`      | str, optional      | If specified, saves the plot to this path.                                                                                                                                   | `None`         |

---

#### **Returns**

`None`: Displays and optionally saves a matplotlib plot.

---

#### **Example**

<pre>
```python
gm.plot_feature_importance('xgboost', top_n=15, color='orange')
```
</pre>

---
## **Import & Export**
---

### Method **`.export_model_package()`**

Export the best model, its summary, and cross-validation results to disk.

This function creates a dedicated subdirectory under the specified folder, containing:
- The final fitted model (`model_final.joblib`)
- A JSON summary of model performance and parameters (`best_model_summary.json`)
- CSV files of cross-validation results for all search stages (e.g., coarse, fine)

---

#### **Args**

| Parameter       | Type               | Description                                                                                                                               | Default               |
|-----------------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| `model_name`    | str                | Name of the model to export.                                                                                                               | —                     |
| `folder_path`   | str, optional      | Base folder to store exported model files. If the folder does not exist, it will be created.                                               | `'model_exports'`     |

---

#### **Returns**

`None`: Writes files to disk, no return value.


---

#### **Example**

<pre>
```python
gm.export_model_package('logistic', folder_path='exports')
```
</pre>

---


---

### Method **`.export_all_models()`**

Export all models and their associated results to disk.

This function loops through all models stored in `self.results` and calls `.export_model_package()` on each of them.  
Optionally, it appends a timestamp to each model’s output directory to avoid overwriting previous exports.

---

#### **Args**

| Parameter        | Type               | Description                                                                                                                      | Default             |
|------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `folder_path`    | str, optional      | Base folder where all model subfolders will be exported. If the folder does not exist, it will be created.                       | `'model_exports'`   |
| `use_timestamp`  | bool, optional     | If `True`, appends a timestamp to each model folder name (useful for versioning and avoiding overwrite).                         | True                |

---

#### **Returns**

`None`: Writes all models and their associated files to disk, no return value.


---

#### **Example**

<pre>
```python
gm.export_all_models(folder_path='exports', use_timestamp=True)
```
</pre>

---

---
### Method **`.load_model_package()`**

Load a saved model package from disk, including the estimator, summary, and cross-validation results.

This function reads:
- The saved model (`model_final.joblib`)
- The JSON summary (`best_model_summary.json`)
- The CSV files of cross-validation results

and loads them back into the `self.results` dictionary.

---

#### **Args**

| Parameter       | Type               | Description                                                                                                               | Default             |
|-----------------|--------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------|
| `model_name`    | str                | Name of the model to load.                                                                                                | —                   |
| `folder_path`   | str, optional      | Base path where the model files are stored. The method looks for a subfolder named after the model inside this directory. | `'model_exports'`   |

---

#### **Returns**

`None`: Updates `self.results` and sets `self.best_model_name`.

---

#### **Example**

<pre>
```python
gm.load_model_package('logistic', folder_path='exports')
```
</pre>

---

---
### Method **`.load_all_models()`**

Load all saved model packages from a specified folder.

This function iterates through all subdirectories under the given `folder_path`,  
assumes each contains a model export (created by `.export_model_package()`),  
and calls `.load_model_package()` to load them into `self.results`.

Model names are inferred from subdirectory names (before the first underscore if timestamped).

---

#### **Args**

| Parameter       | Type               | Description                                                                                                                       | Default             |
|-----------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `folder_path`   | str, optional      | Directory containing exported model subfolders. Each subfolder should follow the expected naming and file structure.              | `'model_exports'`   |

---

#### **Returns**

`None`: Updates `self.results` with loaded model data for all found models.

---

#### **Example**

<pre>
```python
gm.load_all_models(folder_path='exports')
```
</pre>
