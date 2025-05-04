
# Advanced Usage
> ⚠️ **Note:**

> These functions are intended for advanced users who want to customize the internal behavior of GridMaster or integrate its components into their own pipelines.

---

### Advanced Setting - **CPU Parallelism `(n_jobs)`**

GridMaster internally uses `GridSearchCV`, which supports parallel computation through the n_jobs parameter.

By **default**, GridMaster detects the number of CPU cores on your machine at runtime and uses **half of the available cores** to balance performance and system load.

- To maximize speed, set `n_jobs = -1` when initializing GridMaster (uses all CPU cores).
- To fully control parallelism, pass any integer value to `n_jobs` during GridMaster initialization.



⚠️ **Warning**:

Using all cores (`n_jobs = -1`) may **overload your system** if you run multiple processes in parallel. Make sure to monitor system load, especially on shared or production environments.

---

#### Example

```python
gm = GridMaster(
    models=['xgboost', 'lightgbm'],
    X_train=X_train,
    y_train=y_train,
    n_jobs=-1  # ← use all cpu cores
)
```

---

---

### Advanced Setting – Custom Estimator Parameters `(custom_estimator_params)`

GridMaster allows you to directly pass **custom initialization parameters** to underlying models (estimators) through the custom_estimator_params argument.



This is particularly useful if you want to:

✅ Enable GPU acceleration in models like XGBoost, LightGBM, or CatBoost

✅ Change default internal settings (e.g., booster, subsample, colsample_bytree)

✅ Fine-tune model-level behavior **before** the hyperparameter search even starts

---

#### Example

By **default**, GridMaster uses standard CPU-based estimators. To leverage GPU or other advanced options, pass a dictionary like this:

```python
gm = GridMaster(
    models=['xgboost', 'lightgbm', 'catboost'],
    X_train=X_train,
    y_train=y_train,
    custom_estimator_params={
        'xgboost': {'tree_method': 'gpu_hist'},
        'lightgbm': {'device': 'gpu'},
        'catboost': {'task_type': 'GPU'}
    }
)
```

------



⚠️ **Warning**:

To use GPU modes, ensure you have the **correct libraries installed** and your environment supports GPU.

For example:



- **XGBoost** must be compiled with GPU support.
- **LightGBM** needs the GPU-enabled version.
- **CatBoost** requires proper CUDA drivers.



Trying to enable GPU without the right setup may lead to **silent fallback to CPU** or runtime errors.



---

---

### Advanced Utility **`.auto_generate_fine_grid()`**

Automatically generate a fine-grained hyperparameter grid around the best parameters from the coarse search.

This method intelligently determines whether a parameter should be scaled linearly or logarithmically,  and creates a narrowed search space centered on the best-known value.

---

#### **Args**

| Parameter           | Type            | Description                                                  | Default |
| ------------------- | --------------- | ------------------------------------------------------------ | ------- |
| `best_params`       | dict            | Best parameter values obtained from the coarse search.       | —       |
| `scale`             | float, optional | Scaling factor for narrowing the search range (e.g., `0.5` → ±50% around the best value). | 0.5     |
| `steps`             | int, optional   | Number of steps/grid points per parameter in the fine grid.  | 5       |
| `keys`              | list, optional  | Specific parameter keys to include. If None, defaults to ['learning_rate']. Used in expert or smart fine-tuning modes to focus search on important parameters.<br />If None, defaults to ['learning_rate'] in expert mode, or uses smart selection. | `None`  |
| `coarse_param_grid` | Dict, optional  | Original coarse grid. Used to ensure parameters selected for fine-tuning had meaningful variation in the coarse stage. | `None`  |

---

#### **Returns**

`dict`: A refined hyperparameter grid dictionary suitable for fine-tuning.

---

#### **Details**

- For parameters commonly using log-scale (e.g., `'clf__C'`, `'clf__learning_rate'`),  
  this method applies multiplicative scaling.  
- For linear-scale parameters (e.g., `'clf__max_depth'`),  
  it applies additive scaling.

This ensures that grid search explores meaningful regions  
of the hyperparameter space without redundant or invalid values.

---

⚠️ **Important:**  

This function **only** refines numeric hyperparameters.

Categorical parameters — such as:

\- `'clf__penalty': ['l1', 'l2']` (Logistic Regression)

\- `'clf__booster': ['gbtree', 'dart']` (XGBoost)

are **not** included in fine grids  and remain fixed at their best coarse value during fine-tuning.

If you want to adjust these,  you must explicitly pass a `custom_fine_params` dictionary to override.

---

#### **Example**

```python
fine_grid = gm.auto_generate_fine_grid(best_params={'clf__C': 1.0}, auto_scale=0.5, auto_steps=5)
print(fine_grid)
```

---

---

### Advanced Utility **`.build_model_config()`**

Generate a default configuration dictionary for a specified model.

This internal utility returns the predefined pipeline and hyperparameter grid  
for supported models such as `'logistic'`, `'random_forest'`, `'xgboost'`, `'lightgbm'`, and `'catboost'`.

---

#### **Args**

| Parameter                 | Type           | Description                                                  | Default |
| ------------------------- | -------------- | ------------------------------------------------------------ | ------- |
| `model_name`              | str            | Name of the model to configure. Must be one of `'logistic'`, `'random_forest'`, `'xgboost'`, `'lightgbm'`, or `'catboost'`. | —       |
| `custom_coarse_params`    | dict, optional | User-defined hyperparameter grid to override default.        | `None`  |
| `custom_estimator_params` | dict, optional | Additional estimator-specific parameters (e.g., GPU settings, tree method) to inject into the model. | `None`  |

---

#### **Returns**

`dict`: A configuration dictionary containing:

- `'pipeline'`: A scikit-learn `Pipeline` object combining preprocessing and the model.
- `'param_grid'`: A coarse hyperparameter grid for initial search.

---

#### **Notes**

✨ **New in v0.5.x:**  

You can now pass a **`mode`** argument to select between:

\- `'fast'` → Lightweight, quick coarse search for experiments.

\- `'industrial'` → Comprehensive, production-grade coarse search.

⚠️ **This is intended for advanced users or developers**  
who want to access and possibly customize the model configurations before passing them into the search functions.

---

#### **Example**

```python
config = build_model_config('logistic')
print(config['pipeline'])
print(config['param_grid'])
```

---

---
### 🔍 **Fine-Tuning vs. Coarse Mode Clarification**

| Setting       | Where to Set                     | Controls                                                     |
| ------------- | -------------------------------- | ------------------------------------------------------------ |
| `mode`        | GridMaster initialization        | Size and complexity of the **coarse search grid** (`fast` vs. `industrial`). |
| `search_mode` | fine_search / multi_stage_search | Strategy used in **fine-tuning** (`smart`, `expert`, or `custom`) after coarse search. |

These settings work **independently** and can be combined to fine-tune both  
the **breadth** (coarse grid) and **depth** (fine search) of your hyperparameter search.

---

This distinction helps users understand which parameter impacts which phase,  
avoiding confusion when configuring multi-stage pipelines.

---

---

### Advanced Utility **`.set_plot_style()`**

Apply a consistent global plotting style across all GridMaster visualizations.


By default, this utility sets a coherent font family and font size scheme to ensure all plots look clean, balanced, and publication-ready.  

The underlying font size logic is explained in the **Details** section below.

---

#### **Args**

| Parameter       | Type          | Description                                                                                     | Default |
| --------------- | ------------- | ----------------------------------------------------------------------------------------------- | ------- |
| `base_fontsize` | int, optional | Base font size (in points) for all plot elements; controls titles, labels, ticks, legends, etc. | 14      |

---
#### **Details**

This function synchronizes the following `matplotlib.rcParams` settings based on `base_fontsize`:

| rcParam               | Computed Value                    | Default when `base_fontsize=14` |
|------------------------|----------------------------------|---------------------------------|
| `'font.size'`         | `base_fontsize`                  | 14                              |
| `'axes.titlesize'`    | `base_fontsize + 2`              | 16                              |
| `'axes.labelsize'`    | `base_fontsize`                  | 14                              |
| `'xtick.labelsize'`   | `base_fontsize - 2`              | 12                              |
| `'ytick.labelsize'`   | `base_fontsize - 2`              | 12                              |
| `'legend.fontsize'`   | `base_fontsize - 2`              | 12                              |

All these values increase or decrease together when you change `base_fontsize`.

---

#### **Returns**

`None`: Updates the global `matplotlib.pyplot.rcParams`.

---

#### **Notes**

⚠️ **Advanced use only:**  
In most cases, this function is called automatically by GridMaster plotting methods.  
You only need to call it manually if you want to apply the style before running your own custom plots.

---

#### **Example**

```python
from gridmaster.plot_utils import set_plot_style

# Set a larger base font size for all following plots
set_plot_style(base_fontsize=16)
```

---

---

### **Advanced Example Combining Both Modes**

This example shows how to combine:

✅ Coarse search mode selection (`mode='industrial'`)  

✅ Fine search strategy selection (`search_mode='smart'`)

```python
gm = GridMaster(

  models=['xgboost'],

  X_train=X_train,

  y_train=y_train,

  mode='industrial', # controls coarse grid size

  custom_estimator_params={'xgboost': {'tree_method': 'gpu_hist'}}

)



gm.multi_stage_search(

  search_mode='smart' # controls fine-tuning strategy

)
```


---



