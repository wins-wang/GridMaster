
# Advanced Usage
> ⚠️ **Note:**

> These functions are intended for advanced users who want to customize the internal behavior of GridMaster or integrate its components into their own pipelines.

---

### Advanced Utility **`.auto_generate_fine_grid()`**

Automatically generate a fine-grained hyperparameter grid  
around the best parameters from the coarse search.

This method intelligently determines whether a parameter  
should be scaled linearly or logarithmically,  
and creates a narrowed search space centered on the best-known value.

---

#### **Args**

| Parameter       | Type               | Description                                                                                                                                                    | Default    |
|-----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| `best_params`   | dict               | Best parameter values obtained from the coarse search.                                                                                                          | —          |
| `auto_scale`    | float, optional    | Scaling factor for narrowing the search range (e.g., `0.5` → ±50% around the best value).                                                                      | 0.5        |
| `auto_steps`    | int, optional      | Number of steps/grid points per parameter in the fine grid.                                                                                                    | 5          |

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

| Parameter      | Type    | Description                                                                                                       | Default  |
|----------------|---------|-----------------------------------------------------------------------------------------------------------------|----------|
| `model_name`   | str     | Name of the model to configure. Must be one of `'logistic'`, `'random_forest'`, `'xgboost'`, `'lightgbm'`, or `'catboost'`. | —        |

---

#### **Returns**

`dict`: A configuration dictionary containing:
- `'pipeline'`: A scikit-learn `Pipeline` object combining preprocessing and the model.
- `'param_grid'`: A coarse hyperparameter grid for initial search.

---

#### **Notes**

⚠️ **This is intended for advanced users or developers**  
who want to access and possibly customize the model configurations  
before passing them into the search functions.

---

#### **Example**

```python
config = build_model_config('logistic')
print(config['pipeline'])
print(config['param_grid'])
```

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
