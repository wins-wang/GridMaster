# Tech Specs

### Default Coarse Search Parameter Grids (by Mode)

| Model            | Parameter                | `fast` Mode        | `industrial` Mode          |
| ---------------- | ------------------------ | ------------------ | -------------------------- |
| **Logistic**     | `clf__C`                 | [0.01, 0.1, 1, 10] | [0.01, 0.1, 1, 10]         |
|                  | `clf__penalty`           | ['l1', 'l2']       | ['l1', 'l2', 'elasticnet'] |
|                  | `clf__l1_ratio`          | _Not used_         | [0.1, 0.5, 0.9]            |
| **RandomForest** | `clf__n_estimators`      | [100, 200]         | [300, 500, 1000]           |
|                  | `clf__max_depth`         | [5, 10]            | [10, 20, None]             |
|                  | `clf__min_samples_split` | [2, 5, 10]         | [2, 5, 10]                 |
| **XGBoost**      | `clf__n_estimators`      | [100, 200]         | [300, 500, 1000]           |
|                  | `clf__max_depth`         | [3, 5]             | [3, 5, 7]                  |
|                  | `clf__learning_rate`     | [0.1, 0.2]         | [0.01, 0.05, 0.1]          |
|                  | `clf__subsample`         | _Not used_         | [0.6, 0.8, 1.0]            |
| **LightGBM**     | `clf__n_estimators`      | [100, 200]         | [300, 500, 1000]           |
|                  | `clf__max_depth`         | [5, 10]            | [10, 15, 20]               |
|                  | `clf__learning_rate`     | [0.1, 0.2]         | [0.01, 0.05, 0.1]          |
|                  | `clf__num_leaves`        | _Not used_         | [15, 31, 63]               |
| **CatBoost**     | `clf__iterations`        | [200, 300]         | [500, 1000]                |
|                  | `clf__depth`             | [4, 6]             | [4, 6, 8]                  |
|                  | `clf__learning_rate`     | [0.1, 0.2]         | [0.01, 0.05, 0.1]          |
|                  | `clf__l2_leaf_reg`       | _Not used_         | [1, 3, 5, 7]               |

---

✅ **`fast`** → lightweight grid for quick experimentation  
✅ **`industrial`** → larger, production-ready grid covering more parameter combinations

You can set this using the `mode` argument when calling `build_model_config()` or initializing `GridMaster`.

---

---

### **Default Estimator Settings and Recommended Use Cases**

| Model            | Random State | Special Defaults                                             | Recommended Use Cases                                        |
| ---------------- | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Logistic**     | 66           | `solver='liblinear'` if ≤10,000 samples or mode='fast'; `solver='saga'` with `penalty=['l1', 'l2', 'elasticnet']` if >10,000 samples or mode='industrial' | Best for small-to-medium datasets or when you need interpretable models; `saga` supports large-scale data and elasticnet but requires standardized inputs. |
| **RandomForest** | 66           | Uses sklearn `RandomForestClassifier`; adjusts trees and depth based on mode | Excellent general-purpose model, robust to overfitting, works well on tabular data with mixed feature types; fast mode for quick trials, industrial mode for robust tuning. |
| **XGBoost**      | 66           | `eval_metric='logloss'`, `use_label_encoder=False`, `verbosity=0`; optional GPU configs allowed via `custom_estimator_params` | Highly performant on structured data, handles missing values natively; recommended for competition-grade or production tasks; supports GPU for large-scale runs. |
| **LightGBM**     | 66           | `verbosity=-1`; optional GPU configs allowed via `custom_estimator_params` | Similar to XGBoost but faster on large datasets; works well with categorical features; recommended for fast iteration and industrial pipelines. |
| **CatBoost**     | 66           | `verbose=0`; optional GPU configs allowed via `custom_estimator_params` | Best when working with categorical data; often requires less parameter tuning out-of-the-box; GPU acceleration improves scalability on big data. |

---

---

### **Fine Search Modes: Parameter Details Table**

| **Model**           | **Smart Mode (Auto Top 2)**                                  | **Expert Mode (Pre-Selected)**                    | **Custom Mode (User-Defined)**                 |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------- | ---------------------------------------------- |
| Logistic Regression | Based on top test score variation — usually `clf__C`, `clf__penalty` | Always `clf__C`                                   | Whatever user provides in `custom_fine_params` |
| Random Forest       | Based on top test score variation — usually `clf__max_depth`, `clf__min_samples_split` | Always `clf__max_depth`, `clf__min_samples_split` | Whatever user provides in `custom_fine_params` |
| XGBoost             | Based on top test score variation — usually `clf__learning_rate`, `clf__max_depth` | Always `clf__learning_rate`, `clf__max_depth`     | Whatever user provides in `custom_fine_params` |
| LightGBM            | Based on top test score variation — usually `clf__learning_rate`, `clf__max_depth` | Always `clf__learning_rate`, `clf__max_depth`     | Whatever user provides in `custom_fine_params` |
| CatBoost            | Based on top test score variation — usually `clf__learning_rate`, `clf__depth` | Always `clf__learning_rate`, `clf__depth`         | Whatever user provides in `custom_fine_params` |

---

#### **Mode Selection Key Points**

- **Smart Mode** dynamically detects which parameters matter most for each dataset — great for flexible, adaptive tuning.
- **Expert Mode** sticks to proven influential parameters, reducing grid size and focusing search.
- **Custom Mode** gives you complete freedom but requires you to define a meaningful and valid parameter grid yourself.

---

---

### Parallelization Strategy

By default, **GridMaster** uses half of the detected CPU cores (`n_jobs`) to balance system load and optimization speed.

You can override this by setting:

- `n_jobs = -1` → use **all** available cores.

- `n_jobs = <int>` → explicitly set the number of parallel jobs.

**Tip**: On shared or production servers, test carefully before using full CPU to avoid resource contention.

---

---

### GPU Acceleration Support

Supported for:

- **XGBoost** → via `tree_method='gpu_hist'`

- **LightGBM** → via `device='gpu'`

- **CatBoost** → via `task_type='GPU'`

These can be passed through the `custom_estimator_params` argument.

**Important**: Requires proper GPU drivers and library installations; otherwise, the system may silently fall back to CPU without warnings.

---

---

### Pipeline Preprocessing Details

\**Logistic Regression**:

- Always uses a `StandardScaler` for feature normalization.

\**Tree-based models** (Random Forest, XGBoost, LightGBM, CatBoost):

- Use `'passthrough'` because they are scale-invariant and don’t require normalization.

---

---

### Evaluation Metric Defaults

**Logistic Regression**, **Random Forest**:

- Follow sklearn’s default scoring (`accuracy`).

**XGBoost**:

- Uses `eval_metric='logloss'` by default.

**LightGBM**, **CatBoost**:

- Rely on their own internal defaults unless overridden.