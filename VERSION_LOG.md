# 📘 GridMaster Version History

---
## v0.5.3 - 2025-05-04

### ✨ New Fixes & Improvements

**Full Parameter Recovery in Best Summary**  
- Fixed an issue where non-numeric parameters (like `'clf__penalty'` in Logistic Regression) selected during coarse search were missing from the final `summary['best_params']` when fine search only refined numeric parameters.  
- Now merges coarse stage’s best parameters with fine stage’s, ensuring the reported best parameter set reflects all crucial hyperparameter decisions.

**Improved Reporting Accuracy**  
- The `generate_search_report()` output now correctly lists all key-value pairs for the best model per classifier, matching the merged parameter set.



## v0.5.2 – 2025-05-04

### ✨ Improvements

- **Search Report Fixes**
    - `generate_search_report()` now consistently uses the `summary` dictionary’s **global best parameters** and **best cross-validation score** instead of relying on local `best_model.best_params_`, ensuring:
        - ✅ Full and correct parameter display in each model’s conclusion section.
        - ✅ Accurate overall best model selection across all models for summary comparison.

## v0.5.1 – 2025-05-04

### 🚀 New Features
- **Robust model export & import system**
  - Enhanced `export_all_models()`:
    - Added `use_timestamp` flag to avoid overwriting export folders.
    - Creates separate timestamped subfolders per model.
    - Wraps each model export in try-except to avoid breaking full export on single failure.
  - Enhanced `import_all_models()`:
    - Automatically detects whether subfolder names include timestamps and restores original model names.
    - Gracefully handles missing directories, providing clear warnings.
    - Skips individual models with issues without interrupting the full import process.

### Bug Fixes
- Fixed `fine_search()` `NoneType` error when `custom_fine_params` is not provided.
- Corrected multi-stage search numeric key detection to skip non-numeric (e.g., string) parameters.
- Added missing metric imports (`precision_score`, `recall_score`, etc.) to prevent `compare_best_models` crashes.
- Fixed incorrect path handling in `import_all_models()` when model files were not found.

### Other Improvements
- Improved internal parameter checks to avoid passing unsupported parameters (e.g., `max_depth` to `LogisticRegression`).

## v0.5.0 - 2025-05-04
✨ New Features
	•	Smart Mode Selection
	•	Introduced mode='fast' (default) and mode='industrial' options in build_model_config() and GridMaster.
	•	Automatically adjusts hyperparameter grids for different application scenarios.
	•	Fast Mode: Lightweight, minimal combinations, focused on core parameters (for fast prototyping, teaching, small experiments).
	•	Industrial Mode: Extended grid with more parameters, finer control, and better optimization for large-scale or production use.
	•	Solver Auto-Selector (Logistic Regression)
	•	Automatically switches to saga solver with elasticnet support for large datasets or industrial mode.
	•	Keeps liblinear for small datasets and fast mode to ensure compatibility and speed.


🛠 Improvements
	•	Coarse search parameter grids updated:
	•	XGBoost, LightGBM, CatBoost: Added or expanded important hyperparameters (subsample, num_leaves, l2_leaf_reg) under industrial mode.
	•	RandomForest: Adjusted n_estimators and max_depth ranges for better scaling.
	•	Cleaner separation between fast and industrial mode configurations.
	•	Enhanced consistency across models, aligning design choices to real-world use cases.


🐛 Fixes
	•	Prevented fine search errors on non-numeric parameters (e.g., clf__penalty) by skipping them in automatic fine grid generation.
	•	Improved stability when working with limited coarse grids or narrow parameter ranges.


## v0.3.3 - 2025-05-04
🛠 Fixed multi_stage_search() setting

## v0.3.2 - 2025-05-03
✨ Updates
	•	[multi_stage_search]
	•	Updated multi_stage_search() to default to searching all initialized models when no model_name is specified.
		Users can still explicitly pass a subset of model names if they want to tune only specific models.
	•	[Documentation Improvements]
	•	Added a prerequisite warning in the fine_search() section to clarify that it requires prior execution of coarse_search().
	•	Enhanced the Quick Start guide to highlight the difference in scope:
	•	coarse_search() and fine_search() always run across all initialized models.
	•	multi_stage_search() allows targeting specific models or defaulting to all.

## v0.3.1 – 2025-05-02 Bugfixes & Usability Enhancements
	•	Fixed stage naming in reports (removed raw stage1 suffix).
	•	Improved fallback when fine grids are too narrow or degenerate.
	•	Enhanced user tips and warning messages for clearer guidance.
	•	Report generation now skips models without valid search results.
	•	Polished docstrings, outputs, and internal messages for consistency.

## v0.3.0 – 2025-05-02 Smart Fine-Tuning & Multi-Stage Upgrade
	•	Added smart, expert, custom modes to fine_search and multi_stage_search.
	•	Implemented _identify_important_params for automatic selection of key tuning parameters.
	•	Enhanced auto_generate_fine_grid() to handle parameter constraints and fallback safely.
	•	Added custom_estimator_params interface for injecting model-specific (e.g., GPU) arguments.
	•	Improved logging with clear stage names, icons (🔍, 🔧), and better readability.
	•	Enhanced final report: now summarizes stage-by-stage best parameters and overall best model.

## v0.2.2 – 2025-05-02 Advanced Parameters Added
	•	Exposed n_jobs, verbose, refit, return_train_score as configurable arguments.
	•	Passed these into all GridSearchCV calls to enable advanced parallelism and control.
	•	Updated docstrings for clarity.
	•	Fully backward-compatible.

## v0.2.1 - 2025-05-01

✨ New
	•	Added a comprehensive MkDocs-powered user manual with quickstart guide, API reference, and advanced utilities.
	•	Integrated ReadTheDocs deployment for online documentation access.

🛠 Fixed
	•	Updated contact email in setup.py and all documentation files to replace the old GitHub noreply address.
	•	Minor corrections to documentation formatting and navigation.


## v0.2.0 – 2025-05-01

✨ **Functionality Enhancements**
- Improved `auto_generate_fine_grid()` with support for log-scale parameters; parameters like `C`, `learning_rate`, etc. are now interpolated logarithmically where appropriate.
- Added fine-grained control over all plotting functions (`plot_cv_score_curve`, `plot_confusion_matrix`, etc.) via new optional arguments (e.g., `title_fontsize`, `label_fontsize`, `figsize`, etc.).

🎨 **Visualization Improvements**
- Optimized default font sizes for high-resolution (HiDPI/Retina) displays, ensuring clearer axis labels and titles.
- Introduced a global font control mechanism for plots, allowing users to easily standardize typography across all visualizations.

🧾 **Backward Compatibility**
- All new features are fully backward compatible and optional.

## v0.1.1 – 2025-04-30

- Added detailed docstrings and descriptions for all methods
- Improved `demo_usage.ipynb` with clearer outputs and warnings filtered
- Cleaned terminal output, removed redundant `utils.py`
- Refactored verbosity control for LightGBM, XGBoost, CatBoost
- Enhanced export functionality with timestamped folders
- Ready for second PyPI release

---

## v0.1.0 – 2025-04-29

- 📦 Initial release on PyPI
- Support for Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Multi-stage coarse-to-fine grid search
- Comparison, scoring, and export tools
