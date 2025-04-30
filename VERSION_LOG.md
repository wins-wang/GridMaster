# 📘 GridMaster Version History

---
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
