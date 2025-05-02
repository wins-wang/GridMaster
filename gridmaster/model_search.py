# gridmaster/model_search.py

# Core dependencies
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Scikit-learn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# External models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------------------------------------------
# GridMaster AutoML Utility for Multi-Stage Grid Search
# -------------------------------------------------------------

def auto_generate_fine_grid(best_params, scale=0.5, steps=5, keys=None, coarse_param_grid=None):
    """
    Automatically generate a refined hyperparameter grid around given best values.

    This function takes the best hyperparameters from a previous coarse search
    and generates a fine-tuned grid using linear or log-scale interpolation.
    It also applies parameter-specific constraints to ensure all values are valid.

    Args:
        best_params (dict): Dictionary of best parameters from a prior GridSearchCV result.
        scale (float, optional): Relative range to expand above and below each numeric value.
            Defaults to 0.5.
        steps (int, optional): Number of values to generate per parameter. Defaults to 5.
        keys (list, optional): Specific parameter keys to include. If None, defaults to ['learning_rate']. 
            Used in expert or smart fine-tuning modes to focus search on important parameters.
        coarse_param_grid (dict, optional): Original coarse grid.
            Used to ensure parameters selected for fine-tuning had meaningful variation in the coarse stage.

    Returns:
        dict: A refined parameter grid suitable for use in GridSearchCV.
    """
    fine_grid = {}

    log_scale_keywords = ['lr', 'learning_rate', 'alpha', 'lambda', 'reg', 'C', 'gamma']

    # Constraints for common classifier hyperparameters
    param_constraints = {
        'clf__C': lambda v: isinstance(v, float) and v > 0,
        'clf__max_depth': lambda v: v is None or (isinstance(v, int) and v > 0),
        'clf__min_samples_split': lambda v: (isinstance(v, int) and v >= 2) or (isinstance(v, float) and 0.0 < v <= 1.0),
        'clf__n_estimators': lambda v: isinstance(v, int) and v > 0,
        'clf__learning_rate': lambda v: isinstance(v, float) and 0.0 < v <= 1.0,
        'clf__iterations': lambda v: isinstance(v, int) and v > 0,
        'clf__depth': lambda v: isinstance(v, int) and v > 0,
    }

    # If keys are provided, only refine those; else, use all numeric
    params_to_refine = keys or [k for k in best_params if isinstance(best_params[k], (int, float))]

    for param in params_to_refine:
        value = best_params[param]

        # Check if coarse grid had more than one value
        coarse_values = coarse_param_grid.get(param, []) if coarse_param_grid else []
        if isinstance(coarse_values, list) and len(coarse_values) <= 1:
            continue  # skip if coarse grid didn't vary this param

        is_log_scale = any(kw in param.lower() for kw in log_scale_keywords)

        try:
            if is_log_scale:
                low = max(value * (1 - scale), 1e-8)
                high = value * (1 + scale)
                values = np.logspace(np.log10(low), np.log10(high), steps)
            else:
                low = max(0, value * (1 - scale))
                high = value * (1 + scale)
                values = np.linspace(low, high, steps)

            values = sorted(set(np.round(values, 8)))

            if param in param_constraints:
                values = [v for v in values if param_constraints[param](v)]

            if len(values) >= 2:
                fine_grid[param] = values
        except Exception:
            continue

    return fine_grid


def build_model_config(model_name, custom_coarse_params=None, custom_estimator_params=None):
    """
    Build a model configuration dictionary including pipeline, coarse hyperparameter grid,
    and optional custom estimator parameters.

    Args:
        model_name (str): Name of the model. Must be one of:
            - 'logistic'
            - 'random_forest'
            - 'xgboost'
            - 'lightgbm'
            - 'catboost'
        custom_coarse_params (dict, optional): User-defined hyperparameter grid to override default.
        custom_estimator_params (dict, optional): Additional estimator-specific parameters
            (e.g., GPU settings, tree method) to inject into the model.

    Returns:
        dict: A dictionary containing:
            - 'pipeline': sklearn Pipeline object with preprocessing and classifier
            - 'coarse_params': Coarse-level hyperparameter grid (dict)
    """
    model_name = model_name.lower()

    if model_name == 'logistic':
        estimator = LogisticRegression(solver='liblinear', random_state=42, **(custom_estimator_params or {}))
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', estimator)
        ])
        coarse = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1', 'l2']
        }

    elif model_name == 'random_forest':
        estimator = RandomForestClassifier(random_state=42, **(custom_estimator_params or {}))
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', estimator)
        ])
        coarse = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5, 10]
        }

    elif model_name == 'xgboost':
        estimator = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0, random_state=42,
                                  **(custom_estimator_params or {}))
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', estimator)
        ])
        coarse = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }

    elif model_name == 'lightgbm':
        estimator = LGBMClassifier(random_state=42, verbosity=-1, **(custom_estimator_params or {}))
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', estimator)
        ])
        coarse = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [5, 10, 15],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }

    elif model_name == 'catboost':
        estimator = CatBoostClassifier(verbose=0, random_state=42, **(custom_estimator_params or {}))
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', estimator)
        ])
        coarse = {
            'clf__iterations': [100, 200, 300],
            'clf__depth': [4, 6, 8],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return {
        'pipeline': pipeline,
        'coarse_params': custom_coarse_params or coarse
    }

from .gridmaster_class import GridMaster
