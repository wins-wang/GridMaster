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

def auto_generate_fine_grid(best_params, scale=0.5, steps=5):
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

    for param, value in best_params.items():
        if isinstance(value, (int, float)) and value > 0:
            is_log_scale = (
                any(kw in param.lower() for kw in log_scale_keywords) or
                (value >= 1e-3 and value <= 1e3 and np.log10(value) % 1 != 0)
            )
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
                # Apply param-specific constraints if defined
                if param in param_constraints:
                    values = [v for v in values if param_constraints[param](v)]
                if len(values) >= 2:
                    fine_grid[param] = values
            except Exception:
                continue
        else:
            # Non-numeric or unsupported: keep original as a single option
            fine_grid[param] = [value]

    return fine_grid

def build_model_config(model_name, custom_coarse_params=None):
    """
    Build a model configuration dictionary including pipeline and coarse hyperparameter grid.

    This function selects a model by name, constructs a preprocessing pipeline with a classifier,
    and defines a default coarse search parameter grid. Optionally, the default grid can be
    overridden with user-provided custom parameters.

    Args:
        model_name (str): Name of the model. Must be one of:
            - 'logistic'
            - 'decision_tree'
            - 'random_forest'
            - 'xgboost'
            - 'lightgbm'
            - 'catboost'
        custom_coarse_params (dict, optional): User-defined hyperparameter grid to override default.
            If provided, replaces the default coarse grid for the selected model.

    Returns:
        dict: A dictionary containing:
            - 'pipeline': sklearn Pipeline object with preprocessing and classifier
            - 'coarse_params': Coarse-level hyperparameter grid (dict)
    """
    model_name = model_name.lower()

    if model_name == 'logistic':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='liblinear', random_state=42))
        ])
        coarse = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1', 'l2']
        }

    elif model_name == 'random_forest':
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        coarse = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5]
        }

    elif model_name == 'xgboost':
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0, random_state=42))
        ])
        coarse = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 5],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }

    elif model_name == 'lightgbm':
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', LGBMClassifier(random_state=42, verbosity=-1))
        ])
        coarse = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [5, 10],
            'clf__learning_rate': [0.01, 0.1]
        }

    elif model_name == 'catboost':
        pipeline = Pipeline([
            ('scaler', 'passthrough'),
            ('clf', CatBoostClassifier(verbose=0, random_state=42))
        ])
        coarse = {
            'clf__iterations': [100, 200],
            'clf__depth': [4, 6, 8],
            'clf__learning_rate': [0.01, 0.1]
        }

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return {
        'pipeline': pipeline,
        'coarse_params': custom_coarse_params or coarse
    }

from .gridmaster_class import GridMaster
