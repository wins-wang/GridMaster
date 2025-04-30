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
    Generate a fine-tuning parameter grid centered around best_params,
    and automatically fix or filter out invalid values for compatibility
    with scikit-learn and other supported classifiers.
    """
    fine_grid = {}
    for param, value in best_params.items():
        if isinstance(value, (int, float)):
            low = max(0, value * (1 - scale))
            high = value * (1 + scale)
            values = np.linspace(low, high, steps)

            if isinstance(value, int):
                values = sorted(set(int(round(v)) for v in values))
            else:
                values = np.round(values, 5).tolist()

            fine_grid[param] = values
        else:
            fine_grid[param] = [value]

    # 🔒 Constraints for common classifier hyperparameters
    param_constraints = {
        'clf__C': lambda v: isinstance(v, float) and v > 0,
        'clf__max_depth': lambda v: v is None or (isinstance(v, int) and v > 0),
        'clf__min_samples_split': lambda v: (isinstance(v, int) and v >= 2) or (isinstance(v, float) and 0.0 < v <= 1.0),
        'clf__n_estimators': lambda v: isinstance(v, int) and v > 0,
        'clf__learning_rate': lambda v: isinstance(v, float) and 0.0 < v <= 1.0,
        'clf__iterations': lambda v: isinstance(v, int) and v > 0,
        'clf__depth': lambda v: isinstance(v, int) and v > 0,
    }

    for param, check_fn in param_constraints.items():
        if param in fine_grid:
            fine_grid[param] = [v for v in fine_grid[param] if check_fn(v)]

    return fine_grid

def build_model_config(model_name, custom_coarse_params=None):
    """
    Define pipeline and coarse grid based on model name.
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
