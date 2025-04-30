# gridmaster/gridmaster_class.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from .model_search import auto_generate_fine_grid, build_model_config

class GridMaster:
    def __init__(self, models, X_train, y_train, custom_params=None):
        self.model_dict = {}
        for model_name in models:
            params = custom_params.get(model_name) if custom_params else None
            self.model_dict[model_name] = build_model_config(model_name, custom_coarse_params=params)
        self.X_train = X_train
        self.y_train = y_train
        self.results = {}
        self.best_model_name = None
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]

    def coarse_search(self, scoring='accuracy', cv=5):
        for name, config in self.model_dict.items():
            grid = GridSearchCV(config['pipeline'], config['coarse_params'], scoring=scoring, cv=cv, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)
            self.results[name] = {'coarse': grid}

    def fine_search(self, scoring='accuracy', cv=5, auto_scale=0.5, auto_steps=5):
        for name in self.results:
            coarse_grid = self.results[name]['coarse']
            best_params = coarse_grid.best_params_
            fine_params = auto_generate_fine_grid(best_params, scale=auto_scale, steps=auto_steps)
            grid = GridSearchCV(self.model_dict[name]['pipeline'], fine_params, scoring=scoring, cv=cv, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)
            self.results[name]['fine'] = grid

    def multi_stage_search(self, model_name, cv=5, scoring='accuracy', stages=[(0.5, 5), (0.2, 5)], verbose=True):
        if model_name not in self.model_dict:
            raise ValueError(f"Model '{model_name}' not found.")
        if 'coarse' not in self.results.get(model_name, {}):
            if verbose:
                print(f"[Coarse Grid Search] {model_name}")
            coarse_grid = GridSearchCV(self.model_dict[model_name]['pipeline'], self.model_dict[model_name]['coarse_params'], scoring=scoring, cv=cv, n_jobs=-1)
            coarse_grid.fit(self.X_train, self.y_train)
            self.results.setdefault(model_name, {})['coarse'] = coarse_grid
        else:
            coarse_grid = self.results[model_name]['coarse']

        last_best = coarse_grid.best_params_
        for i, (scale, steps) in enumerate(stages):
            stage_name = f'stage{i+1}'
            if verbose:
                print(f"[{stage_name.upper()} Grid Search] scale={scale}, steps={steps}")
            fine_params = auto_generate_fine_grid(last_best, scale=scale, steps=steps)
            fine_grid = GridSearchCV(self.model_dict[model_name]['pipeline'], fine_params, scoring=scoring, cv=cv, n_jobs=-1)
            fine_grid.fit(self.X_train, self.y_train)
            self.results[model_name][stage_name] = fine_grid
            last_best = fine_grid.best_params_
        self.results[model_name]['final'] = fine_grid
        self.results[model_name]['best_model'] = fine_grid

    def compare_best_models(self, X_test, y_test, metrics=['accuracy', 'f1', 'roc_auc'], strategy='rank_sum', weights=None):
        metric_fn = {'accuracy': accuracy_score, 'f1': f1_score, 'roc_auc': roc_auc_score}
        comparison_table = []
        for name, result in self.results.items():
            best_model = result.get('final', result.get('fine', result.get('coarse')))
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            scores = {m: round(metric_fn[m](y_test, y_prob if m == 'roc_auc' and y_prob is not None else y_pred), 4) for m in metrics}
            self.results[name]['test_scores'] = scores
            self.results[name]['best_model'] = best_model
            comparison_table.append((name, scores))

        # Rank models
        score_df = pd.DataFrame({name: scores for name, scores in comparison_table}).T[metrics]
        if strategy == 'rank_sum':
            score_df['score'] = score_df.rank(ascending=False).sum(axis=1)
            best_model = score_df['score'].idxmin()
        elif strategy == 'weighted':
            weights = weights or {m: 1.0 for m in metrics}
            score_df['score'] = sum(score_df[m] * weights.get(m, 1.0) for m in metrics)
            best_model = score_df['score'].idxmax()
        else:
            raise ValueError("Unsupported strategy. Use 'rank_sum' or 'weighted'.")

        self.best_model_name = best_model
        return best_model, score_df

    def get_best_model_summary(self, model_name=None):
        name = model_name or self.best_model_name
        result = self.results[name]
        best_obj = result.get('final', result.get('fine', result.get('coarse')))
        return {
            'model_name': name,
            'best_estimator': best_obj.best_estimator_,
            'best_params': best_obj.best_params_,
            'cv_best_score': round(best_obj.best_score_, 4),
            'test_scores': result.get('test_scores', {})
        }

    def plot_cv_score_curve(self, model_name, metric='mean_test_score', plot_train=True):
        grid_obj = self.results[model_name].get('final', self.results[model_name].get('fine', self.results[model_name].get('coarse')))
        results = grid_obj.cv_results_
        scores = results.get(metric)
        train_scores = results.get('mean_train_score') if plot_train else None
        best_idx = grid_obj.best_index_
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(scores)), scores, marker='o', label='Validation')
        if train_scores is not None:
            plt.plot(range(len(train_scores)), train_scores, marker='x', label='Training')
        plt.axvline(best_idx, color='red', linestyle='--', label='Best Model')
        plt.scatter(best_idx, scores[best_idx], color='red', s=80, zorder=5)
        plt.title(f'{model_name.upper()} – Grid Search: {metric}')
        plt.xlabel('Parameter Set Index')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, model_name, X_test, y_test, labels=None, normalize=None):
        best_model = self.results[model_name].get('final', self.results[model_name].get('fine', self.results[model_name].get('coarse')))
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix – {model_name.upper()}")
        plt.grid(False)
        plt.show()

    def plot_model_coefficients(self, model_name, top_n=20, sort_descending=True):
        best_model = self.results[model_name].get('final', self.results[model_name].get('fine', self.results[model_name].get('coarse')))
        coef = best_model.best_estimator_.named_steps['clf'].coef_.flatten()
        feature_names = self.feature_names[:len(coef)]
        df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef, 'AbsValue': np.abs(coef)})
        df = df.sort_values(by='AbsValue', ascending=not sort_descending).head(top_n)
        plt.figure(figsize=(10, 5))
        plt.barh(df['Feature'], df['Coefficient'], color='teal')
        plt.xlabel("Coefficient Value")
        plt.title(f"Top {top_n} Coefficients – {model_name.upper()}")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()

    def plot_feature_importance(self, model_name, top_n=20, sort_descending=True):
        best_model = self.results[model_name].get('final', self.results[model_name].get('fine', self.results[model_name].get('coarse')))
        importances = best_model.best_estimator_.named_steps['clf'].feature_importances_
        feature_names = self.feature_names[:len(importances)]
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=not sort_descending).head(top_n)
        plt.figure(figsize=(10, 5))
        plt.barh(df['Feature'], df['Importance'], color='darkgreen')
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances – {model_name.upper()}")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()

    def get_cv_results(self, model_name, use_fine=True, as_dataframe=True):
        search_result = self.results[model_name].get('final' if use_fine else 'coarse')
        return pd.DataFrame(search_result.cv_results_) if as_dataframe else search_result.cv_results_

    def export_model_package(self, model_name, folder_path='model_exports'):
        model_folder = os.path.join(folder_path, model_name)
        os.makedirs(model_folder, exist_ok=True)
        best_model = self.results[model_name].get('final', self.results[model_name].get('fine', self.results[model_name]['coarse']))
        joblib.dump(best_model.best_estimator_, os.path.join(model_folder, 'model_final.joblib'))
        summary = self.get_best_model_summary(model_name)
        with open(os.path.join(model_folder, 'best_model_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        for stage_name, grid_obj in self.results[model_name].items():
            if hasattr(grid_obj, 'cv_results_'):
                pd.DataFrame(grid_obj.cv_results_).to_csv(os.path.join(model_folder, f'cv_results_{stage_name}.csv'), index=False)

    def export_all_models(self, folder_path='model_exports', use_timestamp=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
        for model_name in self.results.keys():
            name_suffix = f"{model_name}_{timestamp}" if use_timestamp else model_name
            model_folder = os.path.join(folder_path, name_suffix)
            try:
                self.export_model_package(model_name, folder_path=model_folder)
            except Exception as e:
                print(f"[⚠️ Skipped] {model_name}: {e}")

    def load_model_package(self, model_name, folder_path='model_exports'):
        import_path = os.path.join(folder_path, model_name)
        model_file = os.path.join(import_path, 'model_final.joblib')
        summary_file = os.path.join(import_path, 'best_model_summary.json')
        estimator = joblib.load(model_file)
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        cv_results = {}
        for file in os.listdir(import_path):
            if file.startswith('cv_results_') and file.endswith('.csv'):
                stage_name = file.replace('cv_results_', '').replace('.csv', '')
                cv_results[stage_name] = pd.read_csv(os.path.join(import_path, file))
        self.results[model_name] = {
            'loaded_model': estimator,
            'summary': summary,
            'cv_results_loaded': cv_results
        }
        self.best_model_name = model_name

    def import_all_models(self, folder_path='model_exports'):
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for sub in subdirs:
            model_name = sub.split('_')[0]
            try:
                self.load_model_package(model_name, folder_path=folder_path)
            except Exception as e:
                print(f"[⚠️ Skipped] {sub}: {e}")
