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
    def __init__(self, models, X_train, y_train, custom_params=None, custom_estimator_params=None, n_jobs=None, verbose=1, refit=True, return_train_score=False):
        """
        Initialize the GridMaster with specified models and training data.

        This constructor sets up the internal model configuration for each model using the provided training dataset and optional custom hyperparameters.


        Args:
            models (list): A list of model names (e.g., ['logistic', 'random_forest', 'xgboost']).
            X_train (array-like or DataFrame): Training features.
            y_train (array-like): Training labels.
            custom_params (dict, optional): Dictionary of custom coarse-level hyperparameters for specific models. Format: {model_name: param_dict}. Defaults to None.
            custom_estimator_params(dict, optional): Dictionary of custom estimator (model) initialization parameters.
                Format: {model_name: param_dict}. Useful for enabling options like GPU. Defaults to None.
            n_jobs (int, optional): Number of parallel jobs for GridSearchCV. Defaults to half of the total detected CPU cores (based on system hardware).  Set to -1 to use all cores, or specify an exact number.
            verbose (int, optional): Verbosity level for GridSearchCV. Controls how much logging is printed. Defaults to 1.
            refit (bool, optional): Whether to refit the best estimator on the entire dataset after search. Defaults to True.
            return_train_score (bool, optional): Whether to include training set scores in cv_results_. Defaults to False.

        Attributes:
            model_dict (dict): Dictionary storing initialized models and their search spaces.
            X_train (array-like): Feature training set.
            y_train (array-like): Label training set.
            results (dict): Stores search results for each model.
            best_model_name (str): Name of the currently best-performing model.
            feature_names (list): List of feature names for plotting and explanation.
            n_jobs (int): Number of parallel jobs for GridSearchCV.
            verbose (int): Verbosity level for GridSearchCV.
            refit (bool): Whether to refit the best estimator after grid search.
            return_train_score (bool): Whether training scores are included in cv_results_.
        """
        self.model_dict = {}
        for model_name in models:
            coarse_params = custom_params.get(model_name) if custom_params else None
            estimator_params = custom_estimator_params.get(model_name) if custom_estimator_params else None
            self.model_dict[model_name] = build_model_config(model_name, custom_coarse_params=coarse_params, custom_estimator_params=estimator_params)
        self.X_train = X_train
        self.y_train = y_train
        self.results = {}
        self.best_model_name = None
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
        import multiprocessing
        if n_jobs is None: n_jobs = max(1, multiprocessing.cpu_count() // 2)

        self.n_jobs = n_jobs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.return_train_score = return_train_score

    def _identify_important_params(self, cv_results, top_n=2):
        param_cols = [k for k in cv_results.keys() if k.startswith('param_')]
        score_diffs = {}

        for param in param_cols:
            values = cv_results[param]
            mean_scores = {}
            for v, score in zip(values, cv_results['mean_test_score']):
                mean_scores.setdefault(v, []).append(score)
            param_range = max([np.mean(scores) for scores in mean_scores.values()]) - min([np.mean(scores) for scores in mean_scores.values()])
            score_diffs[param] = param_range

        sorted_params = sorted(score_diffs.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_params[:top_n]]


    def coarse_search(self, scoring='accuracy', cv=5):
        """
        Perform coarse-level hyperparameter grid search across all models.

        This method iterates through all configured models and performs GridSearchCV using their predefined coarse parameter grids.

        Args:
            scoring (str, optional): Evaluation metric to optimize.
                Must be one of:
                - 'accuracy' (default)
                - 'f1'
                - 'roc_auc'
                - 'precision'
                - 'recall'
                - or any valid sklearn scorer string
                Defaults to 'accuracy'.

            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Notes:
            This method internally uses the following advanced GridSearchCV parameters, set during GridMaster initialization:
            - `n_jobs`: Number of parallel jobs. Defaults to None (single-threaded). Use -1 for all CPU cores.
            - `verbose`: Verbosity level. Controls logging detail.
            - `refit`: Whether to refit the best model on the entire dataset after search.
            - `return_train_score`: Whether to include training set scores in the results.

        Returns:
        None: Updates the `results` dictionary with fitted GridSearchCV objects under the 'coarse' key.
        """
        for name, config in self.model_dict.items():
            grid = GridSearchCV(
                config['pipeline'],
                config['coarse_params'],
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=self.refit,
                return_train_score=self.return_train_score)
            grid.fit(self.X_train, self.y_train)
            self.results[name] = {'coarse': grid}

    def fine_search(self, scoring='accuracy', cv=5, auto_scale=0.5, auto_steps=5,search_mode='smart', custom_fine_params=None):
        """
        Perform fine-level hyperparameter tuning based on coarse search results.

        This method supports three fine-tuning modes:
        1. Smart: Automatically identify the most influential hyperparameters from coarse search and fine-tune only those.
        2. Expert: Use pre-defined common parameters (like learning_rate, C, etc.) for fine-tuning.
        3. Custom: Allow the user to specify an explicit parameter grid.

        Args:
            scoring (str, optional): Scoring metric to optimize.
                Must be one of:
                - 'accuracy' (default)
                - 'f1'
                - 'roc_auc'
                - 'precision'
                - 'recall'
                - or any valid sklearn scorer string
                Defaults to 'accuracy'.

            cv (int, optional): Number of cross-validation folds. Defaults to 5.

                search_mode (str, optional): Fine-tuning strategy.
                    One of:
                    - 'smart': Auto-detect key hyperparameters and fine-tune (default).
                    - 'expert': Fine-tune commonly important parameters.
                    - 'custom': Use a user-provided parameter grid (`custom_fine_params`).
                    Defaults to 'smart'.

            auto_scale (float, optional): Scaling factor for narrowing the search range (e.g., 0.5 = +/-50% around the best value). Defaults to 0.5.

            auto_steps (int, optional): Number of steps/grid points per parameter in fine grid. Defaults to 5.

            custom_fine_params (dict, optional): User-specified fine-tuning parameter grid (only used if `search_mode='custom'`).


        Notes:
            This method internally uses the following advanced GridSearchCV parameters, set during GridMaster initialization:
            - `n_jobs`: Number of parallel jobs. Defaults to None (single-threaded). Use -1 for all CPU cores.
            - `verbose`: Verbosity level. Controls logging detail.
            - `refit`: Whether to refit the best model on the entire dataset after search.
            - `return_train_score`: Whether to include training set scores in the results.

        Returns:
        None: Updates the `results` dictionary with the fine-tuned GridSearchCV objects under the 'fine' key.
        """
        for name in self.results:
            coarse_grid = self.results[name]['coarse']
            best_params = coarse_grid.best_params_


            if search_mode == 'smart':
                important_params = self._identify_important_params(coarse_grid.cv_results_, top_n=2)
                important_keys = [p.replace('param_', '') for p in important_params]
                fine_params = auto_generate_fine_grid(best_params, scale=auto_scale, steps=auto_steps, keys=important_keys,
                                                      coarse_param_grid=self.model_dict[name]['coarse_params'])
            elif search_mode == 'expert':
                expert_keys = [k for k in best_params.keys() if 'learning_rate' in k or 'max_depth' in k]
                fine_params = auto_generate_fine_grid(best_params, scale=auto_scale, steps=auto_steps, keys=expert_keys,
                                                      coarse_param_grid=self.model_dict[name]['coarse_params'])
            elif search_mode == 'custom' and custom_fine_params:
                fine_params = custom_fine_params
            else:
                fine_params = auto_generate_fine_grid(best_params, scale=auto_scale, steps=auto_steps,
                                                      coarse_param_grid=self.model_dict[name]['coarse_params'])

            total_combinations = 1
            for values in fine_params.values():
                total_combinations *= len(values)

            if total_combinations <= 1:
                print(f"⚠️ Skipping fine search for {name}: only one parameter combination detected after generating fine grid.\n"
                      f"Reason: likely due to narrow coarse grid or constrained expert/custom keys.\n"
                      f"Tip: Adjust auto_scale, auto_steps, or provide a richer custom_fine_params.\n")
                continue

            grid = GridSearchCV(
                self.model_dict[name]['pipeline'],
                fine_params,
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=self.refit,
                return_train_score=self.return_train_score
            )
            grid.fit(self.X_train, self.y_train)
            self.results[name]['fine'] = grid
            self.results[name]['best_model'] = grid

    def multi_stage_search(self, model_name, cv=5, scoring='accuracy', stages=[(0.5, 5), (0.2, 5)],search_mode='smart', custom_fine_params=None):
        """
        Perform a multi-stage grid search consisting of one coarse and multiple fine-tuning stages.

        This method first performs a coarse search (if not already done), then iteratively refines the hyperparameter space using a list of `(scale, steps)` tuples.

        Args:
            model_name (str): Name of the model to search (must be present in `model_dict`).
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            scoring (str, optional): Scoring metric to optimize.
                Must be one of:
                - 'accuracy' (default)
                - 'f1'
                - 'roc_auc'
                - 'precision'
                - 'recall'
                - or any valid sklearn scorer string
                Defaults to 'accuracy'.

            stages (list of tuple, optional): List of (scale, steps) for each fine-tuning stage.
             For example, [(0.5, 5), (0.2, 5)] means two rounds of fine tuning:
             ±50% grid with 5 points, then ±20% with 5 points. Defaults to [(0.5, 5), (0.2, 5)].

            search_mode (str, optional): 'smart', 'expert', or 'custom' (with custom_fine_params).
            custom_fine_params (dict, optional): User-provided fine-tuning grid (if search_mode='custom').
            verbose (bool, optional): Whether to print progress messages. Defaults to True.

        Notes:
            This method internally uses the following advanced GridSearchCV parameters, set during GridMaster initialization:
            - `n_jobs`: Number of parallel jobs. Defaults to None (single-threaded). Use -1 for all CPU cores.
            - `verbose`: Verbosity level. Controls logging detail.
            - `refit`: Whether to refit the best model on the entire dataset after search.
            - `return_train_score`: Whether to include training set scores in the results.

        Returns:
        None: Updates the `results` dictionary with intermediate GridSearchCV results for each stage.
        """
        if model_name not in self.model_dict:
            raise ValueError(f"Model '{model_name}' not found.")
        if 'coarse' not in self.results.get(model_name, {}):
            if self.verbose:
                print(f"🔍 [COARSE SEARCHING] for: {model_name}")
            coarse_grid = GridSearchCV(
                self.model_dict[model_name]['pipeline'],
                self.model_dict[model_name]['coarse_params'],
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=self.refit,
                return_train_score=self.return_train_score
                )
            coarse_grid.fit(self.X_train, self.y_train)
            self.results.setdefault(model_name, {})['coarse'] = coarse_grid
        else:
            coarse_grid = self.results[model_name]['coarse']

        last_best = coarse_grid.best_params_
        for i, (scale, steps) in enumerate(stages):
                stage_name = f'stage{i+1}'
                if self.verbose:
                    print(f"🔧 [STAGE {i+1} FINE SEARCHING] for: {model_name} | Scale: {scale} | Steps: {steps}")

                if search_mode == 'smart':
                    important_params = self._identify_important_params(coarse_grid.cv_results_, top_n=2)
                    important_keys = [p.replace('param_', '') for p in important_params]
                    fine_params = auto_generate_fine_grid(last_best, scale=scale, steps=steps, keys=important_keys,
                                                          coarse_param_grid=self.model_dict[model_name]['coarse_params'])
                elif search_mode == 'expert':
                    expert_keys = [k for k in last_best.keys() if 'learning_rate' in k or 'max_depth' in k]
                    fine_params = auto_generate_fine_grid(last_best, scale=scale, steps=steps, keys=expert_keys,
                                                          coarse_param_grid=self.model_dict[model_name]['coarse_params'])
                elif search_mode == 'custom' and custom_fine_params:
                    fine_params = custom_fine_params
                else:
                    fine_params = auto_generate_fine_grid(last_best, scale=scale, steps=steps,
                                                          coarse_param_grid=self.model_dict[model_name]['coarse_params'])

                total_combinations = 1
                for values in fine_params.values():
                    total_combinations *= len(values)

                if total_combinations <= 1:
                    print(f"⚠️ Skipping {stage_name} for {model_name}: only one parameter combination detected.\n"
                          f"Reason: likely due to narrow coarse grid or constrained keys.\n"
                          f"Tip: Adjust auto_scale, auto_steps, or provide a richer custom_fine_params.\n")
                    continue

                fine_grid = GridSearchCV(
                    self.model_dict[model_name]['pipeline'],
                    fine_params,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    refit=self.refit,
                    return_train_score=self.return_train_score
                )
                fine_grid.fit(self.X_train, self.y_train)
                self.results[model_name][stage_name] = fine_grid
                last_best = fine_grid.best_params_

        self.results[model_name]['final'] = fine_grid
        self.results[model_name]['best_model'] = fine_grid

    def compare_best_models(self, X_test, y_test, metrics=['accuracy', 'f1', 'roc_auc'], strategy='rank_sum', weights=None):
        """
        Compare all trained models on test data using specified evaluation metrics.

        This method selects the best estimator for each model (final → fine → coarse), computes scores on the provided test set, and stores results for later access.

        Args:

            X_test (array-like): Feature test set.

            y_test (array-like): Ground truth labels for the test set.

            metrics (list of str, optional): Evaluation metrics to compute. List of evaluation metrics to compute.
                Valid values include:
                - 'accuracy'
                - 'f1'
                - 'roc_auc'
                - 'precision'
                - 'recall'
                Defaults to ['accuracy', 'f1', 'roc_auc'].

            strategy (str, optional): Placeholder for future ranking strategies (currently unused). Defaults to 'rank_sum'.

            weights (dict, optional): Placeholder for future weighted metric strategies (currently unused). Defaults to None.

        Returns:
        None: Updates `results` with 'test_scores' and 'best_model' for each model.
        """
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
        """
        Retrieve a summary of the best model's configuration and performance.

        This includes the best estimator, parameters, cross-validation score,
        and test set scores if available.


        Args:

            model_name (str, optional): Name of the model to summarize. If None, uses the current `best_model_name` set by the user or internal logic. Defaults to None.


        Returns:
            dict: A dictionary with the following keys:
                - 'model_name' (str): Name of the model.
                - 'best_estimator' (sklearn.BaseEstimator): Best estimator object.
                - 'best_params' (dict): Best hyperparameters from search.
                - 'cv_best_score' (float): Best cross-validation score.
                - 'test_scores' (dict): Optional test metrics if available.
        """
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

    def plot_cv_score_curve(
        self,
        model_name,
        metric='mean_test_score',
        plot_train=True,
        figsize=(10, 5),
        show_best_point=True,
        title=None,
        xlabel='Parameter Set Index',
        ylabel=None,
        save_path=None
    ):
        """
        Plot cross-validation scores for each parameter set tried during grid search.

        This visualization helps analyze how performance varies across different hyperparameter combinations.

        Args:
            model_name (str): Name of the model whose results will be plotted.
            metric (str, optional): Score metric to plot Score metric to plot.
            Must be one of the following:
                - 'mean_test_score' (default)
                - 'mean_train_score'
                - 'std_test_score'
                - 'std_train_score'
                - 'rank_test_score'
            Defaults to 'mean_test_score'.

            plot_train (bool, optional): Whether to include training scores in the plot. Defaults to True.

            figsize (tuple, optional): Size of the plot in inches (width, height). Defaults to (10, 5).

            show_best_point (bool, optional): Whether to mark the best score point on the plot. Defaults to True.

            title (str, optional): Custom plot title. If None, a default title will be used. Defaults to None.

            xlabel (str, optional): Label for x-axis. Defaults to 'Parameter Set Index'.

            ylabel (str, optional): Label for y-axis. If None, uses the `metric`. Defaults to None.

            save_path (str, optional): If provided, saves the plot to the specified file path. Defaults to None.

        Returns:
        None: Displays and optionally saves a matplotlib plot.
        """
        grid_obj = self.results[model_name].get('final',
                    self.results[model_name].get('fine',
                    self.results[model_name].get('coarse')))

        results = grid_obj.cv_results_
        scores = results.get(metric)
        train_scores = results.get('mean_train_score') if plot_train else None
        best_idx = grid_obj.best_index_

        plt.figure(figsize=figsize)
        plt.plot(range(len(scores)), scores, marker='o', label='Validation')

        if train_scores is not None:
            plt.plot(range(len(train_scores)), train_scores, marker='x', label='Training')

        if show_best_point:
            plt.axvline(best_idx, color='red', linestyle='--', label='Best Model')
            plt.scatter(best_idx, scores[best_idx], color='red', s=80, zorder=5)

        plt.title(title or f'{model_name.upper()} – Grid Search: {metric}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_confusion_matrix(
        self,
        model_name,
        X_test,
        y_test,
        labels=None,
        normalize=None,
        figsize=(6, 5),
        cmap='Blues',
        title=None,
        save_path=None
    ):
        """
        Plot the confusion matrix for a model's predictions on the test set.


        Args:
            model_name (str): Name of the model to use for prediction.

            X_test (array-like): Test feature set.

            y_test (array-like): True labels.

            labels (list, optional): List of label names to display in the matrix.

            normalize (str or None, optional): Normalization method.
                Must be one of:
                - None (default): No normalization
                - 'true': Normalize over the true condition (rows)
                - 'pred': Normalize over the predicted condition (columns)
                - 'all': Normalize over all values

            figsize (tuple, optional): Size of the figure in inches. Defaults to (6, 5).

            cmap (str or Colormap, optional): Colormap to use (e.g., 'Blues', 'viridis'). Defaults to 'Blues'.

            title (str, optional): Title of the plot. If None, uses default format. Defaults to None.

            save_path (str, optional): If specified, saves the figure to this path. Defaults to None.


        Returns:
            None: Displays and optionally saves the confusion matrix plot.
        """
        from .plot_utils import set_plot_style
        set_plot_style()
        best_model = self.results[model_name].get(
            'final',
            self.results[model_name].get('fine', self.results[model_name].get('coarse'))
        )
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        fig, ax = plt.subplots(figsize=figsize)
        disp.plot(cmap=cmap, ax=ax)
        plt.title(title or f"Confusion Matrix – {model_name.upper()}")
        plt.grid(False)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_model_coefficients(
        self,
        model_name,
        top_n=20,
        sort_descending=True,
        figsize=(10, 5),
        color='teal',
        title=None,
        xlabel='Coefficient Value',
        save_path=None
    ):
        """
        Plot the top N coefficients of a linear model for interpretability.


        Args:
            model_name (str): Name of the model whose coefficients will be plotted.
            top_n (int, optional): Number of most important features to display. Defaults to 20.
            sort_descending (bool, optional): Whether to sort by absolute value in descending order. Defaults to True.
            figsize (tuple, optional): Figure size in inches. Defaults to (10, 5).
            color (str, optional): Bar color. Defaults to 'teal'.
            title (str, optional): Plot title. If None, uses default. Defaults to None.
            xlabel (str, optional): Label for x-axis. Defaults to 'Coefficient Value'.
            save_path (str, optional): If provided, saves the plot to this file path. Defaults to None.

        Returns:
            None: Displays and optionally saves a matplotlib bar chart of model coefficients.
        """
        from .plot_utils import set_plot_style
        set_plot_style()
        best_model = self.results[model_name].get(
            'final',
            self.results[model_name].get('fine', self.results[model_name].get('coarse'))
        )
        coef = best_model.best_estimator_.named_steps['clf'].coef_.flatten()
        feature_names = self.feature_names[:len(coef)]
        df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef,
            'AbsValue': np.abs(coef)
        })
        df = df.sort_values(by='AbsValue', ascending=not sort_descending).head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(df['Feature'], df['Coefficient'], color=color)
        plt.xlabel(xlabel)
        plt.title(title or f"Top {top_n} Coefficients – {model_name.upper()}")
        plt.tight_layout()
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_feature_importance(
        self,
        model_name,
        top_n=20,
        sort_descending=True,
        figsize=(10, 5),
        color='darkgreen',
        title=None,
        xlabel='Feature Importance',
        save_path=None
    ):
        """
        Plot the top N feature importances from a tree-based model.


        Args:
            model_name (str): Name of the model to visualize.
            top_n (int, optional): Number of top features to display. Defaults to 20.
            sort_descending (bool, optional): Whether to sort by importance in descending order. Defaults to True.
            figsize (tuple, optional): Figure size in inches. Defaults to (10, 5).
            color (str, optional): Bar color. Defaults to 'darkgreen'.
            title (str, optional): Plot title. If None, a default will be used. Defaults to None.
            xlabel (str, optional): X-axis label. Defaults to 'Feature Importance'.
            save_path (str, optional): If specified, saves the plot to this path. Defaults to None.

        Returns:
            None: Displays and optionally saves a matplotlib plot.
        """
        from .plot_utils import set_plot_style
        set_plot_style()  # Automatically applies once

        best_model = self.results[model_name].get(
            'final',
            self.results[model_name].get('fine', self.results[model_name].get('coarse'))
        )
        importances = best_model.best_estimator_.named_steps['clf'].feature_importances_
        feature_names = self.feature_names[:len(importances)]

        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=not sort_descending).head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(df['Feature'], df['Importance'], color=color)
        plt.xlabel(xlabel)
        plt.title(title or f"Top {top_n} Feature Importances – {model_name.upper()}")
        plt.tight_layout()
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def get_cv_results(self, model_name, use_fine=True, as_dataframe=True):
        """
        Retrieve cross-validation results from GridSearchCV for a specific model.


        Args:
            model_name (str): Name of the model to retrieve results for.
            use_fine (bool, optional): Whether to retrieve from fine-tuned results (`final`) or coarse-level results. Defaults to True.
            as_dataframe (bool, optional): If True, returns results as a pandas DataFrame. If False, returns the raw `cv_results_` dictionary. Defaults to True.

        Returns:
            Union[pd.DataFrame, dict]: Cross-validation results in the selected format.
        """
        search_result = self.results[model_name].get('final' if use_fine else 'coarse')
        return pd.DataFrame(search_result.cv_results_) if as_dataframe else search_result.cv_results_

    def generate_search_report(self):
        """
        Generate a detailed multi-stage search report across all models, summarizing parameter grids, best parameter sets, and best metric scores.

        Returns:
            str: A formatted multi-line text report summarizing the entire search process.
        """
        report_lines = []
        overall_best_score = -float('inf')
        overall_best_model = None
        overall_best_params = None
        overall_best_metric = None

        for model_name, stages in self.results.items():
            report_lines.append(f"\nFor {model_name.capitalize()} model:")

            best_model = stages.get('best_model')
            if not best_model:
                # Skip models that have no final best_model (i.e., no meaningful search done)
                continue

            scoring = best_model.scoring or 'accuracy'
            report_lines.append(f"Scoring metric used: '{scoring}'")

            stage_counter = 1
            sorted_stage_keys = sorted(
                [k for k in stages.keys() if k not in ['best_model', 'final', 'test_scores']],
                key=lambda x: (0 if x == 'coarse' else 1 if x == 'fine' else int(x.replace('stage', '')) + 2)
            )

            for stage_key in sorted_stage_keys:
                grid = stages[stage_key]
                params = grid.param_grid if hasattr(grid, 'param_grid') else {}
                total_combinations = 1
                param_ranges = []

                for param, values in params.items():
                    clean_values = [float(v) if isinstance(v, np.generic) else v for v in values]
                    param_ranges.append(f"- {param} in {clean_values}")
                    total_combinations *= len(values)

                if stage_key == 'coarse':
                    stage_title = f"Stage {stage_counter}: Coarse grid search"
                elif stage_key == 'fine':
                    stage_title = f"Stage {stage_counter}: Fine grid search"
                elif stage_key.startswith('stage'):
                    stage_number = stage_key.replace('stage', '')
                    stage_title = f"Stage {stage_counter}: Multi-stage fine grid search (Round {stage_number})"
                else:
                    stage_title = f"Stage {stage_counter}: {stage_key}"

                report_lines.append(f"{stage_title}:")

                if total_combinations <= 1:
                    report_lines.append("⚠️ Skipped (no meaningful parameter combinations).\n")
                else:
                    report_lines.extend(param_ranges)
                    report_lines.append(f"Total of {total_combinations} parameter combinations.")
                    clean_best_params = {k: (float(v) if isinstance(v, np.generic) else v) for k, v in grid.best_params_.items()}
                    report_lines.append(f"Best parameters: {clean_best_params}\n")

                stage_counter += 1

            best_score = best_model.best_score_
            best_params = {k: (float(v) if isinstance(v, np.generic) else v) for k, v in best_model.best_params_.items()}

            report_lines.append(f"✅ Conclusion: Best model for {model_name.capitalize()} is {best_params} "
                                f"with best '{scoring}' score of {best_score:.4f}")
            report_lines.append("-" * 60)

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_model = model_name
                overall_best_params = best_params
                overall_best_metric = scoring

        if overall_best_model:
            report_lines.append(f"\n🌟 Summary:")
            report_lines.append(f"The ultimate best model is {overall_best_model.capitalize()} "
                                f"with parameters {overall_best_params} "
                                f"and best '{overall_best_metric}' score of {overall_best_score:.4f}")
        else:
            report_lines.append("\n⚠️ No models completed search. No summary available.")

        final_report = "\n".join(report_lines)
        print(final_report)
        return final_report

    def export_model_package(self, model_name, folder_path='model_exports'):
        """
        Export the best model, model summary, and cross-validation results to disk.

        This function creates a subdirectory under `folder_path` named after the model,
        and stores:

            - The final fitted model (`model_final.joblib`)
            - A JSON summary of model performance and parameters (`best_model_summary.json`)
            - CSV files of cross-validation results for all search stages (e.g., coarse, fine)


        Args:
            model_name (str): Name of the model to export.
            folder_path (str, optional): Base folder to store model files. Defaults to 'model_exports'.

        Returns:
            None
        """
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
        """
        Export all models and their associated results to disk.

        This function loops through all models stored in `self.results` and calls
        `export_model_package()` on each of them. Optionally appends a timestamp
        to each model's output directory to avoid overwriting.


        Args:
            folder_path (str, optional): Base folder to export model files. Defaults to 'model_exports'.
            use_timestamp (bool, optional): If True, appends a timestamp to each model folder name. Useful for versioning and avoiding overwrite. Defaults to True.

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
        for model_name in self.results.keys():
            name_suffix = f"{model_name}_{timestamp}" if use_timestamp else model_name
            model_folder = os.path.join(folder_path, name_suffix)
            try:
                self.export_model_package(model_name, folder_path=model_folder)
            except Exception as e:
                print(f"[⚠️ Skipped] {model_name}: {e}")

    def load_model_package(self, model_name, folder_path='model_exports'):
        """
        Load a saved model package from disk, including the estimator, summary, and CV results.

        This function reads the saved model (.joblib), summary (.json), and cross-validation result files (.csv) from the given folder, and stores them in `self.results`.


        Args:
            model_name (str): Name of the model to load.
            folder_path (str, optional): Base path where model files are stored. Defaults to 'model_exports'.

        Returns:
            None: Updates `self.results` and sets `self.best_model_name`.
        """
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
        """
        Load all saved model packages from the specified folder.

        This function iterates through all subdirectories in `folder_path`, assumes each contains a model export (via `export_model_package()`), and calls
        `load_model_package()` to load them into `self.results`.
        Model names are inferred from subdirectory names (before first underscore if timestamped).

        Args:
            folder_path (str, optional): Directory containing exported model subfolders. Defaults to 'model_exports'.

        Returns:
            None: Updates `self.results` with loaded model data.
        """
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for sub in subdirs:
            model_name = sub.split('_')[0]
            try:
                self.load_model_package(model_name, folder_path=folder_path)
            except Exception as e:
                print(f"[⚠️ Skipped] {sub}: {e}")
