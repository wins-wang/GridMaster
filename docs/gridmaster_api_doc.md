# 📘 GridMaster Python Package: API Documentation

### `GridMaster.__init__`

```python
def __init__(self, models, X_train, y_train, custom_params=None):
    """
    Initialize the GridMaster with specified models and training data.

    This constructor sets up the internal model configuration for each model 
    using the provided training dataset and optional custom hyperparameters.

    Args:
        models (list): A list of model names (e.g., ['logistic', 'random_forest', 'xgboost']).
        X_train (array-like or DataFrame): Training features.
        y_train (array-like): Training labels.
        custom_params (dict, optional): Dictionary of custom coarse-level hyperparameters 
            for specific models. Format: {model_name: param_dict}. Defaults to None.

    Attributes:
        model_dict (dict): Dictionary storing initialized models and their search spaces.
        X_train (array-like): Feature training set.
        y_train (array-like): Label training set.
        results (dict): Stores search results for each model.
        best_model_name (str): Name of the currently best-performing model.
        feature_names (list): List of feature names for plotting and explanation.
    """
```

...

(中略：这里只展示了第一段内容以节省空间，我们将在保存时还原全部内容)
