Quickstart Usage
================

Here's a basic usage example of GridMaster:

.. code-block:: python

    from gridmaster import GridMaster, build_model_config

    models = ['logistic', 'random_forest']
    X_train, y_train = ...
    gm = GridMaster(models, X_train, y_train)
    gm.coarse_search()
    gm.fine_search()
    gm.compare_best_models(X_test, y_test)
    gm.plot_cv_score_curve('logistic')

You can then export the best model:

.. code-block:: python

    gm.export_model_package('logistic')
