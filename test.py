import numpy as np
import pandas as pd
from gridmaster import GridMaster

def test_generate_search_report():
    # Fake small dataset
    X_train = pd.DataFrame(np.random.rand(20, 4), columns=[f"feature_{i}" for i in range(4)])
    y_train = np.random.randint(0, 2, size=20)

    # Setup GridMaster with two models
    gm = GridMaster(models=['logistic', 'random_forest'], X_train=X_train, y_train=y_train)

    # Simulate fake results
    gm.results = {
        'logistic': {
            'summary': {
                'best_params': {'clf__C': 1.0},
                'cv_best_score': 0.88,
                'scoring': 'accuracy'
            },
            'coarse': type('Dummy', (), {'param_grid': {'clf__C': [0.1, 1, 10]}, 'best_params_': {'clf__C': 1.0}})()
        },
        'random_forest': {
            'summary': {
                'best_params': {'clf__n_estimators': 200, 'clf__max_depth': 10, 'clf__min_samples_split': 2},
                'cv_best_score': 0.90,
                'scoring': 'accuracy'
            },
            'coarse': type('Dummy', (), {'param_grid': {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10], 'clf__min_samples_split': [2, 5]}, 'best_params_': {'clf__n_estimators': 200}})()
        }
    }

    # Generate report
    report = gm.generate_search_report()

    # Assertions (manual check via print; you can integrate with pytest assert if needed)
    print("\n==== Generated Report ====\n")
    print(report)
    print("\n==========================\n")

    assert "Best model for Logistic is {'clf__C': 1.0}" in report, "Logistic model best params mismatch"
    assert "Best model for Random_forest is {'clf__n_estimators': 200, 'clf__max_depth': 10, 'clf__min_samples_split': 2}" in report, "Random Forest best params mismatch"
    assert "ultimate best model is Random_forest" in report, "Overall best model selection mismatch"

if __name__ == "__main__":
    test_generate_search_report()
