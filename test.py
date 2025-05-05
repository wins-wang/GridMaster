import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from gridmaster import GridMaster

# Load binary classification dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Initialize GridMaster
gm = GridMaster(models=['logistic', 'random_forest'], X_train=X_train, y_train=y_train)

# Perform search
gm.multi_stage_search(scoring='f1')

# Summarize best model
summary = gm.get_best_model_summary()
print(summary)

# Plot ROC curve
print("✅ Plotting ROC Curve...")
gm.plot_roc_curve(model_name=summary['model_name'], X_test=X_test, y_test=y_test)

# Plot Precision-Recall curve
print("✅ Plotting Precision-Recall Curve...")
gm.plot_precision_recall_curve(model_name=summary['model_name'], X_test=X_test, y_test=y_test)
