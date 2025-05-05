import pandas as pd
from sklearn.datasets import load_iris
from gridmaster import GridMaster

# Load a small dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Initialize GridMaster with at least two models
gm = GridMaster(models=['logistic', 'random_forest'], X_train=X, y_train=y, verbose=True)

# Run multi-stage search
gm.multi_stage_search(scoring='accuracy')

# Check if 'summary' is filled for each model
for model_name, result in gm.results.items():
    if 'summary' in result:
        print(f"✅ Summary found for {model_name}:")
        print(result['summary'])
    else:
        print(f"❌ No summary recorded for {model_name}!")

# Generate and print the full search report
print("\n=== Generated Search Report ===")
report = gm.generate_search_report()
print(report)
