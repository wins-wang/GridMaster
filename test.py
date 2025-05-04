import pandas as pd
from sklearn.datasets import load_breast_cancer
from gridmaster import GridMaster

# 准备数据
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 初始化
gm = GridMaster(models=['logistic', 'random_forest'], X_train=X, y_train=y)

# Coarse search
gm.coarse_search()
print("✅ Coarse search completed.")

# Case 2: Expert fine search
gm.fine_search(search_mode='expert')
print("✅ Expert fine search completed.")

# Case 3: Multi-stage search
gm.multi_stage_search(stages=[(0.5, 5), (0.3, 5), (0.1, 5)])
print("✅ Multi-stage search completed.")

# Case 4: Custom fine search
custom_grid = {'clf__n_estimators': [100, 150, 200], 'clf__max_depth': [5, 10, 15]}
gm.fine_search(search_mode='custom', custom_fine_params=custom_grid)
print("✅ Custom fine search completed.")

# Case 5: Compare and export
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gm.generate_search_report()

gm.compare_best_models(X_test, y_test)
summary = gm.get_best_model_summary()
print(f"✅ Best model summary:\n{summary}")

# 运行 coarse_search 先产生模型结果（保
# 定义测试用的导出文件夹
test_folder = 'test_exports'

# 导出所有模型
print("✅ Starting export...")
gm.export_all_models(folder_path=test_folder, use_timestamp=True)
print("✅ Models exported.")

# 检查导出目录内容
import os
print("📂 Exported folders:", os.listdir(test_folder))

# 导入所有模型
print("✅ Starting import...")
gm.import_all_models(folder_path=test_folder)
print("✅ Models re-imported.")

# 验证导入后的结果
print("🔍 Loaded models after import:", list(gm.results.keys()))
for model_name, data in gm.results.items():
    print(f"Model: {model_name}")
    print("  - Loaded model:", type(data.get('loaded_model')))
    print("  - Summary keys:", data.get('summary').keys())
    print("  - CV result stages:", list(data.get('cv_results_loaded').keys()))
