# 导入所需要的库和数据，这次我们使用的是经典的数据集，鸢尾花数据集
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# 创建数据目录（如果不存在）
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"已创建数据目录: {DATA_DIR}")

iris = load_iris()
X, y = iris.data, iris.target

# 保存数据集到本地
df_iris = pd.DataFrame(X, columns=iris.feature_names)
df_iris['target'] = y
df_iris['target_name'] = [iris.target_names[i] for i in y]

csv_path = os.path.join(DATA_DIR, 'iris.csv')
df_iris.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n鸢尾花数据集已保存到本地: {csv_path}")
print(f"数据形状: {df_iris.shape}")
print(f"类别: {iris.target_names}")

# 同样划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
## 在分类任务中，划分数据集时使用stratify=y参数，可以保证训练集和测试集中各个类别的比例与原始数据一致。

# 初始化并训练多个模型
# 1. 初始化模型
log_reg = LogisticRegression(max_iter=200)
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear')
tree = DecisionTreeClassifier(random_state=42)

# 2. 创建模型字典，方便管理
models = {
    "Logistic Regression": log_reg,
    "K-Nearest Neighbors": knn,
    "Support Vector Machine": svm,
    "Decision Tree": tree
}

# 3. 循环训练并评估每个模型
for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} 的准确率: {acc:.4f}")


# 单独测试一个预测
# 用训练好的决策树模型来预测一个新样本
# 假设我们有一朵新的鸢尾花，其特征为 [花萼长, 花萼宽, 花瓣长, 花瓣宽]
new_flower = [[5.1, 3.5, 1.4, 0.2]] 
predicted_class = tree.predict(new_flower)
class_name = iris.target_names[predicted_class[0]]

print(f"\n新样本 {new_flower} 被决策树预测为: {class_name} (类别 {predicted_class[0]})")

