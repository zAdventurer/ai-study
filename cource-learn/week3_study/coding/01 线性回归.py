# 导入所需要的库
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# 创建数据目录（如果不存在）
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"已创建数据目录: {DATA_DIR}")

# 加载数据，Scikit-learn 提供了一些内置的数据集，比如加州房价数据集
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# 转换为Pandas DataFrame方便查看
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# 保存数据集到本地
csv_path = os.path.join(DATA_DIR, 'california_housing.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n数据集已保存到本地: {csv_path}")
print(f"数据形状: {df.shape}")

print("数据集前5行:")
print(df.head())
print("\n数据集信息:")
df.info()

# 准备数据：划分训练集和测试集
X = df.drop('PRICE', axis=1) # 特征
y = df['PRICE'] # 标签

# 划分数据集，80%训练，20%测试，random_state保证每次划分结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 创建并训练模型
# 1. 创建模型实例
model = LinearRegression()

# 2. 训练模型
model.fit(X_train, y_train)

print("模型训练完成!")
print(f"模型截距 (b): {model.intercept_}")
print(f"模型系数 (w): {model.coef_}")

# 测试与评估，使用评估指标来衡量模型的好坏
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n均方误差 (MSE): {mse:.2f}")
print(f"R² 分数: {r2:.2f}")

# 分析模型系数
print("\n=== 特征重要性分析 ===")
feature_importance = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': model.coef_
})
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
print("特征重要性（按系数绝对值排序）:")
print(feature_importance)

# 计算RMSE（均方根误差）
rmse = np.sqrt(mse)
print(f"\n均方根误差 (RMSE): {rmse:.2f}")

# 计算平均绝对误差
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"平均绝对误差 (MAE): {mae:.2f}")


