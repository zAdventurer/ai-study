import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免图形显示问题
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 多元线性回归作业 ===")
print("请完成TODO标记的部分\n")

# 加载加州房价数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("数据集信息:")
print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"房价范围: {y.min():.2f} - {y.max():.2f}")

# 数据探索
print("\n数据探索:")
print("前5行数据:")
print(X.head())
print("\n基本统计信息:")
print(X.describe())
print("\n=== 基础多元线性回归 ===")

# TODO 1: 实现多元线性回归（约8行代码）
# 1. 划分训练集和测试集（比例8:2）
# 2. 训练线性回归模型
# 3. 在测试集上预测
# 4. 计算MSE和R²分数

# 你的代码：
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

print("\n=== 特征工程与改进 ===")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TODO 2: 使用标准化后的特征重新训练模型（约6行代码）
# 1. 划分标准化后的数据
# 2. 训练新模型
# 3. 预测并评估性能
# 4. 比较标准化前后的效果

# 你的代码：
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)
print(f"标准化后 MSE: {mse_scaled:.4f}")
print(f"标准化后 R²: {r2_scaled:.4f}")

print("\n=== 结果分析 ===")

# 特征重要性分析
def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    coefficients = model.coef_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    feature_importance = feature_importance.sort_values('coefficient', key=abs, ascending=False)
    print("特征重要性（按系数绝对值排序）:")
    print(feature_importance)
    return feature_importance

# 可视化结果
def plot_results(y_true, y_pred, title):
    """绘制预测结果"""
    plt.figure(figsize=(10, 4))

    # 预测值vs真实值
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{title} - 预测值vs真实值')

    # 残差图
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title(f'{title} - 残差图')

    plt.tight_layout()
    # 保存图片而不是显示
    plt.savefig(f'{title}_结果分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存为: {title}_结果分析.png")

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

print("\n作业完成")

# 分析基础模型
print("\n--- 基础模型分析 ---")
analyze_feature_importance(model, housing.feature_names)
mae, rmse = calculate_metrics(y_test, y_pred)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
plot_results(y_test, y_pred, "基础模型")

# 分析标准化后的模型
print("\n--- 标准化模型分析 ---")
analyze_feature_importance(model_scaled, housing.feature_names)
mae_scaled, rmse_scaled = calculate_metrics(y_test, y_pred_scaled)
print(f"MAE: {mae_scaled:.4f}")
print(f"RMSE: {rmse_scaled:.4f}")
plot_results(y_test, y_pred_scaled, "标准化模型")

# 模型对比
print("\n--- 模型对比 ---")
print("基础模型 vs 标准化模型:")
print(f"MSE: {mse:.4f} vs {mse_scaled:.4f}")
print(f"R²: {r2:.4f} vs {r2_scaled:.4f}")
print(f"MAE: {mae:.4f} vs {mae_scaled:.4f}")
print(f"RMSE: {rmse:.4f} vs {rmse_scaled:.4f}")


# 额外添加
print("\n=== 揭秘：为什么结果一样？ ===")
print("虽然MSE一样，但请观察系数的变化：")

# 获取两个模型的系数
coef_original = pd.Series(model.coef_, index=housing.feature_names)
coef_scaled = pd.Series(model_scaled.coef_, index=housing.feature_names)

compare_df = pd.DataFrame({
    '未标准化系数': coef_original,
    '标准化后系数': coef_scaled
})

print(compare_df.head())
print("\n结论：特征数值变了，模型自动调整了系数，导致最终预测结果一模一样。")