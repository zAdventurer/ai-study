"""
Week2 机器学习综合练习 - 参考答案

本参考答案展示了如何完成以下任务：
1. 数据处理（数据清洗、特征工程）
2. 分类任务（判断是否为篮球运动员）
3. 回归任务（房价预测）
4. 模型评估（分类和回归的评估指标）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 辅助函数：处理未知标签的编码
# ============================================================================

def encode_with_unknown(value, encoder, default_value=None, default_label=None):
    """
    处理可能包含未知标签的编码
    
    参数:
        value: 要编码的值
        encoder: 已训练的LabelEncoder
        default_value: 默认编码值（如果为None，则使用default_label计算）
        default_label: 默认标签（训练集中最常见的标签）
    
    返回:
        编码后的值
    """
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # 如果遇到未知标签，使用默认值
        if default_value is None and default_label is not None:
            default_value = encoder.transform([default_label])[0]
        return default_value if default_value is not None else 0

# ============================================================================
# 任务1：数据处理 - 数据清洗和特征工程
# ============================================================================

def task1_data_processing():
    """
    任务1：数据处理
    
    完成数据清洗和特征工程：
    1. 处理缺失值（用均值填充数值型特征，用众数填充类别型特征）
    2. 处理异常值（年龄应该在18-100之间，房价应该在0-10000之间）
    3. 对类别型特征进行编码（使用LabelEncoder）
    4. 对数值型特征进行标准化（使用StandardScaler）
    """
    # 创建包含"脏"数据的示例数据
    data = {
        'Name': ['Mike', 'Jerry', 'Bryan', 'Patricia', 'Elodie', 'Remy', 'John', 'Marine', 'Julien', 'Fred'],
        'City': ['Miami', 'New York', 'Orlando', 'Miami', 'Phoenix', 'Chicago', 'New York', 'Miami', None, 'Orlando'],
        'Age': [42, 32, 18, 45, 35, 72, 48, 45, 52, 200],  # 包含异常值200
        'Salary': [5000, 6000, None, 5500, 5800, 7000, 6500, 5200, 6800, 3000],
        'HousePrice': [1000, 1300, 700, 1100, 850, 1500, 1200, 1050, 1400, -100]  # 包含异常值-100
    }
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("任务1：数据处理")
    print("=" * 60)
    print("\n原始数据：")
    print(df)
    print("\n数据信息：")
    print(df.info())
    print("\n缺失值统计：")
    print(df.isnull().sum())
    
    # 复制数据框进行处理
    df_processed = df.copy()
    
    # 步骤1：处理缺失值
    print("\n步骤1：处理缺失值...")
    
    # 对类别型特征用众数填充
    if df_processed['City'].isnull().any():
        city_mode = df_processed['City'].mode()[0]  # 获取众数
        df_processed['City'].fillna(city_mode, inplace=True)
        print(f"City缺失值用众数 '{city_mode}' 填充")
    
    # 对数值型特征用均值填充
    numeric_cols = ['Age', 'Salary', 'HousePrice']
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            mean_val = df_processed[col].mean()
            df_processed[col].fillna(mean_val, inplace=True)
            print(f"{col}缺失值用均值 {mean_val:.2f} 填充")
    
    # 步骤2：处理异常值
    print("\n步骤2：处理异常值...")
    
    # 处理Age异常值（应该在18-100之间）
    age_outliers = (df_processed['Age'] < 18) | (df_processed['Age'] > 100)
    if age_outliers.any():
        print(f"发现 {age_outliers.sum()} 个Age异常值")
        # 将异常值替换为边界值或中位数
        df_processed.loc[df_processed['Age'] > 100, 'Age'] = df_processed['Age'].median()
        df_processed.loc[df_processed['Age'] < 18, 'Age'] = 18
    
    # 处理HousePrice异常值（应该在0-10000之间）
    price_outliers = (df_processed['HousePrice'] < 0) | (df_processed['HousePrice'] > 10000)
    if price_outliers.any():
        print(f"发现 {price_outliers.sum()} 个HousePrice异常值")
        # 将异常值替换为边界值或中位数
        df_processed.loc[df_processed['HousePrice'] < 0, 'HousePrice'] = df_processed['HousePrice'].median()
        df_processed.loc[df_processed['HousePrice'] > 10000, 'HousePrice'] = df_processed['HousePrice'].median()
    
    # 步骤3：对类别型特征进行编码
    print("\n步骤3：对类别型特征进行编码...")
    le_city = LabelEncoder()
    df_processed['City_encoded'] = le_city.fit_transform(df_processed['City'])
    print(f"City编码映射: {dict(zip(le_city.classes_, range(len(le_city.classes_))))}")
    
    # 步骤4：对数值型特征进行标准化
    print("\n步骤4：对数值型特征进行标准化...")
    scaler = StandardScaler()
    numeric_features = ['Age', 'Salary', 'HousePrice']
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    print("数值型特征已标准化（均值为0，标准差为1）")
    
    print("\n处理后的数据：")
    print(df_processed)
    print("\n处理后的数据信息：")
    print(df_processed.info())
    
    return df_processed


# ============================================================================
# 任务2：分类任务 - 判断是否为篮球运动员
# ============================================================================

def task2_classification():
    """
    任务2：分类任务
    
    训练一个分类模型，判断一个人是否为篮球运动员。
    """
    print("\n" + "=" * 60)
    print("任务2：分类任务 - 判断是否为篮球运动员")
    print("=" * 60)
    
    # 创建训练数据（基于演讲稿中的例子）
    train_data = {
        'City': ['Miami', 'New York', 'Orlando', 'Miami', 'Phoenix', 'Chicago', 'New York'],
        'Age': [42, 32, 18, 45, 35, 72, 48],
        'IsBasketballPlayer': [1, 0, 0, 1, 0, 1, 1]  # 1表示是，0表示否
    }
    df_train = pd.DataFrame(train_data)
    
    # 创建测试数据
    test_data = {
        'City': ['Miami', 'Miami', 'Orlando', 'Boston', 'Phoenix'],
        'Age': [45, 52, 20, 34, 90],
        'IsBasketballPlayer': [1, 1, 0, 0, 0]  # 真实标签（用于评估）
    }
    df_test = pd.DataFrame(test_data)
    
    print("\n训练数据：")
    print(df_train)
    print("\n测试数据：")
    print(df_test)
    
    # 步骤1：对City进行编码
    le = LabelEncoder()
    df_train['City_encoded'] = le.fit_transform(df_train['City'])
    
    # 获取默认值（训练集中最常见的城市）
    most_common_city = df_train['City'].mode()[0]
    default_city_code = le.transform([most_common_city])[0]
    
    # 对测试集的City进行编码，处理未知标签
    df_test['City_encoded'] = df_test['City'].apply(
        lambda x: encode_with_unknown(x, le, default_city_code, most_common_city)
    )
    
    # 检查是否有未知标签
    unknown_cities = df_test[~df_test['City'].isin(le.classes_)]
    if not unknown_cities.empty:
        print(f"\n警告：测试集中发现训练时未见过的城市: {unknown_cities['City'].unique()}")
        print(f"这些城市将被编码为最常见城市 '{most_common_city}' 的编码值: {default_city_code}")
    
    # 步骤2：准备特征X和标签y
    X_train = df_train[['City_encoded', 'Age']].values
    y_train = df_train['IsBasketballPlayer'].values
    
    X_test = df_test[['City_encoded', 'Age']].values
    y_test = df_test['IsBasketballPlayer'].values
    
    # 步骤3：标准化特征（可选，但通常有助于提高性能）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 步骤4：训练模型
    # 使用逻辑回归
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 步骤5：预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率
    
    # 步骤6：评估模型
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['不是', '是'])
    
    print("\n模型预测结果：")
    print(f"预测类别: {y_pred}")
    print(f"预测概率: {y_pred_proba}")
    
    print("\n模型准确率：", accuracy)
    print("\n混淆矩阵：")
    print("       预测")
    print("     不是  是")
    print(f"不是  {cm[0,0]}  {cm[0,1]}")
    print(f"是    {cm[1,0]}  {cm[1,1]}")
    print("\n分类报告：")
    print(report)
    
    # 解释混淆矩阵
    print("\n混淆矩阵解释：")
    print(f"TP (真正例): {cm[1,1]} - 实际是，预测也是")
    print(f"TN (真负例): {cm[0,0]} - 实际不是，预测也不是")
    print(f"FP (假正例): {cm[0,1]} - 实际不是，但预测是")
    print(f"FN (假负例): {cm[1,0]} - 实际是，但预测不是")
    
    return model, accuracy, cm


# ============================================================================
# 任务3：回归任务 - 房价预测
# ============================================================================

def task3_regression():
    """
    任务3：回归任务
    
    训练一个回归模型，预测房价。
    """
    print("\n" + "=" * 60)
    print("任务3：回归任务 - 房价预测")
    print("=" * 60)
    
    # 创建训练数据（基于演讲稿中的例子）
    train_data = {
        'SchoolDistrict': [8, 9, 6, 9, 3, 7, 8, 5],
        'Orientation': ['南', '西南', '北', '东南', '南', '东', '南', '西'],
        'Area': [100, 120, 60, 80, 95, 110, 105, 75],
        'Price': [1000, 1300, 700, 1100, 850, 1200, 1150, 900]
    }
    df_train = pd.DataFrame(train_data)
    
    # 创建测试数据
    test_data = {
        'SchoolDistrict': [3, 7, 8, 6],
        'Orientation': ['南', '东', '南', '北'],
        'Area': [95, 110, 105, 60],
        'Price': [850, 1200, 1150, 700]  # 真实标签（用于评估）
    }
    df_test = pd.DataFrame(test_data)
    
    print("\n训练数据：")
    print(df_train)
    print("\n测试数据：")
    print(df_test)
    
    # 步骤1：对Orientation进行编码
    le = LabelEncoder()
    df_train['Orientation_encoded'] = le.fit_transform(df_train['Orientation'])
    
    # 获取默认值（训练集中最常见的朝向）
    most_common_orientation = df_train['Orientation'].mode()[0]
    default_orientation_code = le.transform([most_common_orientation])[0]
    
    # 对测试集的Orientation进行编码，处理未知标签
    df_test['Orientation_encoded'] = df_test['Orientation'].apply(
        lambda x: encode_with_unknown(x, le, default_orientation_code, most_common_orientation)
    )
    
    # 检查是否有未知标签
    unknown_orientations = df_test[~df_test['Orientation'].isin(le.classes_)]
    if not unknown_orientations.empty:
        print(f"\n警告：测试集中发现训练时未见过的朝向: {unknown_orientations['Orientation'].unique()}")
        print(f"这些朝向将被编码为最常见朝向 '{most_common_orientation}' 的编码值: {default_orientation_code}")
    
    # 步骤2：准备特征X和标签y
    X_train = df_train[['SchoolDistrict', 'Orientation_encoded', 'Area']].values
    y_train = df_train['Price'].values
    
    X_test = df_test[['SchoolDistrict', 'Orientation_encoded', 'Area']].values
    y_test = df_test['Price'].values
    
    # 步骤3：标准化特征（回归任务通常也需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 步骤4：训练模型
    # 使用线性回归
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 步骤5：预测
    y_pred = model.predict(X_test_scaled)
    
    # 步骤6：评估模型
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n预测结果对比：")
    comparison = pd.DataFrame({
        '真实价格': y_test,
        '预测价格': y_pred,
        '误差': np.abs(y_test - y_pred)
    })
    print(comparison)
    
    print("\n回归模型评估指标：")
    print(f"MAE (平均绝对误差): {mae:.2f}")
    print(f"  - 含义：平均每个预测值与真实值相差 {mae:.2f} 个单位")
    print(f"  - 单位：与目标变量相同（房价单位）")
    print(f"  - 越小越好")
    
    print(f"\nMSE (均方误差): {mse:.2f}")
    print(f"  - 含义：预测误差的平方的平均值")
    print(f"  - 单位：目标变量单位的平方")
    print(f"  - 对大的误差惩罚更重")
    print(f"  - 越小越好")
    
    print(f"\nR² (决定系数): {r2:.4f}")
    print(f"  - 含义：模型解释了 {r2*100:.2f}% 的数据变异")
    print(f"  - 取值范围：[0, 1]")
    print(f"  - R² = 1：完美拟合")
    print(f"  - R² = 0：模型与简单预测平均值一样")
    print(f"  - R² < 0：模型比简单预测平均值还差")
    print(f"  - 越大越好")
    
    return model, mae, mse, r2


# ============================================================================
# 任务4：模型评估分析
# ============================================================================

def task4_model_evaluation():
    """
    任务4：模型评估分析
    
    基于任务2和任务3的结果，完成模型评估分析。
    """
    print("\n" + "=" * 60)
    print("任务4：模型评估分析")
    print("=" * 60)
    
    analysis = """
    ============================================================
    模型评估分析报告
    ============================================================
    
    1. 分类模型性能分析：
    
    准确率（Accuracy）：
    - 含义：分类正确的样本数占总样本数的比例
    - 优点：直观易懂，适合类别平衡的数据集
    - 局限：在类别不平衡时可能不够准确
    
    混淆矩阵（Confusion Matrix）：
    - TP (真正例)：实际是正类，预测也是正类
    - TN (真负例)：实际是负类，预测也是负类
    - FP (假正例)：实际是负类，但预测为正类（第一类错误）
    - FN (假负例)：实际是正类，但预测为负类（第二类错误）
    
    从混淆矩阵可以计算：
    - 精确率（Precision）= TP / (TP + FP)：预测为正类中真正为正类的比例
    - 召回率（Recall）= TP / (TP + FN)：实际正类中被正确预测的比例
    - F1分数：精确率和召回率的调和平均
    
    ============================================================
    
    2. 回归模型性能分析：
    
    MAE (平均绝对误差)：
    - 含义：预测值与真实值差值的绝对值的平均
    - 优点：单位与目标变量相同，易于理解；对异常值不敏感
    - 缺点：所有误差权重相同，无法突出大误差
    
    MSE (均方误差)：
    - 含义：预测值与真实值差值的平方的平均
    - 优点：对大误差惩罚更重，常用于模型训练
    - 缺点：单位是目标变量单位的平方，不够直观
    
    R² (决定系数)：
    - 含义：模型解释了数据变异的多少比例
    - 优点：无量纲，便于比较不同模型
    - 取值范围：[0, 1]，值越大越好
    - R² = 1：完美拟合
    - R² = 0：模型与简单预测平均值一样
    - R² < 0：模型比简单预测平均值还差
    
    ============================================================
    
    3. 模型泛化能力讨论：
    
    泛化能力是模型最重要的特性，指的是模型在新数据上的表现。
    
    训练误差 vs 泛化误差：
    - 训练误差：模型在训练集上的误差，反映模型对训练数据的拟合程度
    - 泛化误差：模型在新样本上的误差，反映模型的真实能力
    
    过拟合 vs 欠拟合：
    - 欠拟合（Underfitting）：
      * 训练误差大，泛化误差也大
      * 模型太简单，未学到数据特征
      * 解决方法：增加模型复杂度、增加特征、减少正则化
    
    - 过拟合（Overfitting）：
      * 训练误差小但泛化误差大
      * 模型太复杂，学习了噪声
      * 解决方法：简化模型、增加数据、增加正则化、使用交叉验证
    
    偏差-方差权衡：
    - 偏差（Bias）：模型预期预测值与正确值的差异
    - 方差（Variance）：模型预测结果在均值附近的偏移幅度
    - 理想状态：低偏差 + 低方差
    - 需要找到偏差和方差的平衡点
    
    ============================================================
    
    4. 改进建议：
    
    对于分类任务：
    1. 增加更多特征：如身高、体重、运动历史等
    2. 收集更多训练数据：提高模型的泛化能力
    3. 尝试不同的算法：如决策树、随机森林、SVM等
    4. 处理类别不平衡：如果数据不平衡，使用SMOTE等方法
    5. 特征工程：创建组合特征，如年龄与城市的交互项
    6. 交叉验证：使用k折交叉验证评估模型稳定性
    
    对于回归任务：
    1. 特征工程：创建新特征，如面积与学区的交互项
    2. 非线性模型：如果关系非线性，尝试多项式回归或树模型
    3. 特征选择：使用相关性分析或特征重要性选择重要特征
    4. 正则化：使用L1或L2正则化防止过拟合
    5. 数据标准化：确保所有特征在同一尺度
    6. 异常值处理：识别并处理异常值
    
    通用改进方法：
    1. 数据质量：确保数据清洗充分，处理缺失值和异常值
    2. 数据量：收集更多高质量的训练数据
    3. 模型选择：尝试不同的算法，选择最适合的模型
    4. 超参数调优：使用网格搜索或随机搜索优化超参数
    5. 集成方法：使用集成学习（如随机森林、梯度提升）提高性能
    6. 交叉验证：使用交叉验证评估模型，避免过拟合
    
    ============================================================
    """
    
    print(analysis)
    return analysis


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：依次执行所有任务
    """
    print("\n" + "=" * 60)
    print("Week2 机器学习综合练习 - 参考答案")
    print("=" * 60)
    
    # 执行任务1：数据处理
    df_processed = task1_data_processing()
    
    # 执行任务2：分类任务
    classification_model, accuracy, cm = task2_classification()
    
    # 执行任务3：回归任务
    regression_model, mae, mse, r2 = task3_regression()
    
    # 执行任务4：模型评估分析
    analysis = task4_model_evaluation()
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

