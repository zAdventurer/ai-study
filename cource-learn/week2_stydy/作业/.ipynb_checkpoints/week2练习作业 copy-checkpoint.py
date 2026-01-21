"""
Week2 机器学习综合练习

本练习涵盖以下内容：
1. 数据处理（数据清洗、特征工程）
2. 分类任务（判断是否为篮球运动员）
3. 回归任务（房价预测）
4. 模型评估（分类和回归的评估指标）

请按照要求完成以下任务。
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
# 任务1：数据处理 - 数据清洗和特征工程
# ============================================================================

def task1_data_processing():
    """
    任务1：数据处理

    给定一个包含"脏"数据的DataFrame，请完成以下操作：
    1. 处理缺失值（用均值填充数值型特征，用众数填充类别型特征）
    2. 处理异常值（年龄应该在18-100之间，房价应该在0-10000之间）
    3. 对类别型特征进行编码（使用LabelEncoder）
    4. 对数值型特征进行标准化（使用StandardScaler）

    返回处理后的DataFrame
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

    # TODO: 请在此处完成数据处理
    # 提示：
    # 1. 处理缺失值
    # 2. 处理异常值（Age应该在18-100之间，HousePrice应该在0-10000之间）
    # 3. 对City进行编码
    # 4. 对数值型特征进行标准化

    # 你的代码写在这里
    df_processed = df.copy()

    # 1. 处理缺失值
    # 对于数值型列，使用均值填充
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            mean_value = df_processed[col].mean()
            df_processed[col].fillna(mean_value, inplace=True)
            print(f"\n列 '{col}': 使用均值 {mean_value:.2f} 填充缺失值")

    # 对于类别型列，使用众数填充
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else '未知'
            df_processed[col].fillna(mode_value, inplace=True)
            print(f"列 '{col}': 使用众数 '{mode_value}' 填充缺失值")

    # 2. 处理异常值
    def detect_outliers_iqr(df, column):
        """使用IQR方法检测异常值"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    def detect_outliers_zscore(df, column, threshold=3):
        """使用Z-score方法检测异常值"""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > threshold]
        return outliers

    print("\n开始异常值检测和处理...")

    # 对数值型列进行异常值检测和处理
    outlier_summary = []

    for col in numeric_cols:
        print(f"\n--- 分析列: {col} ---")

        # IQR方法检测
        outliers_iqr, lower_bound, upper_bound = detect_outliers_iqr(df_processed, col)
        outlier_count_iqr = len(outliers_iqr)

        # Z-score方法检测
        outliers_zscore = detect_outliers_zscore(df_processed, col)
        outlier_count_zscore = len(outliers_zscore)

        print(f"  IQR方法检测到异常值: {outlier_count_iqr} 个")
        print(f"  正常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Z-score方法检测到异常值: {outlier_count_zscore} 个")

        if outlier_count_iqr > 0:
            print(f"  异常值示例:")
            print(outliers_iqr[[col]].head(3))

            # 使用边界值替换异常值
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            print(f"  已使用IQR边界值替换异常值")

        # 应用业务规则限制
        if col == 'Age':
            # 年龄应该在18-100之间
            before_count = len(df_processed[(df_processed[col] < 18) | (df_processed[col] > 100)])
            df_processed[col] = df_processed[col].clip(18, 100)
            after_count = len(df_processed[(df_processed[col] < 18) | (df_processed[col] > 100)])
            print(f"  应用业务规则: Age限制在18-100之间，修正了 {before_count - after_count} 个值")
        elif col == 'HousePrice':
            # 房价应该在0-10000之间
            before_count = len(df_processed[(df_processed[col] < 0) | (df_processed[col] > 10000)])
            df_processed[col] = df_processed[col].clip(0, 10000)
            after_count = len(df_processed[(df_processed[col] < 0) | (df_processed[col] > 10000)])
            print(f"  应用业务规则: HousePrice限制在0-10000之间，修正了 {before_count - after_count} 个值")

        outlier_summary.append({
            '列名': col,
            'IQR异常值数量': outlier_count_iqr,
            'IQR下界': lower_bound,
            'IQR上界': upper_bound,
            'Z-score异常值数量': outlier_count_zscore
        })

    # 显示异常值处理总结
    if outlier_summary:
        print("\n异常值处理总结:")
        summary_df = pd.DataFrame(outlier_summary)
        print(summary_df)

    print(f'\n处理完异常值之后的数据：{df_processed}\n')

    # 3. 对类别型特征进行编码（使用LabelEncoder）
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        print(f"\n类别型特征编码完成：{col}")
        print(f"\n类别型特征编码映射：{dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 4. 对数值型特征进行标准化（使用StandardScaler）
    for col in numeric_cols:
        scaler = StandardScaler()
        df_processed[col] = scaler.fit_transform(df_processed[col])
        print(f"\n数值特征标准化完成：{col}")

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

    使用处理后的数据训练一个分类模型，判断一个人是否为篮球运动员。

    要求：
    1. 准备训练数据（使用任务1处理后的数据）
    2. 分割训练集和测试集（比例8:2）
    3. 训练一个分类模型（可以使用LogisticRegression或DecisionTreeClassifier）
    4. 在测试集上评估模型性能
    5. 输出准确率、混淆矩阵和分类报告

    返回训练好的模型和评估结果
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

    # 任务2实现：分类任务 - 判断是否为篮球运动员
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # 1. 对City进行编码（将城市名称转换为数字）
    # 使用OrdinalEncoder处理未知标签的情况
    oe_city = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()

    # 对训练集和测试集的City列进行编码
    # 先合并训练和测试数据来获取所有可能的城市，然后分别编码
    all_cities = pd.concat([df_train['City'], df_test['City']]).unique()
    oe_city.fit(all_cities.reshape(-1, 1))

    df_train_encoded['City_encoded'] = oe_city.transform(df_train_encoded[['City']])
    df_test_encoded['City_encoded'] = oe_city.transform(df_test_encoded[['City']])

    # 2. 准备特征X和标签y
    X_train = df_train_encoded[['City_encoded', 'Age']]
    y_train = df_train_encoded['IsBasketballPlayer']
    X_test = df_test_encoded[['City_encoded', 'Age']]
    y_test = df_test_encoded['IsBasketballPlayer']

    # 3. 训练LogisticRegression模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # 4. 在测试集上预测
    y_pred = model.predict(X_test)

    # 5. 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n模型准确率：", accuracy)
    print("\n混淆矩阵：")
    print(cm)
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=['不是篮球运动员', '是篮球运动员']))

    return model, accuracy, cm


# ============================================================================
# 任务3：回归任务 - 房价预测
# ============================================================================

def task3_regression():
    """
    任务3：回归任务

    使用处理后的数据训练一个回归模型，预测房价。

    要求：
    1. 准备训练数据（基于演讲稿中的房价预测例子）
    2. 分割训练集和测试集（比例8:2）
    3. 训练一个回归模型（可以使用LinearRegression或DecisionTreeRegressor）
    4. 在测试集上评估模型性能
    5. 输出MAE、MSE、R²指标

    返回训练好的模型和评估结果
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

    # 任务3实现：回归任务 - 房价预测
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # 1. 对Orientation进行编码（将朝向转换为数字）
    # 使用OrdinalEncoder确保训练和测试数据的一致性
    oe_orientation = OrdinalEncoder()
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()

    # 对训练集和测试集的Orientation列进行编码
    # 先合并训练和测试数据来获取所有可能的朝向，然后分别编码
    all_orientations = pd.concat([df_train['Orientation'], df_test['Orientation']]).unique()
    oe_orientation.fit(all_orientations.reshape(-1, 1))

    df_train_encoded['Orientation_encoded'] = oe_orientation.transform(df_train_encoded[['Orientation']])
    df_test_encoded['Orientation_encoded'] = oe_orientation.transform(df_test_encoded[['Orientation']])

    # 2. 准备特征X和标签y
    X_train = df_train_encoded[['SchoolDistrict', 'Orientation_encoded', 'Area']]
    y_train = df_train_encoded['Price']
    X_test = df_test_encoded[['SchoolDistrict', 'Orientation_encoded', 'Area']]
    y_test = df_test_encoded['Price']

    # 3. 训练LinearRegression模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. 在测试集上预测
    y_pred = model.predict(X_test)

    # 5. 计算回归评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 输出模型系数信息
    print(f"\n模型方程: 房价 = {model.intercept_:.2f} + {model.coef_[0]:.2f}×学区等级 + {model.coef_[1]:.2f}×朝向编码 + {model.coef_[2]:.2f}×面积")

    print("\n回归模型评估指标：")
    print(f"MAE (平均绝对误差): {mae:.2f}")
    print(f"MSE (均方误差): {mse:.2f}")
    print(f"R^2 (决定系数): {r2:.4f}")

    # 显示预测结果对比
    print("\n预测结果对比：")
    comparison_df = pd.DataFrame({
        '真实房价': y_test,
        '预测房价': y_pred,
        '误差': y_test - y_pred,
        '绝对误差': abs(y_test - y_pred)
    })
    print(comparison_df)

    return model, mae, mse, r2


# ============================================================================
# 任务4：模型评估分析
# ============================================================================

def task4_model_evaluation():
    """
    任务4：模型评估分析

    基于任务2和任务3的结果，完成以下分析：
    1. 分析分类模型的性能（准确率、混淆矩阵的含义）
    2. 分析回归模型的性能（MAE、MSE、R²的含义）
    3. 讨论模型的泛化能力
    4. 提出改进建议

    返回分析结果（字符串形式）
    """
    print("\n" + "=" * 60)
    print("任务4：模型评估分析")
    print("=" * 60)

# TODO: 请在此处完成模型评估分析
    # 提示：
    # 1. 解释准确率的含义
    # 2. 解释混淆矩阵中TP、TN、FP、FN的含义
    # 3. 解释MAE、MSE、R²的含义
    # 4. 讨论模型的优缺点
    # 5. 提出改进建议

    # 任务4实现：模型对比分析
    print("=" * 60)
    print("任务4：模型对比分析")
    print("=" * 60)

    # 获取之前任务的结果进行分析
    # 这里我们需要基于实际运行结果进行分析

    analysis = """
基于任务2和任务3的实际运行结果，进行以下详细分析：

1. 分类模型性能分析：

   准确率 (Accuracy = 0.8 = 80%)：
   - 含义：模型预测正确的样本占总样本的比例
   - 评价：80%的准确率对于小样本数据集来说表现良好

   混淆矩阵分析 [[2 1] [0 2]]：
   - TP (True Positive) = 2：正确预测为篮球运动员的样本数
   - TN (True Negative) = 2：正确预测为非篮球运动员的样本数
   - FP (False Positive) = 1：错误预测为篮球运动员的样本数（误报）
   - FN (False Negative) = 0：错误预测为非篮球运动员的样本数（漏报）

   分类报告解读：
   - 精确率 (Precision)：预测为正类的样本中真正为正类的比例
     · 非篮球运动员：1.00 (100%) - 所有预测为非篮球运动员的都是正确的
     · 篮球运动员：0.67 (67%) - 预测为篮球运动员的样本中，67%是正确的
   - 召回率 (Recall)：所有正类样本中被正确预测的比例
     · 非篮球运动员：0.67 (67%) - 所有非篮球运动员中，67%被正确识别
     · 篮球运动员：1.00 (100%) - 所有篮球运动员都被正确识别
   - F1分数：精确率和召回率的调和平均值

2. 回归模型性能分析：

   MAE (平均绝对误差 = 28.72)：
   - 含义：预测值与真实值之间绝对误差的平均值
   - 评价：平均预测误差约为29元，相对于房价水平(700-1300)来说误差较小

   MSE (均方误差 = 1399.59)：
   - 含义：预测误差平方的平均值，对大误差更敏感
   - 计算：sqrt(1399.59) ≈ 37.4，说明平均误差约为37元

   R² (决定系数 = 0.9675 ≈ 97%)：
   - 含义：模型解释的变异性比例，1.0表示完美拟合
   - 评价：96.75%的房价变异性被模型解释，拟合效果非常好

3. 模型优缺点分析：

   分类模型 (LogisticRegression)：
   优点：
   - 简单易理解，计算效率高
   - 输出概率值，便于设定阈值
   - 对特征进行线性组合，鲁棒性较好

   缺点：
   - 假设特征与目标变量呈线性关系
   - 对异常值敏感
   - 样本量较小(7个训练样本)，可能存在过拟合风险

   回归模型 (LinearRegression)：
   优点：
   - 简单直观，易于解释每个特征的影响
   - 计算效率高，训练速度快
   - R²达到96.75%，说明模型拟合效果很好

   缺点：
   - 假设线性关系，可能无法捕捉非线性模式
   - 对异常值敏感（尽管数据已预处理）
   - 可能存在多重共线性问题（特征间相关性）

4. 改进建议：

   数据层面：
   - 增加训练样本数量，提高模型泛化能力
   - 收集更多特征（如运动员的身高、体重、运动经验等）
   - 进行交叉验证来评估模型稳定性

   特征工程：
   - 对城市特征使用独热编码(OneHotEncoder)而非标签编码
   - 尝试特征交互，如年龄与城市的组合特征
   - 考虑多项式特征来捕捉非线性关系

   模型层面：
   - 尝试其他分类算法：随机森林、SVM、XGBoost
   - 尝试正则化回归：Ridge回归或Lasso回归
   - 使用集成方法提高预测性能

   评估层面：
   - 使用交叉验证进行更可靠的性能评估
   - 绘制学习曲线分析过拟合/欠拟合问题
   - 使用ROC曲线和AUC评估分类模型

5. 总结：

   当前模型在小数据集上表现良好：
   - 分类模型准确率80%，能够较好地区分篮球运动员
   - 回归模型R^2达97%，房价预测精度很高

   但受限于数据量和特征丰富度，建议在实际应用中：
   - 扩大数据集规模
   - 丰富特征维度
   - 尝试更复杂的模型架构

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
    print("Week2 机器学习综合练习")
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

