# ============================================
# Week1 综合练习作业 - 参考答案
# ============================================
# 本文件提供了完整的参考答案
# 注意：答案不唯一，这里提供一种标准解法
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================
# 题目1：Python基础与字符串格式化（10分）
# ============================================

print("=" * 50)
print("题目1：Python基础与字符串格式化")
print("=" * 50)

# 参考答案
name = "张三"  # 替换为你的姓名
age = 20      # 替换为你的年龄
language = "Python"  # 替换为你最喜欢的编程语言

# 使用f-string格式化输出
print(f"我是{name}，今年{age}岁，最喜欢的编程语言是{language}")


# ============================================
# 题目2：NumPy数组操作（20分）
# ============================================

print("\n" + "=" * 50)
print("题目2：NumPy数组操作")
print("=" * 50)

# 参考答案
# 1. 创建包含10个随机整数（范围1-100）的一维数组
np.random.seed(42)  # 设置随机种子，保证结果可复现
arr = np.random.randint(1, 101, size=10)
print(f"1. 创建的数组: {arr}")

# 2. 计算最大值、最小值和平均值
max_val = arr.max()
min_val = arr.min()
mean_val = arr.mean()
print(f"2. 最大值: {max_val}, 最小值: {min_val}, 平均值: {mean_val:.2f}")

# 3. 找出数组中大于50的元素
greater_than_50 = arr[arr > 50]
print(f"3. 大于50的元素: {greater_than_50}")

# 4. 将数组重塑为2行5列的二维数组
arr_2d = arr.reshape(2, 5)
print(f"4. 重塑后的二维数组:\n{arr_2d}")

# 5. 计算二维数组每行的平均值
row_means = arr_2d.mean(axis=1)
print(f"5. 每行的平均值: {row_means}")


# ============================================
# 题目3：Pandas数据处理（30分）
# ============================================

print("\n" + "=" * 50)
print("题目3：Pandas数据处理")
print("=" * 50)

# 参考答案
# 1. 创建DataFrame
data = {
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 28, 35, 22],
    '城市': ['北京', '上海', '广州', '北京', '深圳'],
    '薪资': [8000, 12000, 10000, 15000, 7000]
}
df = pd.DataFrame(data)
print("1. 创建的DataFrame:")
print(df)

# 2. 显示前3行
print("\n2. 前3行数据:")
print(df.head(3))

# 3. 筛选出年龄大于25岁的员工
age_filter = df[df['年龄'] > 25]
print("\n3. 年龄大于25岁的员工:")
print(age_filter)

# 4. 筛选出来自北京且薪资大于10000的员工
beijing_high_salary = df[(df['城市'] == '北京') & (df['薪资'] > 10000)]
print("\n4. 来自北京且薪资大于10000的员工:")
print(beijing_high_salary)

# 5. 计算每个城市的平均薪资
city_avg_salary = df.groupby('城市')['薪资'].mean()
print("\n5. 每个城市的平均薪资:")
print(city_avg_salary)

# 6. 按薪资降序排列，并显示薪资最高的员工信息
sorted_df = df.sort_values(by='薪资', ascending=False)
top_employee = sorted_df.iloc[0]
print("\n6. 薪资最高的员工:")
print(f"姓名: {top_employee['姓名']}, 年龄: {top_employee['年龄']}, "
      f"城市: {top_employee['城市']}, 薪资: {top_employee['薪资']}")


# ============================================
# 题目4：数据可视化（30分）
# ============================================

print("\n" + "=" * 50)
print("题目4：数据可视化")
print("=" * 50)

# 参考答案
# 1. 设置matplotlib支持中文显示（兼容Windows和Mac）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 
                                   'STHeiti', 'Arial Unicode MS', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 2. 创建2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 3. 子图1：绘制sin(x)和cos(x)的折线图
x = np.linspace(0, 2*np.pi, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
axes[0, 0].plot(x, np.cos(x), 'r--', label='cos(x)', linewidth=2)
axes[0, 0].set_title('三角函数', fontsize=12)
axes[0, 0].set_xlabel('x', fontsize=10)
axes[0, 0].set_ylabel('y', fontsize=10)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 4. 子图2：柱状图
categories = ['Python', 'Java', 'C++', 'JavaScript', 'Go']
values = [85, 70, 60, 90, 75]
axes[0, 1].bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
axes[0, 1].set_title('编程语言受欢迎程度', fontsize=12)
axes[0, 1].set_xlabel('编程语言', fontsize=10)
axes[0, 1].set_ylabel('得分', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 5. 子图3：饼图
sizes = [30, 25, 20, 15, 10]
labels = ['北京', '上海', '广州', '深圳', '杭州']
axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('城市分布占比', fontsize=12)

# 6. 子图4：散点图
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
axes[1, 1].scatter(x_scatter, y_scatter, c=colors, s=50, alpha=0.6, cmap='viridis')
axes[1, 1].set_title('随机散点图', fontsize=12)
axes[1, 1].set_xlabel('X值', fontsize=10)
axes[1, 1].set_ylabel('Y值', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

# 7. 调整布局并保存
plt.tight_layout()
plt.savefig('作业_可视化结果.png', dpi=150, bbox_inches='tight')
print("可视化图表已保存为: 作业_可视化结果.png")


# ============================================
# 题目5：综合应用（10分）
# ============================================

print("\n" + "=" * 50)
print("题目5：综合应用")
print("=" * 50)

# 参考答案
# 1. 使用NumPy生成100个学生的数学成绩
np.random.seed(42)
scores = np.random.normal(75, 10, 100)
# 确保成绩在0-100范围内
scores = np.clip(scores, 0, 100)
print(f"1. 生成了100个学生的成绩，范围: [{scores.min():.1f}, {scores.max():.1f}]")

# 2. 使用Pandas创建DataFrame
student_df = pd.DataFrame({
    '学生ID': range(1, 101),
    '成绩': scores
})
print("\n2. 前5个学生的成绩:")
print(student_df.head())

# 3. 计算统计信息
avg_score = student_df['成绩'].mean()
max_score = student_df['成绩'].max()
min_score = student_df['成绩'].min()
pass_rate = (student_df['成绩'] >= 60).sum() / len(student_df) * 100

print("\n3. 成绩统计信息:")
print(f"   平均分: {avg_score:.2f}")
print(f"   最高分: {max_score:.2f}")
print(f"   最低分: {min_score:.2f}")
print(f"   及格率: {pass_rate:.2f}%")

# 4. 绘制成绩分布直方图
plt.figure(figsize=(10, 6))
plt.hist(student_df['成绩'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(avg_score, color='red', linestyle='--', linewidth=2, label=f'平均分: {avg_score:.2f}')
plt.xlabel('成绩', fontsize=12)
plt.ylabel('学生人数', fontsize=12)
plt.title('学生数学成绩分布直方图', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('作业_成绩分布.png', dpi=150, bbox_inches='tight')
print("\n4. 成绩分布图已保存为: 作业_成绩分布.png")


print("\n" + "=" * 50)
print("所有题目完成！")
print("=" * 50)
print("\n生成的文件:")
print("  - 作业_可视化结果.png")
print("  - 作业_成绩分布.png")

