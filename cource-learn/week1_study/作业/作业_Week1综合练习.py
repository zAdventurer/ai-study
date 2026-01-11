# ============================================
# Week1 综合练习作业
# ============================================
# 作业说明：
# 1. 请完成以下所有题目
# 2. 每个题目都有明确的输出要求
# 3. 请确保代码能够正常运行
# 4. 注意代码规范和注释
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================
# 题目1：Python基础与字符串格式化（10分）
# ============================================
# 要求：
# 1. 创建一个变量存储你的姓名
# 2. 创建一个变量存储你的年龄
# 3. 创建一个变量存储你最喜欢的编程语言
# 4. 使用f-string格式化输出：我是[姓名]，今年[年龄]岁，最喜欢的编程语言是[语言]
# ============================================

print("=" * 50)
print("题目1：Python基础与字符串格式化")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目1

my_name = "Ada"
my_age = 25
my_favorite_language = "JavaScript"
print(f"我是{my_name}，今年{my_age}岁，最喜欢的编程语言是{my_favorite_language}")

# ============================================
# 题目2：NumPy数组操作（20分）
# ============================================
# 要求：
# 1. 创建一个包含10个随机整数（范围1-100）的一维数组
# 2. 计算该数组的最大值、最小值和平均值
# 3. 找出数组中大于50的元素
# 4. 将数组重塑为2行5列的二维数组
# 5. 计算二维数组每行的平均值
# ============================================

print("\n" + "=" * 50)
print("题目2：NumPy数组操作")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目2
random_arr = np.random.randint(1, 101, 10) # np.random.randint(start, stop, size) 生成一个 size 大小的由 [start, stop) 范围内的随机数数组
max_value = random_arr.max() # 获取数组的最大值
min_value = random_arr.min() # 获取数组的最小值
mean_value = random_arr.mean() # 获取数组的平均值
plus_arr = random_arr[random_arr > 50] # 获取数组中大于50的值
reshape_arr = random_arr.reshape(2, 5) # 将数组重塑为2行5列
row_means = reshape_arr.mean(axis=1) # 计算二维数组每行的平均值

print(f'''
生成的随机数数组为：{random_arr}\n
该数组的最大值为{max_value}，最小值为{min_value}，平均值为{mean_value}\n
其中大于50的值为{plus_arr}\n
重塑为2行5列后为\n{reshape_arr}\n
二维数组每行的平均值为{row_means}
''')


# ============================================
# 题目3：Pandas数据处理（30分）
# ============================================
# 要求：
# 1. 创建一个包含以下信息的DataFrame：
#    - 姓名：['张三', '李四', '王五', '赵六', '钱七']
#    - 年龄：[25, 30, 28, 35, 22]
#    - 城市：['北京', '上海', '广州', '北京', '深圳']
#    - 薪资：[8000, 12000, 10000, 15000, 7000]
# 2. 显示DataFrame的基本信息（前3行）
# 3. 筛选出年龄大于25岁的员工
# 4. 筛选出来自北京且薪资大于10000的员工
# 5. 计算每个城市的平均薪资
# 6. 按薪资降序排列，并显示薪资最高的员工信息
# ============================================

print("\n" + "=" * 50)
print("题目3：Pandas数据处理")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目3

info_dict = {
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 28, 35, 22],
    '城市': ['北京', '上海', '广州', '北京', '深圳'],
    '薪资': [8000, 12000, 10000, 15000, 7000]
}
info_df = pd.DataFrame(info_dict) # 创建一个 DataFrame 对象
info_df.head(3) # 显示 DataFrame 的前3行
over25_peoples = info_df[info_df['年龄'] > 25] # 筛选出年龄大于25岁的员工
beijing_peoples = info_df[(info_df['城市'] == '北京') & (info_df['薪资'] > 10000)] # 筛选出来自北京且薪资大于10000的员工
average_wage = info_df.groupby('城市')['薪资'].mean() # 计算每个城市的平均薪资
sort_df = info_df.sort_values(by='薪资', ascending=False) # 按薪资降序排列
top_people = sort_df.head(1) # 显示薪资最高的员工信息
print(f'''
DataFrame的基本信息：
{info_df.head(3)}

年龄大于25岁的员工：
{over25_peoples}

来自北京且薪资大于10000的员工：
{beijing_peoples}

每个城市的平均薪资：
{average_wage}

薪资最高的员工信息：
{top_people}
''')

# ============================================
# 题目4：数据可视化（30分）
# ============================================
# 要求：
# 1. 设置matplotlib支持中文显示（兼容Windows和Mac）
# 2. 创建2x2的子图布局
# 3. 子图1：绘制sin(x)和cos(x)的折线图（x范围0到2π），添加图例和网格
# 4. 子图2：绘制一个包含5个类别的柱状图，类别为['Python', 'Java', 'C++', 'JavaScript', 'Go']，数值为[85, 70, 60, 90, 75]
# 5. 子图3：绘制一个饼图，展示5个城市的占比：[30, 25, 20, 15, 10]，标签为['北京', '上海', '广州', '深圳', '杭州']
# 6. 子图4：绘制一个散点图，x和y都是100个随机数（正态分布），添加颜色映射
# 7. 使用tight_layout调整布局，并保存图片为'作业_可视化结果.png'
# ============================================

print("\n" + "=" * 50)
print("题目4：数据可视化")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目4

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 生成总画布(12*10)，子图大小为2*2

# 子图1：绘制sin(x)和cos(x)的折线图（x范围0到2π），添加图例和网格
points = np.linspace(0, 2*np.pi, 100) # 生成 0~2π 的等差数列，共100个点（默认不写为50）
axes[0, 0].plot(points, np.sin(points), 'b-', label='sin', linewidth=2) # 绘制 0~2π 的 sin 曲线
axes[0, 0].plot(points, np.cos(points), 'r--', label='cos', linewidth=2)
axes[0, 0].set_xlabel('x') # 设置 x 轴标题
axes[0, 0].set_ylabel('sin(x) & cos(x)')
axes[0, 0].set_title('sin(x)和cos(x)的折线图') # 设置子图标题
axes[0, 0].legend() # 显示图例
axes[0, 0].grid(True, alpha=0.3) #设置网格，透明度

# 子图2：绘制一个包含5个类别的柱状图，类别为['Python', 'Java', 'C++', 'JavaScript', 'Go']，数值为[85, 70, 60, 90, 75]
categories = ['Python', 'Java', 'C++', 'JavaScript', 'Go']
values = [85, 70, 60, 90, 75]
axes[0, 1].bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
axes[0, 1].set_ylabel('热度')
axes[0, 1].set_title('各语言热度的柱状图')

# 子图3：绘制一个饼图，展示5个城市的占比：[30, 25, 20, 15, 10]，标签为['北京', '上海', '广州', '深圳', '杭州']
sizes_pie = [30, 25, 20, 15, 10]
labels_pie = ['北京', '上海', '广州', '深圳', '杭州']
# autopct='%1.1f%%' 自动展示百分比。格式为 xx.x%
# startangle=90 设置扇形图的起始角度为 90°
axes[1, 0].pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('城市占比饼图')

# 子图4：绘制一个散点图，x和y都是100个随机数（正态分布），添加颜色映射
x_scatter = np.random.randn(100) #生成100个正态分布的随机数（均值为0，方差为1）
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)
# c=colors：颜色映射。根据 colors 数组中的值（0到1），配合 cmap 参数决定每个点的颜色。
# s=sizes：大小映射。根据 sizes 数组决定每个点的大小（面积）。
# cmap='viridis'：颜色盘。指定使用 Matplotlib 经典的 'viridis' 色系（从紫到黄的渐变）。
scatter_plot = axes[1, 1].scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter_plot, ax=axes[1, 1]) # 显示颜色条
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('100个随机数的散点图')

# 使用tight_layout调整布局，并保存图片为'作业_可视化结果.png'
plt.tight_layout()
plt.savefig('作业_可视化结果.png')
plt.show()

# ============================================
# 题目5：综合应用（10分）
# ============================================
# 要求：
# 1. 使用NumPy生成100个学生的数学成绩（正态分布，均值75，标准差10）
# 2. 使用Pandas创建一个DataFrame，包含学生ID（1-100）和成绩
# 3. 计算成绩的统计信息：平均分、最高分、最低分、及格率（>=60）
# 4. 使用matplotlib绘制成绩分布直方图（bins=20），并标注平均分线
# 5. 保存图表为'作业_成绩分布.png'
# ============================================

print("\n" + "=" * 50)
print("题目5：综合应用")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目5

student_scores = np.random.normal(75, 10, 100) # 生成100个学生的数学成绩（正态分布，均值75，标准差10）
# 如果 np.random.normal(75, 10, 100) 生成的数有大于100的或小于0的，则需要进行处理
student_scores = np.clip(student_scores, 0, 100) # np.clip(a, a_min, a_max) 将数组a中的元素限制在[a_min, a_max]之间
student_df = pd.DataFrame({'学生Id': range(1, 101), '成绩': student_scores})
average_score = student_df['成绩'].mean()
max_score = student_df['成绩'].max()
min_score = student_df['成绩'].min()
pass_rate = student_df[student_df['成绩'] >= 60].shape[0] / student_df.shape[0]
plt.hist(student_scores, bins=20, alpha=0.7, color='blue') # plt.hist(x, bins, alpha, color) 绘制直方图
plt.axvline(average_score, color='green', linestyle='--', label='平均分') # plt.axvline(x, color, linestyle, label) 在x轴上绘制一条垂直于x轴的线
plt.xlabel('成绩')
plt.ylabel('人数')
plt.title('数学成绩分布直方图')
plt.legend()
plt.savefig('作业_成绩分布.png')
plt.show()

print("\n" + "=" * 50)
print("作业完成！")
print("=" * 50)

