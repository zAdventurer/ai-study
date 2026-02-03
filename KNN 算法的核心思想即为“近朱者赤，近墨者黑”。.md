# KNN 算法简易实现

KNN 算法的核心思想即为“近朱者赤，近墨者黑”。

而如何判断所谓的近，则成了关键。

K 值超参数的选取也是重中之重，因为它代表着取该点周围最近的 K 个点进行判断分类，为了避免分类的大小一致，我们一般选取基数作为 K 值。

```python
import numpy as np # 导入 numpy 库并重命名为 np
import matplotlib.pyplot as plt #导入 matplotlib 库中的 pyplot 包并重命名为 plt】

# 定义样本特征
data_X = [
    [1.1, 6],
    [3.7, 4],
    [4.2, 2],
    [5, 3.5],
    [2, 8],
    [5.3, 6.9],
    [6.6, 4.4 ],
    [8.1, 8],
    [9, 2.5]
]

# 定义样本的标记
data_y = [0,0,0,0,1,1,1,1,1]

# 定义训练集
train_X = np.array(data_X)
train_y = np.array(data_y)

# 定义新的样本点  
data_new = np.array([4,5])

# 定义 type 为 0 的点为红色的 x
plt.scatter(train_X[train_y==0,0], train_X[train_y==0,1], color='red', marker='x')
# 定义 type 为 1 的点为黑色的 o
plt.scatter(train_X[train_y==1,0], train_X[train_y==1,1],color='black', marker='o')
# 定义新的样本点为蓝色的三角形
plt.scatter(data_new[0], data_new[1],color='blue', marker='^')
plt.show() # 展示图片

'''--------------------上述为样本定义------------------------'''

# KNN 预测的过程
# 1. 计算新样本点与已知样本点的距离，这里我们采用欧式举例，可以解开注释打印看看具体的距离
# for data in train_X:
#     print(np.sqrt(np.sum((data - data_new)**2)))
distances = [np.sqrt(np.sum((data - data_new)**2)) for data in train_X]

# 2. 按照距离进行索引的排序
sort_index = np.argsort(distances)

# 3. 超参数 K 值的选取
K = 5

# 4. “代表公投~”，选出对应的类型
first_K =  [train_y[i] for i in sort_index[:K]]

# 5. 统计投票
count_0 = first_K.count(0) # 数数有几个 0
count_1 = first_K.count(1) # 数数有几个 1

# 6. 打印类型
predict_data_type = '0' if count_0 > count_1 else '1'
print(predict_data_type)
```