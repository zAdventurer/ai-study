# 作业

1、创建一个长度为10的一维全为0的ndarray对象，然后让第5个元素等于1

```Python
import numpy as np

arr = np.zeros(10,dtype = 'int')

# 注意第5个元素对应索引是4
arr[4] = 1
arr
```

2、创建一个元素为从10到49（包含49）的ndarray对象，间隔是1

```Python
import numpy as np

# 注意，包含49，所以范围给的是10 ~ 50，可以取到10，但是取不到50
arr = np.arange(10,50)
arr
```

3、将第2题的所有元素位置反转

```Python
import numpy as np

# 注意，包含49，所以范围给的是10 ~ 50，可以取到10，但是取不到50
arr = np.arange(10,50)

# 注意::-1表示切片颠倒，数据倒过来了
arr = arr[::-1]
arr
```

4、使用np.random.random创建一个10*10的ndarray对象，并打印出最大最小元素

```Python
import numpy as np

# 生成0~1之间的随机小数
arr = np.random.random(size = (10,10))

# 考察函数使用
print('输出最大值：',arr.max())
print('输出最小值：',arr.min())
```

5、创建一个10*10的ndarray对象，且矩阵边界全为1，里面全为0

```Python
import numpy as np

arr = np.full(shape = (10,10),fill_value=0,dtype=np.int)

# 第一行和最后一行重新赋值为：1
arr[[0,-1]] = 1

# 第一列和最后一列重新赋值为：1
arr[:,[0,-1]] = 1
arr
```

6、创建一个每一行都是从0到4的5*5矩阵

```Python
import numpy as np

# 创建的形状5*5全部是0的数组
arr = np.zeros((5,5))

# 进行赋值操作，值是0~4
arr += np.arange(5)
# 输出显示
print(arr)
```

7、创建一个范围在(0,1)之间的长度为12的等差数列，创建[1,    2,    4,    8,   16,   32,   64,  128,  256,  512, 1024]等比数列。

```Python
import numpy as np

# 等差数列
arr = np.linspace(0,1,12)

display(arr)

# 等比数列，base = 2表示2的倍数
arr2 = np.logspace(0,10,base=2,num = 11,dtype='int')
arr2
```

8、创建一个长度为10的正太分布数组np.random.randn并排序

```Python
import numpy as np

# 创建随机数组
arr = np.random.randn(10)
print('未排序：',arr)

# 注意调用np.sort()方法，要接收一下数据
# 原来的arr并没有改变
arr2 = np.sort(arr)
print('排序后：',arr2)
```

9、创建一个长度为10的随机数组并将最大值替换为-100

```Python
import numpy as np

# 设置不显示科学计数法，默认情况显示情况是：3.22449e-01
np.set_printoptions(suppress = True)

arr = np.random.random(10)
print('原数据：\n',arr)

# 找到最大值
v = arr.max()
# 进行条件判断
cond = arr == v
# 根据条件，赋值
arr[cond] = -100
print('修改之后数据：\n',arr)
```

10、如何根据第3列大小顺序来对一个5*5矩阵排序？（考察argsort()方法）

```Python
np.random.seed(10)
arr = np.random.randint(0,10,(5,5))
print('原始数据：\n',arr)

# 查看第三列数据，并获取索引排序
print('第三列数据：', arr[:,2])
# argsort函数返回的是数组值从小到大的索引值
print('第三列数据的排序：',np.argsort(arr[:,2]))

# 根据第三列索引排序，进行重新索引
print('根据第三列大小排序：\n',arr[np.argsort(arr[:,2])])
```







