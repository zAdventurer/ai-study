"""
目标：理解独热编码的工作原理和局限性
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

print("="*50)
print("第一节：独热编码 (One-Hot Encoding)")
print("="*50)

# 1. 创建词汇表
vocabulary = ['猫', '狗', '苹果', '汽车', '学习', '快乐']
word_to_index = {word: i for i, word in enumerate(vocabulary)}

print(f"词汇表：{vocabulary}")
print(f"词汇表大小：{len(vocabulary)}")

# 2. 独热编码函数
def one_hot_encode(word, vocab_size):
    """将单词转换为独热编码向量"""
    vector = np.zeros(vocab_size)
    if word in word_to_index:
        index = word_to_index[word]
        vector[index] = 1
    return vector

# 3. 演示独热编码
print(f"\n独热编码演示：")
test_words = ['猫', '狗', '苹果']
vectors = {}

for word in test_words:
    vector = one_hot_encode(word, len(vocabulary))
    vectors[word] = vector
    print(f"'{word}': {vector}")

# 4. 计算相似度
def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# 5. 相似度问题演示
cat_vector = vectors['猫']
dog_vector = vectors['狗']
apple_vector = vectors['苹果']

sim_cat_dog = cosine_similarity(cat_vector, dog_vector)
sim_cat_apple = cosine_similarity(cat_vector, apple_vector)

print(f"\n相似度计算：")
print(f"'猫' 和 '狗' 的相似度：{sim_cat_dog}")
print(f"'猫' 和 '苹果' 的相似度：{sim_cat_apple}")
print("❌ 问题：语义相近的词相似度也是0！")

# 6. 存储问题演示
print(f"\n存储问题：")
large_vocab_sizes = [1000, 10000, 100000]

for vocab_size in large_vocab_sizes:
    memory_mb = vocab_size * 4 / (1024 * 1024)  # 4字节浮点数
    sparsity = (vocab_size - 1) / vocab_size * 100
    print(f"词汇表{vocab_size:,}: {memory_mb:.2f}MB/词, 稀疏度{sparsity:.1f}%")

# 7. 可视化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('独热编码向量可视化')

for i, word in enumerate(test_words):
    vector = vectors[word]
    axes[i].bar(range(len(vector)), vector)
    axes[i].set_title(f"'{word}'")
    axes[i].set_ylim(0, 1.2)
    
    # 标注对应位置
    for j, vocab_word in enumerate(vocabulary):
        if vector[j] == 1:
            axes[i].text(j, 1.05, vocab_word, ha='center')

plt.tight_layout()
plt.show()

# 8. 总结
print(f"\n" + "="*50)
print("独热编码总结")
print("="*50)
print("✅ 优点：简单直观，每个词有唯一表示")
print("❌ 缺点：")
print("   1. 高维稀疏，浪费存储空间")
print("   2. 无法表达语义关系")
print("   3. 维度随词汇表增长")

print(f"\n🎯 这就是我们需要分布式词向量的原因！")
print("   下一节学习Word2Vec解决这些问题")

# 9. 简单对比演示
print(f"\n对比展示：")
print("独热编码：高维稀疏，无语义")
print("理想词向量：低维稠密，有语义")

# 模拟理想词向量（随机生成，仅作演示）
np.random.seed(42)
ideal_cat = np.random.rand(5)
ideal_dog = np.random.rand(5) + 0.3 * ideal_cat  # 模拟相似性
ideal_apple = np.random.rand(5)

ideal_sim_cat_dog = cosine_similarity(ideal_cat, ideal_dog)
ideal_sim_cat_apple = cosine_similarity(ideal_cat, ideal_apple)

print(f"\n假设我们有5维的理想词向量：")
print(f"'猫' 和 '狗' 相似度：{ideal_sim_cat_dog:.3f}")
print(f"'猫' 和 '苹果' 相似度：{ideal_sim_cat_apple:.3f}")
print("✅ 这样就能体现语义关系了！") 