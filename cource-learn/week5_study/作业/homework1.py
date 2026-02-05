"""
**目的**：理解独热编码的表示方式及其在语义度量上的局限。

**要求**：

1. 给定词表 `['猫', '狗', '苹果', '香蕉', '学习', '快乐']`，为每个词生成**独热编码**（One-Hot），并保存为一个字典或列表，便于根据词查向量。
2. 实现**余弦相似度**函数：输入两个向量，输出其余弦相似度（取值在 [-1, 1]）。
3. 使用该函数计算**任意两个不同词**在独热编码下的余弦相似度（例如「猫」与「狗」、「苹果」与「香蕉」），并打印结果。
4. 用一句话在注释中说明：为什么独热编码下不同词之间的相似度都是 0？这与“语义相似”有什么关系？

**提示**：独热编码下不同词对应的向量两两正交，点积为 0，因此余弦相似度为 0，无法区分“猫/狗”与“猫/苹果”的语义远近。
"""

import numpy as np

def build_one_hot_vectors(vocabulary):

    vocab_size = len(vocabulary)
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    word_to_vector = {}
    for word in vocabulary:
        # 初始化全 0 向量
        vector = np.zeros(vocab_size)
        # 独热编码：当前位置为 1，其余为 0
        index = word_to_index[word]
        vector[index] = 1.0
        word_to_vector[word] = vector

    return word_to_vector

# 计算两个向量之间的余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def main():
    # 1. 定义词表
    vocabulary = ['猫', '狗', '苹果', '香蕉', '学习', '快乐']

    # 2. 构建独热编码字典
    word_to_vector = build_one_hot_vectors(vocabulary)

    print("=" * 40)
    print("独热编码字典：")
    for word, vector in word_to_vector.items():
        print(f"{word} -> {vector}")

    print("\n余弦相似度计算：")

    # 3. 选择若干“不同词”的词对进行相似度计算
    test_pairs = [
        ('猫', '狗'),
        ('苹果', '香蕉'),
        ('学习', '快乐'),
        ('猫', '苹果'),
    ]

    for w1, w2 in test_pairs:
        v1 = word_to_vector[w1]
        v2 = word_to_vector[w2]
        sim = cosine_similarity(v1, v2)
        print(f"'{w1}' 和 '{w2}' 的相似度：{sim:.4f}")

    # 为什么独热编码下不同词之间的相似度都是 0？这与“语义相似”有什么关系？
    # 在独热编码下，不同词的向量在不同维度上为 1，其余为 0，向量两两正交，点积恒为 0，所以余弦相似度也为 0，无法体现“猫/狗”和“猫/苹果”在语义上的远近差异。


if __name__ == "__main__":
    main()

