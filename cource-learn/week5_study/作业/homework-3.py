"""
## 第3题：句向量与文本相似度

**目的**：掌握由词向量得到句向量的方法，以及用余弦相似度衡量文本相似度。

**要求**：

1. 使用**第2题**训练好的 Word2Vec 模型（或重新训练同一语料），实现**句向量**计算函数：输入为分词后的词列表，输出为定长向量。要求采用**平均池化**：对句中所有在词表中的词的词向量取平均；若某词不在词表中则跳过；若整句无有效词，则返回全 0 向量。
2. 对语料中的**前 3 句话**分别计算句向量，并两两计算**余弦相似度**，打印 3×3 的相似度矩阵（或等价形式）。
3. 再任选**两句新句子**（需分词成列表），计算它们的句向量及这两句之间的余弦相似度，并打印结果。用一句话在注释中说明：为什么用“平均池化 + 余弦相似度”可以衡量两句子的语义相近程度？

**提示**：句向量用 `sklearn.metrics.pairwise.cosine_similarity` 或自己实现的余弦相似度均可；若使用 sklearn，注意输入为二维数组（如 `[[v1], [v2]]`）。
"""

import numpy as np
from gensim.models import Word2Vec

# 构建语料
def build_corpus():
    corpus = [
        "我 喜欢 猫 和 狗",
        "猫 是 可爱 的 宠物",
        "狗 很 忠诚 聪明",
        "苹果 是 健康 的 水果",
        "我 每天 吃 苹果 香蕉",
        "水果 含有 维生素",
        "我 喜欢 学习 编程",
        "学习 使人 快乐 进步",
        "编程 是 有趣 的 技能",
        "今天 我 很 快乐",
        "快乐 的 心情 重要",
    ]
    sentences = [sentence.split() for sentence in corpus]
    return corpus, sentences

# 训练 Word2Vec 模型
def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,
        epochs=100,
        seed=42,
    )
    return model

# 使用“平均池化”计算句向量
def sentence_vector(words, model):
    vectors = []
    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])

    if len(vectors) == 0:
        return np.zeros(model.wv.vector_size, dtype=float)

    return np.mean(vectors, axis=0)

# 计算两个向量的余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def main():
    # 1. 构建语料并训练模型
    corpus, sentences = build_corpus()
    model = train_word2vec(sentences)

    print("=" * 50)
    print("句向量与文本相似度")

    # 2. 取语料中前 3 句，计算句向量
    first_three_sentences = corpus[:3]
    first_three_tokens = sentences[:3]

    vectors = []
    for tokens in first_three_tokens:
        vec = sentence_vector(tokens, model)
        vectors.append(vec)

    vectors = np.stack(vectors, axis=0)

    # 3. 构造 3x3 相似度矩阵
    sim_matrix = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            sim_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])

    print("\n语料前 3 句：")
    for idx, sent in enumerate(first_three_sentences):
        print(f"{idx}: {sent}")

    print("\n前 3 句之间的相似度矩阵：")
    for i in range(3):
        row_str = "  ".join(f"{sim_matrix[i, j]:.4f}" for j in range(3))
        print(row_str)

    # 4. 任意两句新句子，计算句向量和相似度
    new_sent1 = "我 喜欢 狗"
    new_sent2 = "今天 心情 快乐"

    words1 = new_sent1.split()
    words2 = new_sent2.split()

    vec1 = sentence_vector(words1, model)
    vec2 = sentence_vector(words2, model)

    sim_new = cosine_similarity(vec1, vec2)

    print("\n两句新句子：")
    print(f"句1：{new_sent1}")
    print(f"句2：{new_sent2}")
    print(f"句1 与 句2 的相似度：{sim_new:.4f}")

    # 为什么用“平均池化 + 余弦相似度”可以衡量两句子的语义相近程度？
    # 平均池化可以求出句子中所有词向量的平均值，这个平均值可以代表句子的语义信息，余弦相似度可以比较两个向量的方向是否接近，从而衡量两句在语义上的相似程度。


if __name__ == "__main__":
    main()
