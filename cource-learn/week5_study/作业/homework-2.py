"""
## 第2题：Word2Vec 训练与词语相似度

**目的**：掌握 Word2Vec 的基本训练与词语相似度、简单词类比。

**要求**：

1. 使用下面提供的**语料**（或等价的分词后的句子列表），训练一个 **Word2Vec** 模型（**Skip-gram**，如 `sg=1`），建议参数：`vector_size=50, window=3, min_count=1, epochs=100, seed=42`。
2. 训练完成后，对以下词对计算**词语相似度**并打印结果：
   `('猫', '狗')`、`('苹果', '香蕉')`、`('学习', '快乐')`、`('猫', '苹果')`。
   比较语义相近词对与语义较远词对的相似度差异。
3. 实现一个简单的**词类比函数**：给定三个词 `a, b, c`，计算 `vec(b) - vec(a) + vec(c)`，在词表中找到与结果向量**最相似**的词（排除 a、b、c 本身），作为答案并打印。
   用该函数完成一例词类比（例如：若词表中有「国王」「男人」「女人」「王后」等，可做「国王-男人+女人≈？」；否则自选词表中存在的三元组，如「苹果-水果+动物≈？」等）。
"""

from gensim.models import Word2Vec

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
    return sentences


def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,       # Skip-gram
        epochs=100,
        seed=42,
    )
    return model


def show_word_similarities(model):
    pairs = [
        ('猫', '狗'),
        ('苹果', '香蕉'),
        ('学习', '快乐'),
        ('猫', '苹果'),
    ]

    print("=" * 40)
    print("词语相似度：")
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            sim = model.wv.similarity(w1, w2)
            print(f"'{w1}' 和 '{w2}'：{sim:.4f}")
        else:
            print(f"'{w1}' 或 '{w2}' 不在词表中，无法计算相似度。")


def word_analogy(model, a, b, c, topn=5):
    """
    简单词类比：b - a + c ≈ ?

    思路：
    - 使用 gensim 提供的 most_similar(positive=[b, c], negative=[a])
      找到与 b - a + c 向量最接近的词。

    参数：
    - model: 已训练好的 Word2Vec 模型
    - a, b, c: 三个中文词语
    - topn: 检索前 topn 个候选，用于排除 a/b/c 本身

    返回：
    - best_word: 类比结果中最合适的词（排除 a/b/c），若失败返回 None
    """
    # 如果有词不在词表中，直接返回 None
    for w in (a, b, c):
        if w not in model.wv:
            return None

    # 使用 gensim 内置的类比接口
    try:
        candidates = model.wv.most_similar(
            positive=[b, c],
            negative=[a],
            topn=topn
        )
    except KeyError:
        return None

    # 从候选词中选择一个不等于 a/b/c 的词
    exclude = {a, b, c}
    for word, _score in candidates:
        if word not in exclude:
            return word

    return None


def main():
    """
    主函数：训练模型、展示词语相似度，并做一次简单词类比。
    """
    # 1. 构建语料并训练模型
    sentences = build_corpus()
    model = train_word2vec(sentences)

    print("=" * 40)
    print("Word2Vec 模型训练完成")
    print(f"词汇表大小：{len(model.wv.key_to_index)}")

    # 2. 打印若干词对的相似度
    show_word_similarities(model)

    # 3. 做一次示例词类比
    a, b, c = '水果', '苹果', '狗'
    print("\n简单词类比示例：")
    print(f"类比形式：{b} - {a} + {c} ≈ ？")

    result = word_analogy(model, a, b, c)
    if result is not None:
        print(f"模型给出的类比结果：{result}")
    else:
        print("类比失败：可能有词不在词表中，或没有合适的候选。")


if __name__ == "__main__":
    main()
