"""
第4题：附加练习（可选）

- **可视化**：对第2题中的词向量做 **PCA 降维**（如 2 维），将至少 6 个词在二维平面上画成散点图，并标注词标签。观察语义相近的词是否在图中更接近。
- **GloVe 对比**：若已下载 `glove.6B.100d.txt`（见课程 GloVe 说明文档），加载该预训练向量，对其中若干英文词计算相似度或词类比，与第2题的中文 Word2Vec 结果做简要对比说明（可写在注释或打印中）。
"""

import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

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
    return sentences

# 训练 Word2Vec 模型
def train_word2vec(sentences):
    """
    训练 Word2Vec 模型。
    """
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

# 选择一些语义上有关系的词，使用PCA将词向量降维到二维并可视化
def visualize_words(model):
    # 尝试选择一些语义上有关系的词
    words_to_plot = [
        '猫', '狗', '宠物',
        '苹果', '香蕉', '水果',
        '学习', '编程', '快乐', '心情',
    ]

    # 过滤掉不在词表中的词
    available_words = [w for w in words_to_plot if w in model.wv]
    if len(available_words) < 2:
        print("可视化的词太少，无法绘图。")
        return

    vectors = [model.wv[w] for w in available_words]

    # PCA 降维到 2 维
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.8)

    # 在点附近标注词语
    for i, word in enumerate(available_words):
        plt.annotate(
            word,
            (vectors_2d[i, 0], vectors_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Word2Vec 词向量二维可视化（PCA）")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 训练模型并可视化若干词向量
def main():
    sentences = build_corpus()
    model = train_word2vec(sentences)

    visualize_words(model)


if __name__ == "__main__":
    main()
