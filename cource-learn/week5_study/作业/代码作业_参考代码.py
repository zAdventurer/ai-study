"""
词向量与句向量 - 代码作业参考代码
对应课件第1-30页：独热编码、Word2Vec、句向量与文本相似度
"""

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("词向量与句向量 - 代码作业 参考实现")
print("=" * 60)

# ===================== 第1题：独热编码与余弦相似度 =====================
print("\n【第1题】独热编码与余弦相似度")
print("-" * 40)

vocab = ['猫', '狗', '苹果', '香蕉', '学习', '快乐']
V = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}

# 1.1 独热编码
def one_hot(word):
    vec = np.zeros(V)
    if word in word2idx:
        vec[word2idx[word]] = 1.0
    return vec

one_hot_vectors = {w: one_hot(w) for w in vocab}
print("词表与独热编码维度:", vocab, "-> 向量维度", V)

# 1.2 余弦相似度
def cos_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# 1.3 不同词对的相似度
pairs = [('猫', '狗'), ('苹果', '香蕉'), ('学习', '快乐'), ('猫', '苹果')]
for w1, w2 in pairs:
    sim = cos_sim(one_hot_vectors[w1], one_hot_vectors[w2])
    print(f"  独热编码下 '{w1}' 与 '{w2}' 的余弦相似度: {sim:.4f}")

# 1.4 说明（注释）
# 独热编码下不同词对应的向量两两正交（点积为0），因此余弦相似度均为0。
# 这与“语义相似”矛盾：我们无法用独热编码区分“猫/狗”这种语义相近与“猫/苹果”这种语义较远的词对，因此需要分布式表示（如Word2Vec）。

# ===================== 第2题：Word2Vec 训练与词语相似度 =====================
print("\n【第2题】Word2Vec 训练与词语相似度")
print("-" * 40)

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
sentences = [s.strip().split() for s in corpus]

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,
    epochs=100,
    seed=42,
)
print(f"Word2Vec 训练完成，词汇表大小: {len(model.wv.key_to_index)}")

# 2.2 词语相似度
similarity_pairs = [('猫', '狗'), ('苹果', '香蕉'), ('学习', '快乐'), ('猫', '苹果')]
print("词语相似度（Word2Vec）:")
for w1, w2 in similarity_pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"  '{w1}' vs '{w2}': {sim:.4f}")
    else:
        print(f"  '{w1}' vs '{w2}': (词不在词表中)")

# 2.3 词类比
def word_analogy(a, b, c, model, topn=5):
    """ 词类比: a:b ≈ c:? ，返回与 vec(b)-vec(a)+vec(c) 最相似的词（排除 a,b,c）"""
    if a not in model.wv or b not in model.wv or c not in model.wv:
        return None
    va, vb, vc = model.wv[a], model.wv[b], model.wv[c]
    target = vb - va + vc
    # 取最相似的 topn+3 个，再排除 a,b,c
    sims = model.wv.similar_by_vector(target, topn=topn + 3)
    result = [w for w, _ in sims if w not in (a, b, c)][:topn]
    return result[0] if result else None

# 词表中没有“国王/王后”等，用现有词做类比示例
analogy_triples = [('猫', '狗', '苹果'), ('学习', '快乐', '编程')]
print("词类比示例:")
for a, b, c in analogy_triples:
    ans = word_analogy(a, b, c, model)
    if ans is not None:
        print(f"  {a} : {b} ≈ {c} : ?  ->  {ans}")
    else:
        print(f"  {a} : {b} ≈ {c} : ?  ->  (词不在词表中)")

# ===================== 第3题：句向量与文本相似度 =====================
print("\n【第3题】句向量与文本相似度")
print("-" * 40)

def sentence_vector(words, model):
    """平均池化：句中在词表中的词的词向量取平均；无有效词则返回全0向量"""
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return np.zeros(model.wv.vector_size)
    return np.mean(vectors, axis=0)

# 3.2 前3句话的句向量与相似度矩阵
first_three = [sentences[0], sentences[1], sentences[2]]
sent_vecs = np.array([sentence_vector(s, model) for s in first_three])
# 相似度矩阵（3x3）
sim_matrix = cosine_similarity(sent_vecs, sent_vecs)
print("前3句话的句向量余弦相似度矩阵:")
print("  句子0:", " ".join(first_three[0]))
print("  句子1:", " ".join(first_three[1]))
print("  句子2:", " ".join(first_three[2]))
print("  相似度矩阵:\n", np.round(sim_matrix, 4))

# 3.3 两句新句子的相似度
new_s1 = "我 爱 小猫 和 小狗".split()
new_s2 = "苹果 香蕉 都 很 好吃".split()
v1 = sentence_vector(new_s1, model)
v2 = sentence_vector(new_s2, model)
sim_new = cosine_similarity([v1], [v2])[0, 0]
print("两句新句子:")
print("  句子A:", " ".join(new_s1))
print("  句子B:", " ".join(new_s2))
print("  余弦相似度:", round(sim_new, 4))

# 平均池化将句子表示为“词向量的平均”，语义相近的句子会包含相近的词或同主题词，
# 其平均向量在空间中方向更接近，因此余弦相似度可以衡量两句子的语义相近程度。

# ===================== 加分项：PCA 可视化（可选） =====================
print("\n【加分项】词向量 PCA 可视化（可选）")
print("-" * 40)

try:
    import matplotlib
    matplotlib.use("Agg")
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_words = [w for w in ["猫", "狗", "苹果", "香蕉", "学习", "快乐", "编程", "水果"] if w in model.wv]
    if len(plot_words) >= 4:
        X = np.array([model.wv[w] for w in plot_words])
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_2d[:, 0], X_2d[:, 1])
        for i, w in enumerate(plot_words):
            plt.annotate(w, (X_2d[i, 0], X_2d[i, 1]))
        plt.title("Word2Vec 词向量 PCA 二维可视化")
        plt.tight_layout()
        plt.savefig("word_vectors_pca.png", dpi=100)
        plt.close()
        print("  已保存 PCA 图: word_vectors_pca.png")
    else:
        print("  词表不足，跳过 PCA 图")
except Exception as e:
    print("  跳过可视化:", e)

print("\n" + "=" * 60)
print("参考代码运行完毕")
print("=" * 60)
