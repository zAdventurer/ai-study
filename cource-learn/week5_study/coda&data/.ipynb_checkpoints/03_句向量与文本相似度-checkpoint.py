"""
第三节：句向量与文本相似度 - 简洁版
目标：掌握句向量计算和文本相似度应用
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import jieba

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

print("="*50)
print("第三节：句向量与文本相似度")
print("="*50)

# 1. 准备Word2Vec模型
print("1. 准备Word2Vec模型")
corpus = [
    "我 喜欢 猫 和 狗", "猫 是 可爱 的 宠物", "狗 很 忠诚 聪明",
    "苹果 是 健康 的 水果", "我 吃 苹果 香蕉", "水果 含有 维生素",
    "我 喜欢 学习 编程", "学习 使人 快乐", "编程 是 有趣 技能",
    "今天 我 很 快乐", "快乐 心情 重要", "学习 让我 开心"
]

sentences = [sentence.split() for sentence in corpus]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=100, seed=42)
print(f"✅ 模型训练完成，词汇表：{len(model.wv.key_to_index)}个词")

# 2. 句向量计算 - 平均池化
print(f"\n2. 句向量计算 (平均池化)")

def sentence_vector(sentence_words, model):
    """使用平均池化计算句向量"""
    vectors = []
    for word in sentence_words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        return np.zeros(model.wv.vector_size)
    
    return np.mean(vectors, axis=0)

# 测试句向量计算
test_sentences = [
    "我 喜欢 小猫",
    "苹果 很 好吃", 
    "学习 编程 有趣",
    "今天 心情 好"
]

print("句向量计算示例：")
sentence_vectors = {}
for sent in test_sentences:
    words = sent.split()
    vector = sentence_vector(words, model)
    sentence_vectors[sent] = vector
    print(f"'{sent}' -> 向量维度:{vector.shape}, 前3维:{vector[:3].round(3)}")

# 3. 句子相似度计算
print(f"\n3. 句子相似度计算")

def calculate_similarity(sent1, sent2, model):
    """计算两个句子的余弦相似度"""
    words1 = sent1.split()
    words2 = sent2.split()
    
    vec1 = sentence_vector(words1, model)
    vec2 = sentence_vector(words2, model)
    
    # 余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

# 测试相似句子
similar_pairs = [
    ("我 喜欢 小猫", "我 喜欢 小狗"),
    ("苹果 很 好吃", "香蕉 很 甜"),
    ("学习 编程", "编程 学习"),
    ("今天 心情 好", "今天 很 开心")
]

print("相似句子的相似度：")
for sent1, sent2 in similar_pairs:
    sim = calculate_similarity(sent1, sent2, model)
    print(f"'{sent1}' 和 '{sent2}': {sim:.4f}")

# 不相似句子
different_pairs = [
    ("我 喜欢 小猫", "苹果 很 好吃"),
    ("学习 编程", "快乐 心情")
]

print("不相似句子的相似度：")
for sent1, sent2 in different_pairs:
    sim = calculate_similarity(sent1, sent2, model)
    print(f"'{sent1}' 和 '{sent2}': {sim:.4f}")

# 4. 简单问答系统
print(f"\n4. 简单问答系统")

# 构建问答知识库
qa_database = [
    ("什么是宠物", "宠物是人类饲养的动物比如猫和狗"),
    ("什么水果好", "苹果和香蕉都很好吃有营养"),
    ("如何学习", "学习需要耐心和练习"),
    ("怎样开心", "学习新知识让人快乐")
]

def find_answer(question, qa_database, model):
    """根据问题找最相似的答案"""
    question_words = " ".join(jieba.cut(question))
    best_similarity = -1
    best_answer = "抱歉，没有找到相关答案"
    
    for qa_q, answer in qa_database:
        qa_q_words = " ".join(jieba.cut(qa_q))
        sim = calculate_similarity(question_words, qa_q_words, model)
        
        if sim > best_similarity:
            best_similarity = sim
            best_answer = answer
    
    return best_answer, best_similarity

# 测试问答
test_questions = ["猫狗是什么", "吃什么水果好", "怎么学习"]

print("问答系统测试：")
for q in test_questions:
    answer, sim = find_answer(q, qa_database, model)
    print(f"问题：{q}")
    print(f"答案：{answer} (相似度：{sim:.3f})")
    print()

# 5. 句向量可视化
print("5. 句向量可视化")

# 选择代表性句子
sentences_to_plot = [
    "我 喜欢 猫",
    "苹果 很 好吃",
    "学习 很 有趣", 
    "今天 很 快乐",
    "宠物 很 可爱",
    "水果 有 营养"
]

# 计算句向量
sent_vectors = []
for sent in sentences_to_plot:
    vector = sentence_vector(sent.split(), model)
    sent_vectors.append(vector)

# PCA降维可视化
if len(sent_vectors) > 0:
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(sent_vectors)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=120, alpha=0.7)
    
    # 添加句子标签
    for i, sent in enumerate(sentences_to_plot):
        plt.annotate(sent, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('句向量二维可视化 (平均池化 + PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.grid(True, alpha=0.3)
    plt.show()
