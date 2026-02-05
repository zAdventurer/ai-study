"""
第二节：Word2Vec训练与应用 - 简洁版
目标：掌握Word2Vec的基本训练和应用
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

print("="*50)
print("第二节：Word2Vec训练与应用")
print("="*50)

# 1. 准备训练数据
print("1. 准备训练语料")
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
    "快乐 的 心情 重要"
]

sentences = [sentence.split() for sentence in corpus]
print(f"语料库：{len(sentences)}个句子")

# 2. 训练Word2Vec模型
print(f"\n2. 训练Word2Vec模型")
model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,  # Skip-gram
    epochs=100,
    seed=42
)

print(f"✅ 模型训练完成！")
print(f"词汇表大小：{len(model.wv.key_to_index)}")

# 3. 词语相似度计算
print(f"\n3. 词语相似度")
similar_pairs = [
    ('猫', '狗'),
    ('苹果', '香蕉'),
    ('学习', '编程'),
    ('快乐', '开心')
]

print("语义相近词的相似度：")
for word1, word2 in similar_pairs:
    if word1 in model.wv and word2 in model.wv:
        similarity = model.wv.similarity(word1, word2)
        print(f"  '{word1}' 和 '{word2}': {similarity:.4f}")

different_pairs = [('猫', '苹果'), ('狗', '编程')]
print("语义不同词的相似度：")
for word1, word2 in different_pairs:
    if word1 in model.wv and word2 in model.wv:
        similarity = model.wv.similarity(word1, word2)
        print(f"  '{word1}' 和 '{word2}': {similarity:.4f}")

# 4. 寻找相似词
print(f"\n4. 寻找相似词")
query_words = ['猫', '苹果', '学习']
for word in query_words:
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=2)
        print(f"与'{word}'最相似：{[w[0] for w in similar_words]}")

# 5. 词向量可视化
print(f"\n5. 词向量可视化")
words_to_plot = ['猫', '狗', '宠物', '苹果', '香蕉', '水果', 
                 '学习', '编程', '快乐', '开心']

available_words = [word for word in words_to_plot if word in model.wv]
print(f"可视化词语：{available_words}")

if len(available_words) > 5:
    # 获取词向量
    vectors = [model.wv[word] for word in available_words]
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.7)
    
    # 添加标签
    for i, word in enumerate(available_words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Word2Vec词向量二维可视化')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.grid(True, alpha=0.3)
    plt.show()

# 6. 模型保存和加载
print(f"\n6. 模型保存")
model_path = "word2vec_simple.model"
model.save(model_path)
print(f"✅ 模型保存到：{model_path}")

# 验证加载
loaded_model = Word2Vec.load(model_path)
if '猫' in loaded_model.wv:
    print(f"验证：'猫'向量前3维 = {loaded_model.wv['猫'][:3].round(3)}")

# 7. 总结对比
print(f"\n" + "="*50)
print("Word2Vec vs 独热编码对比")
print("="*50)

# 与独热编码对比
print("Word2Vec优势：")
print("✅ 低维稠密：50维 vs 词汇表维度")  
print("✅ 捕获语义：相似词有相似向量")
print("✅ 无监督学习：从大量文本自动学习")

print(f"\n实际效果验证：")
if '猫' in model.wv and '狗' in model.wv and '苹果' in model.wv:
    cat_dog_sim = model.wv.similarity('猫', '狗')
    cat_apple_sim = model.wv.similarity('猫', '苹果')
    print(f"'猫'和'狗'相似度：{cat_dog_sim:.3f} (应该较高)")
    print(f"'猫'和'苹果'相似度：{cat_apple_sim:.3f} (应该较低)")
