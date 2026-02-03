"""
第四节：综合实践 - 文本分析系统 - 简洁版
目标：整合所有技术，构建完整的文本分析系统
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import jieba
import re

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

print("="*60)
print("第四节：综合实践 - 文本分析系统")
print("="*60)

# 1. 构建数据集
print("1. 构建数据集")
text_data = {
    '科技': ["人工智能技术快速发展", "机器学习图像识别进展", "自然语言处理技术", "云计算大数据技术"],
    '教育': ["在线教育平台便利", "教师培养创新思维", "课堂结合教育技术", "学生综合素质发展"],
    '健康': ["健康饮食习惯重要", "定期锻炼增强免疫力", "充足睡眠大脑恢复", "心理健康身体健康"],
    '环境': ["环境保护人人责任", "可再生能源减少排放", "垃圾分类环保措施", "森林保护生态平衡"]
}

# 构建训练数据
documents = []
labels = []
for category, texts in text_data.items():
    for text in texts:
        documents.append(text)
        labels.append(category)

print(f"数据集：{len(documents)}个文档，{len(set(labels))}个类别")

# 2. 文本预处理
print("\n2. 文本预处理")
stop_words = set(['的', '了', '在', '是', '我', '有', '和', '很', '要', '也'])

def preprocess_text(text):
    """文本预处理"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    words = jieba.cut(text)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return words

processed_documents = [preprocess_text(doc) for doc in documents]
print("预处理完成，示例：")
print(f"原文：{documents[0]}")
print(f"处理后：{' '.join(processed_documents[0])}")

# 3. 训练Word2Vec模型
print("\n3. 训练Word2Vec模型")
model = Word2Vec(
    sentences=processed_documents,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,
    epochs=50,
    seed=42
)
print(f"✅ 模型训练完成，词汇表：{len(model.wv.key_to_index)}个词")

# 4. 计算文档向量
print("\n4. 计算文档向量")
def document_vector(doc_words, model):
    """计算文档向量（平均池化）"""
    vectors = [model.wv[word] for word in doc_words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.wv.vector_size)

doc_vectors = np.array([document_vector(doc, model) for doc in processed_documents])
print(f"文档向量矩阵：{doc_vectors.shape}")

# 5. 文本聚类
print("\n5. 文本聚类")
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(doc_vectors)

print("聚类结果分析：")
for i in range(4):
    cluster_docs = [j for j, label in enumerate(cluster_labels) if label == i]
    cluster_categories = [labels[j] for j in cluster_docs]
    main_category = max(set(cluster_categories), key=cluster_categories.count)
    print(f"聚类{i}: {len(cluster_docs)}个文档, 主要类别: {main_category}")

# 6. 文本分类
print("\n6. 文本分类")
label_map = {label: i for i, label in enumerate(set(labels))}
numeric_labels = [label_map[label] for label in labels]

X_train, X_test, y_train, y_test = train_test_split(
    doc_vectors, numeric_labels, test_size=0.3, random_state=42
)

classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"分类准确率：{accuracy:.3f}")

# 7. 可视化结果
print("\n7. 可视化结果")
pca = PCA(n_components=2)
doc_vectors_2d = pca.fit_transform(doc_vectors)

# 创建颜色映射
colors = {'科技': 'red', '教育': 'blue', '健康': 'green', '环境': 'orange'}
point_colors = [colors[label] for label in labels]

plt.figure(figsize=(12, 4))

# 真实类别分布
plt.subplot(1, 2, 1)
plt.scatter(doc_vectors_2d[:, 0], doc_vectors_2d[:, 1], c=point_colors, alpha=0.7)
plt.title('真实类别分布')
for category, color in colors.items():
    plt.scatter([], [], c=color, label=category)
plt.legend()

# 聚类结果
plt.subplot(1, 2, 2)
plt.scatter(doc_vectors_2d[:, 0], doc_vectors_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('聚类结果')

plt.tight_layout()
plt.show()

# 8. 新文本分析功能
print("\n8. 新文本分析")
def analyze_text(text, model, classifier, label_map):
    """分析新文本"""
    processed = preprocess_text(text)
    vector = document_vector(processed, model)
    
    # 预测类别
    pred_num = classifier.predict([vector])[0]
    reverse_map = {v: k for k, v in label_map.items()}
    predicted_category = reverse_map[pred_num]
    
    # 预测概率
    proba = classifier.predict_proba([vector])[0]
    
    print(f"📄 文本：{text}")
    print(f"🎯 预测类别：{predicted_category}")
    print(f"📊 各类别概率：")
    for i, prob in enumerate(proba):
        category = reverse_map[i]
        print(f"   {category}: {prob:.3f}")
    print()

# 测试新文本
test_texts = [
    "深度学习神经网络图像处理应用",
    "学校重视学生身心健康发展",
    "气候变化全球生态环境影响"
]

print("新文本分析结果：")
for text in test_texts:
    analyze_text(text, model, classifier, label_map)