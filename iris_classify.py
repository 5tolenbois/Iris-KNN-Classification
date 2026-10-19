import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ========== 路径配置，和你其他项目保持一致 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(BASE_DIR, "result")
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# 中文设置
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 加载鸢尾花数据集，保存csv =====================
iris = load_iris(as_frame=True)
df = iris.frame
# 保存原始数据集到data文件夹
csv_save_path = os.path.join(data_dir, "iris_dataset.csv")
df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

print("鸢尾花数据集前5行：")
print(df.head())
print(f"\n数据集已保存至 {csv_save_path}")
print("特征：萼片长、萼片宽、花瓣长、花瓣宽")
print("标签：0=山鸢尾setosa，1=变色鸢尾versicolor，2=维吉尼亚鸢尾virginica")

# ===================== 2. 划分训练集、测试集 8:2 =====================
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集：{X_train.shape}, 测试集：{X_test.shape}")

# ===================== 3. KNN分类模型训练 =====================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 模型评估
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\n====模型评估结果====")
print(f"测试集准确率Accuracy：{acc:.2f}")
print("混淆矩阵：")
print(cm)

# ===================== 4. 可视化：花瓣长、花瓣宽散点图，保存图片 =====================
plt.figure(figsize=(8,6))
scatter = plt.scatter(df["petal length (cm)"], df["petal width (cm)"], c=df["target"], cmap="viridis")
plt.xlabel("花瓣长度(cm)")
plt.ylabel("花瓣宽度(cm)")
plt.title("鸢尾花：花瓣长宽分布（3个品种）")
plt.legend(*scatter.legend_elements(), title="品种")
plt.grid(alpha=0.3)
plt.tight_layout()

img_path = os.path.join(result_dir, "iris_scatter.png")
plt.savefig(img_path, dpi=150)
plt.show()
print(f"\n可视化图片保存到: {img_path}")
