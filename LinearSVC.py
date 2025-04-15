import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def augment_positive_samples(X, y, n_aug=5, noise_std=0.01):
    """
    对每个正样本（标签为 -1）生成 n_aug 个扩增样本，
    并在每个特征上添加高斯噪声，noise_std 为噪声标准差（需要根据数据尺度调整）。
    """
    pos_idx = np.where(y == -1)[0]
    X_pos = X[pos_idx]
    augmented_samples = []
    augmented_labels = []

    for sample in X_pos:
        for _ in range(n_aug):
            noise = np.random.normal(loc=0, scale=noise_std, size=sample.shape)
            augmented_samples.append(sample + noise)
            augmented_labels.append(-1)

    if augmented_samples:
        X_aug = np.vstack(augmented_samples)
        y_aug = np.array(augmented_labels)
        # 将扩增样本与原始样本合并
        X_new = np.vstack((X, X_aug))
        y_new = np.hstack((y, y_aug))
    else:
        X_new, y_new = X, y

    return X_new, y_new


# 读取标注后的特征数据
# df = pd.read_csv("feature_data_labeled_expanded_3194.csv")
df = pd.read_csv("2025-04-09/feature_data_labeled_span_1608.csv")
feature_cols = [
    "meanx", "meany", "meanz",
    "sdx",   "sdy",   "sdz",
    "meanxg", "meanyg", "meanzg",
    "sdxg",  "sdyg",  "sdzg"
]
X = df[feature_cols].values
y = df["label"].values  # -1 表示坑洼, 1 表示正常

# 对正样本扩增：例如每个正样本扩增5个样本，噪声标准差根据数据尺度调整
X_aug, y_aug = augment_positive_samples(X, y, n_aug=5, noise_std=0.02)
print("扩增后数据量:", X_aug.shape[0])

# 划分数据（这里你可以直接用全部数据训练）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 LinearSVC 训练
clf = LinearSVC(C=1.0, max_iter=10000, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Test Accuracy: {:.4f}".format(accuracy))

# 获取模型参数：w 和 b
w = clf.coef_[0]      # shape (12,)
b = clf.intercept_[0] # 单个数值

print("static const double w[12] = {")
print(",\n".join(["\t{:.17f}".format(val) for val in w]))
print("};")
print("static const double b = {:.17f};".format(b))


