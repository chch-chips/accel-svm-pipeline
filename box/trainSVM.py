import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
np.set_printoptions(threshold=np.inf)
# 1) 读取特征数据
df = pd.read_csv("2025-04-09/feature_data_labeled_span_439.csv")

# 假设这 12 个特征列名与你项目中一致
feature_cols = [
    "meanx", "meany", "meanz",
    "sdx",   "sdy",   "sdz",
    "meanxg","meanyg","meanzg",
    "sdxg",  "sdyg",  "sdzg"
]
X = df[feature_cols].values
y = df["label"].values  # -1 / +1

# 2) 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) 训练线性 SVM （可根据需要调参 C、添加正则化或特征缩放）
clf = SVC(kernel="linear", C=1.0)
clf.fit(X_train, y_train)

# 4) 在测试集上评估
accuracy = clf.score(X_test, y_test)
print("Test Accuracy: {:.4f}".format(accuracy))

# 5) 获取训练好的模型参数
#    对于二分类线性SVM，决策函数是 w · x + b
#    - w 即 clf.coef_[0]
#    - b 即 clf.intercept_[0]
#    - support_vectors_ 是所有支持向量
#    - dual_coef_ 对应 alpha（对偶系数）
w = clf.coef_[0]                # shape: (12,)
b = clf.intercept_[0]           # shape: (1,)
support_vectors = clf.support_vectors_   # shape: (nSV, 12)
alpha = clf.dual_coef_[0]       # shape: (1, nSV)

print("w (coef):", w)
print("b (intercept):", b)
print("Support Vectors:\n", support_vectors)
print("Alpha (dual_coef):", alpha)

# 6) 如果你希望完全模仿项目中 '支持向量数量 = 12' 的写法
#    你可能需要控制正则化程度、或对数据进行筛选，才能使 nSV = 12。
#    否则，需要同步修改 C 端的数组大小和计算逻辑。
