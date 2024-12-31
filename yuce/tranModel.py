from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# 训练和预测模型
def train_predict_model(features, labels):
    # 映射标签到 [0, 1]
    labels_mapped = labels.map({-1: 0, 1: 1})

    # 数据拆分：训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels_mapped, test_size=0.2, random_state=42)

    # 使用 XGBoost 模型，启用 GPU 加速
    xgb_model = XGBClassifier(
        random_state=42,
        tree_method='hist',  # GPU 加速的方法
        device='cuda',
        max_depth=10,
        learning_rate=0.1,
        n_estimators=100
    )

    # 训练模型
    xgb_model.fit(X_train, y_train)

    # 预测测试集
    y_pred = xgb_model.predict(X_test)

    # 将预测结果映射回原始标签 [-1, 1]
    y_pred = pd.Series(y_pred).map({0: -1, 1: 1})
    y_test = pd.Series(y_test).map({0: -1, 1: 1})
    # y_test = y_test.map({0: -1, 1: 1})  # 测试集标签也映射回 [-1, 1]

    # 输出模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost 准确率: {accuracy_score(y_test, y_pred):.2f}")
    print(f"XGBoost Classification Report:\n{classification_report(y_test, y_pred)}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # 你还可以绘制模型评估图，如 ROC 曲线
    # 例如，绘制 ROC AUC 曲线：
    # y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    # fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='best')
    # plt.show()

    
    return xgb_model


# 保存模型
def save_model(model, path='model.pkl'):
    joblib.dump(model, path)
    print(f"[INFO] 模型已保存至 {path}")

# 加载模型
def load_model(path='model.pkl'):
    model = joblib.load(path)
    print(f"[INFO] 模型已加载")
    return model