import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import random

# ====================== Utility ======================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def gini(y):
    counts = Counter(y)
    impurity = 1.0
    for lbl in counts:
        prob = counts[lbl] / len(y)
        impurity -= prob**2
    return impurity

# ====================== Decision Tree ======================
class DecisionTree:
    def __init__(self, max_depth=10, min_size=2, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_features = self.n_features or int(np.sqrt(X.shape[1]))
        self.tree = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return y[0]
        if depth >= self.max_depth or len(y) <= self.min_size:
            return Counter(y).most_common(1)[0][0]

        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)
        best_feat, best_thresh, best_score, best_split = None, None, 1e9, None

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx = X[:, feat] <= t
                right_idx = ~left_idx
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                gini_left = gini(y[left_idx])
                gini_right = gini(y[right_idx])
                score = (sum(left_idx)*gini_left + sum(right_idx)*gini_right) / len(y)
                if score < best_score:
                    best_feat, best_thresh, best_score = feat, t, score
                    best_split = (left_idx, right_idx)

        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        left = self._build_tree(X[best_split[0]], y[best_split[0]], depth+1)
        right = self._build_tree(X[best_split[1]], y[best_split[1]], depth+1)
        return (best_feat, best_thresh, left, right)

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feat, thresh, left, right = node
        return self._predict_one(x, left if x[feat] <= thresh else right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ====================== Random Forest ======================
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_size=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.trees.clear()
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTree(self.max_depth, self.min_size, self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        final_pred = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final_pred)
    
    def predict_proba(self, X):
        # Predict each tree’s label for each sample
        preds = np.array([tree.predict(X) for tree in self.trees])
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            probs[:, i] = np.mean(preds == cls, axis=0)
        return probs

# ====================== Loss Function ======================
def classification_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

# ====================== Main ======================
if __name__ == "__main__":
    set_seed(42)
    data = np.load("features.npy", allow_pickle=True).item()
    X, y, paths = data["features"], data["labels"], data["paths"]

    print(f"Loaded features = {X.shape}, labels = {y.shape}")

    train_idx = [i for i, p in enumerate(paths) if "train" in p]
    val_idx   = [i for i, p in enumerate(paths) if "val" in p]
    test_idx  = [i for i, p in enumerate(paths) if "test" in p]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Normalize once
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ====================== Train ======================
    best_acc, best_model = 0, None
    epoches = 10
    for epoch in range(epoches):
        rf = RandomForest(n_trees=100, max_depth=15, n_features=int(np.sqrt(X_train.shape[1])))
        rf.fit(X_train, y_train)
        y_pred_val = rf.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        print(f"[Epoch {epoch+1}/{epoches}] Val Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = rf
            print(f"✅ New best model saved (Accuracy={best_acc*100:.2f}%)")

    # ====================== Evaluate ======================
    target_names = ["Non-Defective", "Defective", "Non-Phone"]

    print("\nClassification Report (Validation set):")
    print(classification_report(y_val, best_model.predict(X_val), target_names=target_names))

    print("\nClassification Report (Test set):")
    y_pred_test = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    # ====================== Confusion Matrices ======================
    # --- Validation set ---
    y_pred_val = best_model.predict(X_val)
    cm_val = confusion_matrix(y_val, y_pred_val)
    print("\nConfusion Matrix (Validation set):")
    print(cm_val)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title("Confusion Matrix - Validation set")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val.png")
    plt.close()

    # --- Test set ---
    y_pred_test = best_model.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix (Test set):")
    print(cm_test)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title("Confusion Matrix - Test set")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png")
    plt.close()

    # # ====================== PCA Visualization ======================
    # pca = PCA(n_components=2)
    # X_2D = pca.fit_transform(X_test)
    # plt.figure(figsize=(8,6))
    # colors = ["blue", "orange", "green"]
    # for i in range(len(y_test)):
    #     color = colors[y_test[i]]
    #     marker = "o" if y_pred_test[i] == y_test[i] else "x"
    #     plt.scatter(X_2D[i,0], X_2D[i,1], c=color, marker=marker, s=70)
    # plt.title("PCA Visualization - Random Forest Test Results")
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.savefig("pca_test.png")
    # plt.close()

    # ====================== Save ======================
    joblib.dump(best_model, "randomforest_best.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(f"✅ Best model saved to randomforest_best.pkl with Accuracy={best_acc*100:.2f}%")
