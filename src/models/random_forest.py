import numpy as np


def variance(y):
    return np.var(y)


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        gain=0.0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain


def build_tree(
    X,
    y,
    depth,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    min_impurity_decrease
):

    if (
        len(y) < min_samples_split
        or depth >= max_depth
        or variance(y) == 0
    ):
        return Node(value=y.mean())

    n_samples, n_features = X.shape

    if max_features == "sqrt":
        n_feats = int(np.sqrt(n_features))
    elif max_features == "log2":
        n_feats = int(np.log2(n_features))
    else:
        n_feats = max_features

    feature_idxs = np.random.choice(
        n_features, n_feats, replace=False
    )

    best_gain = 0.0
    best_feature = None
    best_threshold = None

    parent_var = variance(y)

    for f in feature_idxs:
        x_f = X[:, f]

        thresholds = np.unique(
            np.quantile(x_f, np.linspace(0.1, 0.9, 10))
        )

        for t in thresholds:
            left_mask = x_f <= t
            right_mask = ~left_mask

            if (
                left_mask.sum() < min_samples_leaf
                or right_mask.sum() < min_samples_leaf
            ):
                continue

            gain = parent_var - (
                left_mask.sum() / n_samples * variance(y[left_mask])
                + right_mask.sum() / n_samples * variance(y[right_mask])
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = t

    if best_gain < min_impurity_decrease:
        return Node(value=y.mean())

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    return Node(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_gain,
        left=build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            min_impurity_decrease
        ),
        right=build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            min_impurity_decrease
        )
    )


def predict_tree(node, x):

    if node.value is not None:
        return node.value

    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)


class RandomForestRegressor:

    def __init__(
        self,
        n_trees=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        min_impurity_decrease=1e-4,
        random_state=42
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):

        np.random.seed(self.random_state)
        self.trees = []

        n_samples = len(X)

        for _ in range(self.n_trees):
            idxs = np.random.choice(
                n_samples, n_samples, replace=True
            )

            tree = build_tree(
                X[idxs],
                y[idxs],
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease
            )

            self.trees.append(tree)

    def predict(self, X):

        tree_preds = np.array(
            [
                [predict_tree(tree, x) for x in X]
                for tree in self.trees
            ]
        )

        return np.mean(tree_preds, axis=0)
