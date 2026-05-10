import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        """
        K-Nearest Neighbors classifier
        """

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data
        """

        self.X_train = X
        self.y_train = y

    def compute_distance(self, x1, x2):
        """
        Euclidean distance from scratch

        d(x,y) = sqrt(sum((x-y)^2))
        """

        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, x):
        """
        Predict one sample
        """

        distances = []

        # distance to every training sample
        for i in range(len(self.X_train)):

            d = self.compute_distance(x, self.X_train[i])

            distances.append((d, self.y_train[i]))

        # sort by distance
        distances.sort(key=lambda item: item[0])

        # k nearest
        k_nearest = distances[:self.k]

        # labels only
        k_labels = [label for _, label in k_nearest]

        # majority vote
        prediction = Counter(k_labels).most_common(1)[0][0]

        return prediction

    def predict_batch(self, X):
        """
        Predict multiple samples
        """

        predictions = []

        for x in X:

            pred = self.predict(x)

            predictions.append(pred)

        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        """
        Compute accuracy
        """

        return np.mean(y_true == y_pred)


if __name__ == "__main__":

    print("Loading MRMR-selected features...")

    # ==========================================
    # LOAD TRAIN DATA (NON-AUGMENTED)
    # ==========================================

    train_data = np.load("features/train_mrmr.npz")

    X_train = train_data["X"]
    y_train = train_data["y"]

    # ==========================================
    # LOAD VALIDATION DATA
    # ==========================================

    val_data = np.load("features/val_mrmr.npz")

    X_val = val_data["X"]
    y_val = val_data["y"]

    # ==========================================
    # LOAD TEST DATA
    # ==========================================

    test_data = np.load("features/test_mrmr.npz")

    X_test = test_data["X"]
    y_test = test_data["y"]

    # ==========================================
    # DEBUG INFO
    # ==========================================

    print(f"\nTraining shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    print("\nUnique labels:")
    print(np.unique(y_train))

    print("\nChecking for NaN values...")
    print("Train NaNs:", np.isnan(X_train).sum())
    print("Val NaNs:", np.isnan(X_val).sum())
    print("Test NaNs:", np.isnan(X_test).sum())

    # ==========================================
    # FEATURE NORMALIZATION
    # ==========================================

    print("\nNormalizing features...")

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print("Normalization complete.")

    print("\nFeature range after normalization:")
    print("Min:", np.min(X_train))
    print("Max:", np.max(X_train))

    # ==========================================
    # K SWEEP
    # ==========================================

    k_values = [1, 3, 5, 7, 9]

    best_k = None
    best_acc = 0

    print("\nRunning K sweep...\n")

    for k in k_values:

        print(f"Testing k = {k}")

        model = KNN(k=k)

        model.fit(X_train, y_train)

        predictions = model.predict_batch(X_val)

        acc = model.accuracy(y_val, predictions)

        print(f"Validation Accuracy = {acc:.4f}\n")

        if acc > best_acc:

            best_acc = acc
            best_k = k

    # ==========================================
    # BEST MODEL
    # ==========================================

    print("===================================")
    print(f"Best k = {best_k}")
    print(f"Best Validation Accuracy = {best_acc:.4f}")
    print("===================================")

    # ==========================================
    # TEST EVALUATION
    # ==========================================

    print("\nEvaluating on test set...")

    best_model = KNN(k=best_k)

    best_model.fit(X_train, y_train)

    test_predictions = best_model.predict_batch(X_test)

    test_acc = best_model.accuracy(y_test, test_predictions)

    print(f"\nTest Accuracy = {test_acc:.4f}")