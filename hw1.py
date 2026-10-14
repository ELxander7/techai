import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class SimpleDenseNet:
    def __init__(self, input_size=28 * 28, hidden_size=128, output_size=10, lr=0.05):
        rng = np.random.default_rng(42)

        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        self.lr = lr

    @staticmethod
    def relu(x):
        # ReLU(x) = max(0, x)
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def softmax(x):
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def cross_entropy(self, probs, Y):
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0)
        return -np.sum(Y * np.log(probs)) / probs.shape[0]

    def backward(self, Y):
        N = Y.shape[0]

        dZ2 = (self.A2 - Y) / N

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_grad(self.Z1)

        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def evaluate(self, X, Y):
        probs = self.forward(X)
        loss = self.cross_entropy(probs, Y)
        acc = np.mean(np.argmax(probs, axis=1) == np.argmax(Y, axis=1))
        return loss, acc

    def fit(self, X_train, Y_train, X_val, Y_val,
            epochs=30, batch_size=128, verbose=True):

        history = {"train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": []}

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X_train[idx]
            Y_shuffled = Y_train[idx]

            for start in range(0, n_samples, batch_size):
                xb = X_shuffled[start:start + batch_size]
                yb = Y_shuffled[start:start + batch_size]

                self.forward(xb)
                self.backward(yb)

            train_loss, train_acc = self.evaluate(X_train, Y_train)
            val_loss, val_acc = self.evaluate(X_val, Y_val)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Epoch {epoch + 1:2d}/{epochs} | "
                      f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        return history


def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                         parser='liac-arff')

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1, random_state=42, stratify=y_train_full
    )

    def one_hot(a, num_classes=10):
        out = np.zeros((a.size, num_classes), dtype=np.float32)
        out[np.arange(a.size), a] = 1.0
        return out

    return (X_train, one_hot(y_train),
            X_val,   one_hot(y_val),
            X_test,  one_hot(y_test))


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png", dpi=150)
    plt.show()



if __name__ == "__main__":
    np.random.seed(42)

    print("Загружаю MNIST...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    model = SimpleDenseNet(input_size=784, hidden_size=128,
                           output_size=10, lr=0.1)

    history = model.fit(X_train, Y_train, X_val, Y_val,
                        epochs=30, batch_size=128)

    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"\nTest loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

    plot_history(history)