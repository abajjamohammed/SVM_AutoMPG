import numpy as np


class LinearSVRFromScratch:
    """
    Régresseur SVR Linéaire avec perte epsilon-insensible (Slide 15) :
    L_eps(y, y_pred) = max(0, |y - y_pred| - epsilon)
    Optimisation par descente de sous-gradient du problème primal régularisé L2.
    """
    def __init__(self, epsilon=1.0, C=1.0, lr=0.005, max_iter=2500):
        self.epsilon = epsilon
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.w0 = None
        self.support_vectors_idx_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        np.random.seed(42)
        self.w = np.zeros(n_features)
        self.w0 = 0.0

        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.w) + self.w0
            residuals = y - y_pred

            grad_w = self.w.copy()
            grad_w0 = 0.0

            upper_mask = residuals > self.epsilon
            lower_mask = residuals < -self.epsilon

            if np.any(upper_mask):
                grad_w -= self.C * np.sum(X[upper_mask], axis=0)
                grad_w0 -= self.C * np.sum(upper_mask)

            if np.any(lower_mask):
                grad_w += self.C * np.sum(X[lower_mask], axis=0)
                grad_w0 += self.C * np.sum(lower_mask)

            self.w -= self.lr * grad_w / n_samples
            self.w0 -= self.lr * grad_w0 / n_samples

        final_residuals = np.abs(y - (np.dot(X, self.w) + self.w0))
        self.support_vectors_idx_ = np.where(final_residuals >= self.epsilon)[0]

    def predict(self, X):
        return np.dot(X, self.w) + self.w0