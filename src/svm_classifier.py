import numpy as np
import cvxopt
import cvxopt.solvers

# Désactiver les logs verbeux de cvxopt
cvxopt.solvers.options['show_progress'] = False


class SVMFromScratch:
    """
    Classifieur SVM Binaire résolvant la forme duale (Slide 9, 12, 14) via cvxopt :
    max -1/2 * sum_ij (alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)) + sum_i alpha_i
    sous contrainte : 0 <= alpha_i <= C  et  sum(alpha_i * y_i) = 0
    """
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        
        self.alpha = None
        self.sv_X = None
        self.sv_y = None
        self.w0 = None
        self.w = None

    def _kernel_func(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            # k(X, X') = exp(-gamma * ||X - X'||^2)
            dist_sq = (
                np.sum(X1**2, axis=1)[:, np.newaxis]
                + np.sum(X2**2, axis=1)
                - 2 * np.dot(X1, X2.T)
            )
            return np.exp(-self.gamma * dist_sq)
        elif self.kernel == 'poly':
            return (1.0 + np.dot(X1, X2.T)) ** self.degree
        else:
            raise ValueError(f"Noyau {self.kernel} non reconnu.")

    def fit(self, X, y):
        n_samples = X.shape[0]
        y_mod = y.astype(np.double)
        
        # 1. Matrice de Gram et formulation QP cvxopt
        K = self._kernel_func(X, X)
        P = cvxopt.matrix(np.outer(y_mod, y_mod) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # Contraintes 0 <= alpha_i <= C
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        # Contrainte sum(alpha_i * y_i) = 0
        A = cvxopt.matrix(y_mod.reshape(1, -1))
        b = cvxopt.matrix(0.0)
        
        # Résolution
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        
        # Vecteurs de support (alpha > 1e-5)
        is_sv = alphas > 1e-5
        self.alpha = alphas[is_sv]
        self.sv_X = X[is_sv]
        self.sv_y = y_mod[is_sv]
        
        # Calcul de w0* (Slide 10)
        K_sv = self._kernel_func(self.sv_X, self.sv_X)
        pred_sv = np.sum((self.alpha * self.sv_y)[:, np.newaxis] * K_sv, axis=0)
        self.w0 = np.mean(self.sv_y - pred_sv)
        
        if self.kernel == 'linear':
            self.w = np.sum((self.alpha * self.sv_y)[:, np.newaxis] * self.sv_X, axis=0)

    def decision_function(self, X):
        K = self._kernel_func(self.sv_X, X)
        return np.sum((self.alpha * self.sv_y)[:, np.newaxis] * K, axis=0) + self.w0

    def predict(self, X):
        return np.sign(self.decision_function(X))


class OneVsRestSVM:
    """
    Stratégie One-vs-Rest (OvR) pour la classification multiclasse (3 classes).
    """
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.classifiers = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classifiers = []
        for c in self.classes:
            y_bin = np.where(y == c, 1.0, -1.0)
            clf = SVMFromScratch(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
            clf.fit(X, y_bin)
            self.classifiers.append(clf)

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            scores[:, i] = clf.decision_function(X)
        return self.classes[np.argmax(scores, axis=1)]