import numpy as np
import warnings

class KMeansClustering:

    def __init__(self, x_train: np.ndarray, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = None) -> None:
        if x_train.size == 0:
            raise ValueError("x_train cannot be empty")
        
        self.x_train = x_train
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self):
        n_samples = self.x_train.shape[0]
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n_samples})")

        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = self.x_train[indices].copy()
        
        for iteration in range(self.max_iter):
            labels = self._assign_clusters(self.x_train)
            new_centroids = self._compute_centroids(self.x_train, labels)
            
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            
            if centroid_shift < self.tol:
                break
        else:
            warnings.warn(f"KMeans did not converge after {self.max_iter} iterations")
        
        self.labels_ = labels
        return self

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = X[np.random.randint(len(X))]
        return new_centroids

    def predict(self, X: np.ndarray):
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Model must be fitted before prediction")
        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray = None):
        self.fit()
        return self.predict(X if X is not None else self.x_train)

    def get_centroids(self):
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Model must be fitted first")
        return self.centroids_

    def get_inertia(self):
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Model must be fitted first")
        distances = np.linalg.norm(self.x_train[:, np.newaxis] - self.centroids_, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)


class MiniBatchKMeans:

    def __init__(self, n_clusters: int, batch_size: int = 100, max_iter: int = 300, 
                 tol: float = 1e-4, random_state: int = None) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.counts_ = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n_samples})")

        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[indices].copy()
        self.counts_ = np.zeros(self.n_clusters)
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids_.copy()
            
            batch_indices = np.random.choice(n_samples, min(self.batch_size, n_samples), replace=False)
            batch = X[batch_indices]
            
            self.partial_fit(batch)
            
            centroid_shift = np.linalg.norm(self.centroids_ - old_centroids)
            if centroid_shift < self.tol:
                break
        else:
            warnings.warn(f"MiniBatchKMeans did not converge after {self.max_iter} iterations")
        
        return self

    def partial_fit(self, X: np.ndarray):
        if self.centroids_ is None:
            n_samples = X.shape[0]
            init_size = min(self.n_clusters, n_samples)
            indices = np.random.choice(n_samples, init_size, replace=False)
            self.centroids_ = X[indices].copy()
            self.counts_ = np.ones(self.n_clusters)
        
        labels = self._assign_clusters(X)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                n_new = np.sum(mask)
                self.counts_[k] += n_new
                learning_rate = n_new / self.counts_[k]
                self.centroids_[k] = (1 - learning_rate) * self.centroids_[k] + learning_rate * X[mask].mean(axis=0)
        
        return self

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray):
        if self.centroids_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.predict(X)

    def get_centroids(self):
        if self.centroids_ is None:
            raise RuntimeError("Model must be fitted first")
        return self.centroids_


class FuzzyCMeans:
    
    def __init__(self, x_train: np.ndarray, n_clusters: int, m: float = 2.0, 
                 max_iter: int = 300, tol: float = 1e-4, random_state: int = None) -> None:
        self.x_train = x_train
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if m <= 1:
            raise ValueError("m must be > 1")
        if n_clusters > x_train.shape[0]:
            raise ValueError(f"n_clusters ({n_clusters}) cannot exceed n_samples ({x_train.shape[0]})")
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self):
        n_samples = self.x_train.shape[0]
        
        self.membership_ = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples).T
        
        for iteration in range(self.max_iter):
            old_membership = self.membership_.copy()
            
            self.centroids_ = self._compute_centroids()
            self.membership_ = self._update_membership()
            
            if np.linalg.norm(self.membership_ - old_membership) < self.tol:
                break
        else:
            warnings.warn(f"FuzzyCMeans did not converge after {self.max_iter} iterations")
        
        return self
    
    def _compute_centroids(self):
        u_m = self.membership_ ** self.m
        centroids = (u_m @ self.x_train) / u_m.sum(axis=1, keepdims=True)
        return centroids
    
    def _update_membership(self):
        distances = np.linalg.norm(self.x_train[:, np.newaxis] - self.centroids_, axis=2).T
        distances = np.fmax(distances, 1e-10)
        
        power = 2 / (self.m - 1)
        inv_distances = 1 / distances
        membership = inv_distances ** power
        membership /= membership.sum(axis=0, keepdims=True)
        
        return membership
    
    def predict(self, X: np.ndarray):
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Model must be fitted before prediction")
        
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2).T
        distances = np.fmax(distances, 1e-10)
        
        power = 2 / (self.m - 1)
        inv_distances = 1 / distances
        membership = inv_distances ** power
        membership /= membership.sum(axis=0, keepdims=True)
        
        return np.argmax(membership, axis=0)
    
    def predict_proba(self, X: np.ndarray):
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Model must be fitted before prediction")
        
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2).T
        distances = np.fmax(distances, 1e-10)
        
        power = 2 / (self.m - 1)
        inv_distances = 1 / distances
        membership = inv_distances ** power
        membership /= membership.sum(axis=0, keepdims=True)
        
        return membership.T
    
    def fit_predict(self, X: np.ndarray = None):
        self.fit()
        if X is None:
            return np.argmax(self.membership_, axis=0)
        return self.predict(X)
    
    def get_centroids(self):
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Model must be fitted first")
        return self.centroids_
    
    def get_membership_matrix(self):
        if not hasattr(self, "membership_"):
            raise RuntimeError("Model must be fitted first")
        return self.membership_.T


class MiniBatchFuzzyCMeans:
    
    def __init__(self, n_clusters: int, batch_size: int = 100, m: float = 2.0,
                 max_iter: int = 300, tol: float = 1e-4, random_state: int = None) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if m <= 1:
            raise ValueError("m must be > 1")
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n_samples})")
        
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[indices].copy()
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids_.copy()
            
            batch_indices = np.random.choice(n_samples, min(self.batch_size, n_samples), replace=False)
            batch = X[batch_indices]
            
            self.partial_fit(batch)
            
            centroid_shift = np.linalg.norm(self.centroids_ - old_centroids)
            if centroid_shift < self.tol:
                break
        else:
            warnings.warn(f"MiniBatchFuzzyCMeans did not converge after {self.max_iter} iterations")
        
        return self
    
    def partial_fit(self, X: np.ndarray):
        if self.centroids_ is None:
            n_samples = X.shape[0]
            init_size = min(self.n_clusters, n_samples)
            indices = np.random.choice(n_samples, init_size, replace=False)
            self.centroids_ = X[indices].copy()
        
        membership = self._compute_membership(X)
        u_m = membership ** self.m
        new_centroids = (u_m @ X) / u_m.sum(axis=1, keepdims=True)
        
        learning_rate = 0.1
        self.centroids_ = (1 - learning_rate) * self.centroids_ + learning_rate * new_centroids
        
        return self
    
    def _compute_membership(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2).T
        distances = np.fmax(distances, 1e-10)
        
        power = 2 / (self.m - 1)
        inv_distances = 1 / distances
        membership = inv_distances ** power
        membership /= membership.sum(axis=0, keepdims=True)
        
        return membership
    
    def predict(self, X: np.ndarray):
        if self.centroids_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        membership = self._compute_membership(X)
        return np.argmax(membership, axis=0)
    
    def predict_proba(self, X: np.ndarray):
        if self.centroids_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self._compute_membership(X).T
    
    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.predict(X)
    
    def get_centroids(self):
        if self.centroids_ is None:
            raise RuntimeError("Model must be fitted first")
        return self.centroids_