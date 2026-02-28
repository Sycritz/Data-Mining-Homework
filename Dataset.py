import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import pandas as pd


class SyntheticDatasetGenerator:
    
    def __init__(self, random_state=42): # For reproducability we set a default random state to the answer of the universe
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_dataset1_easy(self, n_samples_per_cluster=500, n_clusters=3, 
                                dim=2, separation_factor=6.0): # Dataset 1 is as decribed in the report
        
        sigma = 1.0  # Fixed standard deviation
        total_samples = n_samples_per_cluster * n_clusters
        
        # Generate well-separated cluster centers
        centers = self._generate_separated_centers(
            n_clusters, dim, min_distance=separation_factor * sigma
        )
        
        # Generate isotropic Gaussian clusters with equal variance
        X = np.zeros((total_samples, dim))
        y = np.zeros(total_samples, dtype=int)
        
        for i in range(n_clusters):
            start_idx = i * n_samples_per_cluster
            end_idx = (i + 1) * n_samples_per_cluster
            
            # Isotropic covariance: sigma^2 * I
            cov = sigma**2 * np.eye(dim)
            X[start_idx:end_idx] = np.random.multivariate_normal(
                centers[i], cov, size=n_samples_per_cluster
            )
            y[start_idx:end_idx] = i
        
        # Shuffle data
        shuffle_idx = np.random.permutation(total_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        metadata = {
            'n_clusters': n_clusters,
            'n_samples_per_cluster': n_samples_per_cluster,
            'dim': dim,
            'sigma': sigma,
            'separation_factor': separation_factor,
            'balanced': True,
            'isotropic': True
        }
        
        return X, y, centers, metadata
    
    def generate_dataset2_hard(self, n_samples_base=1000, n_clusters=3, dim=2): # Again as described in report
        
        if dim != 2:
            raise ValueError("Dataset 2 requires dim=2 for visualization")
        
        # Imbalanced proportions
        proportions = [0.60, 0.30, 0.10][:n_clusters]
        proportions = np.array(proportions) / sum(proportions)  # Normalize
        n_samples_per_cluster = (proportions * n_samples_base).astype(int)
        
        # Overlapping cluster centers (2-3 sigma apart)
        sigma = 1.0
        if n_clusters == 3:
            centers = np.array([
                [0, 0],
                [2.5 * sigma, 0],      # 2.5 sigma away (overlap)
                [1.25 * sigma, 2.2 * sigma]  # Forms triangle, overlaps with both
            ])
        else:
            # General case: closer spacing
            centers = self._generate_separated_centers(
                n_clusters, dim, min_distance=2.0 * sigma
            )
        
        # Anisotropic covariances (elongated, different orientations)
        covariances = []
        for i in range(n_clusters):
            # Create elongated covariance matrix
            # Different aspect ratios for each cluster
            eigenvalues = [sigma**2 * (3 + i), sigma**2 * (0.3 + i*0.1)]
            angle = i * np.pi / n_clusters  # Different rotation for each cluster
            
            cov = self._create_rotated_covariance(eigenvalues, angle)
            covariances.append(cov)
        
        # Generate data
        X_list = []
        y_list = []
        
        for i in range(n_clusters):
            n_samples = n_samples_per_cluster[i]
            X_cluster = np.random.multivariate_normal(
                centers[i], covariances[i], size=n_samples
            )
            y_cluster = np.full(n_samples, i, dtype=int)
            
            X_list.append(X_cluster)
            y_list.append(y_cluster)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # Shuffle data
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        metadata = {
            'n_clusters': n_clusters,
            'n_samples_per_cluster': n_samples_per_cluster.tolist(),
            'dim': dim,
            'sigma': sigma,
            'proportions': proportions.tolist(),
            'balanced': False,
            'isotropic': False,
            'covariances': covariances
        }
        
        return X, y, centers, metadata
    
    def _generate_separated_centers(self, n_clusters, dim, min_distance):
        """Generate cluster centers with minimum pairwise distance"""
        centers = []
        max_attempts = 1000
        
        # First center at origin
        centers.append(np.zeros(dim))
        
        # Generate remaining centers
        for i in range(1, n_clusters):
            for attempt in range(max_attempts):
                # Random center in reasonable range
                new_center = np.random.randn(dim) * min_distance
                
                # Check distance to all existing centers
                distances = [np.linalg.norm(new_center - c) for c in centers]
                
                if all(d >= min_distance for d in distances):
                    centers.append(new_center)
                    break
            else:
                # Fallback: place on a grid/circle
                angle = 2 * np.pi * i / n_clusters
                if dim == 2:
                    new_center = min_distance * np.array([np.cos(angle), np.sin(angle)])
                else:
                    new_center = np.zeros(dim)
                    new_center[0] = min_distance * np.cos(angle)
                    new_center[1] = min_distance * np.sin(angle)
                centers.append(new_center)
        
        return np.array(centers)
    
    def _create_rotated_covariance(self, eigenvalues, angle):
        """Create a 2D covariance matrix with specified eigenvalues and rotation"""
        # Diagonal matrix with eigenvalues
        D = np.diag(eigenvalues)
        
        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Rotated covariance: R * D * R^T
        cov = R @ D @ R.T
        return cov
    
    def visualize_dataset(self, X, y, centers, metadata, title="Dataset", 
                          save_path=None):
        """Visualize 2D dataset with true clusters and centers"""
        
        if metadata['dim'] != 2:
            print(f"Visualization only available for 2D data (dim={metadata['dim']})")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                            alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        # Plot centers
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
                  s=300, edgecolors='black', linewidth=2, label='True Centers')
        
        # Add ellipses for anisotropic clusters
        if not metadata['isotropic'] and 'covariances' in metadata:
            for i, cov in enumerate(metadata['covariances']):
                self._plot_covariance_ellipse(ax, centers[i], cov, 
                                             n_std=2.0, alpha=0.3)
        
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Add metadata text
        info_text = f"Clusters: {metadata['n_clusters']}\n"
        info_text += f"Balanced: {metadata['balanced']}\n"
        info_text += f"Isotropic: {metadata['isotropic']}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def _plot_covariance_ellipse(self, ax, center, cov, n_std=2.0, alpha=0.3):
        """Plot an ellipse representing the covariance matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        ellipse = Ellipse(xy=center, width=width, height=height, 
                         angle=angle, alpha=alpha, facecolor='gray', 
                         edgecolor='black', linewidth=2)
        ax.add_patch(ellipse)
    
    def save_dataset(self, X, y, centers, metadata, prefix="dataset"):
        """Save dataset to CSV files"""
        # Save data
        df_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df_data['label'] = y
        df_data.to_csv(f"datasets/{prefix}_data.csv", index=False)
        
        # Save centers
        df_centers = pd.DataFrame(centers, 
                                 columns=[f'feature_{i}' for i in range(centers.shape[1])])
        df_centers.to_csv(f"datasets/{prefix}_centers.csv", index=False)
        
        # Save metadata
        with open(f"datasets/{prefix}_metadata.txt", 'w') as f:
            for key, value in metadata.items():
                if key != 'covariances':  # Skip large numpy arrays
                    f.write(f"{key}: {value}\n")
        
        print(f"Dataset saved with prefix '{prefix}'")


# Example usage
if __name__ == "__main__":
    generator = SyntheticDatasetGenerator(random_state=42)
    
    print("="*60)
    print("DATASET 1: Clearly Separated (Easy Case)")
    print("="*60)
    
    # Generate Dataset 1 (2D for visualization)
    X1_2d, y1_2d, centers1_2d, meta1_2d = generator.generate_dataset1_easy(
        n_samples_per_cluster=500,
        n_clusters=3,
        dim=2,
        separation_factor=6.0
    )
    
    print(f"Shape: {X1_2d.shape}")
    print(f"Cluster distribution: {np.bincount(y1_2d)}")
    print(f"Centers:\n{centers1_2d}")
    
    generator.visualize_dataset(X1_2d, y1_2d, centers1_2d, meta1_2d,
                               title="Dataset 1: Well-Separated Clusters",
                               save_path="datasets/dataset1_2d.png")
    
    # Generate Dataset 1 (10D for high-dimensional test)
    X1_10d, y1_10d, centers1_10d, meta1_10d = generator.generate_dataset1_easy(
        n_samples_per_cluster=500,
        n_clusters=4,
        dim=10,
        separation_factor=6.0
    )
    
    print(f"\n10D version shape: {X1_10d.shape}")
    
    # Save Dataset 1
    generator.save_dataset(X1_2d, y1_2d, centers1_2d, meta1_2d, prefix="dataset1_2d")
    generator.save_dataset(X1_10d, y1_10d, centers1_10d, meta1_10d, prefix="dataset1_10d")
    
    print("\n" + "="*60)
    print("DATASET 2: Poorly Separated (Hard Case)")
    print("="*60)
    
    # Generate Dataset 2
    X2, y2, centers2, meta2 = generator.generate_dataset2_hard(
        n_samples_base=1000,
        n_clusters=3,
        dim=2
    )
    
    print(f"Shape: {X2.shape}")
    print(f"Cluster distribution: {np.bincount(y2)}")
    print(f"Proportions: {meta2['proportions']}")
    print(f"Centers:\n{centers2}")
    
    generator.visualize_dataset(X2, y2, centers2, meta2,
                               title="Dataset 2: Overlapping & Anisotropic Clusters",
                               save_path="datasets/dataset2.png")
    
    # Save Dataset 2
    generator.save_dataset(X2, y2, centers2, meta2, prefix="dataset2")
    
    print("\n" + "="*60)
    print("Datasets generated successfully!")
    print("="*60)