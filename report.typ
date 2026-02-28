#import "@preview/flow-way:0.1.0": *

#show: flow.with(
  title: "Mini-Batch Clustering Analysis: Scalable K-Means and Fuzzy C-Means",
  subtitle: "Comparative Study on High-Dimensional and Large-Scale Tabular Data",
  authors: ("Medjdoul Anis",),
  affiliation: "National Higher School of Mathematics (NHSM)",
  toc: true,
  breaks: true,
)

#set par(justify: true)

The objective is to quantify the tradeoff between computational efficiency (runtime) and clustering quality (Silhouette, V-Measure, ARI), providing a rigorous assessment of when mini-batch stochasticity is preferable to full-batch exactitude.

In many real‑world applications, datasets are huge—think of customer data or image collections. Running traditional K‑Means on millions of rows can take hours or even crash your laptop. So we wanted to see if the ‘mini‑batch’ trick really works: does it save time without ruining the clusters? And does it behave differently for ‘hard’ clustering (K‑Means) vs. ‘soft’ clustering (Fuzzy C‑Means)? We picked two very different datasets to test this: one big and tabular, one high‑dimensional and image‑based.

= Hypotheses

Based on what we learned in class, we had a few guesses:
- Mini‑batch K‑Means would be way faster than full K‑Means, but might give slightly messier clusters (lower Silhouette score).
- For Fuzzy C‑Means, the full version is super slow because it updates a huge membership matrix. We thought mini‑batch FCM would be much faster, but maybe too noisy—or maybe the noise would actually help it avoid bad local minima.
- We also expected the high‑dimensional Fashion‑MNIST to be harder for any distance‑based method, so preprocessing with PCA would be crucial.

= Theoretical Framework

== K-Means and the Mini-Batch Optimization

K-Means objective is to minimize the total within-cluster variance (Inertia):
$ J(X, mu) = sum_(i=1)^(k) sum_(x in C_i) ||x - mu_i||^2 $

The standard algorithm alternates between an *Assignment Step* and an *Update Step*. The bottleneck lies in the assignment $x arrow.long scripts(op("argmin"))_i ||x - mu_i||^2$, which must be computed for all $n$ points.

*Mini-Batch K-Means* (Sculley, 2010) relaxes this constraint by using a small batch $B$ of size $b$. For each $x in B$, the algorithm finds the nearest center $mu_i$ and updates that center using a learning rate $eta$ that decreases as more points are assigned to that cluster:
$ mu_i arrow.long (1 - eta) mu_i + eta x $
This stochastic gradient descent-like update allows the algorithm to converge much faster, often reaching an "acceptable" solution before having seen the entire dataset even once.

== Fuzzy C-Means (FCM)

FCM introduces the concept of fuzzy sets to clustering. Each point $x_j$ belongs to all clusters with varying degrees of membership $u_(i j) in [0, 1]$. The fuzziness is controlled by the parameter $m$:
$ J_m = sum_(i=1)^(k) sum_(j=1)^(n) (u_(i j))^m ||x_j - mu_i||^2 $

When $m=1$, FCM reduces to K-Means. Typically $m=2$ is used to balance the "influence" of distant points. The fuzzy approach is particularly useful in datasets where clusters are not well-separated or where a point's shared identity between two classes is semantically meaningful.

*Mini-Batch FCM* is an adaptation where only the points in the current batch contribute to the update of the membership matrix $U$ and the centroids $mu$. This drastically reduces the memory footprint of the $n times k$ membership matrix, which would otherwise be $O(n k)$.

= Experimental Data and Preprocessing

== Covertype: Large-Scale Tabular Data

The *Covertype* dataset (UCI) describes forest cover types from cartographic variables.
- *Cardinality*: 581,012 samples.
- *Dimensionality*: 54 features (Elevation, Aspect, Slope, Distances, etc.).
- *Preprocessing*: We apply *StandardScaler* to ensure all features contribute equally to the Euclidean distance calculation. This is critical because features like "Elevation" (thousands of meters) would otherwise dominate binary features like "Soil_Type" (0 or 1).
- *Subsampling*: 15,000 samples are used for the main evaluation to allow for multiple repetitions (10 seeds) while maintaining statistical significance.

Covertype is huge (over half a million rows) – perfect for testing scalability. It also has a known number of classes (7), so we can evaluate how well the clusters match the real labels. We subsampled it to 15k to run multiple trials (10 seeds) without waiting forever, while still being representative of the overall distribution.

== Fashion-MNIST: High-Dimensional Imagery

Fashion-MNIST consists of $28 times 28$ grayscale images.
- *Feature Space*: 784 dimensions (pixels).
- *Complexity*: Direct clustering in 784D is susceptible to the "Curse of Dimensionality," where distances between all pairs of points become roughly equal.
- *Dimensionality Reduction*: We apply *Principal Component Analysis (PCA)* to reduce the space to 50 dimensions. This preserves ~84% of the variance while significantly accelerating the distance computations ($||x - mu||^2$) and improving the signal-to-noise ratio.

Fashion‑MNIST is not as large as Covertype (70k samples) but has 784 dimensions—a classic ‘curse of dimensionality’ challenge. We wanted to see if mini‑batch methods handle high-dimensional spatial data effectively.

#figure(
  image("results/pca_visualization.png", width: 90%),
  caption: [PCA 2D Visualization. Fashion-MNIST (right) shows distinct globular clusters corresponding to clothing categories, whereas Covertype (left) exhibits a more continuous and overlapping structure.]
)

= Parameter Selection Strategy

As emphasized in the study requirements, parameter selection must be rigorously justified rather than arbitrarily chosen.

== Determining the Optimal $k$

We utilize the *Elbow Method* and *Silhouette Analysis* on a subset of the Covertype data.

#figure(
  grid(
    columns: (1fr),
    gutter: 1em,
    image("results/parameter_selection_k.png", width: 90%),
    [
      *Analysis*: The Elbow plot (Inertia) shows a clear "bend" at $k=7$. The Silhouette score, which measures cluster separation, also peaks or stabilizes around this region. This aligns with the physical reality of the dataset, which contains 7 ground-truth cover types.
    ]
  ),
  caption: [Justification of k=7 for Covertype.]
)

== Batch Size and Stochasticity

The *Batch Size* parameter is a critical hyperparameter for mini-batch variants.

#figure(
  image("results/batch_size_analysis.png", width: 90%),
  caption: [Effect of Batch Size on Runtime and Silhouette Quality. We observe that increasing the batch size beyond 1024 yields diminishing returns in quality while significantly increasing the time per iteration.]
)

Based on this analysis, we select *1024* as our default batch size. It provides a stable gradient update while maintaining the sub-linear computational benefit.

= Methodology

This section consolidates the technical details of our experimental environment to ensure reproducibility.

- **Hardware**: All experiments were run on an AMD Ryzen 7 PRO 5850U with Radeon Graphics, 16GB RAM, running Arch Linux (Kernel 6.18.9).
- **Software**: The analysis was performed using Python 3.12 within a dedicated `ML` conda environment. Key libraries include:
  - `scikit-learn` (v1.8.0): Used for K-Means and standard data preprocessing.
  - `scikit-fuzzy`: Used for Fuzzy C-Means membership updates.
  - `numpy` (v2.4.0), `pandas` (v2.3.3), `matplotlib`: Used for numerical computation and data visualization.
- **Experimental setup**: Each algorithm was executed 10 times using distinct random seeds (1-10) to assess stability. We used $k=7$ for Covertype and $k=10$ for Fashion-MNIST. The batch size was fixed at 1024 for all mini-batch variants. Evaluation metrics include Silhouette Score, V-Measure, and Adjusted Rand Index (ARI).
- **Reproducibility**: We used default parameters for all algorithms except where specifically noted (e.g., fuzziness $m=2.0$ for FCM). The source code and raw result files are included with this report.

= Comparative Results and Discussion

The following results represent the mean and standard deviation across *10 independent trials* with different random initializations (seeds).

== Quantitative Performance

#table(
  columns: (1.5fr, 1.2fr, 1.2fr, 1fr, 1fr),
  inset: 7pt,
  align: center + horizon,
  [*Algorithm*], [*Runtime (s)*], [*Silhouette*], [*V-Measure*], [*ARI*],
  [K-Means], [0.645 ± 0.38], [0.1131 ± 0.01], [0.176 ± 0.03], [0.080 ± 0.02],
  [MB K-Means], [0.386 ± 0.07], [0.0943 ± 0.01], [0.147 ± 0.04], [0.068 ± 0.02],
  [Fuzzy C-Means], [0.571 ± 0.07], [0.0817 ± 0.02], [0.070 ± 0.00], [0.001 ± 0.00],
  [MB FCM], [0.325 ± 0.02], [0.0869 ± 0.01], [0.070 ± 0.00], [0.001 ± 0.00],
)

*Table 1: Performance metrics for Covertype (k=7)*

#table(
  columns: (1.5fr, 1.2fr, 1.2fr, 1fr, 1fr),
  inset: 7pt,
  align: center + horizon,
  [*Algorithm*], [*Runtime (s)*], [*Silhouette*], [*V-Measure*], [*ARI*],
  [K-Means], [1.904 ± 0.41], [0.1840 ± 0.01], [0.510 ± 0.01], [0.357 ± 0.01],
  [MB K-Means], [0.523 ± 0.08], [0.1773 ± 0.01], [0.500 ± 0.02], [0.345 ± 0.02],
  [FCM (Standard)], [14.359 ± 1.66], [0.0560 ± 0.02], [0.298 ± 0.01], [0.189 ± 0.01],
  [MB FCM], [0.362 ± 0.01], [0.0799 ± 0.01], [0.372 ± 0.01], [0.270 ± 0.01],
)

*Table 2: Performance metrics for Fashion-MNIST (k=10)*

== Analysis of Observed Trends

=== 1. Speed-Precision Tradeoff
In Fashion-MNIST, where the dimensionality is high, Mini-Batch K-Means achieved a *3.6x speedup* compared to standard K-Means while maintaining over *98% of the V-Measure* (0.500 vs 0.510). This confirms that for high-dimensional data, the stochastic sampling effectively captures the global structure without requiring the computation of all $n times k$ distances. On Covertype, standard K‑Means was not that much slower (0.65s vs 0.39s) because 15k samples isn't huge. On Fashion‑MNIST, the speed difference is dramatic (1.9s vs 0.52s) due to expensive 784‑d distance computations. Mini‑batch retains 98% of V‑Measure, confirming stochastic sampling captures global structure.

=== 2. The Curious Case of Mini-Batch FCM
Surprisingly, on Fashion-MNIST, *Mini-Batch FCM* outperformed the standard FCM in terms of clustering quality (ARI 0.270 vs 0.189) while being roughly *40 times faster*. Full FCM is extremely slow (14s) because it updates a 70k×10 membership matrix each iteration. Mini‑batch FCM runs in 0.36s and even achieves higher ARI (0.27 vs 0.19). This phenomenon suggests that the mini-batch update acts as a form of *simulated annealing* or regularization. The standard FCM, by considering the whole dataset, likely gets trapped in a local minimum early on. The stochastic nature of mini-batches allows the centroids to "escape" these local traps, eventually finding a better partition.

=== 3. Statistical Robustness
The standard deviation for Mini-Batch K-Means runtime (0.07s on Fashion-MNIST) is significantly lower than that of K-Means (0.41s). This indicates that mini-batches provide a more predictable execution time, which is a desirable property for real-time applications or cloud computing environments with budget constraints. Lower standard deviation in mini‑batch runtimes (0.07s vs 0.41s for K‑Means) indicates more predictable execution, beneficial for real‑time or cloud environments.

#figure(
  image("results/boxplots_comparison.png", width: 95%),
  caption: [Boxplots across 10 trials. Note the extreme runtime reduction in the lower-right quadrant for Fashion-MNIST.]
)

= Scalability and Complexity Analysis

To further demonstrate the complexity benefits, we conducted a scalability benchmark varying $n$ from 10,000 to 500,000 samples.

#figure(
  image("results/scalability_benchmark.png", width: 75%),
  caption: [Runtime scaling. Standard K-Means shows a linear increase in time, whereas Mini-Batch K-Means remains nearly constant once the batch size is fixed.]
)

Once the batch size is fixed and sufficient iterations are allowed, the runtime remains constant regardless of the total number of samples $n$.

= Limitations and Future Work

One limitation is that we only tried one batch size (1024) for the main experiments. Maybe a smaller batch would be even faster, or a larger one might improve quality. We also didn’t tune the fuzziness parameter $m$ in FCM – we just used $m=2$ because that’s common. Different $m$ might change the comparison. Also, we only looked at Euclidean distance; other distance metrics might work better for image data. If we had more time, we’d test on even bigger datasets (like millions of rows) and see if the speedup holds.

= Conclusion

This comparative study highlights the efficiency of mini-batch clustering. Our findings demonstrate that:
1.  *Mini-Batch K-Means* is highly effective for high-dimensional image data, providing massive speedups with negligible quality loss.
2.  *Mini-Batch FCM* is not only a faster alternative to standard Fuzzy C-Means but can actually lead to better clustering results by avoiding poor local minima through stochastic updates.
3.  *Preprocessing matters*: PCA for imagery and Scaling for tabular data are essential prerequisites for meaningful Euclidean distance-based clustering.

Overall, this homework made me realize that ‘exact’ algorithms aren’t always better – sometimes a smart approximation (like mini‑batches) can be both faster and more robust. For big data, I’ll definitely use mini‑batch from now on.
