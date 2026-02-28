#import "@preview/flow-way:0.1.0": *

#show: flow.with(
  title: "Mini-Batch Clustering Analysis: Scalable K-Means and Fuzzy C-Means",
  subtitle: "Comparative Study on High-Dimensional and Large-Scale Tabular Data",
  authors: ("Medjdoul Anis",),
  affiliation: "National Higher School of Mathematics (NHSM)",
  year: 2025,
  toc: true,
  toc-depth: 3,
  breaks: false,
  main-color: "003F88",
)

#set par(justify: true, leading: 0.7em)
#set text(size: 11pt)

// ── Table styling helpers ──────────────────────────────────────────────────────
#let header-fill = rgb("003F88")
#let alt-fill    = rgb("003F88").lighten(92%)

#let styled-table(columns: (), data: ()) = {
  let header = data.first()
  let rows   = data.slice(1)
  table(
    columns:  columns,
    inset:    8pt,
    align:    center + horizon,
    fill: (col, row) =>
      if row == 0          { header-fill }
      else if calc.odd(row){ alt-fill    }
      else                 { white       },
    stroke: (col, row) => (
      bottom: 0.5pt + if row == 0 { white } else { rgb("cccccc") },
    ),
    ..header.map(h => text(fill: white, weight: "bold", h)),
    ..rows.flatten(),
  )
}

// ── Callout box for key insights ─────────────────────────────────────────────
#let insight-box(body) = block(
  width: 100%,
  inset: (x: 14pt, y: 10pt),
  radius: 4pt,
  fill: rgb("003F88").lighten(93%),
  stroke: (left: 3pt + rgb("003F88")),
  body,
)

// ── 1. INTRODUCTION ──────────────────────────────────────────────────────────
= Introduction

In the era of modern data science, clustering remains a cornerstone of
exploratory data analysis, pattern recognition, and data compression. However,
as datasets grow in both cardinality (number of samples) and dimensionality
(number of features), traditional clustering algorithms face significant
computational barriers. The most prominent example is Lloyd's algorithm for
K-Means, which requires a full pass over the dataset at every iteration,
resulting in $O(n k d)$ complexity per iteration — prohibitive at scale.

In many real-world applications, datasets are huge — think of customer records
or image collections. Running traditional K-Means on millions of rows can take
hours or even crash a workstation. So the key question is: does the "mini-batch"
trick actually work? Does it save time without ruining the clusters? And does it
behave differently for "hard" clustering (K-Means) versus "soft" clustering
(Fuzzy C-Means)? We selected two very different datasets to explore this: one
large and tabular, one high-dimensional and image-based.

This study explores *Mini-Batch* variants of two fundamental clustering
paradigms: the "hard" boundary approach of K-Means and the "soft" membership
approach of Fuzzy C-Means (FCM). Mini-batch algorithms leverage stochastic
approximation by updating model parameters using random subsets of data — a
particularly judicious strategy when the dataset is too large to fit in memory,
or when rapid convergence is preferred over exact global optimization. The
objective is to quantify the tradeoff between computational efficiency (runtime)
and clustering quality (Silhouette Score, V-Measure, ARI), providing a rigorous
assessment of when mini-batch stochasticity is preferable to full-batch
exactitude.

== Research Hypotheses

Based on theoretical considerations and intuition from class material, we
formulate the following working hypotheses:

- *H1 — Speed-Quality Tradeoff*: Mini-Batch K-Means will be significantly
  faster than full K-Means, potentially at the cost of slightly lower cluster
  quality (lower Silhouette and V-Measure scores).
- *H2 — FCM Scalability*: Standard Fuzzy C-Means will be severely bottlenecked
  by its $O(n k)$ membership-matrix update; its mini-batch variant should
  provide dramatic speedups — but maybe the noise would actually help it avoid
  bad local minima.
- *H3 — Dimensionality Effect*: High-dimensional Fashion-MNIST will be harder
  for any distance-based method, so preprocessing with PCA will be crucial.

// ── 2. THEORETICAL FRAMEWORK ─────────────────────────────────────────────────
= Theoretical Framework

== K-Means and the Mini-Batch Optimization

The K-Means objective is to minimize total within-cluster variance (Inertia):

$ J(X, mu) = sum_(i=1)^(k) sum_(x in C_i) ||x - mu_i||^2 $

The standard algorithm alternates between an *Assignment Step*, which maps each
point to its nearest centroid, and an *Update Step*, which recomputes centroids
as the cluster mean. The computational bottleneck is the assignment
$x arrow.long scripts(op("argmin"))_i ||x - mu_i||^2$, which scales with all
$n$ points at every iteration.

*Mini-Batch K-Means* (Sculley, 2010) relaxes this constraint by drawing a
random batch $B$ of size $b << n$ at each iteration. For each $x in B$, the
algorithm finds the nearest centroid $mu_i$ and applies an incremental update
weighted by a learning rate $eta$ that decays as the cluster accumulates more
assignments:

$ mu_i arrow.long (1 - eta) mu_i + eta x $

This stochastic gradient descent-like update allows the algorithm to converge
orders of magnitude faster, often reaching an acceptable solution before having
seen the entire dataset even once.

== Fuzzy C-Means (FCM)

FCM introduces the concept of fuzzy sets to clustering. Rather than assigning
each point to a single cluster, each point $x_j$ belongs to all clusters
simultaneously with varying degrees of membership $u_(i j) in [0, 1]$. The
fuzziness is controlled by the parameter $m$, and the objective becomes:

$ J_m = sum_(i=1)^(k) sum_(j=1)^(n) (u_(i j))^m ||x_j - mu_i||^2 $

subject to $sum_(i=1)^k u_(i j) = 1$ for all $j$. When $m = 1$, FCM reduces
to hard K-Means. In practice, $m = 2$ is the standard choice, balancing
sensitivity to distant points without over-smoothing memberships. The fuzzy
approach is particularly meaningful in overlapping or ambiguous data
distributions.

*Mini-Batch FCM* restricts the membership update to only the points in the
current batch. This reduces the memory footprint of the $n times k$ membership
matrix — otherwise $O(n k)$ — to $O(b k)$ per iteration, enabling processing
of datasets that would otherwise not fit in RAM.

// ── 3. EXPERIMENTAL DATA AND PREPROCESSING ───────────────────────────────────
= Experimental Data and Preprocessing

== Covertype: Large-Scale Tabular Data

The *Covertype* dataset (UCI) describes forest cover types derived from
cartographic variables such as elevation, slope, and soil type.

- *Cardinality*: 581,012 samples.
- *Dimensionality*: 54 features (Elevation, Aspect, Slope, Distances, binary
  Soil_Type indicators, etc.).
- *Preprocessing*: We apply *StandardScaler* to standardize all features to
  zero mean and unit variance. This is critical because continuous features like
  "Elevation" (measured in metres) would otherwise numerically dominate binary
  indicator features like "Soil_Type" (values 0 or 1), distorting Euclidean
  distances.
- *Subsampling*: 15,000 samples are used for the main evaluation phase to allow
  10 independent repetitions per algorithm (for statistical robustness) while
  remaining computationally feasible.

#insight-box[
  *Why Covertype?* — It is huge (over half a million rows), making it perfect
  for testing scalability. It also has a known number of classes (7), so we can
  evaluate how well the clusters match the real labels. We subsampled to 15k
  to run multiple trials (10 seeds) without waiting indefinitely, while still
  being representative of the overall distribution.
]

== Fashion-MNIST: High-Dimensional Imagery

Fashion-MNIST consists of $28 times 28$ grayscale images of clothing items
across 10 categories (T-shirts, trousers, pullovers, etc.).

- *Feature Space*: 784 dimensions (one per pixel).
- *Complexity*: Direct clustering in 784D is susceptible to the "Curse of
  Dimensionality," where the contrast between nearest and farthest neighbours
  deteriorates, making Euclidean distance an increasingly poor discriminator.
- *Dimensionality Reduction*: We apply *Principal Component Analysis (PCA)* to
  project the data onto 50 principal components, retaining approximately 84% of
  the total variance. This accelerates distance computations and substantially
  improves the signal-to-noise ratio of the feature space.

#insight-box[
  *Why Fashion-MNIST?* — It is not as large as Covertype (70k samples), but its
  784 dimensions make it a classic "curse of dimensionality" challenge. We wanted
  to see whether mini-batch methods handle high-dimensional spatial data more
  effectively than their full-batch counterparts.
]

#figure(
  image("results/pca_visualization.png", width: 90%),
  caption: [PCA 2D visualisation. Fashion-MNIST (right) shows distinct globular clusters
    corresponding to clothing categories, whereas Covertype (left) exhibits a more
    continuous and overlapping structure.],
)

// ── 4. PARAMETER SELECTION STRATEGY ─────────────────────────────────────────
= Parameter Selection Strategy

As required by the study guidelines, parameter selection must be rigorously
justified rather than arbitrarily chosen.

== Determining the Optimal $k$

We use the *Elbow Method* and *Silhouette Analysis* jointly on a 5,000-sample
subset of the Covertype data.

#figure(
  image("results/parameter_selection_k.png", width: 90%),
  caption: [Justification of $k = 7$ for Covertype. The Elbow plot (Inertia, left) shows
    a clear inflection at $k = 7$; the Silhouette score (right) peaks or stabilises
    in the same region.],
)

The Inertia curve exhibits a clear "bend" at $k = 7$, after which additional
clusters yield diminishing reductions in variance. The Silhouette score
corroborates this choice. This is also consistent with the physical reality of
the dataset, which contains exactly 7 ground-truth cover types.

== Batch Size and Stochasticity

The batch size $b$ governs the speed-quality tradeoff in mini-batch algorithms.
We performed a sweep over $b in {64, 256, 512, 1024, 2048}$ and measured both
runtime and Silhouette score.

#figure(
  image("results/batch_size_analysis.png", width: 90%),
  caption: [Effect of batch size on runtime (left) and Silhouette quality (right). Beyond
    $b = 1024$, quality improvements become marginal while runtime increases
    substantially.],
)

Based on this analysis, we select *$b = 1024$* as our default batch size for
all mini-batch experiments. It provides stable gradient estimates and
reproducible cluster quality while maintaining sub-linear computational cost
relative to the full dataset.

// ── 5. METHODOLOGY ───────────────────────────────────────────────────────────
= Methodology

This section consolidates the technical details of our experimental environment
to ensure full reproducibility.

#v(0.4em)

*Hardware.* #h(0.3em) All experiments were executed on an AMD Ryzen 7 PRO 5850U (8 cores,
16 threads) with 16 GB RAM, running Arch Linux (Kernel 6.18.9). No GPU
acceleration was used, ensuring results reflect CPU-bound performance
characteristics typical of data science workloads.

*Software stack.* #h(0.3em) The analysis was implemented in Python 3.12 within a dedicated
Conda environment. Key libraries include:
- `scikit-learn` (v1.8.0) — K-Means, standard preprocessing, and evaluation metrics.
- `scikit-fuzzy` — Fuzzy C-Means membership update rules.
- `numpy` (v2.4.0), `pandas` (v2.3.3), `matplotlib` — numerical computation and visualisation.

*Experimental protocol.* #h(0.3em) Each algorithm was executed 10 times using distinct
integer random seeds (1 through 10) to assess stability across initialisations.
We fixed $k = 7$ for Covertype (matching ground-truth classes) and $k = 10$ for
Fashion-MNIST (matching the number of clothing categories). The fuzziness
exponent was set to $m = 2.0$ for all FCM variants, as is standard in the
literature. Timing was measured using Python's `time.perf_counter()` to
minimise overhead; only the clustering step itself was timed (excluding
preprocessing).

*Evaluation metrics:*
- *Silhouette Score* — Measures intra-cluster cohesion vs. inter-cluster
  separation. Range: $[-1, 1]$, higher is better.
- *V-Measure* — Harmonic mean of homogeneity and completeness with respect to
  true labels. Range: $[0, 1]$, higher is better.
- *Adjusted Rand Index (ARI)* — Measures agreement between predicted clusters
  and true labels, adjusted for chance. Range: $[-1, 1]$, higher is better.

*Reproducibility.* #h(0.3em) We used default parameters for all algorithms except where
specifically noted (e.g., fuzziness $m = 2.0$ for FCM). The source code and raw
result files are included with this report.

// ── 6. COMPARATIVE RESULTS AND DISCUSSION ────────────────────────────────────
= Comparative Results and Discussion

The following results represent the mean and standard deviation across *10
independent trials* with different random initialisations (seeds).

== Quantitative Performance

#figure(
  styled-table(
    columns: (1.7fr, 1.3fr, 1.3fr, 1fr, 1fr),
    data: (
      ("Algorithm", "Runtime (s)", "Silhouette", "V-Measure", "ARI"),
      ("K-Means",       "0.645 ± 0.38", "0.1131 ± 0.01", "0.176 ± 0.03", "0.080 ± 0.02"),
      ("MB K-Means",    "0.386 ± 0.07", "0.0943 ± 0.01", "0.147 ± 0.04", "0.068 ± 0.02"),
      ("Fuzzy C-Means", "0.571 ± 0.07", "0.0817 ± 0.02", "0.070 ± 0.00", "0.001 ± 0.00"),
      ("MB FCM",        "0.325 ± 0.02", "0.0869 ± 0.01", "0.070 ± 0.00", "0.001 ± 0.00"),
    )
  ),
  caption: [Performance metrics for Covertype ($k = 7$), mean ± std over 10 trials.],
)

#v(0.5em)

#figure(
  styled-table(
    columns: (1.7fr, 1.3fr, 1.3fr, 1fr, 1fr),
    data: (
      ("Algorithm", "Runtime (s)", "Silhouette", "V-Measure", "ARI"),
      ("K-Means",        "1.904 ± 0.41",  "0.1840 ± 0.01", "0.510 ± 0.01", "0.357 ± 0.01"),
      ("MB K-Means",     "0.523 ± 0.08",  "0.1773 ± 0.01", "0.500 ± 0.02", "0.345 ± 0.02"),
      ("FCM (Standard)", "14.359 ± 1.66", "0.0560 ± 0.02", "0.298 ± 0.01", "0.189 ± 0.01"),
      ("MB FCM",         "0.362 ± 0.01",  "0.0799 ± 0.01", "0.372 ± 0.01", "0.270 ± 0.01"),
    )
  ),
  caption: [Performance metrics for Fashion-MNIST ($k = 10$), mean ± std over 10 trials.],
)

== Analysis of Observed Trends

=== The Speed-Precision Tradeoff

On Fashion-MNIST, Mini-Batch K-Means achieved a *3.6× speedup* over standard
K-Means (0.52s vs 1.90s) while retaining over *98% of the V-Measure* (0.500
vs 0.510) and 97% of the ARI (0.345 vs 0.357). This confirms that for
high-dimensional data, stochastic sampling over batches effectively captures
the global cluster structure without computing all $n times k$ pairwise
distances per iteration.

On Covertype, the speedup is more modest (1.7×), because 15k samples is small
enough that full-batch computation is not yet the bottleneck. The quality drop
is also more noticeable here (V-Measure 0.147 vs 0.176), likely because the
dataset's overlapping cluster geometry demands more precise centroid estimation.
This contrast highlights a nuanced message: mini-batch is not universally
superior — its advantage is most pronounced for large $n$ and high $d$.

=== The Curious Case of Mini-Batch FCM

Perhaps the most striking finding is that on Fashion-MNIST, Mini-Batch FCM not
only runs *40 times faster* than standard FCM (0.36s vs 14.36s), but also
achieves *substantially better clustering quality* (ARI 0.270 vs 0.189,
V-Measure 0.372 vs 0.298). This result initially appears counterintuitive.

Full FCM is extremely slow because it updates a 70k × 10 membership matrix each
iteration. The explanation for the quality gain lies in the optimisation
landscape. Standard FCM, by computing memberships over the entire dataset at
each step, follows the gradient very closely — and is therefore susceptible to
early convergence into a poor local minimum. The stochastic noise introduced by
mini-batches effectively acts as a form of *simulated annealing*, perturbing
centroid updates enough to escape suboptimal basins while still making
consistent progress toward a better partition. Mini-batch FCM is thus not merely
an approximation of standard FCM; for datasets with complex manifold structure,
it may be a strictly superior optimisation procedure.

=== Statistical Robustness

The standard deviation of Mini-Batch K-Means runtimes on Fashion-MNIST is
0.07s, compared to 0.41s for standard K-Means — a six-fold reduction in
variance. Mini-batch runtimes are determined primarily by the fixed batch size
and number of iterations, rather than by the variable convergence speed of
Lloyd's algorithm. More predictable execution times are highly desirable in
latency-sensitive or resource-constrained environments such as cloud computing
or real-time inference pipelines.

#figure(
  image("results/boxplots_comparison.png", width: 95%),
  caption: [Boxplots of Silhouette score and runtime across 10 independent trials.
    The lower-right quadrant (Fashion-MNIST runtime) clearly illustrates the
    dramatic runtime reduction of mini-batch methods, particularly Mini-Batch FCM.],
)

// ── 7. SCALABILITY AND COMPLEXITY ANALYSIS ───────────────────────────────────
= Scalability and Complexity Analysis

To empirically demonstrate the complexity advantages, we conducted a scalability
benchmark by varying $n$ from 10,000 to 500,000 samples and measuring the
wall-clock runtime of K-Means versus Mini-Batch K-Means.

#figure(
  image("results/scalability_benchmark.png", width: 75%),
  caption: [Runtime scaling as a function of dataset size $n$. Standard K-Means grows
    linearly with $n$, whereas Mini-Batch K-Means remains approximately constant once
    the batch size is fixed.],
)

Standard K-Means exhibits the expected $O(n k d)$ scaling per iteration, with
runtime growing roughly linearly with $n$. Mini-Batch K-Means, by contrast, has
iteration cost $O(b k d)$ where $b$ is fixed — meaning that once the algorithm's
convergence behaviour is established, the per-iteration cost is independent of
the total dataset size. In the benchmark, Mini-Batch K-Means processes 500,000
samples in approximately the same time it takes standard K-Means to process
10,000. This sub-linear scaling is the key property that makes mini-batch
methods the de facto standard for big-data clustering applications.

// ── 8. LIMITATIONS AND FUTURE WORK ───────────────────────────────────────────
= Limitations and Future Work

Several limitations of this study should be acknowledged.

First, all main experiments used a single fixed batch size ($b = 1024$). While
Section 4.2 justifies this choice via a sensitivity analysis, adaptive batch
size schedules — where $b$ increases over training — could potentially combine
the early speed of small batches with the later precision of larger ones.

Second, we fixed the FCM fuzziness parameter at $m = 2.0$ throughout. Different
values of $m$ may alter the relative performance of FCM variants, particularly
on datasets with strongly overlapping clusters where higher fuzziness could be
beneficial.

Third, all distance computations use the Euclidean metric. For image data,
perceptually-motivated metrics (e.g., Structural Similarity or learned
embeddings from a neural network) may yield better-separated cluster spaces.

Finally, the scalability benchmark was conducted only for K-Means variants. A
full comparison including FCM scalability at $n > 100k$ would complete the
picture. Future work could also explore integrating mini-batch clustering with
approximate nearest-neighbour search structures (e.g., KD-trees, FAISS) to push
scalability further, or experiment on even bigger datasets (millions of rows) to
confirm that the speedup observed here generalises.

// ── 9. CONCLUSION ────────────────────────────────────────────────────────────
= Conclusion

This comparative study demonstrates the practical superiority of mini-batch
clustering for large-scale and high-dimensional data. Our key findings are:

+ *Mini-Batch K-Means* provides massive runtime reductions with negligible
  quality loss for high-dimensional data, retaining over 98% of V-Measure
  quality at 3.6× the speed on Fashion-MNIST.
+ *Mini-Batch FCM* is not merely a faster approximation of standard Fuzzy
  C-Means — on Fashion-MNIST, it achieves better clustering quality
  (ARI 0.270 vs 0.189) by leveraging stochastic updates to escape poor local
  minima.
+ *Preprocessing is indispensable*: StandardScaler for tabular data and PCA
  for imagery are essential prerequisites for meaningful Euclidean
  distance-based clustering.
+ *Statistical predictability*: Mini-batch methods exhibit significantly lower
  runtime variance than their full-batch counterparts, making them more
  suitable for production and real-time environments.

In summary, for datasets exceeding $10^4$ samples, mini-batch algorithms should
be the preferred implementation choice. The stochastic noise introduced by
sampling is not a drawback — it is a feature that enables faster convergence,
lower memory consumption, and in some cases, better optimisation outcomes.

Overall, this homework made me realise that "exact" algorithms are not always
better — sometimes a smart approximation (like mini-batches) can be both faster
and more robust. For large-scale data, mini-batch clustering will be my go-to
approach.
