import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, v_measure_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_covtype, fetch_openml, make_blobs
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time, os

from Algorithms import KMeansClustering, MiniBatchKMeans, FuzzyCMeans, MiniBatchFuzzyCMeans

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs('results', exist_ok=True)
np.random.seed(42)

N = 15000

print("Loading Covertype...")
cov = fetch_covtype(download_if_missing=False)
idx = np.random.choice(len(cov.data), N, replace=False)
X_cov = StandardScaler().fit_transform(cov.data[idx])
y_cov = cov.target[idx]
print(f"  {X_cov.shape} subsampled from {len(cov.data)}")

print("Loading Fashion-MNIST...")
fm = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X_fm_raw = fm.data.astype(np.float64) / 255.0
y_fm_all = fm.target.astype(int)
pca = PCA(n_components=50, random_state=42)
X_fm_pca = pca.fit_transform(X_fm_raw)
idx2 = np.random.choice(len(X_fm_pca), N, replace=False)
X_fm = X_fm_pca[idx2]
y_fm = y_fm_all[idx2]
print(f"  {X_fm.shape} (PCA 50d, {pca.explained_variance_ratio_.sum():.1%} var)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
p1 = PCA(n_components=2).fit_transform(X_cov[:3000])
axes[0].scatter(p1[:, 0], p1[:, 1], c=y_cov[:3000], cmap='tab10', alpha=0.4, s=5)
axes[0].set_title('Covertype (PCA)')
p2 = PCA(n_components=2).fit_transform(X_fm[:3000])
axes[1].scatter(p2[:, 0], p2[:, 1], c=y_fm[:3000], cmap='tab10', alpha=0.4, s=5)
axes[1].set_title('Fashion-MNIST (PCA)')
plt.tight_layout()
plt.savefig('results/pca_visualization.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved pca_visualization.png")

print("\nParameter selection...")
sub = X_cov[:5000]
k_range = range(2, 13)
inertias, sils = [], []
for k in k_range:
    km = KMeansClustering(sub, n_clusters=k, max_iter=50, random_state=42)
    km.fit()
    inertias.append(km.get_inertia())
    sils.append(silhouette_score(sub, km.predict(sub), sample_size=3000))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, 'bo-')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method')
axes[1].plot(list(k_range), sils, 'ro-')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette'); axes[1].set_title('Silhouette Analysis')
plt.tight_layout()
plt.savefig('results/parameter_selection_k.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved parameter_selection_k.png")

batch_sizes = [64, 256, 512, 1024, 2048]
bres = []
for bs in batch_sizes:
    t0 = time.time()
    mb = MiniBatchKMeans(n_clusters=7, batch_size=bs, max_iter=100, random_state=42)
    mb.fit(sub)
    rt = time.time() - t0
    sil = silhouette_score(sub, mb.predict(sub), sample_size=3000)
    bres.append({'bs': bs, 'rt': rt, 'sil': sil})
bdf = pd.DataFrame(bres)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar([str(b) for b in batch_sizes], bdf['rt'], color='steelblue')
axes[0].set_xlabel('Batch Size'); axes[0].set_ylabel('Runtime (s)'); axes[0].set_title('Batch Size vs Runtime')
axes[1].plot(batch_sizes, bdf['sil'], 'go-', linewidth=2)
axes[1].set_xlabel('Batch Size'); axes[1].set_ylabel('Silhouette'); axes[1].set_title('Batch Size vs Quality')
plt.tight_layout()
plt.savefig('results/batch_size_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved batch_size_analysis.png")

def evaluate(cls, X, y, k, seeds=range(1, 11), **kw):
    res = {'seed': [], 'runtime': [], 'silhouette': [], 'v_measure': [], 'ari': []}
    sn = min(5000, len(X))
    for s in seeds:
        t0 = time.time()
        if cls in [MiniBatchKMeans, MiniBatchFuzzyCMeans]:
            m = cls(n_clusters=k, random_state=s, **kw); m.fit(X)
        else:
            m = cls(X, n_clusters=k, random_state=s, **kw); m.fit()
        rt = time.time() - t0
        lb = m.predict(X)
        res['seed'].append(s); res['runtime'].append(rt)
        res['silhouette'].append(silhouette_score(X, lb, sample_size=sn))
        res['v_measure'].append(v_measure_score(y, lb))
        res['ari'].append(adjusted_rand_score(y, lb))
    return pd.DataFrame(res)

algos = {
    'KMeans': (KMeansClustering, {}),
    'MiniBatch KMeans': (MiniBatchKMeans, {'batch_size': 1024}),
    'Fuzzy C-Means': (FuzzyCMeans, {'m': 2.0}),
    'MiniBatch FCM': (MiniBatchFuzzyCMeans, {'batch_size': 1024, 'm': 2.0})
}

print("\n=== Covertype (k=7) ===")
r_cov = {}
for name, (cls, p) in algos.items():
    print(f"  {name}...", end=" ", flush=True)
    r_cov[name] = evaluate(cls, X_cov, y_cov, 7, **p)
    print(f"rt={r_cov[name]['runtime'].mean():.2f}s sil={r_cov[name]['silhouette'].mean():.4f}")

print("\n=== Fashion-MNIST (k=10) ===")
r_fm = {}
for name, (cls, p) in algos.items():
    print(f"  {name}...", end=" ", flush=True)
    r_fm[name] = evaluate(cls, X_fm, y_fm, 10, **p)
    print(f"rt={r_fm[name]['runtime'].mean():.2f}s sil={r_fm[name]['silhouette'].mean():.4f}")

def summarize(rd):
    rows = []
    for n, df in rd.items():
        rows.append({'Algorithm': n,
            'Runtime': f"{df['runtime'].mean():.3f} +/- {df['runtime'].std():.3f}",
            'Silhouette': f"{df['silhouette'].mean():.4f} +/- {df['silhouette'].std():.4f}",
            'V-Measure': f"{df['v_measure'].mean():.4f} +/- {df['v_measure'].std():.4f}",
            'ARI': f"{df['ari'].mean():.4f} +/- {df['ari'].std():.4f}"})
    return pd.DataFrame(rows)

print("\n--- Covertype ---")
sc = summarize(r_cov); print(sc.to_string(index=False)); sc.to_csv('results/covertype_summary.csv', index=False)
print("\n--- Fashion-MNIST ---")
sf = summarize(r_fm); print(sf.to_string(index=False)); sf.to_csv('results/fmnist_summary.csv', index=False)

ds = [(r_cov, "Covertype"), (r_fm, "Fashion-MNIST")]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for col, (res, t) in enumerate(ds):
    d = [df['silhouette'].values for df in res.values()]
    bp = axes[0, col].boxplot(d, labels=list(res.keys()), patch_artist=True)
    for patch, c in zip(bp['boxes'], sns.color_palette("husl", 4)): patch.set_facecolor(c)
    axes[0, col].set_title(f'{t} — Silhouette'); axes[0, col].tick_params(axis='x', rotation=25)
    d2 = [df['runtime'].values for df in res.values()]
    bp2 = axes[1, col].boxplot(d2, labels=list(res.keys()), patch_artist=True)
    for patch, c in zip(bp2['boxes'], sns.color_palette("husl", 4)): patch.set_facecolor(c)
    axes[1, col].set_title(f'{t} — Runtime (s)'); axes[1, col].tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig('results/boxplots_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved boxplots_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for col, (res, t) in enumerate(ds):
    x = np.arange(len(res)); w = 0.35
    vm = [df['v_measure'].mean() for df in res.values()]
    ar = [df['ari'].mean() for df in res.values()]
    axes[col].bar(x - w/2, vm, w, label='V-Measure', color='steelblue')
    axes[col].bar(x + w/2, ar, w, label='ARI', color='coral')
    axes[col].set_xticks(x); axes[col].set_xticklabels(list(res.keys()), rotation=25)
    axes[col].set_title(f'{t}'); axes[col].legend(); axes[col].set_ylabel('Score')
plt.tight_layout()
plt.savefig('results/vmeasure_ari_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved vmeasure_ari_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for col, (res, t) in enumerate(ds):
    for n, df in res.items():
        axes[col].scatter(df['runtime'].mean(), df['silhouette'].mean(), s=200, label=n, zorder=5)
        axes[col].annotate(n, (df['runtime'].mean(), df['silhouette'].mean()),
                          textcoords="offset points", xytext=(10, 5), fontsize=8)
    axes[col].set_xlabel('Runtime (s)'); axes[col].set_ylabel('Silhouette')
    axes[col].set_title(f'{t}'); axes[col].legend(fontsize=7)
plt.tight_layout()
plt.savefig('results/runtime_vs_quality.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved runtime_vs_quality.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, met in enumerate(['silhouette', 'v_measure']):
    for j, (res, t) in enumerate(ds):
        for n, df in res.items():
            axes[i, j].plot(df['seed'], df[met], marker='o', label=n, linewidth=1.5)
        axes[i, j].set_xlabel('Seed'); axes[i, j].set_ylabel(met.replace('_', ' ').title())
        axes[i, j].set_title(f'{t}'); axes[i, j].legend(fontsize=7)
plt.tight_layout()
plt.savefig('results/convergence_stability.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved convergence_stability.png")

print("\nStatistical tests (silhouette):")
for res, t in ds:
    names = list(res.keys()); n = len(names)
    pv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pv[i,j] = 1.0 if i==j else stats.ttest_ind(res[names[i]]['silhouette'], res[names[j]]['silhouette'])[1]
    print(f"\n{t}:")
    print(pd.DataFrame(pv, index=names, columns=names).round(4))

rows = []
for res, dn in ds:
    for a, df in res.items():
        rows.append({'Dataset': dn, 'Algorithm': a,
            'Runtime_mean': df['runtime'].mean(), 'Runtime_std': df['runtime'].std(),
            'Silhouette_mean': df['silhouette'].mean(), 'Silhouette_std': df['silhouette'].std(),
            'V_Measure_mean': df['v_measure'].mean(), 'V_Measure_std': df['v_measure'].std(),
            'ARI_mean': df['ari'].mean(), 'ARI_std': df['ari'].std()})
pd.DataFrame(rows).to_csv('results/detailed_comparison.csv', index=False)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
viz = X_cov[:3000]; yv = y_cov[:3000]
p2d = PCA(n_components=2).fit_transform(viz)
for col, (name, (cls, par)) in enumerate(algos.items()):
    if cls in [MiniBatchKMeans, MiniBatchFuzzyCMeans]:
        m = cls(n_clusters=7, random_state=42, **par); m.fit(viz)
    else:
        m = cls(viz, n_clusters=7, random_state=42, **par); m.fit()
    pred = m.predict(viz)
    axes[0, col].scatter(p2d[:, 0], p2d[:, 1], c=yv, cmap='tab10', alpha=0.3, s=3)
    axes[0, col].set_title('True Labels')
    axes[1, col].scatter(p2d[:, 0], p2d[:, 1], c=pred, cmap='tab10', alpha=0.3, s=3)
    axes[1, col].set_title(name)
axes[0, 0].set_ylabel('True'); axes[1, 0].set_ylabel('Predicted')
plt.suptitle('Covertype — Clustering (PCA)', fontweight='bold')
plt.tight_layout()
plt.savefig('results/clustering_pca_covertype.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved clustering_pca_covertype.png")

print("\n=== Scalability ===")
sizes = [10000, 50000, 100000, 500000]
kt, mt = [], []
for n in sizes:
    X_b, _ = make_blobs(n_samples=n, centers=5, n_features=10, random_state=42)
    t0 = time.time()
    KMeansClustering(X_b, n_clusters=5, max_iter=50, random_state=42).fit()
    kt.append(time.time() - t0)
    t0 = time.time()
    mb = MiniBatchKMeans(n_clusters=5, batch_size=1024, max_iter=50, random_state=42); mb.fit(X_b)
    mt.append(time.time() - t0)
    print(f"  N={n}: KM={kt[-1]:.2f}s MB={mt[-1]:.2f}s x{kt[-1]/mt[-1]:.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(sizes)); w = 0.35
ax.bar(x - w/2, kt, w, label='KMeans', color='skyblue')
ax.bar(x + w/2, mt, w, label='MiniBatch', color='salmon')
ax.set_xticks(x); ax.set_xticklabels([f'{s//1000}K' for s in sizes])
ax.set_xlabel('N'); ax.set_ylabel('Runtime (s)'); ax.set_title('Scalability')
ax.legend()
plt.tight_layout()
plt.savefig('results/scalability_benchmark.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved scalability_benchmark.png")

print("\nDone.")
