import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time, os

from Algorithms import KMeansClustering, MiniBatchKMeans, FuzzyCMeans, MiniBatchFuzzyCMeans

os.makedirs('results', exist_ok=True)
np.random.seed(42)

def load_img(path, factor=0.5):
    img = Image.open(path)
    img = img.resize((int(img.size[0]*factor), int(img.size[1]*factor)), Image.LANCZOS)
    if img.mode != 'RGB': img = img.convert('RGB')
    arr = np.array(img)
    px = arr.reshape(-1, 3).astype(np.float64) / 255.0
    print(f"Image: {arr.shape[1]}x{arr.shape[0]}, {px.shape[0]} pixels")
    return arr, px, arr.shape

def load_img_spatial(path, factor=0.5, sw=0.2):
    img = Image.open(path)
    img = img.resize((int(img.size[0]*factor), int(img.size[1]*factor)), Image.LANCZOS)
    if img.mode != 'RGB': img = img.convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    xx, yy = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
    rgb = arr.reshape(-1, 3).astype(np.float64) / 255.0
    sp = np.column_stack([xx.ravel(), yy.ravel()]) * sw
    return arr, np.column_stack([rgb, sp]), arr.shape

def seg(arr, px, shape, model, k):
    lb = model.predict(px)
    c = model.get_centroids()
    s = np.clip(c[lb][:, :3], 0, 1)
    return (s.reshape(shape[0], shape[1], 3)*255).astype(np.uint8), lb

path = 'datasets/image.png'
if not os.path.exists(path):
    print(f"Not found: {path}"); exit(1)

arr, px, shape = load_img(path, 0.5)
k = 5

results = {}

print(f"K-Means (k={k})...")
t0 = time.time()
m = KMeansClustering(px, n_clusters=k, max_iter=100, random_state=42); m.fit()
print(f"  {time.time()-t0:.2f}s")
results['K-Means'] = seg(arr, px, shape, m, k)

print(f"MiniBatch K-Means (k={k})...")
t0 = time.time()
m = MiniBatchKMeans(n_clusters=k, batch_size=1000, max_iter=100, random_state=42); m.fit(px)
print(f"  {time.time()-t0:.2f}s")
results['MiniBatch K-Means'] = seg(arr, px, shape, m, k)

print(f"Fuzzy C-Means (k={k})...")
t0 = time.time()
m = FuzzyCMeans(px, n_clusters=k, max_iter=100, random_state=42); m.fit()
print(f"  {time.time()-t0:.2f}s")
results['Fuzzy C-Means'] = seg(arr, px, shape, m, k)

for name, (si, lb) in results.items():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(arr); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(si); axes[1].set_title(f'{name} ({k} seg)'); axes[1].axis('off')
    axes[2].imshow(lb.reshape(shape[0], shape[1]), cmap='tab10'); axes[2].set_title('Cluster Map'); axes[2].axis('off')
    plt.tight_layout()
    fn = name.lower().replace(' ', '_').replace('-', '')
    plt.savefig(f'results/{fn}_segmentation.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"Saved {fn}_segmentation.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes[0,0].imshow(arr); axes[0,0].set_title('Original', fontweight='bold'); axes[0,0].axis('off')
for i, (n, (si, _)) in enumerate(results.items()):
    r, c = divmod(i+1, 2)
    axes[r,c].imshow(si); axes[r,c].set_title(f'{n} ({k} seg)', fontweight='bold'); axes[r,c].axis('off')
plt.tight_layout()
plt.savefig('results/all_methods_comparison.png', dpi=200, bbox_inches='tight'); plt.close()
print("Saved all_methods_comparison.png")

ks = [3, 5, 7, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.ravel()
for i, kk in enumerate(ks):
    print(f"k={kk}...")
    m = KMeansClustering(px, n_clusters=kk, max_iter=100, random_state=42); m.fit()
    si, _ = seg(arr, px, shape, m, kk)
    axes[i].imshow(si); axes[i].set_title(f'k={kk}', fontweight='bold'); axes[i].axis('off')
plt.tight_layout()
plt.savefig('results/kmeans_multiple_clusters.png', dpi=200, bbox_inches='tight'); plt.close()
print("Saved kmeans_multiple_clusters.png")

arr_sp, px_sp, shape_sp = load_img_spatial(path, 0.5, 0.2)
m = KMeansClustering(px_sp, n_clusters=5, max_iter=100, random_state=42); m.fit()
seg_sp, _ = seg(arr_sp, px_sp, shape_sp, m, 5)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(arr); axes[0].set_title('Original', fontweight='bold'); axes[0].axis('off')
axes[1].imshow(results['K-Means'][0]); axes[1].set_title('Color Only', fontweight='bold'); axes[1].axis('off')
axes[2].imshow(seg_sp); axes[2].set_title('Color + Spatial', fontweight='bold'); axes[2].axis('off')
plt.tight_layout()
plt.savefig('results/spatial_vs_color_only.png', dpi=200, bbox_inches='tight'); plt.close()
print("Saved spatial_vs_color_only.png")

print("Done.")
