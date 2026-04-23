# 04. Kernel k-Means

## 🎯 핵심 질문

- Kernel k-means는 어떻게 **명시적 feature $\phi$ 없이** feature 공간에서의 k-means를 실행하는가?
- Distance $\|\phi(x_i) - \mu_k\|^2$를 **Gram 행렬만으로** 계산하는 공식은?
- Kernel k-means와 **spectral clustering**의 관계 — **가중 kernel k-means**가 spectral clustering과 equivalent (Dhillon et al. 2004)?
- 임의 모양 클러스터를 포착할 수 있는 이유는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

K-means는 **빠르지만 convex cluster만 찾는** 한계. Kernel k-means는 "feature space에서 k-means"로 이 한계를 우회해 **비구형 cluster 포착**. 특히 (i) 구현 간단 (k-means와 거의 동일), (ii) spectral clustering의 multiplicative weight variant와 equivalence로 **두 알고리즘의 통합 관점 제공**, (iii) Nyström sketching과 결합해 scalable. 실무에서는 spectral clustering의 대안으로 **scaling 더 좋음** (eigendecomp 불필요). Chang & Yeung (2008) "Robust kernel k-means"는 outlier에도 robust.

---

## 📐 수학적 선행 조건

- [Ch5-02 Kernel PCA](./02-kernel-pca.md)
- [Ch5-03 Spectral Clustering](./03-spectral-clustering.md)
- 기초 알고리즘: K-means (Lloyd's algorithm), expectation-maximization

---

## 📖 직관적 이해

### K-means의 Kernel 확장

**Standard K-means**: $\mathcal{X} = \mathbb{R}^d$, cluster means $\mu_k \in \mathbb{R}^d$, assignment:

$$c_i = \arg\min_k \|x_i - \mu_k\|^2.$$

Update: $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$.

**Kernel K-means**: $\phi(x_i) \in \mathcal{H}$, cluster means $\mu_k \in \mathcal{H}$:

$$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} \phi(x_i).$$

Distance in feature space:

$$\|\phi(x_i) - \mu_k\|_{\mathcal{H}}^2 = \langle \phi(x_i), \phi(x_i) \rangle - 2 \langle \phi(x_i), \mu_k \rangle + \langle \mu_k, \mu_k \rangle$$

$$= k(x_i, x_i) - \frac{2}{|C_k|} \sum_{j \in C_k} k(x_i, x_j) + \frac{1}{|C_k|^2} \sum_{j, l \in C_k} k(x_j, x_l).$$

**$\mu_k$는 명시적 계산 불필요**, **Gram** $K$만 있으면 됨.

### 알고리즘 (Scholkopf et al. 1998)

1. Initialize cluster assignment $c_i \in \{1, \ldots, k\}$.
2. **Repeat**:
   a. For each $i, k$, compute $d_{ik} = \|\phi(x_i) - \mu_k\|_{\mathcal{H}}^2$ using Gram.
   b. $c_i = \arg\min_k d_{ik}$.
   c. Update cluster memberships.
3. **Until** no change.

### 복잡도

각 iter마다:
- All $n \times k$ distances: $O(n k)$ distance 계산. 각 $d_{ik}$는 $|C_k|$ 관련 $O(|C_k|^2)$ — 총 $O(n^2)$ per iter.
- 최악 $O(n^2 T)$, $T$ = iterations.

메모리 $O(n^2)$ (Gram). $n \leq 10^4$에서 실용적.

### 비구형 Cluster 포착

RBF kernel이면 feature space에서 $\phi(x_i)$가 구형 구조를 잘 유지 → feature space k-means가 **원래 공간의 임의 모양 cluster 구별 가능**.

예: Two moons 데이터 → feature space에서 각 moon이 "쉽게 분리되는 구형" → kernel k-means 성공.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Kernel K-means Objective

$$\min_{c_1, \ldots, c_n \in \{1, \ldots, k\}} \sum_{j=1}^k \sum_{i \in C_j} \|\phi(x_i) - \mu_j\|_{\mathcal{H}}^2$$

where $\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} \phi(x_i)$.

### 정의 4.2 — Distance Computation

$$d_{ij}^2 = \|\phi(x_i) - \mu_j\|^2 = K_{ii} - \frac{2}{|C_j|} \sum_{l \in C_j} K_{il} + \frac{1}{|C_j|^2} \sum_{l, m \in C_j} K_{lm}.$$

### 정의 4.3 — Weighted Kernel K-means

Weights $w_i \geq 0$로 일반화:

$$\min_{c} \sum_j \sum_{i \in C_j} w_i \|\phi(x_i) - \mu_j\|^2, \quad \mu_j = \frac{\sum_{i \in C_j} w_i \phi(x_i)}{\sum_{i \in C_j} w_i}.$$

---

## 🔬 정리와 증명

### 정리 4.1 — Distance의 Gram-Only 표현

**명제**: 정의 4.2의 공식이 성립. $\phi$는 명시적으로 필요 없음.

**증명**: 

$\|\phi(x_i) - \mu_j\|^2 = \langle \phi(x_i) - \mu_j, \phi(x_i) - \mu_j \rangle$

$= \langle \phi(x_i), \phi(x_i) \rangle - 2 \langle \phi(x_i), \mu_j \rangle + \langle \mu_j, \mu_j \rangle$

$= k(x_i, x_i) - 2 \langle \phi(x_i), \frac{1}{|C_j|} \sum_{l \in C_j} \phi(x_l) \rangle + \langle \frac{1}{|C_j|} \sum_{l \in C_j} \phi(x_l), \frac{1}{|C_j|} \sum_{m \in C_j} \phi(x_m) \rangle$

$= K_{ii} - \frac{2}{|C_j|} \sum_l K_{il} + \frac{1}{|C_j|^2} \sum_{l, m} K_{lm}. \quad \square$

### 정리 4.2 — Kernel k-means의 수렴

**명제**: Kernel k-means는 **monotone** objective 감소를 보장 (standard k-means와 마찬가지). 유한 step 내 수렴 (local minimum).

**증명**: Assignment 단계 또는 re-computation 단계 모두 objective 감소 또는 유지. Finite cluster assignments → 유한 step. $\square$

**Local minimum**: Global 아님. 여러 initialization + best selection 필요.

### 정리 4.3 — Kernel K-means ⇔ Spectral Clustering (Dhillon et al. 2004)

**명제**: **Weighted kernel k-means**는 **normalized cut**의 objective와 equivalent.

구체적으로, similarity $K = W$, weights $w_i = d_i$ (degree)로 놓으면 weighted kernel k-means = normalized cut relaxation (Ch5-03).

**증명 개요**:

NCut objective (Ch5-03 정리 3.3):
$$\text{NCut}(c) = \sum_j \frac{\text{links}(C_j, C_j^c)}{\text{degree}(C_j)}.$$

Weighted kernel k-means objective ($w_i = d_i$, kernel = $W$):
$$\sum_j \sum_{i \in C_j} d_i \left(W_{ii} - \frac{2}{\sum_{l \in C_j} d_l} \sum_{l \in C_j} d_l W_{il} + \cdots\right).$$

대수 전개로 NCut과 동등 (세부: Dhillon et al.).

**함의**: **두 알고리즘이 통합됨** — 한쪽이 eigendecomp 기반 ($O(n^3)$), 다른 쪽이 iterative k-means 기반 ($O(n^2 T)$). 큰 $n$에서 kernel k-means 선호.

### 정리 4.4 — Initialization의 중요성

**명제**: Kernel k-means의 해는 **initialization에 의존**. Global minimum 보장 없음.

**실무 초기화**:
1. **K-means++**: 데이터 기반 greedy initialization. Feature 공간 distance 사용 가능 (kernel k-means++).
2. **Spectral initialization**: Spectral clustering의 embedding으로 초기 clustering 후 kernel k-means로 refine.
3. **Multi-start**: 10~100 random init, best (minimum objective) 선택.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons, make_blobs

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. 비구형 데이터 — moons
# ─────────────────────────────────────────────
X, labels_true = make_moons(n_samples=200, noise=0.08, random_state=0)
n = len(X)

# ─────────────────────────────────────────────
# 2. Kernel K-means bottom-up
# ─────────────────────────────────────────────
def rbf_kernel(X, Y, s=0.3):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K = rbf_kernel(X, X)

def kernel_kmeans(K, k, max_iter=100, seed=0):
    rng_local = np.random.default_rng(seed)
    n = len(K)
    # Random initialization
    c = rng_local.integers(0, k, n)
    
    for iteration in range(max_iter):
        # Compute d_ij^2 for each (i, j)
        d2 = np.zeros((n, k))
        for j in range(k):
            C_j = np.where(c == j)[0]
            if len(C_j) == 0:
                d2[:, j] = np.inf
                continue
            # K_ii - (2/|C|) Σ_l K_il - (1/|C|^2) Σ_{l, m} K_lm
            K_ii = np.diag(K)
            term2 = K[:, C_j].sum(axis=1) * 2 / len(C_j)
            term3 = K[np.ix_(C_j, C_j)].sum() / (len(C_j) ** 2)
            d2[:, j] = K_ii - term2 + term3
        
        c_new = np.argmin(d2, axis=1)
        if np.array_equal(c, c_new):
            print(f'수렴: iter {iteration}')
            break
        c = c_new
    return c

# 여러 random init, best selected
best_obj = np.inf
best_labels = None
for seed in range(10):
    labels = kernel_kmeans(K, k=2, seed=seed)
    # Compute total within-cluster distance
    obj = 0
    for j in range(2):
        C_j = np.where(labels == j)[0]
        if len(C_j) == 0:
            continue
        for i in C_j:
            # d^2
            K_ii = K[i, i]
            term2 = 2 / len(C_j) * K[i, C_j].sum()
            term3 = K[np.ix_(C_j, C_j)].sum() / len(C_j) ** 2
            obj += K_ii - term2 + term3
    if obj < best_obj:
        best_obj = obj
        best_labels = labels

# ─────────────────────────────────────────────
# 3. 비교
# ─────────────────────────────────────────────
km_direct = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1/(2*0.3**2), random_state=0)
sc_labels = sc.fit_predict(X)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_true, cmap='coolwarm', s=20)
axes[0].set_title('Ground Truth')

axes[1].scatter(X[:, 0], X[:, 1], c=km_direct, cmap='coolwarm', s=20)
axes[1].set_title('K-means (fails)')

axes[2].scatter(X[:, 0], X[:, 1], c=best_labels, cmap='coolwarm', s=20)
axes[2].set_title('Kernel K-means (RBF)')

axes[3].scatter(X[:, 0], X[:, 1], c=sc_labels, cmap='coolwarm', s=20)
axes[3].set_title('Spectral Clustering')

plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 4. 다른 kernel 시도 — Polynomial
# ─────────────────────────────────────────────
def poly_kernel(X, Y, c=1, d=3):
    return (X @ Y.T + c) ** d

# Blob data (convex) → polynomial kernel이 유용할 수도 vs. 그냥 k-means
X_blob, labels_blob = make_blobs(n_samples=200, centers=3, random_state=0)
K_poly = poly_kernel(X_blob, X_blob)
# Center & scale for numerical stability
K_poly = K_poly / K_poly.max()

labels_poly = kernel_kmeans(K_poly, k=3, seed=0)
km_blob = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X_blob)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=labels_blob, cmap='tab10', s=20)
axes[0].set_title('Blobs GT')
axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=km_blob, cmap='tab10', s=20)
axes[1].set_title('K-means (blobs = convex → OK)')
axes[2].scatter(X_blob[:, 0], X_blob[:, 1], c=labels_poly, cmap='tab10', s=20)
axes[2].set_title('Poly Kernel K-means')
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 5. Scaling 검증 — $O(n^2)$ per iter
# ─────────────────────────────────────────────
import time
for n_size in [200, 500, 1000]:
    X_s, _ = make_moons(n_samples=n_size, noise=0.08)
    K_s = rbf_kernel(X_s, X_s, s=0.3)
    
    t0 = time.time()
    kernel_kmeans(K_s, k=2, max_iter=50)
    t_kkm = time.time() - t0
    
    t0 = time.time()
    SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0).fit_predict(K_s)
    t_sc = time.time() - t0
    
    print(f'n={n_size}: Kernel K-means {t_kkm:.3f}s, Spectral {t_sc:.3f}s')
```

**출력 예시**:
```
n=200: Kernel K-means 0.012s, Spectral 0.089s
n=500: Kernel K-means 0.043s, Spectral 0.531s
n=1000: Kernel K-means 0.152s, Spectral 3.642s
```

→ Kernel k-means가 spectral clustering보다 빠르게 scaling. Moons에서 K-means 실패, kernel k-means 성공.

---

## 🔗 실전 활용

- **Alternative to spectral clustering**: 더 빠른 $O(n^2 T)$ vs $O(n^3)$, large $n$에서 유리.
- **Text clustering**: String kernel·BM25 similarity로 문서 clustering.
- **Bioinformatics**: Sequence kernels로 DNA/protein sequence clustering.
- **Streaming/online**: Online kernel k-means variants — incremental cluster update.
- **Scalability**: Nyström-based kernel k-means로 $n > 10^5$ 가능.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Local minima | Multi-start 필요 |
| $k$ 지정 | Pre-specified (elbow, silhouette, BIC로 heuristic) |
| **Empty clusters** | Iteration 중 cluster size 0 가능 → 재초기화 |
| $O(n^2)$ memory | Gram 저장 |
| **Kernel choice** | RBF $\sigma$ 선택 critical |

---

## 📌 핵심 정리

$$\boxed{d_{ij}^2 = K_{ii} - \frac{2}{|C_j|} \sum_{l \in C_j} K_{il} + \frac{1}{|C_j|^2} \sum_{l, m \in C_j} K_{lm}}$$

$$\boxed{\text{Weighted Kernel K-means} \iff \text{Normalized Cut} \text{ (Dhillon 2004)}}$$

| 알고리즘 | Complexity | Eigendecomp | 비구형 cluster |
|----------|-----------|-------------|----------------|
| K-means | $O(nk)$ per iter | No | No |
| Kernel K-means | $O(n^2)$ per iter | No | Yes |
| Spectral Clustering | $O(n^3)$ | Yes | Yes |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Kernel k-means에서 새 점 $x_*$ (training 아닌)의 cluster assignment는 어떻게 하는가?

<details>
<summary>힌트 및 해설</summary>

Trained assignment $\{c_i\}$, cluster $C_j$로부터:

$$d^2(x_*, \mu_j) = k(x_*, x_*) - \frac{2}{|C_j|} \sum_{l \in C_j} k(x_*, x_l) + \frac{1}{|C_j|^2} \sum_{l, m \in C_j} k(x_l, x_m).$$

마지막 항은 training 시 저장 가능 (cluster별 상수). 두 번째 항은 새 점과 각 cluster 점들의 kernel 계산 — $O(|C_j|)$ per cluster.

$c_* = \arg\min_j d^2(x_*, \mu_j)$.

**실무**: Cluster center를 명시적으로 저장 못 하지만, "cluster 멤버들의 set"은 저장 → prediction 시 필요.

</details>

**문제 2** (심화): Kernel k-means가 **K-means의 일반화**임을 직접 확인하라.

<details>
<summary>힌트 및 해설</summary>

Linear kernel $k(x, y) = x^\top y$이면:

- $K_{ii} = x_i^\top x_i = \|x_i\|^2$.
- $\sum_{l \in C_j} K_{il} = x_i^\top \sum_l x_l = |C_j| \cdot x_i^\top \bar{x}_j$.
- $\sum_{l, m} K_{lm} = (\sum_l x_l)^\top (\sum_m x_m) = |C_j|^2 \|\bar{x}_j\|^2$.

$$d_{ij}^2 = \|x_i\|^2 - 2 x_i^\top \bar{x}_j + \|\bar{x}_j\|^2 = \|x_i - \bar{x}_j\|^2.$$

**정확히 standard k-means distance**. ✓

**해석**: Kernel k-means는 linear kernel이면 standard k-means, non-linear kernel이면 feature space k-means.

</details>

**문제 3** (ML 연결): Kernel k-means의 **deep learning 버전** (DeepCluster, Caron et al. 2018)의 핵심 아이디어는?

<details>
<summary>힌트 및 해설</summary>

**DeepCluster**:
1. CNN $f_\theta$로 unlabeled 이미지에서 feature 추출.
2. 이 feature들을 **k-means**로 clustering.
3. Cluster 번호를 **pseudo-label**로 사용해 CNN 재학습 (classification loss).
4. 반복.

**Kernel k-means 관점**:
- $f_\theta$는 learned **implicit feature map** — neural kernel $k(x, y) := f_\theta(x)^\top f_\theta(y)$.
- K-means on $\{f_\theta(x_i)\}$ = kernel k-means with this learned kernel.
- 핵심 차이: **Kernel $f_\theta$가 fixed가 아닌 learning된다** → clustering이 feature learning을 유도.

**Connection to self-supervised**: DeepCluster는 초기 self-supervised learning의 한 접근. 현재 SimCLR·MoCo·DINO 등이 더 성공적이지만, DeepCluster의 "kernel k-means as self-supervision"은 개념적으로 중요.

**NTK 관점**: NN의 feature가 NTK를 정의 → "NN training + clustering" = "evolving kernel k-means".

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Spectral Clustering — Graph Laplacian](./03-spectral-clustering.md) | [Ch6-01. MMD의 정의와 RKHS 해석 ▶](../ch6-mmd/01-mmd-definition.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
