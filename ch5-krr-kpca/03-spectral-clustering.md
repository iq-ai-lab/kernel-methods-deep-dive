# 03. Spectral Clustering — Graph Laplacian

## 🎯 핵심 질문

- **Graph Laplacian** $L = D - W$에서 $W$ (similarity matrix)와 $D$ (degree matrix)의 역할은?
- $L$의 **작은 고유값의 고유벡터**로 clustering이 왜 가능한가?
- Spectral clustering이 왜 **Kernel PCA의 특수 사례**이고, **Normalized Cut** (Shi & Malik 1997)과 어떻게 연결되는가?
- Unnormalized vs Normalized Laplacian의 차이는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

K-means는 convex cluster만 찾을 수 있어 moons/circles 같은 비구형 클러스터에 실패. Spectral clustering은 "**graph에서의 minimum cut**" 관점으로 이 문제 해결. Kernel PCA의 특수 사례로 해석되어 **kernel method framework**에 통합되고, Normalized Cut의 relaxation으로 조합최적화 문제와도 연결. 실무에서는 (i) **image segmentation** (Shi & Malik), (ii) **community detection** (graph data), (iii) **manifold learning** (지구통계학). 또한 Diffusion maps·Laplacian Eigenmaps·t-SNE와도 이론적 기반 공유.

---

## 📐 수학적 선행 조건

- [Ch5-02 Kernel PCA](./02-kernel-pca.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 스펙트럴 분해, Rayleigh quotient, Courant-Fischer
- 그래프 이론 기초: weighted graph, adjacency, degree

---

## 📖 직관적 이해

### Graph Laplacian의 구성

Similarity $w_{ij} = k(x_i, x_j)$ (RBF 등)으로 edge weight. Adjacency $W = [w_{ij}]$. Degree $d_i = \sum_j w_{ij}$, $D = \text{diag}(d_i)$.

**Unnormalized Laplacian**: $L = D - W$.

$L$의 성질:
- 대칭, PSD (모든 고유값 $\geq 0$).
- $L \mathbf{1} = 0$ → $\lambda_1 = 0$, 고유벡터 $\mathbf{1}$.
- Connected components: $\dim \ker L$ = components 수.

### 왜 작은 고유값의 고유벡터가 Cluster 찾는가

$L$의 Rayleigh quotient:

$$f^\top L f = \sum_{i, j} w_{ij} (f_i - f_j)^2 / 2.$$

**해석**: $f$가 "**많이 변하지 않는**" 함수 (유사한 점들에서 값 비슷)일수록 작은 $f^\top L f$. 즉 "cluster 내에서 일정"한 함수.

Top-$k$ smallest non-zero eigenvalues의 eigenvectors = "$k$개 cluster를 잘 구별하는" 지표 함수의 relaxation.

### Normalized Cut (Shi & Malik 1997)

Graph partitioning 목표: $A, B = V \setminus A$로 분할, 다음을 최소화:

$$\text{NCut}(A, B) := \frac{\text{cut}(A, B)}{\text{vol}(A)} + \frac{\text{cut}(A, B)}{\text{vol}(B)}$$

$\text{cut}(A, B) = \sum_{i \in A, j \in B} w_{ij}$, $\text{vol}(A) = \sum_{i \in A} d_i$.

**NP-hard** (combinatorial). Relaxation (indicator vector를 real-valued로):

$$\min_{f \ne 0, f \perp D \mathbf{1}} \frac{f^\top L f}{f^\top D f} = \lambda_2(L_{\text{sym}} \text{ or } L_{\text{rw}})$$

**해**: Normalized Laplacian의 **두 번째 smallest** 고유벡터 = optimal relaxed partition (**Fiedler vector**).

### Spectral Clustering as KPCA

Laplacian $L$ 대신 **$D^{-1/2} W D^{-1/2}$**의 eigendecomp 생각:

- 이것은 "**normalized similarity matrix**"와 유사.
- Top eigenvectors의 row로 embedding → k-means.

RBF similarity이면 이 matrix가 **centered-Gram의 Jacobi variant** → KPCA의 variant.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Weighted Adjacency Matrix

Similarity function $w : \mathcal{X} \times \mathcal{X} \to [0, \infty)$ (예: $w(x, y) = \exp(-\|x-y\|^2 / 2\sigma^2)$ 또는 $k$-NN graph).

$W_{ij} = w(x_i, x_j)$, 대각 0 또는 self-loop 허용.

### 정의 3.2 — Graph Laplacians

- **Unnormalized**: $L = D - W$.
- **Symmetric normalized**: $L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}$.
- **Random walk**: $L_{\text{rw}} = D^{-1} L = I - D^{-1} W$.

### 정의 3.3 — Spectral Clustering Algorithm (Ng, Jordan, Weiss 2002)

1. Similarity matrix $W$ 구성.
2. Normalized Laplacian $L_{\text{sym}}$ 계산.
3. Top-$k$ smallest eigenvectors $\{u_1, \ldots, u_k\}$ 구함.
4. 행렬 $U = [u_1, \ldots, u_k] \in \mathbb{R}^{n \times k}$, **row-normalize** $U_{ij} \to U_{ij} / \sqrt{\sum_l U_{il}^2}$.
5. 각 행을 point로 보고 **k-means**로 clustering.

### 정의 3.4 — Fiedler Vector

$L$의 $\lambda_2$ (두 번째 smallest) eigenvalue에 대응하는 eigenvector. Graph의 bipartition에 대한 optimal relaxation 해.

---

## 🔬 정리와 증명

### 정리 3.1 — Laplacian의 기본 성질

**명제**: $L = D - W$ (symmetric, non-negative weights)에 대해:

1. $L$은 대칭 PSD.
2. $L \mathbf{1} = 0$, $\mathbf{1}$이 $\lambda_1 = 0$ eigenvector.
3. 고유값 $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$.
4. $\dim \ker L = $ graph의 connected components 수.

**증명**:

(1) $f^\top L f = f^\top D f - f^\top W f = \sum_i d_i f_i^2 - \sum_{i,j} w_{ij} f_i f_j = \frac{1}{2} \sum_{i, j} w_{ij} (f_i - f_j)^2 \geq 0$.

(2) $L_{ij} = d_i \delta_{ij} - w_{ij}$, $(L\mathbf{1})_i = d_i - \sum_j w_{ij} = 0$.

(3) PSD + $\lambda_1 = 0$.

(4) Connected components로 graph 분해되면 각 component에서 $L|_{\text{comp}} \mathbf{1}_{\text{comp}} = 0$ → 해당 indicator function이 eigenvector with $\lambda = 0$. (Details in von Luxburg 2007.) $\square$

### 정리 3.2 — Rayleigh Quotient와 Courant-Fischer

**명제**: $\lambda_k(L)$는

$$\lambda_k = \min_{\text{dim}(V) = k} \max_{f \in V, f \ne 0} \frac{f^\top L f}{f^\top f}.$$

$k = 2$이면 $\lambda_2 = \min_{f \perp \mathbf{1}, f \ne 0} \frac{f^\top L f}{f^\top f}$.

**증명**: Courant-Fischer (표준 스펙트럴 이론). $\square$

**해석**: $\lambda_2$ 작음 = "$\mathbf{1}$과 수직인 smooth 함수 존재" = graph이 2개 cluster로 쉽게 분리 가능.

### 정리 3.3 — Normalized Cut Relaxation (Shi & Malik)

**명제**: Discrete NCut 문제

$$\min_{A} \text{NCut}(A, \bar{A}) = \min_{f \in \{a, -b\}^n} \frac{f^\top L f}{f^\top D f}$$

의 real-valued relaxation

$$\min_{f: f^\top D \mathbf{1} = 0, f^\top D f = 1} f^\top L f$$

의 해는 $L_{\text{rw}} = D^{-1} L$의 두 번째 smallest eigenvector (또는 equivalently $L_{\text{sym}}$의 $D^{1/2}$-scaled).

**증명 개요**: Lagrangian $f^\top L f - \lambda (f^\top D f - 1) - \mu f^\top D \mathbf{1}$에서 $\nabla = 0$: $L f = \lambda D f$ + constraint. Generalized eigenvalue problem $L f = \lambda D f \Leftrightarrow L_{\text{rw}} f = \lambda f$. $\square$

### 정리 3.4 — Spectral Clustering as KPCA

**명제**: Spectral clustering은 **$L_{\text{sym}}$의 top-$k$ eigenvectors로 embedding** 후 k-means. 이것은 Kernel PCA의 **특수 사례** ("kernel" $= D^{-1/2} W D^{-1/2}$).

**증명 아이디어**: $D^{-1/2} W D^{-1/2}$는 "normalized similarity = kernel-like 행렬". 이것의 top eigenvectors = KPCA projection. Row-normalization은 각 "point"를 unit vector로 정규화 → angular distance-based k-means.

**함의**: Spectral clustering = KPCA-based clustering with specific (normalized) kernel.

### 정리 3.5 — Ideal Case — Perfect Clusters

**명제**: $K$개의 완전 disconnected 같은 크기 cluster: each cluster = clique, inter-cluster weights = 0.

$L$의 smallest eigenvalues $\lambda_1 = \cdots = \lambda_K = 0$ (K-fold degenerate). 대응 eigenvectors = **indicator functions** $\mathbf{1}_{C_k}$ (cluster $k$).

**결과**: Spectral clustering이 **정확한 클러스터 복구**.

**Noise가 있는 경우**: Perturbation theory (Davis-Kahan)로 near-perfect recovery가 noise가 작으면 성립.

### 정리 3.6 — Spectral vs K-means 비교

| | K-means | Spectral |
|--|---------|----------|
| Cluster shape | Convex (구형) | Arbitrary (graph 구조) |
| Complexity | $O(n k)$ per iter | $O(n^3)$ eigendecomp |
| Scalability | Great | Poor (Nyström 근사 필요) |
| Initialization | 민감 | 없음 (eigenvector fixed) |
| Number of clusters | 미리 지정 | 미리 지정 (eigengap으로 자동 추정도 가능) |

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. 비구형 데이터 — moons
# ─────────────────────────────────────────────
X, labels_true = make_moons(n_samples=200, noise=0.08, random_state=0)
n = len(X)

# ─────────────────────────────────────────────
# 2. Spectral clustering bottom-up
# ─────────────────────────────────────────────
def rbf_similarity(X, s=0.3):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T
    return np.exp(-d2 / (2 * s**2))

W = rbf_similarity(X, s=0.3)
np.fill_diagonal(W, 0)  # no self-loop

D = np.diag(W.sum(axis=1))
L = D - W

# Normalized Laplacian
d_sqrt_inv = 1 / np.sqrt(np.diag(D))
L_sym = np.eye(n) - d_sqrt_inv[:, None] * W * d_sqrt_inv[None, :]

# Eigendecomposition: smallest k eigenvectors
eigvals, eigvecs = np.linalg.eigh(L_sym)
k = 2
U = eigvecs[:, :k]  # top-k smallest

# Row normalization (Ng, Jordan, Weiss)
U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

# K-means on embedding
km = KMeans(n_clusters=k, n_init=10, random_state=0)
labels_pred = km.fit_predict(U_norm)

# ─────────────────────────────────────────────
# 3. 비교: K-means 직접, Spectral
# ─────────────────────────────────────────────
km_direct = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_true, cmap='coolwarm', s=20)
axes[0].set_title('Ground Truth')

axes[1].scatter(X[:, 0], X[:, 1], c=km_direct, cmap='coolwarm', s=20)
axes[1].set_title('K-means (fails)')

axes[2].scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='coolwarm', s=20)
axes[2].set_title('Spectral (bottom-up)')

# sklearn Spectral
sc = SpectralClustering(n_clusters=k, affinity='rbf', gamma=5, random_state=0)
labels_sklearn = sc.fit_predict(X)
axes[3].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='coolwarm', s=20)
axes[3].set_title('Spectral (sklearn)')

plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 4. Eigenvalue gap (automatic k selection)
# ─────────────────────────────────────────────
plt.figure(figsize=(9, 4))
plt.plot(eigvals[:15], 'o-')
plt.xlabel('Eigenvalue index'); plt.ylabel('λ')
plt.title('Laplacian Eigenvalues — eigengap으로 cluster 수 추정')
plt.grid(True, alpha=0.3); plt.show()

# 큰 gap 위치 = 적정 cluster 수
gaps = np.diff(eigvals)
print(f'첫 5 eigenvalues: {eigvals[:5]}')
print(f'첫 4 gaps: {gaps[:4]}')

# ─────────────────────────────────────────────
# 5. Image-like example (두 개 연결 블록)
# ─────────────────────────────────────────────
# 3-cluster 데이터
def three_clusters():
    c1 = rng.standard_normal((40, 2)) * 0.3 + [2, 2]
    c2 = rng.standard_normal((40, 2)) * 0.3 + [-2, 2]
    c3 = rng.standard_normal((40, 2)) * 0.3 + [0, -2]
    return np.vstack([c1, c2, c3]), np.concatenate([np.zeros(40), np.ones(40), 2*np.ones(40)])

X3, labels3_true = three_clusters()
W3 = rbf_similarity(X3, s=0.5)
np.fill_diagonal(W3, 0)
D3 = np.diag(W3.sum(axis=1))
d3_sqrt_inv = 1 / np.sqrt(np.diag(D3))
L3_sym = np.eye(120) - d3_sqrt_inv[:, None] * W3 * d3_sqrt_inv[None, :]
eig3, V3 = np.linalg.eigh(L3_sym)
U3 = V3[:, :3]
U3_norm = U3 / np.linalg.norm(U3, axis=1, keepdims=True)
labels3_pred = KMeans(n_clusters=3, random_state=0).fit_predict(U3_norm)

print(f'\nTree-cluster eigenvalues: {eig3[:5]}')
print(f'→ 처음 3개가 0에 가까움 (3 cluster indicator)')
```

**출력 예시**:
```
첫 5 eigenvalues: [1e-15  0.0132  0.0421  0.1234  0.2341]
첫 4 gaps: [0.0132  0.0289  0.0813  0.1107]

Tree-cluster eigenvalues: [1e-15  2e-15  3e-15  0.1234  0.4521]
→ 처음 3개가 0에 가까움 (3 cluster indicator)
```

→ Moons에서 K-means 실패, spectral clustering 성공. Eigenvalue gap으로 cluster 수 추정 가능.

---

## 🔗 실전 활용

- **Image segmentation**: Normalized Cut으로 pixel-level clustering. Shi & Malik의 원래 응용.
- **Community detection**: Graph data (social network, citation)에서 community 찾기.
- **Manifold learning**: Laplacian Eigenmaps (Belkin & Niyogi 2003).
- **Diffusion Maps**: Laplacian을 diffusion operator로 해석 → non-linear dimensionality reduction.
- **sklearn**: `SpectralClustering(affinity='rbf', gamma=..., n_clusters=k)`.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| $O(n^3)$ eigendecomp | $n > 10^4$이면 Nyström, Lanczos approximation |
| **Similarity kernel 선택** | RBF $\sigma$ 나쁘면 clustering 결과 극단적 (완전 연결 or 너무 많은 cluster) |
| $k$ 지정 | Eigengap heuristic으로 추정 가능하나 완전 자동 아님 |
| Connected graph 필요 | Disconnected이면 eigenvalue 0이 여러 번 → 주의 |
| Scalability | Large graph에서 Nyström or Stochastic SVD |

---

## 📌 핵심 정리

$$\boxed{L = D - W, \quad L \mathbf{1} = 0, \quad f^\top L f = \frac{1}{2} \sum_{i, j} w_{ij} (f_i - f_j)^2}$$

$$\boxed{\text{Spectral clustering: top-}k\text{ smallest eigenvectors of } L_{\text{sym}} \to \text{embed} \to \text{k-means}}$$

| Step | 내용 |
|------|------|
| 1. Graph 구성 | $W$ = similarity matrix (RBF, $k$-NN 등) |
| 2. Laplacian | $L = D - W$ 또는 $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$ |
| 3. Eigendecomp | Top-$k$ smallest eigenvectors $U \in \mathbb{R}^{n \times k}$ |
| 4. Row-normalize | $U_{i, :} \to U_{i, :} / \|U_{i, :}\|$ (Ng, Jordan, Weiss) |
| 5. K-means | Each row as a point, cluster |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Graph Laplacian의 작은 고유값이 cluster 구조를 감지하는 이유를 Rayleigh quotient로 설명하라.

<details>
<summary>힌트 및 해설</summary>

$\lambda_k = \min_{f \perp \{u_1, \ldots, u_{k-1}\}} f^\top L f / f^\top f = \min \frac{\sum w_{ij} (f_i - f_j)^2 / 2}{\|f\|^2}$.

**"작은 $\lambda_k$"** = "전체적으로 smooth한 $f$ 존재 (point들 사이 value 변화 작음)".

**Cluster 관점**: 
- 같은 cluster 내: $w_{ij}$ 크고 $(f_i - f_j)^2$ 작음 → 기여 적음.
- 다른 cluster 사이: $w_{ij}$ 작음 → $(f_i - f_j)^2$ 가 커도 기여 작음.

→ $f$가 cluster 내 일정, cluster 사이 차이 → $f^\top L f$ 작음.

**Extreme case**: Perfect clusters ($K$ disconnected)이면 $\lambda_1 = \cdots = \lambda_K = 0$ with indicator eigenvectors.

</details>

**문제 2** (심화): $L_{\text{sym}}$와 $L_{\text{rw}}$의 차이는 언제 중요한가?

<details>
<summary>힌트 및 해설</summary>

**$L_{\text{sym}} = D^{-1/2} L D^{-1/2}$**: 대칭, 수치적으로 안정적. Ng-Jordan-Weiss 알고리즘 기본.

**$L_{\text{rw}} = D^{-1} L$**: 비대칭이지만 Markov chain interpretation ($P = D^{-1} W$는 random walk transition). Shi-Malik 알고리즘.

**언제 다른가**: Degree가 크게 불균형할 때 ($d_{\max} \gg d_{\min}$):
- $L$: High-degree nodes 지배.
- $L_{\text{rw}}$: Degree로 정규화 → community 구조에 더 민감.
- $L_{\text{sym}}$: 비슷하지만 대칭 유지.

**동등성**: $L_{\text{sym}} f = \lambda f \iff L_{\text{rw}} (D^{-1/2} f) = \lambda (D^{-1/2} f)$ — 고유벡터가 $D^{1/2}$로 관련.

**실무**: $L_{\text{sym}}$이 표준 (numerical stability).

</details>

**문제 3** (ML 연결): Spectral clustering과 **Graph Neural Networks (GNN)**의 관계는?

<details>
<summary>힌트 및 해설</summary>

**GNN의 기초**: Message passing $h_i^{(l+1)} = \text{AGG}(\{h_j^{(l)} : j \in \mathcal{N}(i)\})$. 

**Graph convolution (GCN, Kipf & Welling 2017)**:
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

$\tilde{A} = A + I$, $\tilde{D}$ = degree. 이것은 **$I - L_{\text{sym}}$ 유사 operator**의 반복 적용.

**연결**:
- GCN = Graph Laplacian의 parametrized linear transformation + nonlinearity.
- Spectral clustering = Graph Laplacian의 eigenvectors.

**함의**:
- GCN은 **Laplacian의 low-frequency eigenvectors를 근사 학습** — spectral clustering의 "supervised 버전".
- Signal processing on graphs: GCN = filter on graph Fourier basis (eigenvectors of $L$).

**Modern**: GNN은 end-to-end learnable, deeper, supervised. Spectral clustering은 unsupervised + 1-shot eigendecomp. 두 techniques가 같은 수학적 기반 공유.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. Kernel PCA의 수학](./02-kernel-pca.md) | [04. Kernel k-Means ▶](./04-kernel-kmeans.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
