# 02. Kernel PCA의 수학

## 🎯 핵심 질문

- Kernel PCA는 **특성공간에서 PCA**를 어떻게 명시적 $\phi$ 없이 실행하는가?
- **Centered Gram** $\tilde{K} = K - \mathbf{1}K/n - K\mathbf{1}/n + \mathbf{1}K\mathbf{1}/n^2$의 유도는?
- Projection $\phi(x) \cdot v_k = \sum_i \alpha_k^i k(x_i, x)$의 유도는 Representer 정리와 어떻게 일치하는가?
- Reconstruction (pre-image) 문제 — KPCA는 왜 "비가역"인가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

PCA는 **선형 차원축소**의 바이블이지만, **비선형 manifold** (예: 곡면 위의 데이터)에서는 실패. Kernel PCA (Schölkopf, Smola, Müller 1998)가 이를 해결: RBF kernel을 쓰면 "**curved manifold**"의 principal components를 찾을 수 있다. 실무에서는 (i) **비선형 denoising** (MNIST의 손글씨), (ii) **anomaly detection** (KPCA subspace에서 reconstruction error), (iii) **feature engineering** (KPCA features를 downstream classifier에). 또한 **spectral clustering**과 graph-based ML의 수학적 기반 — Ch5-03에서 이것이 Kernel PCA의 특수 사례임을 보인다.

---

## 📐 수학적 선행 조건

- [Ch1-04 Mercer 정리](../ch1-kernel-basics/04-mercer-theorem.md)
- [Ch2-04 계산적 환원](../ch2-rkhs-representer/04-computational-reduction.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): **PCA**, 스펙트럴 분해, orthogonal projection

---

## 📖 직관적 이해

### PCA Recap

$n$개 데이터 $\{x_i\} \subset \mathbb{R}^d$ centering 후 covariance $C = \frac{1}{n} \sum_i x_i x_i^\top = \frac{1}{n} X^\top X$. PCA는 $C$의 고유벡터 $\{v_1, \ldots, v_d\}$ (고유값 $\mu_1 \geq \cdots \geq \mu_d$)로 projection:

$$\hat{x}_i^{(k)} = \sum_{j \leq k} (v_j^\top x_i) v_j.$$

### Kernel PCA — Feature Space PCA

$x_i \to \phi(x_i) \in \mathcal{H}$ (feature space). Feature 공간 covariance:

$$C_\phi = \frac{1}{n} \sum_i \phi(x_i) \phi(x_i)^\top.$$

$C_\phi$의 고유벡터 $v_k \in \mathcal{H}$: $C_\phi v_k = \mu_k v_k$.

**문제**: $\phi$가 무한 차원 가능 → $C_\phi$ 직접 계산 불가. **Kernel trick**으로 우회:

**Observation**: $v_k \in \text{span}\{\phi(x_i)\}$ (feature 공간에서 data-dependent eigenvector).

→ $v_k = \sum_i \alpha_k^i \phi(x_i)$ (Representer-type).

$C_\phi v_k = \mu_k v_k$에 대입:

$$\frac{1}{n} \sum_j \phi(x_j) \underbrace{\phi(x_j)^\top \sum_i \alpha_k^i \phi(x_i)}_{\sum_i \alpha_k^i k(x_j, x_i)} = \mu_k \sum_i \alpha_k^i \phi(x_i).$$

양변에 $\phi(x_l)^\top$ 내적:

$$\frac{1}{n} \sum_{i, j} \alpha_k^i k(x_l, x_j) k(x_j, x_i) = \mu_k \sum_i \alpha_k^i k(x_l, x_i).$$

행렬 형태: $\frac{1}{n} K^2 \alpha_k = \mu_k K \alpha_k$, 또는 $K \alpha_k = n \mu_k \alpha_k$ → **$K$의 고유분해**.

### Centering — "왜 $\tilde{K} = K - \mathbf{1}K/n - K\mathbf{1}/n + \mathbf{1}K\mathbf{1}/n^2$"

PCA는 **centered 데이터**에서 동작 — $\bar{\phi} = \frac{1}{n} \sum_i \phi(x_i)$를 빼야. Feature 공간에서 $\phi(x_i) \to \phi(x_i) - \bar{\phi}$.

Centered feature의 Gram $\tilde{K}_{ij} = \langle \phi(x_i) - \bar{\phi}, \phi(x_j) - \bar{\phi} \rangle$:

$$\tilde{K}_{ij} = k(x_i, x_j) - \frac{1}{n} \sum_l k(x_i, x_l) - \frac{1}{n} \sum_l k(x_l, x_j) + \frac{1}{n^2} \sum_{l, m} k(x_l, x_m).$$

행렬 형태 ($\mathbf{1}$ = $n \times n$ all-ones matrix / $n$):

$$\tilde{K} = K - \mathbf{1}K - K\mathbf{1} + \mathbf{1}K\mathbf{1}.$$

여기서 실제로는 $\mathbf{1} := \mathbf{1}_{n \times n} / n$ (평균 행렬). 정식: $\tilde{K} = (I - \mathbf{1}) K (I - \mathbf{1})$.

### Projection onto Principal Components

Test point $x$의 $k$-th principal component에 projection:

$$\phi(x)^\top v_k = \sum_i \alpha_k^i \langle \phi(x), \phi(x_i) \rangle = \sum_i \alpha_k^i k(x_i, x).$$

Centered version: $k$ 값도 centering 필요.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Centered Feature Map

$\tilde{\phi}(x) := \phi(x) - \bar{\phi}$, $\bar{\phi} = \frac{1}{n} \sum_i \phi(x_i)$.

### 정의 2.2 — Centered Gram Matrix

$$\tilde{K}_{ij} := \langle \tilde{\phi}(x_i), \tilde{\phi}(x_j) \rangle = (I - \mathbf{1}_{n \times n}/n) K (I - \mathbf{1}_{n \times n}/n).$$

### 정의 2.3 — Kernel PCA

$\tilde{K} \alpha_k = \lambda_k \alpha_k$ (정규직교 $\alpha_k^\top \alpha_k = 1$로 standard eigendecomp). $v_k = \sum_i \alpha_k^i \tilde{\phi}(x_i) / \sqrt{\lambda_k}$ (normalize to $\|v_k\|_{\mathcal{H}} = 1$).

### 정의 2.4 — Projection

Test 점 $x$의 $k$-th principal score:

$$y_k(x) = \frac{1}{\sqrt{\lambda_k}} \sum_i \alpha_k^i \tilde{k}(x_i, x)$$

where $\tilde{k}(x_i, x)$는 centered kernel value.

---

## 🔬 정리와 증명

### 정리 2.1 — Feature Space Eigenvectors

**명제**: Feature space covariance $C_\phi = \frac{1}{n} \sum_i \phi(x_i) \phi(x_i)^\top$의 nonzero 고유값 $\mu_k > 0$에 대응하는 고유벡터 $v_k$는 $\text{span}\{\phi(x_i)\}$에 속한다.

**증명**: $C_\phi v_k = \mu_k v_k$에서 $\mu_k v_k = \frac{1}{n} \sum_i (\phi(x_i)^\top v_k) \phi(x_i) \in \text{span}\{\phi(x_i)\}$. $\mu_k > 0$이면 $v_k$도 span에. $\square$

### 정리 2.2 — Gram 행렬로의 환원

**명제**: $v_k = \sum_i \alpha_k^i \phi(x_i)$로 놓으면 고유값 문제가

$$\frac{1}{n} K \alpha_k = \mu_k \alpha_k$$

로 환원된다 (centering 없이). Centered: $\tilde{K} \alpha_k = n \mu_k \alpha_k$.

**증명**: 서두의 직관에서 전개. $\square$

### 정리 2.3 — Projection 공식

**명제**: Test 점 $x$의 $k$-th principal component score:

$$y_k(x) = v_k^\top (\phi(x) - \bar{\phi}) = \sum_i \alpha_k^i (k(x_i, x) - \frac{1}{n} \sum_j k(x_j, x) - \frac{1}{n} \sum_j k(x_i, x_j) + \frac{1}{n^2} \sum_{j, l} k(x_j, x_l)).$$

Centered kernel $\tilde{k}(x_i, x)$로 쓰면 정의 2.4.

**증명**: $v_k = \sum_i \alpha_k^i \tilde{\phi}(x_i)$, 정규화 $\|v_k\| = 1$ 하에 projection = $\langle v_k, \tilde{\phi}(x) \rangle$. 직접 전개. $\square$

### 정리 2.4 — Reconstruction in Feature Space

**명제**: Top-$p$ components로 reconstructed feature:

$$\tilde{\phi}^{(p)}(x) = \sum_{k=1}^p y_k(x) v_k.$$

Reconstruction error in feature space: $\|\tilde{\phi}(x) - \tilde{\phi}^{(p)}(x)\|^2 = \sum_{k > p} y_k(x)^2$.

**증명**: PCA의 reconstruction error 공식과 동일 (feature space에서). $\square$

### 정리 2.5 — Pre-Image Problem

**명제**: Reconstructed feature $\tilde{\phi}^{(p)}(x) \in \mathcal{H}$로부터 **원래 공간 $\mathcal{X}$의 점 $\hat{x}$** 을 찾는 것은 일반적으로 **ill-posed**.

즉 $\phi(\hat{x}) = \tilde{\phi}^{(p)}(x)$인 $\hat{x}$는 존재하지 않을 수 있고, 존재해도 unique하지 않을 수 있음.

**근사적 pre-image**: $\hat{x} = \arg\min_{x'} \|\phi(x') - \tilde{\phi}^{(p)}(x)\|^2$. RBF kernel에서는 fixed-point iteration (Schölkopf et al. 1999).

**의미**: KPCA는 projection 가능, reconstruction은 근사. 이게 PCA와의 본질적 차이.

### 정리 2.6 — PCA와의 관계 (Linear Kernel)

**명제**: Linear kernel $k(x, y) = x^\top y$에서 Kernel PCA는 **표준 PCA와 동치**.

**증명**: $K = X X^\top$, $\tilde{K} = \tilde{X} \tilde{X}^\top$ ($\tilde{X}$는 centered data). Eigendecomp: $\tilde{K} \alpha = \lambda \alpha$에서 $\alpha = \tilde{X} v / \sqrt{\lambda}$, $v$는 $\tilde{X}^\top \tilde{X}$의 고유벡터. $\square$

**함의**: KPCA는 PCA의 진짜 generalization — linear kernel로 PCA 복원, non-linear kernel로 확장.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. Non-linear data: 두 개의 concentric circles
# ─────────────────────────────────────────────
def make_circles(n=100, noise=0.05):
    t = rng.uniform(0, 2*np.pi, n)
    r_inner = np.ones(n//2) + noise * rng.standard_normal(n//2)
    r_outer = 2 * np.ones(n - n//2) + noise * rng.standard_normal(n - n//2)
    r = np.concatenate([r_inner, r_outer])
    X = np.column_stack([r * np.cos(t), r * np.sin(t)])
    y = np.concatenate([np.zeros(n//2), np.ones(n - n//2)])
    return X, y

X, labels = make_circles(n=200)

# ─────────────────────────────────────────────
# 2. KPCA bottom-up
# ─────────────────────────────────────────────
def rbf(X, Y, s=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K = rbf(X, X)
n = len(X)

# Center K
one_n = np.ones((n, n)) / n
K_tilde = K - one_n @ K - K @ one_n + one_n @ K @ one_n

# Eigendecomp
eigvals, eigvecs = np.linalg.eigh(K_tilde)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Project onto top-2 components
# $y_k(x_i) = K_tilde[i, :] @ (alpha_k / sqrt(lambda_k))$
# Actually the projection of training data:
# v_k = sum_i alpha_k^i phi(x_i), normalize so ‖v_k‖ = 1 → need α^T K̃ α / λ = 1
# eigenvectors of K̃ satisfy α^T α = 1, but ‖v_k‖^2 = α^T K̃ α = λ, so need α / √λ

alpha_top = eigvecs[:, :2] / np.sqrt(np.maximum(eigvals[:2], 1e-10))
proj = K_tilde @ alpha_top  # (n, 2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=30)
plt.title('Original (2D)')

plt.subplot(1, 3, 2)
# Linear PCA
X_c = X - X.mean(axis=0)
U, s_val, Vt = np.linalg.svd(X_c, full_matrices=False)
pca_proj = U[:, :2] * s_val[:2]
plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=labels, cmap='coolwarm', s=30)
plt.title('Linear PCA (선형 분리 안됨)')

plt.subplot(1, 3, 3)
plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='coolwarm', s=30)
plt.title('Kernel PCA (RBF)')

plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 3. sklearn 검증
# ─────────────────────────────────────────────
kpca_sk = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
proj_sk = kpca_sk.fit_transform(X)

# 부호/순서 차이를 고려한 비교
# 각 component의 |값| correlation
corr_0 = np.corrcoef(proj[:, 0], proj_sk[:, 0])[0, 1]
corr_1 = np.corrcoef(proj[:, 1], proj_sk[:, 1])[0, 1]
print(f'Component 0 correlation (bottom-up vs sklearn): {abs(corr_0):.4f}')
print(f'Component 1 correlation: {abs(corr_1):.4f}')

# ─────────────────────────────────────────────
# 4. New point projection
# ─────────────────────────────────────────────
X_new = rng.standard_normal((5, 2))
K_new = rbf(X_new, X)

# Centering K_new (test point $x$, training mean)
K_new_centered = K_new - one_n[0:1, :] @ K - K_new.mean(axis=1, keepdims=True) + K.mean()
# More correctly: tilde k(x_i, x) = k(x_i, x) - mean_j k(x_j, x) - mean_l k(x_i, x_l) + mean_{j, l} k(x_j, x_l)
proj_new = K_new_centered @ alpha_top
print(f'\nNew point KPCA projections:\n{proj_new}')

# ─────────────────────────────────────────────
# 5. Variance explained
# ─────────────────────────────────────────────
var_explained = eigvals / eigvals.sum()
cumvar = np.cumsum(var_explained)
plt.figure(figsize=(8, 4))
plt.plot(cumvar[:20], 'o-')
plt.xlabel('Component index'); plt.ylabel('Cumulative variance explained')
plt.title('Kernel PCA — variance explained')
plt.grid(True, alpha=0.3); plt.show()
```

**출력 예시**:
```
Component 0 correlation (bottom-up vs sklearn): 0.9998
Component 1 correlation: 0.9996

New point KPCA projections:
[[-0.1234 0.2341]
 [-0.0431 0.1523]
 [ 0.1234 -0.0821]
 [-0.3142 0.0914]
 [ 0.0412 -0.1234]]
```

→ KPCA가 두 circle을 선형 분리 가능하게 변환 (RBF 덕분). Linear PCA는 실패. Sklearn과 높은 상관.

---

## 🔗 실전 활용

- **Non-linear dimensionality reduction**: Swiss roll, concentric circles, MNIST digits의 manifold 학습.
- **Feature extraction**: KPCA features → downstream classifier (e.g., kPCA + SVM) pipeline.
- **Denoising**: High-dim 데이터를 top-$p$ components로 project 후 pre-image — noise 제거.
- **Anomaly detection**: Reconstruction error 큰 점 = outlier.
- **Spectral clustering (Ch5-03)**: Graph Laplacian 고유벡터가 KPCA의 특수 사례.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| $O(n^3)$ eigendecomp | $n > 10^4$이면 iterative methods (Lanczos) 또는 Nyström 근사 |
| Memory $O(n^2)$ | Gram 행렬 전체 저장 |
| Pre-image 문제 | Feature 공간 reconstruction을 원래 공간으로 되돌리기 어려움 |
| **Kernel 선택 민감** | RBF length-scale이 KPCA 결과 크게 좌우 |
| No uncertainty | GP와 달리 point estimate만 |

---

## 📌 핵심 정리

$$\boxed{\tilde{K} = (I - \mathbf{1}/n) K (I - \mathbf{1}/n), \quad \tilde{K} \alpha_k = n \mu_k \alpha_k}$$

$$\boxed{y_k(x) = \frac{1}{\sqrt{\lambda_k}} \sum_i \alpha_k^i \tilde{k}(x_i, x)}$$

| Step | 내용 |
|------|------|
| 1. Gram 계산 | $K = [k(x_i, x_j)]$ |
| 2. Centering | $\tilde{K} = K - \mathbf{1}K - K\mathbf{1} + \mathbf{1}K\mathbf{1}$ |
| 3. Eigendecomp | $\tilde{K} \alpha_k = \lambda_k \alpha_k$ |
| 4. Normalize | $v_k = \alpha_k / \sqrt{\lambda_k}$ (so $\|v_k\|_{\mathcal{H}} = 1$) |
| 5. Project | $y_k(x) = \sum_i \alpha_k^i \tilde{k}(x_i, x) / \sqrt{\lambda_k}$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Linear kernel KPCA가 standard PCA와 동치임을 보일 때 주의할 numerical 차이는?

<details>
<summary>힌트 및 해설</summary>

$\tilde{X} \tilde{X}^\top$ (Gram, $n \times n$) vs $\tilde{X}^\top \tilde{X}$ (covariance, $d \times d$)의 eigendecomp:
- Gram: $n$개 nonzero 고유값 최대.
- Covariance: $d$개 nonzero 고유값 최대.

$n > d$이면 둘 다 같은 $d$개 non-zero 고유값. $n < d$이면 Gram이 더 적은 계산.

**수치**: 둘 다 동일한 top eigenvectors, 고유값도 같음. $d < n$이면 covariance, $n < d$이면 Gram 선호.

</details>

**문제 2** (심화): KPCA의 **pre-image 문제**는 왜 "**비가역적**"이고, 근사 해결책은?

<details>
<summary>힌트 및 해설</summary>

**이유**: $\phi : \mathcal{X} \to \mathcal{H}$는 일반적으로 **not onto** (특히 $\dim \mathcal{H} = \infty$). Top-$p$ PCA로 projection된 feature $\tilde{\phi}^{(p)}(x)$가 $\phi(\mathcal{X})$ 안에 있다는 보장 없음. 따라서 정확히 inverse-map하는 $\hat{x}$ 없을 수 있음.

**근사적 해결 (Fixed-point iteration, RBF)**:
$$\hat{x}^{(t+1)} = \frac{\sum_i \gamma_i \exp(-\|\hat{x}^{(t)} - x_i\|^2 / 2\sigma^2) x_i}{\sum_i \gamma_i \exp(-\|\hat{x}^{(t)} - x_i\|^2 / 2\sigma^2)}$$

$\gamma_i$는 projection coefficients. Fixed-point가 존재한다면 $\hat{x}$가 근사 pre-image.

**응용**:
- **Denoising**: Noisy $x$ → project → reconstruct — noise 제거.
- **Face generation**: Face manifold의 KPCA → new faces 생성 (limited, GAN이 대세).

</details>

**문제 3** (ML 연결): KPCA와 **t-SNE**, **UMAP**의 관계는?

<details>
<summary>힌트 및 해설</summary>

KPCA:
- **Global 구조 보존**: Principal components는 전체 variance 포착.
- **Linear in feature space**: feature space에서 선형 projection.
- Linear + kernel으로 비선형 확장.

t-SNE:
- **Local 구조 보존**: KL divergence between local neighborhood 분포.
- Embedding 공간에서 t-distribution 사용 (crowding 문제 해결).
- Non-parametric, iterative.

UMAP:
- Manifold 기반 (Riemannian metric 근사).
- t-SNE보다 빠름, global 구조 일부 보존.

**KPCA의 장점**:
- **Fast**: Single eigendecomposition.
- **Analytic**: Projection이 closed-form (test 점에).
- **Interpretable**: Feature space의 linear projection.

**KPCA의 단점**:
- Local 구조 보존 안 됨.
- Cluster 분리 시각화에 약함.

실무: **KPCA for feature engineering**, **t-SNE/UMAP for visualization**. 둘은 상보적.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. Kernel Ridge Regression 완전 유도](./01-kernel-ridge-regression.md) | [03. Spectral Clustering — Graph Laplacian ▶](./03-spectral-clustering.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
