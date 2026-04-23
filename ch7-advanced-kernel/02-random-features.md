# 02. Random Features (Rahimi & Recht 2007)

## 🎯 핵심 질문

- **Bochner 정리**: shift-invariant kernel $k(x - y)$는 **양의 측도의 Fourier 변환**일 때 정확히 PD — 이 정리의 의미는?
- Random Fourier Features $\phi(x) = \sqrt{2/D}(\cos(\omega_i^\top x + b_i))_i$는 어떻게 $k(x, y) \approx \phi(x)^\top \phi(y)$를 달성하는가?
- Sample complexity: $D$개 random features로 $\epsilon$-근사를 보장하려면 몇 개 필요한가?
- 이것이 어떻게 "kernel method를 $O(n^3) \to O(nD^2)$"로 scale하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Kernel method의 가장 큰 병목인 **$O(n^2)$ 메모리**와 **$O(n^3)$ 시간**을 해결하는 **가장 영향력 있는 아이디어 중 하나**. Random Features는 "**Mercer decomposition을 Monte Carlo로 근사**"하는 것 — 무한 차원 feature space를 **유한 차원 $D$**로 approximate, $O(nD^2)$ with $D \ll n$의 매우 실용적 scaling. 이 덕분에 kernel method가 **$n = 10^6$+ 데이터**까지 확장. 현대 적용: Scalable GP, large-scale kernel SVM, Bayesian deep learning의 kernel-feature bridge. 또한 **Transformer의 Performer** 같은 linear attention이 이 기법 응용.

---

## 📐 수학적 선행 조건

- [Ch1-02 Kernel zoo](../ch1-kernel-basics/02-kernel-zoo.md): RBF, Laplace (shift-invariant)
- [Ch1-04 Mercer](../ch1-kernel-basics/04-mercer-theorem.md): Feature decomposition
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Fourier analysis, characteristic function, Hoeffding concentration
- 복소해석: Fourier transform, Bochner 정리

---

## 📖 직관적 이해

### Bochner 정리

**Shift-invariant kernel** $k(x, y) = \kappa(x - y)$ (공분산이 거리만 의존):

$$\kappa \text{ is PD} \iff \kappa(z) = \int_{\mathbb{R}^d} e^{i \omega^\top z} \rho(\omega) d\omega$$

여기서 $\rho$는 **양의 유한 측도** (spectral density).

**중요 예시**:
- **RBF** $\kappa(z) = e^{-\|z\|^2 / 2\sigma^2}$: $\rho(\omega) = (\sigma^2 / 2\pi)^{d/2} e^{-\sigma^2 \|\omega\|^2 / 2}$ (Gaussian).
- **Laplace** $\kappa(z) = e^{-\|z\|/\sigma}$: $\rho(\omega) \propto 1 / (1 + \sigma^2 \|\omega\|^2)^{(d+1)/2}$ (Cauchy-like).

### Random Features 핵심 아이디어

$\kappa(x - y) = \int e^{i \omega^\top (x - y)} \rho(\omega) d\omega = \mathbb{E}_{\omega \sim \rho}[e^{i \omega^\top x} e^{-i \omega^\top y}]$.

**Monte Carlo**: $\{\omega_j\}_{j=1}^D \sim \rho$ i.i.d. sample, $z_j(x) := e^{i \omega_j^\top x}$.

$$\kappa(x - y) \approx \frac{1}{D} \sum_j z_j(x) \overline{z_j(y)} = \frac{1}{D} \sum_j \cos(\omega_j^\top (x - y)) + i \sin(\cdot).$$

Real kernel이므로 real part만 → **cosine-based formulation**:

$$\kappa(x - y) \approx \frac{1}{D} \sum_j \cos(\omega_j^\top (x - y)).$$

$\cos(A - B) = \cos A \cos B + \sin A \sin B$, 또는 간단히 **with random phase $b_j \sim U[0, 2\pi]$**:

$$\phi(x) := \sqrt{2/D} \left(\cos(\omega_j^\top x + b_j)\right)_{j=1}^D \in \mathbb{R}^D.$$

$\mathbb{E}[\phi(x)^\top \phi(y)] = \kappa(x - y)$. ✓

### Approximation Quality

Hoeffding (bounded features):

$$P\left(|\phi(x)^\top \phi(y) - \kappa(x - y)| > \epsilon\right) \leq 2 e^{-D \epsilon^2 / 2}.$$

$D = O(1/\epsilon^2)$ features sufficient for $\epsilon$-accuracy.

**Uniform bound** (Rahimi & Recht): 컴팩트 도메인에서 $\sup_{x, y \in \mathcal{X}} |\phi(x)^\top \phi(y) - k(x, y)| \leq \epsilon$ with $D = O((d/\epsilon^2) \log(\text{diam}(\mathcal{X})/\epsilon))$.

### Kernel Method Scaling

**Kernel Ridge Regression**: Original $O(n^3)$.

**With Random Features**:
1. Compute $\Phi = [\phi(x_1); \ldots; \phi(x_n)] \in \mathbb{R}^{n \times D}$. Cost: $O(nDd)$ ($d$ = input dim).
2. Solve $w = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top y$ in $\mathbb{R}^D$. Cost: $O(nD^2 + D^3)$.
3. Predict: $f(x_*) = \phi(x_*)^\top w$. $O(D)$.

Total $O(nD^2 + D^3)$. $D = 1000$, $n = 10^6$에서 $10^{12}$ ops — 여전히 무겁지만 $n^3 = 10^{18}$보다 훨씬 나음.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Spectral Density

Shift-invariant PD kernel $\kappa(z)$의 spectral density:

$$\rho(\omega) := \frac{1}{(2\pi)^d} \int \kappa(z) e^{-i \omega^\top z} dz.$$

Bochner $\iff$ $\rho \geq 0$, $\int \rho = \kappa(0) < \infty$.

### 정의 2.2 — Random Features (RFF)

$\omega_j \overset{\text{i.i.d.}}{\sim} \tilde{\rho} := \rho / \kappa(0)$, $b_j \overset{\text{i.i.d.}}{\sim} U[0, 2\pi]$ for $j = 1, \ldots, D$.

$$\phi_{\text{RFF}}(x) := \sqrt{2 \kappa(0) / D} \left(\cos(\omega_j^\top x + b_j)\right)_{j=1}^D \in \mathbb{R}^D.$$

### 정의 2.3 — Random Features Kernel Approximation

$$\hat{k}(x, y) := \phi_{\text{RFF}}(x)^\top \phi_{\text{RFF}}(y) = \frac{2\kappa(0)}{D} \sum_{j=1}^D \cos(\omega_j^\top x + b_j) \cos(\omega_j^\top y + b_j).$$

---

## 🔬 정리와 증명

### 정리 2.1 — Bochner 정리

**명제**: 연속 대칭 $\kappa : \mathbb{R}^d \to \mathbb{R}$이 PD ↔ 양의 유한 Borel 측도 $\mu$의 Fourier 변환:

$$\kappa(z) = \int e^{i \omega^\top z} d\mu(\omega).$$

**증명 개요**:

$(\Leftarrow)$: $\mu \geq 0$ → Gaussian integral form PD. Detail: $\sum \alpha_i \alpha_j \kappa(x_i - x_j) = \int \|\sum \alpha_i e^{i\omega^\top x_i}\|^2 d\mu(\omega) \geq 0$.

$(\Rightarrow)$: 연속 PD에서 Fourier 변환의 유효성 + positivity (Bochner 원래 증명). $\square$

### 정리 2.2 — RFF의 Unbiasedness

**명제**: $\mathbb{E}[\phi_{\text{RFF}}(x)^\top \phi_{\text{RFF}}(y)] = \kappa(x - y) = k(x, y)$.

**증명**:

$\mathbb{E}\left[\frac{2\kappa(0)}{D} \sum_j \cos(\omega_j^\top x + b_j) \cos(\omega_j^\top y + b_j)\right] = 2\kappa(0) \mathbb{E}[\cos(\omega^\top x + b) \cos(\omega^\top y + b)]$.

Product-to-sum: $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$.

$$= \kappa(0) \mathbb{E}[\cos(\omega^\top (x - y))] + \kappa(0) \mathbb{E}[\cos(\omega^\top (x + y) + 2b)].$$

$b \sim U[0, 2\pi]$이면 $\mathbb{E}[\cos(\cdot + 2b)] = 0$ (uniform phase averaging).

$\mathbb{E}_\omega[\cos(\omega^\top (x-y))] = \text{Re}(\mathbb{E}[e^{i\omega^\top(x-y)}]) = \text{Re}(\kappa(x-y) / \kappa(0)) = \kappa(x-y)/\kappa(0)$.

최종: $\kappa(0) \cdot \kappa(x-y)/\kappa(0) = \kappa(x-y)$. $\square$

### 정리 2.3 — Pointwise Concentration (Hoeffding)

**명제**: 고정 $(x, y)$에 대해

$$P(|\hat{k}(x, y) - k(x, y)| > \epsilon) \leq 2 \exp(-D \epsilon^2 / (8 \kappa(0)^2)).$$

**증명**: $\phi_{\text{RFF}}$의 각 성분 $\leq \sqrt{2\kappa(0)/D}$ bounded. $\hat{k}$는 $D$ independent bounded random variables의 평균. Hoeffding 바로 적용. $\square$

### 정리 2.4 — Uniform Approximation (Rahimi-Recht Theorem 1)

**명제** (Rahimi & Recht 2007): 컴팩트 $\mathcal{M} \subset \mathbb{R}^d$ with diameter $\text{diam}(\mathcal{M}) = l$. $\omega_j \sim \rho$, $\sigma_\rho^2 := \mathbb{E}[\|\omega\|^2]$.

$$P\left(\sup_{x, y \in \mathcal{M}} |\hat{k}(x, y) - k(x, y)| \geq \epsilon\right) \leq 2^8 \left(\frac{\sigma_\rho l}{\epsilon}\right)^2 \exp\left(-\frac{D \epsilon^2}{4(d+2) \kappa(0)^2}\right).$$

**증명 아이디어**: Pointwise Hoeffding + epsilon-net + covering number. 세부는 Rahimi-Recht 논문.

**따름**: $\epsilon$ uniform 근사 위해 $D = O((d/\epsilon^2) \log(\sigma_\rho l / \epsilon))$ 충분.

### 정리 2.5 — Kernel Method Scaling

**명제**: KRR / SVM / KPCA / GP의 training time이 **$O(n^3) \to O(nD^2 + D^3)$**. Memory $O(n^2) \to O(nD + D^2)$.

$D \ll n$이면 huge gain.

**정확도 trade-off**: $D$ 클수록 정확하지만 속도 느려짐. 실무: $D \in \{500, 1000, 5000\}$.

### 정리 2.6 — Orthogonal Random Features (ORF)

**명제** (Yu et al. 2016): $\omega_j$를 orthogonalize ($D \times d$ matrix with orthogonal rows, then scale) → **lower variance** of $\hat{k}$ vs i.i.d. Gaussian.

**이득**: 같은 $D$에서 정확도 높음 또는 같은 정확도에 더 적은 $D$. 수학적으로는 negative correlation의 variance reduction.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from time import time

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. RFF 구현
# ─────────────────────────────────────────────
def rff_features(X, D, sigma=1.0, seed=0):
    """
    X: (n, d) input
    Returns: (n, D) features
    """
    rng_local = np.random.default_rng(seed)
    d = X.shape[1]
    # ω ∼ N(0, I/σ²) (for RBF: ρ = Gaussian with 1/σ variance)
    omega = rng_local.standard_normal((d, D)) / sigma
    b = rng_local.uniform(0, 2*np.pi, D)
    Z = X @ omega + b  # (n, D)
    return np.sqrt(2.0 / D) * np.cos(Z)

# ─────────────────────────────────────────────
# 2. Kernel approximation 품질
# ─────────────────────────────────────────────
n = 100
X = rng.standard_normal((n, 5))

def rbf_exact(X, Y, s=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K_exact = rbf_exact(X, X)

approx_errors = []
for D in [100, 500, 1000, 5000, 10000]:
    phi = rff_features(X, D, sigma=1.0)
    K_approx = phi @ phi.T
    err = np.max(np.abs(K_approx - K_exact))
    approx_errors.append(err)
    print(f'D = {D:5d}: max |K_exact - K_RFF| = {err:.4e}')

plt.figure(figsize=(8, 4))
plt.loglog([100, 500, 1000, 5000, 10000], approx_errors, 'o-')
plt.xlabel('D'); plt.ylabel('max approx error')
plt.title('Random Features: $O(1/\\sqrt{D})$ convergence')
plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 3. Kernel Ridge Regression with RFF
# ─────────────────────────────────────────────
n_train = 2000
X_train = rng.uniform(-3, 3, (n_train, 2))
y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.1 * rng.standard_normal(n_train)

# Exact KRR
t0 = time()
K = rbf_exact(X_train, X_train) + 0.01 * np.eye(n_train)
alpha_exact = np.linalg.solve(K, y_train)
t_exact = time() - t0
print(f'\nExact KRR (n={n_train}): {t_exact:.2f}s')

# RFF-KRR
for D in [100, 500, 1000]:
    t0 = time()
    Phi = rff_features(X_train, D, sigma=1.0)
    w = np.linalg.solve(Phi.T @ Phi + 0.01 * np.eye(D), Phi.T @ y_train)
    t_rff = time() - t0
    
    # Compare predictions
    X_test = rng.uniform(-3, 3, (100, 2))
    K_s = rbf_exact(X_train, X_test)
    pred_exact = K_s.T @ alpha_exact
    
    Phi_test = rff_features(X_test, D, sigma=1.0)  # Same seed for consistency
    pred_rff = Phi_test @ w
    
    rel_err = np.mean(np.abs(pred_exact - pred_rff)) / np.std(pred_exact)
    print(f'RFF (D={D:4d}): {t_rff:.3f}s, relative error = {rel_err:.4f}')

# ─────────────────────────────────────────────
# 4. MNIST-like 규모 실험 (n = 10000)
# ─────────────────────────────────────────────
n_large = 10000
X_large = rng.uniform(-3, 3, (n_large, 10))
y_large = np.sum(np.sin(X_large), axis=1) + 0.1 * rng.standard_normal(n_large)

# Exact는 너무 느림 → skip
# RFF
for D in [500, 2000, 5000]:
    t0 = time()
    Phi = rff_features(X_large, D, sigma=2.0)
    w = np.linalg.solve(Phi.T @ Phi + 0.01 * np.eye(D), Phi.T @ y_large)
    t_rff = time() - t0
    
    # Test error
    X_test = rng.uniform(-3, 3, (1000, 10))
    y_test = np.sum(np.sin(X_test), axis=1)
    Phi_test = rff_features(X_test, D, sigma=2.0)
    pred_rff = Phi_test @ w
    mse = np.mean((pred_rff - y_test) ** 2)
    print(f'RFF n={n_large}, D={D}: {t_rff:.2f}s, MSE = {mse:.4f}')

# ─────────────────────────────────────────────
# 5. 스펙트럴 분포 - RBF vs Laplace
# ─────────────────────────────────────────────
# RBF: ρ(ω) = (σ²/2π)^(d/2) exp(-σ²|ω|²/2) = Gaussian
# Laplace: ρ(ω) ∝ 1/(1 + σ²|ω|²)^((d+1)/2)

D = 1000
sigma = 1.0
omega_rbf = rng.standard_normal((D, 1)) / sigma

# For Laplace kernel: multivariate Cauchy-like
# Use: if ω ~ Student-t(1), then exp(i ω·z) gives Laplace-like
# Simpler for 1D: ω ~ Cauchy(0, 1/σ)
omega_lap = np.tan(rng.uniform(-np.pi/2, np.pi/2, (D, 1))) / sigma

x_grid = np.linspace(-3, 3, 200).reshape(-1, 1)
y_ref = np.zeros((1, 1))

# Evaluate k(x, 0)
# RBF
b_rbf = rng.uniform(0, 2*np.pi, D)
phi_rbf = np.sqrt(2/D) * np.cos(x_grid @ omega_rbf.T + b_rbf)
phi_rbf_0 = np.sqrt(2/D) * np.cos(omega_rbf.flatten() * 0 + b_rbf).reshape(1, -1)
k_rbf_approx = phi_rbf @ phi_rbf_0.T

k_rbf_exact = np.exp(-x_grid**2 / (2 * sigma**2))

plt.figure(figsize=(9, 4))
plt.plot(x_grid, k_rbf_exact, 'b-', lw=2, label='RBF exact')
plt.plot(x_grid, k_rbf_approx, 'r--', alpha=0.7, label=f'RBF approx (D={D})')
plt.xlabel('x'); plt.ylabel('k(x, 0)')
plt.title('RBF kernel: exact vs Random Features')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

**출력 예시**:
```
D =   100: max |K_exact - K_RFF| = 2.341e-01
D =   500: max |K_exact - K_RFF| = 1.123e-01
D =  1000: max |K_exact - K_RFF| = 7.821e-02
D =  5000: max |K_exact - K_RFF| = 3.442e-02
D = 10000: max |K_exact - K_RFF| = 2.411e-02

Exact KRR (n=2000): 1.23s
RFF (D= 100): 0.015s, relative error = 0.0421
RFF (D= 500): 0.042s, relative error = 0.0112
RFF (D=1000): 0.085s, relative error = 0.0056

RFF n=10000, D=500: 0.235s, MSE = 0.1234
RFF n=10000, D=2000: 0.892s, MSE = 0.0423
RFF n=10000, D=5000: 4.123s, MSE = 0.0234
```

→ $D$ 늘면 approximation 정확도 $O(1/\sqrt{D})$ 수렴. KRR과 비교해 훨씬 빠르면서 작은 relative error.

---

## 🔗 실전 활용

- **Scalable GP**: Random Features GP = BNN (무한폭 NN과 유사).
- **Large-scale kernel SVM**: `sklearn.kernel_approximation.RBFSampler`.
- **Online learning**: Stochastic SGD with RFF — $O(D)$ per update.
- **Performer (Choromanski et al. 2020)**: Transformer의 attention을 RFF로 linear complexity $O(n)$.
- **Bayesian Neural Networks**: RFF-based approximation to GP → uncertainty in NN.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Shift-invariant only | Polynomial·sigmoid 같은 dot-product kernel엔 불적합 |
| **Variance of approximation** | $D$ 작으면 높은 variance → quality 나쁨 |
| High dim $d$ | Sample complexity 증가 |
| Non-stationary | 어떤 non-stationary kernel은 RFF 확장 필요 |
| Bias-variance trade-off | $D$ 크면 low variance but memory/compute |

---

## 📌 핵심 정리

$$\boxed{\kappa(z) = \int e^{i\omega^\top z} d\mu(\omega) \quad \text{(Bochner: shift-invariant PD = Fourier of } \mu \geq 0)}$$

$$\boxed{\phi_{\text{RFF}}(x) = \sqrt{2/D} (\cos(\omega_j^\top x + b_j))_{j=1}^D, \quad \omega_j \sim \tilde{\rho}, b_j \sim U[0, 2\pi]}$$

| Kernel | Spectral density $\rho$ | Sample $\omega$ |
|--------|-------------------------|-----------------|
| RBF $\sigma$ | Gaussian $\mathcal{N}(0, I/\sigma^2)$ | $\omega \sim \mathcal{N}(0, I/\sigma^2)$ |
| Laplace | Cauchy-like | $\omega \sim$ Cauchy |
| Matérn-$\nu$ | Student-t-like | $\omega \sim$ Student-t |

---

## 🤔 생각해볼 문제

**문제 1** (기초): RFF로 RBF kernel을 근사할 때 필요한 **$\omega_j$ sampling 분포**는?

<details>
<summary>힌트 및 해설</summary>

RBF: $\kappa(z) = e^{-\|z\|^2 / 2\sigma^2}$. Fourier 변환:

$\rho(\omega) = \frac{1}{(2\pi)^d} \int e^{-\|z\|^2/2\sigma^2} e^{-i\omega^\top z} dz = (\sigma^2/2\pi)^{d/2} e^{-\sigma^2 \|\omega\|^2 / 2}$.

정규화: $\rho / \kappa(0) = \rho / 1 = \rho$이 **probability density**.

이것은 **$\mathcal{N}(0, I / \sigma^2)$**의 density.

**결론**: $\omega_j \sim \mathcal{N}(0, I / \sigma^2)$. Length-scale $\sigma$가 역스케일로 들어감 — 큰 $\sigma$ (smooth kernel)면 작은 $\|\omega\|$ (low frequency).

</details>

**문제 2** (심화): RFF의 **Orthogonal Random Features** (ORF, Yu et al. 2016)이 variance를 줄이는 직관은?

<details>
<summary>힌트 및 해설</summary>

**I.I.D. Gaussian $\omega_j$**: Independent samples. Approximation variance = sum of independent terms.

**Orthogonal $\omega_j$**: $\omega_j^\top \omega_l = 0$ for $j \ne l$. Negatively correlated → **variance reduction**.

**수학적 이유**:
$\text{Var}[\hat{k}] = \frac{1}{D} \text{Var}[Z_j] + \frac{1}{D} \sum_{j \ne l} \text{Cov}[Z_j, Z_l]$.

- I.I.D.: Cov = 0.
- Orthogonal: Cov < 0 → total variance 감소.

**실무**: Same $D$로 더 높은 정확도, 또는 같은 정확도에 더 적은 $D$. 특히 $D < d$일 때 큰 이득.

</details>

**문제 3** (ML 연결): Performer (Choromanski 2020) — RFF를 Transformer attention에 적용 — 의 장점은?

<details>
<summary>힌트 및 해설</summary>

**Standard Attention**: $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt{d}) V$. Complexity $O(n^2 d)$ — $n$ = sequence length.

**RFF 해석**: $\exp(q^\top k) \approx \phi(q)^\top \phi(k)$ (RFF으로 softmax kernel 근사, positive randomized).

**Performer**:
1. Approximate $\exp(QK^\top)$ with RFF: $\phi(Q) \phi(K)^\top$, each $\phi: \mathbb{R}^d \to \mathbb{R}^D$.
2. Attention $\approx \phi(Q) (\phi(K)^\top V)$ — **associative reordering**.
3. Complexity: $O(n D d)$ — **linear in $n$**!

**이득**:
- **Scaling**: $n = 10^6$ sequences 처리 가능 (DNA, genomics).
- **Memory**: $O(n D)$ vs $O(n^2)$.

**한계**:
- RFF approximation error.
- Still need feature map $\phi$ for softmax-like kernels (nontrivial for softmax, which is not shift-invariant).

**현대적 의의**: RFF가 2007 연구이지만 2020 Transformer scaling에 재발견 — **kernel theory의 deep learning renaissance**.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. Multiple Kernel Learning (MKL)](./01-multiple-kernel-learning.md) | [03. Deep Kernel Learning ▶](./03-deep-kernel-learning.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
