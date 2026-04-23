# 01. Kernel Ridge Regression 완전 유도

## 🎯 핵심 질문

- Ridge regression의 **dual form** $\alpha = (K + \lambda I)^{-1} y$은 어떻게 유도되는가?
- Representer 정리의 직접 응용 — 왜 KRR의 해가 이 closed-form인가?
- 예측 공식 $f(x) = k(x)^\top (K + \lambda I)^{-1} y$에서 **$\lambda$의 역할**은 smoothness를 어떻게 제어하는가?
- $\lambda \to 0$ (interpolation)과 $\lambda \to \infty$ (maximum shrinkage) 극한에서는 무엇이 일어나는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

KRR은 kernel method의 **"Hello World"**: 가장 간단하고 가장 자주 쓰이는 kernel regression. **Closed-form 해**가 존재해 수치적으로 안정적이고, GP posterior mean과 정확히 일치(Ch4-03)하여 Bayesian 해석 가능. 또한 NTK 이론(Ch7-04)에서 "무한폭 NN의 gradient descent"가 KRR로 수렴함을 보여 — **NN과 kernel method의 연결고리**. 실무에서 SVM이 분류의 표준이라면 KRR은 회귀의 "baseline + uncertainty bonus". 특히 noise 크고 target smooth한 scientific regression (물리, 화학 모델링)에서 default.

---

## 📐 수학적 선행 조건

- [Ch2-03 Representer 정리](../ch2-rkhs-representer/03-representer-theorem.md)
- [Ch2-04 계산적 환원](../ch2-rkhs-representer/04-computational-reduction.md)
- [Ch4-03 GP = KRR](../ch4-gaussian-process/03-gp-equals-krr.md)
- 선형대수: Woodbury matrix identity, Cholesky

---

## 📖 직관적 이해

### Ridge Regression → Kernel Ridge Regression

**Ridge regression** (linear):

$$\min_w \|y - Xw\|^2 + \lambda \|w\|^2.$$

Closed-form: $w^* = (X^\top X + \lambda I)^{-1} X^\top y$.

**Kernel Ridge Regression** (non-linear, via RKHS $\mathcal{H}_k$):

$$\min_{f \in \mathcal{H}_k} \sum_i (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}_k}^2.$$

Representer 정리: $f^* = \sum_i \alpha_i k(\cdot, x_i)$. Closed-form: $\alpha = (K + \lambda I)^{-1} y$.

### Primal ↔ Dual Form

**Primal** (linear만):

$$w^* = (X^\top X + \lambda I)^{-1} X^\top y \in \mathbb{R}^d.$$

Prediction: $f(x) = x^\top w^*$.

**Dual** (kernel-ready):

$$\alpha = (XX^\top + \lambda I)^{-1} y \in \mathbb{R}^n.$$

Prediction: $f(x) = x^\top X^\top \alpha = \sum_i \alpha_i x^\top x_i$.

$x^\top x_i \to k(x, x_i)$ 치환 → **KRR**:

$$\alpha = (K + \lambda I)^{-1} y, \quad f(x) = \sum_i \alpha_i k(x_i, x).$$

**Woodbury identity**가 primal ↔ dual 변환의 수학적 근거.

### $\lambda$의 Smoothness 제어

$(K + \lambda I)^{-1}$에서 $K$의 고유값 $\mu_i$가 $\mu_i + \lambda$로 shift:

- $\mu_i \gg \lambda$: 거의 영향 없음 → 큰 변동 허용.
- $\mu_i \ll \lambda$: shrinkage 강함 → 해당 모드 억제.

$K$의 **작은 고유값** = 고주파 (진동) 모드. $\lambda$가 이들을 더 많이 억제 → smoother solution.

RKHS 관점: $\lambda \|f\|_{\mathcal{H}_k}^2$ 페널티가 "진동" (small eigenvalue mode) 함수에 큰 벌 → smoothness 강제.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — KRR 최적화 문제

PD kernel $k$, RKHS $\mathcal{H}_k$, $\lambda > 0$:

$$\min_{f \in \mathcal{H}_k} L(f) := \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}_k}^2.$$

### 정의 1.2 — KRR의 Closed-Form 해

Representer 정리 후:

$$\alpha^* = (K + \lambda I)^{-1} y \in \mathbb{R}^n$$

$$f^*(x) = k(x)^\top \alpha^* = \sum_{i=1}^n \alpha_i^* k(x_i, x)$$

where $k(x) = (k(x, x_1), \ldots, k(x, x_n))^\top$.

---

## 🔬 정리와 증명

### 정리 1.1 — KRR Closed-Form 유도

**명제**: 정의 1.1의 최적해는 정의 1.2.

**증명**:

Representer 정리로 $f = \sum_i \alpha_i k_{x_i}$. 대입:

- $f(x_j) = \sum_i \alpha_i k(x_j, x_i) = (K\alpha)_j$.
- $\|f\|_{\mathcal{H}_k}^2 = \alpha^\top K \alpha$.

Objective:
$$J(\alpha) = \|y - K\alpha\|^2 + \lambda \alpha^\top K \alpha.$$

$\nabla_\alpha J = -2 K (y - K \alpha) + 2 \lambda K \alpha = 0$:

$$K y = K K \alpha + \lambda K \alpha = K(K + \lambda I) \alpha.$$

$K$ invertible이면 $\alpha = (K + \lambda I)^{-1} y$. $\lambda > 0$이면 $K + \lambda I$ strictly PD → 항상 invertible. $\square$

### 정리 1.2 — 예측 공식

**명제**: 새 점 $x$에서

$$f^*(x) = k(x)^\top (K + \lambda I)^{-1} y = \sum_{i=1}^n \alpha_i k(x_i, x).$$

**증명**: 정리 1.1 + Representer. $\square$

### 정리 1.3 — Primal-Dual Equivalence (Linear Kernel)

**명제**: Linear kernel $k(x, y) = x^\top y$에서 KRR의 dual 해 $\alpha = (XX^\top + \lambda I)^{-1} y$와 primal 해 $w = (X^\top X + \lambda I)^{-1} X^\top y$는 **Woodbury identity**로 일치.

$$f^*(x) = x^\top w = x^\top (X^\top X + \lambda I)^{-1} X^\top y = (Xx)^\top (XX^\top + \lambda I)^{-1} y.$$

**증명 (Woodbury)**:

$$A^{-1} U (I + V^\top A^{-1} U)^{-1} V^\top = (A + UV^\top)^{-1} UV^\top \cdot \text{(complicated)}$$

표준 Woodbury: $(A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}$.

간단 경로: $(X^\top X + \lambda I) X^\top = X^\top (XX^\top + \lambda I) \Rightarrow X^\top (XX^\top + \lambda I)^{-1} = (X^\top X + \lambda I)^{-1} X^\top$. 양변에 $y$ 곱하기. $\square$

**함의**: $d < n$ (저차원)이면 primal이 효율적 ($O(d^3)$), $n < d$ (고차원)이면 dual이 효율적 ($O(n^3)$). Kernel SVM은 $d = \infty$라 항상 dual.

### 정리 1.4 — $\lambda$의 극한

**명제**:
1. $\lambda \to 0^+$: $\alpha \to K^{-1} y$ (interpolating). $f^*(x_i) = y_i$ exactly. RKHS norm $\|f^*\|^2 = y^\top K^{-1} y$.
2. $\lambda \to \infty$: $\alpha \to 0$, $f^* \to 0$ (prior mean). Training 완전 무시.

**증명**: 

(1) $K + \lambda I \to K$. $K^{-1}$이 존재하면 ($K$ strict PD) 바로. Training 점에서 $(K \alpha)_i = y_i$ 확인.

(2) $K + \lambda I \to \lambda I$, $(K + \lambda I)^{-1} \to \lambda^{-1} I \to 0$, 따라서 $\alpha \to 0$. $\square$

**실무**: 적절한 $\lambda$는 cross-validation 또는 GP marginal likelihood (Ch4-05).

### 정리 1.5 — Leave-One-Out CV의 Closed Form

**명제**: LOO CV error도 closed form:

$$\text{LOOCV}(\lambda) = \frac{1}{n} \sum_{i=1}^n \left(\frac{y_i - f^{(-i)}(x_i)}{1 - h_{ii}}\right)^2$$

여기서 $f^{(-i)}$는 $i$-th 점 제외하고 학습, $H = K (K + \lambda I)^{-1}$ (hat matrix), $h_{ii} = H_{ii}$.

**증명**: Sherman-Morrison formula를 이용해 $(K^{(-i)} + \lambda I)^{-1}$을 $(K + \lambda I)^{-1}$로부터 rank-1 update로 표현. 결과는 위 공식. (Rasmussen & Williams 5.4.1) $\square$

**실무 의미**: $\lambda$ tuning을 **$O(n^3)$ 한 번** ($K^{-1}$ 계산)에 LOO CV 값을 모든 $i$에 대해 얻음. Grid search over $\lambda$: $O(n^3 + n L)$ for $L$ values of $\lambda$.

### 정리 1.6 — KRR의 Bias-Variance 분해

**명제 (비공식)**: Target $f^*$ (미지), noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$. KRR prediction $\hat{f}_\lambda$의 expected squared error:

$$\mathbb{E}[(f^*(x) - \hat{f}_\lambda(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2.$$

- $\lambda \uparrow$: Bias $\uparrow$, Variance $\downarrow$.
- $\lambda \downarrow$: Bias $\downarrow$, Variance $\uparrow$.

**Minimum**: 적절한 $\lambda^*$에서 total error 최소. 이것이 "bias-variance trade-off".

**정확한 rate**: Target이 RKHS에 있으면 KRR error $\mathcal{O}(n^{-\alpha})$, $\alpha$는 kernel 고유값 감쇠율에 의존 (Ch2-05).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

rng = np.random.default_rng(42)

# 데이터
n = 30
X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
y = np.sin(X).flatten() + 0.2 * rng.standard_normal(n)

def rbf(X, Y, s=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K = rbf(X, X)
X_test = np.linspace(-4, 4, 300).reshape(-1, 1)
K_s = rbf(X, X_test)

# ─────────────────────────────────────────────
# 1. KRR with various λ
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, lam in zip(axes, [0.001, 0.01, 0.1, 1.0]):
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    pred = K_s.T @ alpha
    ax.plot(X_test, np.sin(X_test), 'k--', alpha=0.3, label='true sin')
    ax.plot(X_test, pred, 'b-', lw=2, label=f'KRR (λ={lam})')
    ax.scatter(X, y, c='red', s=30, zorder=5)
    ax.set_title(f'λ = {lam}')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 2. sklearn과 수치 일치
# ─────────────────────────────────────────────
lam = 0.1
alpha_manual = np.linalg.solve(K + lam * np.eye(n), y)

krr = KernelRidge(alpha=lam, kernel='rbf', gamma=0.5)
krr.fit(X, y)
# sklearn의 alpha는 dual_coef_로 접근
alpha_sklearn = krr.dual_coef_

print(f'α 최대 차이 (manual vs sklearn): {np.max(np.abs(alpha_manual - alpha_sklearn)):.2e}')

# ─────────────────────────────────────────────
# 3. LOO CV closed-form
# ─────────────────────────────────────────────
from scipy.linalg import solve
lams = np.logspace(-4, 2, 30)
loocv_errs = []
for lam in lams:
    A_inv = np.linalg.inv(K + lam * np.eye(n))
    H = K @ A_inv  # hat matrix
    h_diag = np.diag(H)
    y_hat = H @ y
    resid = y - y_hat
    loocv = np.mean((resid / (1 - h_diag + 1e-10))**2)
    loocv_errs.append(loocv)

plt.figure(figsize=(8, 4))
plt.semilogx(lams, loocv_errs, 'o-')
best_lam = lams[np.argmin(loocv_errs)]
plt.axvline(best_lam, color='red', linestyle='--', label=f'λ* = {best_lam:.4f}')
plt.xlabel('λ (log)'); plt.ylabel('LOO CV MSE')
plt.title('LOO CV closed-form')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. Primal-Dual equivalence (Linear kernel)
# ─────────────────────────────────────────────
X_high = rng.standard_normal((n, 100))  # d=100
y_high = X_high @ rng.standard_normal(100) * 0.1 + 0.1 * rng.standard_normal(n)

lam = 0.1

# Dual form (n=30)
K_lin = X_high @ X_high.T
alpha_dual = np.linalg.solve(K_lin + lam * np.eye(n), y_high)
w_from_dual = X_high.T @ alpha_dual

# Primal form (d=100)
w_primal = np.linalg.solve(X_high.T @ X_high + lam * np.eye(100), X_high.T @ y_high)

print(f'w 차이 (primal vs dual): {np.linalg.norm(w_primal - w_from_dual):.2e}')

# ─────────────────────────────────────────────
# 5. 극한 case
# ─────────────────────────────────────────────
# λ → 0: interpolation
alpha_interp = np.linalg.solve(K + 1e-10 * np.eye(n), y)
pred_train_interp = K @ alpha_interp
print(f'\nλ → 0: training points fit error = {np.max(np.abs(pred_train_interp - y)):.2e}')

# λ → ∞: → 0
alpha_shrink = np.linalg.solve(K + 1e10 * np.eye(n), y)
print(f'λ → ∞: |α| max = {np.max(np.abs(alpha_shrink)):.2e} (→ 0)')
```

**출력 예시**:
```
α 최대 차이 (manual vs sklearn): 4.44e-16
w 차이 (primal vs dual): 1.23e-14
λ → 0: training points fit error = 3.32e-08
λ → ∞: |α| max = 7.82e-11 (→ 0)
```

→ sklearn과 완전 일치. Primal-dual equivalence 수치적 확인. 극한 case 예상대로.

---

## 🔗 실전 활용

- **Baseline regressor**: 복잡 모델 전에 KRR로 baseline. `sklearn.kernel_ridge.KernelRidge`.
- **Uncertainty 추가**: KRR을 GP로 재해석 → posterior variance 얻음.
- **Efficient hyperparameter tuning**: LOO CV closed form으로 $\lambda$, kernel length-scale 빠르게 선택.
- **NTK / Deep learning theory**: 무한폭 NN의 gradient flow가 KRR로 수렴 → KRR이 deep learning의 "theoretical proxy".
- **Feature engineering 대안**: Kernel 선택으로 비선형성을 자동 포함.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Squared loss | Heavy-tail noise에서 덜 robust; Huber loss 또는 $\epsilon$-insensitive 대안 |
| $(K + \lambda I)^{-1}$ | $O(n^3)$, $n \geq 10^4$이면 approximation 필요 |
| **$\lambda$ 수동 선택** | CV 또는 GP marginal likelihood로 자동화 |
| No sparsity | 모든 점이 $\alpha_i \ne 0$ → SVM의 sparsity 대조 |
| Numerical stability | $K$ 조건수 나쁘면 jitter $10^{-6} I$ 추가 |

---

## 📌 핵심 정리

$$\boxed{\alpha = (K + \lambda I)^{-1} y \quad ; \quad f^*(x) = k(x)^\top \alpha}$$

$$\boxed{\lambda \|f\|_{\mathcal{H}_k}^2 = \lambda \alpha^\top K \alpha \quad \text{(smoothness penalty)}}$$

| Case | 결과 |
|------|------|
| $\lambda \to 0$ | Interpolation: $f(x_i) = y_i$ exactly |
| $\lambda \to \infty$ | $\alpha \to 0$: prior mean |
| Linear kernel | Ridge regression의 dual form |
| Computation | $O(n^3)$ for inverse, $O(n^2)$ memory |

---

## 🤔 생각해볼 문제

**문제 1** (기초): KRR에서 **$\lambda$를 작게** 설정했을 때 가능한 실패 모드는?

<details>
<summary>힌트 및 해설</summary>

1. **Overfitting**: Training data interpolation, noise 포착 → test error 큼.
2. **Numerical instability**: $K$의 작은 고유값에서 $(K + \lambda I)^{-1}$의 큰 성분 → 증폭된 수치 오차.
3. **Condition number**: $\kappa(K + \lambda I) = (\mu_{\max} + \lambda) / (\mu_{\min} + \lambda)$, $\lambda \to 0$에서 폭발.

**해결**:
- $\lambda$ 너무 작게 설정 금지 (최소한 $10^{-6}$).
- Jitter trick: $K + \max(\lambda, 10^{-6}) I$.
- Regularization으로 충분, interpolation 원하면 GP interpolation 별도 고려.

</details>

**문제 2** (심화): KRR과 Kernel Smoothers (Nadaraya-Watson)의 차이는?

<details>
<summary>힌트 및 해설</summary>

**Nadaraya-Watson** (NW):
$$\hat{f}_{\text{NW}}(x) = \frac{\sum_i K_h(x - x_i) y_i}{\sum_i K_h(x - x_i)}.$$

각 점의 값을 **kernel 가중 평균**. **Local** 방법.

**KRR**:
$$\hat{f}_{\text{KRR}}(x) = \sum_i \alpha_i k(x_i, x), \quad \alpha = (K + \lambda I)^{-1} y.$$

**Global 최적화** + L2 regularization.

**차이**:
- NW: simpler, local, bias larger at boundaries.
- KRR: more accurate, global, handles noise better via $\lambda$.
- Both use kernel for similarity, but NW is purely local averaging, KRR is RKHS-based.

**Connection**: NW의 weight $\alpha_i(x) = K_h(x - x_i) / \sum_j K_h(x - x_j)$은 data-dependent이지만 KRR의 $\alpha$는 training 시 한 번 결정 → faster inference.

</details>

**문제 3** (ML 연결): Why is KRR often described as "**neural network's limit**" in NTK theory?

<details>
<summary>힌트 및 해설</summary>

Jacot et al. (2018): 무한폭 NN $f_\theta$, gradient flow $\dot{\theta} = -\nabla L$, MSE loss. Infinite width limit:

$$\dot{f}(x) = -\sum_i (f(x_i) - y_i) \Theta(x, x_i)$$

$\Theta$ = NTK, fixed kernel.

해: $t \to \infty$ with gradient flow (no regularization, no noise):

$$f(x_*) = \Theta_*^\top \Theta^{-1} y$$

이것은 **KRR with $\lambda = 0$**.

Finite width NN with gradient descent + weight decay → approximately **KRR with $\lambda \propto$ weight decay**.

**함의**:
- NN의 "minimum 점" = KRR 해 (무한폭에서).
- NN training의 generalization = NTK RKHS analysis.
- Deep learning theory ↔ kernel theory 연결 (Layer 2 Generalization).

**실무적 의미**: KRR은 NN의 theoretical model로 이해. NN의 수렴·generalization 분석이 kernel theory 도구로 가능.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch4-06. Sparse GP와 Inducing Points](../ch4-gaussian-process/06-sparse-gp.md) | [02. Kernel PCA의 수학 ▶](./02-kernel-pca.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
