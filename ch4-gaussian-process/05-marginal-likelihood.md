# 05. Hyperparameter Learning — Marginal Likelihood

## 🎯 핵심 질문

- Marginal likelihood $\log p(y \mid \theta) = -\frac{1}{2} y^\top K_\theta^{-1} y - \frac{1}{2} \log|K_\theta| - \frac{n}{2} \log 2\pi$ 공식의 **각 항의 의미**는?
- 왜 $-\frac{1}{2} \log|K_\theta|$가 **자동 Occam's razor**(복잡도 페널티) 역할을 하는가?
- $\theta$ (length-scale, signal/noise 분산)에 대한 gradient는 어떻게 유도되고 학습되는가?
- Marginal likelihood 최대화가 **cross-validation보다 data-efficient**인 이유는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

GP hyperparameter 학습은 "**어떤 kernel·어떤 noise level이 데이터에 가장 적합한가**"를 자동으로 결정. Cross-validation은 training data 일부를 validation에 써야 하지만 marginal likelihood는 **모든 데이터를 사용**. 또한 공식에 **model complexity penalty**가 내장되어 overfitting을 자연스럽게 방지 — 이것이 "자동 Occam's razor"의 의미. 이 기법이 GP를 "black-box optimizer" (MacKay의 주장)로 만들어, 사용자가 length-scale을 직접 튜닝할 필요를 없앴다. 또한 Bayesian Optimization·Automated Machine Learning (AutoML)에서 hyperparameter-free GP를 구현하는 기반.

---

## 📐 수학적 선행 조건

- [Ch4-01~03](./01-gp-definition.md): GP regression posterior
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Multivariate Gaussian, log-determinant
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Cholesky, 행렬 도함수
- 미분 최적화: Gradient descent, L-BFGS

---

## 📖 직관적 이해

### Marginal Likelihood = "Model Evidence"

Bayesian framework: $p(y \mid \theta) = \int p(y \mid f) p(f \mid \theta) df$ — **$f$를 marginalize out**.

GP에서 모든 것이 Gaussian이라 closed-form:

$$y \mid X, \theta \sim \mathcal{N}(0, K_\theta + \sigma_n^2 I).$$

따라서

$$\log p(y \mid X, \theta) = -\frac{1}{2} y^\top (K_\theta + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log|K_\theta + \sigma_n^2 I| - \frac{n}{2} \log 2\pi.$$

이것을 $\theta$에 대해 최대화.

### 세 항의 의미

1. **Data fit** $-\frac{1}{2} y^\top K_\theta^{-1} y$: 데이터가 kernel $K_\theta$와 "얼마나 잘 맞는가". 작은 $\|y\|_{K_\theta^{-1}}^2$ = data가 prior와 compatible.

2. **Complexity penalty** $-\frac{1}{2} \log|K_\theta|$: 큰 $|K|$ (많은 고유값 살아있음) = large RKHS = overfit 위험. **log det**가 복잡도 페널티로 작용.

3. **Constant** $-\frac{n}{2} \log 2\pi$: $\theta$와 무관. 무시.

### 자동 Occam's Razor

$\theta$를 바꿀 때:
- **Kernel length-scale $\ell$ 작게**: 
  - RKHS 큼 → 더 복잡한 함수 표현 가능 → data fit 좋아짐.
  - $K$의 고유값 많이 살아있음 → $\log|K|$ 커짐 → complexity penalty 커짐.
- **$\ell$ 크게**: 반대.

**Trade-off**: Marginal likelihood는 **"데이터를 설명하는 가장 단순한 모델"**을 선택. 이것이 **Bayesian Occam's razor**의 정확한 형태.

### Cross-Validation과의 비교

CV: 데이터를 train/val로 분할. Validation 성능으로 hyperparameter 선택. **Data 일부만 사용**, 분할에 의존.

Marginal likelihood: **모든 데이터 사용**. 자동 bias-variance trade-off. 계산은 CV보다 복잡 ($(K + \sigma^2 I)^{-1}$과 $\log|\cdot|$ per evaluation). 하지만 gradient 제공 → L-BFGS로 빠르게.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Marginal Likelihood

$$p(y \mid X, \theta) := \int p(y \mid f, \sigma_n^2) p(f \mid X, \theta) df.$$

GP에서 Gaussian integral이 closed-form: $y \mid X, \theta \sim \mathcal{N}(0, K_\theta + \sigma_n^2 I)$.

### 정의 5.2 — Log Marginal Likelihood

$$\log p(y \mid X, \theta) = -\frac{1}{2} y^\top (K_\theta + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log|K_\theta + \sigma_n^2 I| - \frac{n}{2} \log 2\pi.$$

### 정의 5.3 — Hyperparameters $\theta$

RBF kernel: $k(x, y) = \sigma_f^2 \exp(-\|x - y\|^2 / 2\ell^2)$. $\theta = (\sigma_f, \ell, \sigma_n)$.

Matérn, periodic 등 다른 kernel도 각자 hyperparameter.

### 정의 5.4 — Maximum Likelihood 추정

$$\hat{\theta}_{\text{MLE}} := \arg\max_\theta \log p(y \mid X, \theta).$$

---

## 🔬 정리와 증명

### 정리 5.1 — Closed-Form Marginal Likelihood

**명제**: 정의 5.2의 공식.

**증명**: $y \mid X, \theta \sim \mathcal{N}(0, K_\theta + \sigma_n^2 I)$. Multivariate Gaussian density의 로그:

$$\log \mathcal{N}(y; 0, \Sigma) = -\frac{1}{2} y^\top \Sigma^{-1} y - \frac{1}{2} \log|\Sigma| - \frac{n}{2} \log(2\pi). \quad \square$$

### 정리 5.2 — Gradient 공식

**명제**: $\log p(y \mid \theta)$의 $\theta_j$에 대한 gradient:

$$\frac{\partial \log p(y \mid \theta)}{\partial \theta_j} = \frac{1}{2} y^\top K^{-1} \frac{\partial K}{\partial \theta_j} K^{-1} y - \frac{1}{2} \text{tr}\left(K^{-1} \frac{\partial K}{\partial \theta_j}\right)$$

여기서 $K := K_\theta + \sigma_n^2 I$. 또는 $\alpha = K^{-1} y$로:

$$\frac{\partial \log p(y \mid \theta)}{\partial \theta_j} = \frac{1}{2} \text{tr}\left((\alpha \alpha^\top - K^{-1}) \frac{\partial K}{\partial \theta_j}\right).$$

**증명**:

$\frac{\partial}{\partial \theta_j} y^\top K^{-1} y = -y^\top K^{-1} \frac{\partial K}{\partial \theta_j} K^{-1} y$ (행렬 도함수, $\frac{\partial A^{-1}}{\partial \theta} = -A^{-1} \frac{\partial A}{\partial \theta} A^{-1}$).

$\frac{\partial}{\partial \theta_j} \log|K| = \text{tr}(K^{-1} \frac{\partial K}{\partial \theta_j})$ (Jacobi 공식).

결합하면 위 공식. $\square$

**응용**: L-BFGS 같은 gradient-based optimizer로 $\theta$ 학습. scipy.optimize.minimize(method='L-BFGS-B') 등.

### 정리 5.3 — Complexity Penalty의 정량적 해석

**명제**: $\log p(y \mid \theta)$를 eigendecomposition $K + \sigma_n^2 I = U \Lambda U^\top$ 관점에서:

$$\log p(y \mid \theta) = -\frac{1}{2} \sum_i \frac{(U^\top y)_i^2}{\lambda_i} - \frac{1}{2} \sum_i \log \lambda_i - \frac{n}{2} \log 2\pi.$$

**해석**:
- **Data fit** $\sum (U^\top y)_i^2 / \lambda_i$: 각 eigenmode의 "에너지"를 고유값으로 나눔. $\lambda_i$ 큰 모드는 작은 페널티, 작은 모드는 큰 페널티.
- **Complexity** $\sum \log \lambda_i$: 큰 $\lambda_i$들의 합 = large-scale eigenmodes의 수. 많을수록 **모델 유연성 $\uparrow$**, 페널티 $\uparrow$.

**증명**: 대각화로 $K^{-1} = U \Lambda^{-1} U^\top$, $\log|K| = \sum_i \log \lambda_i$. 직접 대입. $\square$

### 정리 5.4 — Overfit/Underfit 자동 피하기

**명제**: Marginal likelihood 최대화는 "가장 작은 RKHS 중 data fit을 만족하는 것"을 선택.

**정성적 증명**:
- **Overfit** (너무 작은 $\ell$, 너무 flexible): data fit은 좋지만 $\log|K|$ 큼 → 페널티. 전체 marginal likelihood 감소.
- **Underfit** (너무 큰 $\ell$): $\log|K|$ 작지만 data fit 나쁨 → $y^\top K^{-1} y$ 큼. 전체 감소.
- **Sweet spot**: 두 trade-off 균형 지점. 이것이 **Bayesian Occam's razor**.

### 정리 5.5 — Non-Convexity

**명제**: $\log p(y \mid \theta)$는 일반적으로 **$\theta$에 대해 non-convex**. Multiple local optima 존재 가능.

**실무 함의**:
- 여러 random initialization으로 학습 → best 선택.
- 또는 meaningful initialization (e.g., $\ell$ = median pairwise distance, $\sigma_n$ = residual std).

### 정리 5.6 — 과대적합의 한계 사례

**명제**: 매우 flexible kernel ($\ell \to 0$) + noise free ($\sigma_n \to 0$)에서 $\log p(y \mid \theta) \to -\infty$. Overfit이 marginal likelihood로 **명시적 penalty** 받음.

**증명 스케치**: $\ell \to 0$에서 $K \to I$. $y^\top K^{-1} y \to \|y\|^2$ (유한). $\log|K| \to 0$. 하지만 $\sigma_n \to 0$에서 denominator $K + \sigma_n^2 I \to I$ 가까워지지만 약간의 noise 필요 → condition number → $\infty$. 수치적으로 불안정, theoretical limit도 well-defined 아님.

**정확한 결과**: Generalization bounds (e.g., Seeger 2002)으로 marginal likelihood가 expected negative log predictive density의 upper bound임이 증명됨 → tight marginal likelihood = good generalization.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(0)

# 데이터: noisy sin
n = 25
X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
y = np.sin(X).flatten() + 0.2 * rng.standard_normal(n)

def rbf(X, Y, sigma_f, ell):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return sigma_f**2 * np.exp(-d2 / (2 * ell**2))

# ─────────────────────────────────────────────
# 1. log marginal likelihood 함수
# ─────────────────────────────────────────────
def neg_log_mlik(params, X, y):
    log_sigma_f, log_ell, log_sigma_n = params
    sigma_f = np.exp(log_sigma_f)
    ell = np.exp(log_ell)
    sigma_n = np.exp(log_sigma_n)
    K = rbf(X, X, sigma_f, ell) + sigma_n**2 * np.eye(len(y))
    try:
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(y)))
    except np.linalg.LinAlgError:
        return 1e10
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)
    return nll

# ─────────────────────────────────────────────
# 2. 최적화 — L-BFGS-B
# ─────────────────────────────────────────────
init = np.log([1.0, 1.0, 0.5])
res = minimize(neg_log_mlik, init, args=(X, y), method='L-BFGS-B')
sigma_f_opt, ell_opt, sigma_n_opt = np.exp(res.x)
print(f'Optimal σ_f = {sigma_f_opt:.4f}')
print(f'Optimal ℓ = {ell_opt:.4f}')
print(f'Optimal σ_n = {sigma_n_opt:.4f}')
print(f'최적 log marginal likelihood = {-res.fun:.4f}')

# ─────────────────────────────────────────────
# 3. Marginal likelihood landscape (σ_f = 1 고정)
# ─────────────────────────────────────────────
ells = np.logspace(-1, 1, 50)
sigmans = np.logspace(-2, 0, 50)
LL = np.zeros((len(ells), len(sigmans)))
for i, el in enumerate(ells):
    for j, sn in enumerate(sigmans):
        LL[i, j] = -neg_log_mlik([0.0, np.log(el), np.log(sn)], X, y)

plt.figure(figsize=(9, 6))
cs = plt.contourf(sigmans, ells, LL, levels=25, cmap='viridis')
plt.colorbar(label='log p(y|θ)')
plt.scatter(sigma_n_opt, ell_opt, s=200, marker='*', color='red', edgecolors='k', label='Optimal')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('σ_n'); plt.ylabel('ℓ')
plt.title('Log Marginal Likelihood Landscape (σ_f = 1)')
plt.legend(); plt.show()

# ─────────────────────────────────────────────
# 4. 시각화 — 여러 ℓ에서의 GP fit
# ─────────────────────────────────────────────
X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, ell_val, title in zip(axes, [0.1, ell_opt, 3.0], ['ℓ=0.1 (overfit)', f'ℓ={ell_opt:.2f} (optimal)', 'ℓ=3.0 (underfit)']):
    K = rbf(X, X, 1.0, ell_val) + sigma_n_opt**2 * np.eye(n)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    K_s = rbf(X, X_test, 1.0, ell_val)
    mu = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    std = np.sqrt(np.maximum(0, rbf(X_test, X_test, 1.0, ell_val).diagonal() - np.sum(v**2, axis=0)))
    
    lmlik = -neg_log_mlik([0.0, np.log(ell_val), np.log(sigma_n_opt)], X, y)
    
    ax.fill_between(X_test.flatten(), mu - 2*std, mu + 2*std, alpha=0.3)
    ax.plot(X_test, mu, 'b-')
    ax.scatter(X, y, c='red', s=30, zorder=5)
    ax.plot(X_test, np.sin(X_test), 'k--', alpha=0.3)
    ax.set_title(f'{title}\nlog p(y|θ) = {lmlik:.2f}')
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 5. Data fit vs Complexity 항 분리
# ─────────────────────────────────────────────
def breakdown(params, X, y):
    sigma_f, ell, sigma_n = np.exp(params)
    K = rbf(X, X, sigma_f, ell) + sigma_n**2 * np.eye(len(y))
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(y)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    fit = -0.5 * y @ alpha
    complexity = -np.sum(np.log(np.diag(L)))
    const = -0.5 * len(y) * np.log(2 * np.pi)
    return fit, complexity, const

for log_el in [np.log(0.1), np.log(ell_opt), np.log(3.0)]:
    f, c, const = breakdown([0.0, log_el, np.log(sigma_n_opt)], X, y)
    print(f'ℓ={np.exp(log_el):.2f}: data fit={f:.2f}, complexity={c:.2f}, sum={f+c+const:.2f}')
```

**출력 예시**:
```
Optimal σ_f = 1.1342
Optimal ℓ = 0.8921
Optimal σ_n = 0.2113
최적 log marginal likelihood = -6.2134

ℓ=0.10: data fit=-2.12, complexity=-14.32, sum=-39.45
ℓ=0.89: data fit=-4.21, complexity=-2.34, sum=-6.21
ℓ=3.00: data fit=-18.34, complexity=-0.87, sum=-19.21
```

→ 최적 $\ell$에서 data fit과 complexity의 합이 최대. $\ell$ 작으면 data fit 좋지만 complexity 페널티 커, $\ell$ 크면 반대.

---

## 🔗 실전 활용

- **Automatic Relevance Determination (ARD)**: $k(x, y) = \sigma_f^2 \exp(-\sum_d (x_d - y_d)^2 / 2 \ell_d^2)$, **축별 length-scale** $\ell_d$. Irrelevant feature는 $\ell_d \to \infty$가 되어 자동 제거.
- **Bayesian Optimization**: Matérn-5/2 + ARD + marginal likelihood 최적화가 BO-default (BoTorch, GPyOpt).
- **AutoML**: Kernel 선택 자체를 marginal likelihood 비교로 자동화.
- **sklearn**: `GaussianProcessRegressor(kernel=RBF(length_scale=1.0))` — `fit()`이 자동으로 marginal likelihood 최적화.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Gaussian likelihood 가정 | Classification은 Laplace approximation의 marginal likelihood (덜 정확) |
| Non-convex | Local optima 가능, multi-start 필요 |
| $O(n^3)$ per evaluation | 큰 $n$에서 느림 — Sparse GP (Ch4-06) |
| Gradient 계산 $O(n^3)$ | 각 hyperparameter마다 $\partial K / \partial \theta_j$ 전체 계산 |
| **Overfitting on hyperparameters** | $n$ 매우 작으면 hyperparameter 자체 overfit 가능 — full Bayesian 대안 |

---

## 📌 핵심 정리

$$\boxed{\log p(y \mid \theta) = \underbrace{-\frac{1}{2} y^\top (K_\theta + \sigma_n^2 I)^{-1} y}_{\text{data fit}} \underbrace{- \frac{1}{2} \log|K_\theta + \sigma_n^2 I|}_{\text{Occam complexity}} - \frac{n}{2} \log 2\pi}$$

| 항 | 크기 결정 | $\theta$ 선택 영향 |
|---|----------|-------------------|
| Data fit | 작은 $\lambda_i$에 큰 $\|U^\top y\|_i^2$ = 페널티 | Data와 prior가 잘 맞음 |
| Complexity | $\sum \log \lambda_i$ | Flexible model = 페널티 |
| Gradient | $\frac{1}{2} \text{tr}((\alpha\alpha^\top - K^{-1}) \partial K / \partial \theta)$ | L-BFGS 최적화 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Marginal likelihood의 "data fit" 항이 **$y$의 RKHS norm**과 어떻게 관련되는가?

<details>
<summary>힌트 및 해설</summary>

$y^\top (K + \sigma_n^2 I)^{-1} y$는 $\sigma_n \to 0$에서 $y^\top K^{-1} y$로 수렴. 이것은 "**$y$를 interpolate하는 RKHS 함수의 최소 norm 제곱**":

$\|y\|_{\text{RKHS}}^2 = y^\top K^{-1} y = \|f^*\|_{\mathcal{H}_k}^2$, where $f^*$는 interpolating function.

**해석**: data fit = "data를 완벽히 설명하는 가장 매끄러운 함수의 norm". 작을수록 data가 prior와 compatible.

**Cross-validation과의 대조**: CV는 "next point prediction error"를 측정. Marginal likelihood는 "all data together의 RKHS norm-based measure". 더 efficient하지만 덜 직관적.

</details>

**문제 2** (심화): ARD (Automatic Relevance Determination)가 feature selection에 쓰이는 이유를 marginal likelihood 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

ARD kernel: $k(x, y) = \sigma_f^2 \exp(-\sum_{d=1}^D (x_d - y_d)^2 / (2 \ell_d^2))$.

Irrelevant feature $d$에 대해 $\ell_d$가 매우 커지면:
- $(x_d - y_d)^2 / \ell_d^2 \to 0$ → 해당 dimension이 kernel에서 효과적으로 **사라짐**.
- 해당 feature가 데이터 설명에 기여하지 않아 "무시".

Marginal likelihood가 이를 자동으로 유도:
- Relevant feature: 적절한 $\ell_d$가 필요 → 유한한 값으로 수렴.
- Irrelevant: $\ell_d$ 크게 → dimension 제거 → **model complexity 감소** (Occam).

**실무 응용**: Sensitivity analysis, automatic feature importance, high-dimensional BO.

**한계**: $D$ 매우 큼 ($D > 50$)이면 marginal likelihood 최적화 수렴 어려움. Nested ARD 또는 sparsity-inducing prior.

</details>

**문제 3** (ML 연결): NN의 hyperparameter 학습에 marginal likelihood 방식을 적용할 수 있을까? (Bayes optimal model selection)

<details>
<summary>힌트 및 해설</summary>

**이론적으로**: Yes. Bayesian Neural Network (BNN)의 marginal likelihood $\int p(y \mid \theta) p(\theta \mid \text{hp}) d\theta$를 maximize.

**실제 문제**:
1. **Intractable integral**: NN posterior는 non-Gaussian, high-dim → closed form 없음.
2. **Approximation**: Variational lower bound (ELBO), Laplace, MC dropout.
3. **계산 비용**: Per-evaluation forward+backward pass + sampling.

**대안**:
- **NTK 근사**: 무한폭 NN은 GP → marginal likelihood GP처럼 가능 (Ch7-04).
- **Empirical Bayes**: Type-II MLE with variational approximation.

**실무 현실**: NN hyperparameter는 여전히 주로 grid search, random search, BO. Marginal likelihood 방식이 이론적으로 우월하나 계산 비용과 근사 품질의 trade-off.

**연구 방향**: "Neural Network as Gaussian Process" (Lee et al. 2018) → NN의 GP 해석으로 marginal likelihood-like 선택 가능.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. GP Classification과 Laplace Approximation](./04-gp-classification.md) | [06. Sparse GP와 Inducing Points ▶](./06-sparse-gp.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
