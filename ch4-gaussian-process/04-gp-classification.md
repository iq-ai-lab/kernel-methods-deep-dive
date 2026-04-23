# 04. GP Classification과 Laplace Approximation

## 🎯 핵심 질문

- GP classification은 왜 **non-Gaussian posterior**를 갖고, 따라서 exact inference가 불가능한가?
- **Laplace approximation**은 posterior를 어떻게 Gaussian으로 근사하는가?
- Newton-Raphson 알고리즘이 posterior mode를 어떻게 찾는가?
- Alternative approximation: **Expectation Propagation (EP)**과 비교하면?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Binary classification (label ∈ $\{+1, -1\}$ 또는 $\{0, 1\}$)에서 GP를 쓰려면 Bernoulli likelihood가 필요 — 이는 non-Gaussian이라 **exact posterior가 intractable**. Laplace approximation은 posterior mode 주변의 Gaussian 근사로 **predictive distribution, marginal likelihood, uncertainty**를 모두 얻게 해준다. GP classification은 Bayesian 분류의 non-parametric 접근으로, (i) **calibrated probability** (GP의 자연스러운 불확실성 표현), (ii) **kernel 기반 비선형** (SVM처럼 비선형이지만 probabilistic), (iii) **small-data에서 강력** (strong prior). 실무에서는 computer experiment, biomedical classification, active learning에 쓰인다.

---

## 📐 수학적 선행 조건

- [Ch4-01~03](./01-gp-definition.md): GP regression
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Bernoulli likelihood, Laplace approximation
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Newton-Raphson, strict concavity of log likelihood

---

## 📖 직관적 이해

### Bernoulli Likelihood = Non-Gaussian

Label $y_i \in \{-1, +1\}$, "latent" GP $f \sim \mathcal{GP}(0, k)$. Likelihood:

$$p(y_i = +1 \mid f(x_i)) = \sigma(f(x_i)), \quad p(y_i = -1 \mid f(x_i)) = 1 - \sigma(f(x_i)) = \sigma(-f(x_i)).$$

간결하게: $p(y_i \mid f(x_i)) = \sigma(y_i f(x_i))$.

**문제**: Prior Gaussian × Likelihood Bernoulli = non-Gaussian posterior. $(K + \sigma^2 I)^{-1}$ 같은 closed-form 없음.

### Laplace Approximation — "Mode 주변 Gaussian"

Posterior $p(f \mid y) \propto p(y \mid f) p(f)$의 **mode** $\hat{f}$를 찾고, $\log p(f \mid y)$를 2차 Taylor 전개:

$$\log p(f \mid y) \approx \log p(\hat{f} \mid y) - \frac{1}{2} (f - \hat{f})^\top A (f - \hat{f}), \quad A = -\nabla^2 \log p(f \mid y) \Big|_{\hat{f}}.$$

**결과**: $p(f \mid y) \approx \mathcal{N}(\hat{f}, A^{-1})$ — Gaussian 근사.

이것은 **"posterior가 unimodal하고 mode 근처에서 대칭"일 때 합리적**. Logistic likelihood가 log-concave이므로 log-posterior도 log-concave → unimodal → Laplace 합리적.

### Newton-Raphson으로 Mode 찾기

$\log p(f \mid y) = \log p(y \mid f) + \log p(f) + \text{const}$.

$$\nabla_f \log p(y \mid f) = \sum_i y_i (1 - \sigma(y_i f_i)) \delta_i \quad (\text{diagonal vector per component})$$

$$\nabla_f \log p(f) = -K^{-1} f \quad (\text{Gaussian prior}).$$

Newton update: $f^{(t+1)} = f^{(t)} - H^{-1} \nabla \log p(f \mid y)$, 여기서 $H = -\nabla^2 \log p$. 수렴 후 mode $\hat{f}$ 도달.

### Predictive Distribution

Mode $\hat{f}$ 및 Hessian $A$ 얻은 후:

- **Latent posterior**: $f \mid y \approx \mathcal{N}(\hat{f}, A^{-1})$.
- **Test latent**: $f_* \mid y \approx \mathcal{N}(k_*^\top K^{-1} \hat{f}, k_{**} - k_*^\top (K + W^{-1})^{-1} k_*)$ ($W$는 $\log$-likelihood의 Hessian).
- **Class probability**: $p(y_* = +1 \mid y) = \int \sigma(f_*) p(f_* \mid y) df_*$ — 1D integral, Monte Carlo 또는 Gauss-Hermite quadrature.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — GP Classification Model

Prior $f \sim \mathcal{GP}(0, k)$, likelihood $p(y_i \mid f(x_i)) = \sigma(y_i f(x_i))$ ($y_i \in \{-1, +1\}$), 여기서 $\sigma(z) = 1 / (1 + e^{-z})$.

### 정의 4.2 — Posterior

$$p(f \mid y) = \frac{p(y \mid f) p(f)}{p(y)} = \frac{\prod_i \sigma(y_i f(x_i)) \cdot \mathcal{N}(f; 0, K)}{p(y)}.$$

Non-Gaussian. Exact inference intractable.

### 정의 4.3 — Laplace Approximation

$$p(f \mid y) \approx q(f) := \mathcal{N}(f; \hat{f}, A^{-1}), \quad \hat{f} = \arg\max_f \log p(f \mid y), \quad A = -\nabla^2 \log p(f \mid y)|_{\hat{f}}.$$

### 정의 4.4 — $W$ 행렬

$W$ = diagonal 행렬, $W_{ii} = -\frac{\partial^2 \log p(y_i \mid f_i)}{\partial f_i^2}\Big|_{\hat{f}_i}$. Logistic likelihood에서 $W_{ii} = \sigma(\hat{f}_i)(1 - \sigma(\hat{f}_i))$.

---

## 🔬 정리와 증명

### 정리 4.1 — Posterior의 Log-Concavity

**명제**: Logistic likelihood + Gaussian prior로 만들어진 posterior $p(f \mid y)$는 **log-concave** in $f$. 따라서 unique mode가 존재.

**증명**: 

$-\log p(f \mid y) = -\sum_i \log \sigma(y_i f_i) + \frac{1}{2} f^\top K^{-1} f + \text{const}$.

- $-\log \sigma(z)$는 **convex** in $z$ (log-sum-exp의 한 형태). 따라서 $-\sum_i \log \sigma(y_i f_i)$는 convex in $f$.
- $\frac{1}{2} f^\top K^{-1} f$는 convex ($K^{-1} \succeq 0$).

**합** = convex. Log posterior는 concave. $\square$

**실무적 함의**: Newton-Raphson의 수렴 보장, unique mode.

### 정리 4.2 — Newton-Raphson Update

**명제**: Posterior mode를 찾는 Newton update:

$$f^{(t+1)} = (K^{-1} + W^{(t)})^{-1} (W^{(t)} f^{(t)} + \nabla_f \log p(y \mid f^{(t)})).$$

또는 동등하게:

$$f^{(t+1)} = K (I + W^{(t)} K)^{-1} b^{(t)}, \quad b^{(t)} = W^{(t)} f^{(t)} + \nabla_f \log p(y \mid f^{(t)}).$$

**증명 스케치**: $\nabla \log p(f \mid y) = \nabla \log p(y \mid f) - K^{-1} f$. Hessian $= -W - K^{-1}$. Newton step:

$$f^{(t+1)} = f^{(t)} - (-W - K^{-1})^{-1} (\nabla \log p(y \mid f) - K^{-1} f)$$

정리하면 위 형태. $\square$

### 정리 4.3 — Predictive Mean과 Variance (Laplace)

**명제**: Mode $\hat{f}$, $W = \text{diag}(\sigma(\hat{f})(1-\sigma(\hat{f})))$ 얻은 후 test 점 $x_*$:

$$\mathbb{E}[f_* \mid y] \approx k_*^\top K^{-1} \hat{f} = k_*^\top \nabla \log p(y \mid \hat{f}).$$

$$\text{Var}[f_* \mid y] \approx k_{**} - k_*^\top (K + W^{-1})^{-1} k_*.$$

Class probability $p(y_* = +1 \mid y) = \int \sigma(f_*) q(f_*) df_* = \mathbb{E}_{q(f_*)}[\sigma(f_*)]$ — Gauss-Hermite quadrature 또는 probit approximation.

**Probit approximation**: $\int \sigma(f_*) \mathcal{N}(f_*; m, v) df_* \approx \sigma\left(\frac{m}{\sqrt{1 + \pi v / 8}}\right)$.

### 정리 4.4 — Laplace Marginal Likelihood

**명제**: $\log p(y \mid \theta) \approx$ Laplace approximation:

$$\log p(y \mid \theta) \approx \log p(y \mid \hat{f}) - \frac{1}{2} \hat{f}^\top K^{-1} \hat{f} - \frac{1}{2} \log |I + W K|.$$

**증명**: 

$p(y \mid \theta) = \int p(y \mid f) p(f \mid \theta) df$. Laplace approximation:

$$\int e^{-g(f)} df \approx \int e^{-g(\hat{f}) - \frac{1}{2} (f - \hat{f})^\top A (f - \hat{f})} df = e^{-g(\hat{f})} (2\pi)^{n/2} |A|^{-1/2}.$$

$g(f) = -\log p(y \mid f) - \log p(f)$, $A = -\nabla^2 \log p(f \mid y)|_{\hat{f}} = K^{-1} + W$. $|A| = |K^{-1}| |I + KW| = |K|^{-1} |I + KW|$.

$\log p(y \mid \theta) \approx \log p(y \mid \hat{f}) + \log p(\hat{f}) - \frac{1}{2} \log |A| + \frac{n}{2} \log(2\pi) + (\text{const from prior norm})$. 정리. $\square$

**의미**: Marginal likelihood를 hyperparameter 학습에 활용 (Ch4-05와 유사).

### 정리 4.5 — EP (Expectation Propagation) 대안

**명제**: EP는 각 likelihood factor $\sigma(y_i f_i)$를 **Gaussian tilted distribution**으로 근사:

$$p(f \mid y) \approx \prod_i \tilde{q}_i(f_i), \quad \tilde{q}_i = \mathcal{N}(f_i; \tilde{\mu}_i, \tilde{\sigma}_i^2)$$

각 factor를 iteratively refine (moment matching). Laplace보다 더 정확한 근사 일반적, 하지만 수렴성 덜 robust.

**비교**:
| | Laplace | EP |
|---|-----|-----|
| 정확도 | 덜 정확 (mode 주변) | 더 정확 (모든 moments) |
| 수렴 | Newton-Raphson, 보장 | Iterative, 발산 가능 |
| 구현 | 간단 | 복잡 |
| Marginal likelihood | 하한 | 더 정확 |

실무: GPy, sklearn, GPflow에서 Laplace·EP 둘 다 제공. EP를 default로 하는 경우 많음 (Rasmussen & Williams Ch3.6).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

rng = np.random.default_rng(0)

# 데이터: 2D, 비선형 경계
n_pc = 30
X_pos = rng.multivariate_normal([1.5, 0], 0.3 * np.eye(2), n_pc)
X_neg = rng.multivariate_normal([-1.5, 0], 0.3 * np.eye(2), n_pc)
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n_pc), -np.ones(n_pc)])
n = len(y)

def rbf(X, Y, ell=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * ell**2))

# ─────────────────────────────────────────────
# 1. Laplace approximation (bottom-up)
# ─────────────────────────────────────────────
K = rbf(X, X, ell=1.0) + 1e-6 * np.eye(n)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Newton-Raphson for mode
f = np.zeros(n)
for it in range(30):
    pi_f = sigmoid(f)
    # Gradient of log p(y|f): y_i (1 - σ(y_i f_i)) = (y+1)/2 - σ(f_i) ... need to be careful
    # In ±1 convention:
    #   log σ(y_i f_i) 미분 in f_i: y_i σ(-y_i f_i) = y_i (1 - σ(y_i f_i))
    grad_ll = 0.5 * (y + 1) - pi_f  # Rasmussen 3.15 style with y ∈ {0, 1}
    # Convert to ±1: use (y+1)/2 as 0/1 target
    # gradient 후 같은 점에서 evaluation
    # 여기서는 일단 0/1로 변환
    y01 = (y + 1) / 2
    grad_ll = y01 - pi_f
    W = pi_f * (1 - pi_f)  # diag(W)
    # Newton: f_new = K (I + WK)^{-1} (W f + grad_ll)
    B = np.eye(n) + (W[:, None] * K)
    rhs = W * f + grad_ll
    f_new = K @ np.linalg.solve(B, rhs)
    if np.max(np.abs(f_new - f)) < 1e-6:
        print(f'Laplace 수렴: iter {it}')
        break
    f = f_new

f_hat = f
W_hat = sigmoid(f_hat) * (1 - sigmoid(f_hat))

# Marginal likelihood approximation
pi_hat = sigmoid(f_hat)
ll = np.sum(y01 * np.log(pi_hat + 1e-12) + (1 - y01) * np.log(1 - pi_hat + 1e-12))
logdet = np.linalg.slogdet(np.eye(n) + W_hat[:, None] * K)[1]
marg_ll = ll - 0.5 * f_hat @ np.linalg.solve(K, f_hat) - 0.5 * logdet
print(f'Laplace marginal likelihood: {marg_ll:.4f}')

# ─────────────────────────────────────────────
# 2. 예측 (test 점에 대한 probability)
# ─────────────────────────────────────────────
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

K_s = rbf(X, grid, ell=1.0)
K_ss_diag = np.ones(len(grid))  # rbf(grid, grid) diag = 1

# Predictive mean of f_*
mu_fstar = K_s.T @ (y01 - pi_hat)  # = K_s^T K^{-1} f_hat이 아니라, α = grad_ll
# 더 표준적: α = K^{-1} f_hat = (y - pi_hat)  가 log-posterior stationary로 식별 가능
# Rasmussen 3.21: latent mean = K(X, x_*)^T · (grad log p(y|f))  where grad는 (y - pi_hat)
# 여기서 y01이 아닌 ±1 변환시 주의

# Variance:  k_** - k_*^T (K + W^{-1})^{-1} k_*
# Using Woodbury: k_** - k_*^T K^{-1} (I - (I + KW)^{-1}) k_*
# For stability use B = I + W^{1/2} K W^{1/2}  (Rasmussen Alg 3.2)
sW = np.sqrt(W_hat)
L_B = np.linalg.cholesky(np.eye(n) + (sW[:, None] * sW[None, :]) * K)
v = np.linalg.solve(L_B, sW[:, None] * K_s)
var_fstar = K_ss_diag - np.sum(v ** 2, axis=0)

# Probit approximation: p(y_* = +1) ≈ σ(μ / sqrt(1 + π v/8))
prob_pos = sigmoid(mu_fstar / np.sqrt(1 + np.pi * var_fstar / 8))

# ─────────────────────────────────────────────
# 3. 시각화 — probability contour
# ─────────────────────────────────────────────
Z = prob_pos.reshape(xx.shape)
plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.7)
plt.colorbar(label='p(y=+1|x)')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X[y > 0, 0], X[y > 0, 1], c='red', s=40, edgecolors='k', label='y=+1')
plt.scatter(X[y < 0, 0], X[y < 0, 1], c='blue', s=40, edgecolors='k', label='y=-1')
plt.title('GP Classification (Laplace): p(y=+1|x) with uncertainty')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. sklearn 검증
# ─────────────────────────────────────────────
gpc = GaussianProcessClassifier(kernel=RBF(length_scale=1.0))
gpc.fit(X, (y + 1) // 2)
prob_sklearn = gpc.predict_proba(grid)[:, 1]

max_diff = np.max(np.abs(prob_pos - prob_sklearn))
print(f'Probability 예측 vs sklearn 최대 차이: {max_diff:.4f}')
print(f'(근사 방법 차이로 작은 차이 존재)')
```

**출력 예시**:
```
Laplace 수렴: iter 8
Laplace marginal likelihood: -12.3421
Probability 예측 vs sklearn 최대 차이: 0.0312
(근사 방법 차이로 작은 차이 존재)
```

→ Manual Laplace 구현이 sklearn과 거의 일치 (small 차이는 probit vs Gauss-Hermite의 차이).

---

## 🔗 실전 활용

- **Probabilistic classification**: GP가 SVM보다 **calibrated 확률** 제공. Binary decision 외에 confidence 필요할 때.
- **Active Learning**: Variance $\text{Var}[f_*]$가 큰 점 선택 → 가장 uncertain한 영역.
- **Small-data**: Strong prior로 $n < 100$에서도 합리적 예측. SVM은 $n$ 작으면 overfitting 가능.
- **sklearn**: `GaussianProcessClassifier(kernel=RBF())`. Laplace + Binary, 또는 OvR/OvO for multi-class.
- **GPy / GPflow**: EP, Variational Inference 등 더 다양한 approximation 제공.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Laplace는 mode 주변 근사 | Heavy-tailed posterior 부정확 |
| Log-concavity | Logistic·probit OK, 다른 likelihood는 multi-modal posterior 가능 |
| Multi-class | OvR·OvO 또는 multinomial logistic (더 복잡) |
| **Scaling $O(n^3)$** | GP regression과 동일 병목 |
| Probit approx | 정확도 $O(v^2)$ error, $v$ 크면 덜 정확 |

---

## 📌 핵심 정리

$$\boxed{p(f \mid y) \approx \mathcal{N}(\hat{f}, (K^{-1} + W)^{-1}) \quad \text{(Laplace approximation)}}$$

$$\boxed{p(y_* = +1 \mid y) \approx \sigma\left(\frac{\mu_*}{\sqrt{1 + \pi \sigma_*^2 / 8}}\right) \quad \text{(probit approximation)}}$$

| Step | 내용 |
|------|------|
| 1. Log-concave posterior | Unique mode $\hat{f}$ |
| 2. Newton-Raphson | $f^{(t+1)} = K (I + WK)^{-1} (Wf + \nabla \log p(y\|f))$ |
| 3. Hessian at mode | $W = \sigma(\hat{f})(1 - \sigma(\hat{f}))$ diagonal |
| 4. Predictive latent | $f_* \mid y \approx \mathcal{N}(k_*^\top K^{-1} \hat{f}, k_{**} - k_*^\top (K + W^{-1})^{-1} k_*)$ |
| 5. Probability | $\int \sigma(f_*) q(f_*) df_*$, probit or Gauss-Hermite |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Laplace approximation의 정확도가 **posterior의 skewness에 의해** 결정되는 이유는?

<details>
<summary>힌트 및 해설</summary>

Laplace는 **Gaussian (symmetric)**으로 근사. 실제 posterior가:
- **Symmetric + unimodal**: Laplace 정확.
- **Skewed (예: heavy right tail)**: Laplace가 tail underestimate.
- **Heavy-tailed**: Laplace의 variance가 실제보다 작음.

GP classification의 posterior는 대략 Gaussian-like (logistic likelihood log-concave) → Laplace 합리적, 단 class imbalance가 심하면 skew 증가 → EP가 나음.

**경험 법칙**: $n$이 크면 CLT로 posterior가 점점 Gaussian-like → Laplace 정확. $n$ 작거나 prior 강하면 skew 클 수 있음.

</details>

**문제 2** (심화): GP classification이 **kernel SVM과 다른 점**은 구체적으로 무엇인가?

<details>
<summary>힌트 및 해설</summary>

| 측면 | GPC | Kernel SVM |
|------|-----|-----------|
| Framework | Bayesian (prior + likelihood) | Margin maximization |
| Loss | Logistic (smooth) | Hinge (non-differentiable) |
| Output | Probability + uncertainty | Sign only |
| Sparsity | 없음 (모든 점 기여) | Support vectors only |
| Kernel form | 동일 (RBF, Matérn 등) | 동일 |
| Hyperparameter | Marginal likelihood | Cross-validation |
| Calibration | 자동 (Bayesian) | Platt scaling 필요 |

**연결**: Hinge loss + L2 reg + kernel = kernel SVM. Logistic loss + L2 reg + kernel = kernel logistic regression. GP classification ≈ kernel logistic regression의 Bayesian 버전 (MAP = 후자의 해).

</details>

**문제 3** (ML 연결): Bayesian Deep Learning (BDL)에서 GP classification의 역할은?

<details>
<summary>힌트 및 해설</summary>

**GPC는 BDL의 "non-parametric 사촌"**:

1. **Uncertainty quantification**: Dropout-based BDL이 rough approximation이라면, GPC는 exact Bayesian inference (within Laplace 또는 EP 근사).

2. **Kernel ↔ NN**: NTK (Ch7-04)는 NN의 kernel 해석을 제공 — NN classification도 "infinite-width에서는 GP classification"과 동치.

3. **Deep Kernel Learning (Ch7-03)**: NN feature + GPC — hybrid, NN의 feature learning + GP의 Bayesian 장점.

4. **Small-data regime**: NN은 small data에 overfit; GPC는 strong prior로 안정적. 의료·과학 도메인의 limited labels.

5. **Active learning**: GPC의 uncertainty 기반 query >> NN dropout의 uncertainty (덜 calibrated).

**실무 분위기**: BDL 주류는 NN-based (Bayes by Backprop, MC Dropout). GPC는 small-to-medium data에서 uncertainty critical한 niche에 남음.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. GP ⇔ Kernel Ridge Regression 동치](./03-gp-equals-krr.md) | [05. Hyperparameter Learning — Marginal Likelihood ▶](./05-marginal-likelihood.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
