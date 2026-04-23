# 03. GP ⇔ Kernel Ridge Regression 동치

## 🎯 핵심 질문

- 왜 GP posterior mean $m_* = k_*^\top (K + \sigma^2 I)^{-1} y$가 KRR의 해 $f_{\text{KRR}} = k^\top (K + \lambda I)^{-1} y$ **와 값 단위로 정확히 일치**하는가 ($\lambda = \sigma^2$)?
- 두 방법은 **같은 계산을 다른 해석**으로 푸는 것 — **Bayesian prior vs regularized risk** 관점은 어떻게 같은 수식으로 수렴하는가?
- 차이는 무엇인가 — GP가 **추가로** 제공하는 posterior variance의 의미는?
- 이 동치성이 Deep Learning의 NTK 이론(Ch7-04)과 어떻게 연결되는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

GP와 KRR이 "같은 예측을 내놓는다"는 사실은 **두 이론적 전통을 통합**한다. KRR은 **statistical learning theory** 시각 (Tikhonov regularization, VC bounds), GP는 **Bayesian inference** 시각 (prior/posterior, marginal likelihood). 동치성 덕분에:
1. **KRR에 uncertainty를 무료로 얻음** — GP posterior variance가 그대로 활용 가능.
2. **Hyperparameter $\lambda$의 Bayesian 해석** — marginal likelihood 최대화(Ch4-05)로 자동 학습.
3. **Regularization의 prior 해석** — $\lambda \|f\|^2$ 페널티 = Gaussian prior on $f$.
4. **NTK 이론** — 무한폭 NN의 gradient flow가 이 동치성을 무한차원으로 확장.

---

## 📐 수학적 선행 조건

- [Ch2-03 Representer 정리](../ch2-rkhs-representer/03-representer-theorem.md)
- [Ch4-02 GP posterior](./02-gp-posterior.md): $m_* = k_*^\top (K + \sigma^2 I)^{-1} y$
- [Ch5-01 Kernel Ridge Regression](../ch5-krr-kpca/01-kernel-ridge-regression.md) (미리 preview): $\alpha = (K + \lambda I)^{-1} y$
- 기본 Bayesian 추론

---

## 📖 직관적 이해

### "같은 식, 두 얼굴"

**KRR 관점** (frequentist, regularized risk):

$$\min_{f \in \mathcal{H}_k} \sum_i (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}_k}^2.$$

Representer: $f = \sum \alpha_i k_{x_i}$, $\alpha = (K + \lambda I)^{-1} y$. 예측: $f(x_*) = k_*^\top (K + \lambda I)^{-1} y$.

**GP 관점** (Bayesian):

Prior $f \sim \mathcal{GP}(0, k)$, Likelihood $y = f(x) + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$. Posterior mean $m_* = k_*^\top (K + \sigma^2 I)^{-1} y$.

**$\lambda = \sigma^2$로 두면 두 식이 정확히 일치**. 차이는 해석의 framework와 "추가로 GP가 제공하는 posterior variance".

### MAP ⟺ Regularized Risk

GP의 posterior가 Gaussian이므로 **MAP (Maximum A Posteriori)** = mean. MAP를 직접 푸는 것이 "negative log posterior minimize":

$$-\log p(f \mid y) \propto \underbrace{\frac{1}{2 \sigma^2} \|y - f(x)\|^2}_{\text{likelihood}} + \underbrace{\frac{1}{2} \|f\|_{\mathcal{H}_k}^2}_{\text{Gaussian prior}}.$$

$2\sigma^2$를 곱하면 $\|y - f\|^2 + \sigma^2 \|f\|^2$ — KRR의 objective with $\lambda = \sigma^2$.

**결론**: Gaussian prior with covariance $k$ = L2 penalty with RKHS norm. Gaussian likelihood = squared loss. 두 frameworks의 통합.

### 차이는 Uncertainty

**GP만의 추가 출력**: posterior variance $\sigma_*^2 = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*$.

이것은 **KRR 계산에 전혀 추가 연산 없이** 얻을 수 있음 — 이미 $(K + \lambda I)^{-1}$을 계산했으므로.

따라서 실무적으로 "KRR 하고 있는데 uncertainty 필요" = "GP 해석으로 바꾸고 variance 출력" — 같은 계산을 재활용.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — KRR 문제

$$\min_{f \in \mathcal{H}_k} \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}_k}^2.$$

### 정의 3.2 — GP Regression 문제

Prior $f \sim \mathcal{GP}(0, k)$, $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. Posterior mean $m_*$, variance $\sigma_*^2$ 계산.

---

## 🔬 정리와 증명

### 정리 3.1 — KRR과 GP Posterior Mean의 동치

**명제**: 정의 3.1의 KRR 해 $f_{\text{KRR}}$과 정의 3.2의 GP posterior mean $m_*$는 **$\lambda = \sigma^2$일 때 모든 $x_*$에서 정확히 일치**:

$$f_{\text{KRR}}(x_*) = m_*(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y.$$

**증명**:

**KRR side**: Representer 정리로 $f_{\text{KRR}} = \sum_i \alpha_i k(\cdot, x_i)$. Empirical risk:

$$J(\alpha) = \|y - K\alpha\|^2 + \lambda \alpha^\top K \alpha.$$

$\nabla_\alpha J = -2 K (y - K\alpha) + 2 \lambda K \alpha = 0 \Rightarrow K(y - K \alpha) = \lambda K \alpha \Rightarrow K y = (K^2 + \lambda K) \alpha = K(K + \lambda I) \alpha$.

$K$가 가역이면 $\alpha = (K + \lambda I)^{-1} y$. 예측: $f_{\text{KRR}}(x_*) = k_*^\top \alpha = k_*^\top (K + \lambda I)^{-1} y$.

**GP side**: Ch4-02 정리 2.2에서 $m_* = k_*^\top (K + \sigma^2 I)^{-1} y$.

**일치**: $\lambda = \sigma^2$이면 두 식 동일. $\square$

### 정리 3.2 — MAP과 KRR의 동치 (Finite-dim 유도)

**명제**: GP posterior의 MAP (= posterior mean, Gaussian이므로)는 다음 minimization과 동치:

$$\arg\min_f -\log p(y \mid f) - \log p(f) = \arg\min_f \frac{1}{2\sigma^2} \|y - f(x)\|^2 + \frac{1}{2} \|f\|_{\mathcal{H}_k}^2.$$

이것은 $\lambda = \sigma^2$인 KRR.

**증명**:

$-\log p(y \mid f) = \frac{1}{2\sigma^2} \sum_i (y_i - f(x_i))^2 + \text{const}$.

$-\log p(f) = \frac{1}{2} \|f\|_{\mathcal{H}_k}^2 + \text{const}$ (GP의 negative log prior).

둘 다 최소화: $\frac{1}{2 \sigma^2} \|y - f(x)\|^2 + \frac{1}{2} \|f\|_{\mathcal{H}_k}^2$. $2\sigma^2$ 곱하면 KRR. $\square$

**주의**: GP의 "prior on $f$"에서 $-\log p(f) = \frac{1}{2} \|f\|_{\mathcal{H}_k}^2$는 **finite-dim marginal 관점**에서만 정확한 해석. Sample path는 $\mathcal{H}_k$에 속하지 않지만, **finite 차원 projection은 Gaussian with covariance $K$**.

### 정리 3.3 — KRR에 없는 GP의 정보

**명제**: GP와 KRR은 posterior mean만 일치. GP는 추가로:

1. **Posterior variance** $\sigma_*^2$.
2. **Marginal likelihood** $\log p(y \mid \theta)$ (hyperparameter 학습, Ch4-05).
3. **Posterior samples** (함수 샘플링).
4. **Predictive distribution** $y_* \mid y \sim \mathcal{N}(m_*, \sigma_*^2 + \sigma^2)$.

**증명**: GP의 Bayesian framework에서 직접. KRR은 point estimate만. $\square$

### 정리 3.4 — Ridge Regression (Linear Kernel)로의 축퇴

**명제**: $k(x, y) = x^\top y$ (linear kernel)일 때 GP ⇔ KRR ⇔ **Ridge Regression**.

Prior $f(x) = w^\top x$, $w \sim \mathcal{N}(0, I_d)$, likelihood $y = w^\top x + \epsilon$.

Posterior:
$$w \mid y \sim \mathcal{N}((X^\top X + \sigma^2 I)^{-1} X^\top y, \sigma^2 (X^\top X + \sigma^2 I)^{-1}).$$

$f(x_*) = x_*^\top w$의 posterior mean: $x_*^\top (X^\top X + \sigma^2 I)^{-1} X^\top y$.

**검증**: KRR form $k_*^\top (K + \lambda I)^{-1} y = X x_*^\top (X X^\top + \lambda I)^{-1} y$. Woodbury identity로 $X x_*^\top (X X^\top + \lambda I)^{-1} = x_*^\top (X^\top X + \lambda I)^{-1} X^\top$. 일치.

**결론**: Ridge regression의 **두 form (primal $w$ 기반, dual $\alpha$ 기반)**이 이 관점에서 연결된다.

### 정리 3.5 — $\lambda = \sigma^2$ 해석

**명제**: Hyperparameter $\lambda$ (KRR)와 noise variance $\sigma^2$ (GP)의 일치는 **임의 choice가 아니라 본질적 일치**이다.

**증명 (통찰)**:
- $\lambda$ 크면 → 큰 regularization → 매끄러운 해 → "데이터를 덜 신뢰".
- $\sigma^2$ 크면 → 큰 noise → likelihood 약함 → prior 지배 → 매끄러운 posterior mean → "데이터를 덜 신뢰".

둘 다 "데이터의 신뢰도"를 제어. Bayesian은 이것을 **noise variance**로, frequentist는 **regularization strength**로 해석. 같은 parameter. $\square$

**실무 함의**: KRR을 쓸 때 $\lambda$를 "관측 noise level의 제곱"으로 추정 가능. GP는 marginal likelihood로 $\sigma^2$를 자동 학습.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 데이터
n = 25
X_train = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
y_train = np.sin(X_train).flatten() + 0.2 * rng.standard_normal(n)
X_test = np.linspace(-4, 4, 300).reshape(-1, 1)

def rbf(X, Y, sigma_f=1.0, ell=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return sigma_f**2 * np.exp(-d2 / (2 * ell**2))

sigma_f, ell = 1.0, 1.0
K = rbf(X_train, X_train, sigma_f, ell)
K_s = rbf(X_train, X_test, sigma_f, ell)
K_ss = rbf(X_test, X_test, sigma_f, ell)

# ─────────────────────────────────────────────
# 1. KRR with λ
# ─────────────────────────────────────────────
lam_krr = 0.04  # = σ_n² 가정
alpha_krr = np.linalg.solve(K + lam_krr * np.eye(n), y_train)
pred_krr = K_s.T @ alpha_krr

# ─────────────────────────────────────────────
# 2. GP with σ_n² = λ_krr
# ─────────────────────────────────────────────
sigma_n = np.sqrt(lam_krr)
L = np.linalg.cholesky(K + sigma_n**2 * np.eye(n))
alpha_gp = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
pred_gp_mean = K_s.T @ alpha_gp
v = np.linalg.solve(L, K_s)
pred_gp_var = np.diag(K_ss) - np.sum(v**2, axis=0)
pred_gp_std = np.sqrt(np.maximum(0, pred_gp_var))

# ─────────────────────────────────────────────
# 3. 수치적 일치 확인
# ─────────────────────────────────────────────
max_diff = np.max(np.abs(pred_krr - pred_gp_mean))
print(f'KRR vs GP posterior mean 최대 차이: {max_diff:.4e}')
print(f'  (기계 정밀도 수준 → 정확히 같음)')

# alpha도 같은지
alpha_diff = np.max(np.abs(alpha_krr - alpha_gp))
print(f'α 최대 차이: {alpha_diff:.4e}')

# ─────────────────────────────────────────────
# 4. 시각화 — KRR 예측 + GP uncertainty
# ─────────────────────────────────────────────
plt.figure(figsize=(11, 5))
plt.fill_between(X_test.flatten(), pred_gp_mean - 2*pred_gp_std, 
                  pred_gp_mean + 2*pred_gp_std, color='lightblue', alpha=0.5, 
                  label='GP 95% CI (KRR에 무료 추가)')
plt.plot(X_test, pred_krr, 'b-', lw=2, label='KRR prediction')
plt.plot(X_test, pred_gp_mean, 'r--', lw=1, label='GP posterior mean (동일)')
plt.scatter(X_train, y_train, c='red', s=30, zorder=5, label='Training')
plt.title(f'GP = KRR (λ=σ_n²={lam_krr}): 같은 예측, GP는 uncertainty 추가')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 5. Ridge regression(linear kernel)로의 축퇴
# ─────────────────────────────────────────────
X_lin = X_train  # d=1, so X = scalar column
# Ridge: w = (X^T X + λ I)^{-1} X^T y
w_ridge = np.linalg.solve(X_lin.T @ X_lin + lam_krr, X_lin.T @ y_train)
pred_ridge = X_test @ w_ridge

# KRR with linear kernel
K_lin = X_train @ X_train.T
K_s_lin = X_train @ X_test.T
alpha_krr_lin = np.linalg.solve(K_lin + lam_krr * np.eye(n), y_train)
pred_krr_lin = K_s_lin.T @ alpha_krr_lin

print(f'\nLinear kernel KRR vs Ridge 최대 차이: {np.max(np.abs(pred_ridge.flatten() - pred_krr_lin)):.4e}')
```

**출력 예시**:
```
KRR vs GP posterior mean 최대 차이: 2.22e-16
  (기계 정밀도 수준 → 정확히 같음)
α 최대 차이: 2.22e-16

Linear kernel KRR vs Ridge 최대 차이: 4.44e-15
```

→ 완전한 수치 일치. 기계 정밀도 수준. Linear kernel에서는 Ridge와도 일치.

---

## 🔗 실전 활용

- **KRR + Uncertainty**: KRR 계산 후 GP 해석으로 variance 추가 — 원래 KRR 사용자에게 "무료 uncertainty".
- **Regularization $\lambda$ 튜닝**: 
  - KRR: Cross-validation.
  - GP: Marginal likelihood 최대화 (Ch4-05) — **자동, data-efficient**.
- **Probabilistic Classification**: KRR로 학습 후 posterior mean + Platt scaling → calibrated 확률. GP classification (Ch4-04)은 더 엄밀하지만 approximation.
- **Surrogate modeling**: KRR 기반 surrogate에 uncertainty 추가 → BO acquisition 사용 가능.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| 동치는 **mean만** | Variance·marginal likelihood는 GP만 |
| Gaussian likelihood 가정 | Non-Gaussian이면 KRR ≠ GP |
| Squared loss | Hinge·logistic에서는 다른 correspondence (e.g., kernel logistic ↔ GP classification) |
| $\lambda = \sigma^2$ 해석 | 실무에서 CV로 튜닝한 $\lambda$가 정확히 noise²이 아닐 수 있음 |
| **Sample path ≠ RKHS 원소** | GP sample은 $\mathcal{H}_k$에 거의 확실히 속하지 않음 (Driscoll 1973) |

---

## 📌 핵심 정리

$$\boxed{f_{\text{KRR}}(x_*) = m_{\text{GP}}(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y \quad (\lambda = \sigma^2)}$$

$$\boxed{\text{GP의 추가 정보: posterior variance} = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*}$$

| 관점 | KRR | GP |
|------|-----|-----|
| Framework | Regularized risk | Bayesian |
| Objective | $\sum \ell(y, f(x)) + \lambda \|f\|^2$ | MAP = posterior mode |
| Output | Point prediction | Predictive distribution |
| Hyperparameter | $\lambda$: CV | $\sigma^2$: marginal likelihood |
| Computation | $(K + \lambda I)^{-1} y$ | 동일 + variance |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GP posterior mean을 "L2 regularized interpolation"으로 해석하는 방법은?

<details>
<summary>힌트 및 해설</summary>

$m_* = \arg\min_f \{\|y - f(x)\|^2 + \sigma^2 \|f\|_{\mathcal{H}_k}^2\}$ — 정리 3.2.

**해석**: Training 점에서 "$y$에 가깝게" + "RKHS norm 작게 (매끄럽게)".

$\sigma = 0$: interpolating solution with minimal RKHS norm — "정확히 통과하는 가장 매끄러운 함수".
$\sigma > 0$: Trade-off.
$\sigma \to \infty$: 거의 prior mean (0) — 데이터 무시.

이 해석이 **"GP prior + Gaussian likelihood = RKHS regularization"** 이라는 통일의 핵심 관점.

</details>

**문제 2** (심화): GP가 **KRR에 없는 marginal likelihood 최대화** (Ch4-05)를 쓸 수 있는 이유를 수학적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

GP의 marginal likelihood:
$$p(y \mid X, \theta) = \int p(y \mid f) p(f \mid \theta) df = \mathcal{N}(y; 0, K_\theta + \sigma^2 I)$$

**Bayesian의 핵심**: "모든 $f$를 marginalize" → $\theta$만 남은 확률. 이것을 **model selection**에 씀.

KRR은 **frequentist point estimate** — $f$를 marginalize한다는 개념이 없음. $\lambda$ 선택은 CV로만 가능.

**수학적 차이**:
- KRR: $\min_f L(f) + \lambda R(f)$. $f^*$는 $\lambda$에 의존. $\lambda$ 최적화는 meta-level.
- GP: $p(y \mid \theta) = \int p(y \mid f) p(f \mid \theta) df$. $\int$로 자동 marginalize.

이 차이가 GP의 **"hyperparameter 자동 학습"** 능력의 근원.

</details>

**문제 3** (ML 연결): NTK 이론(Ch7-04)과의 연결 — 무한폭 NN이 왜 "kernel method"의 일종인가?

<parameter name="description" />
<details>
<summary>힌트 및 해설</summary>

Jacot et al. (2018): 무한폭 NN $f_\theta$에서 gradient flow $\dot{\theta} = -\nabla_\theta L$는

$$\dot{f}(x) = -\sum_i (f(x_i) - y_i) \Theta(x, x_i)$$

여기서 NTK $\Theta(x, y) = \lim_{\text{width} \to \infty} \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle$.

이 ODE의 해 (MSE loss, gradient flow, $t \to \infty$):

$$f_\infty(x_*) = \Theta_*^\top \Theta^{-1} y$$

이것은 **NTK kernel의 kernel regression** (KRR without $\lambda$ limit, or $\sigma^2 \to 0$ GP).

**함의**:
- 무한폭 NN의 gradient descent = KRR (with NTK kernel).
- NN의 "수렴 함수" = GP posterior mean with NTK prior.
- NN의 **early stopping** = KRR with finite $\lambda$ (implicit regularization).

이 연결로 "NN의 generalization"을 kernel theory 도구 (Rademacher complexity, eigenfunction analysis)로 분석 가능. Generalization Theory의 핵심 연결 (Layer 2).

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. GP Regression — Posterior 유도](./02-gp-posterior.md) | [04. GP Classification과 Laplace Approximation ▶](./04-gp-classification.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
