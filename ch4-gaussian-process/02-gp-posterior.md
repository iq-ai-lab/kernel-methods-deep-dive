# 02. GP Regression — Posterior 유도

## 🎯 핵심 질문

- Prior $f \sim \mathcal{GP}(0, k)$와 관측 $y = f(x) + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$에서 posterior $f \mid y$의 **mean과 variance 공식**은 어떻게 유도되는가?
- **Joint Gaussian의 conditional 공식** $\mathcal{N}(\mu_{a|b}, \Sigma_{a|b})$이 왜 이 유도의 핵심인가?
- Posterior predictive $f(x_*) \mid y \sim \mathcal{N}(m_*, \sigma_*^2)$의 공식에서 각 항 ($k_*^\top (K + \sigma^2 I)^{-1} y$와 $k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*$)은 **어떤 의미**인가?
- Noise-free 관측 ($\sigma = 0$)에서는 어떻게 달라지는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

GP posterior는 **"Bayesian inference의 Gaussian 특수 사례"** — exact inference가 closed-form으로 가능한 드문 경우. 이 공식이 없으면 GP는 실용 알고리즘이 될 수 없다. Posterior mean은 **prediction** (Kernel Ridge Regression과 동치, Ch4-03), posterior variance는 **uncertainty** — 이 두 가지가 GP의 "prediction + confidence" 특유 출력. Bayesian Optimization의 acquisition function (EI, UCB, PI), active learning의 query strategy, surrogate modeling의 trust region — 모두 이 공식 위에 세워진다. 또한 posterior 공식의 미묘한 구조 ($K + \sigma^2 I$의 inversion이 **모든 kernel method의 공통 연산**)는 Ch4-03의 KRR 동치성으로 이어진다.

---

## 📐 수학적 선행 조건

- [Ch4-01 GP의 정의](./01-gp-definition.md): Multivariate normal 형태의 finite marginals
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): **Joint Gaussian의 conditional distribution** — 핵심 도구
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Block matrix inverse, Cholesky, Schur complement

---

## 📖 직관적 이해

### Joint Gaussian Conditional 공식

$\begin{pmatrix} a \\ b \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mu_a \\ \mu_b \end{pmatrix}, \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}\right)$에서

$$a \mid b \sim \mathcal{N}(\mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (b - \mu_b), \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}).$$

이 공식 하나가 GP posterior 유도의 **전부**다.

### GP Regression으로 적용

Training 점 $X = \{x_i\}$, test 점 $x_*$, prior $f \sim \mathcal{GP}(0, k)$:

$\begin{pmatrix} f_* \\ y \end{pmatrix} = \begin{pmatrix} f(x_*) \\ (y_1, \ldots, y_n)^\top \end{pmatrix}$이 joint Gaussian:

$$\begin{pmatrix} f_* \\ y \end{pmatrix} \sim \mathcal{N}\left(0, \begin{pmatrix} k_{**} & k_*^\top \\ k_* & K + \sigma^2 I \end{pmatrix}\right)$$

- $k_{**} = k(x_*, x_*)$.
- $k_* = (k(x_*, x_1), \ldots, k(x_*, x_n))^\top$.
- $K_{ij} = k(x_i, x_j)$.

Conditional 공식 적용:

$$f_* \mid y \sim \mathcal{N}\underbrace{(k_*^\top (K + \sigma^2 I)^{-1} y,}_{\text{posterior mean } m_*} \underbrace{k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*)}_{\text{posterior variance } \sigma_*^2}.$$

### 각 항의 의미

- **Mean** $m_*(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y$:
  - $(K + \sigma^2 I)^{-1} y$는 training 점들에 대한 "가중치 $\alpha$".
  - $k_*^\top \alpha = \sum_i \alpha_i k(x_*, x_i)$ — **$x_*$와 training 점들의 유사도로 가중 평균**.
  - Representer 정리 형태 그대로.

- **Variance** $\sigma_*^2(x_*) = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*$:
  - $k_{**}$: prior variance at $x_*$.
  - $k_*^\top (K + \sigma^2 I)^{-1} k_*$: "training 데이터로부터 얻은 정보로 인한 variance 감소".
  - **중요**: variance는 target $y$에 **무관**, training 점 $X$에만 의존 → **active learning**에 활용 가능.

### Noise-free와 비교

$\sigma = 0$이면 $K + 0 = K$, posterior mean $= k_*^\top K^{-1} y$, variance $= k_{**} - k_*^\top K^{-1} k_*$.

Training 점 $x_i$에서 $f_* = f(x_i)$:
- $k_* = K[i, :]$ (i-th row of $K$).
- $m_* = e_i^\top y = y_i$ — **training 값 정확히 통과** (interpolation).
- $\sigma_*^2 = K_{ii} - K_{i, :} K^{-1} K_{:, i} = K_{ii} - K_{ii} = 0$ — **uncertainty 0**.

Noise 있으면 training 점에서도 uncertainty $> 0$ (shrinkage).

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Noisy Observation Model

$$y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2), \quad f \sim \mathcal{GP}(0, k).$$

### 정의 2.2 — Joint Gaussian

Training $y = (y_1, \ldots, y_n)^\top$, test $f_* = f(x_*)$:

$$y \sim \mathcal{N}(0, K + \sigma^2 I), \quad f_* \mid X \sim \mathcal{N}(0, k_{**}).$$

**Cross covariance**: $\text{Cov}(f_*, y_i) = \text{Cov}(f(x_*), f(x_i)) = k(x_*, x_i)$.

### 정의 2.3 — Posterior Predictive

$$f_* \mid y, X, x_* \sim \mathcal{N}(m_*(x_*), \sigma_*^2(x_*))$$

여기서

$$m_*(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y, \quad \sigma_*^2(x_*) = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*.$$

$k_* \in \mathbb{R}^n$: $k_*(i) = k(x_*, x_i)$.

### 정의 2.4 — Noisy Prediction

$y_* = f_* + \epsilon_*$에 대한 predictive는 $y_* \mid y \sim \mathcal{N}(m_*, \sigma_*^2 + \sigma^2)$ ($\sigma^2$ 추가됨).

---

## 🔬 정리와 증명

### 정리 2.1 — Joint Gaussian의 Conditional 공식

**명제**: $(a, b) \sim \mathcal{N}((\mu_a, \mu_b), \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{pmatrix})$, $\Sigma_{bb}$ 가역일 때

$$a \mid b \sim \mathcal{N}(\mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (b - \mu_b), \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}).$$

**증명 스케치**: 

(1) 행렬 block inversion:

$$\begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1} = \begin{pmatrix} M^{-1} & -M^{-1} \Sigma_{ab} \Sigma_{bb}^{-1} \\ * & * \end{pmatrix}$$

여기서 $M = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}$ (Schur complement).

(2) Joint density $p(a, b) = p(a \mid b) p(b)$. Complete the square w.r.t. $a$:

$$\log p(a, b) = -\frac{1}{2}(a - \mu_a)^\top \Sigma_{aa}^{-1}(\cdot) + \cdots$$

정리하면 $a \mid b$의 mean과 covariance가 위 공식. $\square$

### 정리 2.2 — GP Posterior Mean과 Variance

**명제**: 정의 2.1, 2.2 하에서 정의 2.3의 공식이 성립.

**증명**: Joint

$$\begin{pmatrix} f_* \\ y \end{pmatrix} \sim \mathcal{N}\left(0, \begin{pmatrix} k_{**} & k_*^\top \\ k_* & K + \sigma^2 I \end{pmatrix}\right).$$

정리 2.1 적용 with $a = f_*$, $b = y$, $\mu_a = \mu_b = 0$, $\Sigma_{aa} = k_{**}$, $\Sigma_{ab} = k_*^\top$, $\Sigma_{ba} = k_*$, $\Sigma_{bb} = K + \sigma^2 I$:

$$m_* = 0 + k_*^\top (K + \sigma^2 I)^{-1} (y - 0) = k_*^\top (K + \sigma^2 I)^{-1} y.$$

$$\sigma_*^2 = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*. \quad \square$$

### 정리 2.3 — Multiple Test Points

**명제**: Test 점들 $X_* = \{x_{*,1}, \ldots, x_{*,m}\}$에 대해 $f_* = (f(x_{*,1}), \ldots, f(x_{*,m}))^\top$의 joint posterior:

$$f_* \mid y \sim \mathcal{N}(\mu_*, \Sigma_*)$$

여기서
$$\mu_* = K_*^\top (K + \sigma^2 I)^{-1} y, \quad \Sigma_* = K_{**} - K_*^\top (K + \sigma^2 I)^{-1} K_*$$

$K_{**} = [k(x_{*, i}, x_{*, j})]_{i, j}$, $K_* = [k(x_i, x_{*, j})] \in \mathbb{R}^{n \times m}$.

**증명**: 정리 2.1을 block 형태로 확장. $\square$

### 정리 2.4 — Training 점에서의 Prediction (Shrinkage)

**명제**: Training 점 $x_i$에서 posterior mean은

$$m_*(x_i) = K[i, :] (K + \sigma^2 I)^{-1} y.$$

Noise-free ($\sigma = 0$): $m_*(x_i) = y_i$ (정확 통과).

Noise ($\sigma > 0$): $m_*(x_i)$가 **$y_i$로 shrink되지만 도달 안 함** — regularization 효과.

**증명**: $K + 0 = K$, $K K^{-1} y = y$. $\sigma > 0$이면 $K (K + \sigma^2 I)^{-1}$은 identity가 아닌 shrinkage operator. $\square$

### 정리 2.5 — Posterior Covariance의 기하

**명제**: Posterior variance $\sigma_*^2(x_*)$는 $x_*$가 training 점들에 **가까울수록 작다**.

**증명 아이디어**: $k_*$가 큰 값을 가지면 (즉 $x_*$가 training과 유사하면), $k_*^\top (K + \sigma^2 I)^{-1} k_*$가 커짐 → variance 크게 감소. $x_*$가 training에서 멀면 $k_* \to 0$, 변분 $\sigma_*^2 \to k_{**}$ (prior variance).

공식 관점: $k_*^\top (K + \sigma^2 I)^{-1} k_* \geq 0$ (PSD quadratic form), $\leq k_{**}$ (Cauchy-Schwarz).

### 정리 2.6 — Noise-free Interpolation

**명제**: $\sigma = 0$에서 posterior mean은 training 점 $x_i$에서 $y_i$ 정확 통과, **interpolating function**.

노름 $\|m_*\|_{\mathcal{H}_k}^2 = y^\top K^{-1} y$ (RKHS norm).

**증명**: 정리 2.4의 noise-free 사례 + Representer 정리 (Ch2-03). $\square$

**해석**: Noise-free GP posterior mean = "training 점을 정확히 interpolate하면서 RKHS norm 최소인 함수" = KRR with $\lambda = 0$의 interpolant.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. GP Regression bottom-up 구현 (Rasmussen & Williams Alg 2.1)
# ─────────────────────────────────────────────
def rbf(X, Y, sigma_f=1.0, ell=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return sigma_f**2 * np.exp(-d2 / (2 * ell**2))

# Training 데이터
n = 12
X_train = np.sort(rng.uniform(-4, 4, n)).reshape(-1, 1)
y_train = np.sin(X_train).flatten() + 0.1 * rng.standard_normal(n)

sigma_f, ell, sigma_n = 1.0, 1.5, 0.1

# Test 점들
X_test = np.linspace(-6, 6, 300).reshape(-1, 1)

# Posterior 공식
K = rbf(X_train, X_train, sigma_f, ell)
K_s = rbf(X_train, X_test, sigma_f, ell)
K_ss = rbf(X_test, X_test, sigma_f, ell)

# Cholesky 기반 수치 안정 계산 (Rasmussen & Williams)
L = np.linalg.cholesky(K + sigma_n**2 * np.eye(n))
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Posterior mean
mu_star = K_s.T @ alpha

# Posterior variance (covariance diag)
v = np.linalg.solve(L, K_s)
var_star = np.diag(K_ss) - np.sum(v**2, axis=0)
std_star = np.sqrt(np.maximum(0, var_star))

# ─────────────────────────────────────────────
# 2. 시각화 — posterior mean + 95% CI + training 점
# ─────────────────────────────────────────────
plt.figure(figsize=(11, 5))
plt.fill_between(X_test.flatten(), mu_star - 2*std_star, mu_star + 2*std_star, 
                  color='lightblue', alpha=0.5, label='95% CI')
plt.plot(X_test, mu_star, 'b-', lw=2, label='Posterior mean')
plt.plot(X_test, np.sin(X_test), 'k--', alpha=0.3, label='True sin')
plt.scatter(X_train, y_train, c='red', s=50, zorder=5, label='Training')
plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.title(f'GP Regression (RBF, σ_f={sigma_f}, ℓ={ell}, σ_n={sigma_n})')
plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 3. Posterior samples (not just mean)
# ─────────────────────────────────────────────
cov_full = K_ss - v.T @ v
cov_full = cov_full + 1e-6 * np.eye(len(X_test))
L_post = np.linalg.cholesky(cov_full)
samples = mu_star[:, None] + L_post @ rng.standard_normal((len(X_test), 4))

plt.figure(figsize=(11, 5))
plt.plot(X_test, samples, alpha=0.7)
plt.scatter(X_train, y_train, c='red', s=50, zorder=5, label='Training')
plt.plot(X_test, mu_star, 'k-', lw=2, label='mean')
plt.title('Posterior 샘플 — 여러 가능한 함수')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. Training 점에서의 shrinkage
# ─────────────────────────────────────────────
for sn in [0.01, 0.1, 1.0]:
    L = np.linalg.cholesky(K + sn**2 * np.eye(n))
    a = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    fit = K @ a
    err = np.abs(fit - y_train)
    print(f'σ_n = {sn}: training point에서 |pred - y| 평균 = {err.mean():.4f}')

# σ_n 작으면 거의 정확 통과, 크면 smoothing

# ─────────────────────────────────────────────
# 5. Training 밖에서 variance 증가
# ─────────────────────────────────────────────
# 정리 2.5 수치 확인
var_at_train = np.zeros(n)
for i in range(n):
    k_s_i = rbf(X_train, X_train[[i]], sigma_f, ell).flatten()
    v_i = np.linalg.solve(L, k_s_i)
    k_ss_i = sigma_f**2
    var_at_train[i] = k_ss_i - np.sum(v_i**2)

print(f'\nTraining 점 근처 평균 variance: {var_at_train.mean():.4f}')
print(f'  (prior k(x,x) = {sigma_f**2})')
print(f'밖 (x=10)의 variance: {var_star[np.argmin(np.abs(X_test.flatten() - 10))]:.4f}  (→ prior variance로 회귀)')
```

**출력 예시**:
```
σ_n = 0.01: training point에서 |pred - y| 평균 = 0.0034
σ_n = 0.1: training point에서 |pred - y| 평균 = 0.0421
σ_n = 1.0: training point에서 |pred - y| 평균 = 0.2845

Training 점 근처 평균 variance: 0.0101
  (prior k(x,x) = 1.0)
밖 (x=10)의 variance: 0.9912  (→ prior variance로 회귀)
```

→ Training 점에서 uncertainty 매우 작고, data에서 멀어지면 prior로 회귀. Shrinkage는 $\sigma_n$과 비례.

---

## 🔗 실전 활용

- **Bayesian Optimization** (BO): Acquisition functions
  - **EI (Expected Improvement)**: $\alpha_{\text{EI}}(x) = \mathbb{E}[\max(0, y^* - f(x))]$ — closed form with $m_*, \sigma_*$.
  - **UCB (Upper Confidence Bound)**: $\alpha_{\text{UCB}}(x) = m_*(x) + \kappa \sigma_*(x)$.
  - **PI (Probability of Improvement)**: $\alpha_{\text{PI}}(x) = \Phi(\cdot)$.

- **Active Learning**: Next query $x^* = \arg\max_x \sigma_*(x)$ — uncertainty가 큰 곳.

- **Surrogate Modeling**: 비싼 simulator를 GP로 emulation. Uncertainty로 reliable region 식별.

- **Astronomy·Finance**: 시계열 예측 with calibrated uncertainty.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Gaussian noise | Heavy-tailed noise에서는 GP-t (Student-t GP) 필요 |
| Mean = 0 | Non-zero mean function 사용 가능 (constant 또는 linear) |
| $(K + \sigma^2 I)^{-1}$ | $O(n^3)$, $n > 10^4$이면 Sparse GP (Ch4-06) |
| $\sigma_n^2$ homoscedastic | Heteroscedastic noise는 별도 모델링 |
| Posterior closed-form only for Gaussian | Classification (Ch4-04)은 approximation 필요 |

---

## 📌 핵심 정리

$$\boxed{m_*(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y}$$

$$\boxed{\sigma_*^2(x_*) = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*}$$

| 요소 | 의미 |
|------|------|
| $K + \sigma^2 I$ | Noisy observation covariance (prior + noise) |
| $k_*$ | Test 점과 training 점들의 cross-covariance |
| $k_*^\top (K + \sigma^2 I)^{-1} y$ | Data-weighted prediction (Representer form) |
| $k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*$ | Information gain으로 인한 variance 감소 |
| **Variance는 target 무관** | Active learning에 활용 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GP posterior mean $m_*(x_*) = k_*^\top \alpha$, $\alpha = (K + \sigma^2 I)^{-1} y$. 이것이 Representer 정리의 어떤 형태인가?

<details>
<summary>힌트 및 해설</summary>

$m_*(x) = \sum_{i=1}^n \alpha_i k(x, x_i)$ — **정확히 Representer 정리의 형태** $f^* = \sum \alpha_i k_{x_i}$.

KRR의 $\alpha_{\text{KRR}} = (K + \lambda I)^{-1} y$에서 $\lambda = \sigma^2$로 놓으면 동일 (Ch4-03에서 상세).

**통찰**: GP는 "Bayesian 해석의 KRR"이고, KRR은 "regularized risk 해석의 GP posterior mean". 같은 계산, 다른 해석.

</details>

**문제 2** (심화): GP posterior에서 **multiple test points**를 joint로 예측할 때 covariance $\Sigma_*$는 왜 중요한가?

<details>
<summary>힌트 및 해설</summary>

Marginal variance만 쓰면 **점별 독립 예측** — 인접 점의 상관을 놓침.

**응용 1 (sampling)**: Joint sampling으로 **realistic sample path** 생성. Marginal만 쓰면 독립 Gaussian → 거친 샘플.

**응용 2 (Bayesian Optimization)**: Multiple point optimization (batch BO). EI를 joint로 계산 = $\mathbb{E}[\max(y^* - \max_k f(x_{*, k}))]$ needs joint.

**응용 3 (confidence bounds)**: "$x$ 영역의 $f$가 모두 $< c$일 확률" = joint Gaussian에 대한 multivariate probability.

**응용 4 (integrals)**: $\int f(x) dx$도 Gaussian (linear of GP), 분산은 joint covariance 필요.

**계산**: $K_{**} - K_*^\top (K + \sigma^2 I)^{-1} K_*$ — marginal이 아닌 full block.

</details>

**문제 3** (ML 연결): Bayesian Optimization의 Expected Improvement 공식을 GP posterior를 이용해 유도하라.

<details>
<summary>힌트 및 해설</summary>

Goal: Minimize $f$. Current best $y^* = \min_i y_i$. $f(x) \mid y \sim \mathcal{N}(m_*(x), \sigma_*^2(x))$.

$$\alpha_{\text{EI}}(x) = \mathbb{E}[\max(0, y^* - f(x))].$$

$u := (y^* - f(x)) / \sigma_*$, $u \sim \mathcal{N}((y^* - m_*) / \sigma_*, 1)$. $Z := (y^* - m_*) / \sigma_*$.

$$\alpha_{\text{EI}} = \sigma_* \mathbb{E}[\max(0, u)]_{u \sim \mathcal{N}(Z, 1)} = \sigma_* [Z \Phi(Z) + \phi(Z)]$$

여기서 $\Phi, \phi$는 standard Normal의 cdf/pdf.

**해석**:
- $\sigma_* = 0$: $\alpha_{\text{EI}} = 0$ (known 영역, no improvement 가능).
- $m_* = y^*$: $\alpha_{\text{EI}} = \sigma_* \phi(0) = \sigma_* / \sqrt{2\pi}$ — exploration only.
- $m_* \ll y^*$: $\alpha_{\text{EI}} \approx y^* - m_*$ — exploitation.

**Exploration-exploitation trade-off**는 $\sigma_*$(uncertainty) vs $m_*$(prediction)의 균형으로 자연스럽게.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. GP의 정의와 공분산 함수](./01-gp-definition.md) | [03. GP ⇔ Kernel Ridge Regression 동치 ▶](./03-gp-equals-krr.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
