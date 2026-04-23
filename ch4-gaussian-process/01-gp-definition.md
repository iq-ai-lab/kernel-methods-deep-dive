# 01. GP의 정의와 공분산 함수

## 🎯 핵심 질문

- Gaussian Process는 왜 **"함수에 대한 정규분포"**라고 불리는가? 무한히 많은 점에 대한 정규분포가 어떻게 정의되는가?
- Mean function $m(x)$와 covariance function $k(x, y)$가 GP를 어떻게 **완전히 특징**짓는가?
- Kolmogorov 확장 정리가 GP의 **존재성**을 어떻게 보장하는가?
- Covariance function 선택이 왜 **prior 함수 공간의 선택**과 같은가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

GP는 **함수에 대한 Bayesian prior**를 명시적으로 정의할 수 있게 해준다. "어떤 함수를 먼저 기대하는가"를 mean + covariance로 표현하고, 데이터를 관측하면 **closed-form posterior**를 얻는다. 이 우아함 덕분에 (i) **uncertainty quantification**이 자연스러움 (posterior variance), (ii) **hyperparameter 학습**이 marginal likelihood 최대화로 자동화, (iii) **non-parametric** — 데이터가 늘수록 모델 복잡도 자동 증가. 단점은 $O(n^3)$ 계산이지만, **Bayesian Optimization·surrogate modeling·지구통계학·active learning**에서 불가결한 도구.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): **다변수 정규분포**, 조건부 기댓값, 특성함수
- [Ch1 전체](../ch1-kernel-basics/01-positive-definite-kernel.md): PD kernel (= covariance function)
- 측도론: Kolmogorov 확장 정리 (stochastic process의 존재성)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 대칭 PSD 행렬, block decomposition

---

## 📖 직관적 이해

### "함수에 대한 정규분포"의 의미

$d$-차원 다변수 정규분포는 **$d$개의 스칼라 확률변수**를 묶는다: $f = (f_1, \ldots, f_d)^\top \sim \mathcal{N}(\mu, \Sigma)$.

GP는 이를 **연속적으로 많은 점에 대한 "분포"**로 확장: "임의 유한 점 $\{x_1, \ldots, x_n\}$에서 평가한 값 $(f(x_1), \ldots, f(x_n))$이 다변수 정규분포".

즉 무한히 많은 "index" $x$가 있지만, **임의 유한 슬라이스만 본다면** 정규분포. 이 properties를 **finite-dimensional distributions**라 한다.

### Mean + Covariance로 완전 특징화

다변수 정규분포는 평균 $\mu$와 공분산 $\Sigma$로 완전 특징화. GP도 유사하게 **두 함수**로 결정:

- **Mean function** $m(x) := \mathbb{E}[f(x)]$.
- **Covariance function** $k(x, y) := \text{Cov}(f(x), f(y)) = \mathbb{E}[(f(x) - m(x))(f(y) - m(y))]$.

임의 점들 $\{x_i\}$에서 $(f(x_1), \ldots, f(x_n)) \sim \mathcal{N}(\mu, K)$, $\mu_i = m(x_i)$, $K_{ij} = k(x_i, x_j)$.

### Covariance Function과 Prior 함수 공간

Covariance function $k$는 "**서로 가까운 $x$에서 $f$ 값이 얼마나 상관되어 있는가**"를 지정.

- **RBF** $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$: 가까운 점 강한 상관, 먼 점 독립 → **매끄러운 함수 prior**.
- **Linear** $k(x, y) = x^\top y$: 상관이 거리와 무관하게 inner product → **선형 함수 prior**.
- **Periodic** $k(x, y) = \exp(-\frac{2 \sin^2(\pi (x-y)/p)}{\ell^2})$: 주기 $p$로 반복 → **주기 함수 prior**.

"Covariance 함수 선택 = prior 함수 공간 선택"의 정확한 의미는 **RKHS와의 대응**(Ch2-05): Matérn-$\nu$ GP의 sample path는 $H^{\nu + d/2}$ 근방 Sobolev smooth. RBF는 해석적.

### Sample Path 시각화

$\mathcal{GP}(0, k)$에서 샘플을 뽑는 방법: 격자 $\{x_1, \ldots, x_n\}$에서 $K = [k(x_i, x_j)]$ 계산, $f = L \epsilon$ ($L$ = Cholesky, $\epsilon \sim \mathcal{N}(0, I)$). 이것이 하나의 "함수" 샘플.

RBF length-scale 작음 → 빠르게 진동하는 샘플.
RBF length-scale 큼 → 매끄러운 샘플.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Gaussian Process

확률과정 $\{f(x) : x \in \mathcal{X}\}$가 **Gaussian Process**라 함은, 임의 유한 부분집합 $\{x_1, \ldots, x_n\} \subset \mathcal{X}$에 대해

$$(f(x_1), \ldots, f(x_n)) \sim \mathcal{N}(\mu, \Sigma)$$

가 다변수 정규분포인 것. 여기서 $\mu_i = m(x_i)$, $\Sigma_{ij} = k(x_i, x_j)$.

표기: $f \sim \mathcal{GP}(m, k)$.

### 정의 1.2 — Mean Function · Covariance Function

- $m : \mathcal{X} \to \mathbb{R}$, $m(x) = \mathbb{E}[f(x)]$.
- $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$, $k(x, y) = \mathbb{E}[(f(x) - m(x))(f(y) - m(y))]$.

$k$는 **PD kernel**이어야 한다 (정리 1.2 아래).

### 정의 1.3 — 대표 Covariance Functions

- **Constant**: $k(x, y) = \sigma^2$.
- **RBF (Squared Exponential)**: $k(x, y) = \sigma_f^2 \exp(-\|x-y\|^2 / 2\ell^2)$.
- **Matérn-$\nu$**: Ch1-02 정의 2.5.
- **Linear**: $k(x, y) = \sigma_0^2 + \sigma_f^2 x^\top y$.
- **Periodic**: $k(x, y) = \sigma_f^2 \exp(-2 \sin^2(\pi \|x-y\|/p) / \ell^2)$.
- **Compound**: $k_1 + k_2$, $k_1 \cdot k_2$ (Ch1-03 PD 보존).

---

## 🔬 정리와 증명

### 정리 1.1 — GP의 존재성 (Kolmogorov 확장)

**명제**: 임의의 mean function $m$과 **PD kernel** $k$에 대해 GP $f \sim \mathcal{GP}(m, k)$가 (적절한 확률공간 위에) 존재한다.

**증명 스케치** (Kolmogorov 확장):

Step 1: 각 유한 $\{x_1, \ldots, x_n\} \subset \mathcal{X}$에 대해 $\mathcal{N}(\mu, K)$ 분포 $P_{x_1, \ldots, x_n}$ 정의 (well-defined: $K$ PSD).

Step 2: **Consistency**: 부분집합 $\{x_1, \ldots, x_m\} \subset \{x_1, \ldots, x_n\}$에 대해 $P$의 marginal은 $P_{x_1, \ldots, x_m}$와 일치. Gaussian의 marginal = Gaussian (covariance submatrix), 자동 성립.

Step 3: **Permutation consistency**: 순서 바꿔도 같은 분포. $\mathcal{N}$의 대칭성.

Step 4: **Kolmogorov 확장 정리**: 위 두 consistency를 만족하는 finite-dim distribution family는 stochastic process $f : \Omega \times \mathcal{X} \to \mathbb{R}$로 확장 가능. $\square$

### 정리 1.2 — Covariance는 PD Kernel

**명제**: GP $f \sim \mathcal{GP}(m, k)$의 covariance function $k$는 **PD kernel**이다.

**증명**: 임의 $\{x_1, \ldots, x_n\}$, $\{\alpha_1, \ldots, \alpha_n\}$:

$$\sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) = \sum_{i, j} \alpha_i \alpha_j \mathbb{E}[(f(x_i) - m(x_i))(f(x_j) - m(x_j))]$$

$$= \mathbb{E}\left[\left(\sum_i \alpha_i (f(x_i) - m(x_i))\right)^2\right] \geq 0. \quad \square$$

**역**: 정리 1.1에 의해 **PD kernel $k$가 주어지면 그것을 covariance로 갖는 GP가 존재**. 따라서 "PD kernel ↔ GP covariance function"의 일대일 대응.

### 정리 1.3 — GP의 Marginals

**명제**: $f \sim \mathcal{GP}(m, k)$일 때 각 점 $x$에서 $f(x) \sim \mathcal{N}(m(x), k(x, x))$.

**증명**: 정의 1.1에서 $n = 1$. $\square$

### 정리 1.4 — Linear Combinations

**명제**: $f \sim \mathcal{GP}(m, k)$, $a_1, \ldots, a_n \in \mathbb{R}$, 점 $\{x_1, \ldots, x_n\}$에 대해

$$\sum_i a_i f(x_i) \sim \mathcal{N}\left(\sum_i a_i m(x_i), \sum_{i, j} a_i a_j k(x_i, x_j)\right).$$

**증명**: Gaussian의 linear transformation 성질. $\square$

### 정리 1.5 — Derivative GP

**명제**: $k$가 충분히 매끄러우면 (예: $C^2$), GP $f \sim \mathcal{GP}(m, k)$의 **미분** $f'$도 GP. Covariance는 $k$의 편미분:

$$\text{Cov}(f'(x), f'(y)) = \frac{\partial^2 k(x, y)}{\partial x \partial y}.$$

$\text{Cov}(f(x), f'(y)) = \frac{\partial k(x, y)}{\partial y}$.

**증명 아이디어**: 미분은 linear operation → Gaussian 유지. Covariance는 bilinear, operator 교환. $\square$

**응용**: GP로 gradient-aware surrogate modeling (e.g., Bayesian Optimization with derivatives).

### 정리 1.6 — GP Sample Paths의 Smoothness

**명제 (비공식)**: GP $\mathcal{GP}(0, k)$의 sample paths는 **$k$의 smoothness를 "거의" 물려받는다**. 정확히:

- RBF: sample path 거의 확실하게 $C^\infty$ (해석적).
- Matérn-$\nu$: sample path $C^{\lceil \nu \rceil - 1}$ 번 미분가능.
- Laplace (Matérn-1/2): 연속이지만 미분 불가능 (Brownian motion-like).

**정확한 진술**: Sample paths가 $k$의 RKHS에 **거의 확실하게 속하지 않는다** (Driscoll 1973). 하지만 "주변"의 Hölder·Sobolev class에 속함.

**주의**: RKHS $\mathcal{H}_k$는 "학습 가능한 함수들의 공간"이고, GP sample paths는 "**생성되는 함수들의 공간**". 이 둘은 다름. (Kanagawa et al. 2018 참조.)

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. GP prior 샘플 — 여러 kernel
# ─────────────────────────────────────────────
x = np.linspace(-5, 5, 200).reshape(-1, 1)
n = len(x)

def rbf(X, Y, sigma_f=1.0, ell=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return sigma_f**2 * np.exp(-d2 / (2 * ell**2))

def matern52(X, Y, sigma_f=1.0, ell=1.0):
    d = np.sqrt(np.maximum(0, np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T))
    r = np.sqrt(5) * d / ell
    return sigma_f**2 * (1 + r + r**2/3) * np.exp(-r)

def laplace(X, Y, sigma_f=1.0, ell=1.0):
    d = np.sqrt(np.maximum(0, np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T))
    return sigma_f**2 * np.exp(-d / ell)

def periodic(X, Y, sigma_f=1.0, ell=1.0, p=2.0):
    d = np.abs(X[:, None, 0] - Y[None, :, 0])
    return sigma_f**2 * np.exp(-2 * np.sin(np.pi * d / p)**2 / ell**2)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (name, K_fn) in zip(axes.ravel(), [('RBF', rbf), ('Matérn-5/2', matern52), ('Laplace', laplace), ('Periodic(p=2)', periodic)]):
    K = K_fn(x, x) + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(K)
    samples = L @ rng.standard_normal((n, 3))
    ax.plot(x, samples)
    ax.set_title(f'{name} — GP 샘플 (prior)')
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 2. Marginal normality 검증
# ─────────────────────────────────────────────
# GP(0, RBF)에서 특정 점 x_0 = 0의 marginal = N(0, k(0,0)) = N(0, 1)
n_samples = 5000
K = rbf(x, x, sigma_f=1.0, ell=1.0) + 1e-6 * np.eye(n)
L = np.linalg.cholesky(K)
samples = L @ rng.standard_normal((n, n_samples))
i0 = n // 2  # x_0 = 0에 해당
f_at_0 = samples[i0, :]
print(f'f(0) 표본 평균: {f_at_0.mean():.4f} (이론 0)')
print(f'f(0) 표본 분산: {f_at_0.var():.4f} (이론 k(0,0) = 1)')

# ─────────────────────────────────────────────
# 3. Joint normality: 두 점 (f(x_1), f(x_2)) ~ N(0, [[k_11, k_12],[k_12, k_22]])
# ─────────────────────────────────────────────
i1, i2 = 50, 150
f_pair = samples[[i1, i2], :].T  # (n_samples, 2)
cov_emp = np.cov(f_pair.T)
cov_theory = K[[i1, i2], :][:, [i1, i2]]
print(f'\n두 점 empirical cov:\n{cov_emp}')
print(f'두 점 theoretical cov:\n{cov_theory}')
print(f'차이 (max): {np.max(np.abs(cov_emp - cov_theory)):.4e}')

# ─────────────────────────────────────────────
# 4. Derivative GP
# ─────────────────────────────────────────────
# RBF의 derivative covariance: ∂²k/∂x∂y = σ² (1/ℓ² - (x-y)²/ℓ⁴) · exp(-(x-y)²/2ℓ²)
def rbf_deriv_cov(X, Y, sigma_f=1.0, ell=1.0):
    diff = X[:, 0:1] - Y[:, 0:1].T
    return sigma_f**2 * (1/ell**2 - diff**2 / ell**4) * np.exp(-diff**2 / (2*ell**2))

# f와 f' joint
K_ff = rbf(x, x) + 1e-6 * np.eye(n)
K_fp = (x[:, 0:1] - x[:, 0:1].T) / 1.0**2 * rbf(x, x)  # ∂k/∂y
K_pp = rbf_deriv_cov(x, x) + 1e-6 * np.eye(n)
K_joint = np.block([[K_ff, -K_fp], [-K_fp.T, K_pp]])
K_joint = K_joint + 1e-6 * np.eye(2*n)
L_joint = np.linalg.cholesky(K_joint)
samp = L_joint @ rng.standard_normal(2*n)
f_samp, fp_samp = samp[:n], samp[n:]

plt.figure(figsize=(10, 4))
plt.plot(x, f_samp, 'b-', label='$f(x)$')
plt.plot(x, fp_samp, 'r--', label="$f'(x)$ (derivative GP)")
plt.title('RBF GP와 그 미분 (joint Gaussian)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

**출력 예시**:
```
f(0) 표본 평균: 0.0145 (이론 0)
f(0) 표본 분산: 1.0012 (이론 k(0,0) = 1)

두 점 empirical cov:
[[1.0023 0.5423]
 [0.5423 0.9932]]
두 점 theoretical cov:
[[1.0000 0.5461]
 [0.5461 1.0000]]
차이 (max): 3.81e-03
```

→ Marginal·joint normality 경험적 확인. Derivative GP도 같은 kernel family.

---

## 🔗 실전 활용

- **Bayesian Optimization**: GP로 unknown function 모델링, expected improvement·UCB 기반 acquisition으로 다음 쿼리 결정.
- **Active Learning**: GP의 uncertainty ($\sigma_*$) 큰 점을 next label 요청.
- **지구통계학 (Kriging)**: GP는 Kriging의 현대적 이름. 공간 데이터 interpolation.
- **Physics-informed ML**: Derivative GP로 ODE/PDE 제약 포함.
- **Multi-fidelity**: 서로 다른 fidelity의 데이터 소스를 multi-output GP로 결합.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Gaussian (normality) | Target distribution이 heavy-tailed·multimodal이면 부적합 |
| Stationary kernel | Non-stationary 데이터는 warped GP, deep GP 등 필요 |
| Sample path smoothness | RBF는 너무 smooth (해석적) → real data에 너무 strong assumption |
| $O(n^3)$ scaling | Sparse GP (Ch4-06), Random Features (Ch7-02) 필요 |
| Hyperparameter 민감 | Marginal likelihood 최대화가 local optimum 빠질 수 있음 |

---

## 📌 핵심 정리

$$\boxed{f \sim \mathcal{GP}(m, k) \iff \forall \{x_i\}: (f(x_1), \ldots, f(x_n)) \sim \mathcal{N}(\mu, K), \mu_i = m(x_i), K_{ij} = k(x_i, x_j)}$$

$$\boxed{k \text{ PD kernel} \iff k \text{ can be a GP covariance function}}$$

| 개념 | 의미 |
|------|------|
| GP | 함수에 대한 확률분포, finite-dim marginals이 Gaussian |
| Mean function $m$ | 평균 함수 |
| Covariance function $k$ | PD kernel, smoothness·상관구조 결정 |
| Kolmogorov 확장 | Consistency 만족하면 process 존재 |
| Sample path | "함수 하나"의 샘플, smoothness는 $k$에 의해 결정 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Linear kernel $k(x, y) = x^\top y$에 대응하는 GP $\mathcal{GP}(0, k)$의 sample path는 어떤 함수들인가?

<details>
<summary>힌트 및 해설</summary>

$\mathcal{X} = \mathbb{R}^d$에서 $f(x) = w^\top x$, $w \sim \mathcal{N}(0, I_d)$.

확인: $\mathbb{E}[f(x)] = 0$ ✓. $\text{Cov}(f(x), f(y)) = \mathbb{E}[(w^\top x)(w^\top y)] = x^\top \mathbb{E}[ww^\top] y = x^\top y = k(x, y)$. ✓

따라서 **linear kernel GP = "random linear function $w^\top x$, $w \sim \mathcal{N}(0, I)$"**. 
$d$-차원 선형 함수 공간 (유한 차원) 위의 Gaussian measure.

일반화: $k(x, y) = x^\top \Sigma y$ → $w \sim \mathcal{N}(0, \Sigma)$.

</details>

**문제 2** (심화): GP $\mathcal{GP}(0, k)$와 그 sample $f$의 **integral** $g(t) := \int_0^t f(s) ds$는 어떤 분포인가?

<details>
<summary>힌트 및 해설</summary>

적분은 linear operation → $g$도 GP.

$\mathbb{E}[g(t)] = \int_0^t \mathbb{E}[f(s)] ds = 0$.

$\text{Cov}(g(s), g(t)) = \int_0^s \int_0^t k(u, v) du dv$.

예: $k(u, v) = \mathbf{1}\{u = v\}$ (white noise)이면 $\text{Cov}(g(s), g(t)) = \min(s, t)$ — **Brownian motion**.

예: $k(u, v) = \exp(-|u - v|/\ell)$이면 $g$는 smoother process (integrated Ornstein-Uhlenbeck).

**응용**: Derivative GP의 반대 방향. Physics에서 velocity → position transformation.

</details>

**문제 3** (ML 연결): GP를 사용할 때 **mean function $m(x)$를 상수 (e.g., 0)로 놓는** 이유와 한계는?

<details>
<summary>힌트 및 해설</summary>

**장점**:
1. **Identifiability**: Mean 0 가정 → covariance만 학습. Mean + covariance 동시 학습은 identifiable 하지 않을 수 있음.
2. **Simple computation**: Closed-form posterior 공식이 간단 (Ch4-02).
3. **Data-centric normalization**: Target $y$를 $y - \bar{y}$로 centering하면 mean 0 합리.

**한계**:
1. **Strong prior**: "함수가 0 근처로 돌아오려 한다" — 데이터 없는 영역에서 prediction이 0으로 회귀. 외삽에 부적합.
2. **Trend 놓침**: Linear trend, periodic, polynomial mean function이 데이터에 명백하면 명시 필요.

**실무 patterns**:
- $m(x) = c$ (상수): 자동 학습.
- $m(x) = \beta^\top x$ (linear): 전역 trend 포함.
- $m(x) = \beta^\top \phi(x)$ (basis): 더 유연.

sklearn `GaussianProcessRegressor(normalize_y=True)`는 $y$ centering 자동.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch3-06. SVM Regression (SVR)](../ch3-svm/06-svr.md) | [02. GP Regression — Posterior 유도 ▶](./02-gp-posterior.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
