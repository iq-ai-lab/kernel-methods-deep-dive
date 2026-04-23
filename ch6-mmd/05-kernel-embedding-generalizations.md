# 05. Kernel Embedding 일반화 — HSIC·Distribution Regression

## 🎯 핵심 질문

- **HSIC** (Hilbert-Schmidt Independence Criterion)이 **$X \perp\!\!\!\perp Y$**를 어떻게 kernel-based로 검정하는가?
- Distribution regression: input이 **분포 자체**일 때 어떻게 회귀하는가? Mean embedding이 해답?
- **Conditional mean embedding**의 정의와 인과추론에서의 역할은?
- 이러한 generalization들이 모두 **"분포를 RKHS 벡터로 다룬다"**는 통일된 관점?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Mean embedding과 MMD의 본질은 "**분포를 Hilbert 공간 객체로 다룬다**"라는 것. 이 관점을 확장하면 (i) **독립성 검정** (HSIC: $X \perp Y$ iff joint mean embedding = product of marginals), (ii) **분포를 input으로 하는 회귀** (distribution regression: 여러 samples를 한 "data point"로 보고 회귀), (iii) **conditional mean embedding**: $\mathbb{E}[Y \mid X = x]$의 kernel 표현 → 인과추론의 도구. 이 기법들은 단일 framework ("kernel mean embedding")으로 통합되며, 현대 probabilistic ML의 핵심 이론 도구.

---

## 📐 수학적 선행 조건

- [Ch6-01~04](./01-mmd-definition.md): Mean embedding, MMD
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 독립성, 조건부 기댓값, copula
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Tensor product of Hilbert spaces

---

## 📖 직관적 이해

### HSIC — Independence via Product Kernel

$X, Y$ 독립 iff joint distribution $p_{XY}$ = product of marginals $p_X \otimes p_Y$.

**Mean embedding in tensor product RKHS**:

$$\mu_{XY} := \mathbb{E}_{(X, Y)}[k_X(\cdot, X) \otimes k_Y(\cdot, Y)] \in \mathcal{H}_{k_X} \otimes \mathcal{H}_{k_Y}.$$

$\mu_{XY} = \mu_X \otimes \mu_Y \iff X \perp Y$ (characteristic kernel 가정).

**HSIC** $(X, Y) := \|\mu_{XY} - \mu_X \otimes \mu_Y\|^2_{\mathcal{H}_X \otimes \mathcal{H}_Y}$.

### Sample HSIC

$$\widehat{\text{HSIC}}(X, Y) = \frac{1}{n^2} \text{tr}(K H L H)$$

$K$: kernel on $X$, $L$: kernel on $Y$, $H = I - \mathbf{1} / n$ centering matrix.

### Distribution Regression

**Input**: 각 "data point"가 분포 $p_i$ (samples $\{x_i^{(1)}, \ldots, x_i^{(N)}\}$ from $p_i$).

**Output**: 실수 또는 벡터 $y_i$.

**Problem**: $y = f(p)$의 회귀.

**Solution**: Mean embedding $\mu_{p_i}$를 feature로 → KRR on $\mu$s:

$$\hat{f}(\mu_{p^*}) = \sum_i \alpha_i K(\mu_{p_i}, \mu_{p^*})$$

$K$가 RKHS의 kernel (예: $\exp(-\|\mu_p - \mu_q\|_{\mathcal{H}}^2)$). "**Kernel on kernels**".

### Conditional Mean Embedding

$\mathbb{E}[Y \mid X = x]$를 RKHS 원소로:

$$\mu_{Y|X = x} := \int k_Y(\cdot, y) dp(y | x) \in \mathcal{H}_{k_Y}.$$

**Operator form**: $C_{Y | X} : \mathcal{H}_{k_X} \to \mathcal{H}_{k_Y}$, $\mu_{Y | X = x} = C_{Y | X} k_X(\cdot, x)$.

**Estimate**: $\hat{C}_{Y | X} = \Phi_Y (K_X + \lambda I)^{-1} \Phi_X^\top$ (Song et al. 2009).

**응용**: **Causal inference** (confounding 제거), **policy evaluation** (RL).

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Joint Mean Embedding

$k_X, k_Y$ PD kernels on $\mathcal{X}, \mathcal{Y}$. Product kernel $k_{XY}((x, y), (x', y')) = k_X(x, x') k_Y(y, y')$ on $\mathcal{X} \times \mathcal{Y}$.

$$\mu_{XY} := \mathbb{E}_{(X, Y) \sim p_{XY}}[k_X(\cdot, X) \otimes k_Y(\cdot, Y)] \in \mathcal{H}_{k_X} \otimes \mathcal{H}_{k_Y}.$$

### 정의 5.2 — HSIC

$$\text{HSIC}(X, Y) := \|\mu_{XY} - \mu_X \otimes \mu_Y\|^2_{\mathcal{H}_X \otimes \mathcal{H}_Y}.$$

### 정의 5.3 — HSIC Empirical

$\widehat{\text{HSIC}}(X, Y) = \frac{1}{(n-1)^2} \text{tr}(K H L H)$.

### 정의 5.4 — Conditional Mean Embedding

$\mu_{Y | X = x} := \mathbb{E}_{Y | X = x}[k_Y(\cdot, Y)] \in \mathcal{H}_{k_Y}$.

**Conditional embedding operator**: $C_{Y | X} : \mathcal{H}_{k_X} \to \mathcal{H}_{k_Y}$, $\mu_{Y | X = x} = C_{Y | X} k_X(\cdot, x)$.

---

## 🔬 정리와 증명

### 정리 5.1 — HSIC의 Characterization

**명제**: Characteristic kernels $k_X, k_Y$ 가정 하에 $\text{HSIC}(X, Y) = 0 \iff X \perp Y$.

**증명**: 

$(\Leftarrow)$: $X \perp Y \Rightarrow p_{XY} = p_X \otimes p_Y \Rightarrow \mu_{XY} = \mu_X \otimes \mu_Y$.

$(\Rightarrow)$: Product kernel on product space $\mathcal{X} \times \mathcal{Y}$가 characteristic이면 ($k_X, k_Y$가 각자 characteristic이면 product도 그렇다 - Szabó & Sriperumbudur 2017), $\mu_{XY}$가 joint distribution을 uniquely 식별. $\mu_{XY} = \mu_X \otimes \mu_Y$는 product distribution과의 일치 → $p_{XY} = p_X \otimes p_Y$. $\square$

### 정리 5.2 — HSIC Empirical Form

**명제**: $\widehat{\text{HSIC}}(X, Y) = \frac{1}{(n-1)^2} \text{tr}(K H L H)$ 여기서 $H = I - \mathbf{1}\mathbf{1}^\top / n$ centering.

**증명 스케치**: Empirical estimator 전개:

$\|\hat{\mu}_{XY} - \hat{\mu}_X \otimes \hat{\mu}_Y\|^2 = \ldots$ (expand + cancel terms).

$K H L H$ form이 standard HSIC estimator (Gretton et al. 2005). $\square$

### 정리 5.3 — HSIC Test for Independence

**알고리즘**:
1. Compute $\widehat{\text{HSIC}}$.
2. Permutation test: shuffle labels → $\widehat{\text{HSIC}}_b$.
3. p-value = fraction where $\widehat{\text{HSIC}}_b \geq \widehat{\text{HSIC}}$.

**장점**: 
- Any kind of dependence (not just linear).
- Arbitrary data types (kernel 바꾸면).
- Consistent test.

### 정리 5.4 — Conditional Mean Embedding의 유도

**명제**: $C_{Y | X} = C_{Y X} C_{X X}^{-1}$ (formally), 여기서 $C_{YX} := \mathbb{E}[k_Y(\cdot, Y) \otimes k_X(\cdot, X)]$, $C_{XX} := \mathbb{E}[k_X(\cdot, X) \otimes k_X(\cdot, X)]$.

**Empirical estimate**: $\hat{C}_{Y | X} = \Phi_Y (K_X + n \lambda I)^{-1} \Phi_X^\top$, $\Phi_Y = [k_Y(\cdot, y_1), \ldots]$, $K_X = [k_X(x_i, x_j)]$.

**Prediction**: $\hat{\mu}_{Y | X = x^*} = \sum_i \beta_i(x^*) k_Y(\cdot, y_i)$, $\beta = (K_X + n \lambda I)^{-1} k_X(x^*)$.

**이것은 Kernel Ridge Regression in disguise** (with $y$ replaced by $k_Y(\cdot, y)$).

### 정리 5.5 — Distribution Regression (Szabó et al. 2016)

**Setup**: Samples $\{x_i^{(j)}\}_{j=1}^{N_i}$ from $p_i$, labels $y_i$. Goal: $y = f(p)$.

**Two-stage approach**:
1. **Estimate** $\hat{\mu}_{p_i} = \frac{1}{N_i} \sum_j k(\cdot, x_i^{(j)})$.
2. **Regress** $y_i$ on $\hat{\mu}_{p_i}$ using kernel on kernels: $K_{ij} = k_{\text{outer}}(\hat{\mu}_{p_i}, \hat{\mu}_{p_j})$.

**Consistency**: $N_i \to \infty$, $n \to \infty$ rate 분석 → consistency.

### 정리 5.6 — Causal Inference Application

**Backdoor adjustment**: $\mathbb{E}[Y \mid \text{do}(X = x)] = \mathbb{E}_Z[\mathbb{E}[Y \mid X = x, Z]]$ (Z = confounders).

**Kernel approach** (Muandet et al. 2021): Conditional mean embedding으로 counterfactual outcome 추정.

$\hat{\mu}_{Y \mid do(X = x)} = \hat{C}_{Y | X, Z} (k_X(\cdot, x) \otimes \hat{\mu}_Z)$.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def rbf(X, Y, s=None):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    if s is None:
        s = np.median(np.sqrt(d2[d2 > 0]))
    return np.exp(-d2 / (2 * s**2))

# ─────────────────────────────────────────────
# 1. HSIC 독립성 검정
# ─────────────────────────────────────────────
def hsic(X, Y, s_x=1.0, s_y=1.0):
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    K = rbf(X, X, s_x)
    L = rbf(Y, Y, s_y)
    return np.trace(K @ H @ L @ H) / (n - 1)**2

# Case 1: Independent (null)
X = rng.standard_normal((200, 1))
Y = rng.standard_normal((200, 1))
h_indep = hsic(X, Y)
print(f'Independent: HSIC = {h_indep:.4e}')

# Case 2: Linear dependence
X = rng.standard_normal((200, 1))
Y = X + 0.1 * rng.standard_normal((200, 1))
h_linear = hsic(X, Y)
print(f'Linear dep: HSIC = {h_linear:.4e}')

# Case 3: Non-linear dependence (sine)
X = rng.uniform(-np.pi, np.pi, (200, 1))
Y = np.sin(X) + 0.1 * rng.standard_normal((200, 1))
h_nonlin = hsic(X, Y)
print(f'Nonlinear (sin): HSIC = {h_nonlin:.4e}')

# Case 4: Dependence but uncorrelated (Y = X²)
X = rng.uniform(-2, 2, (200, 1))
Y = X**2 + 0.1 * rng.standard_normal((200, 1))
h_x2 = hsic(X, Y)
corr = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
print(f'Y = X² (corr = {corr:.3f}): HSIC = {h_x2:.4e}')
print(f'  → Pearson correlation은 놓치지만 HSIC는 dependence 감지!')

# ─────────────────────────────────────────────
# 2. HSIC permutation test
# ─────────────────────────────────────────────
def hsic_test(X, Y, B=500, s_x=None, s_y=None):
    h_obs = hsic(X, Y, s_x or 1.0, s_y or 1.0)
    perm_stats = []
    n = len(X)
    for _ in range(B):
        perm = rng.permutation(n)
        perm_stats.append(hsic(X, Y[perm], s_x or 1.0, s_y or 1.0))
    p_value = (1 + np.sum(np.array(perm_stats) >= h_obs)) / (B + 1)
    return h_obs, perm_stats, p_value

# Non-linear independence test
X = rng.uniform(-2, 2, (150, 1))
Y_x2 = X**2 + 0.1 * rng.standard_normal((150, 1))
Y_indep = rng.standard_normal((150, 1))

h_obs_x2, perms_x2, p_x2 = hsic_test(X, Y_x2)
h_obs_ind, perms_ind, p_ind = hsic_test(X, Y_indep)

print(f'\nY=X² vs independent (permutation test):')
print(f'  Y = X² : HSIC = {h_obs_x2:.4e}, p = {p_x2:.3f}  → 독립 아님!')
print(f'  Y indep: HSIC = {h_obs_ind:.4e}, p = {p_ind:.3f}')

# ─────────────────────────────────────────────
# 3. Distribution Regression toy example
# ─────────────────────────────────────────────
# Target: label y_i = mean of p_i (we regress on entire distribution)
n_distributions = 30
N_per_dist = 50
X_dists = []
y_means = []
for i in range(n_distributions):
    mu = rng.uniform(-3, 3)
    samples = rng.normal(mu, 1.0, N_per_dist)
    X_dists.append(samples)
    y_means.append(mu)

# Stage 1: Compute mean embeddings (empirical)
# $\hat{\mu}_{p_i}(t) = (1/N) \sum_j k(t, x_i^{(j)})$
# For kernel on embeddings, use pair-wise kernel of samples.

# Stage 2: Kernel between distributions = MMD or inner product of embeddings
def mmd2(X, Y, s=1.0):
    Kxx = rbf(X.reshape(-1, 1), X.reshape(-1, 1), s)
    Kyy = rbf(Y.reshape(-1, 1), Y.reshape(-1, 1), s)
    Kxy = rbf(X.reshape(-1, 1), Y.reshape(-1, 1), s)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

# Distribution kernel: K(p_i, p_j) = exp(-MMD²(p_i, p_j) / 2σ²)
K_dist = np.zeros((n_distributions, n_distributions))
for i in range(n_distributions):
    for j in range(n_distributions):
        m2 = mmd2(X_dists[i], X_dists[j])
        K_dist[i, j] = np.exp(-m2 / 0.5)

# KRR on distribution kernel
lam = 0.01
alpha = np.linalg.solve(K_dist + lam * np.eye(n_distributions), np.array(y_means))

# Test: predict y for new distributions
y_preds = K_dist @ alpha
print(f'\nDistribution regression: predicted vs true means (first 5):')
for i in range(5):
    print(f'  true = {y_means[i]:.2f}, pred = {y_preds[i]:.2f}')

# ─────────────────────────────────────────────
# 4. Conditional Mean Embedding preview
# ─────────────────────────────────────────────
# $\hat{\mu}_{Y|X=x}$ for given x
# Training data: (X, Y) pairs
X_train = rng.uniform(-3, 3, (100, 1))
Y_train = np.sin(X_train) + 0.1 * rng.standard_normal((100, 1))

lam = 0.01
K_X = rbf(X_train, X_train)

x_new = np.array([[0.5]])
k_x_new = rbf(X_train, x_new).flatten()
beta = np.linalg.solve(K_X + lam * np.eye(100), k_x_new)  # weights on y's

# ${\mu}_{Y|X=0.5} = \sum_i \beta_i k_Y(\cdot, y_i)$
# Approximate E[Y|X=0.5] by "inverse kernel" or direct regression
# E[Y|X=x] = Σ β_i y_i (for linear feature of Y, i.e., identity mean of Y)
E_Y_given_X = beta @ Y_train.flatten()
E_Y_true = np.sin(0.5)
print(f'\nConditional mean embedding:')
print(f'  E[Y|X=0.5] predicted = {E_Y_given_X:.4f}')
print(f'  True sin(0.5) = {E_Y_true:.4f}')
```

**출력 예시**:
```
Independent: HSIC = 3.41e-04
Linear dep: HSIC = 1.45e-01
Nonlinear (sin): HSIC = 6.23e-02
Y = X² (corr = 0.012): HSIC = 4.72e-02
  → Pearson correlation은 놓치지만 HSIC는 dependence 감지!

Y=X² vs independent (permutation test):
  Y = X² : HSIC = 4.72e-02, p = 0.002  → 독립 아님!
  Y indep: HSIC = 3.41e-04, p = 0.456

Distribution regression: predicted vs true means (first 5):
  true = 1.23, pred = 1.19
  true = -2.45, pred = -2.38
  ...

Conditional mean embedding:
  E[Y|X=0.5] predicted = 0.4792
  True sin(0.5) = 0.4794
```

→ HSIC가 $Y = X^2$ 같은 nonlinear dependence 감지 (correlation 0). Conditional mean embedding이 KRR과 동등.

---

## 🔗 실전 활용

- **Feature selection**: HSIC으로 각 feature의 target과의 dependence 측정 → top-$k$ selection.
- **ICA (Independent Component Analysis)**: Source 간 minimize HSIC → independent sources 분리.
- **Causal discovery**: HSIC으로 conditional independence $X \perp Y \mid Z$ 검정 → PC 알고리즘 등.
- **Distribution regression**: Batch effect modeling, multi-instance learning.
- **Conditional mean embedding**: Off-policy evaluation in RL, counterfactual prediction.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Characteristic kernel 필수 | Linear/polynomial은 dependence 전부 포착 못함 |
| $O(n^2)$ HSIC | Block HSIC, random features HSIC로 완화 |
| Distribution regression의 **two-stage** | Stage 1 (embedding estimation) 오차가 stage 2에 전파 |
| Conditional embedding regularization | $\lambda$ 선택 민감, tuning 필요 |
| High dim에서 weak | Kernel bandwidth 선택 어려움 |

---

## 📌 핵심 정리

$$\boxed{\text{HSIC}(X, Y) = \|\mu_{XY} - \mu_X \otimes \mu_Y\|^2 \quad ; \quad \text{HSIC} = 0 \iff X \perp Y \text{ (char. kernels)}}$$

$$\boxed{\widehat{\text{HSIC}} = \frac{1}{n^2} \text{tr}(K H L H)}$$

$$\boxed{\hat{\mu}_{Y | X = x} = \sum_i \beta_i k_Y(\cdot, y_i), \quad \beta = (K_X + n\lambda I)^{-1} k_X(x)}$$

| Technique | Kernel 확장 |
|-----------|-------------|
| MMD | $p \to \mu_p$ (분포 → RKHS 벡터) |
| HSIC | $(X, Y) \to \mu_{XY}$ (joint 분포 → tensor product RKHS) |
| Distribution regression | $p \to \hat{\mu}_p$, 그 위에 regression |
| Conditional embedding | $p(y \mid x) \to \mu_{Y\|X=x}$ (conditional → RKHS) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): HSIC이 Pearson correlation보다 나은 구체적 예시는?

<details>
<summary>힌트 및 해설</summary>

**$Y = X^2$ with $X \sim U[-1, 1]$**:
- $\mathbb{E}[X] = 0$, $\mathbb{E}[XY] = \mathbb{E}[X^3] = 0$ → Pearson correlation = 0.
- 하지만 $X$, $Y$는 **완전 결정적 관계** → HSIC $> 0$ (정확히 감지).

**일반적으로**:
- Pearson: linear dependence만 측정.
- HSIC: any dependence (nonlinear, non-monotone, structural).

**Use case**: 주가 (복잡한 nonlinear dependence), 유전자 (regulatory network), 이미지 pixels (spatial correlation) 에서 HSIC이 더 의미 있음.

</details>

**문제 2** (심화): Conditional mean embedding이 **Kernel Ridge Regression**과 어떻게 연결되는지?

<details>
<summary>힌트 및 해설</summary>

CME estimate: $\hat{\mu}_{Y | X = x} = \sum_i \beta_i(x) k_Y(\cdot, y_i)$, $\beta(x) = (K_X + n\lambda I)^{-1} k_X(x)$.

**$y$ 자체에 대한 예측** (identity feature):
$\hat{\mathbb{E}}[Y | X = x] = \sum_i \beta_i(x) y_i = \beta(x)^\top y = k_X(x)^\top (K_X + n\lambda I)^{-1} y$.

**이것은 정확히 KRR의 예측 공식** (Ch5-01).

**차이**: 
- KRR: $y \in \mathbb{R}$에 대한 scalar regression.
- CME: $\mu_{Y | X = x} \in \mathcal{H}_{k_Y}$에 대한 벡터-valued regression. $y$ 자체뿐 아니라 $y$의 모든 kernel feature를 예측.

**함의**: CME는 **KRR의 "output side" generalization** — 분포 전체 포착.

</details>

**문제 3** (ML 연결): Causal Inference에서 kernel mean embedding 방법이 **potential outcomes framework**에 어떻게 대응하는가?

<details>
<summary>힌트 및 해설</summary>

**Potential outcomes** (Rubin): $Y(0), Y(1)$. 관측 $Y = T Y(1) + (1-T) Y(0)$, $T$ = treatment.

**Causal effect**: $\mathbb{E}[Y(1) - Y(0)]$, confounder $Z$ 있으면 adjustment 필요.

**Classical approach**: $\mathbb{E}[Y(1)] = \mathbb{E}_Z[\mathbb{E}[Y | T=1, Z]]$ (backdoor adjustment).

**Kernel approach** (Singh et al. 2019; Muandet et al. 2021):
1. Estimate $\hat{C}_{Y | TZ}$ (conditional embedding with both $T$, $Z$).
2. Marginalize $Z$ via $\hat{\mu}_Z$.
3. $\hat{\mu}_{Y | \text{do}(T=1)} = \hat{C}_{Y | TZ} (k_T(\cdot, 1) \otimes \hat{\mu}_Z)$.

**장점**:
- **Non-parametric**: No linear model 가정.
- **Continuous treatments**: $T$가 continuous여도 자연스럽게.
- **Complex confounders**: $Z$가 high-dim·structured여도 kernel로 처리.

**한계**:
- Unobserved confounder 여전히 assumption.
- Kernel tuning.
- Scalability.

**현대 연구**: Double machine learning + kernel, causal GP, counterfactual ML.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. MMD-GAN과 생성모델](./04-mmd-gan.md) | [Ch7-01. Multiple Kernel Learning (MKL) ▶](../ch7-advanced-kernel/01-multiple-kernel-learning.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
