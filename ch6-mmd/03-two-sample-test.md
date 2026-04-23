# 03. Two-Sample Test (Gretton et al. 2012)

## 🎯 핵심 질문

- Two-sample test $H_0: p = q$ vs $H_1: p \ne q$를 MMD²로 어떻게 검정하는가?
- **Null distribution of $\widehat{\text{MMD}}^2$**의 점근분포는 어떻게 유도되는가?
- **Permutation test** 방법 — 왜 정확한 null critical value를 제공하는가?
- MMD test가 **고차원·구조화 데이터**에서 왜 Kolmogorov-Smirnov, energy distance보다 강력한가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Two-sample test는 "두 샘플이 같은 분포에서 왔는가?"라는 과학적 기본 질문. 기존 방법 (KS test, t-test)은 (i) 1D 또는 간단한 분포에만 작동, (ii) moment 기반이라 high-moment 차이 놓침. MMD-based two-sample test (Gretton 2012)는 **high-dim data, structured data (text, graph, image)**에 적용 가능한 **원칙적 generalization**. 실무 응용: (i) **A/B testing** (두 실험 분포 비교), (ii) **distribution drift detection** (deployed model), (iii) **Generative model evaluation** (GAN samples vs real data), (iv) **Biomedical trials** (treatment vs control).

---

## 📐 수학적 선행 조건

- [Ch6-01~02](./01-mmd-definition.md): MMD 정의, estimators
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): **Hypothesis testing**, p-value, Type I/II error
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Bootstrap, permutation, 점근분포

---

## 📖 직관적 이해

### Hypothesis Testing 설정

- $H_0$ (null): $p = q$.
- $H_1$ (alternative): $p \ne q$.

**Test statistic**: $\widehat{\text{MMD}}^2$. 크면 $H_1$ 지지.

**Critical value**: $H_0$ 하에서 $\widehat{\text{MMD}}^2$의 분포 (null distribution)의 $(1 - \alpha)$-quantile. $\widehat{\text{MMD}}^2 > c_{1-\alpha}$이면 reject $H_0$ at level $\alpha$.

### 왜 Critical Value 찾기가 어려운가

Null 분포는 $p = q$ 가정 하 $\widehat{\text{MMD}}^2$의 분포. **$p$ 자체가 미지**이므로 이 분포를 바로 알 수 없다.

**해결책 2가지**:
1. **점근분포** (asymptotic): $n \to \infty$에서 null distribution의 수학적 형태 유도 → 근사 critical value.
2. **Permutation test**: 샘플 랜덤 혼합으로 null 근사 → empirical critical value.

### 점근분포 (Null)

**정리**: $p = q$ 하에서

$$n \widehat{\text{MMD}}_u^2 \xrightarrow{d} \sum_{l=1}^\infty \lambda_l (z_l^2 - 1), \quad z_l \sim \mathcal{N}(0, 1) \text{ i.i.d.}$$

($\lambda_l$은 kernel 관련 operator의 고유값, Ch1-04 Mercer.) Infinite mixture of chi-squared minus constants. 복잡해서 직접 critical value 어려움.

**Gamma approximation** (Gretton 2012): Moments matching으로 Gamma 분포 근사 → 실용적 critical value.

### Permutation Test

**아이디어**: $p = q$이면 $\{x_i\}$와 $\{y_j\}$의 **label은 random**. 따라서 random 섞은 후 MMD² 계산 = null 분포 샘플.

**알고리즘**:
1. Compute $\widehat{\text{MMD}}^2_{\text{obs}}$.
2. Repeat $B$ times:
   a. Random permute labels of $\{x_1, \ldots, x_n, y_1, \ldots, y_m\}$.
   b. Compute $\widehat{\text{MMD}}^2_b$ from shuffled.
3. p-value $= \frac{1}{B} \sum_b \mathbf{1}\{\widehat{\text{MMD}}^2_b \geq \widehat{\text{MMD}}^2_{\text{obs}}\}$.
4. Reject $H_0$ if p-value $< \alpha$.

**장점**: Exact null distribution (finite sample), no asymptotic approximation.

**단점**: $O(B n^2)$ compute. Typical $B = 1000$.

### Why MMD Test Powerful

- **Characteristic kernel** → **all distribution differences** 포착 (high-moment 포함).
- **RKHS**로 high-dim 데이터 자연스럽게 처리.
- **Kernel 선택 유연**: Gaussian (default), Matérn, custom kernel (graph, string) 가능.

**Limitation**: **Kernel bandwidth** 선택이 power 결정. Median heuristic 또는 **Type-II error 최소화**로 튜닝.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Two-Sample Test

Given samples $\{x_i\}_{i=1}^n \sim p$, $\{y_j\}_{j=1}^m \sim q$. Test $H_0: p = q$ vs $H_1: p \ne q$ at level $\alpha$.

### 정의 3.2 — MMD-based Test

1. Compute $T_n := \widehat{\text{MMD}}_u^2(\{x_i\}, \{y_j\})$.
2. Determine critical value $c_{1-\alpha}$ (permutation or asymptotic).
3. Reject $H_0$ if $T_n > c_{1-\alpha}$.

### 정의 3.3 — Permutation Test

$B$ random label permutations $\pi_b$ of $\{x_1, \ldots, x_n, y_1, \ldots, y_m\}$. Shuffled MMD² $T_n^{(b)}$. p-value:

$$\hat{p} := \frac{1 + \sum_b \mathbf{1}\{T_n^{(b)} \geq T_n\}}{B + 1}.$$

### 정의 3.4 — Null 점근분포 (Gretton et al.)

$p = q$ 하:

$$n \widehat{\text{MMD}}_u^2 \xrightarrow{d} \sum_{l=1}^\infty \lambda_l (z_l^2 - 1),$$

$\{\lambda_l\}$은 centered kernel $\tilde{k}(x, y) = k(x, y) - \mathbb{E}_p[k(X, y)] - \mathbb{E}_p[k(x, Y)] + \mathbb{E}[k(X, Y)]$의 integral operator 고유값.

---

## 🔬 정리와 증명

### 정리 3.1 — 점근 Null Distribution

**명제** (Gretton et al. 2012, Theorem 8): 정의 3.4.

**증명 개요** (U-statistic의 limit theory):

1. $\widehat{\text{MMD}}_u^2$는 degenerate U-statistic under $H_0$ (first-order kernel zero).
2. **Second-order U-statistic limit theorem**: Spectral decomposition of centered kernel → chi-squared mixture.
3. $\lambda_l$은 integral operator $T_{\tilde{k}}$의 고유값 (Mercer, Ch1-04 정리 4.1).

세부는 Serfling (1980) Chapter 5. $\square$

### 정리 3.2 — Permutation Test의 Validity

**명제**: 정의 3.3의 permutation test는 **exactly valid** — $p = q$ 하 Type-I error $\leq \alpha$.

**증명 아이디어** (exchangeability):

$p = q$이면 $\{x_1, \ldots, x_n, y_1, \ldots, y_m\}$이 **exchangeable**. 임의 permutation $\pi$에 대해 $T_n(\pi(\text{data}))$가 같은 분포. 

$\{T_n^{(b)}\}$와 $T_n$은 iid sample of null distribution → $T_n$의 rank가 uniform on $\{1, \ldots, B+1\}$ under $H_0$.

$P(T_n \text{ in top } \alpha \text{ of } \{T_n\} \cup \{T_n^{(b)}\}) \leq \alpha$. $\square$

### 정리 3.3 — Test의 Power

**명제**: $p \ne q$ 이고 $k$ characteristic이면 test power $\to 1$ as $n \to \infty$.

**증명 아이디어**: $\widehat{\text{MMD}}_u^2 \to \text{MMD}^2 > 0$ (characteristic). 한편 critical value $c_{1-\alpha} = O(1/\sqrt{n})$ → $T_n > c_{1-\alpha}$ with probability $\to 1$. $\square$

**Rate**: Finite sample에서 power는 $\text{MMD}^2 / \text{SD}(\widehat{\text{MMD}}^2)$ — "effect size / noise" ratio.

### 정리 3.4 — Kernel Bandwidth 선택 (Power Optimization)

**명제**: $\sigma$ 선택이 test power에 큰 영향.

**휴리스틱**:
1. **Median heuristic**: $\sigma = \text{median}\{\|x_i - x_j\| : i \ne j\}$. 간단하지만 optimal 아님.
2. **Maximum power**: $\sigma^* = \arg\max_\sigma \frac{\text{MMD}^2_\sigma}{\sqrt{\text{Var}[\widehat{\text{MMD}}^2_\sigma]}}$. Gretton의 Deep kernel MMD (ICML 2017).
3. **Cross-validated**: Data split, half-sample에서 $\sigma$ 튜닝.

### 정리 3.5 — 다른 Tests와 비교

| Test | Data | Sensitivity | Scaling |
|------|------|-------------|---------|
| Kolmogorov-Smirnov | 1D only | CDF 차이 | $O(n \log n)$ |
| Chi-squared | Categorical / histogram | Bin 선택 민감 | $O(n)$ |
| Anderson-Darling | 1D | Tail-sensitive | $O(n \log n)$ |
| **MMD + RBF** | Any dim | All moments (char. kernel) | $O(n^2)$ |
| Energy distance | Any dim | Similar to MMD with specific kernel | $O(n^2)$ |

**Energy distance = MMD with $k(x, y) = -\|x-y\| + \|x\| + \|y\|$**. 관련 있지만 bandwidth-free.

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
        # Median heuristic
        d2_all = np.concatenate([np.sqrt(d2.flatten()), np.sqrt(np.diag(d2))])
        s = np.median(d2_all[d2_all > 0])
    return np.exp(-d2 / (2 * s**2))

def mmd2_unbiased(X, Y, s=1.0):
    n, m = len(X), len(Y)
    Kxx = rbf(X, X, s); Kyy = rbf(Y, Y, s); Kxy = rbf(X, Y, s)
    return (Kxx.sum() - np.trace(Kxx)) / (n*(n-1)) + \
           (Kyy.sum() - np.trace(Kyy)) / (m*(m-1)) - \
           2 * Kxy.mean()

# ─────────────────────────────────────────────
# 1. Permutation test
# ─────────────────────────────────────────────
def permutation_test(X, Y, B=1000, s=1.0, seed=0):
    rng_local = np.random.default_rng(seed)
    T_obs = mmd2_unbiased(X, Y, s)
    all_data = np.concatenate([X, Y])
    n, m = len(X), len(Y)
    T_perm = []
    for _ in range(B):
        perm = rng_local.permutation(n + m)
        X_p = all_data[perm[:n]]
        Y_p = all_data[perm[n:]]
        T_perm.append(mmd2_unbiased(X_p, Y_p, s))
    T_perm = np.array(T_perm)
    p_value = (1 + np.sum(T_perm >= T_obs)) / (B + 1)
    return T_obs, T_perm, p_value

# ─────────────────────────────────────────────
# 2. Null test (p = q)
# ─────────────────────────────────────────────
X_null = rng.standard_normal((200, 2))
Y_null = rng.standard_normal((200, 2))
T_obs, T_perm, p_null = permutation_test(X_null, Y_null, B=500, s=1.0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(T_perm, bins=30, alpha=0.7, label='Permutation null')
plt.axvline(T_obs, color='red', lw=2, label=f'Observed ({T_obs:.4f})')
plt.title(f'Null test (p=q): p-value = {p_null:.3f}')
plt.legend(); plt.grid(True, alpha=0.3)

# ─────────────────────────────────────────────
# 3. Alternative test (p != q)
# ─────────────────────────────────────────────
X_alt = rng.standard_normal((200, 2))
Y_alt = rng.standard_normal((200, 2)) + 0.5
T_obs_alt, T_perm_alt, p_alt = permutation_test(X_alt, Y_alt, B=500, s=1.0)

plt.subplot(1, 2, 2)
plt.hist(T_perm_alt, bins=30, alpha=0.7, label='Permutation null')
plt.axvline(T_obs_alt, color='red', lw=2, label=f'Observed ({T_obs_alt:.4f})')
plt.title(f'Alt test (p≠q): p-value = {p_alt:.3f}')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

print(f'Null (p=q)       : p = {p_null:.3f}  → Fail to reject H_0 at α=0.05' if p_null > 0.05 else f'Null (p=q): p = {p_null:.3f} → Rejected (Type I error)')
print(f'Alt  (mean shift): p = {p_alt:.3f}  → Reject H_0 at α=0.05' if p_alt < 0.05 else f'Alt  (mean shift): p = {p_alt:.3f} → Fail to reject (Type II error)')

# ─────────────────────────────────────────────
# 4. Type I / II error rate (simulation)
# ─────────────────────────────────────────────
alpha = 0.05
n_sim = 200

# Type I (null case)
type1_rejections = 0
for _ in range(n_sim):
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2))
    _, _, p = permutation_test(X, Y, B=200, s=1.0, seed=None)
    if p < alpha:
        type1_rejections += 1
print(f'\nType-I error rate (p=q, target {alpha}): {type1_rejections/n_sim:.3f}')

# Type II (alternative case)
type2_failures = 0
for _ in range(n_sim):
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2)) + 0.3
    _, _, p = permutation_test(X, Y, B=200, s=1.0, seed=None)
    if p >= alpha:
        type2_failures += 1
print(f'Type-II error rate (p≠q, small shift): {type2_failures/n_sim:.3f}')
print(f'Power (1 - Type II): {1 - type2_failures/n_sim:.3f}')

# ─────────────────────────────────────────────
# 5. KS test와 비교 (1D)
# ─────────────────────────────────────────────
from scipy.stats import ks_2samp

X_1d = rng.standard_normal(200)
# 같은 평균, 같은 분산, 다른 skewness
Y_1d = rng.standard_normal(200)
Y_1d = np.where(rng.uniform(0, 1, 200) < 0.5, Y_1d - 0.5, Y_1d + 0.5 + 0.3)
Y_1d = (Y_1d - Y_1d.mean()) / Y_1d.std()

ks_stat, ks_p = ks_2samp(X_1d, Y_1d)
mmd_obs, _, mmd_p = permutation_test(X_1d.reshape(-1, 1), Y_1d.reshape(-1, 1), B=500, s=0.5)
print(f'\n같은 mean/var, 다른 skew:')
print(f'  KS test: p = {ks_p:.4f}')
print(f'  MMD test: p = {mmd_p:.4f}')
```

**출력 예시**:
```
Null (p=q)       : p = 0.124  → Fail to reject H_0 at α=0.05
Alt  (mean shift): p = 0.001  → Reject H_0 at α=0.05

Type-I error rate (p=q, target 0.05): 0.050
Type-II error rate (p≠q, small shift): 0.235
Power (1 - Type II): 0.765

같은 mean/var, 다른 skew:
  KS test: p = 0.0823
  MMD test: p = 0.0321
```

→ Type-I error rate가 target $\alpha$에 근사. MMD가 KS보다 skew 차이 민감.

---

## 🔗 실전 활용

- **A/B Testing**: 두 그룹의 metric 분포 비교 (전체 분포, not just mean).
- **Model drift detection**: Training vs deployment input 분포 비교.
- **GAN evaluation**: Real data sample vs generated sample의 MMD test.
- **Biomedical**: Treatment vs control effect on high-dim outcomes (gene expression).
- **Python libraries**: `torch_two_sample`, `scipy.stats` (간단한 경우), 직접 구현.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Kernel bandwidth 선택 | Power 크게 좌우; median heuristic이 default but suboptimal |
| Permutation test $O(B n^2)$ | Large $n$, $B$ = 1000이면 $10^9$ ops |
| Small $n$ power | Effect size 작으면 false negative 높음 |
| **Multiple testing** | 여러 two-sample test 동시 수행 시 Bonferroni correction 필요 |
| **Characteristic 가정** | Non-characteristic이면 Type-II error 높음 |

---

## 📌 핵심 정리

$$\boxed{T_n = \widehat{\text{MMD}}_u^2 \to \text{Reject } H_0 \text{ if } T_n > c_{1-\alpha}}$$

$$\boxed{c_{1-\alpha} \text{ from permutation test (exact) or asymptotic } \sum \lambda_l (z_l^2 - 1)}$$

| 방법 | 장점 | 단점 |
|------|------|------|
| **Permutation** | Exact null, any kernel | $O(B n^2)$ |
| **Asymptotic** | Fast 1-shot | Approximation error |
| **Gamma approx** | Moments-based, simple | Small $n$ 부정확 |
| **Block-wise** | Scalable | Slight power loss |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 Permutation test가 null distribution의 **exact** approximation을 제공하는가?

<details>
<summary>힌트 및 해설</summary>

**핵심 근거**: $H_0: p = q$ 하에서 $\{x_i, y_j\}$는 **exchangeable** — 라벨을 random shuffle해도 joint distribution 불변.

**Exactness**: 가능한 모든 permutation (또는 $B$ random subset)에서 $T_n$의 분포가 exact null distribution.

**Conservativeness**: $p$-value $= (1 + \#\{T^{(b)} \geq T_{\text{obs}}\}) / (B + 1)$. $+1$ in numerator and denominator는 $T_{\text{obs}}$ 자체를 null sample 중 하나로 카운트 → Type-I error guaranteed $\leq \alpha$.

**Limitation**: $B$ finite이면 random 변동. $B$ 충분 크게 ($B \geq 1000$) 설정.

</details>

**문제 2** (심화): Kernel bandwidth $\sigma$ 선택이 test power에 미치는 영향을 RBF의 경우 구체적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

RBF $k(x, y) = \exp(-\|x-y\|^2 / 2\sigma^2)$:

- **$\sigma$ 작음**: 
  - $k \approx \mathbf{1}\{x \approx y\}$ (Kronecker-like).
  - 모든 pair의 kernel ≈ 0 (except $i = j$).
  - MMD² 작고 noisy → **low power**.

- **$\sigma$ 큼**:
  - $k \to 1$ for all pairs.
  - 분포 차이 정보 사라짐 → MMD² → 0 → **low power**.

- **$\sigma$ 중간**:
  - Scale of differences 포착 → **optimal power**.

**Optimal $\sigma$**: Maximum $\text{MMD}^2 / \sqrt{\text{Var}}$.

**Median heuristic**: $\sigma$ = median pairwise distance. Data scale에 자동 맞춰짐. 일반적으로 합리적이지만 optimal 아님.

**Multi-kernel MMD**: $k = \sum_l k_{\sigma_l}$ (여러 scale 혼합) — scale-invariant, MMD-GAN에서 채택.

</details>

**문제 3** (ML 연결): Model drift detection에서 MMD test의 **실무적 주의사항**은?

<details>
<summary>힌트 및 해설</summary>

**Setup**: Deployed model, reference data $\{x_i^{\text{ref}}\}$, new data $\{x_i^{\text{new}}\}$. Test if $p_{\text{new}} = p_{\text{ref}}$.

**주의사항**:

1. **Multiple testing**: 매일/매주 체크 → False positive 누적. Bonferroni correction 또는 FDR control.

2. **Feature engineering**: Raw input보다 **model의 intermediate feature** (embedding) 추천. Raw pixel drift는 meaningless하지만 semantic feature drift는 의미 있음.

3. **Sample size**: Reference는 large ($n \geq 10^4$), new data는 smaller이어도 OK. Unbalanced case에서 MMD² estimator 조정.

4. **Kernel bandwidth 일관성**: Reference에서 결정 후 고정 (매번 re-fit하지 말 것).

5. **Threshold setting**: 단순 $\alpha = 0.05$ 대신 historical drift scores의 분포에서 threshold.

6. **Scalability**: $n$ large → block-wise MMD 또는 Random Features MMD.

7. **Alternative**: Uncertainty-based drift detection (prediction entropy, ensemble disagreement) 보완.

**사례**: 자율주행·의료진단에서 input 분포 drift → model degradation → 재훈련 trigger.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. MMD의 샘플 추정량](./02-mmd-estimator.md) | [04. MMD-GAN과 생성모델 ▶](./04-mmd-gan.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
