# 02. MMD의 샘플 추정량

## 🎯 핵심 질문

- MMD의 biased estimator $\widehat{\text{MMD}}_b^2 = \frac{1}{n^2} \sum k(x_i, x_j) - \frac{2}{nm} \sum k(x_i, y_j) + \frac{1}{m^2} \sum k(y_i, y_j)$는 왜 biased인가?
- **Unbiased estimator** (U-statistic form)는 어떻게 유도되는가?
- MMD² 추정량의 **수렴률** $O(1/\sqrt{n})$은 어떻게 증명되는가?
- 두 추정량의 trade-off — bias vs variance는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

MMD의 **샘플 기반 계산**이 실무 MMD의 핵심. Density 모르는 상황에서도 샘플만으로 두 분포의 차이를 측정 가능. **Unbiased U-statistic**은 이론적 정확성 (two-sample test의 null distribution 유도)과 **concentration bound** 제공. **Biased estimator**는 빠르지만 test 통계로 직접 쓰면 null에서도 positive bias → 주의. Ch6-03의 two-sample test와 Ch6-04의 MMD-GAN loss에서 이 estimator 선택이 중요하다.

---

## 📐 수학적 선행 조건

- [Ch6-01 MMD 정의](./01-mmd-definition.md)
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): **U-statistics**, Hoeffding 부등식, 편향 추정량
- 통계: 중심극한 정리, bootstrap

---

## 📖 직관적 이해

### Biased Estimator

Empirical means:

$$\hat{\mu}_p := \frac{1}{n} \sum_i k(\cdot, x_i), \quad \hat{\mu}_q := \frac{1}{m} \sum_j k(\cdot, y_j).$$

$\widehat{\text{MMD}}_b^2 := \|\hat{\mu}_p - \hat{\mu}_q\|^2$ 전개:

$$= \frac{1}{n^2} \sum_{i, j} k(x_i, x_j) - \frac{2}{nm} \sum_{i, j} k(x_i, y_j) + \frac{1}{m^2} \sum_{i, j} k(y_i, y_j).$$

**Biased**: $i = j$일 때 $k(x_i, x_i)$가 포함되어 MMD² 추정을 **overestimate** (특히 작은 $n, m$에서).

### Unbiased Estimator

$i \ne j$ 쌍만 사용:

$$\widehat{\text{MMD}}_u^2 := \frac{1}{n(n-1)} \sum_{i \ne j} k(x_i, x_j) - \frac{2}{nm} \sum_{i, j} k(x_i, y_j) + \frac{1}{m(m-1)} \sum_{i \ne j} k(y_i, y_j).$$

**U-statistic** form (Hoeffding's classic definition).

$\mathbb{E}[\widehat{\text{MMD}}_u^2] = \text{MMD}^2$ — **정확한 unbiased**.

### 수렴률

Concentration inequality (Gretton et al. 2012):

$$P(|\widehat{\text{MMD}}_u^2 - \text{MMD}^2| \geq t) \leq 2 e^{-n t^2 / (2 K^2)}$$

(bounded kernel $k \leq K^2$일 때). 따라서 $\widehat{\text{MMD}}_u^2 \to \text{MMD}^2$ with rate $O(1/\sqrt{n})$.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Biased MMD² Estimator

$$\widehat{\text{MMD}}_b^2 := \frac{1}{n^2} \sum_{i, j = 1}^n k(x_i, x_j) - \frac{2}{nm} \sum_{i, j} k(x_i, y_j) + \frac{1}{m^2} \sum_{i, j = 1}^m k(y_i, y_j).$$

### 정의 2.2 — Unbiased MMD² Estimator (U-statistic)

$$\widehat{\text{MMD}}_u^2 := \frac{1}{n(n-1)} \sum_{i \ne j} k(x_i, x_j) - \frac{2}{nm} \sum_{i, j} k(x_i, y_j) + \frac{1}{m(m-1)} \sum_{i \ne j} k(y_i, y_j).$$

### 정의 2.3 — Kernel Function for U-statistic

$$h(z_1, z_2) := k(x_1, x_2) + k(y_1, y_2) - k(x_1, y_2) - k(y_1, x_2), \quad z = (x, y).$$

$$\widehat{\text{MMD}}_u^2 = \frac{1}{n(n-1)} \sum_{i \ne j} h(z_i, z_j)$$

(같은 sample size $n = m$, paired setup).

---

## 🔬 정리와 증명

### 정리 2.1 — Biased Estimator의 Bias

**명제**: 

$$\mathbb{E}[\widehat{\text{MMD}}_b^2] = \text{MMD}^2 + \frac{1}{n} \mathbb{E}_p[k(X, X)] + \frac{1}{m} \mathbb{E}_q[k(Y, Y)] - \text{MMD}^2 \cdot (\text{correction terms})$$

— 정확히는 $\mathbb{E}[\widehat{\text{MMD}}_b^2] = \text{MMD}^2 + \frac{1}{n} (\mathbb{E}_p[k(X, X)] - \mathbb{E}_p[k(X, X')]) + \frac{1}{m}(\mathbb{E}_q[k(Y, Y)] - \mathbb{E}_q[k(Y, Y')])$.

**증명 스케치**: $i = j$ 포함하는 항들에서:

$$\frac{1}{n^2} \sum_{i, j} k(x_i, x_j) = \frac{1}{n^2} \sum_{i=j} k(x_i, x_i) + \frac{1}{n^2} \sum_{i \ne j} k(x_i, x_j) = \frac{1}{n} \mathbb{E}_p[k(X, X)] + \frac{n-1}{n} \mathbb{E}_p[k(X, X')].$$

MMD² 공식에서는 $\mathbb{E}_p[k(X, X')]$만 필요 → $\frac{1}{n} (\mathbb{E}[k(X, X)] - \mathbb{E}[k(X, X')])$의 overestimate. $\square$

**Bound kernel (RBF)**: $k \leq 1$, bias $= O(1/n)$.

### 정리 2.2 — Unbiased Estimator의 Unbiasedness

**명제**: $\mathbb{E}[\widehat{\text{MMD}}_u^2] = \text{MMD}^2$.

**증명**: 각 항의 기댓값 독립적으로:

$$\mathbb{E}\left[\frac{1}{n(n-1)} \sum_{i \ne j} k(x_i, x_j)\right] = \frac{1}{n(n-1)} \cdot n(n-1) \cdot \mathbb{E}_p[k(X, X')] = \mathbb{E}_p[k(X, X')].$$

비슷하게 다른 항. 합: MMD² 정의 그대로. $\square$

### 정리 2.3 — Concentration (Hoeffding)

**명제** (Gretton et al. 2012, Theorem 10): Bounded kernel $0 \leq k \leq K$. $n = m$. 임의 $t > 0$:

$$P(\widehat{\text{MMD}}_u^2 - \text{MMD}^2 > t) \leq \exp\left(-\frac{n t^2}{4 K^2}\right).$$

**증명 개요**: U-statistic에 Hoeffding's inequality 적용 (U-statistic is dependent, but variance bound is manageable). 세부는 Serfling (1980) 또는 Gretton 논문. $\square$

**따름**: $\widehat{\text{MMD}}_u^2 \to \text{MMD}^2$ at rate $O(1/\sqrt{n})$.

### 정리 2.4 — 비교 — Biased vs Unbiased

| | Biased $\widehat{\text{MMD}}_b^2$ | Unbiased $\widehat{\text{MMD}}_u^2$ |
|--|------|------|
| Expectation | $\text{MMD}^2 + O(1/n)$ | $\text{MMD}^2$ |
| Variance | 약간 작음 | 약간 큼 |
| Null ($p = q$) 값 | $O(1/n)$ positive | 0 (centered) |
| Always $\geq 0$? | Yes | 아님 (negative 가능 — $p = q$ 근처) |
| Computation | $O(n^2)$ | $O(n^2)$ |

**실무 선택**:
- **Test**: Unbiased (null distribution이 0 centered).
- **Training (MMD-GAN)**: Biased OK — variance 약간 작음, loss로 쓸 때 상수 shift 무관.

### 정리 2.5 — Block-wise 추정 (Scalability)

**명제**: $n$ 매우 크면 block estimator:

$$\widehat{\text{MMD}}_B^2 := \frac{1}{B} \sum_{b=1}^B \widehat{\text{MMD}}_{u, b}^2$$

$B$ blocks, 각 block 크기 $n_b = n/B$. $\mathbb{E}[\widehat{\text{MMD}}_B^2] = \text{MMD}^2$. Variance $O(1/B \cdot n_b^2) = O(B/n^2)$. 

**장점**: Memory $O(n_b^2)$만 필요 (full $O(n^2)$ 아님). $n = 10^5$에서도 실용.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def rbf(X, Y, s=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

def mmd2_biased(X, Y, kernel=rbf):
    Kxx = kernel(X, X)
    Kyy = kernel(Y, Y)
    Kxy = kernel(X, Y)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def mmd2_unbiased(X, Y, kernel=rbf):
    n, m = len(X), len(Y)
    Kxx = kernel(X, X)
    Kyy = kernel(Y, Y)
    Kxy = kernel(X, Y)
    # remove diagonal: sum of off-diagonal / n(n-1)
    term_xx = (Kxx.sum() - np.diag(Kxx).sum()) / (n * (n - 1))
    term_yy = (Kyy.sum() - np.diag(Kyy).sum()) / (m * (m - 1))
    term_xy = Kxy.sum() / (n * m)
    return term_xx + term_yy - 2 * term_xy

# ─────────────────────────────────────────────
# 1. Null 분포 (p = q)에서 biased vs unbiased
# ─────────────────────────────────────────────
n_trials = 500
sample_size = 100
biased_null = []
unbiased_null = []
for _ in range(n_trials):
    X = rng.standard_normal((sample_size, 2))
    Y = rng.standard_normal((sample_size, 2))
    biased_null.append(mmd2_biased(X, Y))
    unbiased_null.append(mmd2_unbiased(X, Y))

plt.figure(figsize=(10, 4))
plt.hist(biased_null, bins=30, alpha=0.5, label=f'Biased (mean={np.mean(biased_null):.4f})')
plt.hist(unbiased_null, bins=30, alpha=0.5, label=f'Unbiased (mean={np.mean(unbiased_null):.4f})')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('MMD²'); plt.ylabel('count')
plt.title('Null distribution (p = q): biased는 positive shift')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# Biased mean > 0 (positive bias), unbiased mean ≈ 0

# ─────────────────────────────────────────────
# 2. 수렴률 검증 — O(1/sqrt(n))
# ─────────────────────────────────────────────
true_mmd2 = None  # 정확한 MMD² 모름 — 추정치
sizes = [50, 100, 200, 500, 1000, 2000]
biased_stds = []
unbiased_stds = []
for n_size in sizes:
    biased_vals = []
    unbiased_vals = []
    for _ in range(100):
        X = rng.standard_normal((n_size, 2))
        Y = rng.standard_normal((n_size, 2)) + 0.5
        biased_vals.append(mmd2_biased(X, Y))
        unbiased_vals.append(mmd2_unbiased(X, Y))
    biased_stds.append(np.std(biased_vals))
    unbiased_stds.append(np.std(unbiased_vals))

plt.figure(figsize=(9, 4))
plt.loglog(sizes, biased_stds, 'o-', label='Biased std')
plt.loglog(sizes, unbiased_stds, 's-', label='Unbiased std')
# Reference: O(1/sqrt(n))
ref = biased_stds[0] * np.sqrt(sizes[0] / np.array(sizes))
plt.loglog(sizes, ref, 'k--', alpha=0.5, label='O(1/√n) reference')
plt.xlabel('sample size n'); plt.ylabel('std of MMD² estimator')
plt.title('MMD² 추정량의 $O(1/\\sqrt{n})$ 수렴')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 3. 분포 차이 vs MMD²
# ─────────────────────────────────────────────
shifts = np.linspace(0, 2, 20)
mmd2_vals = []
for shift in shifts:
    X = rng.standard_normal((500, 2))
    Y = rng.standard_normal((500, 2)) + shift
    mmd2_vals.append(mmd2_unbiased(X, Y))

plt.figure(figsize=(8, 4))
plt.plot(shifts, mmd2_vals, 'o-')
plt.xlabel('Mean shift $\\Delta$'); plt.ylabel('$\\widehat{MMD}^2_u$')
plt.title('분포 차이가 클수록 MMD² 증가')
plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. Block estimator
# ─────────────────────────────────────────────
def mmd2_blockwise(X, Y, n_blocks=10, kernel=rbf):
    n_total = len(X)
    block_size = n_total // n_blocks
    results = []
    for b in range(n_blocks):
        idx = slice(b * block_size, (b + 1) * block_size)
        results.append(mmd2_unbiased(X[idx], Y[idx], kernel))
    return np.mean(results), np.std(results) / np.sqrt(n_blocks)

X_large = rng.standard_normal((5000, 2))
Y_large = rng.standard_normal((5000, 2)) + 0.3

mmd_full = mmd2_unbiased(X_large, Y_large)
mmd_block, mmd_block_std = mmd2_blockwise(X_large, Y_large, n_blocks=20)
print(f'Full MMD² (n=5000): {mmd_full:.4e}')
print(f'Block-wise MMD² (20 blocks of 250): {mmd_block:.4e} ± {mmd_block_std:.4e}')
```

**출력 예시**:
```
Full MMD² (n=5000): 2.314e-02
Block-wise MMD² (20 blocks of 250): 2.289e-02 ± 3.12e-03
```

→ Null에서 biased는 positive shift, unbiased는 0 centered. $1/\sqrt{n}$ 수렴 확인.

---

## 🔗 실전 활용

- **Hypothesis testing (Ch6-03)**: Unbiased estimator 필수 — null distribution 정확 유도.
- **MMD-GAN training**: Biased OK (variance 약간 작음, constant bias는 상수 shift).
- **Block estimator**: Large $n$ ($10^5+$) 상황 (streaming data).
- **Random Features MMD**: $O(n D)$ via $\phi_{\text{RF}}$ (Ch7-02).

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| $O(n^2)$ 계산 | Block, Random Features로 완화 |
| Biased의 bias | Small $n$에서 의미 있음, 1/n 수준 |
| Unbiased의 negative 값 | 수학적 해석 없음 (MMD² ≥ 0) 하지만 random 변동 |
| **Kernel bandwidth 선택** | Median heuristic 일반적 |
| Power of test | Small $n$, small effect에서 test power 낮음 |

---

## 📌 핵심 정리

$$\boxed{\widehat{\text{MMD}}_u^2 = \frac{1}{n(n-1)} \sum_{i \ne j} k(x_i, x_j) - \frac{2}{nm} \sum_{i, j} k(x_i, y_j) + \frac{1}{m(m-1)} \sum_{i \ne j} k(y_i, y_j)}$$

$$\boxed{P(|\widehat{\text{MMD}}_u^2 - \text{MMD}^2| > t) \leq 2 \exp(-n t^2 / 4K^2)}$$

| Estimator | Bias | 수렴률 | 용도 |
|-----------|------|--------|------|
| Biased $\widehat{\text{MMD}}_b^2$ | $O(1/n)$ | $O(1/\sqrt{n})$ | Training (GAN loss) |
| Unbiased $\widehat{\text{MMD}}_u^2$ | 0 | $O(1/\sqrt{n})$ | Hypothesis testing |
| Block-wise | 0 | $O(1/\sqrt{B})$ | Large scale |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Unbiased MMD² estimator가 **음수 값**을 반환하는 상황은?

<details>
<summary>힌트 및 해설</summary>

$p = q$일 때, 기댓값은 0이지만 random variation으로:
- Sample $X$가 $Y$와 유사도가 $X \sim X'$ 자체 유사도보다 클 수 있음.
- → $-2 \sum k(x_i, y_j)/nm$가 $\sum k(x_i, x_j) / n(n-1) + \sum k(y_i, y_j) / m(m-1)$보다 크면 negative.

**수학적 해석**: $\text{MMD}^2 \geq 0$ (norm)이지만 estimator는 random 근사. Negative는 "null 근처의 fluctuation".

**실무적 처리**: Test statistic 계산 시 $\max(0, \widehat{\text{MMD}}_u^2)$ clip하거나 그대로 사용 후 permutation으로 p-value.

</details>

**문제 2** (심화): Biased estimator를 test statistic으로 쓰면 어떤 문제가 생기는가?

<details>
<summary>힌트 및 해설</summary>

1. **Null distribution shift**: $\mathbb{E}[\widehat{\text{MMD}}_b^2 \mid p = q] = O(1/n) > 0$ → null에서도 positive.

2. **Test power**: Biased이 positive로 shift되어, true null 상황에서 rejection rate가 target level $\alpha$보다 낮아질 수 있음.

3. **Permutation test 해결**: 바뀐 데이터에서도 같은 bias → **permutation으로 null 근사**하면 bias 취소. Biased statistic도 permutation으로 test 가능.

4. **Asymptotic**: $n \to \infty$에서 bias $\to 0$, 두 estimator 동등. 작은 $n$에서 분명한 차이.

**실무 권장**: Test에서는 unbiased or biased + permutation test (둘 다 같은 결과).

</details>

**문제 3** (ML 연결): Deep Learning generative model evaluation에서 MMD² estimator 선택의 고려사항은?

<details>
<summary>힌트 및 해설</summary>

**Use case**:
1. **Training loss**: Generator가 MMD² minimize → biased OK, computation efficient.
2. **Evaluation metric**: Generator 비교 → unbiased, unbiased 값이 0에 가까움이 better.

**고려사항**:
- **Deep feature MMD**: Raw pixel vs CNN feature space. Deep features의 MMD가 perceptually meaningful.
- **Kernel choice**: Multi-scale RBF (다양한 $\sigma$) 혼합 kernel이 더 robust.
- **Sample size**: Generator samples vs real samples 크기 매칭 중요. 일반적으로 1000~10000 samples로 MMD 계산.
- **Diversity penalty**: MMD만으로는 mode coverage 부족 — intra-class variance 별도 측정.

**대안 metrics**:
- **FID (Frechet Inception Distance)**: Inception feature의 Gaussian fit MMD-유사.
- **Precision & Recall**: Density ratio based.
- **MMD²**: Kernel 기반 fully nonparametric.

실무: FID가 이미지 GAN의 standard, MMD는 less common but theoretical 견고.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. MMD의 정의와 RKHS 해석](./01-mmd-definition.md) | [03. Two-Sample Test (Gretton et al. 2012) ▶](./03-two-sample-test.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
