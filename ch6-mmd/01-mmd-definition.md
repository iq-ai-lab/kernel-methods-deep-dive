# 01. MMD의 정의와 RKHS 해석

## 🎯 핵심 질문

- **Mean embedding** $\mu_p = \mathbb{E}_{X \sim p}[k(\cdot, X)]$이 어떻게 "**분포를 RKHS 벡터로 embedding**"하는가?
- MMD $\text{MMD}(p, q) := \|\mu_p - \mu_q\|_{\mathcal{H}_k}$이 왜 **분포 간 거리**가 되는가?
- **Characteristic kernel** 하에서 $\text{MMD}(p, q) = 0 \iff p = q$의 완전 증명 (Gretton et al. 2012)은?
- MMD의 **Integral Probability Metric (IPM)** 해석 — unit ball의 test function으로서의 RKHS?

---

## 🔍 왜 이 개념이 ML에서 중요한가

MMD는 "**두 분포가 다른가?**"를 판정하는 가장 원칙적인 kernel-based metric. 기존 방법 (KL divergence, chi-squared)과 달리 (i) **분포의 density를 알 필요 없음** — 샘플만으로 계산, (ii) **high-dim·구조화 데이터**에서도 동작 (RBF kernel 있으면), (iii) **closed-form 추정량**과 **consistent convergence**. 이 성질이 (a) **Two-sample test** (Ch6-03), (b) **MMD-GAN** (Ch6-04), (c) **HSIC 독립성 검정** (Ch6-05)의 모두의 기반. 또한 distributional shift detection·generative model evaluation·domain adaptation의 핵심 도구.

---

## 📐 수학적 선행 조건

- [Ch1-05 Characteristic·Universal Kernel](../ch1-kernel-basics/05-characteristic-universal.md)
- [Ch2-02 재생성질](../ch2-rkhs-representer/02-reproducing-property.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률측도, Bochner integral
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Hilbert 공간의 norm, inner product

---

## 📖 직관적 이해

### Mean Embedding — "Distribution을 RKHS 점으로"

확률측도 $p$에 대해 **mean embedding**:

$$\mu_p := \mathbb{E}_{X \sim p}[k(\cdot, X)] = \int k(\cdot, x) dp(x) \in \mathcal{H}_k.$$

이것은 "분포의 kernel 가중 평균" — 각 $x$에서 RKHS 원소 $k(\cdot, x) = k_x$를 뽑아 $p$에 대해 평균낸 것.

**직관**: $\mu_p$는 $p$의 **모든 모멘트를 $\mathcal{H}_k$에 담은 벡터**. RBF kernel이면 $\mu_p$의 $n$-th 성분 (Mercer 전개)이 $p$의 $n$-th kernel moment.

### 재생성질로 기댓값 표현

재생성질 $f(x) = \langle f, k_x \rangle$와 Fubini로:

$$\mathbb{E}_{X \sim p}[f(X)] = \int f(x) dp(x) = \int \langle f, k(\cdot, x) \rangle dp(x) = \langle f, \mu_p \rangle_{\mathcal{H}_k}.$$

즉 **"$p$에 대한 $f$의 기댓값" = "RKHS에서 $f$와 $\mu_p$의 내적"**. $\mu_p$가 "기댓값을 계산하는 대리자".

### MMD = RKHS에서의 거리

$$\text{MMD}^2(p, q) := \|\mu_p - \mu_q\|_{\mathcal{H}_k}^2.$$

전개:

$$\|\mu_p - \mu_q\|^2 = \langle \mu_p, \mu_p \rangle - 2 \langle \mu_p, \mu_q \rangle + \langle \mu_q, \mu_q \rangle$$

$$= \mathbb{E}_{X, X' \sim p}[k(X, X')] - 2 \mathbb{E}_{X \sim p, Y \sim q}[k(X, Y)] + \mathbb{E}_{Y, Y' \sim q}[k(Y, Y')].$$

**각 항**:
- $\mathbb{E}[k(X, X')]$: $p$의 **self-similarity**.
- $\mathbb{E}[k(Y, Y')]$: $q$의 self-similarity.
- $-2 \mathbb{E}[k(X, Y)]$: $p$와 $q$ 사이의 **cross-similarity**.

**작은 MMD**: 두 분포의 self + cross similarities가 balance. **큰 MMD**: differ.

### IPM 해석

$$\text{MMD}(p, q) = \sup_{\|f\|_{\mathcal{H}_k} \leq 1} |\mathbb{E}_p[f] - \mathbb{E}_q[f]|.$$

즉 "unit RKHS ball에서의 test function으로 가장 큰 차이".

- **KL divergence**: density ratio 기반, 하나의 분포만 support 바뀌면 infinite.
- **Total variation**: $\sup$ over indicator functions — 매우 strict.
- **MMD**: RKHS unit ball만 — smooth function class의 제약, 실용적.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Mean Embedding

확률측도 $p$에 대해, $\int \sqrt{k(x, x)} dp(x) < \infty$ 조건에서

$$\mu_p := \int k(\cdot, x) dp(x) \in \mathcal{H}_k$$

(Bochner integral).

### 정의 1.2 — Maximum Mean Discrepancy

$$\text{MMD}(p, q; \mathcal{H}_k) := \|\mu_p - \mu_q\|_{\mathcal{H}_k}.$$

### 정의 1.3 — MMD²의 Expected Form

$$\text{MMD}^2(p, q) = \mathbb{E}_{X, X' \sim p}[k(X, X')] - 2 \mathbb{E}_{X \sim p, Y \sim q}[k(X, Y)] + \mathbb{E}_{Y, Y' \sim q}[k(Y, Y')].$$

### 정의 1.4 — IPM (Integral Probability Metric)

$$\text{IPM}_{\mathcal{F}}(p, q) := \sup_{f \in \mathcal{F}} |\mathbb{E}_p[f] - \mathbb{E}_q[f]|.$$

MMD = IPM with $\mathcal{F} = \{f : \|f\|_{\mathcal{H}_k} \leq 1\}$.

---

## 🔬 정리와 증명

### 정리 1.1 — Mean Embedding의 존재성

**명제**: $\int \sqrt{k(x, x)} dp(x) < \infty$이면 $\mu_p \in \mathcal{H}_k$ (Bochner integral의 convergence).

**증명**: $\|k(\cdot, x)\|_{\mathcal{H}_k}^2 = k(x, x)$. Bochner integral $\int k(\cdot, x) dp(x)$는 $\int \|k(\cdot, x)\| dp(x) = \int \sqrt{k(x, x)} dp(x) < \infty$이면 수렴. $\square$

**Bounded kernel** (예: RBF, $k(x, x) = 1$)이면 모든 $p$에 대해 $\mu_p$ 존재.

### 정리 1.2 — Mean Embedding의 기댓값 표현

**명제**: $f \in \mathcal{H}_k$에 대해

$$\mathbb{E}_{X \sim p}[f(X)] = \langle f, \mu_p \rangle_{\mathcal{H}_k}.$$

**증명**: 재생성질 $f(x) = \langle f, k(\cdot, x) \rangle$과 Bochner integral의 linearity:

$$\mathbb{E}_p[f(X)] = \int f(x) dp(x) = \int \langle f, k(\cdot, x) \rangle dp(x) = \langle f, \int k(\cdot, x) dp(x) \rangle = \langle f, \mu_p \rangle. \quad \square$$

### 정리 1.3 — MMD² 공식

**명제**: 정의 1.3의 공식.

**증명**:

$$\|\mu_p - \mu_q\|^2 = \langle \mu_p, \mu_p \rangle - 2 \langle \mu_p, \mu_q \rangle + \langle \mu_q, \mu_q \rangle.$$

각 항에 정리 1.2 적용 ($f = \mu_q$ 등):

$\langle \mu_p, \mu_p \rangle = \mathbb{E}_{X \sim p}[\mu_p(X)] = \mathbb{E}_{X \sim p}\mathbb{E}_{X' \sim p}[k(X, X')] = \mathbb{E}_{X, X' \sim p}[k(X, X')]$.

(Fubini.) 비슷하게 다른 항. $\square$

### 정리 1.4 — Characteristic Kernel ⇒ MMD = 0 iff p = q

**명제** (Gretton et al. 2012): $k$가 characteristic kernel (Ch1-05)이면

$$\text{MMD}(p, q) = 0 \iff p = q.$$

**증명**:

$(\Leftarrow)$: $p = q$이면 $\mu_p = \mu_q$ (mean embedding은 $p$의 함수), 따라서 $\text{MMD} = 0$.

$(\Rightarrow)$: $\text{MMD}(p, q) = 0 \Rightarrow \mu_p = \mu_q$. Characteristic 정의에 의해 $p \mapsto \mu_p$ 단사 → $p = q$. $\square$

**중요**: Characteristic 없으면 $\text{MMD} = 0$이 $p = q$를 함의하지 않음 → **false negative**. 예: polynomial kernel로 MMD 계산하면 두 분포의 처음 $d$개 모멘트만 비교.

### 정리 1.5 — IPM 해석

**명제**: $\text{MMD}(p, q) = \sup_{\|f\|_{\mathcal{H}_k} \leq 1} |\mathbb{E}_p[f] - \mathbb{E}_q[f]|$.

**증명**:

정리 1.2에서 $\mathbb{E}_p[f] - \mathbb{E}_q[f] = \langle f, \mu_p - \mu_q \rangle$.

Cauchy-Schwarz: $|\langle f, \mu_p - \mu_q \rangle| \leq \|f\| \cdot \|\mu_p - \mu_q\|$.

$\sup_{\|f\| \leq 1} = \|\mu_p - \mu_q\|$ (cauchy-Schwarz equality achieved by $f = (\mu_p - \mu_q) / \|\mu_p - \mu_q\|$).

$= \text{MMD}(p, q). \quad \square$

**해석**: MMD는 "RKHS의 $\|\cdot\|_{\mathcal{H}_k} \leq 1$ 제약 내에서 두 분포를 가장 잘 구별하는 test function"의 기댓값 차이.

### 정리 1.6 — Metric 성질

**명제**: Characteristic kernel 하에서 $\text{MMD}$는 **metric** (on probability measures):

1. $\text{MMD}(p, q) \geq 0$.
2. $\text{MMD}(p, q) = 0 \iff p = q$.
3. $\text{MMD}(p, q) = \text{MMD}(q, p)$.
4. $\text{MMD}(p, r) \leq \text{MMD}(p, q) + \text{MMD}(q, r)$ (triangle inequality).

**증명**: 

1~3 는 norm의 기본. 4는 RKHS norm의 triangle inequality. $\|\mu_p - \mu_r\| \leq \|\mu_p - \mu_q\| + \|\mu_q - \mu_r\|$. $\square$

### 정리 1.7 — 여러 Kernel에 대한 MMD의 Relative Strength

**명제**: $k_1 \leq k_2$ (둘 다 PD, pointwise) 이면 $\text{MMD}_{k_1}(p, q) \leq \text{MMD}_{k_2}(p, q)$ (일반적).

**해석**: Stronger (더 expressive) kernel = 더 sensitive MMD. 

실무: RBF 혼합 kernel $k = \sum_i e^{-\|x-y\|^2 / 2\sigma_i^2}$이 다양한 scale에서 차이 포착 → MMD-GAN의 standard.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. MMD² 계산 — 여러 분포 비교
# ─────────────────────────────────────────────
def rbf(X, Y, s=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

def mmd2_biased(X, Y, kernel=rbf):
    Kxx = kernel(X, X)
    Kyy = kernel(Y, Y)
    Kxy = kernel(X, Y)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

# 여러 시나리오
scenarios = {
    'Same dist': (rng.standard_normal((200, 2)), rng.standard_normal((200, 2))),
    'Mean shift': (rng.standard_normal((200, 2)), rng.standard_normal((200, 2)) + 1),
    'Variance shift': (rng.standard_normal((200, 2)), 2 * rng.standard_normal((200, 2))),
    'Different shape (uniform vs normal)': (rng.standard_normal((200, 2)), rng.uniform(-2, 2, (200, 2))),
}

print(f'{"Scenario":50s} {"MMD²":>10s}')
for name, (X, Y) in scenarios.items():
    m2 = mmd2_biased(X, Y)
    print(f'{name:50s} {m2:10.4f}')

# ─────────────────────────────────────────────
# 2. IPM 해석 — 최적 test function 찾기
# ─────────────────────────────────────────────
# f* = (μ_p - μ_q) / ‖μ_p - μ_q‖
# f*(x) = Σ α_i k(x, x_i) where α_p = 1/n, α_q = -1/m (normalized)
X = rng.standard_normal((200, 1))
Y = rng.standard_normal((200, 1)) + 0.5

x_grid = np.linspace(-4, 4, 200).reshape(-1, 1)
K_xg = rbf(x_grid, X)
K_yg = rbf(x_grid, Y)
# Unnormalized f(x) = (1/n) Σ k(x, x_i) - (1/m) Σ k(x, y_j)
f_unnorm = K_xg.mean(axis=1) - K_yg.mean(axis=1)

# RKHS norm
n_X, n_Y = len(X), len(Y)
K_XX = rbf(X, X); K_YY = rbf(Y, Y); K_XY = rbf(X, Y)
norm_sq = K_XX.mean() - 2*K_XY.mean() + K_YY.mean()
f_norm = f_unnorm / np.sqrt(norm_sq + 1e-12)

plt.figure(figsize=(10, 5))
plt.plot(x_grid, f_norm, 'b-', lw=2, label='optimal test function $f^*$')
plt.fill_between(x_grid.flatten(), 0, f_norm, alpha=0.3)
plt.scatter(X, np.zeros_like(X.flatten()) - 0.3, c='red', s=8, label='p samples', alpha=0.5)
plt.scatter(Y, np.zeros_like(Y.flatten()) - 0.4, c='blue', s=8, label='q samples', alpha=0.5)
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('x'); plt.title('MMD = sup of $E_p[f] - E_q[f]$ for $\\|f\\|_{\\mathcal{H}_k} \\leq 1$')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 3. Characteristic vs non-characteristic
# ─────────────────────────────────────────────
# Gaussian vs Laplace: 같은 mean, 다른 variance 
# 또는 같은 0, 1 moments, 다른 3rd moment
def poly2(X, Y, c=0, d=2):
    return (X @ Y.T + c) ** d

# 같은 mean, 같은 variance, 다른 skewness
p_samp = rng.normal(0, 1, (500, 1))
# Mixture: 비대칭 → 같은 var이지만 skew
q_samp = np.where(rng.uniform(0, 1, (500, 1)) < 0.5,
                    rng.normal(-0.5, np.sqrt(0.75), (500, 1)),
                    rng.normal(0.5, np.sqrt(0.75) + 0.2, (500, 1)))
q_samp = (q_samp - q_samp.mean()) / q_samp.std()  # normalize

print(f'\n같은 mean·var, 다른 skew:')
print(f'  RBF MMD² (characteristic):   {mmd2_biased(p_samp, q_samp, rbf):.4e}')
print(f'  Poly-2 MMD² (not character.): {mmd2_biased(p_samp, q_samp, poly2):.4e}')
print(f'  → Polynomial kernel은 모멘트 2까지만 비교 → 차이 감지 못함')
```

**출력 예시**:
```
Scenario                                               MMD²
Same dist                                              0.0021
Mean shift                                             0.1523
Variance shift                                         0.0823
Different shape (uniform vs normal)                    0.0234

같은 mean·var, 다른 skew:
  RBF MMD² (characteristic):   5.12e-03
  Poly-2 MMD² (not character.): 1.23e-05
  → Polynomial kernel은 모멘트 2까지만 비교 → 차이 감지 못함
```

→ RBF는 분포 차이를 잘 감지, polynomial은 특정 차이 놓침.

---

## 🔗 실전 활용

- **Two-sample test** (Ch6-03): $H_0: p = q$ vs $H_1: p \ne q$. MMD²이 test statistic.
- **Generative model evaluation**: GAN의 output 분포와 data 분포의 MMD 측정.
- **Domain adaptation**: Source와 target domain의 MMD 최소화 (feature alignment).
- **Distributional shift detection**: Deployment에서 input 분포가 training과 다른지 MMD monitoring.
- **Kernel two-sample test in sklearn**: 직접 구현 필요, scipy·KernelTest 패키지.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Characteristic kernel 필요 | Linear·polynomial은 분포 완전 구분 못함 |
| $O(n^2)$ 계산 | Gram 두 분포 각자 + cross — scalable approx 필요 |
| Power vs sample size | Small $n$에서는 noise > signal. Low-power. |
| **Kernel 선택 sensitivity** | Length-scale $\sigma$ 나쁘면 모든 pair가 similar 또는 dissimilar |
| Unbiased estimator 복잡 | Biased estimator는 sample overlap $i=j$ 포함 |

---

## 📌 핵심 정리

$$\boxed{\mu_p = \int k(\cdot, x) dp(x) \in \mathcal{H}_k, \quad \text{MMD}(p, q) = \|\mu_p - \mu_q\|_{\mathcal{H}_k}}$$

$$\boxed{\text{MMD}(p, q) = \sup_{\|f\|_{\mathcal{H}_k} \leq 1} |\mathbb{E}_p[f] - \mathbb{E}_q[f]| \quad \text{(IPM)}}$$

| 성질 | 의미 |
|------|------|
| Mean embedding | 분포를 RKHS 벡터로 encode |
| $\mathbb{E}_p[f] = \langle f, \mu_p \rangle$ | 재생성질로 기댓값 = 내적 |
| $\text{MMD}^2 = \mathbb{E}[k(X, X')] - 2\mathbb{E}[k(X, Y)] + \mathbb{E}[k(Y, Y')]$ | 샘플 기반 계산 |
| Characteristic kernel | $\text{MMD} = 0 \iff p = q$ |
| Metric | 확률측도 공간에서 유효 metric |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MMD²의 각 항 $\mathbb{E}[k(X, X')]$, $-2\mathbb{E}[k(X, Y)]$, $\mathbb{E}[k(Y, Y')]$의 직관적 역할을 설명하라.

<details>
<summary>힌트 및 해설</summary>

- **$\mathbb{E}[k(X, X')]$**: $p$에서 두 독립 샘플의 평균 유사도. $p$가 concentrated (narrow distribution)이면 크고, spread out이면 작음.

- **$\mathbb{E}[k(Y, Y')]$**: 동일하게 $q$.

- **$-2 \mathbb{E}[k(X, Y)]$**: $p$ 샘플과 $q$ 샘플의 cross-similarity. 두 분포가 similar하면 $k(X, Y)$ 커서 이 항이 더 negative → MMD² 감소.

**합의 기하**: "각 분포 내의 self-similarity 합 - 2 × cross similarity".
- $p = q$: Self와 cross가 같음 → self + self - 2 cross = 0.
- $p \ne q$: Cross < self → MMD² > 0.

</details>

**문제 2** (심화): MMD와 Wasserstein distance의 관계는?

<details>
<summary>힌트 및 해설</summary>

둘 다 IPM으로 쓸 수 있음:

- **MMD**: $\text{IPM}(\{f: \|f\|_{\mathcal{H}_k} \leq 1\})$. Smooth test function.
- **Wasserstein-1**: $\text{IPM}(\{f: \text{Lip}(f) \leq 1\})$. Lipschitz test function (Kantorovich-Rubinstein).

**차이**:
- Wasserstein: **geometric** (distance를 직접 integrate), sensitive to "mass movement".
- MMD: **functional** (RKHS에서 similarity), sensitive to "어떤 kernel function으로 구별 가능한가".

**계산**:
- Wasserstein: $O(n^3)$ (optimal transport).
- MMD: $O(n^2)$ (Gram).

**Applications**:
- WGAN: Wasserstein (Lipschitz gradient).
- MMD-GAN: MMD.

**Duality**: MMD는 kernel space에서의 L2 distance, Wasserstein은 probability space의 transport metric.

</details>

**문제 3** (ML 연결): GAN의 발전에서 "MMD를 왜 adversarial loss 대체로 사용하는가"?

<details>
<summary>힌트 및 해설</summary>

**Standard GAN의 문제**:
- Adversarial minimax → 불안정한 training (mode collapse, vanishing gradient).
- Discriminator가 "이기면" generator gradient 0.

**MMD-GAN (Li et al. 2015, Dziugaite et al. 2015)**:
- Loss: $\text{MMD}^2(p_{\text{data}}, p_{\text{gen}})$. Generator는 이것을 minimize.
- **장점 1**: 단일 objective (non-adversarial) → stable training.
- **장점 2**: Characteristic kernel이면 이론적으로 $\text{MMD} = 0 \iff$ 분포 매칭.
- **장점 3**: Gradient가 항상 informative (kernel value 0 아닌 한).

**단점**:
- Kernel bandwidth 선택 sensitive.
- Mode coverage가 adversarial GAN보다 약할 수 있음 (특정 모드만 매칭).
- High-dim 이미지 (pixel space)에서 kernel 효과 감소 → deep feature space에서 MMD 계산하는 hybrid.

**현대**: Pure MMD-GAN은 점유율 낮지만, "feature-space MMD" (with deep encoder)은 여러 곳에서.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch5-04. Kernel k-Means](../ch5-krr-kpca/04-kernel-kmeans.md) | [02. MMD의 샘플 추정량 ▶](./02-mmd-estimator.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
