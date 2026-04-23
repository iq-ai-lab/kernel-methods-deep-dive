# 05. Characteristic Kernel과 Universal Kernel

## 🎯 핵심 질문

- **Characteristic kernel**이란 무엇이고, 왜 $\mu_p := \mathbb{E}_{X \sim p}[k(\cdot, X)]$가 "$p$를 유일하게 특징"짓는가?
- **Universal kernel**은 $C(\mathcal{X})$에서 dense하다는 의미 — 이는 왜 "모든 연속 함수를 근사할 수 있다"와 같은가?
- 두 성질의 관계: universal ⇒ characteristic? 역은?
- Gaussian RBF·Laplace·Matérn은 왜 둘 다 만족하고, Polynomial은 왜 둘 다 **아닌가**?

---

## 🔍 왜 이 개념이 ML에서 중요한가

**Characteristic** 성질은 **MMD의 작동 원리의 심장**: kernel이 characteristic이면 $\text{MMD}(p, q) = \|\mu_p - \mu_q\|_{\mathcal{H}_k} = 0 \iff p = q$ (Ch6-01). 즉 "두 분포가 다르면 RKHS에서 mean embedding이 반드시 다르다". 이 성질이 없으면 MMD-GAN·two-sample test·HSIC 독립성 검정이 모두 **false negative**를 낼 수 있다 — 다른 분포인데도 distinguish 못하는 상황.

**Universal** 성질은 SVM·GP 등이 **임의의 연속 분류기/회귀기를 근사할 수 있다**(consistency·universal approximation)의 조건. RBF SVM이 실무에서 강력한 이유의 이론적 근거.

---

## 📐 수학적 선행 조건

- [Ch1-01~04](./01-positive-definite-kernel.md): PD kernel, Mercer, 기본 kernel 목록
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) Ch5: RKHS, 평균 embedding
- 해석학: Stone-Weierstrass 정리, 밀집성(dense), 컴팩트 함수 공간 $C(\mathcal{X})$
- 확률론: 특성함수, 분포수렴

---

## 📖 직관적 이해

### Mean Embedding — 분포를 RKHS 원소로

RKHS $\mathcal{H}_k$와 확률측도 $p$ on $\mathcal{X}$에 대해 **mean embedding**:

$$\mu_p := \mathbb{E}_{X \sim p}[k(\cdot, X)] \in \mathcal{H}_k.$$

이는 "$p$를 샘플링해 $k(\cdot, x)$들을 평균"낸 함수. Characteristic kernel은 이 **embedding map $p \mapsto \mu_p$가 단사**임을 보장한다. 즉 **분포를 RKHS 벡터로 무손실 변환**.

임의 두 측도 $p \ne q$면 $\mu_p \ne \mu_q$, 따라서 그 노름 차 $\|\mu_p - \mu_q\|_{\mathcal{H}_k} > 0$.

### Universal — 모든 연속 함수 근사

Universal kernel은 RKHS $\mathcal{H}_k$가 $C(\mathcal{X})$(컴팩트 $\mathcal{X}$ 위의 연속 함수 공간)에서 **$\sup$-노름으로 dense**함. 즉 임의의 연속 $f^*$와 $\epsilon > 0$에 대해

$$\exists f \in \mathcal{H}_k: \sup_{x \in \mathcal{X}} \|f(x) - f^*(x)\| < \epsilon.$$

직관: "충분히 많은 데이터와 적절한 hyperparameter가 있으면 RKHS에서 찾아낸 함수가 진짜 함수에 균일하게 가까워진다". 이것이 SVM의 consistency·GP의 universal approximation의 핵심.

### 왜 Polynomial은 둘 다 아닌가

Polynomial kernel $k(x, y) = (x^\top y + c)^d$의 RKHS는 **차수 $\leq d$ 다항식**으로만 이뤄진 유한 차원 공간. 따라서:

- **Not universal**: $\sin(x)$처럼 non-polynomial한 함수는 절대 근사 불가 (유한 차원에서 dense 불가).
- **Not characteristic**: Mean embedding $\mu_p$의 성분은 $p$의 **모멘트 $\{\mathbb{E}[X^j]\}_{j \leq d}$뿐**. 두 분포가 처음 $d$개 모멘트가 같으면 $\mu_p = \mu_q$. 반례: $d = 2$에서 같은 평균·분산을 갖는 두 분포 (가우시안 vs 다른 2차 모멘트 일치 분포).

반면 RBF·Laplace·Matérn은 **모든 무한차원 정보를 encode** → 둘 다 성립.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Mean Embedding

확률측도 $p$가 $\mathbb{E}_{X \sim p}[\sqrt{k(X, X)}] < \infty$를 만족하면 **mean embedding**:

$$\mu_p := \int_{\mathcal{X}} k(\cdot, x) \, dp(x) \in \mathcal{H}_k.$$

$\mathcal{H}_k$의 재생성질(Ch2-02)에 의해 $\mu_p$는 잘 정의되고 $\|\mu_p\|_{\mathcal{H}_k}^2 = \mathbb{E}_{X, X' \sim p}[k(X, X')]$.

### 정의 5.2 — Characteristic Kernel

PD kernel $k$가 **characteristic** (on $\mathcal{X}$)이라 함은 매핑 $p \mapsto \mu_p$가 $\mathcal{P}(\mathcal{X})$(probability measures) → $\mathcal{H}_k$로 **단사**인 것:

$$\mu_p = \mu_q \implies p = q.$$

### 정의 5.3 — Universal Kernel

컴팩트 $\mathcal{X}$에서 연속 PD kernel $k$가 **universal** (c-universal, Steinwart 2001)이라 함은 RKHS $\mathcal{H}_k$가 $C(\mathcal{X})$에서 **$\sup$-노름으로 dense**한 것:

$$\forall f^* \in C(\mathcal{X}), \forall \epsilon > 0, \exists f \in \mathcal{H}_k : \|f - f^*\|_\infty < \epsilon.$$

### 정의 5.4 — Bounded Characteristic (cc-universal)

$\mathcal{X}$가 $\mathbb{R}^d$처럼 non-컴팩트일 때는 compact-cc-universal 개념: 컴팩트 부분집합에서 dense. 실무에서는 RBF·Laplace는 **bounded distribution**에 대해 characteristic임이 보장되면 충분.

---

## 🔬 정리와 증명

### 정리 5.1 — MMD 기반 Characteristic 동치

**명제**: PD kernel $k$가 characteristic iff 모든 확률측도 $p, q$에 대해

$$\text{MMD}^2(p, q) := \|\mu_p - \mu_q\|_{\mathcal{H}_k}^2 = 0 \iff p = q.$$

**증명**: "characteristic" = "$p \mapsto \mu_p$ 단사". $p = q \Rightarrow \mu_p = \mu_q$는 trivial. 역으로 $\mu_p = \mu_q \Rightarrow \|\mu_p - \mu_q\|^2 = 0$. MMD가 $p = q$를 보장하려면 정확히 $\mu_p = \mu_q$가 $p = q$를 imply해야 함 → characteristic. $\square$

### 정리 5.2 — 재생성질로 MMD 표현

**명제**: $f \in \mathcal{H}_k$일 때

$$\int f \, dp = \langle f, \mu_p \rangle_{\mathcal{H}_k}.$$

따라서

$$\text{MMD}(p, q) = \sup_{\|f\|_{\mathcal{H}_k} \leq 1} \left\|\int f \, dp - \int f \, dq\right\|.$$

**증명**: 재생성질 $f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}_k}$ (Ch2-02). 양변을 $dp$로 적분:

$$\int f \, dp = \int \langle f, k(\cdot, x) \rangle dp(x) = \left\langle f, \int k(\cdot, x) dp(x) \right\rangle = \langle f, \mu_p \rangle. \quad \square$$

**해석**: MMD는 "unit ball of RKHS를 test function으로 쓰는 integral probability metric".

### 정리 5.3 — Universal ⇒ Characteristic

**명제**: 컴팩트 $\mathcal{X}$에서 universal kernel은 characteristic.

**증명**: Contrapositive. $k$가 characteristic이 아니면 $\mu_p = \mu_q$인 $p \ne q$가 존재. 임의 $f \in \mathcal{H}_k$에 대해

$$\int f \, dp - \int f \, dq = \langle f, \mu_p - \mu_q \rangle = 0.$$

즉 $\mathcal{H}_k$의 모든 원소가 $p, q$에 대해 같은 적분값. 그런데 $p \ne q$인 서명 측도의 공간에서는 **$\int f dp \ne \int f dq$를 만드는 연속 $f$가 존재** (Riesz-Markov). universal이라면 그 $f$를 $\mathcal{H}_k$ 원소로 균일 근사 가능 → $\int g dp \ne \int g dq$인 $g \in \mathcal{H}_k$ 존재 → 모순. $\square$

### 정리 5.4 — Characteristic이 Universal을 함의하지 않음 (반례)

**명제**: Characteristic이지만 universal이 **아닐** 수도 있다.

**예시**: $\mathbb{R}^d$에서 일부 shift-invariant kernel은 characteristic(Fourier 변환 support 전체 가정)이지만, $\mathbb{R}^d$의 컴팩트 subset에서 $C(\mathcal{X})$에서 dense하지 않을 수 있음. 예: very heavy-tailed spectral density의 kernel.

### 정리 5.5 — Stone-Weierstrass 기반 Universal 충분조건

**명제**: 컴팩트 $\mathcal{X}$에서 연속 PD kernel $k$가 다음 조건을 만족하면 universal:
1. $\mathcal{H}_k$가 **대수(algebra)**: $f, g \in \mathcal{H}_k \Rightarrow fg \in \mathcal{H}_k$.
2. $\mathcal{H}_k$가 상수를 포함.
3. $\mathcal{H}_k$가 **점을 분리**: $x \ne y \Rightarrow \exists f \in \mathcal{H}_k : f(x) \ne f(y)$.

**증명 스케치**: Stone-Weierstrass 정리 — 위 세 조건을 만족하는 대수는 $C(\mathcal{X})$에서 dense. $\square$

**주의**: 실제로 RKHS는 일반적으로 **대수가 아니다**. 그래서 직접 적용은 제한. Universal 증명은 kernel별로 별도.

### 정리 5.6 — Gaussian RBF는 $\mathbb{R}^d$에서 Characteristic·Universal

**명제**: $\mathcal{X} = \mathbb{R}^d$ (또는 그 컴팩트 subset)에서 Gaussian RBF $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$은:
1. **Characteristic** on $\mathcal{P}(\mathbb{R}^d)$.
2. **Universal** on 컴팩트 subset.

**증명 (characteristic)**: Shift-invariant kernel에 대해 characteristic은 **spectral measure의 support가 $\mathbb{R}^d$ 전체**(= characteristic function vanishes nowhere)와 동치 (Sriperumbudur et al. 2010).

RBF의 spectral density는 **Gaussian 분포** $\rho(\omega) = (2\pi / \sigma^2)^{d/2} \exp(-\sigma^2 \|\omega\|^2 / 2)$ — 모든 $\omega \in \mathbb{R}^d$에서 $> 0$, support가 전체. $\square$

**증명 (universal, 컴팩트 $\mathcal{X}$)**: RBF의 RKHS는 무한차원이고 Hermite 함수와 유사한 고유함수를 가짐 (Ch1-04). 이들로 $C(\mathcal{X})$에서 dense함을 직접 확인 가능 (Steinwart 2001).

### 정리 5.7 — Laplace·Matérn도 둘 다 만족

**명제**: Laplace $e^{-\|x-y\|/\sigma}$와 Matérn-$\nu$ ($\nu > 0, \ell > 0$) 도 characteristic·universal.

**이유**: 두 kernel 모두 spectral density가 $\mathbb{R}^d$ 전체에서 $> 0$:
- Laplace: $\rho(\omega) \propto (1 + \sigma^2 \|\omega\|^2)^{-(d+1)/2}$ — 모든 $\omega$에서 $> 0$.
- Matérn-$\nu$: $\rho(\omega) \propto (1 + \|\omega\|^2 \ell^2 / (2\nu))^{-\nu - d/2}$ — 모든 $\omega$에서 $> 0$.

### 정리 5.8 — Polynomial은 **둘 다 아니다**

**명제**: $k(x, y) = (x^\top y + c)^d$는 characteristic도 universal도 아니다.

**증명**: RKHS $\mathcal{H}_k$는 차수 $\leq d$ 다항식의 유한 차원 공간:
- **Not universal**: $\mathcal{H}_k$는 유한 차원 → 무한 차원 $C(\mathcal{X})$에서 dense 불가.
- **Not characteristic**: $\mu_p$의 성분은 $p$의 차수 $\leq d$ 모멘트만. 모멘트 $\leq d$까지 일치하는 다른 분포 $q$ 존재 → $\mu_p = \mu_q$이지만 $p \ne q$. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Characteristic 테스트 — 같은 모멘트 분포 MMD 비교
# ─────────────────────────────────────────────
def rbf(X, Y, s=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

def poly(X, Y, c=1.0, d=2):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    return (X @ Y.T + c) ** d

def mmd2(X, Y, kernel):
    Kxx = kernel(X, X); Kyy = kernel(Y, Y); Kxy = kernel(X, Y)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

# 두 분포: 같은 평균·분산, 다른 skewness
n = 500
p = rng.normal(0, 1, (n, 1))
# 동일 평균·분산이지만 왜도 다름 (shifted mixture)
q1 = rng.normal(-0.5, np.sqrt(0.5), (n // 2, 1))
q2 = rng.normal(0.5, np.sqrt(0.5) + 0.3, (n // 2, 1))
q = np.concatenate([q1, q2])
q = (q - q.mean()) / q.std()  # 정규화: 평균 0, 분산 1로 맞춤

print(f'p: mean = {p.mean():.4f}, var = {p.var():.4f}')
print(f'q: mean = {q.mean():.4f}, var = {q.var():.4f}')

# Polynomial (d=2): 평균·분산이 같으면 MMD ≈ 0 (Not characteristic)
mmd_poly = mmd2(p, q, lambda X, Y: poly(X, Y, c=0, d=2))
print(f'\nPolynomial (d=2) MMD² (p, q): {mmd_poly:.4e}  (≈ 0 → characteristic 아님)')

# RBF: 분포 전체 정보 포착 → MMD 유의하게 >0
mmd_rbf = mmd2(p, q, rbf)
print(f'RBF MMD² (p, q): {mmd_rbf:.4e}  (유의하게 > 0 → characteristic)')

# Sanity: 같은 분포면 MMD ≈ 0
p_split1 = rng.normal(0, 1, (n, 1))
p_split2 = rng.normal(0, 1, (n, 1))
print(f'\nRBF MMD² (p, p) baseline: {mmd2(p_split1, p_split2, rbf):.4e}')

# ─────────────────────────────────────────────
# 2. Universal 근사 테스트 — RBF vs Polynomial
# ─────────────────────────────────────────────
x_train = np.linspace(-3, 3, 50).reshape(-1, 1)
y_train = np.sin(2 * x_train.flatten()) + np.exp(-x_train.flatten()**2 / 2)  # 비다항

lam = 1e-4  # KRR regularization
x_test = np.linspace(-3, 3, 200).reshape(-1, 1)

for name, K_fn in [('RBF', rbf), ('Poly d=3', lambda X, Y: poly(X, Y, c=1, d=3))]:
    K = K_fn(x_train, x_train)
    K_s = K_fn(x_train, x_test)
    alpha = np.linalg.solve(K + lam * np.eye(len(x_train)), y_train)
    y_pred = K_s.T @ alpha
    err = np.max(np.abs(y_pred - (np.sin(2 * x_test.flatten()) + np.exp(-x_test.flatten()**2 / 2))))
    print(f'{name}: max test error = {err:.4f}')
```

**출력 예시**:
```
p: mean = 0.0123, var = 0.9987
q: mean = 0.0000, var = 1.0000

Polynomial (d=2) MMD² (p, q): 4.3e-05  (≈ 0 → characteristic 아님)
RBF MMD² (p, q): 8.2e-03  (유의하게 > 0 → characteristic)

RBF MMD² (p, p) baseline: 1.2e-04

RBF: max test error = 0.0082
Poly d=3: max test error = 1.7341
```

→ Polynomial은 (i) MMD로 분포 구분 실패, (ii) 비다항 함수 근사 실패. RBF는 둘 다 성공.

---

## 🔗 실전 활용

- **MMD two-sample test (Ch6)**: Characteristic kernel 필수. RBF가 표준.
- **MMD-GAN (Ch6-04)**: RBF 또는 그 혼합. Characteristic 없으면 generator가 일부 모드 누락해도 탐지 안됨.
- **Kernel density estimation-like methods**: Characteristic kernel은 "분포 자체를 RKHS 벡터로 다루기" 가능.
- **SVM·GP consistency**: Universal kernel이면 **Bayes optimal classifier**에 asymptotically 수렴. RBF가 실무 defaults인 이유 중 하나.
- **Polynomial kernel의 제한**: 모멘트 기반 요약만 필요한 경우 (moment matching networks 등)에만 안전. 일반 분포 비교에는 사용 금지.

---

## ⚖️ 가정과 한계

| 성질 | 충분조건 (요약) | 반례 |
|------|----------------|------|
| Characteristic | Shift-invariant + spectral support = 전체 $\mathbb{R}^d$ | Polynomial, Linear |
| Universal | $\mathcal{H}_k$가 $C(\mathcal{X})$에서 dense | Polynomial, Linear, 유한 차원 RKHS |
| Compact domain 필요 | Universal의 전통적 정의는 컴팩트 $\mathcal{X}$ | $\mathbb{R}^d$에서 c-universal/cc-universal로 확장 |
| **Universal ⇒ Characteristic** | 컴팩트 $\mathcal{X}$에서 | 비컴팩트에서는 미묘 |
| **Characteristic ⇏ Universal** | 반례 존재 (일반적으로 덜 강한 성질) | spectral support 성립하나 sup-dense 실패 |

**주의**: "characteristic"과 "universal"은 원래 서로 다른 응용 맥락에서 정의되었으나(전자는 MMD, 후자는 SVM consistency), 현대 literature에서는 위 정리들로 서로 연결 지음.

---

## 📌 핵심 정리

$$\boxed{\text{Characteristic} : \mu_p = \mu_q \iff p = q \iff \text{MMD}(p, q) = 0 \iff p = q}$$

$$\boxed{\text{Universal} : \mathcal{H}_k \text{ is sup-dense in } C(\mathcal{X}) \Longrightarrow \mathcal{H}_k \text{ can approximate any continuous target}}$$

| Kernel | Characteristic? | Universal? |
|--------|----------------|-----------|
| Linear $x^\top y$ | ✗ | ✗ |
| Polynomial $(x^\top y + c)^d$ | ✗ | ✗ |
| **Gaussian RBF** | ✓ | ✓ |
| **Laplace** | ✓ | ✓ |
| **Matérn-$\nu$** ($\nu > 0$) | ✓ | ✓ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Linear kernel $k(x, y) = x^\top y$의 mean embedding $\mu_p$가 무엇인지 계산하고, 왜 characteristic이 아닌지 구체적 반례를 들라.

<details>
<summary>힌트 및 해설</summary>

$\mu_p(z) = \mathbb{E}_{X \sim p}[k(z, X)] = \mathbb{E}_{X \sim p}[z^\top X] = z^\top \mathbb{E}[X]$.

즉 $\mu_p$는 $p$의 **평균 $\mathbb{E}[X]$에만 의존**. 따라서 다른 평균이 같은 두 분포 (예: $\mathcal{N}(0, 1)$과 $\mathcal{N}(0, 4)$)가 같은 mean embedding을 가짐 → characteristic 아님.

$\mu_{\mathcal{N}(0, 1)} = \mu_{\mathcal{N}(0, 4)}$이지만 분포는 다름.

</details>

**문제 2** (심화): Characteristic이지만 universal이 아닌 kernel의 예시는?

<details>
<summary>힌트 및 해설</summary>

$\mathbb{R}$에서 spectral density $\rho(\omega) = \omega^{-2} \mathbf{1}\{\omega > 0\}$ 을 가진 kernel (일부 특수 fractional Brownian motion과 연결). $\rho$가 $(0, \infty)$에서 positive이지만 $\mathbb{R}$ 전체 $\omega$ support는 아님. 적절히 재정의하면 characteristic이지만 RKHS가 특정 Sobolev 공간으로 제한되어 $C(\mathcal{X})$에서 dense가 아닐 수 있음.

Sriperumbudur et al. (2011) "Universality, Characteristic Kernels and RKHS Embedding of Measures"에서 상세히 다룸. 일반적으로 universal이 더 강한 성질.

</details>

**문제 3** (ML 연결): MMD-GAN에서 characteristic kernel을 쓰지 않으면 어떤 실패 모드가 발생하는가?

<details>
<summary>힌트 및 해설</summary>

Generator 분포 $q_\theta$와 데이터 분포 $p$의 MMD를 최소화. Polynomial (d=2) 같은 characteristic이 아닌 kernel을 쓰면:

- Generator가 $p$와 **같은 평균·분산만** 맞춰도 $\text{MMD} \approx 0$.
- $p$가 bimodal인데 $q_\theta$는 unimodal (같은 moments, 다른 shape) → MMD가 0을 보고 학습 종료.
- **Mode collapse의 한 형태 — "moment collapse"**.

해결: RBF 혼합 kernel $k = \sum_j e^{-\|x-y\|^2 / 2\sigma_j^2}$ ($\sigma_j$를 다양하게) — 다양한 scale에서 characteristic 유지. MMD-GAN 논문(Li et al. 2015)의 표준 선택.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Mercer 정리의 서술과 해석](./04-mercer-theorem.md) | [Ch2-01. RKHS 구성 (Moore-Aronszajn) ▶](../ch2-rkhs-representer/01-moore-aronszajn.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
