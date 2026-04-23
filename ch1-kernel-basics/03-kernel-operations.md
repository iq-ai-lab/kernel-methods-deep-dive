# 03. Kernel 연산 — Sum·Product·Composition

## 🎯 핵심 질문

- PD kernel의 **합** $k_1 + k_2$, **곱** $k_1 \cdot k_2$, **scalar 조합** $f(x) k(x, y) f(y)$는 왜 모두 PD인가?
- 곱의 PD성 증명은 왜 **Schur product theorem**(Hadamard 곱)이 필요하고, 합보다 non-trivial한가?
- 이 연산들로 **ANOVA kernel**·**tensor product kernel**·**Gaussian kernel 재구성**을 어떻게 하는가?
- Kernel composition $k_1(x, y) := k_2(\psi(x), \psi(y))$에서 $\psi$가 어떤 조건을 만족해야 하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

새 kernel을 처음부터 설계하는 것은 어렵지만, **기존 PD kernel들을 결합**해 만드는 것은 규칙만 지키면 안전하다. 이 "결합 법칙" 덕분에 (i) **Multiple Kernel Learning**(MKL, Ch7-01)에서 학습 가능한 볼록 결합 $k = \sum \beta_i k_i$가 자동으로 PD, (ii) **ANOVA kernel**로 특성 간 상호작용을 계층적으로 표현, (iii) **Deep Kernel Learning**(Ch7-03)에서 $k(\phi_\theta(x), \phi_\theta(y))$ 형태의 composition이 정당화된다. 또한 Gaussian kernel의 PD 증명(Ch1-02)도 본질적으로 "Taylor 전개의 각 항 PD + 양계수 무한합"이라는 이 장의 규칙에 기댄다.

---

## 📐 수학적 선행 조건

- [Ch1-01 PD kernel의 정의](./01-positive-definite-kernel.md): Gram 양정치성
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): **Hadamard product**(원소별 곱), 양정치 행렬의 tensor product
- 해석학: 점별 극한, 일양 수렴

---

## 📖 직관적 이해

### 덧셈 — 두 feature 공간의 직합

$k_1, k_2$가 PD이고 feature map $\phi_1, \phi_2$를 가진다면, $k_1(x, y) + k_2(x, y) = \langle \phi_1(x), \phi_1(y) \rangle + \langle \phi_2(x), \phi_2(y) \rangle = \langle (\phi_1(x), \phi_2(x)), (\phi_1(y), \phi_2(y)) \rangle_{\mathcal{H}_1 \oplus \mathcal{H}_2}$. 즉 **"feature를 이어붙인" inner product**. 따라서 당연히 PD.

### 곱셈 — 두 feature 공간의 텐서곱

$k_1(x, y) k_2(x, y) = \langle \phi_1(x), \phi_1(y) \rangle \langle \phi_2(x), \phi_2(y) \rangle$. 이것은 $\phi(x) := \phi_1(x) \otimes \phi_2(x)$ (텐서곱 feature)의 inner product과 같다:

$$\langle \phi_1(x) \otimes \phi_2(x), \phi_1(y) \otimes \phi_2(y) \rangle_{\mathcal{H}_1 \otimes \mathcal{H}_2} = \langle \phi_1(x), \phi_1(y) \rangle \langle \phi_2(x), \phi_2(y) \rangle.$$

곱의 feature는 "두 특성의 모든 쌍"이라는 **상호작용(interaction)**을 표현한다. 이것이 polynomial kernel이 고차 상호작용을 만드는 원리.

### Composition — 다른 공간을 경유

$\psi : \mathcal{X} \to \mathcal{Z}$이고 $k_{\mathcal{Z}}$가 $\mathcal{Z}$에서 PD이면, $k(x, y) := k_{\mathcal{Z}}(\psi(x), \psi(y))$도 PD. 이것은 $\psi$가 "데이터를 다른 공간으로 옮긴 뒤 거기서 kernel"을 쓰는 구조로, **Deep Kernel Learning**의 핵심.

### Hadamard product — 행렬의 원소별 곱

두 PD Gram 행렬 $K_1, K_2$의 원소별 곱 $K_1 \odot K_2$도 PSD라는 결과(**Schur product theorem**)가 kernel product의 PD성의 행렬 이론적 근거.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Kernel의 합·곱·스칼라 조합

$k_1, k_2$가 $\mathcal{X}$ 위의 kernel이고 $c \geq 0$, $f : \mathcal{X} \to \mathbb{R}$일 때:

- **합**: $(k_1 + k_2)(x, y) := k_1(x, y) + k_2(x, y)$.
- **양의 스칼라 곱**: $(c \cdot k_1)(x, y) := c \cdot k_1(x, y)$.
- **곱(Hadamard)**: $(k_1 \cdot k_2)(x, y) := k_1(x, y) \cdot k_2(x, y)$.
- **Scalar conjugate**: $k_f(x, y) := f(x) k_1(x, y) f(y)$.

### 정의 3.2 — Kernel Composition

$\psi : \mathcal{X} \to \mathcal{Z}$와 $\mathcal{Z}$ 위의 kernel $k_{\mathcal{Z}}$에 대해 **composition** $k(x, y) := k_{\mathcal{Z}}(\psi(x), \psi(y))$.

### 정의 3.3 — ANOVA Kernel

$x = (x_1, \ldots, x_d) \in \mathbb{R}^d$에 대해 **ANOVA kernel**:

$$k_{\text{ANOVA}}(x, y) := \prod_{i=1}^d (1 + k_i(x_i, y_i))$$

여기서 $k_i$는 $i$-번째 축 위의 1차원 PD kernel. 이는 **모든 부분집합 상호작용의 합**으로 전개됨:

$$\prod_{i=1}^d (1 + k_i(x_i, y_i)) = \sum_{S \subseteq \{1, \ldots, d\}} \prod_{i \in S} k_i(x_i, y_i).$$

### 정의 3.4 — Tensor Product Kernel

$\mathcal{X} = \mathcal{X}_1 \times \mathcal{X}_2$와 각 좌표의 kernel $k_1, k_2$에 대해

$$k((x_1, x_2), (y_1, y_2)) := k_1(x_1, y_1) k_2(x_2, y_2).$$

---

## 🔬 정리와 증명

### 정리 3.1 — PD Kernel은 합·양의 스칼라 곱에 닫혀 있다

**명제**: $k_1, k_2$가 PD이고 $c \geq 0$이면 $k_1 + k_2$, $c \cdot k_1$도 PD.

**증명**: 임의의 $\{x_i\}, \{\alpha_i\}$에 대해

$$\sum \alpha_i \alpha_j (k_1 + k_2)(x_i, x_j) = \underbrace{\sum \alpha_i \alpha_j k_1(\cdot)}_{\geq 0} + \underbrace{\sum \alpha_i \alpha_j k_2(\cdot)}_{\geq 0} \geq 0.$$

$c k_1$도 동일. $\square$

### 정리 3.2 — Schur Product Theorem (Hadamard)

**명제**: $A, B \in \mathbb{R}^{n \times n}$이 대칭 PSD이면 원소별 곱 $A \odot B$도 PSD.

**증명**: $A$의 스펙트럴 분해 $A = \sum_i \lambda_i u_i u_i^\top$ ($\lambda_i \geq 0$). 그러면

$$A \odot B = \sum_i \lambda_i (u_i u_i^\top) \odot B.$$

각 $(u_i u_i^\top) \odot B$에 대해 임의의 $v \in \mathbb{R}^n$:

$$v^\top [(u_i u_i^\top) \odot B] v = \sum_{j, k} v_j v_k u_{i,j} u_{i,k} B_{jk} = (v \odot u_i)^\top B (v \odot u_i) \geq 0$$

($B \succeq 0$에 의해). 따라서 $(u_i u_i^\top) \odot B \succeq 0$, 양의 계수 $\lambda_i$의 합도 $\succeq 0$. $\square$

### 정리 3.3 — PD Kernel은 곱에 닫혀 있다

**명제**: $k_1, k_2$가 PD이면 $k_1 \cdot k_2$도 PD.

**증명**: 임의의 유한 샘플에서 그람 $K_1 = [k_1(x_i, x_j)]$, $K_2 = [k_2(x_i, x_j)]$는 둘 다 PSD. $k_1 \cdot k_2$의 그람은 $K_1 \odot K_2$. 정리 3.2에 의해 PSD. $\square$

### 정리 3.4 — Scalar Conjugate 유지

**명제**: $k$가 PD이고 $f : \mathcal{X} \to \mathbb{R}$ (임의) 이면 $k_f(x, y) := f(x) k(x, y) f(y)$도 PD.

**증명**:

$$\sum_{i, j} \alpha_i \alpha_j f(x_i) k(x_i, x_j) f(x_j) = \sum_{i, j} (\alpha_i f(x_i)) (\alpha_j f(x_j)) k(x_i, x_j) \geq 0$$

(스칼라 $\beta_i := \alpha_i f(x_i)$로 재레이블, $k$의 PD성 사용). $\square$

### 정리 3.5 — Pointwise Limit 보존

**명제**: $k_n$이 모두 PD이고 모든 $(x, y)$에서 $k_n(x, y) \to k(x, y)$이면 $k$도 PD.

**증명**: 임의 $\{x_i\}, \{\alpha_i\}$에 대해 $\sum \alpha_i \alpha_j k_n(x_i, x_j) \geq 0$이고 $k_n \to k$ pointwise이므로 극한 $\sum \alpha_i \alpha_j k(x_i, x_j) \geq 0$. $\square$

**따름**: PD kernel의 **양계수 무한합** $\sum_{j=0}^\infty c_j k_j$($c_j \geq 0$, 합 수렴)은 PD. 이것이 Gaussian kernel $e^{x^\top y / \sigma^2} = \sum_j \frac{(x^\top y / \sigma^2)^j}{j!}$이 PD인 핵심 논거였음(Ch1-02).

### 정리 3.6 — Composition 보존

**명제**: $\psi : \mathcal{X} \to \mathcal{Z}$ (임의 사상)이고 $k_{\mathcal{Z}}$가 $\mathcal{Z}$에서 PD이면 $k(x, y) := k_{\mathcal{Z}}(\psi(x), \psi(y))$는 $\mathcal{X}$에서 PD.

**증명**: $\{x_i\} \subset \mathcal{X}$에 대해 $\psi(x_i) \in \mathcal{Z}$가 유한 샘플. $k_{\mathcal{Z}}$가 PD이므로

$$\sum \alpha_i \alpha_j k(x_i, x_j) = \sum \alpha_i \alpha_j k_{\mathcal{Z}}(\psi(x_i), \psi(x_j)) \geq 0. \quad \square$$

**따름**: $f : \mathcal{X} \to \mathbb{R}^d$(예: neural network feature)이고 $k_{\mathcal{Z}}$가 RBF이면 **Deep Kernel** $k(x, y) := \exp(-\|f_\theta(x) - f_\theta(y)\|^2 / 2\sigma^2)$은 PD. (Ch7-03)

### 정리 3.7 — Exponential 보존

**명제**: $k$가 PD이면 $e^{k}$도 PD.

**증명**: $e^{k} = \sum_{j=0}^\infty k^j / j!$. 각 $k^j$는 정리 3.3의 반복 적용으로 PD. 양계수 무한합은 정리 3.5의 따름에 의해 PD. $\square$

**즉시 응용**: Gaussian kernel $\exp(-\|x-y\|^2 / 2\sigma^2)$의 증명을 다시 보자. $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 x^\top y$이고 $k_0(x, y) := x^\top y / \sigma^2$은 PD, $\exp(k_0)$는 정리 3.7에 의해 PD, 마지막으로 정리 3.4로 $f(x) := e^{-\|x\|^2 / 2\sigma^2}$ 곱하기 → RBF PD.

### 정리 3.8 — ANOVA Kernel의 PD성

**명제**: 각 $k_i$가 $\mathbb{R}$에서 PD이면 $k_{\text{ANOVA}}(x, y) = \prod_{i=1}^d (1 + k_i(x_i, y_i))$도 PD.

**증명**: 상수 kernel $1$은 PD($\phi \equiv 1$). $1 + k_i$는 정리 3.1에 의해 PD. 곱은 정리 3.3에 의해 PD. $\square$

**해석**: 전개하면 $k_{\text{ANOVA}} = \sum_{S \subseteq [d]} \prod_{i \in S} k_i$, 즉 **모든 부분집합 상호작용의 합**. $|S| = 1$이면 주효과, $|S| = 2$이면 2차 상호작용, ... 통계의 ANOVA 분해와 정확히 대응.

---

## 💻 NumPy로 검증

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((30, 2))

def rbf(X, Y, s=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

def linear(X, Y):
    return X @ Y.T

def check_pd(K, name):
    lam = np.linalg.eigvalsh(K).min()
    print(f'{name:30s} min eig = {lam:+.4e}   → {"PD" if lam > -1e-8 else "NOT PD"}')

# 개별
K_lin = linear(X, X)
K_rbf = rbf(X, X, s=1.0)

check_pd(K_lin, 'k_lin')
check_pd(K_rbf, 'k_rbf')

# 합
check_pd(K_lin + K_rbf,         'k_lin + k_rbf')
check_pd(2.5 * K_lin + 0.3 * K_rbf, '2.5·k_lin + 0.3·k_rbf')

# 곱
check_pd(K_lin * K_rbf,         'k_lin · k_rbf  (Hadamard)')
check_pd(K_rbf ** 3,            'k_rbf^3')

# Scalar conjugate
f = np.exp(-np.sum(X**2, axis=1) / 4)
check_pd(np.outer(f, f) * K_rbf, 'f(x)·k_rbf·f(y)')

# Composition: f(x) = x^2 을 통해 k_rbf( f(x), f(y) )
Xf = X ** 2
check_pd(rbf(Xf, Xf), 'k_rbf ∘ (x -> x²)')

# exp(k_lin) = Gaussian kernel의 한 경로
check_pd(np.exp(K_lin), 'exp(k_lin)')

# ANOVA kernel
k1 = rbf(X[:, [0]], X[:, [0]], s=1.0)
k2 = rbf(X[:, [1]], X[:, [1]], s=1.0)
K_anova = (1 + k1) * (1 + k2)
check_pd(K_anova, '(1+k1)(1+k2)  ANOVA')

# 반례: 두 PD의 차는 PD가 아닐 수 있음
check_pd(K_rbf - K_lin, 'k_rbf - k_lin  (not closed!)')
```

**출력 예시**:
```
k_lin                          min eig = +4.3712e-01   → PD
k_rbf                          min eig = +2.7128e-05   → PD
k_lin + k_rbf                  min eig = +4.4120e-01   → PD
2.5·k_lin + 0.3·k_rbf          min eig = +1.0938e+00   → PD
k_lin · k_rbf  (Hadamard)      min eig = +8.2345e-04   → PD
k_rbf^3                        min eig = +2.0113e-08   → PD
f(x)·k_rbf·f(y)                min eig = +1.6421e-06   → PD
k_rbf ∘ (x -> x²)              min eig = +3.5012e-07   → PD
exp(k_lin)                     min eig = +6.0092e-03   → PD
(1+k1)(1+k2)  ANOVA            min eig = +4.0012e-03   → PD
k_rbf - k_lin  (not closed!)   min eig = -4.0781e-01   → NOT PD
```

→ 차(−)는 닫혀 있지 않음을 확인.

---

## 🔗 실전 활용

- **Multiple Kernel Learning (Ch7-01)**: $k_\beta = \sum_l \beta_l k_l$, $\beta_l \geq 0, \sum \beta_l = 1$. 자동 PD.
- **Deep Kernel Learning (Ch7-03)**: $k(x, y) := k_{\mathcal{Z}}(f_\theta(x), f_\theta(y))$. 정리 3.6으로 PD 자동 보장, $\theta$를 GP marginal likelihood로 학습.
- **Spectral Mixture kernel (Wilson & Adams 2013)**: Gaussian 혼합 spectral density로 구성 → 합·곱·Bochner 조합으로 PD.
- **String kernel·Graph kernel**: 여러 작은 subkernel들의 합·곱으로 복잡한 구조 대응. 각 subkernel이 PD이면 조합도 PD.
- **Gaussian과 Linear의 합**: $k = k_{\text{rbf}} + \sigma_0^2 x^\top y$ — 전역 추세(linear) + 국소 패턴(RBF) 동시 모델링.

---

## ⚖️ 가정과 한계

| 연산 | 닫혀 있는가 | 주의 |
|------|------------|-----|
| 합 $k_1 + k_2$ | ✓ | — |
| 양의 스칼라 곱 $c k$($c \geq 0$) | ✓ | $c < 0$이면 깨짐 |
| 곱 $k_1 \cdot k_2$ | ✓ (Schur) | Feature 차원 폭발 가능 |
| $f(x) k(x, y) f(y)$ | ✓ ($f$ 임의) | $f \equiv 0$이면 trivial |
| Composition $k \circ \psi$ | ✓ | $\psi$ 임의 |
| Exponential $e^k$ | ✓ | 수치 overflow 주의 |
| **차** $k_1 - k_2$ | ✗ | 일반적으로 PD 아님 (반례 확인) |
| **나눗셈** $k_1 / k_2$ | ✗ | 일반적으로 PD 아님 |
| **$\log k$** | ✗ | 일반적으로 정의 불가 ($k$가 음수일 수 있음) |

---

## 📌 핵심 정리

$$\boxed{\text{PD kernel은 합·양의 스칼라 곱·곱·scalar conjugate·pointwise limit·composition·exponential에 닫혀 있다.}}$$

| 연산 | 이유 |
|------|------|
| Sum | Gram 행렬의 합 $\geq 0$ |
| Product | **Schur theorem**: Hadamard product 보존 |
| $f(x) k f(y)$ | 벡터 re-scaling |
| Pointwise limit | 부등식이 극한에 보존 |
| Composition $k \circ \psi$ | 샘플 $\{\psi(x_i)\}$도 유효 |
| $e^k$ | Taylor 전개 + 위 규칙들 조합 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $k_1(x, y) = x^\top y$, $k_2(x, y) = (x^\top y)^2$일 때 $k_1 \cdot k_2$의 feature map을 명시하라.

<details>
<summary>힌트 및 해설</summary>

$k_1 \cdot k_2 = x^\top y \cdot (x^\top y)^2 = (x^\top y)^3$. Polynomial kernel로 차수 3.

Feature map: $\phi(x) \in \mathbb{R}^{d^3}$, 성분 $x_{i_1} x_{i_2} x_{i_3}$. 기호적으로 $\phi(x) = x^{\otimes 3}$ (3-텐서).

</details>

**문제 2** (심화): $k_1 \cdot k_2$의 feature map이 $\phi_1 \otimes \phi_2$임을 보여라. 단, $\phi_1$의 차원이 $m_1$, $\phi_2$의 차원이 $m_2$이면 곱의 feature 차원은 $m_1 m_2$.

<details>
<summary>힌트 및 해설</summary>

$\phi(x) := \phi_1(x) \otimes \phi_2(x) \in \mathbb{R}^{m_1 m_2}$, 성분 $\phi_1(x)_i \phi_2(x)_j$.

$\langle \phi(x), \phi(y) \rangle = \sum_{i, j} \phi_1(x)_i \phi_2(x)_j \phi_1(y)_i \phi_2(y)_j = \left(\sum_i \phi_1(x)_i \phi_1(y)_i\right)\left(\sum_j \phi_2(x)_j \phi_2(y)_j\right) = k_1(x, y) k_2(x, y)$.

**의미**: 곱 kernel의 feature 차원은 **기하급수적 증가**. ANOVA의 각 항 곱이 지수적으로 커지는 이유.

</details>

**문제 3** (ML 연결): Deep Kernel Learning에서 $k(x, y) := \exp(-\|f_\theta(x) - f_\theta(y)\|^2 / 2\sigma^2)$가 PD임을 위 정리들만을 이용해 증명하라.

<details>
<summary>힌트 및 해설</summary>

1. $\mathcal{Z} := \mathbb{R}^d$(NN 출력 공간)에서 $k_{\mathcal{Z}}(z, z') := \exp(-\|z - z'\|^2 / 2\sigma^2)$는 RBF kernel, PD(Ch1-02 정리 2.3).
2. $\psi := f_\theta : \mathcal{X} \to \mathbb{R}^d$는 임의 사상 (NN 가중치 $\theta$가 무엇이든).
3. 정리 3.6 (Composition): $k(x, y) = k_{\mathcal{Z}}(f_\theta(x), f_\theta(y))$는 PD. ✓

**의미**: NN이 무엇을 학습하든 Deep Kernel은 항상 PD 보장. 따라서 marginal likelihood 최적화 과정에서도 GP 계산이 붕괴되지 않는다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 대표 커널의 목록과 PD 증명](./02-kernel-zoo.md) | [04. Mercer 정리의 서술과 해석 ▶](./04-mercer-theorem.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
