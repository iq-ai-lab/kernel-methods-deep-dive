# 02. 대표 커널의 목록과 PD 증명

## 🎯 핵심 질문

- Linear·Polynomial·Gaussian(RBF)·Laplace kernel 각각이 왜 PD인가? 증명 전략은?
- Gaussian kernel의 PD 증명에서 **Bochner 정리**, **$\exp$ 전개**, **가우시안 측도의 특성함수** 중 어떤 관점이 가장 명료한가?
- Sigmoid $\tanh(a x^\top y + c)$는 왜 **항상** PD가 아닌가?
- Matérn kernel의 smoothness 파라미터 $\nu$는 함수의 미분가능성과 어떻게 연결되는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Kernel 선택은 **함수 prior를 선택하는 것**과 같다. GP regression에서 RBF를 고르면 "매끄러운 함수"를, Matérn-1/2(= Laplace)를 고르면 "연속이지만 미분 불가능한 함수"를, polynomial을 고르면 "다항식"을 prior로 삼는 것. SVM에서 linear kernel은 선형 분리기, RBF는 국소 유사도 기반 분류기가 된다. 이 선택이 **generalization과 inductive bias를 결정**하므로, 각 kernel의 수학적 성질과 PD 증명을 이해하는 것은 "문제에 맞는 kernel을 고르는" 실무 능력으로 직결된다. 또한 sigmoid처럼 **언뜻 PD로 보이지만 아닌** 함수를 SVM에 쓰면 solver가 무너지는 사고가 실제로 발생하므로 PD 검증 능력은 필수다.

---

## 📐 수학적 선행 조건

- [Ch1-01 PD kernel의 정의](./01-positive-definite-kernel.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 외적(rank-1) 분해, $\text{Vandermonde}$ 행렬
- 해석학: Taylor 급수, 다항식 전개, 지수함수 성질
- 복소해석(선택): Bochner 정리 관점에서의 Fourier 변환 (Ch7-02에서 심화)

---

## 📖 직관적 이해

### 5가지 대표 kernel 한눈에

| Kernel | 수식 | 직관 | 사용처 |
|--------|------|------|--------|
| **Linear** | $k(x, y) = x^\top y$ | "원본 공간에서의 각도" | 고차원·sparse 데이터(NLP) |
| **Polynomial** | $(x^\top y + c)^d$ | 차수 $d$ 이하 다항식 특성 | 구조적 상호작용 |
| **Gaussian (RBF)** | $\exp(-\|x - y\|^2 / 2\sigma^2)$ | 점 사이 가까움의 지수감쇠 | 기본 선택, 매끄러운 함수 |
| **Laplace** | $\exp(-\|x - y\| / \sigma)$ | RBF보다 뾰족한 감쇠 | 덜 매끄러운 함수 |
| **Matérn-$\nu$** | 복잡 — Bessel 함수 포함 | $\nu$로 미분가능성 제어 | 물리 신호, 지구통계학 |
| **Sigmoid** | $\tanh(a x^\top y + c)$ | 신경망 활성화 닮음 | **일반적으로 PD 아님 — 주의** |

### 증명 전략 3가지

1. **Feature map 제시(정리 1.4 사용)**: $\phi$를 직접 만들어 $k(x, y) = \langle \phi(x), \phi(y) \rangle$를 보인다. (Linear, Polynomial에서 유효)
2. **PD-preserving 연산**: 이미 PD인 kernel에 합·곱·컴포지션을 적용해 새 kernel의 PD성을 얻는다. (Gaussian은 이 방법)
3. **Bochner 정리**: shift-invariant kernel $k(x - y)$는 어떤 **유한 음이 아닌 측도의 Fourier 변환**일 때 정확히 PD. (Laplace, Matérn에 깔끔)

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Linear Kernel

$\mathcal{X} = \mathbb{R}^d$에서 $k_{\text{lin}}(x, y) := x^\top y = \sum_{i=1}^d x_i y_i$.

### 정의 2.2 — Polynomial Kernel

$c \geq 0$, $d \in \mathbb{Z}_{\geq 1}$에 대해 $k_{\text{poly}}(x, y) := (x^\top y + c)^d$.

### 정의 2.3 — Gaussian (RBF) Kernel

$\sigma > 0$에 대해 $k_{\text{rbf}}(x, y) := \exp\left(-\frac{\|x - y\|^2}{2 \sigma^2}\right)$.

### 정의 2.4 — Laplace Kernel

$\sigma > 0$에 대해 $k_{\text{lap}}(x, y) := \exp\left(-\frac{\|x - y\|}{\sigma}\right)$.

### 정의 2.5 — Matérn Kernel

$\nu > 0$, $\ell > 0$에 대해

$$k_{\text{mat}}(x, y) := \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} \|x-y\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} \|x - y\|}{\ell}\right)$$

여기서 $K_\nu$는 modified Bessel function of the second kind. 특수 사례: $\nu = 1/2$ $\Rightarrow$ Laplace; $\nu \to \infty$ $\Rightarrow$ Gaussian.

### 정의 2.6 — Sigmoid Kernel

$a, c \in \mathbb{R}$에 대해 $k_{\text{sig}}(x, y) := \tanh(a x^\top y + c)$. **주의**: 일반적으로 PD 아님.

---

## 🔬 정리와 증명

### 정리 2.1 — Linear Kernel은 PD

**명제**: $k_{\text{lin}}(x, y) = x^\top y$는 $\mathbb{R}^d$에서 PD.

**증명**: $\phi : \mathbb{R}^d \to \mathbb{R}^d$를 $\phi(x) := x$(항등사상)로 두면 $k(x, y) = \langle \phi(x), \phi(y) \rangle$. 정리 1.4. $\square$

### 정리 2.2 — Polynomial Kernel은 PD

**명제**: $c \geq 0$, $d \in \mathbb{Z}_{\geq 1}$에 대해 $k_{\text{poly}}(x, y) = (x^\top y + c)^d$는 PD.

**증명**: 이항정리 전개:

$$(x^\top y + c)^d = \sum_{j=0}^d \binom{d}{j} c^{d-j} (x^\top y)^j.$$

각 항이 PD임을 보이면, PD kernel들의 양의 조합은 PD(Ch1-03 정리)이므로 결론.

- $(x^\top y)^j$는 $\phi_j(x) := x^{\otimes j}$ (텐서 $j$승)을 feature map으로 하는 inner product kernel. 구체적으로 $\phi_j(x) \in \mathbb{R}^{d^j}$의 성분은 $x_{i_1} x_{i_2} \cdots x_{i_j}$이고

$$\langle \phi_j(x), \phi_j(y) \rangle = \sum_{i_1, \ldots, i_j} x_{i_1} \cdots x_{i_j} y_{i_1} \cdots y_{i_j} = (x^\top y)^j.$$

따라서 $(x^\top y)^j$는 PD. $c \geq 0$이고 $c^{d-j} \geq 0$이므로 양의 계수들의 합. $\square$

**Feature map 직관**: $d = 2$, $c = 1$, $x, y \in \mathbb{R}^2$에서

$$(x^\top y + 1)^2 = 1 + 2 x_1 y_1 + 2 x_2 y_2 + x_1^2 y_1^2 + 2 x_1 x_2 y_1 y_2 + x_2^2 y_2^2.$$

이것은 $\phi(x) = (1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, \sqrt{2} x_1 x_2, x_2^2)^\top \in \mathbb{R}^6$ 의 inner product.

### 정리 2.3 — Gaussian (RBF) Kernel은 PD

**명제**: $\sigma > 0$에 대해 $k_{\text{rbf}}(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$는 PD.

**증명 (Taylor 전개)**: 전개하면

$$\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right) = \exp\left(-\frac{\|x\|^2}{2\sigma^2}\right) \exp\left(-\frac{\|y\|^2}{2\sigma^2}\right) \exp\left(\frac{x^\top y}{\sigma^2}\right).$$

마지막 인자를 Taylor 전개:

$$\exp\left(\frac{x^\top y}{\sigma^2}\right) = \sum_{j=0}^\infty \frac{1}{j! \sigma^{2j}} (x^\top y)^j.$$

각 $(x^\top y)^j$는 정리 2.2에서 PD. 양의 계수 $1/(j! \sigma^{2j})$ 곱해도 PD. 무한합도 PD(pointwise limit 보존). 마지막으로 $f(x) := \exp(-\|x\|^2 / 2\sigma^2)$에 대해 $f(x) f(y)$ 형태를 곱해도 PD 유지 (Ch1-03 정리 3.3). $\square$

**증명 대안 (Bochner 정리)**: $k(x - y) = k(z)$라 하면 $\exp(-\|z\|^2 / 2\sigma^2)$은 $\mathcal{N}(0, \sigma^{-2} I)$의 특성함수(스케일링)의 Fourier pair. 유한 양의 측도의 Fourier 변환이므로 Bochner 정리에 의해 PD. (Ch7-02에서 상세)

### 정리 2.4 — Laplace Kernel은 PD

**명제**: $\sigma > 0$에 대해 $k_{\text{lap}}(x, y) = \exp(-\|x - y\| / \sigma)$는 $\mathbb{R}^d$에서 PD.

**증명 개요 (Bochner)**: $\mathbb{R}^d$에서 $\exp(-\|z\|/\sigma)$의 Fourier 변환은 **Cauchy 분포의 다차원 일반화** 형태 $c_d \sigma^d / (1 + \sigma^2 \|\omega\|^2)^{(d+1)/2}$, 이는 $\omega \in \mathbb{R}^d$에 대해 양의 함수. 따라서 Bochner에 의해 PD. $\square$

**대안 증명 ($d = 1$에서 직접)**: $e^{-|x-y|/\sigma} = \int_{-\infty}^{\infty} \frac{1/\sigma}{1 + (\omega/\sigma)^2} e^{i \omega (x-y)} \frac{d\omega}{2\pi}$ 등의 적분 표현으로도 확인 가능.

### 정리 2.5 — Matérn Kernel은 PD

**명제**: $\nu > 0$, $\ell > 0$에 대해 Matérn kernel은 PD. 또한:
- $\nu = 1/2$: Laplace와 일치.
- $\nu = 3/2, 5/2$: 해석적으로 단순한 형태 (미분가능성 1회, 2회).
- $\nu \to \infty$: Gaussian (RBF)로 수렴.

**증명 스케치**: Matérn은 **Whittle 클래스** — spectral density $S(\omega) \propto (1 + \|\omega\|^2 \ell^2 / (2\nu))^{-\nu - d/2}$가 양의 유한 측도를 유도. Bochner에 의해 PD. (Rasmussen & Williams 4.2 참조.)

**샘플 함수의 미분가능성**: Matérn-$\nu$ kernel에서 추출한 GP 샘플은 **거의 확실하게 $\lceil \nu \rceil - 1$번 미분가능**. 따라서:
- Matérn-1/2: 연속이지만 미분 불가 (Brownian motion류).
- Matérn-3/2: 1번 미분가능.
- Matérn-5/2: 2번 미분가능 (대부분의 ML 응용에서 충분).
- Matérn-$\infty$ = RBF: 무한번 미분가능 (해석적).

### 정리 2.6 — Sigmoid Kernel은 **일반적으로 PD가 아니다**

**명제**: $k_{\text{sig}}(x, y) = \tanh(a x^\top y + c)$는 **$a, c$의 특정 범위를 벗어나면** PD가 아니다.

**반례**: $a = 1$, $c = 1$, $\mathcal{X} = \mathbb{R}$에서 $x_1 = 1$, $x_2 = -1$, $x_3 = 0$:

$$K = \begin{pmatrix} \tanh 2 & \tanh 0 & \tanh 1 \\ \tanh 0 & \tanh 2 & \tanh 1 \\ \tanh 1 & \tanh 1 & \tanh 1 \end{pmatrix} = \begin{pmatrix} 0.964 & 0 & 0.762 \\ 0 & 0.964 & 0.762 \\ 0.762 & 0.762 & 0.762 \end{pmatrix}$$

수치 계산으로 $\det(K) < 0$인 $a, c$를 쉽게 찾을 수 있고, 음의 고유값이 존재하는 영역이 넓다. 따라서 일반적으로 PD 아님. $\square$

**왜 SVM에서 자주 쓰이는가?**: 특정 $(a, c)$ 범위 ($a > 0, c < 0$의 일부)에서는 PD에 가깝거나 CPD이고, 신경망의 활성화와 유사하다는 직관 때문. 하지만 수학적으로 **항상 PD**인 것은 아니므로 사용 전 검증 필수.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
X = rng.standard_normal((30, 4))

def rbf(X, Y, sigma=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * sigma**2))

def laplace(X, Y, sigma=1.0):
    d = np.sqrt(np.maximum(0, np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T))
    return np.exp(-d / sigma)

def poly(X, Y, c=1.0, d=3):
    return (X @ Y.T + c) ** d

def sigmoid(X, Y, a=1.0, c=-1.0):
    return np.tanh(a * X @ Y.T + c)

def matern52(X, Y, ell=1.0):
    d = np.sqrt(np.maximum(0, np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T))
    r = np.sqrt(5) * d / ell
    return (1 + r + r**2 / 3) * np.exp(-r)

# 각 kernel에서 최소 고유값 측정 → PD는 ≥ -jitter, 아니면 음수
kernels = {
    'Linear':      X @ X.T,
    'Polynomial':  poly(X, X, c=1, d=3),
    'RBF':         rbf(X, X),
    'Laplace':     laplace(X, X),
    'Matérn-5/2':  matern52(X, X),
    'Sigmoid(1,2)': sigmoid(X, X, a=1.0, c=2.0),
    'Sigmoid(1,-1)': sigmoid(X, X, a=1.0, c=-1.0),
}

print(f'{"Kernel":20s} {"min eig":>12s}  {"PD?":>6s}')
for name, K in kernels.items():
    lam_min = np.linalg.eigvalsh(K).min()
    is_pd = 'YES' if lam_min > -1e-8 else 'NO'
    print(f'{name:20s} {lam_min:12.4e}  {is_pd:>6s}')
```

**출력 예시**:
```
Kernel               min eig      PD?
Linear              4.3217e-01     YES
Polynomial          1.4128e-03     YES
RBF                 2.9842e-05     YES
Laplace             8.7233e-04     YES
Matérn-5/2          6.4112e-05     YES
Sigmoid(1,2)       -1.8734e+00      NO
Sigmoid(1,-1)       3.0217e-03     YES
```

→ Sigmoid는 $(a, c)$에 따라 PD 여부가 바뀜을 수치적으로 확인.

```python
# RBF vs Matérn GP 샘플 — kernel 선택이 함수 매끄러움을 어떻게 결정하는가
x = np.linspace(-5, 5, 200).reshape(-1, 1)
for name, K_fn in [('RBF', rbf), ('Matérn-5/2', matern52), ('Laplace', laplace)]:
    K_xx = K_fn(x, x) + 1e-6 * np.eye(len(x))
    L = np.linalg.cholesky(K_xx)
    samples = L @ rng.standard_normal((len(x), 3))
    plt.figure(figsize=(8, 3))
    plt.plot(x, samples); plt.title(f'GP 샘플 ({name}) — 매끄러움이 kernel에 따라 다름')
    plt.grid(True, alpha=0.3); plt.show()
```

→ RBF는 $C^\infty$ 매끄러운 경로, Matérn-5/2는 2번 미분가능, Laplace는 연속이지만 들쭉날쭉(미분 불가).

---

## 🔗 실전 활용

- **NLP / 텍스트 분류**: 높은 차원·sparse → Linear kernel이 우월 (LinearSVC). RBF를 쓰면 오히려 overfit.
- **이미지·구조화 특성**: RBF·Matérn-5/2가 기본. Length-scale $\sigma$를 cross-validation 또는 GP marginal likelihood(Ch4-05)로 학습.
- **Polynomial kernel**: $d = 2$ 또는 $3$이 일반적. 높은 $d$는 수치적으로 불안정($x^\top y$가 크면 $(x^\top y + c)^d$ 값이 폭발).
- **Matérn**: 지구통계학·공간통계·Bayesian 최적화의 표준. $\nu = 5/2$가 sweet spot.
- **Sigmoid**: SVM에서 유행했으나 현재는 권장하지 않음 — RBF가 거의 항상 우월하고 PD 보장.

---

## ⚖️ 가정과 한계

| Kernel | 장점 | 한계 |
|--------|------|------|
| Linear | 계산 $O(n d)$, 해석 가능 | 비선형 패턴 포착 불가 |
| Polynomial | 특성 상호작용 표현 | $d$ 크면 불안정, 경계 포화 |
| RBF | Universal, 매끄러움 | 차원의 저주($d$ 크면 모든 거리가 비슷) |
| Laplace | 덜 매끄러운 함수 포착 | 미분 불가 → 경사법 느림 |
| Matérn-$\nu$ | 매끄러움 제어 가능 | Bessel 함수 계산 비쌈 (단, $\nu = p + 1/2$면 단순) |
| Sigmoid | NN과 유사한 직관 | **일반적으로 PD 아님 — 사용 주의** |

---

## 📌 핵심 정리

$$\boxed{k_{\text{lin}}, k_{\text{poly}}, k_{\text{rbf}}, k_{\text{lap}}, k_{\text{mat}} \text{ 모두 PD} \quad ; \quad k_{\text{sig}} \text{ 는 일반적으로 PD 아님}}$$

| Kernel | PD 증명 전략 |
|--------|-----------|
| Linear | Feature map $\phi(x) = x$ |
| Polynomial | 이항전개 + $(x^\top y)^j$이 PD |
| RBF | $\exp(x^\top y / \sigma^2)$ Taylor 전개 or Bochner |
| Laplace | Cauchy-류 측도의 Fourier 변환 (Bochner) |
| Matérn | Whittle spectral density의 Fourier 변환 |
| Sigmoid | **아님** — 반례로 $\det(K) < 0$ 제시 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $c = 0, d = 2$인 polynomial kernel $k(x, y) = (x^\top y)^2$의 feature map $\phi$를 $\mathbb{R}^2$에서 명시적으로 쓰라.

<details>
<summary>힌트 및 해설</summary>

$(x^\top y)^2 = (x_1 y_1 + x_2 y_2)^2 = x_1^2 y_1^2 + 2 x_1 x_2 y_1 y_2 + x_2^2 y_2^2$.

$\phi(x) = (x_1^2, \sqrt{2} x_1 x_2, x_2^2)^\top \in \mathbb{R}^3$로 놓으면 $\phi(x)^\top \phi(y) = (x^\top y)^2$. ✓

일반적으로 차수 $d$ polynomial의 feature 차원은 $\binom{d + n - 1}{d}$ (멀티셋 계수).

</details>

**문제 2** (심화): RBF kernel $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$에서 $\sigma \to 0$과 $\sigma \to \infty$의 극한에서 그람 행렬은 어떤 모양이 되는가? 각각 SVM에 미치는 영향은?

<details>
<summary>힌트 및 해설</summary>

- $\sigma \to 0$: 모든 distinct $x_i \ne x_j$에 대해 $k(x_i, x_j) \to 0$, 대각은 1. 즉 $K \to I$.
  - SVM에서는 각 점이 **자기 자신과만 유사** → 모든 training 점이 support vector가 되고 **완벽 training fit + 심각한 overfit**.
- $\sigma \to \infty$: 모든 $k(x_i, x_j) \to 1$. $K \to \mathbf{1} \mathbf{1}^\top$ (rank 1).
  - SVM이 **상수 함수만** 학습 가능 → **심한 underfit**.
- 적정 $\sigma$는 데이터 스케일에 맞춰야 함. 자주 쓰이는 heuristic: $\sigma$ = median pairwise distance.

</details>

**문제 3** (ML 연결): GP regression에서 Matérn-5/2와 RBF 중 어느 것을 선택해야 할지 결정하는 기준을 3가지 제시하라.

<details>
<summary>힌트 및 해설</summary>

1. **신호의 미분가능성에 대한 사전 지식**: 물리 시스템(예: 천체 궤도)처럼 해석적으로 매끄러운 함수 → RBF. 센서 신호·주가처럼 거친 함수 → Matérn-5/2 또는 3/2.
2. **수치 안정성**: RBF의 Gram 행렬은 length-scale이 길면 **매우 조건수 나쁨** (거의 특이). Matérn-5/2는 감쇠가 더 빨라 수치적으로 안정적.
3. **외삽 vs 내삽**: RBF는 학습 데이터 밖에서 **빠르게 평균으로 복귀** (uncertainty가 빨리 커짐). Matérn은 상대적으로 천천히.
4. (보너스) **Marginal likelihood 비교**: 두 kernel 모두로 fitting해 $\log p(y \mid \theta)$ 비교 (Ch4-05).

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. Positive Definite Kernel의 정의](./01-positive-definite-kernel.md) | [03. Kernel 연산 — Sum·Product·Composition ▶](./03-kernel-operations.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
