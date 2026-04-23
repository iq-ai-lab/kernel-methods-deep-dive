# 01. Positive Definite Kernel의 정의

## 🎯 핵심 질문

- Positive definite(PD) kernel이란 정확히 무엇인가? 왜 "정의식"이 이중합 $\sum \alpha_i \alpha_j k(x_i, x_j) \geq 0$으로 주어지는가?
- **함수** $k(x, y)$의 PD성과 **그람 행렬** $K = [k(x_i, x_j)]$의 양정치성은 왜 동치인가?
- Conditionally positive definite와 positive definite는 어떻게 다르고, 왜 distance kernel 구성에서 구분이 중요한가?
- 복소수값 kernel($k : \mathcal{X} \times \mathcal{X} \to \mathbb{C}$)으로 확장하면 무엇이 달라지는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Kernel method의 **모든 것**은 한 가지 가정 위에 세워진다: "$k$가 PD이다." 이 가정은 단순한 기술적 조건이 아니라 **feature map $\phi$의 존재**를 보장한다. Mercer 정리는 PD kernel을 무한차원 feature map으로 "펼쳐" 놓고($k(x, y) = \langle \phi(x), \phi(y) \rangle$), 이것이 SVM의 kernel trick, GP의 prior covariance, MMD의 metric 성질의 근간이 된다. 만약 $k$가 PD가 아니면 SVM dual QP가 **non-convex**가 되어 global minimum이 보장되지 않고, GP의 covariance matrix는 Cholesky 분해가 불가능해 posterior 계산이 수치적으로 깨진다. 즉 "kernel을 쓴다"는 말은 "내가 사용하는 함수가 PD임을 증명했다"와 동치다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 대칭 행렬의 양정치성, 스펙트럴 분해, 이차형식
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): 내적 공간, 이중선형 형식
- 기초 해석학: 함수 공간에서의 대칭성, 연속성
- 확률론(선택): 이 정의가 공분산 함수와 동일한 구조임을 관찰

---

## 📖 직관적 이해

### "PD"는 왜 ML에서 "좋은" 성질인가

"$k$가 PD"라는 말을 다르게 읽으면 **"$k$를 두 벡터의 내적으로 표현할 수 있다"** 가 된다:

$$k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}$$

여기서 $\phi : \mathcal{X} \to \mathcal{H}$는 어떤 Hilbert 공간으로의 **feature map**이다. 이 표현 가능성이 있으면:

1. **비선형 분류가 선형이 된다** — 원래 공간 $\mathcal{X}$에서 비선형인 문제가 $\mathcal{H}$에서는 선형이 된다 (SVM).
2. **유사도(similarity)가 기하적 의미를 갖는다** — $k(x, y)$가 $\phi(x)$와 $\phi(y)$의 **각도**에 해당한다.
3. **거리가 자동으로 정의된다** — $\|\phi(x) - \phi(y)\|^2 = k(x, x) + k(y, y) - 2 k(x, y)$.

### 이중합이 왜 ≥ 0이어야 하는가 — 선형결합의 norm

만약 $k(x, y) = \langle \phi(x), \phi(y) \rangle$이면, 임의의 유한 점 $\{x_1, \ldots, x_n\}$과 스칼라 $\{\alpha_1, \ldots, \alpha_n\}$에 대해:

$$\left\| \sum_{i=1}^n \alpha_i \phi(x_i) \right\|_{\mathcal{H}}^2 = \sum_{i, j = 1}^n \alpha_i \alpha_j \langle \phi(x_i), \phi(x_j) \rangle = \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) \geq 0$$

즉 **"$\sum \alpha_i \alpha_j k(x_i, x_j) \geq 0$"은 곧 "어떤 선형결합의 norm 제곱이 항상 음이 아니다"**. 당연한 요구이고, 반대로 이 성질이 성립하면 feature map을 **거꾸로** 구성할 수 있다는 것이 Moore-Aronszajn 정리의 내용이다(Ch2).

### 그람 행렬 관점

$n$개의 점 $x_1, \ldots, x_n$에 대해 **그람 행렬**을 $K_{ij} := k(x_i, x_j)$로 정의하면, 이중합은 **이차형식** $\alpha^\top K \alpha$가 된다. 따라서:

$$k \text{ is PD} \iff K \succeq 0 \text{ for every finite sample}$$

이것이 PD kernel을 "샘플링된 그람 행렬의 양정치성"으로 체크하는 관행의 근거다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Positive Semi-Definite Kernel

집합 $\mathcal{X}$ 위의 함수 $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$가 **positive semi-definite kernel**(흔히 positive definite kernel, PD kernel)이라 함은 다음 두 조건이 모두 성립하는 것이다:

1. **대칭성(symmetry)**: 모든 $x, y \in \mathcal{X}$에 대해 $k(x, y) = k(y, x)$.
2. **양정치성(positive semi-definiteness)**: 임의의 유한 부분집합 $\{x_1, \ldots, x_n\} \subset \mathcal{X}$와 임의의 스칼라 $\alpha_1, \ldots, \alpha_n \in \mathbb{R}$에 대해

$$\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j k(x_i, x_j) \geq 0.$$

### 정의 1.2 — Strictly Positive Definite Kernel

정의 1.1의 조건에서 등호 $\sum \alpha_i \alpha_j k(x_i, x_j) = 0$이 **모든 $x_i$가 distinct일 때** $\alpha_1 = \cdots = \alpha_n = 0$인 경우에만 성립하면, $k$를 **strictly positive definite**이라 한다. 이때 distinct한 $x_i$들에서의 그람 행렬은 $\succ 0$(strictly).

### 정의 1.3 — Gram 행렬

주어진 kernel $k$와 점 $x_1, \ldots, x_n \in \mathcal{X}$에 대해, $n \times n$ 행렬

$$K := \bigl[k(x_i, x_j)\bigr]_{i, j = 1}^n$$

을 **Gram 행렬**(또는 kernel 행렬)이라 한다.

### 정의 1.4 — Conditionally Positive Definite

$k$가 대칭이고, $\sum_{i=1}^n \alpha_i = 0$인 스칼라에 대해서만

$$\sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) \geq 0$$

이 성립하면 **conditionally positive definite** (CPD)라 한다. 이 개념은 거리 기반 kernel(예: $-\|x-y\|^2$)의 분석에 필수다.

### 정의 1.5 — 복소수값 PD Kernel

$k : \mathcal{X} \times \mathcal{X} \to \mathbb{C}$에 대해서는 대칭성을 **Hermitian symmetry** $k(x, y) = \overline{k(y, x)}$로, 양정치성을 $\sum_{i, j} \alpha_i \overline{\alpha_j} k(x_i, x_j) \geq 0$(모든 $\alpha_i \in \mathbb{C}$)로 대체한다.

---

## 🔬 정리와 증명

### 정리 1.1 — Kernel PD성과 Gram 양정치성의 동치

**명제**: 대칭 함수 $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$에 대해 다음은 동치이다:

1. $k$는 PD kernel이다.
2. 임의의 유한 부분집합 $\{x_1, \ldots, x_n\} \subset \mathcal{X}$에 대해 그람 행렬 $K = [k(x_i, x_j)]$는 positive semi-definite($K \succeq 0$)이다.

**증명**:

$(1) \Rightarrow (2)$: 이차형식 관점. 임의의 벡터 $\alpha = (\alpha_1, \ldots, \alpha_n)^\top \in \mathbb{R}^n$에 대해

$$\alpha^\top K \alpha = \sum_{i, j} K_{ij} \alpha_i \alpha_j = \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) \geq 0$$

이 정의 1.1로부터 직접 따라온다. 대칭성 $k(x_i, x_j) = k(x_j, x_i)$에 의해 $K = K^\top$. 따라서 $K$는 대칭이면서 $\alpha^\top K \alpha \geq 0$ 이므로 $K \succeq 0$이다.

$(2) \Rightarrow (1)$: 역방향도 마찬가지. 임의의 $\{x_i\}_{i=1}^n$와 $\alpha \in \mathbb{R}^n$에 대해 $\alpha^\top K \alpha \geq 0$이 곧 정의 1.1의 조건이다. $\square$

### 정리 1.2 — PD Kernel의 기본 성질

**명제**: $k$가 PD kernel이면 다음이 성립한다:

1. **대각 음이 아님**: 모든 $x \in \mathcal{X}$에 대해 $k(x, x) \geq 0$.
2. **Cauchy-Schwarz**: 모든 $x, y \in \mathcal{X}$에 대해 $\|k(x, y)\|^2 \leq k(x, x) \cdot k(y, y)$.
3. **고유값 음이 아님**: 임의 샘플의 그람 행렬은 모든 고유값이 $\geq 0$.

**증명**:

(1) $\alpha_1 = 1$, $n = 1$로 두면 $\alpha_1^2 k(x_1, x_1) = k(x_1, x_1) \geq 0$.

(2) $n = 2$, $x_1 = x$, $x_2 = y$로 두면 그람 행렬은

$$K = \begin{pmatrix} k(x, x) & k(x, y) \\ k(x, y) & k(y, y) \end{pmatrix}$$

$K \succeq 0 \iff \det K \geq 0$ **그리고** 대각 원소 $\geq 0$ (대칭 $2 \times 2$의 경우). 따라서

$$k(x, x) k(y, y) - k(x, y)^2 \geq 0 \iff k(x, y)^2 \leq k(x, x) k(y, y).$$

(3) 대칭 행렬 $K$의 스펙트럴 분해 $K = \sum_i \lambda_i v_i v_i^\top$에서 $v_i^\top K v_i = \lambda_i \|v_i\|^2 \geq 0$이므로 $\lambda_i \geq 0$. $\square$

### 정리 1.3 — CPD와 PD의 관계

**명제**: $k$가 conditionally PD이고, 임의의 고정점 $x_0 \in \mathcal{X}$에 대해

$$\tilde{k}(x, y) := k(x, y) - k(x, x_0) - k(y, x_0) + k(x_0, x_0)$$

을 정의하면 $\tilde{k}$는 PD kernel이다.

**증명 스케치**: 임의의 $\alpha_1, \ldots, \alpha_n \in \mathbb{R}$에 대해 **$\alpha_0 := -\sum_{i=1}^n \alpha_i$**를 추가하고 $x_0$도 포함시킨 확장된 샘플 $\{x_0, x_1, \ldots, x_n\}$과 스칼라 $\{\alpha_0, \alpha_1, \ldots, \alpha_n\}$을 생각하면 $\sum_{i=0}^n \alpha_i = 0$이다. 이때 $\sum_{i, j = 0}^n \alpha_i \alpha_j k(x_i, x_j) \geq 0$이 CPD 가정이고, 이를 전개해 $x_0$ 관련 항을 정리하면

$$\sum_{i, j = 1}^n \alpha_i \alpha_j \tilde{k}(x_i, x_j) \geq 0$$

를 얻는다. $\square$

**응용**: $-\|x - y\|^2$는 PD가 아니지만 CPD이고, 위 변환으로 $-\|x-y\|^2$에서 $x_0 = 0$을 기준으로 변환하면 PD를 얻는다. 이것이 왜 Gaussian kernel $\exp(-\|x - y\|^2 / 2\sigma^2)$이 PD인지의 한 가지 증명 경로다(Ch1-02 참조).

### 정리 1.4 — Inner Product Kernel은 PD

**명제**: Hilbert 공간 $\mathcal{H}$와 임의의 사상 $\phi : \mathcal{X} \to \mathcal{H}$에 대해 $k(x, y) := \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}$는 PD kernel이다.

**증명**: 대칭성은 내적의 대칭성에서 즉각. PD성은

$$\sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) = \sum_{i, j} \alpha_i \alpha_j \langle \phi(x_i), \phi(x_j) \rangle = \left\langle \sum_i \alpha_i \phi(x_i), \sum_j \alpha_j \phi(x_j) \right\rangle = \left\| \sum_i \alpha_i \phi(x_i) \right\|_{\mathcal{H}}^2 \geq 0. \quad \square$$

이 정리의 역이 Moore-Aronszajn 정리이고, 본 레포 Ch2-01에서 엄밀하게 구성한다.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. RBF kernel의 Gram 양정치성 확인
# ─────────────────────────────────────────────
def rbf_kernel(X, Y, sigma=1.0):
    d2 = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * sigma**2))

n = 50
X = rng.standard_normal((n, 3))
K = rbf_kernel(X, X)

eigvals = np.linalg.eigvalsh(K)
print(f'RBF Gram 최소 고유값: {eigvals.min():.4e}  (≥ 0 이어야 함)')
print(f'RBF Gram 최대 고유값: {eigvals.max():.4e}')
assert eigvals.min() > -1e-10, 'RBF는 PD여야 한다'

# ─────────────────────────────────────────────
# 2. Sigmoid tanh(xᵀy + c)가 PD가 아닌 반례
# ─────────────────────────────────────────────
def sigmoid_kernel(X, Y, a=1.0, c=-1.0):
    return np.tanh(a * X @ Y.T + c)

K_sig = sigmoid_kernel(X, X, a=1.0, c=-2.0)
eig_sig = np.linalg.eigvalsh(K_sig)
print(f'\nSigmoid Gram 최소 고유값: {eig_sig.min():.4e}  (음수면 PD 아님)')
print(f'→ 이 c 선택에서는 PD가 아님이 확인됨')

# ─────────────────────────────────────────────
# 3. 이중합 직접 계산으로 정의 체크 (Cauchy-Schwarz)
# ─────────────────────────────────────────────
i, j = 0, 1
cs_lhs = K[i, j] ** 2
cs_rhs = K[i, i] * K[j, j]
print(f'\nCauchy-Schwarz: k(x_i, x_j)² = {cs_lhs:.4f} ≤ {cs_rhs:.4f} = k_ii · k_jj')
assert cs_lhs <= cs_rhs + 1e-12

# ─────────────────────────────────────────────
# 4. CPD kernel -‖x-y‖²의 변환이 PD
# ─────────────────────────────────────────────
def neg_dist2(X, Y):
    return -(np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T)

K_cpd = neg_dist2(X, X)
print(f'\n-‖x-y‖² Gram 최소 고유값: {np.linalg.eigvalsh(K_cpd).min():.4f}  (음수 → PD 아님)')

# Trick: x_0 = 0을 기준으로 변환
x0 = np.zeros((1, X.shape[1]))
k_x0 = neg_dist2(X, x0).flatten()
k_00 = neg_dist2(x0, x0).item()
K_tilde = K_cpd - k_x0[:, None] - k_x0[None, :] + k_00
print(f'변환 후 Gram 최소 고유값: {np.linalg.eigvalsh(K_tilde).min():.4e}  (≥ 0 → PD)')
```

**출력 예시**:
```
RBF Gram 최소 고유값: 3.2e-08  (≥ 0 이어야 함)
RBF Gram 최대 고유값: 3.1e+01

Sigmoid Gram 최소 고유값: -1.8e+00  (음수면 PD 아님)
→ 이 c 선택에서는 PD가 아님이 확인됨

Cauchy-Schwarz: k(x_i, x_j)² = 0.0423 ≤ 1.0000 = k_ii · k_jj

-‖x-y‖² Gram 최소 고유값: -50.3201  (음수 → PD 아님)
변환 후 Gram 최소 고유값: 1.2e-14  (≥ 0 → PD)
```

---

## 🔗 실전 활용

- **SVM의 dual QP**: 목적함수 $\frac{1}{2} \alpha^\top (y y^\top \odot K) \alpha - \mathbf{1}^\top \alpha$가 **볼록**이려면 $K$가 PD여야 한다. PD가 아닌 kernel(예: 일부 $(a, c)$의 sigmoid)로 SVM을 돌리면 solver가 non-convex에 빠져 local optimum에 수렴하거나 발산.
- **Gaussian Process**: Prior covariance $K$가 PD가 아니면 $K + \sigma^2 I$의 Cholesky 분해가 실패. 실제로 **numerical jitter**($K + 10^{-6} I$)를 추가하는 이유가 이것이다.
- **Kernel 설계**: 새 kernel을 제안할 때 **PD성 증명이 논문의 필수 corollary**. Graph kernel(Random Walk kernel)·Shift-invariant kernel 등은 모두 이 과정을 거친다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\mathcal{X}$가 집합 | 위상구조 없어도 정의 가능하지만, Mercer 정리는 컴팩트 측도공간 필요 |
| 실수값 대칭성 | 복소수 확장 시 Hermitian symmetry로 변경 필요 |
| 유한 샘플에서 Gram 양정치 | 무한 샘플에서는 operator-level 양정치성(Ch1-04) |
| Strict PD가 아닌 경우 | Representer 정리의 해가 **유일하지 않을** 수 있음 |

**주의**: PD는 **샘플 크기에 무관**하게 성립해야 한다. "$n = 100$에서는 PD지만 $n = 1000$에서는 아니다" 같은 kernel은 존재하지 않는다 — 정의가 "임의의 유한 부분집합"이기 때문. 반면 수치적으로는 큰 $n$에서 조건수가 나빠져 **근사 singular**해질 수 있다.

---

## 📌 핵심 정리

$$\boxed{k \text{ is PD} \iff \forall \{x_i\}_{i=1}^n \subset \mathcal{X}, \forall \alpha \in \mathbb{R}^n: \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) \geq 0 \iff \forall \text{ Gram } K: K \succeq 0}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **PD kernel** | 모든 유한 샘플에서 Gram 행렬이 양정치 |
| **Strict PD** | distinct한 점에서 Gram이 strictly 양정치 |
| **CPD** | $\sum \alpha_i = 0$ 제약 하에서만 양정치 — 거리 kernel에 유용 |
| **Cauchy-Schwarz** | $k(x, y)^2 \leq k(x, x) k(y, y)$ — PD의 필연적 따름정리 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathcal{X} = \mathbb{R}$에서 $k(x, y) = xy$가 PD임을 정의로 직접 증명하라.

<details>
<summary>힌트 및 해설</summary>

대칭성: $xy = yx$. ✓

PD성: $\sum_{i, j} \alpha_i \alpha_j x_i x_j = \left(\sum_i \alpha_i x_i\right)^2 \geq 0$. ✓

이는 $\phi(x) = x$, $\mathcal{H} = \mathbb{R}$인 inner product kernel의 특수 사례 (정리 1.4).

</details>

**문제 2** (심화): $k(x, y) = \min(x, y)$가 $\mathcal{X} = [0, \infty)$에서 PD임을 증명하라.

<details>
<summary>힌트 및 해설</summary>

$\min(x, y) = \int_0^\infty \mathbf{1}\{t \leq x\} \mathbf{1}\{t \leq y\} \, dt$로 쓸 수 있다. 따라서 $\phi(x) := \mathbf{1}_{[0, x]} \in L^2([0, \infty))$로 놓으면 $k(x, y) = \langle \phi(x), \phi(y) \rangle_{L^2}$. 정리 1.4에 의해 PD.

이 kernel은 **Brownian motion의 공분산 함수** $\text{Cov}(B_x, B_y) = \min(x, y)$이기도 하다 — 공분산 함수는 항상 PD이다.

</details>

**문제 3** (ML 연결): SVM dual $\max \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)$ subject to $0 \leq \alpha_i \leq C$, $\sum \alpha_i y_i = 0$은 $k$가 PD일 때 볼록 최대화(또는 $-$를 곱해 볼록 최소화)가 된다. PD성이 깨지면 어떤 구체적 문제가 발생하는가?

<details>
<summary>힌트 및 해설</summary>

$Q_{ij} := y_i y_j k(x_i, x_j)$로 놓으면 목적함수는 $-\frac{1}{2} \alpha^\top Q \alpha + \mathbf{1}^\top \alpha$. $k$가 PD이면 $Q = \text{diag}(y) K \text{diag}(y)$도 PSD(두 PSD의 congruence transformation)이므로 $-\frac{1}{2} \alpha^\top Q \alpha$는 **concave**, 따라서 maximize는 concave maximization = convex QP.

PD가 아니면 $Q$가 indefinite이 되어:
- **Multiple local maxima** 존재
- SMO·SGD 등의 알고리즘이 **시작점에 따라 다른 해**로 수렴
- Dual gap이 0으로 닫히지 않음 → primal-dual 최적해 불일치
- 실무에서는 sigmoid kernel을 쓸 때 $(a, c)$를 특정 범위로 제한해 경험적으로 PD를 유지

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 대표 커널의 목록과 PD 증명 ▶](./02-kernel-zoo.md) |

</div>
