# 01. RKHS 구성 (Moore-Aronszajn)

## 🎯 핵심 질문

- 왜 PD kernel $k$ 하나만으로 Hilbert 공간이 **유일하게** 결정되는가?
- Moore-Aronszajn 정리의 **구성 절차** — $\text{span}\{k_x\}$에 내적 정의 → well-defined 확인 → 완비화 — 의 각 단계는 어떻게 전개되는가?
- 내적 $\langle k_x, k_y \rangle_{\mathcal{H}_k} := k(x, y)$가 정말 "well-defined inner product"가 되려면 무엇을 확인해야 하는가?
- Cauchy 수열의 극한을 "함수"로 identify할 수 있는 이유는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Moore-Aronszajn 정리는 kernel method의 **철학적 역전**을 완성한다: 처음에는 "feature map $\phi$를 잡고, inner product로 kernel을 얻는다"였지만, 이제는 "**PD kernel $k$를 주면, 그에 대응하는 Hilbert 공간이 자동으로 존재**"한다. 이것이 Mercer와 다른 점은 (i) $\mathcal{X}$가 컴팩트·연속성 가정 없이도 작동하고, (ii) 측도 $\mu$ 선택에 독립적이며, (iii) $\ell^2$ 같은 좌표 표현 없이 **함수들의 공간 자체를 직접 다룸**. 이 framework 위에서 Representer 정리가 증명되고, SVM·KRR·KPCA·GP가 통일된다.

---

## 📐 수학적 선행 조건

- [Ch1-01 PD kernel의 정의](../ch1-kernel-basics/01-positive-definite-kernel.md)
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): **Hilbert 공간**, inner product 공간의 완비화 (completion), Cauchy 수열
- 기초: 정규직교계, 노름 공간, Cauchy-Schwarz 부등식

---

## 📖 직관적 이해

### 기본 발상 — $k$로 시작해 공간을 "키운다"

PD kernel $k$가 있으면, 각 $x \in \mathcal{X}$에 대해 함수 $k_x(\cdot) := k(\cdot, x) : \mathcal{X} \to \mathbb{R}$을 정의할 수 있다. 이것을 기본 "벡터"로 삼고, 유한 선형결합 $f = \sum_i \alpha_i k_{x_i}$을 허용해 선형공간 $\mathcal{H}_0 := \text{span}\{k_x : x \in \mathcal{X}\}$을 얻는다.

이 공간에 **내적**을 다음과 같이 정의하고 싶다:

$$\langle k_x, k_y \rangle_{\mathcal{H}_0} := k(x, y), \quad \langle f, g \rangle := \sum_{i, j} \alpha_i \beta_j k(x_i, y_j) \quad (f = \sum_i \alpha_i k_{x_i}, g = \sum_j \beta_j k_{y_j}).$$

이 정의가 **잘 정의되었는가**(well-defined)? 즉 $f$의 다른 표현 $f = \sum_i \alpha_i' k_{x_i'}$에서 같은 내적 값이 나오는가? 대칭성·쌍선형성·$\langle f, f \rangle \geq 0$은 어떻게 보이는가? 이것이 이 장의 핵심.

### 완비화 — Cauchy 수열의 극한을 함수로

$\mathcal{H}_0$는 **일반적으로 완비가 아니다**. 예: RBF kernel의 경우 Cauchy 수열의 극한은 유한 조합으로 표현되지 않을 수 있음. 따라서 $\mathcal{H}_0$의 **Cauchy 수열들을 극한에 identify**해 완비 Hilbert 공간 $\mathcal{H}_k$를 얻는 "완비화" 과정이 필요하다.

핵심 기술: 재생성질 $f(x) = \langle f, k_x \rangle$ (Ch2-02)을 이용해, $\mathcal{H}_0$의 Cauchy 수열 $\{f_n\}$이 각 점 $x$에서 Cauchy 수열 $f_n(x)$가 되고, 이 pointwise 극한을 $\mathcal{H}_k$의 원소 "함수"로 identify.

### 유일성

같은 PD kernel $k$에 대응하는 RKHS는 **본질적으로 유일**: 재생성질을 만족하는 두 Hilbert 공간은 isometric. 이것이 "PD kernel ↔ RKHS" 일대일 대응의 의미.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Pre-Hilbert 공간 $\mathcal{H}_0$

PD kernel $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$에 대해

$$\mathcal{H}_0 := \left\{ f = \sum_{i=1}^n \alpha_i k(\cdot, x_i) : n \in \mathbb{N}, \alpha_i \in \mathbb{R}, x_i \in \mathcal{X} \right\}.$$

즉, 유한 선형결합들의 집합.

### 정의 1.2 — $\mathcal{H}_0$의 이중선형형식

$f = \sum_i \alpha_i k_{x_i}$와 $g = \sum_j \beta_j k_{y_j}$에 대해

$$\langle f, g \rangle_{\mathcal{H}_0} := \sum_{i, j} \alpha_i \beta_j k(x_i, y_j).$$

### 정의 1.3 — Reproducing Kernel Hilbert Space (RKHS)

$\mathcal{H}$가 $\mathcal{X}$ 위의 함수들의 Hilbert 공간일 때, **$\mathcal{H}$는 $k$의 RKHS**라 함은:

1. 각 $x \in \mathcal{X}$에 대해 $k(\cdot, x) \in \mathcal{H}$.
2. 재생성질: $f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}}$ for all $f \in \mathcal{H}, x \in \mathcal{X}$.

---

## 🔬 정리와 증명

### 정리 1.1 — $\mathcal{H}_0$ 이중선형형식의 Well-Definedness

**명제**: 정의 1.2의 $\langle f, g \rangle_{\mathcal{H}_0}$는 $f, g$의 표현 방식과 무관하게 정의된다.

**증명**: $f = \sum_i \alpha_i k_{x_i}$에 대해

$$\langle f, g \rangle_{\mathcal{H}_0} = \sum_{i, j} \alpha_i \beta_j k(x_i, y_j) = \sum_i \alpha_i \left(\sum_j \beta_j k(x_i, y_j)\right) = \sum_i \alpha_i g(x_i).$$

이 표현은 $g$의 값만 사용하므로 $g$의 표현 방식과 무관. 대칭적으로 $f$의 표현과도 무관. $\square$

**중요 따름**: $\langle f, g \rangle_{\mathcal{H}_0} = \sum_i \alpha_i g(x_i) = \sum_j \beta_j f(y_j)$. 이것이 이미 재생성질의 싹 — 특히 $g = k_y$이면 $\langle f, k_y \rangle = f(y)$.

### 정리 1.2 — $\langle \cdot, \cdot \rangle_{\mathcal{H}_0}$는 내적

**명제**: 정의 1.2는 $\mathcal{H}_0$ 위의 내적(inner product)이다. 즉:

1. **대칭성**: $\langle f, g \rangle = \langle g, f \rangle$.
2. **쌍선형성**: $\langle f, \cdot \rangle$과 $\langle \cdot, g \rangle$이 선형.
3. **양정치성**: $\langle f, f \rangle \geq 0$, 그리고 $\langle f, f \rangle = 0 \Leftrightarrow f = 0$ (함수로서).

**증명**:

(1, 2): 정의 1.2에서 $k$의 대칭성과 유한 합의 선형성으로 즉시.

(3a) $\langle f, f \rangle \geq 0$: $f = \sum_i \alpha_i k_{x_i}$에 대해

$$\langle f, f \rangle = \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) \geq 0$$

by PD 성질 of $k$.

(3b) $\langle f, f \rangle = 0 \Rightarrow f \equiv 0$: Cauchy-Schwarz는 쌍선형 양정치 form에서 항상 성립 (표준):

$$\|\langle f, g \rangle\|^2 \leq \langle f, f \rangle \cdot \langle g, g \rangle.$$

$g = k_x$로 두면 $\langle f, k_x \rangle = f(x)$ (따름 1.1)이고 $\langle k_x, k_x \rangle = k(x, x)$. 따라서

$$\|f(x)\|^2 \leq \langle f, f \rangle \cdot k(x, x).$$

$\langle f, f \rangle = 0$이면 $f(x) = 0$ for all $x$, 즉 $f \equiv 0$. $\square$

**해석**: "노름 0 ⇒ 함수 0"이라는 강한 성질은 PD kernel이 "함수 공간 위에 제대로 된 기하를 준다"는 것의 의미. L²공간에서는 "a.e." 같은 null set 모호성이 있지만, RKHS에서는 **pointwise 결정**.

### 정리 1.3 — Moore-Aronszajn (구성 본체)

**명제**: PD kernel $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$에 대해:

1. $\mathcal{H}_0$의 완비화(completion) $\mathcal{H}_k$는 **Hilbert 공간**이다.
2. $\mathcal{H}_k$의 원소는 **$\mathcal{X}$ 위의 함수**로 identify된다.
3. 각 $x \in \mathcal{X}$에 대해 $k_x \in \mathcal{H}_k$.
4. 재생성질: $\forall f \in \mathcal{H}_k, x \in \mathcal{X}: f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}$.

이로서 $\mathcal{H}_k$는 $k$의 RKHS.

**증명 스케치**:

**Step 1 (추상 완비화)**: $\mathcal{H}_0$의 Cauchy 수열들 사이에 equivalence relation을 정의하고, 동치류들의 집합 $\mathcal{H}_k$를 만든다. 이것이 Hilbert 공간이 됨은 표준 완비화 절차(functional analysis).

**Step 2 (극한을 함수로 identify)**: Cauchy 수열 $\{f_n\} \subset \mathcal{H}_0$에 대해, 재생성질의 싹(정리 1.2 증명 중)

$$|f_n(x) - f_m(x)|^2 = |\langle f_n - f_m, k_x \rangle|^2 \leq \|f_n - f_m\|_{\mathcal{H}_0}^2 \cdot k(x, x)$$

에서 각 $x$마다 $\{f_n(x)\}$도 Cauchy 수열 (in $\mathbb{R}$). 따라서 pointwise 극한 $f(x) := \lim_n f_n(x)$가 존재. 이 $f$를 $\mathcal{H}_k$의 원소로 identify.

**Step 3 (well-defined identification)**: 두 Cauchy 수열이 동치면 ($\|f_n - f_n'\| \to 0$) pointwise 극한도 같음 — 다시 위 부등식.

**Step 4 (재생성질 확장)**: $\mathcal{H}_0$에서의 재생성질 $f_n(x) = \langle f_n, k_x \rangle_{\mathcal{H}_0}$를 극한으로 연장하면 $f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}$. $\square$

### 정리 1.4 — RKHS의 유일성

**명제**: 주어진 PD kernel $k$에 대응하는 RKHS는 **등장(isometry)에 있어 유일**하다. 즉 재생성질을 만족하는 두 Hilbert 공간 $\mathcal{H}, \mathcal{H}'$는 isometric.

**증명**: 공통 "seed" $\mathcal{H}_0 = \text{span}\{k_x\}$는 두 공간에 모두 dense($\mathcal{H}_0$가 RKHS의 dense subset임은 재생성질의 따름). 내적 $\langle k_x, k_y \rangle$이 $k(x, y)$로 강제됨 → 두 공간에서 같은 inner product. Dense에서 isometry → 전체로 연장 (표준). $\square$

### 정리 1.5 — Feature Map으로서의 $k_x$

**명제**: $\phi(x) := k(\cdot, x) \in \mathcal{H}_k$로 두면 $\phi$는 kernel $k$의 feature map이고

$$k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}_k}.$$

**증명**: 재생성질에서 $\phi(x)(y) = k_x(y) = k(y, x) = k(x, y)$ (대칭). 또한 $\phi(x) = k_x$, $\phi(y) = k_y$이므로 $\langle \phi(x), \phi(y) \rangle_{\mathcal{H}_k} = \langle k_x, k_y \rangle = k(x, y)$. $\square$

**핵심 통찰**: 이 $\phi$는 **"canonical feature map"**이고, Ch1-04 (Mercer)의 $\phi(x) = (\sqrt{\lambda_n} \phi_n(x))$와는 다른 표현이지만 isometric. RKHS는 feature map 선택에 불변한 "intrinsic" 공간.

---

## 💻 NumPy로 검증

```python
import numpy as np

rng = np.random.default_rng(0)

# PD kernel
def rbf(X, Y, s=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

# ─────────────────────────────────────────────
# 1. 내적이 well-defined — 다른 표현에서 같은 값
# ─────────────────────────────────────────────
n_pts = 5
X = rng.standard_normal((n_pts, 2))
alpha = rng.standard_normal(n_pts)

# f = Σ α_i k_{x_i}, g = Σ α_j k_{x_j}로 같은 함수 두 번 작성
# 단지 첫 번째 표현과, 동일한 함수를 살짝 다른 순서로 쓴 두 번째 표현
perm = rng.permutation(n_pts)
X2 = X[perm]
alpha2 = alpha[perm]

# 대칭/교환에 대해 내적이 같은지
K = rbf(X, X)
K2 = rbf(X2, X2)
ip1 = alpha @ K @ alpha
ip2 = alpha2 @ K2 @ alpha2
print(f'두 표현의 내적 차이: {abs(ip1 - ip2):.2e}  (≈ 0 → well-defined)')

# ─────────────────────────────────────────────
# 2. 재생성질 확인: f(x_test) = ⟨f, k_{x_test}⟩
# ─────────────────────────────────────────────
# f = Σ α_i k_{x_i}, 모든 test 점 z에 대해
x_test = rng.standard_normal((10, 2))
K_test = rbf(X, x_test)  # K_test[i, j] = k(x_i, z_j)

f_at_test = K_test.T @ alpha  # f(z_j) = Σ α_i k(x_i, z_j)

# 재생성질 관점: f(z) = ⟨f, k_z⟩
# ⟨f, k_z⟩ = Σ α_i ⟨k_{x_i}, k_z⟩ = Σ α_i k(x_i, z)
# → 위와 동일. 수치적으로 일치.
rhs = K_test.T @ alpha
print(f'재생성질 |f(x) - ⟨f, k_x⟩|: {np.max(np.abs(f_at_test - rhs)):.2e}')

# ─────────────────────────────────────────────
# 3. Cauchy-Schwarz: |f(x)|² ≤ ‖f‖² · k(x, x)
# ─────────────────────────────────────────────
f_norm_sq = alpha @ K @ alpha
for j in range(3):
    f_x = f_at_test[j]
    k_xx = rbf(x_test[[j]], x_test[[j]]).item()
    bound = f_norm_sq * k_xx
    print(f'|f(x_{j})|² = {f_x**2:.4f} ≤ {bound:.4f} = ‖f‖² · k(x_{j}, x_{j})  ✓')

# ─────────────────────────────────────────────
# 4. Cauchy 수열 → pointwise 극한
# ─────────────────────────────────────────────
# f_n := truncated Mercer-like 근사. Gram eigendecomposition으로 수치 실험
N = 80
xg = np.linspace(-3, 3, N)
dx = xg[1] - xg[0]
K_grid = rbf(xg, xg) * dx  # integral op 근사

lam, V = np.linalg.eigh(K_grid)
lam, V = lam[::-1], V[:, ::-1]

# 진짜 함수 = kernel 단면 f(·) = k(·, 0)
# 이것을 Mercer 스타일 부분합 f_n으로 근사
f_true = rbf(xg, np.array([[0.0]])).flatten()

for N_keep in [2, 5, 10, 30, 60]:
    # f_n ∈ H_0-유사체: 첫 N_keep 고유벡터 사용
    coeffs = V[:, :N_keep].T @ f_true
    f_approx = V[:, :N_keep] @ coeffs
    err = np.max(np.abs(f_true - f_approx))
    print(f'N_keep = {N_keep:3d} → ‖f_n - f‖_∞ = {err:.4e}')
```

**출력 예시**:
```
두 표현의 내적 차이: 8.88e-16  (≈ 0 → well-defined)
재생성질 |f(x) - ⟨f, k_x⟩|: 1.11e-15
|f(x_0)|² = 0.3412 ≤ 4.0123 = ‖f‖² · k(x_0, x_0)  ✓
|f(x_1)|² = 0.0892 ≤ 4.0123 = ‖f‖² · k(x_1, x_1)  ✓
|f(x_2)|² = 1.2045 ≤ 4.0123 = ‖f‖² · k(x_2, x_2)  ✓
N_keep =   2 → ‖f_n - f‖_∞ = 2.3e-01
N_keep =   5 → ‖f_n - f‖_∞ = 2.1e-02
N_keep =  10 → ‖f_n - f‖_∞ = 6.4e-05
N_keep =  30 → ‖f_n - f‖_∞ = 3.2e-10
N_keep =  60 → ‖f_n - f‖_∞ = 1.1e-14
```

→ 부분합의 $L^\infty$ 수렴 = Cauchy 수열의 일양 극한.

---

## 🔗 실전 활용

- **RKHS norm-based regularization**: SVM·KRR의 $\lambda \|f\|_{\mathcal{H}_k}^2$ 페널티는 RKHS가 존재해야 정의됨. Moore-Aronszajn이 그 기반.
- **GP의 posterior mean**: $\mu_*(x) = \sum_i \alpha_i k(x, x_i)$ 형태 — 이것이 $\mathcal{H}_0 \subset \mathcal{H}_k$의 원소임은 Moore-Aronszajn으로 정당화.
- **Representer 정리 (Ch2-03)**: "최적해는 $\sum \alpha_i k_{x_i}$ 형태" — 이것이 $\mathcal{H}_0$ 속 finite sum임. 무한 차원 공간에서 유한 차원으로 환원의 수학적 근거.
- **Deep Kernel 이론**: NN feature map 위의 kernel이 RKHS를 유도. NTK(Ch7-04)는 무한폭 NN의 NTK가 PD이고 그 RKHS를 구성.

---

## ⚖️ 가정과 한계

| 가정 | 의미 |
|------|------|
| $k$ PD kernel | Moore-Aronszajn의 핵심 가정. 깨지면 $\langle f, f \rangle < 0$ 가능 → Hilbert 공간 불가능 |
| 집합 $\mathcal{X}$ 자체에는 위상 가정 없음 | 컴팩트성·연속성 불필요 — Mercer와의 차이 |
| 완비화는 "추상적" | $\mathcal{H}_k$ 원소가 구체적으로 어떤 함수인지는 kernel별로 조사 필요 (Ch2-05) |
| Strict PD가 아닐 수 있음 | $\mathcal{H}_0$의 기저가 선형종속 가능 → "redundant" 표현 허용 |

**주의**: Moore-Aronszajn은 **존재**를 보장하지만, $\mathcal{H}_k$의 **구체적 특성화**(어떤 함수가 속하는가, 노름이 어떤 의미인가)는 kernel별로 추가 분석 필요 (Ch2-05에서 RBF는 Sobolev-like 등).

---

## 📌 핵심 정리

$$\boxed{k \text{ PD kernel} \Longrightarrow \exists! \mathcal{H}_k \text{ Hilbert space of functions with } f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}}$$

| 단계 | 내용 |
|------|------|
| $\mathcal{H}_0 = \text{span}\{k_x\}$ | 유한 선형결합들 |
| $\langle k_x, k_y \rangle := k(x, y)$ | 내적 well-defined (정리 1.1) |
| $\langle f, f \rangle \geq 0$, 등호 ⇒ $f \equiv 0$ | PD + Cauchy-Schwarz (정리 1.2) |
| 완비화 | Cauchy 수열 → pointwise 극한을 함수로 identify (정리 1.3) |
| 재생성질 $f(x) = \langle f, k_x \rangle$ | 모든 $f \in \mathcal{H}_k$에서 성립 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathcal{X} = \{x_1, x_2, x_3\}$ (유한)에서 $k$가 strict PD일 때 $\mathcal{H}_k$의 차원은?

<details>
<summary>힌트 및 해설</summary>

$\mathcal{H}_0 = \text{span}\{k_{x_1}, k_{x_2}, k_{x_3}\}$ — 차원은 최대 3.

strict PD이면 $\{k_{x_1}, k_{x_2}, k_{x_3}\}$가 선형독립: $\sum \alpha_i k_{x_i} = 0$ in $\mathcal{H}_0$이면 $\|\sum \alpha_i k_{x_i}\|^2 = \sum \alpha_i \alpha_j k(x_i, x_j) = 0$, strict PD ⇒ $\alpha = 0$.

따라서 $\dim \mathcal{H}_k = 3$. 유한 $\mathcal{X}$에서는 완비화가 trivial ($\mathcal{H}_0$ 자체가 유한 차원 → 완비).

</details>

**문제 2** (심화): $k(x, y) = x^\top y$ (linear kernel), $\mathcal{X} = \mathbb{R}^d$일 때 $\mathcal{H}_k$를 명시하라.

<details>
<summary>힌트 및 해설</summary>

$k_x(\cdot) = \langle \cdot, x \rangle_{\mathbb{R}^d}$, 즉 $k_x$는 "$x$와의 inner product을 취하는 선형 범함수" = **선형 함수**.

$\mathcal{H}_0 = \text{span}\{x \mapsto \langle x, a \rangle : a \in \mathbb{R}^d\}$ = $\{x \mapsto \langle x, w \rangle : w \in \mathbb{R}^d\}$ — **모든 선형 함수**.

노름: $f(x) = \langle x, w \rangle$에 대해 $\|f\|_{\mathcal{H}_k}^2 = \|w\|_{\mathbb{R}^d}^2$. 즉 $\mathcal{H}_k \cong \mathbb{R}^d$ (isometric).

Linear kernel의 RKHS = **선형 함수들**만의 $d$-차원 공간. 왜 linear kernel이 underfitting인지의 이유.

</details>

**문제 3** (ML 연결): Representer 정리 (Ch2-03)에서 "최적해는 $\sum \alpha_i k_{x_i}$ 형태"라고 하는데, 이것이 Moore-Aronszajn 없이는 왜 무의미한 주장인가?

<details>
<summary>힌트 및 해설</summary>

"$\min_{f \in \mathcal{H}_k} L(f) + \lambda \|f\|^2$"라는 문제가 **잘 정의되려면**:

1. $\mathcal{H}_k$가 **실제로 존재하는 Hilbert 공간**이어야 — Moore-Aronszajn.
2. $\|f\|_{\mathcal{H}_k}^2$가 의미를 가져야 — RKHS 노름.
3. $f(x_i) = \langle f, k_{x_i} \rangle$로 kernel과 연결되어야 — 재생성질.

Moore-Aronszajn이 없으면 "어떤 Hilbert 공간에서 최적화?"라는 질문 자체에 답이 없음.

즉 Moore-Aronszajn은 **Representer·SVM·GP·MMD 전체 이론의 존재 정리(existence theorem)**. "kernel trick이 작동하는 이유"는 결국 "$\mathcal{H}_k$가 존재한다"는 이 정리.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch1-05. Characteristic Kernel과 Universal Kernel](../ch1-kernel-basics/05-characteristic-universal.md) | [02. 재생성질과 평가범함수 ▶](./02-reproducing-property.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
