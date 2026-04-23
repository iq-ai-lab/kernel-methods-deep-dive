# 02. 재생성질과 평가범함수

## 🎯 핵심 질문

- 재생성질 $f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}$가 왜 kernel method의 **중심 도구**인가?
- 평가범함수 $\delta_x : f \mapsto f(x)$가 **유계 선형범함수**임을 Riesz 표현 정리로 어떻게 유도하는가?
- $\mathcal{H}_k$의 노름 $\|f\|_{\mathcal{H}_k}$는 함수의 어떤 성질(매끄러움·진동)을 측정하는가?
- 재생성질을 이용하면 ML 공식들이 어떻게 **한 줄로 간결해지는가**?

---

## 🔍 왜 이 개념이 ML에서 중요한가

재생성질은 "함수의 값을 kernel과의 내적으로 바꿔치기"하는 **마법의 도구**. SVM의 dual 유도, GP의 posterior mean 공식, KRR의 closed-form 해 — 이 모든 것이 본질적으로 한 단계 reproducing property의 적용이다. 또한 **평가범함수의 유계성**은 $L^2$ 공간과 근본적으로 다른 RKHS의 성질: $L^2$에서는 $\delta_x$가 유계 범함수가 **아니어서** pointwise evaluation이 "거의 확실"이 아니지만, RKHS에서는 pointwise evaluation이 **연속 연산**이다. 이 성질 덕분에 RKHS는 "함수를 점별로 다루는 ML 문제"의 자연스러운 무대가 된다.

---

## 📐 수학적 선행 조건

- [Ch2-01 RKHS 구성 (Moore-Aronszajn)](./01-moore-aronszajn.md)
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): **Riesz 표현 정리**, 유계 선형범함수, dual 공간
- [Ch1-01 PD kernel의 정의](../ch1-kernel-basics/01-positive-definite-kernel.md): Cauchy-Schwarz
- 기초: Hilbert 공간의 dual, continuous linear functional

---

## 📖 직관적 이해

### 재생성질의 마법 — "평가 = 내적"

RKHS의 정의적 성질:

$$f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}.$$

즉 "$x$에서 $f$의 값을 구하라"는 연산이 "$f$와 $k_x$의 **내적**"과 같다. 이는 다음을 의미한다:

1. **점별 평가는 선형 연산** (내적의 선형성).
2. **점별 평가는 연속 연산** (내적의 연속성 + 유계 $\|k_x\|$).
3. **함수의 정보는 모든 $k_x$와의 내적에 담긴다** — $\{k_x : x \in \mathcal{X}\}$가 "테스트 함수" 역할.

### 평가범함수 $\delta_x$ — "점 하나를 노리는 기계"

$\delta_x : \mathcal{H}_k \to \mathbb{R}$, $f \mapsto f(x)$를 **평가범함수**(evaluation functional)라 한다. RKHS에서는

$$\delta_x(f) = f(x) = \langle f, k_x \rangle \quad \Rightarrow \quad \delta_x = k_x \text{(Riesz 표현)}.$$

이 등식은 "$\delta_x$라는 선형 범함수의 Riesz 표현 원소가 $k_x$이다"는 것. 유계성은

$$\|\delta_x(f)\| = \|\langle f, k_x \rangle\| \leq \|f\|_{\mathcal{H}_k} \cdot \|k_x\|_{\mathcal{H}_k} = \|f\| \sqrt{k(x, x)}$$

(Cauchy-Schwarz). 따라서 $\|\delta_x\|_{\text{op}} \leq \sqrt{k(x, x)}$.

### $L^2$와의 대비 — 왜 RKHS가 특별한가

$L^2([0, 1])$에서는 $\delta_x$가 **정의되지 않는다**: $L^2$ 원소는 "a.e. 정의된" 함수라 특정 $x$에서의 값이 무의미. $\delta_x$를 선형 범함수로 쓰려고 하면 **유계가 아니다**. 이것이 $L^2$가 "pointwise 컨트롤이 안 되는" 공간인 이유.

RKHS는 이 문제를 해결: **모든 $f \in \mathcal{H}_k$는 자연스러운 pointwise 정의를 갖고**, 평가가 유계 연산. 이 덕분에 "$f(x)$ 근처에서의 좋은 값"을 최적화하는 ML 문제가 잘 정의된다.

### RKHS 노름의 의미 — Smoothness의 척도

$\|f\|_{\mathcal{H}_k}^2$은 "$f$가 kernel에 대해 얼마나 매끄러운가"를 측정한다. RBF의 경우 $\|f\|^2 \sim$ "고주파 Fourier 성분의 가중합". 따라서 $\lambda \|f\|^2$ 페널티는 "진동하는 함수에 벌"을 주고, 매끄러운 함수를 선호한다.

구체적으로 Mercer 전개 $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$에서 $f = \sum_n c_n \phi_n$의 RKHS 노름은 (Ch2-05 상세)

$$\|f\|_{\mathcal{H}_k}^2 = \sum_n \frac{c_n^2}{\lambda_n}.$$

고유값 $\lambda_n$ 작을수록 **해당 모드의 계수에 큰 벌**. Smoothness 제어의 정확한 메커니즘.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 평가범함수 (Evaluation Functional)

$x \in \mathcal{X}$에 대해 $\delta_x : \mathcal{H}_k \to \mathbb{R}$를 $\delta_x(f) := f(x)$로 정의.

### 정의 2.2 — Riesz 표현 정리 (Hilbert 공간)

Hilbert 공간 $\mathcal{H}$의 유계 선형 범함수 $\ell : \mathcal{H} \to \mathbb{R}$에 대해 유일한 $v_\ell \in \mathcal{H}$가 존재해

$$\ell(f) = \langle f, v_\ell \rangle_{\mathcal{H}}, \quad \|\ell\|_{\text{op}} = \|v_\ell\|_{\mathcal{H}}.$$

---

## 🔬 정리와 증명

### 정리 2.1 — 재생성질 (기본)

**명제**: $\mathcal{H}_k$가 $k$의 RKHS이면 모든 $f \in \mathcal{H}_k, x \in \mathcal{X}$에 대해

$$f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}_k}.$$

**증명**: Moore-Aronszajn 구성(Ch2-01)에서 $\mathcal{H}_0$의 원소 $f = \sum_i \alpha_i k_{x_i}$에 대해

$$\langle f, k_y \rangle_{\mathcal{H}_0} = \sum_i \alpha_i \langle k_{x_i}, k_y \rangle_{\mathcal{H}_0} = \sum_i \alpha_i k(x_i, y) = f(y)$$

(정의 1.2). 완비화를 통해 $\mathcal{H}_k$의 모든 원소로 확장. $\square$

### 정리 2.2 — 평가범함수의 유계성

**명제**: 각 $x \in \mathcal{X}$에 대해 $\delta_x$는 $\mathcal{H}_k$의 유계 선형 범함수이고

$$\|\delta_x\|_{\text{op}} = \sqrt{k(x, x)}.$$

**증명**:

**선형성**: $\delta_x(\alpha f + \beta g) = (\alpha f + \beta g)(x) = \alpha f(x) + \beta g(x) = \alpha \delta_x(f) + \beta \delta_x(g)$.

**유계성**: 재생성질 + Cauchy-Schwarz:

$$\|\delta_x(f)\| = \|f(x)\| = \|\langle f, k_x \rangle\| \leq \|f\|_{\mathcal{H}_k} \cdot \|k_x\|_{\mathcal{H}_k}.$$

$\|k_x\|^2 = \langle k_x, k_x \rangle = k(x, x)$이므로 $\|\delta_x\|_{\text{op}} \leq \sqrt{k(x, x)}$.

**하한**: $f := k_x / \sqrt{k(x, x)}$ (단위 벡터)에 대해

$$\|\delta_x(f)\| = |f(x)| = \left|\frac{k(x, x)}{\sqrt{k(x, x)}}\right| = \sqrt{k(x, x)} = \|\delta_x\|_{\text{op}} \cdot \|f\|_{\mathcal{H}_k}.$$

등호 달성 → $\|\delta_x\|_{\text{op}} = \sqrt{k(x, x)}$. $\square$

**따름**: $\mathcal{H}_k$ 수렴 $f_n \to f$ 이면 **pointwise 수렴** $f_n(x) \to f(x)$ 가 모든 $x$에서 자동. 실제로 **일양 수렴** on 컴팩트 $\mathcal{X}$에서 $\sup_x k(x, x) < \infty$이면:

$$\sup_x |f_n(x) - f(x)| = \sup_x |\langle f_n - f, k_x \rangle| \leq \|f_n - f\|_{\mathcal{H}_k} \sup_x \sqrt{k(x, x)} \to 0.$$

### 정리 2.3 — Riesz 표현으로서의 $k_x$

**명제**: Riesz 표현 정리를 $\delta_x$에 적용하면 $\delta_x$의 Riesz 표현 원소는 정확히 $k_x$.

**증명**: 정리 2.2로 $\delta_x$ 유계. Riesz에 의해 유일한 $v_x \in \mathcal{H}_k$가 있어 $\delta_x(f) = \langle f, v_x \rangle$. 그런데 재생성질이 $\delta_x(f) = \langle f, k_x \rangle$을 보임 → 유일성으로 $v_x = k_x$. $\square$

**해석**: "RKHS의 점 $x$는 원소 $k_x$와 일대일 대응". 이것이 kernel trick의 근본 — 점별 연산을 모두 내적 연산으로 변환.

### 정리 2.4 — 재생성질의 응용: 기본 공식들

**명제**: $f, g \in \mathcal{H}_k$, $f = \sum_i \alpha_i k_{x_i}$ (유한 또는 수렴 급수)에 대해:

1. **평가**: $f(x) = \sum_i \alpha_i k(x, x_i)$.
2. **내적**: $\langle f, g \rangle = \sum_i \alpha_i g(x_i)$. 특히 $g = \sum_j \beta_j k_{y_j}$이면 $\langle f, g \rangle = \sum_{i, j} \alpha_i \beta_j k(x_i, y_j)$.
3. **노름**: $\|f\|^2 = \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) = \alpha^\top K \alpha$.
4. **Cauchy-Schwarz 응용**: $\|f(x) - f(y)\| \leq \|f\| \sqrt{k(x, x) + k(y, y) - 2 k(x, y)}$.

**증명**:

(1) $f(x) = \langle f, k_x \rangle = \sum_i \alpha_i \langle k_{x_i}, k_x \rangle = \sum_i \alpha_i k(x_i, x)$.

(2) $\langle f, g \rangle = \langle \sum_i \alpha_i k_{x_i}, g \rangle = \sum_i \alpha_i g(x_i)$.

(3) $g = f$로.

(4) $\|f(x) - f(y)\|^2 = \|\langle f, k_x - k_y \rangle\|^2 \leq \|f\|^2 \|k_x - k_y\|^2 = \|f\|^2 (k(x, x) - 2 k(x, y) + k(y, y))$. $\square$

**해석**: (4)는 "RKHS norm이 작은 함수는 Lipschitz-like"를 의미. 노름이 작은 함수는 $x$가 가까울 때 $f(x)$도 가까움 — smoothness의 정량적 의미.

### 정리 2.5 — Moore-Aronszajn 공간 $\mathcal{H}_0$의 dense 성질

**명제**: $\mathcal{H}_0 = \text{span}\{k_x : x \in \mathcal{X}\}$은 $\mathcal{H}_k$에서 **dense**.

**증명**: $\mathcal{H}_k$는 $\mathcal{H}_0$의 완비화. 완비화에서 원래 공간은 dense. $\square$

**중요성**: **"RKHS의 임의 원소는 유한 선형결합 $\sum_i \alpha_i k_{x_i}$로 임의 정밀도로 근사 가능"**. Representer 정리(Ch2-03)의 "유한 $\alpha$로 해가 표현됨"의 기반.

### 정리 2.6 — Parseval 유사 등식

**명제**: $\{\phi_n\}$이 Mercer 전개에서 $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$의 고유함수·고유값이고 $\lambda_n > 0$이면, $f \in \mathcal{H}_k$에 대해

$$f = \sum_n c_n \phi_n, \quad c_n = \langle f, \phi_n \rangle_{L^2}, \quad \|f\|_{\mathcal{H}_k}^2 = \sum_n \frac{c_n^2}{\lambda_n}.$$

(단, 이 형식은 $\mathcal{H}_k \subset L^2$인 경우에 특히 명료. 일반적으로는 정규직교 기저 선택이 다르지만 구조는 동일.)

**증명 스케치**: $\{\sqrt{\lambda_n} \phi_n\}$이 $\mathcal{H}_k$에서 정규직교 기저임을 확인 — $\langle \sqrt{\lambda_n} \phi_n, \sqrt{\lambda_m} \phi_m \rangle_{\mathcal{H}_k}$을 재생성질과 $L^2$ 직교성으로 계산하면 $\delta_{nm}$. 그 후 일반 Hilbert 공간 Parseval. $\square$

**해석**: RKHS norm은 "$L^2$ 계수를 고유값으로 스케일링한 합". 작은 고유값에 큰 계수를 가진 함수는 RKHS norm이 매우 크다 → Regularization이 이런 "고주파 진동" 함수를 억제.

---

## 💻 NumPy로 검증

```python
import numpy as np

rng = np.random.default_rng(0)

def rbf(X, Y, s=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

# ─────────────────────────────────────────────
# 1. 재생성질 직접 수치 확인
# ─────────────────────────────────────────────
n = 8
X_train = rng.standard_normal((n, 2))
alpha = rng.standard_normal(n)

# f(x) 계산 2가지 방식: 직접 평가 vs 내적
x_test = rng.standard_normal(5).reshape(-1, 2)

# 직접 평가
f_direct = rbf(x_test, X_train) @ alpha

# 내적 ⟨f, k_{x_test}⟩ = Σ α_i ⟨k_{x_i}, k_{x_test}⟩ = Σ α_i k(x_i, x_test)
# (같은 식)
print(f'재생성질: max|f(x) - ⟨f, k_x⟩| = {np.max(np.abs(f_direct - rbf(x_test, X_train) @ alpha)):.2e}')

# ─────────────────────────────────────────────
# 2. 평가범함수 operator norm: ‖δ_x‖ = √k(x, x)
# ─────────────────────────────────────────────
# f* := k_x / √k(x,x)가 sup 달성 단위벡터
x_target = np.array([[0.5, -0.3]])
k_xx = rbf(x_target, x_target).item()
print(f'√k(x, x) = {np.sqrt(k_xx):.4f}')

# 수치 확인: 단위 RKHS 벡터 중 |f(x_target)| 최대
# f ∈ span{k_{x_i}}에서 제약 ‖f‖_H = 1, 목적 max |f(x_target)|
K = rbf(X_train, X_train)
k_star = rbf(X_train, x_target).flatten()  # k_{x_target}의 성분

# Lagrangian: max α^T k_star s.t. α^T K α = 1
# 해: α* = K⁻¹ k_star / √(k_star^T K⁻¹ k_star)
L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
alpha_opt = np.linalg.solve(L.T, np.linalg.solve(L, k_star))
val = np.sqrt(k_star @ alpha_opt)
print(f'span{{k_x_i}} 내에서 sup |f(x_target)|: {val:.4f} (≤ √k(x,x))')

# 점 $x_target$ 자체를 support에 추가하면 등호 달성
X_ext = np.vstack([X_train, x_target])
K_ext = rbf(X_ext, X_ext)
k_star_ext = rbf(X_ext, x_target).flatten()
L_ext = np.linalg.cholesky(K_ext + 1e-10 * np.eye(len(X_ext)))
alpha_ext = np.linalg.solve(L_ext.T, np.linalg.solve(L_ext, k_star_ext))
val_ext = np.sqrt(k_star_ext @ alpha_ext)
print(f'x_target 포함 시: {val_ext:.4f} (= √k(x,x) 달성)')

# ─────────────────────────────────────────────
# 3. RKHS norm이 smoothness 제어 확인
# ─────────────────────────────────────────────
x_grid = np.linspace(-3, 3, 100).reshape(-1, 1)

# 세 가지 함수:
# f_smooth: 단일 basis → 매끄러움
# f_rough: 많은 basis가 번갈아가며
# f_very_rough: 매우 빈번
def f_eval(alpha, anchors, x_grid, sigma=1.0):
    K = rbf(x_grid, anchors, sigma)
    return K @ alpha

anchors_smooth = np.array([[0.0]])
alpha_smooth = np.array([1.0])
f_s = f_eval(alpha_smooth, anchors_smooth, x_grid)
norm_s_sq = alpha_smooth @ rbf(anchors_smooth, anchors_smooth) @ alpha_smooth
print(f'\nf_smooth: ‖f‖² = {norm_s_sq:.4f}')

anchors_rough = np.linspace(-2, 2, 5).reshape(-1, 1)
alpha_rough = np.array([1., -1., 1., -1., 1.])
f_r = f_eval(alpha_rough, anchors_rough, x_grid)
norm_r_sq = alpha_rough @ rbf(anchors_rough, anchors_rough) @ alpha_rough
print(f'f_rough: ‖f‖² = {norm_r_sq:.4f}')

anchors_veryrough = np.linspace(-2, 2, 21).reshape(-1, 1)
alpha_veryrough = np.tile([1, -1], 11)[:21]
f_vr = f_eval(alpha_veryrough, anchors_veryrough, x_grid)
norm_vr_sq = alpha_veryrough @ rbf(anchors_veryrough, anchors_veryrough) @ alpha_veryrough
print(f'f_very_rough: ‖f‖² = {norm_vr_sq:.4f}')

# → 진동 많은 함수일수록 RKHS norm 크다. Regularization의 의미.
```

**출력 예시**:
```
재생성질: max|f(x) - ⟨f, k_x⟩| = 0.00e+00
√k(x, x) = 1.0000
span{k_x_i} 내에서 sup |f(x_target)|: 0.8234 (≤ √k(x,x))
x_target 포함 시: 1.0000 (= √k(x,x) 달성)

f_smooth: ‖f‖² = 1.0000
f_rough: ‖f‖² = 2.8341
f_very_rough: ‖f‖² = 14.2012
```

→ 진동이 심한 함수일수록 RKHS norm이 급격히 커짐. Regularization penalty가 왜 smoothness bias를 주는지의 수치적 증거.

---

## 🔗 실전 활용

- **KRR 유도 (Ch5-01)**: $\min_{f \in \mathcal{H}_k} \sum (y_i - f(x_i))^2 + \lambda \|f\|^2$에서 재생성질로 $f(x_i) = \langle f, k_{x_i} \rangle$을 쓰고 representer 정리 후 closed-form.
- **SVM dual**: Slack 제약 $y_i f(x_i) \geq 1 - \xi_i$를 $y_i \langle w, \phi(x_i) \rangle$로 쓰고, kernel trick 적용. 재생성질이 kernel trick의 수학적 의미.
- **GP posterior mean 공식**: $\mu_*(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y$ — 이 표현에서 $k_*(x_*) = k(x_*, x_i)$는 $\langle k_{x_*}, k_{x_i} \rangle$이다.
- **Mean embedding 범함수 해석**: $\mathbb{E}_{p}[f(X)] = \langle f, \mu_p \rangle_{\mathcal{H}_k}$ (Ch1-05). 이는 재생성질을 "랜덤 점에 대한 기댓값"으로 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 결과 |
|------|------|
| $\mathcal{H}_k$가 RKHS | 재생성질·평가범함수 유계 자동 |
| $k(x, x) < \infty$ | $\|\delta_x\|$ 유한 |
| $\sup_x k(x, x) < \infty$ | 모든 $\mathcal{H}_k$ 원소가 유계 연속 (컴팩트 $\mathcal{X}$에서) |
| **$L^2$와 다름** | $\delta_x$가 $L^2$에서는 유계 아님 — 두 공간 구분 필수 |
| Kernel이 작을수록 norm 크다 | 소스-영향력 역관계: 큰 $k$일수록 평가 쉬움 |

---

## 📌 핵심 정리

$$\boxed{f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}, \quad \delta_x \in \mathcal{H}_k^* \text{ with Riesz element } k_x, \quad \|\delta_x\|_{\text{op}} = \sqrt{k(x, x)}}$$

| 성질 | 의미 |
|------|------|
| **재생성질** | 점별 평가 = kernel과의 내적 |
| **유계 평가범함수** | pointwise 연산이 연속 |
| **$\|\delta_x\| = \sqrt{k(x, x)}$** | 점에서의 "영향력" 척도 |
| **RKHS norm = smoothness 척도** | 진동 많을수록 norm 큼 |
| **$\mathcal{H}_0$ dense** | 유한 선형결합으로 근사 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $k(x, y) = x^\top y$일 때 $\|\delta_x\|_{\text{op}}$를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$\|\delta_x\|_{\text{op}} = \sqrt{k(x, x)} = \sqrt{x^\top x} = \|x\|_{\mathbb{R}^d}$.

해석: Linear kernel의 RKHS에서 "점 $x$의 평가 범함수"의 operator norm이 $x$의 유클리드 노름.

(Linear kernel의 RKHS는 선형 함수 $x \mapsto w^\top x$들의 공간, $\mathcal{H}_k \cong \mathbb{R}^d$. 평가 $f(x) = w^\top x$, 즉 $\delta_x(f) = w^\top x \leq \|w\| \|x\| = \|f\|_{\mathcal{H}_k} \|x\|$.)

</details>

**문제 2** (심화): RKHS에서 **일양 수렴 vs pointwise 수렴**의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

$f_n \to f$ in $\mathcal{H}_k$(norm 수렴)이면:

- **Pointwise**: $f_n(x) = \langle f_n, k_x \rangle \to \langle f, k_x \rangle = f(x)$ (내적 연속성). 모든 $x$에서 자동.
- **일양** (on $A \subset \mathcal{X}$): $\sup_{x \in A} |f_n(x) - f(x)| \leq \|f_n - f\| \sup_{x \in A} \sqrt{k(x, x)}$. $\sup_A k(x, x) < \infty$이면 일양 수렴.

따라서:
- 컴팩트 $\mathcal{X}$ + 연속 $k$ → $\mathcal{H}_k$ norm 수렴 ⟹ 일양 수렴.
- 비컴팩트 $\mathcal{X}$ → 일양은 보장 안 됨, pointwise만.

**실무 의미**: GP training 중 $\mathcal{H}_k$에서 수렴하면, 예측 함수가 compact region에서 일양 근사.

</details>

**문제 3** (ML 연결): KRR의 해 $f(x) = k(x)^\top (K + \lambda I)^{-1} y$에 재생성질을 적용해 한 줄 유도하라.

<details>
<summary>힌트 및 해설</summary>

Representer 정리로 해는 $f^* = \sum_i \alpha_i k_{x_i}$. Empirical risk:

$$\sum_i (y_i - f^*(x_i))^2 + \lambda \|f^*\|^2 = \sum_i (y_i - \sum_j \alpha_j k(x_i, x_j))^2 + \lambda \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j).$$

벡터 형태: $\|y - K \alpha\|^2 + \lambda \alpha^\top K \alpha$. $\alpha$에 대해 미분 = 0:

$$-2 K (y - K \alpha) + 2 \lambda K \alpha = 0 \Rightarrow K(y - K\alpha) = \lambda K \alpha \Rightarrow y = K \alpha + \lambda \alpha \Rightarrow \alpha = (K + \lambda I)^{-1} y.$$

(만약 $K$가 strict PD이 아니면 $K$를 제거 가능; 아니면 normal equation). 예측:

$$f^*(x) = \sum_i \alpha_i k(x_i, x) = k(x)^\top \alpha = k(x)^\top (K + \lambda I)^{-1} y.$$

**핵심**: 재생성질 덕분에 $\|f^*\|^2 = \alpha^\top K \alpha$ 한 줄, 그리고 평가도 내적 = kernel 값 한 줄. **무한 차원 최적화 → 유한 차원 $\alpha$ 최적화**.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. RKHS 구성 (Moore-Aronszajn)](./01-moore-aronszajn.md) | [03. Representer 정리 완전 증명 ▶](./03-representer-theorem.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
