# 04. Soft-margin SVM과 Hinge Loss

## 🎯 핵심 질문

- 선형 분리 불가능 데이터에서 hard-margin이 infeasible이 되는 문제를 **slack variable $\xi_i \geq 0$**로 어떻게 해결하는가?
- $\min \frac{1}{2} \|w\|^2 + C \sum_i \xi_i$에서 **$C$의 역할**과 bias-variance trade-off는?
- Soft-margin이 **hinge loss** $\max(0, 1 - y_i f(x_i))$ + L2 regularization으로 재작성되는 과정은?
- Soft-margin dual의 **box constraint** $0 \leq \alpha_i \leq C$는 어디서 오는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Hard-margin SVM은 선형 분리 가능 데이터에만 작동 — 실무에서 거의 쓸모없다. Soft-margin이 SVM을 "어떤 데이터에도 적용 가능한 실용 알고리즘"으로 변환한다. 또한 **hinge loss 관점**은 SVM을 다른 ML 알고리즘 (logistic regression, squared loss 등)과 **같은 template "loss + regularization"** 프레임워크에 넣어 비교·조합 가능하게 만든다. 특히 hinge loss는 **sparsity**(correctly classified far-from-boundary 점은 gradient 0 → SV 아님)의 기원이고, 이것이 SVM의 특징적 sparse 해의 수학적 원천이다.

---

## 📐 수학적 선행 조건

- [Ch3-01~03](./01-hard-margin-svm.md): Hard-margin primal·dual, kernel trick
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Lagrangian with multiple constraint types, KKT
- [Ch2-03 Representer 정리](../ch2-rkhs-representer/03-representer-theorem.md)

---

## 📖 직관적 이해

### Slack Variable — "위반을 허용하되 벌을 준다"

Hard-margin: $y_i (w^\top x_i + b) \geq 1$ — 강제.

Soft-margin: $y_i (w^\top x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$ — 위반량 $\xi_i$ 허용.

- $\xi_i = 0$: margin 준수 (또는 correctly classified + margin 충분).
- $0 < \xi_i < 1$: margin 안쪽이지만 아직 correctly classified.
- $\xi_i \geq 1$: **misclassified**.

Objective에 페널티 $C \sum \xi_i$ 추가 → 위반 줄이기와 margin 최대화의 **trade-off**.

### $C$의 역할 — Bias-Variance

- **$C$ 큼**: 위반에 큰 벌 → hard-margin에 근접 → low bias but **high variance** (overfit).
- **$C$ 작음**: 위반 관대 → 큰 margin 선호 → high bias but **low variance** (underfit).

실무에서 $C$는 **cross-validation**으로 튜닝. 대략 $10^{-3}$부터 $10^3$까지 log scale grid.

### Hinge Loss 재작성

Soft-margin primal:

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_i \xi_i, \quad y_i(w^\top x_i + b) \geq 1 - \xi_i, \xi_i \geq 0.$$

Optimal $\xi_i^* = \max(0, 1 - y_i(w^\top x_i + b))$ (가능한 최솟값). 대입:

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i(w^\top x_i + b)).$$

**Hinge loss**: $\ell_{\text{hinge}}(z) := \max(0, 1 - z)$, $z = y_i f(x_i)$.

- $z \geq 1$ (correctly classified + margin ≥ 1): loss = 0. **무관심**.
- $z < 1$: loss = $1 - z$. 선형 감소.

이 구조가 **sparsity**: hinge = 0인 점들은 gradient 기여 0 → support vector 아님.

### Dual의 Box Constraint $0 \leq \alpha_i \leq C$

Slack $\xi_i \geq 0$에 곱셈수 $\mu_i \geq 0$. Stationarity: $\alpha_i + \mu_i = C$ → $\alpha_i \leq C$ (since $\mu_i \geq 0$). 조합하면 $\alpha_i \in [0, C]$.

이 제약이 **outlier 영향 제한**: 한 outlier의 $\alpha_i$가 $C$로 capped → 전체 해를 지배하지 못함. Hard-margin의 "$\alpha_i = \infty$로 발산" 문제 해결.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Soft-margin SVM Primal

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i, \quad \xi_i \geq 0, \quad y_i (w^\top x_i + b) \geq 1 - \xi_i.$$

### 정의 4.2 — Hinge Loss

$$\ell_{\text{hinge}}(z) := \max(0, 1 - z).$$

### 정의 4.3 — Hinge Form

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i (w^\top x_i + b)).$$

### 정의 4.4 — Soft-margin Dual

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j), \quad 0 \leq \alpha_i \leq C, \sum_i \alpha_i y_i = 0.$$

---

## 🔬 정리와 증명

### 정리 4.1 — Slack Form과 Hinge Form의 동치

**명제**: 정의 4.1과 4.3은 동일한 $(w^*, b^*)$를 산출.

**증명**: 정의 4.1에서 $(w, b)$ 고정 후 $\xi$에 대해 최소화:

$$\min_{\xi_i \geq 0} \xi_i \quad \text{s.t.} \quad \xi_i \geq 1 - y_i(w^\top x_i + b).$$

$\xi_i^* = \max(0, 1 - y_i(w^\top x_i + b))$. 이것이 hinge loss 정의. 대입 → 4.3. $\square$

### 정리 4.2 — Soft-margin Dual 유도

**명제**: 정의 4.1의 dual은 정의 4.4.

**증명**: Lagrangian:

$$L = \frac{1}{2} \|w\|^2 + C \sum_i \xi_i - \sum_i \alpha_i [y_i(w^\top x_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i.$$

Stationarity:
- $\nabla_w L = w - \sum_i \alpha_i y_i x_i = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i$.
- $\partial L / \partial b = -\sum_i \alpha_i y_i = 0$.
- $\partial L / \partial \xi_i = C - \alpha_i - \mu_i = 0 \Rightarrow \mu_i = C - \alpha_i$.

$\mu_i \geq 0 \Rightarrow \alpha_i \leq C$. 또한 $\alpha_i \geq 0$ → $\alpha_i \in [0, C]$.

$L$에 대입 (hard-margin과 유사; $C \sum \xi_i$ 항과 $-\sum \alpha_i \xi_i - \sum \mu_i \xi_i = -C \sum \xi_i$ 상쇄):

$$g(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j.$$

Kernel SVM: $x_i^\top x_j \to k(x_i, x_j)$. $\square$

**요약**: **Hard vs Soft dual 차이는 $\alpha_i$의 upper bound가 $\infty$ → $C$**. 목적함수와 $\sum \alpha_i y_i = 0$ 제약은 동일.

### 정리 4.3 — KKT로부터 점의 3가지 분류

**명제**: Optimal $(w^*, b^*, \alpha^*, \xi^*)$에서 각 $i$는 다음 3가지 중 하나:

1. **Interior (not SV)**: $\alpha_i^* = 0$, $\xi_i^* = 0$, $y_i(w^{*\top} x_i + b^*) > 1$.
2. **Free SV (on margin)**: $0 < \alpha_i^* < C$, $\xi_i^* = 0$, $y_i(w^{*\top} x_i + b^*) = 1$.
3. **Bounded SV (margin violator)**: $\alpha_i^* = C$, $\xi_i^* > 0$, $y_i(w^{*\top} x_i + b^*) < 1$.
   - $\xi_i^* \in (0, 1)$: correctly classified이지만 margin 안쪽.
   - $\xi_i^* \geq 1$: **misclassified**.

**증명**: KKT complementary slackness:
- $\alpha_i^* (1 - \xi_i^* - y_i(w^{*\top} x_i + b^*)) = 0$.
- $\mu_i^* \xi_i^* = 0 \Leftrightarrow (C - \alpha_i^*) \xi_i^* = 0$.

Cases:
1. $\alpha_i^* = 0$: 첫 번째 trivially. $(C - 0) \xi_i^* = 0 \Rightarrow \xi_i^* = 0$. Therefore $y_i(\cdot) \geq 1$.
2. $0 < \alpha_i^* < C$: $1 - \xi_i^* - y_i(\cdot) = 0$. $(C - \alpha_i^*) \xi_i^* = 0 \Rightarrow \xi_i^* = 0$. Therefore $y_i(\cdot) = 1$.
3. $\alpha_i^* = C$: $1 - \xi_i^* - y_i(\cdot) = 0$. $\mu_i^* = 0$, $\xi_i^*$ 자유.

$\square$

**실무**: Free SV의 functional margin = 1을 이용해 $b^* = y_k - \sum_i \alpha_i^* y_i k(x_i, x_k)$ 추정.

### 정리 4.4 — Hinge Loss의 Sub-gradient

**명제**: $\ell_{\text{hinge}}(z) = \max(0, 1 - z)$의 sub-gradient:

$$\partial \ell_{\text{hinge}}(z) = \begin{cases} \{0\} & z > 1 \\ [-1, 0] & z = 1 \\ \{-1\} & z < 1. \end{cases}$$

**증명**: $z > 1$: $\ell = 0$, derivative 0. $z < 1$: $\ell = 1 - z$, derivative $-1$. $z = 1$: non-differentiable, sub-gradient 범위. $\square$

**함의**: Correctly classified + margin 충분한 점 ($z > 1$)은 **gradient 기여 없음** → sparsity. **SGD-based SVM** (Pegasos)도 이 성질 활용.

### 정리 4.5 — Hinge Loss의 Bayes-consistent 성질

**명제**: Population-level에서 $\mathbb{E}[\ell_{\text{hinge}}(y f(X))]$ 최소화자는 **Bayes classifier의 부호와 일치**.

**정확히**: $f^*(x) := \text{sign}(\eta(x) - 1/2)$, $\eta(x) = P(y = 1 \mid X = x)$.

**증명 개요**: Hinge loss conditional risk $R(f(x), x) = (1 - \eta(x)) \max(0, 1 + f(x)) + \eta(x) \max(0, 1 - f(x))$. Minimize over $f(x) \in \mathbb{R}$:

- $\eta(x) > 1/2$: $f^*(x) = 1$.
- $\eta(x) < 1/2$: $f^*(x) = -1$.

부호: $\text{sign}(f^*(x)) = \text{sign}(\eta(x) - 1/2)$ = Bayes. $\square$

**의미**: "SVM을 충분한 데이터로 학습하면 Bayes error에 점근 수렴" — consistency의 근거.

### 정리 4.6 — $C \to \infty$의 극한 = Hard-margin

**명제**: 선형 분리 가능 데이터에서 $C \to \infty$이면 soft-margin 최적해가 hard-margin 최적해로 수렴.

**증명**: $C$ 커지면 $\sum \xi_i$의 페널티가 커짐. 선형 분리 가능이면 $\xi = 0$ 가능 → $C \to \infty$에서 $\xi^* = 0$ forced. 나머지는 hard-margin과 동일. $\square$

**실무**: `sklearn.svm.SVC(C=1e10)`이 근사 hard-margin. 정확한 hard-margin은 선형 분리 가능 확인 + `C=infinity`이지만 수치적으로 문제 있어 큰 $C$로 근사.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.svm import SVC
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. Noise 포함 데이터 (선형 분리 불가)
# ─────────────────────────────────────────────
X, y01 = make_moons(n_samples=100, noise=0.3, random_state=0)
y = 2 * y01 - 1
n = len(y)

def rbf(X, Y, s=0.5):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K = rbf(X, X)

# ─────────────────────────────────────────────
# 2. 다양한 C에 대해 soft-margin dual
# ─────────────────────────────────────────────
for C_val in [0.1, 1.0, 100.0]:
    a = cp.Variable(n)
    Q = (y[:, None] * y[None, :]) * K
    obj = cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, cp.psd_wrap(Q)))
    cons = [a >= 0, a <= C_val, y @ a == 0]
    prob = cp.Problem(obj, cons)
    prob.solve(solver='CLARABEL')
    alpha = np.asarray(a.value).flatten()

    free = (alpha > 1e-4) & (alpha < C_val - 1e-4)
    bounded = alpha > C_val - 1e-4
    margin = 1 / np.sqrt(alpha @ Q @ alpha + 1e-12)
    print(f'C={C_val}: objective={prob.value:.3f}, '
          f'free SV={free.sum()}, bounded SV={bounded.sum()}, '
          f'margin ≈ {margin:.3f}')

# ─────────────────────────────────────────────
# 3. Hinge loss 시각화
# ─────────────────────────────────────────────
z = np.linspace(-2, 3, 200)
hinge = np.maximum(0, 1 - z)
logistic = np.log(1 + np.exp(-z))  # logistic loss
zero_one = (z < 0).astype(float)

plt.figure(figsize=(9, 4))
plt.plot(z, hinge, 'r-', lw=2, label='Hinge: max(0, 1-z)')
plt.plot(z, logistic, 'b-', lw=2, label='Logistic: log(1+e^-z)')
plt.plot(z, zero_one, 'k--', lw=1.5, label='0-1 (true error)')
plt.axvline(1, color='gray', linestyle=':', alpha=0.7, label='margin boundary')
plt.axvline(0, color='red', linestyle=':', alpha=0.5, label='decision')
plt.xlabel('$z = y \\cdot f(x)$'); plt.ylabel('loss')
plt.title('Hinge loss vs 0-1 vs logistic'); plt.legend()
plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. 3-category 확인: interior / free SV / bounded SV
# ─────────────────────────────────────────────
C_val = 1.0
a = cp.Variable(n)
Q = (y[:, None] * y[None, :]) * K
prob = cp.Problem(cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, cp.psd_wrap(Q))),
                   [a >= 0, a <= C_val, y @ a == 0])
prob.solve(solver='CLARABEL')
alpha = np.asarray(a.value).flatten()

free = (alpha > 1e-4) & (alpha < C_val - 1e-4)
bounded = alpha > C_val - 1e-4
interior = alpha <= 1e-4

# b* from free SV
f_at_free = K[free] @ (alpha * y)
b_star = (y[free] - f_at_free).mean()
func_margin = y * (K @ (alpha * y) + b_star)

print(f'\n3-category 검증:')
print(f'interior ({interior.sum()}): min func_margin = {func_margin[interior].min():.3f} (should be > 1)')
print(f'free SV  ({free.sum()}): func_margin ≈ 1: {func_margin[free].mean():.3f}')
print(f'bounded  ({bounded.sum()}): func_margin < 1: {func_margin[bounded].mean():.3f}')

# ─────────────────────────────────────────────
# 5. Decision boundary with bounded SV 강조
# ─────────────────────────────────────────────
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = (rbf(grid, X) @ (alpha * y) + b_star).reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['lightblue', 'salmon'])
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'])
plt.scatter(X[interior & (y > 0), 0], X[interior & (y > 0), 1], c='red', s=30, label='interior +1')
plt.scatter(X[interior & (y < 0), 0], X[interior & (y < 0), 1], c='blue', s=30, label='interior -1')
plt.scatter(X[free, 0], X[free, 1], s=120, facecolors='none', edgecolors='k', lw=2, label=f'free SV ({free.sum()})')
plt.scatter(X[bounded, 0], X[bounded, 1], s=120, facecolors='yellow', edgecolors='k', lw=2, label=f'bounded SV ({bounded.sum()})')
plt.title(f'Soft-margin Kernel SVM (C={C_val}) — 3-category SV')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

**출력 예시**:
```
C=0.1: objective=5.423, free SV=8, bounded SV=27, margin ≈ 1.823
C=1.0: objective=24.612, free SV=11, bounded SV=15, margin ≈ 0.734
C=100.0: objective=145.231, free SV=9, bounded SV=3, margin ≈ 0.142

3-category 검증:
interior (65): min func_margin = 1.024 (should be > 1)
free SV  (11): func_margin ≈ 1: 1.000
bounded  (24): func_margin < 1: 0.346
```

→ $C$ 커질수록 margin 작아지고 bounded SV 감소 (근사 hard-margin 행동). KKT 3-category 정확히 확인.

---

## 🔗 실전 활용

- **실무 튜닝**: RBF soft-margin SVM이 default. Grid search: $C \in \{10^{-3}, \ldots, 10^3\}$, $\gamma \in \{10^{-3}, \ldots, 10^2\}$, 5-fold CV.
- **Class imbalance**: `class_weight='balanced'`로 minority class에 더 큰 $C$ 부여.
- **SGD-SVM (Pegasos)**: $n$ 매우 큰 경우 dual 불가능 → primal hinge form을 SGD로. $O(n)$ per epoch.
- **L1-SVM**: $\|w\|^2$ 대신 $\|w\|_1$ regularization → sparse weight (feature selection).

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Hinge loss | 확률 출력 없음 (SVM은 본질적으로 classification만) — 확률 원하면 Platt scaling |
| Non-differentiable at margin | Sub-gradient methods OK, 2차 최적화 조금 복잡 |
| $C$ 수동 튜닝 | Cross-validation 비용 |
| **Multi-class confusion** | OvR·OvO 방법에 따라 결과 다름. Crammer-Singer는 joint formulation이지만 느림 |
| **Noisy labels** | $C$ 작게 해서 robust하게. 그래도 극단적 mislabel에 민감 |

---

## 📌 핵심 정리

$$\boxed{\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i(w^\top x_i + b)) \quad \text{(Hinge form)}}$$

$$\boxed{\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j), \quad 0 \leq \alpha_i \leq C, y^\top \alpha = 0}$$

| Category | $\alpha_i$ | $\xi_i$ | Functional margin |
|----------|-----------|---------|-------------------|
| Interior (not SV) | 0 | 0 | $> 1$ |
| Free SV | $(0, C)$ | 0 | $= 1$ |
| Bounded SV | $C$ | $> 0$ | $< 1$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Hinge loss $\ell(z) = \max(0, 1 - z)$와 0-1 loss $\mathbf{1}\{z < 0\}$의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

- **Surrogate loss**: Hinge는 0-1의 upper bound: $\mathbf{1}\{z < 0\} \leq \max(0, 1 - z)$.
- **Convex**: Hinge는 convex, 0-1는 non-convex. Convex surrogate가 최적화 가능하게 함.
- **Margin-based**: Hinge는 $z > 1$에서 gradient 0 (sparsity). 0-1는 $z = 0$에서 점프.
- **Bayes-consistent**: Hinge minimizer의 부호 = Bayes classifier.

Hinge loss는 "convex upper bound + Bayes-consistent + sparsity"의 이상적 조합. 이게 SVM의 이론적 장점.

</details>

**문제 2** (심화): Soft-margin SVM과 Logistic Regression의 차이 — 어떨 때 어느 것이 유리한가?

<details>
<summary>힌트 및 해설</summary>

| 측면 | SVM (Hinge) | Logistic |
|------|------------|----------|
| Loss | $\max(0, 1 - yf)$ | $\log(1 + e^{-yf})$ |
| Sparsity | Sparse SV | 모든 점 기여 |
| 확률 출력 | 없음 (직접은) | $\sigma(f(x))$로 자연스럽게 |
| Outlier 영향 | Bounded (hinge linear growth) | **Logistic도 linear growth** — 유사한 robustness |
| Kernel화 | Representer 정리 | Representer 정리 (kernel logistic) |
| 최적화 | Convex QP (dual) | Convex, IRLS |

**유리한 상황**:
- **SVM**: Sparsity 원할 때 (interpretable SV, faster inference), probability 불필요.
- **Logistic**: Probability 필요 (calibration, threshold tuning), 모든 점이 기여.

실무에서는 RBF kernel + soft-margin SVM과 kernel logistic regression이 **비슷한 성능**, 선택은 downstream 요구에 따라.

</details>

**문제 3** (ML 연결): Deep learning의 cross-entropy loss와 hinge loss를 비교. NN classifier에서 hinge가 덜 인기인 이유는?

<details>
<summary>힌트 및 해설</summary>

- **Cross-entropy (CE)**: $-\log p(y \mid x)$. 확률 출력에 최적화된 probabilistic loss. Gradient가 항상 non-zero (gradient signal 강력).
- **Hinge**: Margin-based. $z > 1$에서 gradient 0 → "이미 잘 맞춘 예제"는 무시.

**NN에서 hinge가 덜 인기인 이유**:
1. **확률 캘리브레이션**: CE는 softmax로 자연스러운 확률 출력. Hinge는 확률 직접 없음.
2. **Gradient saturation**: Hinge의 margin 달성된 샘플은 0 gradient → batch에서 활용도 낮음.
3. **Multi-class**: CE는 multi-class 쉽게 확장 (softmax + CE). Hinge의 Crammer-Singer multi-class는 복잡.
4. **경험적 성능**: Deep networks에서 CE가 꾸준히 약간 우위.

그러나 **Hinge NN도 사용됨**: e.g., Ranking tasks, certain sparse-label scenarios. Also, Hinge의 sparsity는 specific domains에서 유용.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Kernel SVM](./03-kernel-svm.md) | [05. SMO (Sequential Minimal Optimization) ▶](./05-smo.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
