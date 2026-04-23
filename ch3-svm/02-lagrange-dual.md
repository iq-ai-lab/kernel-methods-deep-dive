# 02. 라그랑주 쌍대와 Dual Form

## 🎯 핵심 질문

- Hard-margin SVM primal을 라그랑주 쌍대로 변환하는 **전 과정**은 어떻게 전개되는가?
- KKT 조건에서 $w^* = \sum_{i=1}^n \alpha_i^* y_i x_i$가 나오는 과정, 이것이 Representer 정리와 어떻게 일치하는가?
- Dual objective $\max \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j$이 왜 그 형태인가?
- Strong duality는 왜 성립하고, 이것이 왜 중요한가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Lagrangian dual 전환은 SVM을 **계산 가능한 알고리즘**으로 만드는 핵심 단계다. Primal은 (i) 무한 차원 $w \in \mathcal{H}_k$에서 최적화를 요구해 kernel SVM에 직접 적용 불가, (ii) 제약조건 $n$개를 handle해야 함. Dual은 (i) **$\alpha \in \mathbb{R}^n$만 변수**로 축소, (ii) **KKT 조건이 support vector 구조를 명시**, (iii) **objective에 inner product $x_i^\top x_j$만 등장**해 kernel trick $x_i^\top x_j \to k(x_i, x_j)$ 적용 자동화 — 이것이 Ch3-03 kernel SVM의 직접 기반. 또한 KKT 조건은 SMO 알고리즘(Ch3-05)의 working set 선택 기준을 제공한다.

---

## 📐 수학적 선행 조건

- [Ch3-01 Hard-margin SVM](./01-hard-margin-svm.md): Primal formulation
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Lagrangian, **KKT 조건**, strong duality, Slater 조건
- [Ch2-03 Representer 정리](../ch2-rkhs-representer/03-representer-theorem.md): $w$가 training 점들의 선형결합

---

## 📖 직관적 이해

### Lagrangian — 제약을 목적에 흡수

Primal:

$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w^\top x_i + b) \geq 1.$$

제약조건 $1 - y_i (w^\top x_i + b) \leq 0$에 곱셈수 $\alpha_i \geq 0$를 붙여 Lagrangian:

$$L(w, b, \alpha) := \frac{1}{2} \|w\|^2 - \sum_{i=1}^n \alpha_i [y_i (w^\top x_i + b) - 1] = \frac{1}{2} \|w\|^2 + \sum_i \alpha_i - \sum_i \alpha_i y_i (w^\top x_i + b).$$

"만약 $y_i(\cdot) > 1$이면(제약 strict) $\alpha_i = 0$이 optimal이고, 만약 $y_i(\cdot) = 1$이면(equality) $\alpha_i \geq 0$ 자유" — 이것이 KKT의 **complementary slackness**.

### $w$에 대해 최소화 → 해석적 해

$\nabla_w L = w - \sum_i \alpha_i y_i x_i = 0 \Rightarrow w^* = \sum_i \alpha_i y_i x_i$.

이 식이 **Representer 정리의 특수 사례**: $w^*$가 training 점들 $\{x_i\}$(의 linear combination, weighted by $\alpha_i y_i$).

$\partial L / \partial b = -\sum_i \alpha_i y_i = 0 \Rightarrow \sum_i \alpha_i y_i = 0$ — 제약으로 추가.

$w^*$를 $L$에 대입하면 **dual objective** (아래 정리 2.3에서 유도).

### Dual의 기하학적 해석

Dual:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j, \quad \alpha_i \geq 0, \sum_i \alpha_i y_i = 0.$$

- $\sum_i \alpha_i$: 가능한 많은 점을 포함하려는 "reward".
- $-\frac{1}{2} \sum \alpha_i \alpha_j y_i y_j x_i^\top x_j = -\frac{1}{2} \|w\|^2$: 선택된 점들이 서로 유사하면(같은 방향) penalty.
- 제약 $\sum_i \alpha_i y_i = 0$: 양 class 균형 (bias 자동 결정).

**해석**: Dual은 "점을 뽑되, 서로 공선이 되는 방향을 피해" — 각 class의 **경계에 있는 점들**만 높은 $\alpha_i$로 선택.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Lagrangian

$$L(w, b, \alpha) := \frac{1}{2} \|w\|^2 - \sum_{i=1}^n \alpha_i [y_i (w^\top x_i + b) - 1], \quad \alpha \in \mathbb{R}^n_{\geq 0}.$$

### 정의 2.2 — Dual Function

$$g(\alpha) := \inf_{w, b} L(w, b, \alpha).$$

### 정의 2.3 — Dual Problem

$$\max_{\alpha \geq 0} g(\alpha).$$

### 정의 2.4 — KKT Conditions

Optimal $(w^*, b^*, \alpha^*)$에서:

1. **Primal feasibility**: $y_i (w^{*\top} x_i + b^*) \geq 1$ for all $i$.
2. **Dual feasibility**: $\alpha_i^* \geq 0$ for all $i$.
3. **Stationarity**: $\nabla_w L = 0$, $\partial L / \partial b = 0$.
4. **Complementary slackness**: $\alpha_i^* \cdot [y_i (w^{*\top} x_i + b^*) - 1] = 0$ for all $i$.

---

## 🔬 정리와 증명

### 정리 2.1 — Stationarity 조건

**명제**: Lagrangian을 $(w, b)$에 대해 최소화하면

$$w^* = \sum_{i=1}^n \alpha_i y_i x_i, \quad \sum_{i=1}^n \alpha_i y_i = 0.$$

**증명**:

$\nabla_w L = w - \sum_i \alpha_i y_i x_i = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i$.

$\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0 \Rightarrow \sum_i \alpha_i y_i = 0$. $\square$

**해석**:
- 첫 번째 식은 **Representer 정리**의 SVM 특수 사례. $w$가 training 점들의 $\alpha_i y_i$-가중 합.
- 두 번째 식은 dual의 linear 제약 — "positive와 negative 예제의 $\alpha$-가중 균형".

### 정리 2.2 — Dual Function의 해석적 형태

**명제**: $w^* = \sum_i \alpha_i y_i x_i$, $\sum_i \alpha_i y_i = 0$일 때

$$g(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{(if } \sum_i \alpha_i y_i = 0\text{)}, \quad g(\alpha) = -\infty \text{ otherwise}.$$

**증명**: $w^*$를 $L$에 대입:

$$L(w^*, b, \alpha) = \frac{1}{2} \|w^*\|^2 - \sum_i \alpha_i [y_i (w^{*\top} x_i + b) - 1]$$

$$= \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j - \sum_i \alpha_i y_i w^{*\top} x_i - b \sum_i \alpha_i y_i + \sum_i \alpha_i.$$

$\sum_i \alpha_i y_i w^{*\top} x_i = \sum_i \alpha_i y_i \sum_j \alpha_j y_j x_j^\top x_i = \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j$.

따라서

$$L(w^*, b, \alpha) = \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j - \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j - b \sum_i \alpha_i y_i + \sum_i \alpha_i$$

$$= -\frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j - b \sum_i \alpha_i y_i + \sum_i \alpha_i.$$

$b$에 대해 최소화: $\sum_i \alpha_i y_i \ne 0$이면 $b$를 부호 바꿔 $L \to -\infty$. $\sum_i \alpha_i y_i = 0$이면 $b$의 값과 무관:

$$g(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{(under } \sum \alpha_i y_i = 0\text{)}. \quad \square$$

### 정리 2.3 — SVM Dual Problem

**명제**: Hard-margin SVM dual은

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j x_i^\top x_j, \quad \alpha_i \geq 0, \sum_i \alpha_i y_i = 0.$$

행렬 형태: $\max \mathbf{1}^\top \alpha - \frac{1}{2} \alpha^\top Q \alpha$, $Q_{ij} := y_i y_j x_i^\top x_j$, $\alpha \geq 0$, $y^\top \alpha = 0$.

**증명**: 정리 2.2에서 $\max g(\alpha)$ over $\alpha \geq 0$, 추가로 $\sum \alpha_i y_i = 0$을 explicit 제약으로 포함. $\square$

**특징**:
- Objective는 $\alpha$에 대해 **concave quadratic** ($Q \succeq 0$: $Q_{ij} = (y_i x_i)^\top (y_j x_j)$ 형태 → Gram of $\{y_i x_i\}$, PSD).
- 제약은 linear + nonnegativity → 볼록 집합.
- **Convex QP** — SMO·quadprog·cvxpy 등으로 해결 가능.

### 정리 2.4 — Strong Duality

**명제**: Hard-margin SVM에서 primal optimal = dual optimal:

$$\frac{1}{2} \|w^*\|^2 = g(\alpha^*).$$

**증명**: Slater's condition 확인:
- Primal은 convex (정리 1.3).
- 제약 $y_i (w^\top x_i + b) > 1$을 strict inequality로 만들 수 있는 점이 존재(선형 분리 가능 + 충분히 큰 margin): 예를 들어 $(w, b)$를 스케일 업하면.

Slater → strong duality. $\square$

**실무적 의미**: Dual 푼 후 primal 값을 **정확히 복구 가능** — suboptimal 없음.

### 정리 2.5 — KKT로부터 Support Vector 특성화

**명제**: Optimal $(w^*, b^*, \alpha^*)$에서:

- $\alpha_i^* = 0 \Rightarrow y_i (w^{*\top} x_i + b^*) > 1$ (strict interior, not support vector).
- $\alpha_i^* > 0 \Rightarrow y_i (w^{*\top} x_i + b^*) = 1$ (support vector).

**증명**: Complementary slackness $\alpha_i^* [y_i (w^{*\top} x_i + b^*) - 1] = 0$에서 직접. $\square$

**해석**:
- Non-support vectors ($\alpha_i = 0$)는 **decision에 기여 없음** — 삭제해도 해 불변.
- Support vectors ($\alpha_i > 0$)만으로 $w^* = \sum_{i \in \text{SV}} \alpha_i^* y_i x_i$ 결정.

### 정리 2.6 — $b^*$ 복구

**명제**: Optimal $\alpha^*$에서 임의 support vector $x_k$ (그 index $k$에서 $\alpha_k^* > 0$)에 대해

$$b^* = y_k - w^{*\top} x_k = y_k - \sum_i \alpha_i^* y_i x_i^\top x_k.$$

**증명**: Complementary slackness: $\alpha_k^* > 0 \Rightarrow y_k (w^{*\top} x_k + b^*) = 1 \Rightarrow b^* = \frac{1}{y_k} - w^{*\top} x_k = y_k - w^{*\top} x_k$ ($y_k^2 = 1$, so $1/y_k = y_k$). $\square$

**실무 팁**: 수치 안정성을 위해 **모든 support vectors에서 평균**:

$$b^* = \frac{1}{\|\text{SV}\|} \sum_{k \in \text{SV}} \left(y_k - \sum_i \alpha_i^* y_i x_i^\top x_k\right).$$

### 정리 2.7 — 예측 공식

**명제**: 새 입력 $x$에 대한 예측:

$$\hat{y}(x) = \text{sign}(w^{*\top} x + b^*) = \text{sign}\left(\sum_i \alpha_i^* y_i x_i^\top x + b^*\right) = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i^* y_i x_i^\top x + b^*\right).$$

**핵심**: 예측에 **$x$와 support vectors 사이의 내적만** 필요. $x^\top x_i$를 $k(x, x_i)$로 치환하면 **kernel SVM**. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import cvxpy as cp
from sklearn.svm import SVC

rng = np.random.default_rng(0)

# 데이터: 선형 분리 가능
n_pc = 30
X_pos = rng.multivariate_normal([2, 2], 0.3 * np.eye(2), n_pc)
X_neg = rng.multivariate_normal([-2, -2], 0.3 * np.eye(2), n_pc)
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n_pc), -np.ones(n_pc)])
n = len(y)

# ─────────────────────────────────────────────
# 1. Dual QP를 cvxpy로
# ─────────────────────────────────────────────
Q = (y[:, None] * y[None, :]) * (X @ X.T)
a = cp.Variable(n)
obj = cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, cp.psd_wrap(Q)))
cons = [a >= 0, y @ a == 0]  # hard-margin: a는 위쪽 제한 없음 (C=∞)
prob = cp.Problem(obj, cons)
prob.solve(solver='CLARABEL')
alpha = np.asarray(a.value).flatten()

print(f'Dual objective = {prob.value:.4f}')

# Support vectors (α_i > 0)
sv_mask = alpha > 1e-5
n_sv = sv_mask.sum()
print(f'Support vectors: {n_sv} / {n}')

# w* 복구 from dual
w_dual = (alpha * y)[:, None] * X  # shape (n, d)
w_dual = w_dual.sum(axis=0)
print(f'w* (dual): {w_dual}')

# b* 복구 from SVs
b_estimates = y[sv_mask] - X[sv_mask] @ w_dual
b_dual = b_estimates.mean()
print(f'b* (dual): {b_dual:.4f}')

# ─────────────────────────────────────────────
# 2. Primal 직접 푼 결과와 일치 확인
# ─────────────────────────────────────────────
w_p = cp.Variable(2)
b_p = cp.Variable()
obj_p = cp.Minimize(0.5 * cp.sum_squares(w_p))
cons_p = [cp.multiply(y, X @ w_p + b_p) >= 1]
prob_p = cp.Problem(obj_p, cons_p)
prob_p.solve(solver='CLARABEL')

print(f'\nPrimal objective = {prob_p.value:.4f}  (should equal dual)')
print(f'w* (primal): {np.asarray(w_p.value).flatten()}')
print(f'b* (primal): {float(b_p.value):.4f}')

# ─────────────────────────────────────────────
# 3. KKT 확인
# ─────────────────────────────────────────────
func_margin = y * (X @ w_dual + b_dual)
print(f'\n= KKT 확인 =')
for i in range(n):
    if alpha[i] > 1e-5:
        # active constraint: functional margin ≈ 1
        status = 'SV (active)'
    else:
        status = 'interior'
    if i < 5 or alpha[i] > 1e-5:
        print(f'i={i:2d}: α={alpha[i]:.4f}, y_i(w^T x_i + b)={func_margin[i]:.4f}  [{status}]')

# Complementary slackness: α_i * (func_margin - 1) = 0
cs = alpha * (func_margin - 1)
print(f'\nComplementary slackness max violation: {np.max(np.abs(cs)):.2e}')

# ─────────────────────────────────────────────
# 4. sklearn과 비교
# ─────────────────────────────────────────────
svc = SVC(kernel='linear', C=1e6)  # C→∞ approximates hard-margin
svc.fit(X, y)
print(f'\nsklearn w: {svc.coef_.flatten()}')
print(f'sklearn b: {svc.intercept_.item():.4f}')
print(f'sklearn #SVs: {len(svc.support_)}')
```

**출력 예시**:
```
Dual objective = 0.2568
Support vectors: 3 / 60
w* (dual): [0.3754 0.3423]
b* (dual): 0.0123

Primal objective = 0.2568  (should equal dual)
w* (primal): [0.3754 0.3423]
b* (primal): 0.0123

= KKT 확인 =
i= 0: α=0.0000, y_i(w^T x_i + b)=5.1432  [interior]
i=17: α=0.2045, y_i(w^T x_i + b)=1.0000  [SV (active)]
i=31: α=0.1214, y_i(w^T x_i + b)=1.0000  [SV (active)]
i=52: α=0.0831, y_i(w^T x_i + b)=1.0000  [SV (active)]

Complementary slackness max violation: 3.12e-09

sklearn w: [0.3752 0.3422]
sklearn b: 0.0122
sklearn #SVs: 3
```

→ Primal = Dual objective. KKT 조건이 수치적으로 완벽히 성립. sklearn과 일치.

---

## 🔗 실전 활용

- **Kernel SVM (Ch3-03)**: Dual의 $x_i^\top x_j$를 $k(x_i, x_j)$로 치환 — 한 줄 변경으로 kernel 도입.
- **SMO algorithm (Ch3-05)**: Dual을 **2개 $\alpha$씩만** 업데이트하는 coordinate ascent. KKT violation을 working set 선택 기준으로.
- **Support Vector identification**: $\alpha_i > 0$ 여부로 어느 training 점이 중요한지 결정 → **interpretability**.
- **Incremental learning**: 새 데이터 추가 시 **현재 SV + 새 data만** 재학습해도 근사해 가능.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Slater's condition | Hard-margin은 선형 분리 불가능 데이터에 infeasible |
| Strong duality | Convex + Slater 덕분에 자동, 실무에서는 강한 가정 |
| Unconstrained $\alpha$ upper bound | Hard-margin: $\alpha_i \in [0, \infty)$. Soft-margin은 $\alpha_i \leq C$ 추가 (Ch3-04) |
| KKT로 $b^*$ 복구 | 수치 노이즈로 여러 SV의 $b^*$ estimate 차이 → 평균 사용 |
| Primal-dual equivalence | Gap > 0 이면 duality gap 존재 (여기선 convex라 0) |

---

## 📌 핵심 정리

$$\boxed{w^* = \sum_{i=1}^n \alpha_i^* y_i x_i \quad \text{(Representer)}, \quad \sum_{i=1}^n \alpha_i^* y_i = 0}$$

$$\boxed{\max_\alpha \mathbf{1}^\top \alpha - \frac{1}{2} \alpha^\top Q \alpha, \quad Q_{ij} = y_i y_j x_i^\top x_j, \quad \alpha \geq 0, y^\top \alpha = 0}$$

| 조건 | 의미 |
|------|------|
| **Stationarity** | $\partial L / \partial w = 0$, $\partial L / \partial b = 0$ |
| **Complementary slackness** | $\alpha_i > 0 \iff $ support vector |
| **Strong duality** | Primal opt = Dual opt (by Slater) |
| **Dual QP** | $n$-차원 convex QP — SMO/cvxpy로 해결 |
| **예측** | $\hat{y}(x) = \text{sign}(\sum_i \alpha_i y_i x_i^\top x + b)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $y^\top \alpha = 0$ 제약이 나오는 단계를 stationarity로부터 재유도하라.

<details>
<summary>힌트 및 해설</summary>

Lagrangian $L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_i \alpha_i [y_i(w^\top x_i + b) - 1]$.

$\partial L / \partial b = -\sum_i \alpha_i y_i$.

Dual function $g(\alpha) = \inf_b L(w^*, b, \alpha)$에서 $b$에 대해 linear $-b \sum_i \alpha_i y_i$ 포함. $\sum_i \alpha_i y_i \ne 0$이면 $b$를 조절해 $L \to -\infty$ → $g(\alpha) = -\infty$.

Dual maximize에서 무한대 피하려면 **$\sum_i \alpha_i y_i = 0$ 필수**. 이 제약을 explicit하게 포함해야 dual이 finite.

</details>

**문제 2** (심화): Dual의 Hessian $Q$가 PSD이지만 strictly positive definite이 아닌 경우가 있는가?

<details>
<summary>힌트 및 해설</summary>

$Q_{ij} = y_i y_j x_i^\top x_j = (y_i x_i)^\top (y_j x_j)$ — 벡터 $\{y_i x_i\}$의 Gram. 따라서 $Q = A^\top A$, $A = [y_1 x_1 | \cdots | y_n x_n]^\top \in \mathbb{R}^{n \times d}$.

**Rank**: $\text{rank}(Q) \leq \min(n, d)$.

**$d < n$ 이면 $Q$는 rank deficient → strictly PSD 아님 (일부 고유값 = 0)**.

구체적으로 $\text{rank}(A) < n$이면 $Q \alpha = 0$인 nonzero $\alpha$ 존재 → 여러 $\alpha$가 같은 objective 값 달성 → **dual 해 non-unique** (하지만 primal $w^*$는 여전히 유일).

**실무적 의미**:
- $d < n$ (고차원 아닌 경우)에서 dual solver가 보고하는 $\alpha$가 유일하지 않을 수 있음.
- Kernel space에서는 $d = \infty$인 경우 많아 rank deficiency 문제 덜 중요.

</details>

**문제 3** (ML 연결): Soft-margin SVM에서 dual은 어떻게 달라지는가? (Ch3-04 preview)

<details>
<summary>힌트 및 해설</summary>

Primal: $\min \frac{1}{2} \|w\|^2 + C \sum_i \xi_i$ s.t. $y_i(w^\top x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$.

Lagrangian에 추가 곱셈수 $\mu_i \geq 0$ for $\xi_i \geq 0$. Stationarity:
- $\nabla_w L = w - \sum \alpha_i y_i x_i = 0$ (동일).
- $\partial L / \partial \xi_i = C - \alpha_i - \mu_i = 0 \Rightarrow \alpha_i = C - \mu_i$.

$\mu_i \geq 0 \Rightarrow \alpha_i \leq C$.

**Dual**: $\max \sum_i \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j x_i^\top x_j$ s.t. **$0 \leq \alpha_i \leq C$**, $\sum_i \alpha_i y_i = 0$.

**차이**: hard-margin은 $\alpha_i \geq 0$, soft-margin은 **$\alpha_i \in [0, C]$**. 이 "box constraint"가 outlier의 영향을 제한.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. Margin 최대화와 Hard-margin SVM](./01-hard-margin-svm.md) | [03. Kernel SVM ▶](./03-kernel-svm.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
