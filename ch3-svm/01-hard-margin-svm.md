# 01. Margin 최대화와 Hard-margin SVM

## 🎯 핵심 질문

- 선형 분리 가능 데이터에서 **margin**은 기하학적으로 어떻게 정의되고, 왜 $\text{margin} = 2 / \|w\|$인가?
- Margin 최대화 $\Leftrightarrow \min \frac{1}{2} \|w\|^2$가 되는 유도는?
- 제약조건 $y_i (w^\top x_i + b) \geq 1$의 **"1"**은 어디서 왔는가 — 단순 스케일링 choice인가 아니면 본질적인가?
- 분리 초평면의 **유일성** — 왜 "가장 넓은 margin의 초평면"이 유일한가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

SVM은 "주어진 데이터를 가장 안전하게 분리"하는 원칙 — margin 최대화 — 에 기반한다. 이 원칙은 단순해 보이지만, (i) **statistical learning theory의 VC 이론**에서 generalization error의 upper bound를 margin으로 제어할 수 있음을 보장하고 (Vapnik 1998), (ii) **Lagrangian dual로 변환**해 kernel trick과 결합하면 비선형 분류가 자동으로 가능해지며, (iii) **support vector**라는 sparsity 개념을 제공한다. 이 장의 1단계 기하 유도를 완전히 이해해야 Ch3-02(dual)·Ch3-03(kernel trick)·Ch3-04(soft-margin)으로 매끄럽게 연결된다.

---

## 📐 수학적 선행 조건

- 선형대수 기초: 초평면 $\{x : w^\top x + b = 0\}$, 유클리드 거리
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): 볼록 집합, 볼록 최적화의 기본 형태
- 기본 최적화: 2차 프로그래밍(QP), KKT 조건 (Ch3-02에서 본격 사용)

---

## 📖 직관적 이해

### "가장 안전한 분리선" — 거리로의 직관

훈련 데이터 $\{(x_i, y_i)\}_{i=1}^n$, $y_i \in \{-1, +1\}$, 선형 분리 가능: $w^\top x_i + b > 0 \iff y_i = +1$.

그런 분리선은 **무수히 많다**. 어떤 것이 "가장 좋은"가? SVM의 답: **양 class에서 가장 먼 선**.

$w^\top x + b = 0$ 초평면과 점 $x_i$의 distance: $\frac{\|w^\top x_i + b\|}{\|w\|}$. 양 class로부터의 **minimum distance**의 합을 **margin**이라 한다. 이것을 최대화.

### 정규화 choice — 왜 "$\geq 1$"인가

$(w, b)$를 $(c w, c b)$로 scale하면 분리 초평면은 같지만 "$\|w^\top x + b\|$" 값이 변한다. 이 **스케일 모호성**을 제거하기 위해 "가장 가까운 점에서 $\|w^\top x + b\| = 1$"이 되도록 정규화. 그러면:

- **제약조건**: $y_i (w^\top x_i + b) \geq 1$ for all $i$.
- **Margin**: 가장 가까운 점과의 거리는 $1 / \|w\|$. 양 class 간 거리는 $2 / \|w\|$.

따라서 **margin 최대화 = $\|w\|$ 최소화 = $\frac{1}{2} \|w\|^2$ 최소화** (제곱 사용은 볼록·미분가능을 위해).

### Support Vector의 기하적 의미

분리 초평면에서 정확히 거리 $1 / \|w\|$인 점들 = **margin 경계에 있는 점들**. 이들이 **support vector** — 나중에 dual에서 $\alpha_i > 0$인 점들과 일대일 대응.

직관: "가장 빡빡하게 margin을 제약"하는 점들이 support vector. 이 점들만으로 분리선이 결정된다 (sparse해의 기원).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 선형 분리 가능성

데이터 $\{(x_i, y_i)\}_{i=1}^n \subset \mathbb{R}^d \times \{-1, +1\}$가 **선형 분리 가능**이라 함은 $w \in \mathbb{R}^d$, $b \in \mathbb{R}$가 존재해

$$y_i (w^\top x_i + b) > 0 \quad \forall i.$$

### 정의 1.2 — Margin (Functional · Geometric)

초평면 $H_{w, b} := \{x : w^\top x + b = 0\}$과 데이터에 대해:

- **Functional margin**: $\hat{\gamma}_i := y_i (w^\top x_i + b)$. Data-dependent, 스케일 $(w, b)$에 의존.
- **Geometric margin**: $\gamma_i := \frac{y_i (w^\top x_i + b)}{\|w\|}$. 스케일에 독립, 실제 유클리드 거리.

전체 데이터에 대한 geometric margin: $\gamma := \min_i \gamma_i$.

### 정의 1.3 — Hard-margin SVM (Primal)

선형 분리 가능 데이터에 대해:

$$\min_{w, b} \quad \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i (w^\top x_i + b) \geq 1, \quad i = 1, \ldots, n.$$

---

## 🔬 정리와 증명

### 정리 1.1 — Margin과 $\|w\|$의 관계

**명제**: 초평면 $H_{w, b}$과 점 $x$의 거리는 $\|w^\top x + b\| / \|w\|$. 특히 $y_i (w^\top x_i + b) = 1$인 가장 가까운 점(support vector)의 거리는 $1 / \|w\|$, 두 class 간 margin은 $2 / \|w\|$.

**증명**: 초평면 $w^\top z + b = 0$에 대한 점 $x$의 수직 투영 $z^*$은 $x$로부터 $w/\|w\|$ 방향으로 $d$만큼 이동: $z^* = x - d w / \|w\|$. 초평면 조건으로

$$w^\top (x - d w / \|w\|) + b = 0 \Rightarrow w^\top x + b = d \|w\| \Rightarrow d = \frac{w^\top x + b}{\|w\|}.$$

부호 포함: $d = \frac{|w^\top x + b|}{\|w\|}$.

정규화 $y_i (w^\top x_i + b) = 1$ support vector에서 $d = 1 / \|w\|$. 양 class의 support vector는 반대편이므로 margin(class 간 거리) = $2 / \|w\|$. $\square$

### 정리 1.2 — Margin 최대화와 $\|w\|^2$ 최소화의 동치

**명제**: 정규화 $\min_i y_i (w^\top x_i + b) = 1$ 하에서

$$\max_{w, b} \frac{2}{\|w\|} \text{ subject to } \cdots \iff \min_{w, b} \frac{1}{2} \|w\|^2 \text{ subject to } y_i (w^\top x_i + b) \geq 1.$$

**증명**:

$\max \frac{2}{\|w\|} = \min \frac{\|w\|}{2}$ (감소함수 관계).

$\min \|w\|$는 $\|w\| > 0$에서 $\min \|w\|^2$과 동치 (단조증가).

$\frac{1}{2}$는 미분 편의를 위한 상수: $\frac{d}{dw} \frac{1}{2} \|w\|^2 = w$ (cleaner).

제약조건: 정규화 후 "가장 가까운 점의 functional margin = 1" → $y_i(\cdot) \geq 1$. $\square$

### 정리 1.3 — 볼록 최적화

**명제**: Hard-margin SVM primal은 **볼록 이차 프로그램(QP)**: convex objective + linear 제약.

**증명**:
- Objective: $\frac{1}{2} \|w\|^2 = \frac{1}{2} w^\top I w$ — strictly convex (Hessian = $I \succ 0$).
- 제약: $1 - y_i (w^\top x_i + b) \leq 0$ — affine in $(w, b)$, 따라서 halfspace 교집합 = 볼록 집합.
- 실현 가능성: 선형 분리 가능 가정 → 실현 가능.

따라서 strongly convex QP → **유일한 global minimum**. $\square$

**따름**: 최적 $(w^*, b^*)$는 유일 (selection 모호성 없음). "최적 분리선"의 well-posed 정의.

### 정리 1.4 — Support Vector의 기하적 특성화

**명제**: 최적해 $(w^*, b^*)$에서 제약 $y_i (w^{*\top} x_i + b^*) \geq 1$이 **등호**로 성립하는 점 $x_i$들이 **support vector**. 이 점들만의 부분집합만으로 같은 최적해 얻음.

**증명 아이디어** (정식 증명은 Ch3-02 KKT에서):

- **Active constraint**: $y_i (w^{*\top} x_i + b^*) = 1$인 $i$. 
- 이 점들을 제거하면 최적해가 바뀔 수 있음(support).
- 나머지 점 ($y_i(\cdot) > 1$)을 제거해도 최적해 불변 — 제약이 **strictly 안쪽**이므로 영향 없음.

따라서 **"필요한 최소한의 점들"**이 support vector. Sparsity의 기원. $\square$

### 정리 1.5 — 분리 초평면의 유일성

**명제**: 선형 분리 가능 데이터에 대해 hard-margin SVM의 최적 분리 초평면 $\{x : w^{*\top} x + b^* = 0\}$은 유일하다.

**증명**: Objective $\frac{1}{2} \|w\|^2$이 strictly convex → $w^*$ 유일. 등호 조건 (support vector의 functional margin = 1)에서 $b^*$도 유일. $\square$

### 정리 1.6 — Generalization Bound의 Margin 의존성 (Vapnik-style)

**명제 (비공식)**: Margin $\gamma$가 클수록 generalization error의 upper bound가 작아진다. 구체적으로 VC-dimension 관점에서

$$\text{error}_{\text{test}} \leq \text{error}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{R^2 / \gamma^2 + \log(1/\delta)}{n}}\right)$$

($R$ = 데이터 반경). Margin-based bound는 VC-dimension bound보다 일반적으로 tight.

**의미**: "margin 최대화"는 **generalization 관점에서 직접적 이점**. Rademacher complexity 관점에서도 유사한 결과.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. 선형 분리 가능 2D 데이터 생성
# ─────────────────────────────────────────────
n_per_class = 30
X_pos = rng.multivariate_normal([2, 2], 0.3 * np.eye(2), n_per_class)
X_neg = rng.multivariate_normal([-2, -2], 0.3 * np.eye(2), n_per_class)
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n_per_class), -np.ones(n_per_class)])
n, d = X.shape

# ─────────────────────────────────────────────
# 2. Hard-margin SVM primal을 cvxpy로
# ─────────────────────────────────────────────
w = cp.Variable(d)
b = cp.Variable()
objective = cp.Minimize(0.5 * cp.sum_squares(w))
constraints = [cp.multiply(y, X @ w + b) >= 1]
prob = cp.Problem(objective, constraints)
prob.solve(solver='CLARABEL')

w_star = np.asarray(w.value).flatten()
b_star = float(b.value)
margin = 2.0 / np.linalg.norm(w_star)

print(f'w* = {w_star}')
print(f'b* = {b_star:.4f}')
print(f'‖w*‖ = {np.linalg.norm(w_star):.4f}')
print(f'Margin = 2/‖w*‖ = {margin:.4f}')

# Support vector 식별 (functional margin ≈ 1)
func_margins = y * (X @ w_star + b_star)
sv_mask = np.abs(func_margins - 1) < 1e-3
n_sv = sv_mask.sum()
print(f'Support vectors: {n_sv} / {n}')

# ─────────────────────────────────────────────
# 3. 시각화 — margin, 초평면, SV 강조
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(X[y > 0, 0], X[y > 0, 1], c='red', s=40, label='class +1')
plt.scatter(X[y < 0, 0], X[y < 0, 1], c='blue', s=40, label='class -1')
plt.scatter(X[sv_mask, 0], X[sv_mask, 1], s=150, facecolors='none', edgecolors='k', linewidths=2, label=f'support vectors ({n_sv})')

# 초평면 + margin 라인
xx = np.linspace(-4, 4, 100)
# w1 x1 + w2 x2 + b = 0  →  x2 = -(w1 x + b) / w2
yy = -(w_star[0] * xx + b_star) / w_star[1]
yy_up = -(w_star[0] * xx + b_star - 1) / w_star[1]
yy_dn = -(w_star[0] * xx + b_star + 1) / w_star[1]
plt.plot(xx, yy, 'k-', lw=2, label='decision boundary')
plt.plot(xx, yy_up, 'k--', lw=1, label='margin')
plt.plot(xx, yy_dn, 'k--', lw=1)
plt.legend(); plt.xlim(-4, 4); plt.ylim(-4, 4)
plt.title(f'Hard-margin SVM (margin = {margin:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

# ─────────────────────────────────────────────
# 4. Margin 최대화 = ‖w‖ 최소화 검증
# ─────────────────────────────────────────────
# 임의 valid 분리 (w_alt, b_alt)을 시도해 margin 비교
# w_alt = 2 * w_star (double scale) — 초평면은 같지만 constraint가 달라짐
# 정규화 하면 동일 분리, margin 동일: check invariance
w_alt = 2 * w_star
b_alt = 2 * b_star
func_margins_alt = y * (X @ w_alt + b_alt)
print(f'\n정규화 전 min functional margin: {func_margins_alt.min():.4f}')
# 정규화: 최소 functional margin = 1이 되도록 scale
scale = func_margins_alt.min()
w_norm = w_alt / scale
b_norm = b_alt / scale
margin_norm = 2.0 / np.linalg.norm(w_norm)
print(f'정규화 후 margin: {margin_norm:.4f} (should equal {margin:.4f})')

# 다른 방향의 분리선
w_bad = np.array([1.0, 0.0])
b_bad = 0.0
fm_bad = y * (X @ w_bad + b_bad)
if fm_bad.min() > 0:
    scale_bad = fm_bad.min()
    margin_bad = 2.0 * scale_bad / np.linalg.norm(w_bad)
    print(f'대안 방향 w=(1,0): margin = {margin_bad:.4f}  < {margin:.4f}  (SVM이 더 큼)')
```

**출력 예시**:
```
w* = [0.5123 0.4989]
b* = -0.0012
‖w*‖ = 0.7153
Margin = 2/‖w*‖ = 2.7960
Support vectors: 4 / 60

정규화 전 min functional margin: 2.0000
정규화 후 margin: 2.7960 (should equal 2.7960)
대안 방향 w=(1,0): margin = 2.3421  < 2.7960  (SVM이 더 큼)
```

→ SVM의 margin이 임의 방향의 분리선보다 크다. 정규화는 margin 값에 영향 없음 (geometric margin invariance).

---

## 🔗 실전 활용

- **Linear SVM classifier**: 고차원 sparse 데이터(NLP, 텍스트 분류)에서 표준. `sklearn.svm.LinearSVC`.
- **Margin-based regularization**: 왜 $\frac{1}{2} \|w\|^2$가 "regularization"인가 — 큰 $\|w\|$ = 작은 margin = overfit 위험.
- **Dual로의 전환** (Ch3-02): Hard-margin primal은 그대로 풀기 어려움 ($d$가 무한대인 kernel case). Lagrangian dual로 $n$-차원 문제로 전환 + kernel trick.
- **Soft-margin (Ch3-04)**: 선형 분리 불가능 데이터에 slack $\xi_i \geq 0$로 확장.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **선형 분리 가능** | 실제 데이터는 거의 항상 noise 있음 → soft-margin 필요 |
| 유클리드 거리 = 자연스러운 거리 | 구조화 데이터(그래프, 문자열)에서는 부적절 — kernel로 우회 |
| $y_i \in \{-1, +1\}$ | 다중 클래스는 one-vs-rest, one-vs-one, 또는 Crammer-Singer 확장 |
| $\|w\|$ 정규화 | Scale-invariant 분리선 선택이지만, feature 스케일링 영향 있음 |
| Outlier 1개가 전체 해 지배 | Soft-margin으로 완화 |

---

## 📌 핵심 정리

$$\boxed{\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w^\top x_i + b) \geq 1 \quad \forall i}$$

$$\boxed{\text{Margin} = \frac{2}{\|w\|}, \quad \text{최대화} \Leftrightarrow \min \|w\|}$$

| 개념 | 의미 |
|------|------|
| **Geometric margin** | 분리선과 가장 가까운 점의 실제 거리 |
| **Functional margin = 1** | 스케일 정규화 choice |
| **Support Vector** | 제약이 equality로 성립하는 점들 — 기하적으로 margin 경계 위 |
| **볼록 QP** | Strictly convex objective + linear 제약 → 유일 최적해 |
| **Margin-based generalization** | 큰 margin ⇒ 작은 generalization error upper bound |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 $\frac{1}{2} \|w\|^2$인가 (그냥 $\|w\|$이 아닌 이유)?

<details>
<summary>힌트 및 해설</summary>

1. **미분 편의**: $\nabla_w \frac{1}{2} \|w\|^2 = w$ — 간단한 linear form. 반면 $\nabla_w \|w\| = w / \|w\|$ — $w = 0$에서 미분 불가능.

2. **Smooth & strictly convex**: $\frac{1}{2} \|w\|^2$은 $C^\infty$ strictly convex. $\|w\|$는 $w = 0$에서 kink (cone).

3. **Objective 값이 strict**: minimization에서는 $\|w\|$와 $\|w\|^2$의 최적해 동일 (단조 증가). 2차식은 QP로 standard form.

4. **Dual에서의 깔끔함**: Lagrangian이 quadratic → dual이 quadratic (QP). $\|w\|$를 쓰면 dual이 non-smooth.

</details>

**문제 2** (심화): 정규화 $y_i(w^\top x_i + b) \geq 1$ 의 "1"을 "$M$"으로 일반화하면 최적해가 어떻게 바뀌는가?

<details>
<summary>힌트 및 해설</summary>

$y_i(w^\top x_i + b) \geq M$ 제약 하에서 $\min \frac{1}{2} \|w\|^2$:

변수 치환 $w' = w / M$, $b' = b / M$ 하면

$y_i(w'^\top x_i + b') \geq 1$, $\min \frac{1}{2} \|M w'\|^2 = \frac{M^2}{2} \|w'\|^2$.

최적해: $w'^* = w^*_{\text{original}}$, $w^* = M w'^*$, margin $= 2/\|w^*\| = 2M / \|w'^*\|$.

**결론**: "$M$"은 **scale parameter**일 뿐 기하적 의미 없음. 관습적으로 $M = 1$로 고정. 분리 초평면과 geometric margin은 $M$에 불변.

</details>

**문제 3** (ML 연결): Hard-margin SVM을 kernel space에서 "직접" 실행하려면 어떤 문제가 생기는가? 이것이 dual로 전환하는 motivation은?

<details>
<summary>힌트 및 해설</summary>

Kernel space $\mathcal{H}$ (예: RBF의 무한차원 RKHS)에서 primal:

$$\min_{w \in \mathcal{H}, b} \frac{1}{2} \|w\|_{\mathcal{H}}^2 \quad \text{s.t.} \quad y_i (\langle w, \phi(x_i) \rangle + b) \geq 1.$$

**문제점**:
1. **$w$가 무한 차원** — 직접 최적화 불가능.
2. $\phi$를 명시적으로 구성할 수 없을 수도 있음 (Mercer feature의 $\ell^2$).
3. Inner product $\langle w, \phi(x_i) \rangle$을 계산하려면 $\phi$ 필요.

**Dual로의 motivation** (Ch3-02):
- Representer 정리로 $w^* = \sum_i \alpha_i y_i \phi(x_i)$ → $n$-차원 $\alpha$ 문제.
- Dual의 objective에 **$\langle \phi(x_i), \phi(x_j) \rangle = k(x_i, x_j)$**만 등장 → kernel trick 자동 적용.
- 실제 $\phi$를 계산할 필요 없이 $k$ 값만 필요.

따라서 **kernel과 결합하려면 dual이 필수**. Hard-margin dual이 kernel SVM의 모든 것의 시작.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch2-05. $\mathcal{H}_k$의 함수 공간적 성질](../ch2-rkhs-representer/05-rkhs-function-spaces.md) | [02. 라그랑주 쌍대와 Dual Form ▶](./02-lagrange-dual.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
