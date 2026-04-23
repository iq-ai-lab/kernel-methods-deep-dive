# 03. Kernel SVM

## 🎯 핵심 질문

- Dual에서 $x_i^\top x_j$를 $k(x_i, x_j)$로 치환하는 **"kernel trick"**이 왜 수학적으로 정당한가?
- 이 치환이 Representer 정리와 어떻게 일관되는가?
- Kernel SVM의 예측 공식 $\hat{y}(x) = \text{sign}(\sum_i \alpha_i y_i k(x_i, x) + b)$의 유도는?
- RBF·Polynomial kernel 하의 **decision boundary**는 왜 비선형인가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Kernel SVM은 **SVM을 범용 비선형 분류기로 확장**한 역사적 전환점이다. 선형 분리 불가능 데이터에도 RBF·polynomial 등으로 **원래 feature 공간에서는 비선형인 결정 경계**를 그릴 수 있게 되었고, 이것이 1990~2010년대 SVM 황금기의 기반이다. 수학적으로는 Ch3-02의 dual이 제공한 "**$\langle x_i, x_j \rangle$만 필요한 구조**"가 kernel trick을 **"치환 한 줄"**로 가능케 한다. 이 장은 (i) Kernel trick의 Representer 정리적 근거, (ii) RBF SVM이 왜 거의 universal한 분류기인지, (iii) decision boundary의 기하를 완성한다.

---

## 📐 수학적 선행 조건

- [Ch3-02 Lagrange dual](./02-lagrange-dual.md): Dual QP, KKT, Support Vector
- [Ch1 전체](../ch1-kernel-basics/01-positive-definite-kernel.md): PD kernel, Mercer, 각 kernel의 성질
- [Ch2-03 Representer 정리](../ch2-rkhs-representer/03-representer-theorem.md): $w = \sum \alpha_i \phi(x_i)$
- [Ch2-04 계산적 환원](../ch2-rkhs-representer/04-computational-reduction.md): Kernel trick의 수학적 토대

---

## 📖 직관적 이해

### "한 줄로 비선형이 된다" — Kernel Trick

Linear dual:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j \underbrace{x_i^\top x_j}_{\text{inner product}}.$$

**치환** $x_i^\top x_j \to k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_{\mathcal{H}}$:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j).$$

**의미**: "원래 공간의 linear SVM"을 "feature 공간 $\mathcal{H}$의 linear SVM"으로 upgrade. $\phi$가 비선형 사상이면 **원래 공간에서는 비선형**.

### Representer 정리와의 일관성

Feature 공간 $\mathcal{H}$에서의 linear SVM:

$$\min_{w \in \mathcal{H}, b} \frac{1}{2} \|w\|_{\mathcal{H}}^2 \quad \text{s.t.} \quad y_i (\langle w, \phi(x_i) \rangle + b) \geq 1.$$

Representer 정리(Ch2-03): 최적 $w^* = \sum_i \alpha_i y_i \phi(x_i)$. 대입하면

$$\langle w^*, \phi(x) \rangle = \sum_i \alpha_i y_i \langle \phi(x_i), \phi(x) \rangle = \sum_i \alpha_i y_i k(x_i, x).$$

Norm: $\|w^*\|_{\mathcal{H}}^2 = \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)$.

→ Lagrangian·dual·예측 모두 $k$로 표현됨. **$\phi$를 명시하지 않아도 작동**.

### Decision Boundary의 기하

$$\hat{y}(x) = \text{sign}\left(\sum_i \alpha_i^* y_i k(x_i, x) + b^*\right).$$

- **RBF**: $k(x_i, x) = \exp(-\|x_i - x\|^2 / 2\sigma^2)$ — $x$가 support vector $x_i$에 가까울수록 기여 큼. "influence zones"의 가중 합. Decision boundary는 **복잡한 곡면**.
- **Polynomial**: $k(x_i, x) = (x_i^\top x + c)^d$ — 차수 $d$까지의 feature 상호작용. Polynomial surfaces.
- **Linear**: $k(x_i, x) = x_i^\top x$ — 원래 공간의 선형 분리선으로 축퇴.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Kernel SVM Primal (feature 공간)

PD kernel $k$와 feature map $\phi : \mathcal{X} \to \mathcal{H}_k$에 대해:

$$\min_{w \in \mathcal{H}_k, b} \frac{1}{2} \|w\|_{\mathcal{H}_k}^2 \quad \text{s.t.} \quad y_i (\langle w, \phi(x_i) \rangle_{\mathcal{H}_k} + b) \geq 1.$$

### 정의 3.2 — Kernel SVM Dual

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j), \quad \alpha \geq 0, \sum_i \alpha_i y_i = 0.$$

### 정의 3.3 — Kernel SVM 예측

$$\hat{y}(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i^* y_i k(x_i, x) + b^*\right).$$

---

## 🔬 정리와 증명

### 정리 3.1 — Kernel Trick의 정당성 (dual 관점)

**명제**: Feature 공간 $\mathcal{H}_k$의 linear SVM dual은 정확히 정의 3.2이다. 즉 **"치환 $x_i^\top x_j \to k(x_i, x_j)$"는 feature 공간에서의 dual과 동일**.

**증명**: Ch3-02에서 linear SVM dual은

$$\max \sum_i \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle_{\mathcal{X}}.$$

Feature 공간 $\mathcal{H}_k$에서는 $x_i$가 $\phi(x_i)$로 대체되고, 내적이 $\langle \phi(x_i), \phi(x_j) \rangle_{\mathcal{H}_k} = k(x_i, x_j)$. 이외의 모든 식(제약, 형태)은 동일. $\square$

### 정리 3.2 — Representer 정리와의 일치

**명제**: Kernel SVM primal의 최적해 $w^*$는

$$w^* = \sum_{i=1}^n \alpha_i^* y_i \phi(x_i) = \sum_{i=1}^n \alpha_i^* y_i k(\cdot, x_i).$$

**증명**: KKT stationarity (Ch3-02 정리 2.1)에서 $x$를 $\phi(x)$로 대체하면 $w^* = \sum \alpha_i y_i \phi(x_i)$. RKHS 관점에서 $\phi(x_i) = k(\cdot, x_i)$ (canonical feature map, Ch2-01 정리 1.5). $\square$

### 정리 3.3 — 예측 공식 (kernel form)

**명제**: 새 입력 $x$에 대한 SVM 예측:

$$\hat{y}(x) = \text{sign}(f^*(x) + b^*), \quad f^*(x) := \sum_{i=1}^n \alpha_i^* y_i k(x_i, x).$$

**증명**: 

$$\langle w^*, \phi(x) \rangle = \left\langle \sum_i \alpha_i^* y_i \phi(x_i), \phi(x) \right\rangle = \sum_i \alpha_i^* y_i \langle \phi(x_i), \phi(x) \rangle = \sum_i \alpha_i^* y_i k(x_i, x). \quad \square$$

### 정리 3.4 — $b^*$ 복구 (kernel form)

**명제**: Support vector $x_k$ (with $\alpha_k^* > 0$)에 대해

$$b^* = y_k - \sum_i \alpha_i^* y_i k(x_i, x_k).$$

**증명**: Complementary slackness: $\alpha_k^* > 0 \Rightarrow y_k (f^*(x_k) + b^*) = 1 \Rightarrow b^* = y_k - f^*(x_k)$. $\square$

### 정리 3.5 — Support Vector의 Sparsity와 기하

**명제**: Kernel SVM에서도 support vector는 "margin 경계 위의 점들":

- $\alpha_i^* > 0 \iff y_i (f^*(x_i) + b^*) = 1$ (margin boundary 위, feature 공간에서).
- $\alpha_i^* = 0 \iff y_i (f^*(x_i) + b^*) > 1$ (strict 분리).

**증명**: KKT complementary slackness. $\square$

**중요**: "margin boundary 위"는 **feature 공간** $\mathcal{H}_k$에서의 개념. 원래 공간에서는 복잡한 곡면.

### 정리 3.6 — Kernel 선택과 Decision Boundary

**명제**: 다음은 각 kernel에서의 decision boundary 형태:

| Kernel | $f^*(x) = \sum_i \alpha_i^* y_i k(x_i, x)$ | Boundary |
|--------|-------------------------------------------|----------|
| Linear | $(\sum_i \alpha_i^* y_i x_i)^\top x = w^{*\top} x$ | 초평면 (선형) |
| Polynomial-$d$ | $\sum_i \alpha_i^* y_i (x_i^\top x + c)^d$ | 차수 $d$ 다항 곡면 |
| RBF | $\sum_i \alpha_i^* y_i \exp(-\|x_i - x\|^2 / 2\sigma^2)$ | 부드러운 곡면 ("Gaussian bumps" 합) |

RBF는 **universal** (Ch1-05) → 충분한 SV와 적절한 $\sigma$로 임의 연속 경계 근사 가능.

### 정리 3.7 — $n$, $\sigma$, SV 개수의 관계

**명제 (정성적)**: RBF kernel에서 $\sigma$가 작을수록, 또는 $n$이 클수록, **더 많은 support vector가 필요**해진다. 극단적으로 $\sigma \to 0$이면 모든 training 점이 SV가 된다 (완벽 training fit, overfit).

**증명 스케치**: $\sigma \to 0$에서 $K \to I$ (Ch1-02 문제 2). 이 경우 dual $\max \sum \alpha_i - \frac{1}{2} \alpha^\top \text{diag}(y)^2 \alpha$, 제약 $\alpha \in [0, C]$, $y^\top \alpha = 0$. Objective = $\sum \alpha_i (1 - \alpha_i / 2)$. Max at $\alpha_i = 1$ for each (ignoring $\alpha y$ = 0) → 모든 점이 SV. $\square$

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
# 1. XOR / moon-style 데이터 (선형 분리 불가)
# ─────────────────────────────────────────────
X, y01 = make_moons(n_samples=100, noise=0.2, random_state=0)
y = 2 * y01 - 1  # {-1, +1}

def rbf(X, Y, s=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

# ─────────────────────────────────────────────
# 2. Kernel SVM dual (soft-margin, C=1로 근사 hard)
# ─────────────────────────────────────────────
sigma = 0.5
K = rbf(X, X, sigma)
Q = (y[:, None] * y[None, :]) * K

C = 10.0
n = len(y)
a = cp.Variable(n)
obj = cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, cp.psd_wrap(Q)))
cons = [a >= 0, a <= C, y @ a == 0]
prob = cp.Problem(obj, cons)
prob.solve(solver='CLARABEL')
alpha = np.asarray(a.value).flatten()

sv_mask = alpha > 1e-4
n_sv = sv_mask.sum()
print(f'Kernel SVM dual objective = {prob.value:.4f}')
print(f'Support vectors: {n_sv} / {n}')

# b* from free SV (0 < α < C)
free_sv = (alpha > 1e-4) & (alpha < C - 1e-4)
if free_sv.sum() > 0:
    f_at_sv = K[free_sv] @ (alpha * y)
    b_star = np.mean(y[free_sv] - f_at_sv)
else:
    # Bounded SV만 있으면 fallback
    f_at_sv = K[sv_mask] @ (alpha * y)
    b_star = np.mean(y[sv_mask] - f_at_sv)
print(f'b* = {b_star:.4f}')

# ─────────────────────────────────────────────
# 3. Decision boundary 시각화
# ─────────────────────────────────────────────
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
K_grid = rbf(grid, X, sigma)
f_grid = K_grid @ (alpha * y) + b_star
Z = f_grid.reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['lightblue', 'salmon'])
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'])
plt.scatter(X[y > 0, 0], X[y > 0, 1], c='red', s=30, label='class +1')
plt.scatter(X[y < 0, 0], X[y < 0, 1], c='blue', s=30, label='class -1')
plt.scatter(X[sv_mask, 0], X[sv_mask, 1], s=100, facecolors='none', edgecolors='k', lw=2, label=f'SV ({n_sv})')
plt.title(f'Kernel SVM (RBF, σ={sigma}, C={C}) — 비선형 decision boundary')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. sklearn 검증
# ─────────────────────────────────────────────
gamma_sklearn = 1 / (2 * sigma**2)
svc = SVC(kernel='rbf', C=C, gamma=gamma_sklearn)
svc.fit(X, y)
print(f'\nsklearn SVs: {len(svc.support_)}')

# 예측 일치 확인
y_pred_manual = np.sign(f_grid.reshape(-1))
y_pred_sklearn = svc.predict(grid)
agree = (y_pred_manual == y_pred_sklearn).mean()
print(f'바닥 구현 vs sklearn 예측 일치율: {agree:.4f}')

# ─────────────────────────────────────────────
# 5. σ 변화 영향 — overfit vs underfit
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, s in zip(axes, [0.1, 0.5, 3.0]):
    svc_s = SVC(kernel='rbf', C=C, gamma=1/(2*s**2)).fit(X, y)
    Zs = svc_s.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Zs, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['lightblue', 'salmon'])
    ax.contour(xx, yy, Zs, levels=[0], colors='black')
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c='red', s=20)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c='blue', s=20)
    ax.set_title(f'σ = {s}, SVs = {len(svc_s.support_)}')
plt.tight_layout(); plt.show()
```

**출력 예시**:
```
Kernel SVM dual objective = 18.3421
Support vectors: 31 / 100
b* = -0.1234

sklearn SVs: 31
바닥 구현 vs sklearn 예측 일치율: 0.9998
```

→ Manual 구현과 sklearn 일치. $\sigma$ 작을수록 support vector 많아지고 경계가 복잡해짐.

---

## 🔗 실전 활용

- **실무 defaults**: RBF kernel, $\sigma$를 **data의 pairwise distance median**으로 초기화, $C$는 cross-validation.
- **Sklearn mapping**: `SVC(kernel='rbf', gamma=..., C=...)` where `gamma = 1/(2σ²)`.
- **Polynomial kernel**: `kernel='poly', degree=d, coef0=c`. $d = 2, 3$이 실무 범위.
- **Custom kernel**: `sklearn.svm.SVC(kernel=callable)` — 구조화 데이터(그래프·문자열)에 직접 kernel 정의.
- **Multi-class**: OvR (one-vs-rest) 또는 OvO (one-vs-one). Sklearn default는 OvO.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $k$가 PD | Non-PD kernel (일부 sigmoid, $\tanh$)은 dual non-convex, solver 실패 가능 |
| Dual QP 해결 가능 | $n$ 크면 $O(n^3)$ — SMO, LASVM, 또는 Random Features 필요 |
| Training 점 모두 메모리 | 새 데이터 예측 시 SV 집합 저장 필요 |
| Kernel 선택 = hyperparameter | $\sigma$, $d$, $C$ 전부 튜닝 필요 |
| **Soft-margin 필요** | 선형 분리 불가능 + noise 데이터에서 hard-margin은 infeasible, Ch3-04 소프트 필요 |

---

## 📌 핵심 정리

$$\boxed{\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j), \quad \alpha \geq 0, y^\top \alpha = 0}$$

$$\boxed{\hat{y}(x) = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i^* y_i k(x_i, x) + b^*\right)}$$

| 요소 | Linear SVM | Kernel SVM |
|------|-----------|-----------|
| Primal 변수 | $w \in \mathbb{R}^d$ | $w \in \mathcal{H}_k$ (무한 차원 가능) |
| Dual 변수 | $\alpha \in \mathbb{R}^n$ | 동일 |
| 예측 | $w^{*\top} x + b^*$ | $\sum_i \alpha_i^* y_i k(x_i, x) + b^*$ |
| Decision boundary | 초평면 | 곡면 |
| SV의 의미 | Margin 위 점 | Feature space margin 위 점 |
| 계산 비용 | $O(n d)$ primal | $O(n^2)$ Gram + $O(n^3)$ QP |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Kernel SVM의 예측 $\hat{y}(x)$를 **함수** $f^*(x) = \sum_i \alpha_i y_i k(x_i, x)$의 관점에서 $f^* \in \mathcal{H}_k$임을 보이고 노름 $\|f^*\|_{\mathcal{H}_k}$를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$f^* = \sum_i \alpha_i y_i k(\cdot, x_i) = \sum_i \alpha_i y_i k_{x_i} \in \mathcal{H}_0 \subset \mathcal{H}_k$.

$\|f^*\|_{\mathcal{H}_k}^2 = \sum_{i, j} (\alpha_i y_i)(\alpha_j y_j) \langle k_{x_i}, k_{x_j} \rangle = \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)$.

이는 정확히 dual objective의 quadratic 부분.

**해석**: SVM의 objective $\frac{1}{2} \|w\|^2 = \frac{1}{2} \|f^*\|_{\mathcal{H}_k}^2$. 즉 SVM은 "RKHS norm을 최소화하며 margin을 확보"하는 함수 학습.

</details>

**문제 2** (심화): RBF kernel로 SVM을 학습했을 때 decision boundary의 **"width"**(양쪽 margin 사이 거리)는 $\sigma$와 어떻게 관련되는가?

<details>
<summary>힌트 및 해설</summary>

Feature space의 margin은 $2 / \|w\|_{\mathcal{H}_k}$ = $2 / \sqrt{\sum \alpha_i \alpha_j y_i y_j k(x_i, x_j)}$.

원래 공간의 "width" 정의는 모호하지만, 다음 heuristic:

$f^*(x) = \sum_i \alpha_i y_i \exp(-\|x_i - x\|^2 / 2\sigma^2)$. $\sigma$ 크면 각 Gaussian bump가 넓음 → 경계가 부드럽고 넓은 margin. $\sigma$ 작으면 bump 좁음 → 경계가 국소적, margin도 각 점 근처에서만 의미.

**수치적 관찰**: $\sigma$를 2배 늘리면, decision boundary의 곡률 감소, effective margin 증가. 하지만 training error 증가 가능 (underfit 시작).

</details>

**문제 3** (ML 연결): Neural Tangent Kernel (NTK, Ch7-04)과 연결: 무한폭 NN의 SVM-like training은 어떻게 보이는가?

<details>
<summary>힌트 및 해설</summary>

Jacot et al. (2018): 무한폭 NN의 gradient flow는 fixed kernel $\Theta(x, y) = \lim_{\text{width}} \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle$에 대한 **kernel regression**과 동치.

MSE loss → kernel ridge regression.
Hinge loss → **kernel SVM (NTK kernel)**.

즉 "무한폭 NN으로 SVM 학습" = "NTK kernel의 kernel SVM".

**의미**:
- NN training의 일부는 **fixed kernel method**와 동등 → 수렴성·generalization 분석 kernel theory에서 도구 제공.
- 반대로 "NN이 kernel method보다 좋은 이유" = "feature learning (유한 width에서 kernel 자체가 학습됨)".

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 라그랑주 쌍대와 Dual Form](./02-lagrange-dual.md) | [04. Soft-margin SVM과 Hinge Loss ▶](./04-soft-margin-hinge.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
