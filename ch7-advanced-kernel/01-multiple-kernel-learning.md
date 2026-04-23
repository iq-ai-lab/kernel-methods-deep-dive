# 01. Multiple Kernel Learning (MKL)

## 🎯 핵심 질문

- 여러 kernel의 **볼록 결합** $k_\beta = \sum_l \beta_l k_l$, $\beta_l \geq 0$, $\sum_l \beta_l = 1$이 왜 자동 PD인가?
- **$\beta$ 학습**이 어떤 최적화 문제로 formulate되는가 — SDP vs gradient-based?
- **SimpleMKL** (Rakotomamonjy et al. 2008)의 subgradient 기반 효율적 알고리즘은?
- $\ell_p$-norm MKL, non-sparse MKL의 변형들은 언제 유리한가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

"어떤 kernel을 써야 하는가?"는 kernel method의 근본적 문제. MKL은 **"데이터가 스스로 kernel을 선택"**하게 만드는 automatic approach. (i) **Multi-source data**: 서로 다른 modality (이미지 + 텍스트)의 kernel을 결합, (ii) **Multi-scale**: 여러 length-scale RBF의 자동 조합, (iii) **Interpretable**: $\beta_l$이 각 kernel의 중요도. 이론적으로 **convex optimization** 문제로 formulate되며, 실무에서는 Bioinformatics (protein function prediction, disease classification), computer vision (feature fusion) 에서 standard.

---

## 📐 수학적 선행 조건

- [Ch1-03 Kernel 연산](../ch1-kernel-basics/03-kernel-operations.md): Sum of PD kernels is PD
- [Ch3-02 SVM dual](../ch3-svm/02-lagrange-dual.md)
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Semidefinite Programming (SDP), subgradient, dual decomposition

---

## 📖 직관적 이해

### MKL의 Setup

Candidate kernels $\{k_1, \ldots, k_L\}$ (각 PD). Combined kernel:

$$k_\beta(x, y) = \sum_{l=1}^L \beta_l k_l(x, y), \quad \beta_l \geq 0.$$

Ch1-03 정리 3.1로 $k_\beta$도 PD.

**SVM with MKL**:

$$\min_{f, b, \beta} \frac{1}{2} \|f\|_{\mathcal{H}_{k_\beta}}^2 + C \sum_i \max(0, 1 - y_i (f(x_i) + b)), \quad \beta_l \geq 0, \sum_l \beta_l = 1.$$

**$\beta$ 학습**: Simplex 위에서 구현.

### 각 Kernel의 기여 분해

$f \in \mathcal{H}_{k_\beta}$는 block 분해 가능: $f = \sum_l f_l$, $f_l \in \mathcal{H}_{k_l}$, $\|f\|^2_{k_\beta} = \sum_l \|f_l\|^2_{k_l} / \beta_l$.

**해석**: 각 kernel이 "서로 다른 함수 공간 기여". $\beta_l$ 작으면 해당 kernel의 기여에 큰 페널티 → $\beta_l \to 0$에서 해당 kernel 완전 제외 ("sparse kernel").

### SDP Formulation (Lanckriet et al. 2004)

**원래 MKL**: SDP (semidefinite program) 형태. $O(n^6)$ — 실무 불가.

$$\min_{K \succeq 0, \text{tr}(K) = 1} \max_\alpha \{\cdots\} \text{ (SVM dual in } K\text{)}.$$

### SimpleMKL (Rakotomamonjy 2008)

**아이디어**: Alternating minimization.

1. $\beta$ 고정 → standard SVM with $k_\beta$ → $\alpha$ 학습.
2. $\alpha$ 고정 → $\beta$에 대해 dual objective gradient descent.
3. 수렴까지 반복.

**복잡도**: $O(n^2 L T)$, $T$ = iterations. 실무적.

### $\ell_p$-Norm MKL

$\sum \beta_l = 1$ 대신 $\sum \beta_l^p = 1$ ($p \geq 1$). 

- $p = 1$: Sparse $\beta$ (L1 regularization).
- $p = 2$: Dense $\beta$ (L2, Ridge-like).
- $p = \infty$: 모든 $\beta_l$ 같음.

**선택**:
- Irrelevant kernels 많으면 $p = 1$ (sparse, feature selection).
- 모든 kernel이 조금씩 유용하면 $p = 2$ (non-sparse, 안정적).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Combined Kernel

$k_\beta(x, y) := \sum_{l=1}^L \beta_l k_l(x, y), \beta \in \Delta^{L-1}$ ($L$-simplex).

### 정의 1.2 — MKL-SVM

$$\min_{f_1, \ldots, f_L, b, \beta} \frac{1}{2} \sum_l \frac{\|f_l\|^2_{\mathcal{H}_l}}{\beta_l} + C \sum_i \max\left(0, 1 - y_i \left(\sum_l f_l(x_i) + b\right)\right)$$

s.t. $\beta_l \geq 0$, $\sum_l \beta_l = 1$.

### 정의 1.3 — MKL Dual

SVM dual with combined kernel:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k_\beta(x_i, x_j), \quad 0 \leq \alpha \leq C, y^\top \alpha = 0.$$

Outer minimize over $\beta \in \Delta$.

### 정의 1.4 — SimpleMKL Objective

$$J(\beta) := \max_\alpha \{\text{SVM dual with } k_\beta\}.$$

$\beta$에 대해 subgradient descent on $J(\beta)$ (Convex in $\beta$).

---

## 🔬 정리와 증명

### 정리 1.1 — Combined Kernel의 PD성

**명제**: $\beta_l \geq 0$이면 $k_\beta = \sum_l \beta_l k_l$은 PD.

**증명**: Ch1-03 정리 3.1 (PD kernel의 양 계수 합). $\square$

### 정리 1.2 — Block Decomposition

**명제**: $\mathcal{H}_{k_\beta}$의 모든 원소 $f$는 unique 분해

$$f = \sum_{l=1}^L f_l, \quad f_l \in \mathcal{H}_{k_l}, \quad \|f\|^2_{k_\beta} = \sum_l \frac{\|f_l\|^2_{k_l}}{\beta_l}$$

를 갖는다 (각 분해에서 norm 최소화).

**증명 개요**: $k_\beta$의 RKHS가 direct sum $\bigoplus \mathcal{H}_{k_l}$ (weighted)으로 표현. Aronszajn의 sum kernel formula. Details: Berlinet & Thomas-Agnan (2004) §6. $\square$

### 정리 1.3 — MKL은 Convex Optimization

**명제**: MKL-SVM의 joint optimization $(\beta, \alpha)$가 **convex** (in terms of $(\beta, f)$ with appropriate formulation).

**증명 아이디어**: Lagrangian decomposition + $\beta$에 대한 convexity. Rakotomamonjy 2008, Bach 2008. $\square$

**함의**: Global optimum 보장, local optima 걱정 없음.

### 정리 1.4 — SimpleMKL 알고리즘

**알고리즘**:

1. Initialize $\beta = (1/L, \ldots, 1/L)$.
2. Repeat until convergence:
   a. $\alpha = \arg\max$ SVM dual with $k_\beta$ (standard SVM solver).
   b. Compute $\nabla_\beta J = -\frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k_l(x_i, x_j)$ for each $l$ (SVM objective의 $\beta_l$에 대한 편미분).
   c. Projection onto simplex $\Delta$ with subgradient descent.

**수렴**: $O(1/\epsilon^2)$ iterations for $\epsilon$-optimal (convex + subgradient). 각 iter SVM solve가 주 비용.

### 정리 1.5 — $\ell_p$-Norm MKL Generalization

**명제**: Constraint $\sum \beta_l^p \leq 1$ ($p \geq 1$)로 일반화. $p = 1$: sparse (Lasso-like). $p \to \infty$: uniform.

**선택 가이드**:
- Many irrelevant kernels: $p = 1$.
- All relevant: $p = 2$ 또는 higher.
- Cross-validation으로 결정.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_blobs

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. 데이터: RBF가 잘 맞는 moons
# ─────────────────────────────────────────────
X, y01 = make_moons(n_samples=150, noise=0.15, random_state=0)
y = 2 * y01 - 1
n = len(y)

# Candidate kernels: RBF with different length-scales + polynomial
def rbf(X, Y, s=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

def poly(X, Y, c=1, d=2):
    return (X @ Y.T + c) ** d

# 5 candidate kernels
kernels = {
    'RBF σ=0.1': lambda X, Y: rbf(X, Y, 0.1),
    'RBF σ=0.5': lambda X, Y: rbf(X, Y, 0.5),
    'RBF σ=1.0': lambda X, Y: rbf(X, Y, 1.0),
    'RBF σ=5.0': lambda X, Y: rbf(X, Y, 5.0),
    'Poly d=2 ': lambda X, Y: poly(X, Y, c=1, d=2),
}

K_list = [kfn(X, X) for kfn in kernels.values()]
L = len(K_list)

# ─────────────────────────────────────────────
# 2. SimpleMKL (간이 구현)
# ─────────────────────────────────────────────
def solve_svm_dual(K_combined, y, C=1.0):
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K_combined
    a = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, cp.psd_wrap(Q))),
                       [a >= 0, a <= C, y @ a == 0])
    prob.solve(solver='CLARABEL')
    return np.asarray(a.value).flatten(), prob.value

def project_simplex(v):
    """Project v onto simplex {beta ≥ 0, sum = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cumsum = np.cumsum(u)
    rho = np.where(u - (cumsum - 1) / (np.arange(1, n + 1)) > 0)[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    lam = (cumsum[rho] - 1) / (rho + 1)
    return np.maximum(v - lam, 0)

# SimpleMKL
beta = np.ones(L) / L
C = 1.0
lr = 0.05

history = {'obj': [], 'beta': [beta.copy()]}
for it in range(50):
    K_comb = sum(b * K for b, K in zip(beta, K_list))
    alpha, obj = solve_svm_dual(K_comb, y, C=C)
    
    # Gradient of SVM dual objective w.r.t. β_l
    # d/dβ_l [Σα - 1/2 α^T Y K_β Y α] = -1/2 α^T Y K_l Y α
    Y = np.diag(y)
    grads = np.array([-0.5 * alpha @ (Y @ K_l @ Y) @ alpha for K_l in K_list])
    # We minimize -dual = maximize dual, so grad ascent on dual = grad descent on -dual
    # Actually β minimizes max_α SVM_dual(β). Subgradient of J(β) = ∇_β SVM_dual(β) at optimal α.
    # J is convex in β; to minimize, descent on grads
    beta_new = project_simplex(beta - lr * grads)
    
    history['obj'].append(obj)
    history['beta'].append(beta_new.copy())
    beta = beta_new

print('Final β:')
for name, b in zip(kernels.keys(), beta):
    print(f'  {name}: {b:.4f}')

# Visualize β evolution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['obj'])
plt.xlabel('Iter'); plt.ylabel('SVM dual value')
plt.title('SimpleMKL: Objective convergence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
beta_hist = np.array(history['beta'])
for l, name in enumerate(kernels.keys()):
    plt.plot(beta_hist[:, l], label=name)
plt.xlabel('Iter'); plt.ylabel('β')
plt.title('β evolution')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 3. 단일 kernel과 비교
# ─────────────────────────────────────────────
from sklearn.model_selection import cross_val_score

results = []
for name, kfn in kernels.items():
    # Custom kernel SVM via precomputed
    K_train = kfn(X, X)
    svc = SVC(kernel='precomputed', C=C)
    scores = cross_val_score(svc, K_train, y, cv=5)
    results.append({'kernel': name, 'CV acc': scores.mean(), 'β': beta[list(kernels.keys()).index(name)]})

# MKL CV
K_mkl = sum(b * K for b, K in zip(beta, K_list))
svc_mkl = SVC(kernel='precomputed', C=C)
mkl_score = cross_val_score(svc_mkl, K_mkl, y, cv=5).mean()
print(f'\nMKL (combined) CV accuracy: {mkl_score:.4f}')

import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
print(f'\n→ MKL combines the best kernels automatically')
```

**출력 예시**:
```
Final β:
  RBF σ=0.1: 0.0521
  RBF σ=0.5: 0.7432
  RBF σ=1.0: 0.1821
  RBF σ=5.0: 0.0021
  Poly d=2 : 0.0205

MKL (combined) CV accuracy: 0.9667

kernel      CV acc   β
RBF σ=0.1   0.8400   0.0521
RBF σ=0.5   0.9533   0.7432
RBF σ=1.0   0.9067   0.1821
RBF σ=5.0   0.5533   0.0021
Poly d=2    0.7600   0.0205

→ MKL combines the best kernels automatically
```

→ MKL이 RBF σ=0.5에 큰 가중치 (최고 단일 kernel) + σ=1.0 약간 추가 → 가장 좋은 accuracy.

---

## 🔗 실전 활용

- **Multi-modal learning**: Image kernel + Text kernel + Audio kernel — MKL로 자동 가중치.
- **Bioinformatics**: Protein의 여러 특성 (sequence, structure, function) kernel을 결합.
- **Computer vision**: SIFT, HOG, deep features 등 여러 representation의 kernel 결합.
- **Hyperparameter automation**: RBF의 length-scale 튜닝을 여러 $\sigma$ 제공 후 MKL로.
- **Library**: `shogun-mkl`, MATLAB-MKL. Python은 직접 구현 또는 부분 support.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| $L$ (kernel 수) 늘어나면 | $O(L n^2)$ memory + SimpleMKL iterations |
| Convex but SDP-like | SDP solver는 $O(n^6)$ — 실무 불가, approximation 필요 |
| **Inappropriate candidates** | "Bad" kernel만 많으면 MKL도 좋은 결과 못 낸다 |
| Sparsity vs accuracy | $\ell_1$-MKL은 interpretable but 덜 accurate 가끔 |
| Scalability | $n > 10^4$에서 SVM solver 자체가 병목 |

---

## 📌 핵심 정리

$$\boxed{k_\beta(x, y) = \sum_{l=1}^L \beta_l k_l(x, y), \beta \in \Delta^{L-1}}$$

$$\boxed{\min_\beta \max_\alpha \sum_i \alpha_i - \frac{1}{2} \alpha^\top (Y K_\beta Y) \alpha \text{ s.t. } \alpha \text{ constraints}}$$

| Approach | Complexity | 특성 |
|----------|-----------|------|
| SDP (Lanckriet) | $O(n^6)$ | 이론적, 실무 불가 |
| SimpleMKL | $O(n^2 L T)$ | 실용적 subgradient |
| $\ell_p$-MKL | 동일 | Sparsity 제어 |
| SPG-MKL | Faster | Stochastic PG |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MKL에서 $\beta_l = 0$이 되는 kernel은 해에 전혀 기여하지 않는다. 이것이 **feature selection**과 어떻게 대응되는가?

<details>
<summary>힌트 및 해설</summary>

$\ell_1$-MKL (sum = 1) = **Lasso on kernels**. Sparse $\beta$ → 일부 kernel만 활성.

**Feature selection 관점**:
- Each kernel = "feature representation" of data.
- $\beta_l > 0$ = "이 representation이 유용".
- $\beta_l = 0$ = "이 representation 무시".

**응용**: Multi-source data에서 어떤 source가 유용한지 자동 판단.

**한계**: 각 kernel이 feature 뭉치 (bundle)라서 fine-grained individual feature selection 불가. 개별 feature selection은 ARD (Ch4-05) 또는 sparse linear model.

</details>

**문제 2** (심화): MKL과 Deep Kernel Learning (Ch7-03)의 관계는?

<details>
<summary>힌트 및 해설</summary>

**MKL**: Fixed candidate kernels $\{k_l\}$의 convex combination.

**Deep Kernel Learning**: 하나의 kernel $k(x, y) = k_0(\phi_\theta(x), \phi_\theta(y))$, $\phi_\theta$ = learnable NN feature.

**비교**:
- MKL: Discrete choice (어느 kernel을 얼마나).
- DKL: Continuous learning (어떤 feature를 학습).

**공통**: 둘 다 "data가 스스로 kernel을 선택"하는 시도.

**통합**: Deep MKL — NN feature의 여러 layer의 kernel을 MKL로 결합 (Wilson et al. 2016 extension).

**현대 trend**: DKL이 더 popular (NN의 표현력). MKL은 **interpretable kernel 결정** 필요할 때.

</details>

**문제 3** (ML 연결): MKL이 **Transformer의 multi-head attention**과 어떻게 개념적으로 연관되는가?

<details>
<summary>힌트 및 해설</summary>

**Multi-head attention**: $H$ heads, 각 head $\text{head}_h = \text{softmax}(Q_h K_h^\top / \sqrt{d}) V_h$. Concat heads + linear projection.

**Kernel perspective** (Tsai et al. 2019):
- Attention $\propto \exp(Q K^\top)$ = "exponential kernel" on $(Q, K)$.
- Multi-head = "multiple kernels" with different learned projections.
- Concat + linear projection = "weighted combination" of kernel outputs.

**유사성**:
- 둘 다 **multiple representations** 제공.
- **Weighted combination**으로 최종 예측.
- Learnable **kernel/projection**.

**차이**:
- MKL: Kernel은 **fixed** (RBF, Linear, Poly), $\beta$만 학습.
- Attention: Kernel parameter (Q, K, V projection)도 학습.

**통찰**: Attention = "learnable MKL with feature projections". Transformer의 성공 = "fixed kernel → learnable kernel with deep features".

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch6-05. Kernel Embedding 일반화](../ch6-mmd/05-kernel-embedding-generalizations.md) | [02. Random Features (Rahimi & Recht 2007) ▶](./02-random-features.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
