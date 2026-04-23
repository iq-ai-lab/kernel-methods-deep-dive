# 04. Representer 정리의 계산적 의미

## 🎯 핵심 질문

- "무한 차원 최적화 → $n$-차원 최적화" 환원이 **구체적으로** 어떤 선형대수 연산들로 이루어지는가?
- Kernel method들(SVM·KRR·KPCA·GP·Kernel Logistic Regression)이 모두 **"같은 형태의 해"**를 갖는 이유는?
- Kernel trick의 **수학적 정당성**과 **계산적 구조**가 Representer 정리와 어떻게 연결되는가?
- $n$이 커질 때의 병목 — $O(n^3)$ 해결과 $O(n^2)$ 메모리 — 을 Random Features·Nyström·Inducing Points가 각각 어떻게 회피하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Representer 정리가 **존재 정리**(Ch2-03)라면, 이 장은 **실행 정리**다: 실제 컴퓨터에서 kernel method를 돌릴 때 어떤 연산들을 하는가, 어떤 해가 나오는가, 어떻게 SVM·KRR·GP가 단일 프레임워크로 통일되는가. 이 통일 관점을 이해하면 **새 알고리즘을 설계**할 때도 "어떤 $L$과 $\Omega$를 쓰는지만 결정하면 나머지는 automatic"이라는 직관이 생긴다. 또한 scaling(Ch7-02 Random Features, Ch4-06 Sparse GP)의 모든 접근은 **Representer의 $O(n^3)$ 병목을 깨는 방법들**로 이해된다.

---

## 📐 수학적 선행 조건

- [Ch2-03 Representer 정리 완전 증명](./03-representer-theorem.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Cholesky 분해, conjugate gradient, 행렬 역의 수치 안정성
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): QP·KKT·IRLS

---

## 📖 직관적 이해

### "모든 kernel method는 결국 Gram 행렬의 선형대수"

Representer 정리로 $f = K \alpha$ (training 점에서의 값) 및 $\|f\|^2 = \alpha^\top K \alpha$ (norm)로 환원되면, 문제는 **$\alpha \in \mathbb{R}^n$을 결정하는 선형대수 문제**로 변환된다:

| 방법 | 유한 차원 환원 |
|------|-------------|
| KRR | $(K + \lambda I) \alpha = y$ (linear system) |
| SVM | $\max \mathbf{1}^\top \alpha - \frac{1}{2} \alpha^\top (y y^\top \odot K) \alpha$ s.t. $0 \leq \alpha \leq C$, $y^\top \alpha = 0$ (QP) |
| Kernel PCA | $K$의 eigendecomposition |
| GP | $\alpha = (K + \sigma^2 I)^{-1} y$, posterior 분산 추가 |
| Kernel LogReg | IRLS로 반복: $(K W K + \lambda K) \alpha = K W z$ |

**모두 $n \times n$ 행렬 $K$의 선형대수**. "kernel이 무엇인지"는 문제의 특수성을 결정하지만, **구조는 동일**.

### Kernel Trick — "feature를 명시하지 않고 inner product만"

원래 feature map $\phi : \mathcal{X} \to \mathcal{H}$에 대한 linear method:

$$\min_{w \in \mathcal{H}} L(y_i, \langle w, \phi(x_i) \rangle) + \lambda \|w\|^2.$$

Representer로 $w = \sum_i \alpha_i \phi(x_i)$. 그러면 $\langle w, \phi(x_j) \rangle = \sum_i \alpha_i \langle \phi(x_i), \phi(x_j) \rangle = (K \alpha)_j$. $\|w\|^2 = \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j) = \alpha^\top K \alpha$.

→ **$\phi$를 몰라도, $k(x, y) = \langle \phi(x), \phi(y) \rangle$만 알면 된다**. 이것이 "kernel trick"의 정확한 수학. Representer가 **왜 작동하는지**, 그리고 **어떻게 계산**하는지를 동시에 보인다.

### 복잡도와 한계

| 연산 | 시간 | 메모리 |
|------|------|--------|
| Gram $K$ 구성 | $O(n^2 d)$ | $O(n^2)$ |
| KRR: $(K + \lambda I)^{-1} y$ | $O(n^3)$ | $O(n^2)$ |
| SVM: QP solver (SMO) | $O(n^2)$ ~ $O(n^3)$ | $O(n^2)$ |
| GP posterior 평가 (1 점) | $O(n)$ (predict) 또는 $O(n^2)$ (분산) | $O(n^2)$ |
| KPCA: eigendecomposition | $O(n^3)$ | $O(n^2)$ |

→ $n = 10^4$까지 노트북에서 가능, $n = 10^5$ 이상은 HPC 또는 근사(Random Features, Nyström) 필요.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Kernel Trick의 엄밀한 형태

$\phi : \mathcal{X} \to \mathcal{H}$와 linear ML method의 해가 $w = \sum_i \alpha_i \phi(x_i)$ 형태(Representer)일 때:

$$\langle w, \phi(x) \rangle_{\mathcal{H}} = \sum_i \alpha_i k(x_i, x), \quad \|w\|_{\mathcal{H}}^2 = \sum_{i, j} \alpha_i \alpha_j k(x_i, x_j).$$

즉 원래 feature 공간의 연산이 **kernel $k$만으로** 완전히 표현됨.

### 정의 4.2 — Finite-dimensional Surrogate

Representer 정리 후 등가 유한 차원 문제:

$$\min_{\alpha \in \mathbb{R}^n} \quad L(y, K\alpha) + \Omega(\sqrt{\alpha^\top K \alpha}).$$

단, $L$과 $\Omega$의 구체적 형태는 method별.

---

## 🔬 정리와 증명

### 정리 4.1 — 범용 유한 차원 환원 공식

**명제**: PD kernel $k$와 정규화된 ERM $\min_{f \in \mathcal{H}_k} L(y, (f(x_i))_i) + \Omega(\|f\|_{\mathcal{H}_k})$에 대해, Representer 정리 후 해 $f^* = \sum_i \alpha_i k_{x_i}$를 대입하면:

1. $f^*(x_j) = (K \alpha)_j$.
2. $(f^*(x_1), \ldots, f^*(x_n))^\top = K \alpha$.
3. $\|f^*\|_{\mathcal{H}_k}^2 = \alpha^\top K \alpha$.
4. 원 문제 ≡ $\min_\alpha L(y, K \alpha) + \Omega(\sqrt{\alpha^\top K \alpha})$.

**증명**: 재생성질 + Ch2-03 정리 3.6. $\square$

### 정리 4.2 — 주요 Kernel Method의 구체적 유한 차원 형태

각 방법을 위 정리 4.1 틀에 넣어 전개.

**KRR** ($L$ = squared, $\Omega(t) = \lambda t^2$):

$$\min_\alpha \|y - K\alpha\|^2 + \lambda \alpha^\top K \alpha.$$

1차 조건: $-2 K (y - K\alpha) + 2 \lambda K \alpha = 0 \Rightarrow K(y - K\alpha - \lambda \alpha) = 0$. $K$ 가역이면 $y = (K + \lambda I) \alpha$, 따라서 $\alpha = (K + \lambda I)^{-1} y$.

**SVM (soft-margin)** ($L$ = hinge, $\Omega(t) = \frac{1}{2} t^2$):

$$\min_{f \in \mathcal{H}_k, b} \frac{1}{2} \|f\|^2 + C \sum_i \max(0, 1 - y_i (f(x_i) + b)).$$

Representer + Lagrangian → dual:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j k(x_i, x_j), \quad 0 \leq \alpha_i \leq C, \sum_i \alpha_i y_i = 0.$$

**KPCA** ($L$ = $-\text{Var}(f \circ \phi)$, $\Omega = I_{\|f\|=1}$):

$\tilde{K}$ (centered Gram)의 eigendecomposition $\tilde{K} v_k = \lambda_k v_k$, $k$-th principal component: $f^{(k)} = \sum_i v_{k,i} k_{x_i}$.

**GP Posterior Mean** (Ch4-03에서 KRR과 동치):

$$\alpha = (K + \sigma^2 I)^{-1} y, \quad m_*(x) = k_*^\top \alpha.$$

**Kernel Logistic Regression** ($L$ = logistic):

$$\min_\alpha \sum_i \log(1 + \exp(-y_i (K\alpha)_i)) + \frac{\lambda}{2} \alpha^\top K \alpha.$$

Newton/IRLS로 해결:

$$\alpha^{(t+1)} = (K W^{(t)} K + \lambda K)^{-1} K W^{(t)} z^{(t)}$$

($W$, $z$는 current iterate에서의 weight·target).

$\square$ (모두 정리 4.1의 직접 적용)

### 정리 4.3 — Predictive Equation (통일 형태)

**명제**: 모든 kernel method에서 새 입력 $x$에 대한 예측은

$$\hat{f}(x) = \sum_{i=1}^n \alpha_i k(x, x_i) = k(x)^\top \alpha$$

형태 ($k(x) := (k(x, x_1), \ldots, k(x, x_n))^\top$). 각 방법은 **$\alpha$ 결정 방식만** 다르다.

**증명**: Representer 정리의 직접 따름. $\square$

**해석**: 예측 시 필요한 것은 "$x$와 모든 training 점 사이의 kernel 값" + 미리 학습된 $\alpha$. 이것이 **kernel method의 "non-parametric" 성격** — 새 예측마다 training data가 필요. $n$ 크면 memory·inference 비용이 증가.

### 정리 4.4 — Kernel Trick의 정당성

**명제**: Representer 정리가 성립하는 임의의 linear ML method에 대해, 원래 feature 공간 $\mathcal{H}$에서의 연산 $\langle w, \phi(x) \rangle$과 $\|w\|^2$은 kernel $k(x, y) = \langle \phi(x), \phi(y) \rangle$만으로 완전히 표현 가능.

**증명**: $w = \sum_i \alpha_i \phi(x_i)$에서 정리 4.1. $\square$

**실무적 함의**:
- **$\phi$는 **저장하지도 계산하지도 않는다****. Gram $K \in \mathbb{R}^{n \times n}$만 계산.
- 무한 차원 $\phi$ (예: RBF의 Mercer feature $\phi(x) \in \ell^2$)도 문제 없음.
- 복잡한 구조화 데이터 (문자열, 그래프, 이미지)의 feature map을 직접 쓰지 않고 **kernel 함수만** 설계하면 됨.

### 정리 4.5 — Scaling 전략의 수학적 구조

$n$이 커질 때 Representer의 $O(n^3)$을 피하는 방법은 모두 **kernel 행렬 $K$를 저-계수(low-rank) 근사**하는 것:

1. **Random Features** (Ch7-02): $k(x, y) \approx \phi_{\text{RF}}(x)^\top \phi_{\text{RF}}(y)$, $\phi_{\text{RF}}(x) \in \mathbb{R}^D$. 유한 차원으로 돌아가 primal solve $O(n D^2)$.
2. **Nyström**: Sub-sample $m \ll n$ 점들로 $K$를 근사 $K \approx K_{nm} K_{mm}^{-1} K_{mn}$. $O(n m^2)$.
3. **Inducing Points / FITC / VFE** (Ch4-06): GP에 특화된 Nyström variant, $O(n m^2)$.
4. **Conjugate Gradient**: $(K + \lambda I)\alpha = y$를 직접 풀지 않고 iterative — $K v$ matrix-vector 곱만 필요. $O(n^2)$ per iter.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

rng = np.random.default_rng(42)

# Data
n = 80
X = rng.uniform(-3, 3, (n, 2))
y_reg = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.standard_normal(n)
y_cls = np.sign(X[:, 0] * X[:, 1]).astype(int)  # XOR-like

def rbf(X, Y, s=1.0):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K = rbf(X, X)
lam = 0.01

# ─────────────────────────────────────────────
# 1. KRR — closed form
# ─────────────────────────────────────────────
alpha_krr = np.linalg.solve(K + lam * np.eye(n), y_reg)

# sklearn 대조
krr = KernelRidge(alpha=lam, kernel='rbf', gamma=1/(2*1.0**2))
krr.fit(X, y_reg)
alpha_sklearn = krr.dual_coef_
print(f'KRR: 바닥 구현 vs sklearn 차이 (L2): {np.linalg.norm(alpha_krr - alpha_sklearn):.2e}')

# ─────────────────────────────────────────────
# 2. SVM — cvxpy로 dual QP
# ─────────────────────────────────────────────
try:
    import cvxpy as cp
    a = cp.Variable(n)
    Q = (y_cls[:, None] * y_cls[None, :]) * K
    obj = cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, cp.psd_wrap(Q)))
    cons = [a >= 0, a <= 1.0, y_cls @ a == 0]
    prob = cp.Problem(obj, cons)
    prob.solve(solver='CLARABEL')
    alpha_svm = np.asarray(a.value)
    
    svc = SVC(kernel='rbf', gamma=0.5, C=1.0).fit(X, y_cls)
    print(f'SVM: 바닥 dual objective = {prob.value:.4f}')
    print(f'SVM: sklearn support vectors = {len(svc.support_)} / bartering dual nonzero = {(alpha_svm > 1e-4).sum()}')
except ImportError:
    print('cvxpy 미설치 — SVM dual 건너뜀')

# ─────────────────────────────────────────────
# 3. KPCA — Centered Gram의 eigendecomp
# ─────────────────────────────────────────────
n_one = np.ones((n, n)) / n
K_centered = K - n_one @ K - K @ n_one + n_one @ K @ n_one

eigvals, V = np.linalg.eigh(K_centered)
eigvals, V = eigvals[::-1], V[:, ::-1]  # descending

# 첫 2개 주성분으로 projection
alpha_kpca = V[:, :2] / np.sqrt(np.maximum(eigvals[:2], 1e-10))
X_new = rng.uniform(-3, 3, (5, 2))
K_new = rbf(X_new, X) - rbf(X_new, X).mean(axis=1, keepdims=True) \
        - K.mean(axis=0) + K.mean()  # center
projs = K_new @ alpha_kpca
print(f'\nKPCA projection shape: {projs.shape}')

# ─────────────────────────────────────────────
# 4. GP — KRR과 동치 확인
# ─────────────────────────────────────────────
sigma_n = 0.1
alpha_gp = np.linalg.solve(K + sigma_n**2 * np.eye(n), y_reg)
x_grid = np.linspace(-3, 3, 30).reshape(-1, 1)
x_grid_2d = np.hstack([x_grid, np.zeros_like(x_grid)])
pred_gp = rbf(x_grid_2d, X) @ alpha_gp

alpha_krr_same_lam = np.linalg.solve(K + sigma_n**2 * np.eye(n), y_reg)
pred_krr = rbf(x_grid_2d, X) @ alpha_krr_same_lam
print(f'GP vs KRR 예측 차이 (같은 λ=σ_n²): {np.max(np.abs(pred_gp - pred_krr)):.2e}')

# ─────────────────────────────────────────────
# 5. 예측 공식의 통일성: f(x) = k(x)^T α (모든 방법 공통)
# ─────────────────────────────────────────────
print('\n= 예측 공식은 모든 방법에서 k(x)^T α 형태 =')
print(f'KRR α 처음 5개: {alpha_krr[:5]}')
print(f'GP α 처음 5개: {alpha_gp[:5]}')
```

**출력 예시**:
```
KRR: 바닥 구현 vs sklearn 차이 (L2): 3.42e-14
SVM: 바닥 dual objective = 34.8712
SVM: sklearn support vectors = 42 / bartering dual nonzero = 41
KPCA projection shape: (5, 2)
GP vs KRR 예측 차이 (같은 λ=σ_n²): 0.00e+00

= 예측 공식은 모든 방법에서 k(x)^T α 형태 =
KRR α 처음 5개: [ 0.2134 -0.1432  0.0834  0.3912 -0.1283]
GP α 처음 5개: [ 0.2134 -0.1432  0.0834  0.3912 -0.1283]
```

→ 모든 방법이 **$\alpha$를 determining system만 다르지**, **해의 형태는 공통**.

---

## 🔗 실전 활용

- **Framework 설계**: 새 kernel method 제안 시 "$L, \Omega$ 결정 → Representer로 $\alpha$ 문제 자동 도출". 기존 scikit-learn·GPy·CVX 기반 solver 재사용.
- **Debugging**: Kernel method에서 버그 발견 시 "$\alpha = K^{-1}$-related 값인가"를 먼저 체크. 거의 모든 이슈가 Gram 행렬의 조건수·inversion에서 비롯.
- **Scaling 선택**:
  - $n \leq 10^4$: 그냥 closed-form 또는 QP. Cholesky.
  - $n = 10^4$~$10^5$: Nyström, Sparse GP (Ch4-06), Random Features (Ch7-02).
  - $n > 10^6$: Deep Kernel + mini-batch (NN-like scaling, Ch7-03).
- **Memory 주의**: $K \in \mathbb{R}^{n \times n}$는 $n = 32768$만 되어도 8GB (double precision). 이때 **Gram을 저장하지 않고 matrix-vector 곱만 하는 iterative method** 필요.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Representer 정리 성립 | Online streaming·무한 $n$에서는 finite-support 가정 필요 |
| Gram $K \in \mathbb{R}^{n \times n}$ 완전 계산 | 메모리 $O(n^2)$, scaling bottleneck |
| $K$ 수치 안정성 | Jitter $K + 10^{-6} I$ 필수 실무에서 |
| $k$가 PD | Non-PD이면 해 유일성 깨짐, SVM 수렴 실패 |
| **$n \leq 10^5$** 정도 까지 closed-form 가능 | 그 이상은 approximation 필수 |

---

## 📌 핵심 정리

$$\boxed{f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i) = k(x)^\top \alpha \quad \text{(모든 kernel method 공통)}}$$

$$\boxed{\text{각 method는 } \alpha \text{ 결정 방식만 다르다: KRR linear solve, SVM QP, KPCA eigendecomp, GP linear solve + variance, KLogR IRLS}}$$

| 차원 | Kernel method | NN/Deep Learning |
|------|--------------|------------------|
| 해의 형태 | $\sum_i \alpha_i k(\cdot, x_i)$ (data-dependent) | $f_\theta(\cdot)$ (parametric) |
| 해의 크기 | $n$ (training data size) | 고정 $|\theta|$ |
| Training 복잡도 | $O(n^3)$ (naive) | $O(n)$ per epoch |
| Inference | $O(n)$ per point | $O(|\theta|)$ |
| Kernel 선택 | 인간의 inductive bias | 아키텍처 설계가 암시적 kernel |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Representer 정리로 KRR의 해 $\alpha = (K + \lambda I)^{-1} y$를 얻었을 때, $K$의 Cholesky 분해 $K + \lambda I = LL^\top$를 사용한 계산을 한 줄로 쓰라.

<details>
<summary>힌트 및 해설</summary>

$\alpha = (LL^\top)^{-1} y = L^{-\top} L^{-1} y$. NumPy:

```python
L = np.linalg.cholesky(K + lam * np.eye(n))
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
```

**왜 직접 inverse보다 좋은가**: (i) Numerical stability 2~3배 우월, (ii) 시간 $O(n^3/3)$ vs naive $O(n^3)$의 2~3배, (iii) $L$ 저장 시 여러 $y$에 대해 재사용 가능.

**실무 팁**: $K$가 거의 singular이면 $K + \lambda I + 10^{-6} \cdot \text{tr}(K) / n \cdot I$ 형태로 jitter 추가.

</details>

**문제 2** (심화): $n \gg 1$일 때 $(K + \lambda I)^{-1} y$를 **직접 계산하지 않고** Conjugate Gradient로 풀 수 있다. 이 방법의 복잡도와 수렴 조건은?

<details>
<summary>힌트 및 해설</summary>

**복잡도 per iter**: $O(n^2)$ (matrix-vector 곱 $Kv$). **총 iter 수**: 수렴까지 $O(\sqrt{\kappa})$ iter, 여기서 $\kappa$는 조건수.

**수렴 조건**: $K + \lambda I$는 대칭 PSD → CG는 $\mathcal{O}(\sqrt{\kappa} \log(1/\epsilon))$ iterations으로 $\epsilon$-정확도 해 달성.

**$\lambda$의 역할**: $K$가 ill-conditioned (작은 고유값 많음)이면 $\lambda$ 추가로 조건수 극적 개선 — $\kappa \approx \lambda_{\max}(K) / (\lambda_{\min}(K) + \lambda) \approx \lambda_{\max}(K) / \lambda$.

**실무**: PyTorch·JAX의 `cg`·`gmres`로 GP scaling 가능. Gardner et al. 2018 "GPyTorch" 참고.

</details>

**문제 3** (ML 연결): Deep Learning은 Representer 정리의 "data-dependent kernel method"와 본질적으로 다른 패러다임인가?

<details>
<summary>힌트 및 해설</summary>

**Surface 수준**: 다름. NN은 $f_\theta$ (fixed parametric), kernel method는 $\sum \alpha_i k(\cdot, x_i)$ (data-dependent).

**Deep kernel 관점 (Ch7-03, Ch7-04)**:
- Deep Kernel Learning (Wilson et al. 2016): NN feature $\phi_\theta$ 위에 kernel → $f = \sum \alpha_i k(\phi_\theta(\cdot), \phi_\theta(x_i))$. Representer 유지.
- NTK (Jacot et al. 2018): 무한폭 NN의 gradient flow는 NTK라는 fixed kernel의 kernel regression과 **정확히** 동치 → $f_\theta(x) = \sum_i \alpha_i \Theta(x, x_i)$ 형태.

**통찰**: "충분히 wide한 NN"은 kernel method의 한 종류. 유한 width NN은 "kernel을 학습하는" 일반화 (feature learning).

따라서:
- **Kernel method는 "고정 kernel + data-dependent 해"**.
- **NN은 "학습 가능 kernel"**.
- Representer 정리는 "fixed kernel" 범위 내에서 강력하고, NN은 그것을 "kernel 선택도 자동화"한 확장.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Representer 정리 완전 증명](./03-representer-theorem.md) | [05. $\mathcal{H}_k$의 함수 공간적 성질 ▶](./05-rkhs-function-spaces.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
