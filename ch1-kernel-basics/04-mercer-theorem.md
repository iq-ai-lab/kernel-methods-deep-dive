# 04. Mercer 정리의 서술과 해석

## 🎯 핵심 질문

- Mercer 정리는 정확히 어떤 kernel에 대해, 어떤 분해 $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$를 보장하는가?
- 이 분해에서 나오는 feature map $\phi(x) = (\sqrt{\lambda_n} \phi_n(x))_n$이 왜 **implicit하게 무한차원**인가?
- "kernel trick"의 **수학적 정당성**이 Mercer에서 어떻게 나오는가?
- Mercer의 가정(컴팩트성·연속성)을 약화시키면 무엇이 필요한가? (Moore-Aronszajn과의 관계는 Ch2에서)

---

## 🔍 왜 이 개념이 ML에서 중요한가

"kernel trick이 작동한다"는 주장은 본질적으로 "$k(x, y) = \langle \phi(x), \phi(y) \rangle$인 $\phi$가 존재한다"를 전제로 한다. Mercer 정리는 **이 전제를 실제로 구성**해 준다 — PD kernel은 $\ell^2$ 공간으로 향하는 구체적인 feature map을 가지며, 그 성분은 integral operator의 고유함수다. 이 분해는 (i) **Gaussian kernel의 feature가 무한차원**이라는 ML 문구를 수학적으로 정당화하고, (ii) **Kernel PCA**(Ch5-02)의 기하학을 직접 기술하며, (iii) **RKHS norm의 smoothness 해석**(Ch2-05)을 제공하고, (iv) **Random Features**(Ch7-02)가 왜 "eigendecomposition을 유한 차원으로 근사"하는지를 통합한다.

---

## 📐 수학적 선행 조건

- [Ch1-01 PD kernel의 정의](./01-positive-definite-kernel.md)
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): **컴팩트 자기수반 작용소**의 스펙트럴 정리, Hilbert-Schmidt 작용소
- 측도론: $L^2(\mathcal{X}, \mu)$, Borel 측도, 측도공간의 컴팩트성
- 해석학: 일양 수렴, 연속함수 공간 $C(\mathcal{X})$

---

## 📖 직관적 이해

### "무한차원 feature"의 기원 — 고유함수 분해

PD kernel $k$로부터 **integral operator** $T_k : L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu)$를

$$(T_k f)(x) := \int_{\mathcal{X}} k(x, y) f(y) d\mu(y)$$

로 정의하면, $k$가 연속이고 $\mathcal{X}$가 컴팩트이면 $T_k$는 **compact + self-adjoint + positive**. 따라서 스펙트럴 정리가 적용되어 **음이 아닌 고유값** $\lambda_n \geq 0$과 $L^2$에서 직교정규인 고유함수 $\phi_n$들이 존재:

$$T_k \phi_n = \lambda_n \phi_n, \quad \langle \phi_n, \phi_m \rangle_{L^2} = \delta_{nm}.$$

Mercer 정리는 **kernel 자체**가 이 고유분해로부터 다시 조립됨을 보장한다:

$$k(x, y) = \sum_{n=1}^\infty \lambda_n \phi_n(x) \phi_n(y) \quad \text{(일양 수렴)}.$$

이 표현을 $\phi(x) := (\sqrt{\lambda_1} \phi_1(x), \sqrt{\lambda_2} \phi_2(x), \ldots) \in \ell^2$로 읽으면

$$k(x, y) = \sum_n \sqrt{\lambda_n} \phi_n(x) \cdot \sqrt{\lambda_n} \phi_n(y) = \langle \phi(x), \phi(y) \rangle_{\ell^2}$$

— 곧 **Mercer feature map**. 이것이 ML에서 말하는 "RBF kernel의 implicit feature가 무한차원"의 정확한 의미다.

### 왜 "일양 수렴"이 중요한가

Mercer의 강한 주장은 급수 $\sum_n \lambda_n \phi_n(x) \phi_n(y)$이 **$\mathcal{X} \times \mathcal{X}$에서 일양 수렴**한다는 것이다. 이 덕분에:

1. $k$가 연속임을 유지하며 분해.
2. 유한 부분합 $\sum_{n=1}^N \lambda_n \phi_n(x) \phi_n(y)$이 $k$의 균일한 근사 (Random Features의 근간).
3. 함수 $f = \sum_n c_n \phi_n$의 kernel norm이 $\sum_n c_n^2 / \lambda_n$로 표현 (Ch2-05).

### RBF의 Mercer 분해 — 구체적 예

$\mathcal{X} = \mathbb{R}$, $\mu$가 가우시안 측도 $\mathcal{N}(0, 1/4)$ 일 때 $k(x, y) = \exp(-\gamma (x - y)^2)$의 Mercer 분해는 **Hermite 함수**로 주어진다:

$$k(x, y) = \sum_{n=0}^\infty \lambda_n \phi_n(x) \phi_n(y), \quad \lambda_n = (1 - A)^2 A^n, \quad \phi_n(x) = H_n(x) e^{-(A/2) x^2} \cdot \text{const}$$

($A$는 $\gamma$와 $\mu$ variance로 결정). Hermite 다항식 $H_n$이 $L^2(\mu)$의 직교정규 기저이고, 대응하는 고유값은 **지수적으로 감쇠**. 이 감쇠 속도가 GP smoothness의 수학적 근거.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Integral Operator $T_k$

컴팩트 측도공간 $(\mathcal{X}, \mu)$와 연속 함수 $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$에 대해

$$T_k : L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu), \quad (T_k f)(x) := \int_{\mathcal{X}} k(x, y) f(y) d\mu(y).$$

### 정의 4.2 — $T_k$의 자기수반·양정치

- **자기수반**: $\langle T_k f, g \rangle = \langle f, T_k g \rangle$ iff $k(x, y) = k(y, x)$.
- **양정치**: $\langle T_k f, f \rangle \geq 0$ for all $f \in L^2$.

### 정의 4.3 — Mercer Condition (Sufficient)

$\mathcal{X}$는 컴팩트 Hausdorff 공간, $\mu$는 Borel 유한 측도 (예: Lebesgue on $[a, b]$), $k$는 **연속 PD kernel**.

### 정의 4.4 — Feature Map (Mercer)

$$\phi(x) := (\sqrt{\lambda_n} \phi_n(x))_{n \geq 1} \in \ell^2$$

여기서 $\lambda_n, \phi_n$은 $T_k$의 고유값·고유함수.

---

## 🔬 정리와 증명

### 정리 4.1 — Hilbert-Schmidt 성질

**명제**: 컴팩트 $\mathcal{X}$와 연속 $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$에 대해 $T_k$는 **Hilbert-Schmidt 작용소**이고 따라서 compact. 특히

$$\|T_k\|_{\text{HS}}^2 = \int_{\mathcal{X} \times \mathcal{X}} \|k(x, y)\|^2 d\mu(x) d\mu(y) < \infty.$$

**증명 스케치**: 컴팩트 $\mathcal{X}$와 유한 측도 $\mu$, 연속 $k$는 유계 → $\|k\|_\infty < \infty$ → $\|T_k\|_{\text{HS}}^2 \leq \|k\|_\infty^2 \mu(\mathcal{X})^2 < \infty$. HS 작용소는 compact. $\square$

### 정리 4.2 — 자기수반 + 양정치

**명제**: 연속 PD kernel $k$에 대해 $T_k$는 자기수반 compact **양정치** 작용소.

**증명**:

- 자기수반: $\langle T_k f, g \rangle = \iint k(x, y) f(y) g(x) d\mu = \langle f, T_k g \rangle$ (Fubini + $k$ 대칭).
- 양정치: $\langle T_k f, f \rangle = \iint k(x, y) f(x) f(y) d\mu d\mu$. 유한 샘플 $\{x_i\}$로 근사하면 $\sum_{i, j} f(x_i) f(x_j) k(x_i, x_j) (\Delta\mu)^2 \geq 0$ (PD 정의), 극한도 $\geq 0$. $\square$

### 정리 4.3 — Mercer 정리 (주 정리)

**명제**: $\mathcal{X}$가 컴팩트 Hausdorff, $\mu$가 유한 Borel 측도 (충분 지지), $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$가 **연속 PD kernel**이라 하자. 그러면:

1. $T_k$는 직교정규 고유함수 $\{\phi_n\}_{n \geq 1} \subset L^2(\mathcal{X}, \mu)$와 음이 아닌 고유값 $\lambda_1 \geq \lambda_2 \geq \cdots \geq 0$ ($\lambda_n \to 0$)을 갖는다.
2. 각 고유함수 $\phi_n$ (with $\lambda_n > 0$)은 **연속**이다.
3. 다음의 Mercer 전개가 $\mathcal{X} \times \mathcal{X}$에서 **절대·일양 수렴**한다:

$$k(x, y) = \sum_{n=1}^\infty \lambda_n \phi_n(x) \phi_n(y).$$

**증명 스케치** (핵심 단계만):

**Step 1 (스펙트럴 정리 적용)**: 정리 4.2로부터 $T_k$는 compact + self-adjoint + positive. 이러한 작용소는 직교정규 고유분해를 갖는다:

$$T_k = \sum_n \lambda_n \langle \cdot, \phi_n \rangle \phi_n, \quad \lambda_n \geq 0, \quad \lambda_n \to 0.$$

**Step 2 (고유함수의 연속성)**: $\lambda_n > 0$이면 $\phi_n = \lambda_n^{-1} T_k \phi_n(x) = \lambda_n^{-1} \int k(x, y) \phi_n(y) d\mu(y)$. $k$가 연속이고 컴팩트 $\mathcal{X}$ 위 유계이므로, dominated convergence로 $\phi_n$도 연속.

**Step 3 (pointwise 전개)**: $L^2$에서 $T_k (\cdot, y)의 $y$-section은

$$k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y) \quad \text{in } L^2(\mathcal{X}, \mu).$$

**Step 4 (단조 수렴 + Dini의 정리)**: 부분합 $k_N(x, y) := \sum_{n=1}^N \lambda_n \phi_n(x) \phi_n(y)$에 대해 $r_N(x, y) := k(x, y) - k_N(x, y)$도 연속 PD kernel(고유값 $\lambda_{N+1}, \lambda_{N+2}, \ldots$의 잔여 분해). 따라서 $r_N(x, x) \geq 0$이고 $r_N(x, x) = \sum_{n > N} \lambda_n \phi_n(x)^2$. $N \to \infty$에서 $r_N(x, x) \to 0$ pointwise.

**Dini**: 컴팩트 공간 위 연속함수들의 **단조 감소**하는 수열이 pointwise로 연속 극한으로 수렴하면, 수렴은 **일양**. 따라서 $r_N(x, x) \to 0$ 일양. Cauchy-Schwarz로 $\|r_N(x, y)\|^2 \leq r_N(x, x) r_N(y, y)$ 이므로 $r_N(x, y) \to 0$ 일양. $\square$

### 정리 4.4 — Mercer Feature Map의 $\ell^2$ 속성

**명제**: Mercer 조건 하에서 $\phi(x) := (\sqrt{\lambda_n} \phi_n(x))_{n \geq 1}$은 각 $x$에 대해 $\ell^2$에 속하고

$$k(x, y) = \langle \phi(x), \phi(y) \rangle_{\ell^2}.$$

**증명**: Mercer 전개의 $y = x$로 두면 $k(x, x) = \sum_n \lambda_n \phi_n(x)^2 < \infty$ (일양 수렴이 $k(x, x)$가 유계이므로). 따라서 $\phi(x) \in \ell^2$. 내적은 Mercer 전개 그 자체. $\square$

### 정리 4.5 — Kernel Trick의 정당성

**명제**: Mercer PD kernel $k$에 대해, 임의의 유한 샘플 $\{x_i\}$의 그람 행렬 $K$는 **유한 차원 근사 feature**

$$\phi^{(N)}(x) := (\sqrt{\lambda_1} \phi_1(x), \ldots, \sqrt{\lambda_N} \phi_N(x)) \in \mathbb{R}^N$$

의 Gram $K^{(N)}$과 다음을 만족한다:

$$\max_{i, j} |K_{ij} - K^{(N)}_{ij}| \to 0 \quad (N \to \infty), \quad \text{일양하게 데이터 독립적으로}.$$

따라서 **"kernel trick"은 무한차원 feature map의 완전한 수치적 근사**이고, 근사 오차가 $N$에 따라 기하학적으로 관리된다 (고유값 감쇠율에 따라).

**증명**: Mercer의 일양 수렴 $\sup_{x, y} |k(x, y) - k_N(x, y)| \to 0$에서 즉시. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────
# 1. RBF kernel의 수치적 Mercer 분해
# ─────────────────────────────────────────────
# 격자 위에서 T_k를 근사해 고유값 분해
N = 100
x_grid = np.linspace(-3, 3, N)
dx = x_grid[1] - x_grid[0]

def rbf(X, Y, s=1.0):
    d2 = (X[:, None] - Y[None, :]) ** 2
    return np.exp(-d2 / (2 * s**2))

K = rbf(x_grid, x_grid)

# T_k의 행렬 근사 = K · dx (measure의 quadrature)
# 고유분해 T_k φ = λ φ  →  K φ · dx = λ φ
# 대칭화를 위해 K · dx의 고유값 분해
lam, V = np.linalg.eigh(K * dx)
lam = lam[::-1]
V = V[:, ::-1]
# 고유함수는 √(1/dx) · V[:, n]  (정규화)
phi = V / np.sqrt(dx)

# 고유값 감쇠 확인
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].semilogy(lam[:30], 'o-'); ax[0].set_title('RBF kernel 고유값 λ_n (지수 감쇠)')
ax[0].set_xlabel('n'); ax[0].grid(True, alpha=0.3)

# 첫 6개 고유함수 (Hermite와 닮음)
for n in range(6):
    ax[1].plot(x_grid, phi[:, n], label=f'φ_{n}')
ax[1].set_title('RBF kernel의 Mercer 고유함수')
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.show()

# ─────────────────────────────────────────────
# 2. Mercer 전개의 일양 수렴 — 부분합의 잔차
# ─────────────────────────────────────────────
errors = []
for N_keep in [1, 3, 5, 10, 20, 50]:
    K_approx = phi[:, :N_keep] @ np.diag(lam[:N_keep]) @ phi[:, :N_keep].T
    err = np.max(np.abs(K - K_approx))
    errors.append(err)
    print(f'N = {N_keep:3d} → max|K - K_N| = {err:.4e}')

# ─────────────────────────────────────────────
# 3. 고유값 감쇠율은 kernel smoothness와 직접 연결
# ─────────────────────────────────────────────
# RBF (무한번 미분) → 지수 감쇠
# Laplace (연속) → 다항 감쇠
def laplace(X, Y, s=1.0):
    d = np.abs(X[:, None] - Y[None, :])
    return np.exp(-d / s)

K_lap = laplace(x_grid, x_grid)
lam_lap = np.sort(np.linalg.eigvalsh(K_lap * dx))[::-1]

plt.figure(figsize=(7, 4))
plt.semilogy(lam[:50], 'o-', label='RBF (지수 감쇠)')
plt.semilogy(np.maximum(lam_lap[:50], 1e-12), 's-', label='Laplace (다항 감쇠)')
plt.xlabel('n'); plt.ylabel('λ_n (log)')
plt.title('kernel smoothness → 고유값 감쇠율')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

**관찰**:
- RBF는 $\lambda_n \approx A^n$ (지수) — 소수의 주요 모드로 전체 kernel 근사 가능.
- Laplace는 $\lambda_n \approx n^{-p}$ (다항) — 더 많은 모드 필요, 함수가 덜 매끄러움과 일치.
- $N_{\text{keep}} = 20$이면 RBF에서 `max|K - K_N| < 10^{-10}$ 수준 달성.

---

## 🔗 실전 활용

- **Kernel PCA (Ch5-02)**: Mercer 분해의 **처음 $k$개 고유함수**로 projection — 비선형 차원축소. Centered Gram의 eigendecomposition = 실제 구현.
- **Random Features (Ch7-02)**: Mercer 고유분해를 몬테카를로로 근사 — $\phi_{\text{RF}}(x) = \sqrt{2/D}(\cos(\omega_i^\top x + b_i))_i$. $D$개 샘플 → 유한 차원 feature로 $O(n D^2)$ 계산.
- **Nyström 근사**: Training 집합의 sub-sample $\tilde{x}_1, \ldots, \tilde{x}_m$으로 구성한 sub-Gram의 eigendecomposition을 전체 근사로 사용. 실무 GP scaling의 주력 도구.
- **GP smoothness 해석**: GP sample은 $f(x) = \sum_n z_n \sqrt{\lambda_n} \phi_n(x)$, $z_n \sim \mathcal{N}(0, 1)$ i.i.d. 고유값 감쇠가 빠를수록 함수가 매끄러움.

---

## ⚖️ 가정과 한계

| 가정 | 완화 시 필요 |
|------|-------------|
| $\mathcal{X}$ 컴팩트 Hausdorff | $\mathcal{X} = \mathbb{R}^d$에서는 Mercer가 직접 적용 안됨 — **Moore-Aronszajn**(Ch2-01) 사용 |
| $k$ 연속 | 연속성 없으면 고유함수도 연속 아닐 수 있음; 일양 수렴 실패 |
| $\mu$ 유한 Borel | 측도 선택이 분해 결정 — RBF의 Hermite 분해는 가우시안 측도에서만 |
| PD (positive semi-definite) | CPD에서는 기본 Mercer 미적용 |

**주의**: Mercer의 $\phi$는 **측도 $\mu$에 의존**한다. 같은 kernel이라도 다른 $\mu$를 쓰면 다른 고유함수가 나온다. 실무에서는 "데이터가 뽑힌 분포"를 $\mu$로 쓰는 것이 합리적 (empirical Mercer = Gram eigendecomposition).

---

## 📌 핵심 정리

$$\boxed{k(x, y) = \sum_{n=1}^\infty \lambda_n \phi_n(x) \phi_n(y) \quad \text{(일양 수렴, Mercer)}}$$

$$\boxed{\phi(x) = (\sqrt{\lambda_n} \phi_n(x))_{n \geq 1} \in \ell^2 \quad ; \quad k(x, y) = \langle \phi(x), \phi(y) \rangle_{\ell^2}}$$

| 개념 | 의미 |
|------|------|
| **Integral operator $T_k$** | Kernel을 컴팩트 자기수반 양정치 작용소로 승격 |
| **고유값 $\lambda_n$** | 감쇠율이 kernel smoothness 결정 |
| **고유함수 $\phi_n$** | $L^2(\mu)$의 직교정규 기저 중 kernel-adapted |
| **일양 수렴** | 유한 부분합으로 균일 근사 가능 |
| **Implicit 무한차원 feature** | Mercer feature $\phi(x) \in \ell^2$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathcal{X} = \{x_1, \ldots, x_n\}$ (유한 집합)이고 counting measure $\mu(\{x_i\}) = 1$이면 Mercer 전개는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

$L^2(\mathcal{X}, \mu) = \mathbb{R}^n$. $T_k f = K f$ (Gram 행렬 곱). $T_k$의 고유분해 = $K$의 스펙트럴 분해: $K = \sum_n \lambda_n v_n v_n^\top$.

Mercer 전개: $k(x_i, x_j) = K_{ij} = \sum_n \lambda_n v_n(i) v_n(j)$. 이것이 바로 **스펙트럴 분해의 행렬 성분 표현**.

**통찰**: Mercer 정리는 유한 샘플에서는 **단순한 스펙트럴 정리**이고, 무한 $\mathcal{X}$에서 그 일반화.

</details>

**문제 2** (심화): Mercer 고유값 $\lambda_n$의 감쇠율이 kernel smoothness와 어떻게 관련되는지 증명/직관적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

Kernel이 매끄러울수록 "고주파 모드 $\phi_n$ (큰 $n$)"에 거의 기여하지 않음 → $\lambda_n$ 감쇠 빠름.

구체적으로 $\mathcal{X} = [0, 1]$에서 RBF의 고유함수는 Hermite-류, 고유값 $\lambda_n \asymp A^n$ (지수).
Laplace는 Sobolev-류, $\lambda_n \asymp n^{-2}$.
Matérn-$\nu$: $\lambda_n \asymp n^{-(2\nu + 1)}$.

**Sampling theorem 관점**: 고유함수 $\phi_n$은 빈도 $\sim \sqrt{n}$의 진동 함수. 매끄러운 kernel은 고주파를 가중치 낮게 → $\lambda_n$ 작다.

**결과**: GP의 sample function smoothness = kernel 고유값 감쇠율에 의해 결정.

</details>

**문제 3** (ML 연결): Kernel PCA가 어떻게 Mercer 분해의 **유한 샘플 근사**로 해석되는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

Training 집합 $\{x_1, \ldots, x_n\}$에서:
1. Gram $K = [k(x_i, x_j)]$ 계산.
2. 스펙트럴 분해 $K = U \Lambda U^\top$.
3. 새 점 $x$에 대한 projection onto $k$-th principal component:
   $$\phi^{(k)}(x) = \frac{1}{\sqrt{n \tilde{\lambda}_k}} \sum_{i=1}^n U_{ik} k(x_i, x).$$

이것은 Mercer의 **empirical 근사**: 진짜 $T_k$ 대신 empirical measure $\hat{\mu} = \frac{1}{n} \sum \delta_{x_i}$에서의 integral operator $\hat{T}_k$의 고유분해.

$n \to \infty$에서 Nyström 근사 이론으로 $\hat{\lambda}_k / n \to \lambda_k$, $\phi^{(k)} \to $ Mercer 고유함수. (Braun, 2006).

**즉**: "Kernel PCA = 유한 샘플에서의 Mercer 분해 근사".

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Kernel 연산 — Sum·Product·Composition](./03-kernel-operations.md) | [05. Characteristic Kernel과 Universal Kernel ▶](./05-characteristic-universal.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
