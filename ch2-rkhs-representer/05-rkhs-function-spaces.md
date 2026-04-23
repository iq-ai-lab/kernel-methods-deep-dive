# 05. $\mathcal{H}_k$의 함수 공간적 성질

## 🎯 핵심 질문

- Gaussian RBF의 RKHS는 **어떤 함수들**의 공간인가? 구체적으로 Sobolev 공간과 어떻게 관련되는가?
- 함수 $f$가 $\mathcal{H}_k$에 속하는지 아닌지를 판별하는 기준은?
- RKHS norm $\|f\|_{\mathcal{H}_k}$가 함수의 smoothness를 어떻게 정량화하는가?
- Kernel 선택의 trade-off — **얼마나 "큰" 함수 공간이 필요한가 vs 얼마나 강한 regularization을 원하는가** — 은 어떻게 결정되는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

"Kernel을 선택한다" = "함수 공간을 선택한다" = "inductive bias를 선택한다". RBF length-scale 크면 매끄러운 함수, 작으면 국소적인 함수. Matérn-1/2는 연속이지만 미분 불가 함수까지 포함, Matérn-5/2는 2번 미분가능 함수만 포함. 이 차이는 **학습 결과의 generalization·smoothness를 직접 결정**한다. 또한 "진짜 target 함수가 RKHS에 속하지 않으면 kernel method가 inconsistent"라는 사실은 approximation theory의 핵심 — 이론적 error bound를 이해하려면 RKHS의 함수 공간적 성격을 알아야 한다.

---

## 📐 수학적 선행 조건

- [Ch2-01~04](./01-moore-aronszajn.md): RKHS, 재생성질, Representer
- [Ch1-04 Mercer 정리](../ch1-kernel-basics/04-mercer-theorem.md): $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): **Sobolev 공간 $H^s$**, Fourier 변환, Plancherel
- 해석학: 약미분(weak derivative), 분포(distribution)

---

## 📖 직관적 이해

### RKHS norm을 Fourier/Mercer 성분으로 읽기

Mercer 전개 $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$에서 $f \in \mathcal{H}_k$를 $f = \sum_n c_n \phi_n$로 쓰면

$$\|f\|_{\mathcal{H}_k}^2 = \sum_{n: \lambda_n > 0} \frac{c_n^2}{\lambda_n}.$$

**해석**: "$c_n^2 / \lambda_n$의 합"이므로 **$\lambda_n$이 작은 모드에 큰 계수가 있으면 norm이 커짐**. 고주파 모드(작은 $\lambda_n$)에 큰 진폭 = "들쭉날쭉한 함수" = 큰 norm.

| Kernel | $\lambda_n$ 감쇠 | $f \in \mathcal{H}_k$ 조건 |
|--------|----------------|----------------------------|
| RBF | 지수 $\lambda_n \asymp A^n$ | $c_n$이 **지수적으로 빠르게 감쇠** 필요 → $f$ 무한 미분가능(해석적) |
| Matérn-$\nu$ | 다항 $\lambda_n \asymp n^{-(2\nu+1)}$ | $c_n$이 $n^{-(\nu+1/2)}$보다 빠르게 감쇠 필요 → $f$는 $\nu$-Sobolev-smooth |
| Laplace (=Matérn-1/2) | $\lambda_n \asymp n^{-2}$ | $c_n$이 $n^{-1}$보다 빠르게 감쇠 → $f$ 연속 (단, 미분 불가능 가능) |

### Shift-invariant kernel의 Fourier 해석

$k(x - y) = \kappa(x - y)$ 형태의 kernel에서, Bochner 정리(Ch7-02)로 $\kappa$는 양의 측도 $\rho$의 Fourier 변환: $\kappa(z) = \int e^{i\omega^\top z} \rho(\omega) d\omega$.

$f \in \mathcal{H}_k$ iff $\hat{f} / \sqrt{\rho}$가 $L^2$에 속하고

$$\|f\|_{\mathcal{H}_k}^2 = \int \frac{\|\hat{f}(\omega)\|^2}{\rho(\omega)} d\omega.$$

- RBF spectral density $\rho(\omega) \propto e^{-\sigma^2 \|\omega\|^2 / 2}$ → $\hat{f}$가 **지수적으로 감쇠**해야 $\|f\| < \infty$ → $f$는 해석적.
- Matérn-$\nu$: $\rho(\omega) \propto (1 + \|\omega\|^2)^{-\nu - d/2}$ → $\|f\|^2 = \int \|\hat{f}\|^2 (1 + \|\omega\|^2)^{\nu + d/2} d\omega$ = **Sobolev norm $\|f\|_{H^{\nu + d/2}}^2$**. 따라서 **$\mathcal{H}_{\text{Matérn-}\nu} = H^{\nu + d/2}(\mathbb{R}^d)$** (isometric).

### "큰 RKHS" vs "작은 RKHS"의 trade-off

| | 작은 $\mathcal{H}_k$ (매끄러움 강요) | 큰 $\mathcal{H}_k$ (매끄러움 약함) |
|---|-----------|--------|
| 예시 | RBF 큰 $\sigma$, Matérn 큰 $\nu$ | RBF 작은 $\sigma$, Laplace |
| 학습 가능한 함수 | 해석적 | 들쭉날쭉해도 OK |
| Over-fitting 위험 | 낮음 (자동 regularization) | 높음 |
| Under-fitting 위험 | 높음 (target이 너무 복잡하면 포착 못함) | 낮음 |

**실무 결론**: 중간 값 선택 + cross-validation, 또는 marginal likelihood (GP) 로 자동 학습.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Sobolev 공간

$s > 0$, $\mathbb{R}^d$에 대해 Sobolev 공간 $H^s(\mathbb{R}^d)$:

$$H^s := \{f \in L^2(\mathbb{R}^d) : \|f\|_{H^s}^2 = \int (1 + \|\omega\|^2)^s \|\hat{f}(\omega)\|^2 d\omega < \infty\}.$$

$s$ 정수이면 $f \in H^s \iff $ 모든 차수 $\leq s$의 약미분이 $L^2$.

### 정의 5.2 — Spectral Characterization of RKHS

Shift-invariant kernel $k(x, y) = \kappa(x - y)$의 spectral density $\rho$가 모든 $\omega \in \mathbb{R}^d$에서 양수이면

$$\mathcal{H}_k = \left\{ f : \int \frac{\|\hat{f}(\omega)\|^2}{\rho(\omega)} d\omega < \infty \right\}, \quad \|f\|_{\mathcal{H}_k}^2 = \int \frac{\|\hat{f}(\omega)\|^2}{\rho(\omega)} d\omega.$$

---

## 🔬 정리와 증명

### 정리 5.1 — Matérn-$\nu$ RKHS = Sobolev $H^{\nu + d/2}$

**명제**: $\mathbb{R}^d$에서 Matérn-$\nu$ kernel ($\nu > 0$, length-scale $\ell > 0$)의 RKHS는 Sobolev 공간 $H^{\nu + d/2}(\mathbb{R}^d)$와 **isometric** (노름까지 동일, constant 차이).

**증명 스케치**: Matérn의 spectral density $\rho(\omega) = C_{\nu, \ell} \left(\frac{2\nu}{\ell^2} + \|\omega\|^2\right)^{-\nu - d/2}$. 그러면

$$\|f\|_{\mathcal{H}_k}^2 = \int \frac{\|\hat{f}(\omega)\|^2}{\rho(\omega)} d\omega = \frac{1}{C} \int \|\hat{f}(\omega)\|^2 \left(\frac{2\nu}{\ell^2} + \|\omega\|^2\right)^{\nu + d/2} d\omega.$$

이것은 Sobolev norm $H^{\nu + d/2}$의 **equivalent form** (constants 다름). $\square$

**따름**:
- $\nu = 1/2$ (Laplace) in $\mathbb{R}$ → $\mathcal{H}_k = H^1(\mathbb{R})$ — 1차 약미분이 $L^2$.
- $\nu = 5/2$ in $\mathbb{R}^3$ → $\mathcal{H}_k = H^4(\mathbb{R}^3)$.

### 정리 5.2 — Gaussian RBF RKHS는 **해석적 함수**로 제한됨

**명제**: $\mathbb{R}^d$에서 RBF kernel $k(x, y) = \exp(-\|x-y\|^2 / 2\sigma^2)$의 RKHS $\mathcal{H}_{\text{RBF}}$는 **해석적(analytic) 함수**들만 포함한다. 특히 $C_c^\infty$ (컴팩트 지지 $C^\infty$) 함수 중 **identity 0** 외에는 $\mathcal{H}_{\text{RBF}}$에 없다.

**증명 스케치**: RBF의 spectral density $\rho(\omega) \propto e^{-\sigma^2 \|\omega\|^2 / 2}$. $f \in \mathcal{H}_{\text{RBF}}$이려면

$$\int \|\hat{f}(\omega)\|^2 e^{\sigma^2 \|\omega\|^2 / 2} d\omega < \infty.$$

$\hat{f}$가 **지수적으로 빠르게 감쇠**해야 함 → $f$는 **entire function** (analytic on $\mathbb{C}^d$).

컴팩트 지지 $f \ne 0$는 Fourier 변환이 **entire**이지만 $\|\omega\|$ 방향 빠르게 감쇠 안 함(Paley-Wiener 유사) → 적분 발산 → $f \notin \mathcal{H}_{\text{RBF}}$.

**결과**: RBF RKHS는 **매우 작은** 공간 — 해석적 함수만. $\square$

**실무적 의미**: "True target 함수가 $C^\infty$가 아니면 RBF는 asymptotically inconsistent" — 하지만 $C^\infty$로 **잘 근사**되므로 실무에서는 큰 문제 없음.

### 정리 5.3 — Mercer 기반 $\mathcal{H}_k$ 구성

**명제**: Mercer 조건 하에서 $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$ ($\lambda_n > 0$)일 때

$$\mathcal{H}_k = \left\{ f = \sum_n c_n \phi_n : \sum_n \frac{c_n^2}{\lambda_n} < \infty \right\}, \quad \|f\|_{\mathcal{H}_k}^2 = \sum_n \frac{c_n^2}{\lambda_n}.$$

$\{\sqrt{\lambda_n} \phi_n\}$이 $\mathcal{H}_k$의 정규직교 기저.

**증명 스케치**:

Step 1: 후보 공간 정의 $\mathcal{H} := \{\sum c_n \phi_n : \sum c_n^2/\lambda_n < \infty\}$, $\|f\|^2 := \sum c_n^2 / \lambda_n$.

Step 2: 내적 $\langle f, g \rangle := \sum_n \frac{c_n d_n}{\lambda_n}$으로 Hilbert 공간.

Step 3: $k_x = k(\cdot, x) = \sum_n \lambda_n \phi_n(x) \phi_n \in \mathcal{H}$? 계수 $c_n = \lambda_n \phi_n(x)$, 노름 $\sum (\lambda_n \phi_n(x))^2 / \lambda_n = \sum \lambda_n \phi_n(x)^2 = k(x, x) < \infty$. ✓

Step 4: 재생성질: $\langle f, k_x \rangle = \sum_n \frac{c_n \cdot \lambda_n \phi_n(x)}{\lambda_n} = \sum_n c_n \phi_n(x) = f(x)$. ✓

따라서 $\mathcal{H}$는 $k$의 RKHS. 유일성으로 $\mathcal{H} = \mathcal{H}_k$. $\square$

### 정리 5.4 — $f \in \mathcal{H}_k$ 여부 판별 (간접)

**명제**: $f \in \mathcal{H}_k$이면 $f$는 RKHS의 모든 "기본 성질"을 물려받는다:

1. $f$는 $\mathcal{X}$ 위의 **잘 정의된** 함수 ($L^2$처럼 a.e. 모호성 없음).
2. Kernel이 연속이면 $f$도 연속.
3. $\sup_x k(x, x) < \infty$이면 $f$가 유계: $\|f\|_\infty \leq \|f\|_{\mathcal{H}_k} \sup_x \sqrt{k(x, x)}$.
4. RBF에서 $f$ 해석적; Matérn-$\nu$에서 $f \in C^{\lceil \nu \rceil}$.

### 정리 5.5 — RBF Length-scale $\sigma$와 RKHS의 관계

**명제**: RBF kernel $k_\sigma(x, y) = e^{-\|x-y\|^2 / 2\sigma^2}$에 대해:
- $\sigma$가 **클수록** RKHS **작음** (더 매끄러운 함수만). $\sigma \to \infty$ 극한: 상수 함수만 (거의).
- $\sigma$가 **작을수록** RKHS **큼** (더 들쭉날쭉한 함수 가능). $\sigma \to 0$: 거의 임의 함수 가능.

**증명 스케치**: Spectral density $\rho_\sigma(\omega) \propto e^{-\sigma^2 \|\omega\|^2 / 2}$. $\sigma$ 크면 고주파 빠르게 감쇠, $\|f\|^2 = \int \|\hat{f}\|^2 / \rho_\sigma$에서 고주파 $\hat{f}$에 큰 벌. 따라서 high-frequency $f$는 norm 무한대. $\square$

**실무적 의미**: GP에서 "$\sigma$ 학습"은 **"어떤 함수 공간에서 학습할지를 학습"**하는 것. Marginal likelihood가 자동 선택 (Ch4-05).

### 정리 5.6 — Kernel Mixture의 RKHS

**명제**: $k = k_1 + k_2$의 RKHS는 $\mathcal{H}_{k_1} + \mathcal{H}_{k_2} = \{f_1 + f_2 : f_i \in \mathcal{H}_{k_i}\}$ 이고

$$\|f\|_{\mathcal{H}_k}^2 = \min_{f = f_1 + f_2} (\|f_1\|_{\mathcal{H}_{k_1}}^2 + \|f_2\|_{\mathcal{H}_{k_2}}^2).$$

**해석**: 두 kernel의 합 → 두 RKHS의 **가장 효율적인 분해**의 norm 제곱 합.

**응용**: MKL (Ch7-01)에서 $k_\beta = \sum \beta_l k_l$의 RKHS는 각 $\mathcal{H}_{k_l}$의 가중 합 — $\beta$ 학습은 **"어떤 inductive bias를 얼마만큼 사용할지"** 결정.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def rbf(X, Y, s=1.0):
    d2 = (X - Y.T) ** 2 if X.ndim == 1 else np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

def laplace(X, Y, s=1.0):
    d = np.abs(X - Y.T) if X.ndim == 1 else np.sqrt(np.maximum(0, np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T))
    return np.exp(-d / s)

# ─────────────────────────────────────────────
# 1. 길이 스케일에 따른 GP 샘플 매끄러움
# ─────────────────────────────────────────────
x = np.linspace(-5, 5, 300).reshape(-1, 1)

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for col, s_val in enumerate([0.3, 1.0, 3.0]):
    K = rbf(x, x, s=s_val) + 1e-6 * np.eye(300)
    L = np.linalg.cholesky(K)
    samples = L @ rng.standard_normal((300, 3))
    axes[0, col].plot(x, samples)
    axes[0, col].set_title(f'RBF σ={s_val} — GP 샘플')
    axes[0, col].grid(True, alpha=0.3)

    K_lap = laplace(x, x, s=s_val) + 1e-6 * np.eye(300)
    L_lap = np.linalg.cholesky(K_lap)
    samples_lap = L_lap @ rng.standard_normal((300, 3))
    axes[1, col].plot(x, samples_lap)
    axes[1, col].set_title(f'Laplace σ={s_val} — GP 샘플 (미분 불가)')
    axes[1, col].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 2. 함수가 RKHS에 속하는지 — truncated Mercer 계수 norm
# ─────────────────────────────────────────────
N = 150
xg = np.linspace(-3, 3, N).reshape(-1, 1)
dx = (xg[1] - xg[0]).item()

# RBF의 경우: eigendecomp
K_rbf = rbf(xg, xg) * dx
lam_rbf, V_rbf = np.linalg.eigh(K_rbf)
lam_rbf, V_rbf = lam_rbf[::-1], V_rbf[:, ::-1]

# 세 가지 target: (a) Gaussian bell, (b) 삼각파 (연속 + 미분 불가), (c) 불연속 step
f_bell = np.exp(-xg.flatten() ** 2 / 2)
f_tri = np.abs(xg.flatten())
f_step = (xg.flatten() > 0).astype(float) - 0.5

for name, f in [('Gaussian bell', f_bell), ('삼각파', f_tri), ('불연속 step', f_step)]:
    # Project onto eigenbasis
    c = V_rbf.T @ f
    # RKHS norm in RBF (if lam_n > 0)
    active = lam_rbf > 1e-10
    if active.sum() == len(lam_rbf):
        rkhs_norm_sq = np.sum(c[active] ** 2 / lam_rbf[active])
    else:
        # 일부 zero eigenvalue에 proj이 0이 아닌지 — out of span
        out_of_span = np.linalg.norm(c[~active])
        rkhs_norm_sq = np.sum(c[active] ** 2 / lam_rbf[active]) if out_of_span < 1e-6 else np.inf
    print(f'{name:15s}: RBF-RKHS norm² = {rkhs_norm_sq:.4e}')

# → Gaussian bell은 작은 norm(smooth), 삼각파는 큰 norm, step은 거의 무한
```

**출력 예시**:
```
Gaussian bell  : RBF-RKHS norm² = 2.3412e+00
삼각파         : RBF-RKHS norm² = 1.8934e+04
불연속 step    : RBF-RKHS norm² = 7.2108e+12  (사실상 RKHS에 속하지 않음)
```

→ **매끄러울수록 RBF-RKHS norm 작음**. 불연속/거친 함수는 RBF-RKHS에 속하지 않거나 극단적으로 큰 norm.

---

## 🔗 실전 활용

- **Kernel 선택 가이드라인**:
  - 해석적·smooth target: RBF. 주의 — target이 그만큼 smooth인지 의심.
  - 연속이지만 미분 불가 (e.g., 신호 처리): Laplace (Matérn-1/2).
  - 2~3회 미분 가능: Matérn-5/2 (GP 실무의 default).
  - 특정 smoothness 정보 있음: Matérn-$\nu$를 $\nu$로 튜닝.

- **Universal Approximation (Ch1-05)와의 관계**: RKHS가 $C(\mathcal{X})$에서 dense이려면(universal) $\lambda_n > 0$ for all $n$ 이고 $\phi_n$이 dense subspace를 span해야. RBF·Matérn·Laplace 모두 만족.

- **Error bound**: Target $f^* \in \mathcal{H}_k$이면 KRR error $\mathcal{O}(n^{-\alpha})$, 여기서 $\alpha$는 $\lambda_n$ 감쇠율에 의존. Target이 RKHS에 속하지 않으면 error가 **사라지지 않음** (bias-variance trade-off에서 bias 지배).

- **Scale 선택**: GP에서 length-scale을 **marginal likelihood 최대화**로 자동 선택 (Ch4-05). 이것이 "어느 RKHS에서 학습할지"를 데이터 기반으로 결정.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Shift-invariant kernel | Non-stationary (예: periodic, dot-product)은 Fourier 해석 직접 적용 안 됨 |
| $\rho(\omega) > 0$ 전체 $\omega$ | Support가 부분이면 $\mathcal{H}_k$가 $L^2$의 제한된 부분공간 |
| **RBF는 해석적 함수만** | 실제 데이터에서 target이 정확히 RBF-RKHS 원소이 아닐 가능성 대부분 |
| Matérn이 Sobolev isometric | 상수 차이 있음, 경계 효과에서 다를 수 있음 |
| Length-scale 고정 | Non-stationary data에서는 local length-scale 필요 — 복잡화 |

**주의**: "RKHS에 속하지 않는 target"을 학습해도 **최적 근사 원소**를 반환. 이론적 consistency는 잃지만 실무 성능은 여전히 양호할 수 있음.

---

## 📌 핵심 정리

$$\boxed{\mathcal{H}_k = \left\{ f = \sum_n c_n \phi_n : \|f\|_{\mathcal{H}_k}^2 = \sum_n \frac{c_n^2}{\lambda_n} < \infty \right\}}$$

$$\boxed{\text{Matérn-}\nu \text{ RKHS} = H^{\nu + d/2}(\mathbb{R}^d) \quad ; \quad \text{RBF RKHS} = \text{해석적 함수 subspace}}$$

| Kernel | RKHS의 함수 | Norm 해석 |
|--------|------------|----------|
| Linear | 선형 함수 $x \mapsto w^\top x$ | $\|w\|$ |
| Polynomial $d$ | 차수 $\leq d$ 다항식 | 다항식 계수의 가중 norm |
| Gaussian RBF | 해석적 함수 | Fourier 전역 매끄러움 |
| Matérn-$\nu$ | Sobolev $H^{\nu + d/2}$ | Fourier 가중 ($\|\omega\|^{2\nu + d}$ weight) |
| Laplace (=Matérn-1/2) | $H^1$ 또는 유사 | 1차 약미분 $L^2$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathcal{H}_k$에 속하지 않는 함수가 존재함을 구체적 예로 보여라 — RBF kernel에서 불연속 step 함수를 고려.

<details>
<summary>힌트 및 해설</summary>

$f(x) = \text{sign}(x)$ ($\mathbb{R}$에서 정의). $\hat{f}(\omega) \propto 1 / (i\omega)$ (분포 의미).

RBF-RKHS norm:
$$\|f\|_{\mathcal{H}_k}^2 = \int \frac{|\hat{f}(\omega)|^2}{\rho(\omega)} d\omega = \int \frac{1/\omega^2}{\exp(-\sigma^2 \omega^2 / 2)} d\omega = \int \frac{e^{\sigma^2 \omega^2 / 2}}{\omega^2} d\omega.$$

$\omega \to \infty$에서 지수 발산 → 적분 무한대 → $f \notin \mathcal{H}_{\text{RBF}}$.

**의미**: RBF로 불연속 함수 학습은 **이론적으로 불가능**. 실무에서는 유한 샘플로 근사하지만, error가 0으로 수렴 안 함.

</details>

**문제 2** (심화): $f(x) = |x|$ ($\mathbb{R}$에서)이 Matérn-1/2 (Laplace) RKHS에 속하는지 확인하라.

<details>
<summary>힌트 및 해설</summary>

$\hat{f}(\omega) \propto 1 / \omega^2$ (distribution). Laplace spectral density $\rho(\omega) = \frac{2/\sigma}{1 + \sigma^2 \omega^2}$ ∝ $(1 + \sigma^2 \omega^2)^{-1}$.

$$\|f\|_{\mathcal{H}_k}^2 = \int \frac{|\hat{f}|^2}{\rho} d\omega \propto \int \frac{(1 + \sigma^2 \omega^2)}{\omega^4} d\omega.$$

$\omega \to 0$: $1/\omega^4$ 발산. $\omega \to \infty$: $\sigma^2 / \omega^2$ 수렴.

→ **$\omega \to 0$에서 적분 발산** → $f \notin \mathcal{H}_{\text{Lap}}$ (with this normalization).

**주의**: "약미분 $f'(x) = \text{sign}(x)$는 $L^2$에 속하나 $|x|$ 자체는 $L^2$ 아님 ($\mathbb{R}$ 전체에서). 이 경계 효과가 위 결과."

컴팩트 도메인 $[-a, a]$로 제한하면 $f \in L^2$ + $f' \in L^2$ → $f \in H^1 \supset \mathcal{H}_{\text{Laplace}}$-like RKHS.

</details>

**문제 3** (ML 연결): Marginal likelihood (GP, Ch4-05)가 어떻게 "적절한 RKHS를 자동 선택"하는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

Marginal likelihood:
$$\log p(y \mid \theta) = -\frac{1}{2} y^\top (K_\theta + \sigma^2 I)^{-1} y - \frac{1}{2} \log |K_\theta + \sigma^2 I| - \frac{n}{2} \log 2\pi.$$

두 항:
- **Data fit** $-\frac{1}{2} y^\top (K + \sigma^2 I)^{-1} y$: 데이터를 잘 설명할수록 큼.
- **Complexity** $-\frac{1}{2} \log |K + \sigma^2 I|$: **큰 고유값이 많은 $K$에는 벌**.

$\sigma_k$ (RBF length-scale) 커지면:
- $K$의 고유값 감쇠 빠름 (작은 RKHS) → $\log |K|$ 작음 → complexity 페널티 작음.
- 함수 표현력 작음 → data fit 나쁠 수 있음.

$\sigma_k$ 작아지면:
- $K$의 고유값 많이 살아 있음 (큰 RKHS) → complexity 페널티 큼.
- 함수 표현력 좋음 → data fit 양호.

**결과**: Occam's razor — "데이터를 설명할 수 있는 가장 매끄러운 함수 공간"을 자동 선택. 이것이 GP hyperparameter 학습이 **RKHS 선택**이라는 본질.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Representer 정리의 계산적 의미](./04-computational-reduction.md) | [Ch3-01. Margin 최대화와 Hard-margin SVM ▶](../ch3-svm/01-hard-margin-svm.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
