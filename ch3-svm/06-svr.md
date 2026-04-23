# 06. SVM Regression (SVR)

## 🎯 핵심 질문

- SVR은 **$\epsilon$-insensitive loss** $\max(0, \|y - f(x)\| - \epsilon)$로 회귀 문제를 푼다 — 이 loss는 왜 sparse 해를 만드는가?
- $\epsilon$-tube의 **기하학적 해석**은? Inside-tube는 loss 0, outside는 linear penalty.
- Primal → dual 유도 — classification SVM과 무엇이 다른가? 왜 $\alpha, \alpha^*$ 두 세트가 필요한가?
- $\epsilon$과 $C$의 상호작용 — bias-variance trade-off에서 어떻게 결정되는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Support Vector Regression은 "분류를 위한 SVM을 회귀로 확장"하는 자연스러운 일반화. **특징**: (i) **$\epsilon$-tube 안의 점들은 loss 0** → sparse SV, interpretable, (ii) **outlier 영향 제한** (bounded SV, $\alpha_i \leq C$), (iii) classification과 동일하게 **kernel trick** 적용 가능. 실무에서는 (i) 금융 시계열·센서 데이터 회귀, (ii) Bayesian optimization의 surrogate, (iii) noise-robust한 non-linear 회귀에서 사용. KRR·GP와 비교했을 때 "sparsity + kernel"의 독특한 조합을 제공.

---

## 📐 수학적 선행 조건

- [Ch3-01~05](./01-hard-margin-svm.md): SVM primal·dual, soft-margin, KKT
- [Ch2-03 Representer 정리](../ch2-rkhs-representer/03-representer-theorem.md)
- 볼록 최적화: Lagrangian with multiple Slack variables

---

## 📖 직관적 이해

### $\epsilon$-Insensitive Loss — "충분히 가까우면 무시"

회귀 문제에서 loss는 일반적으로 MSE $(y - f(x))^2$ 또는 MAE $|y - f(x)|$. SVR은 다음을 제안:

$$\ell_\epsilon(y, f(x)) := \max(0, |y - f(x)| - \epsilon).$$

- $|y - f(x)| \leq \epsilon$: loss = 0 (inside tube, acceptable).
- $|y - f(x)| > \epsilon$: loss = $|y - f(x)| - \epsilon$ (linear penalty, outside tube).

**Sparsity 직관**: Inside-tube 점은 loss 0 → gradient 없음 → support vector 아님. Outside-tube와 tube 경계 점들만 support vector. 일반적으로 SV 개수 $\ll n$.

### 두 Slack 변수 — Upper / Lower Violation

$y_i - f(x_i) > \epsilon$ (target 위로 넘음): $\xi_i > 0$ (upper slack).
$f(x_i) - y_i > \epsilon$ (target 아래로): $\xi_i^* > 0$ (lower slack).

둘 다 동시에 non-zero일 수 없음 ($y = f$에 대해 $y - f > \epsilon$과 $f - y > \epsilon$이 둘 다 성립 불가). 따라서 $\xi_i \cdot \xi_i^* = 0$.

### Classification SVM과의 대조

| 측면 | Classification SVM | SVR |
|------|---------|-----|
| Output | $\{-1, +1\}$ | $\mathbb{R}$ |
| Loss | Hinge $\max(0, 1 - yf)$ | $\epsilon$-insensitive $\max(0, |y - f| - \epsilon)$ |
| Slack | $\xi_i$ (한 개) | $\xi_i, \xi_i^*$ (두 개) |
| Dual 변수 | $\alpha_i$ | $\alpha_i, \alpha_i^*$ (두 개) |
| Prediction | $\text{sign}(\sum \alpha_i y_i k + b)$ | $\sum (\alpha_i - \alpha_i^*) k + b$ |
| Sparsity | Marginal + misclassified | Outside $\epsilon$-tube |

---

## ✏️ 엄밀한 정의

### 정의 6.1 — $\epsilon$-Insensitive Loss

$$\ell_\epsilon(u) := \max(0, |u| - \epsilon), \quad u := y - f(x).$$

### 정의 6.2 — SVR Primal

$$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_i (\xi_i + \xi_i^*)$$

s.t.
- $y_i - (w^\top x_i + b) \leq \epsilon + \xi_i$,
- $(w^\top x_i + b) - y_i \leq \epsilon + \xi_i^*$,
- $\xi_i, \xi_i^* \geq 0$.

### 정의 6.3 — SVR Dual

$$\max_{\alpha, \alpha^*} \sum_i (\alpha_i - \alpha_i^*) y_i - \epsilon \sum_i (\alpha_i + \alpha_i^*) - \frac{1}{2} \sum_{i, j} (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) k(x_i, x_j)$$

s.t. $0 \leq \alpha_i, \alpha_i^* \leq C$, $\sum_i (\alpha_i - \alpha_i^*) = 0$.

### 정의 6.4 — SVR 예측

$$f(x) = \sum_{i=1}^n (\alpha_i - \alpha_i^*) k(x_i, x) + b.$$

---

## 🔬 정리와 증명

### 정리 6.1 — SVR Dual 유도

**명제**: 정의 6.2의 primal dual은 정의 6.3.

**증명 스케치**: Lagrangian

$$L = \frac{1}{2} \|w\|^2 + C \sum (\xi + \xi^*) - \sum \alpha_i (\epsilon + \xi_i - y_i + w^\top x_i + b) - \sum \alpha_i^* (\epsilon + \xi_i^* + y_i - w^\top x_i - b) - \sum \mu_i \xi_i - \sum \mu_i^* \xi_i^*.$$

Stationarity:
- $\nabla_w L = w - \sum_i (\alpha_i - \alpha_i^*) x_i = 0 \Rightarrow w = \sum (\alpha_i - \alpha_i^*) x_i$.
- $\partial L / \partial b = -\sum (\alpha_i - \alpha_i^*) = 0$.
- $\partial L / \partial \xi_i = C - \alpha_i - \mu_i = 0 \Rightarrow \alpha_i \leq C$.
- $\partial L / \partial \xi_i^* = C - \alpha_i^* - \mu_i^* = 0 \Rightarrow \alpha_i^* \leq C$.

$w$ 대입 → dual. $\square$

### 정리 6.2 — KKT로부터 SV의 특성화

**명제**: Optimal에서:

1. **Interior** ($|y_i - f(x_i)| < \epsilon$): $\alpha_i = \alpha_i^* = 0$. Support vector 아님.
2. **On $\epsilon$-tube boundary** ($|y_i - f(x_i)| = \epsilon$): $0 \leq \alpha_i$ or $\alpha_i^*$ $\leq C$.
3. **Outside tube** ($|y_i - f(x_i)| > \epsilon$): $\alpha_i = C$ or $\alpha_i^* = C$.

**증명**: Complementary slackness의 여러 식을 case-by-case 해석. $\square$

**해석**: $\epsilon$-tube 내부 점은 **해에 기여하지 않음** → **sparse solution**.

### 정리 6.3 — $\alpha_i$와 $\alpha_i^*$ 동시에 non-zero 불가

**명제**: 최적해에서 $\alpha_i \cdot \alpha_i^* = 0$ for all $i$.

**증명 아이디어**: $y_i - f(x_i) > \epsilon$과 $f(x_i) - y_i > \epsilon$이 동시에 성립 불가 → $\xi_i, \xi_i^*$ 중 하나만 non-zero. KKT로 $\alpha_i$ 또는 $\alpha_i^*$도 하나만 non-zero. $\square$

### 정리 6.4 — Primal 해석: Hinge-like Form

**명제**: Slack 변수를 제거하면 SVR은

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_i \max(0, |y_i - w^\top x_i - b| - \epsilon).$$

**증명**: Optimal $\xi_i^* = \max(0, y_i - w^\top x_i - b - \epsilon)$, $\xi_i^{**} = \max(0, w^\top x_i + b - y_i - \epsilon)$. 대입. $\square$

**의미**: "$\epsilon$-insensitive loss + L2 regularization"의 standard 형태. SGD로도 최적화 가능.

### 정리 6.5 — $\epsilon \to 0$ 극한

**명제**: $\epsilon \to 0$이면 SVR의 loss는 absolute error $|y - f(x)|$ (MAE)로 수렴. 또한 모든 non-boundary 점이 support vector가 될 수 있음 (sparsity 상실).

**증명**: $\epsilon$-insensitive loss의 정의에서 직접. $\square$

**실무 함의**: $\epsilon$을 "허용 가능한 noise level"로 설정. 예: noise가 $\mathcal{N}(0, 0.1^2)$이면 $\epsilon \approx 0.1$.

### 정리 6.6 — $\nu$-SVR (Alternative Parametrization)

**명제** (Schölkopf et al. 2000): $\epsilon$ 대신 $\nu \in (0, 1]$을 파라미터로 쓰면 $\epsilon$이 **자동으로** 학습됨. Constraint: 대략 $n \nu$개의 점이 support vector, 그중 $n \nu / 2$개가 bounded SV.

$\nu$-SVR primal:

$$\min \frac{1}{2} \|w\|^2 + C \left( \nu \epsilon + \frac{1}{n} \sum_i (\xi_i + \xi_i^*) \right), \quad \epsilon \geq 0.$$

$\epsilon$을 **최적화 변수**로 포함. 같은 dual with $\nu$-related constraint.

**장점**: $\nu$ 튜닝이 $\epsilon$ 튜닝보다 직관적 (비율 기반).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.svm import SVR

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 데이터: sinusoidal + noise
# ─────────────────────────────────────────────
n = 60
X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
y_true = np.sin(X).flatten()
y = y_true + 0.1 * rng.standard_normal(n)

def rbf(X, Y, s=0.5):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

K = rbf(X, X)
C_val = 10.0
eps_val = 0.1

# ─────────────────────────────────────────────
# 2. SVR dual (cvxpy)
# ─────────────────────────────────────────────
alpha = cp.Variable(n)
alpha_star = cp.Variable(n)
beta = alpha - alpha_star

obj = cp.Maximize(beta @ y - eps_val * (cp.sum(alpha) + cp.sum(alpha_star))
                    - 0.5 * cp.quad_form(beta, cp.psd_wrap(K)))
cons = [alpha >= 0, alpha <= C_val,
        alpha_star >= 0, alpha_star <= C_val,
        cp.sum(beta) == 0]
prob = cp.Problem(obj, cons)
prob.solve(solver='CLARABEL')

a = np.asarray(alpha.value).flatten()
a_s = np.asarray(alpha_star.value).flatten()
diff = a - a_s

print(f'SVR dual objective = {prob.value:.4f}')

# Support vectors
sv_mask = np.abs(diff) > 1e-4
n_sv = sv_mask.sum()
print(f'Support vectors: {n_sv} / {n} (sparsity: {1 - n_sv/n:.2%})')

# b* 복구 from free SV (where 0 < α < C or 0 < α* < C)
free_mask = ((a > 1e-4) & (a < C_val - 1e-4)) | ((a_s > 1e-4) & (a_s < C_val - 1e-4))
if free_mask.sum() > 0:
    # y_i - f(x_i) = ±ε
    f_train = K @ diff
    # Use free SVs where α > 0 (upper boundary)
    b_upper = y[free_mask & (a > 1e-4)] - f_train[free_mask & (a > 1e-4)] - eps_val
    b_lower = y[free_mask & (a_s > 1e-4)] - f_train[free_mask & (a_s > 1e-4)] + eps_val
    b_all = np.concatenate([b_upper, b_lower])
    b_star = b_all.mean()
else:
    b_star = 0.0
print(f'b* = {b_star:.4f}')

# ─────────────────────────────────────────────
# 3. 예측 + 시각화 (ε-tube 포함)
# ─────────────────────────────────────────────
x_grid = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
f_pred = rbf(x_grid, X) @ diff + b_star

plt.figure(figsize=(10, 5))
plt.plot(X, y_true, 'k--', alpha=0.3, label='true sin')
plt.fill_between(x_grid.flatten(), f_pred - eps_val, f_pred + eps_val, color='lightgray', alpha=0.5, label=f'ε-tube (ε={eps_val})')
plt.plot(x_grid, f_pred, 'b-', lw=2, label='SVR prediction')
plt.scatter(X[~sv_mask], y[~sv_mask], c='gray', s=30, label='Interior (not SV)')
plt.scatter(X[sv_mask], y[sv_mask], c='red', s=60, edgecolors='k', label=f'Support vectors ({n_sv})')
plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.title(f'SVR (RBF, C={C_val}, ε={eps_val})')
plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. sklearn 검증
# ─────────────────────────────────────────────
svr = SVR(kernel='rbf', C=C_val, epsilon=eps_val, gamma=1/(2*0.5**2))
svr.fit(X, y)
pred_sk = svr.predict(x_grid)
print(f'\n예측 max 차이 (manual vs sklearn): {np.max(np.abs(f_pred - pred_sk)):.4e}')
print(f'sklearn #SVs: {len(svr.support_)}')

# ─────────────────────────────────────────────
# 5. ε의 효과 — sparsity vs accuracy
# ─────────────────────────────────────────────
import pandas as pd
eps_list = [0.01, 0.1, 0.3, 0.5]
records = []
for e in eps_list:
    svr_e = SVR(kernel='rbf', C=C_val, epsilon=e, gamma=1/(2*0.5**2)).fit(X, y)
    y_pred = svr_e.predict(X)
    mse = np.mean((y_pred - y_true) ** 2)
    records.append({'ε': e, '#SVs': len(svr_e.support_), 'MSE (true)': f'{mse:.4f}'})
print('\n= ε 효과 =')
print(pd.DataFrame(records).to_string(index=False))
```

**출력 예시**:
```
SVR dual objective = 3.4521
Support vectors: 18 / 60 (sparsity: 70.00%)
b* = 0.0412

예측 max 차이 (manual vs sklearn): 2.34e-05
sklearn #SVs: 18

= ε 효과 =
   ε  #SVs  MSE (true)
0.01    51     0.0089
0.10    18     0.0092
0.30     9     0.0143
0.50     6     0.0312
```

→ $\epsilon$ 커질수록 SV 줄지만 MSE 증가 (bias 증가). Sparsity vs accuracy trade-off.

---

## 🔗 실전 활용

- **Bayesian Optimization surrogate**: GP가 많이 쓰이지만 SVR도 가능. 더 sparse + 더 빠른 예측.
- **시계열 회귀**: Feature engineering (lag features) + RBF SVR. Financial prediction의 한 접근.
- **Noise-robust**: $\epsilon$-insensitive loss가 MSE보다 outlier에 덜 민감 (linear vs quadratic penalty).
- **Hyperparameter tuning**: $(C, \epsilon, \gamma)$ 3차원 grid search. Time-consuming.
- **Alternative**: KRR (Ch5-01) — closed form, sparsity 없음. GP (Ch4) — uncertainty 추가. SVR은 "sparse kernel method 원하고 uncertainty 불필요" 시.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| $\epsilon$ 수동 설정 | $\nu$-SVR으로 자동 학습 가능 but 복잡 |
| 두 slack 변수 | Dual 변수 수 2배 → 메모리·시간 약간 증가 |
| Regression only | 순수 회귀, 확률적 예측 없음 (GP 대체재) |
| **Large $n$** scaling | SMO extension (two-set working set), 그래도 $O(n^2)$ memory |
| Outlier gone wild | Heavy-tailed noise에서는 robust 회귀 (Huber, quantile) 더 나을 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\ell_\epsilon(y, f) = \max(0, |y - f| - \epsilon) \quad ; \quad f(x) = \sum_i (\alpha_i - \alpha_i^*) k(x_i, x) + b}$$

$$\boxed{\max \sum (\alpha_i - \alpha_i^*) y_i - \epsilon \sum (\alpha_i + \alpha_i^*) - \frac{1}{2} \sum_{i, j} (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) k(x_i, x_j)}$$

| 점 | 조건 | $\alpha$, $\alpha^*$ |
|----|------|----------------------|
| Interior ($\epsilon$-tube 안) | $|y - f| < \epsilon$ | $\alpha = \alpha^* = 0$ (not SV) |
| Tube 경계 | $|y - f| = \epsilon$ | Free SV: $0 < \alpha < C$ or $0 < \alpha^* < C$ |
| 외부 (위) | $y - f > \epsilon$ | $\alpha = C$ (bounded) |
| 외부 (아래) | $f - y > \epsilon$ | $\alpha^* = C$ (bounded) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): SVR의 예측 공식에서 $\sum (\alpha_i - \alpha_i^*) k(x_i, x) + b$에 왜 **두 계수의 차이** $\alpha_i - \alpha_i^*$가 나오는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

Primal stationarity: $w = \sum_i (\alpha_i - \alpha_i^*) x_i$.

- $\alpha_i > 0$ (upper slack active): target $y_i$가 $f(x_i)$보다 **위**에 있음 → 해가 $x_i$ 방향으로 "당겨짐".
- $\alpha_i^* > 0$ (lower slack active): target이 **아래** → $x_i$ 반대 방향으로 "당겨짐".
- 부호 있는 contribution: $\alpha_i - \alpha_i^* \in [-C, C]$.

정리 6.3에 따라 $\alpha_i \cdot \alpha_i^* = 0$ → 두 계수 중 하나만 non-zero.

**Classification과 대조**: SVM은 $y_i \in \{-1, +1\}$이 부호 담당 → $\alpha_i y_i$ 단일 계수. SVR은 target이 연속 → 부호 담당할 추가 변수 필요 → $(\alpha, \alpha^*)$ 분리.

</details>

**문제 2** (심화): SVR과 KRR의 비교: 같은 RBF kernel, 같은 $\lambda$ (KRR의 regularization)에서 두 해가 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

- **KRR**: $\alpha_{\text{KRR}} = (K + \lambda I)^{-1} y$. 모든 $\alpha_i \ne 0$ (dense). Loss = squared.
- **SVR**: $\alpha$ 중 다수가 0 (sparse). Loss = $\epsilon$-insensitive.

**같은 $\lambda$/$C$에서**:
- SVR은 tube 안 training 점을 "fit하지 않으려" 함 → bias 작지만 일부 점 무시.
- KRR은 모든 점에 squared penalty → noise에 민감, smoothing.

**실험적 관찰**:
- Noise가 homoscedastic Gaussian → KRR/GP 우월 (ML estimator).
- Noise가 heavy-tailed + outlier → SVR 우월 (robust).
- Sparsity 필요 → SVR.
- Uncertainty 필요 → GP.

**결론**: SVR과 KRR은 다른 loss → 다른 해. "언제 어느 것"은 noise 성격과 downstream task에 의존.

</details>

**문제 3** (ML 연결): SVR을 **time-series forecasting**에 쓸 때 주의할 점?

<details>
<summary>힌트 및 해설</summary>

1. **I.I.D. 가정**: SVR은 i.i.d. 가정 하에 정당화. 시계열은 autocorrelated → cross-validation 설계 주의 (block CV, walk-forward).

2. **Feature engineering**: $x_t = (y_{t-1}, y_{t-2}, \ldots, y_{t-p})$ lag features. Exogenous variables 포함 가능.

3. **Stationarity**: Non-stationary 시계열 (trend, seasonality)은 전처리 (차분, detrend, deseasonalize) 후 SVR.

4. **$\epsilon$ 선택**: Noise level에 비례. 관측 잡음 표준편차 $\sigma$ → $\epsilon \approx 0.5\sigma$ ~ $\sigma$.

5. **Forecasting horizon**: Multi-step ahead는 recursive (1-step 예측 반복) 또는 direct (별도 모델). Recursive는 error 누적.

6. **대안 비교**: ARIMA, Prophet, LSTM 등과 비교. SVR은 short-to-medium horizon + moderate data에 적합.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 05. SMO (Sequential Minimal Optimization)](./05-smo.md) | [Ch4-01. GP의 정의와 공분산 함수 ▶](../ch4-gaussian-process/01-gp-definition.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
