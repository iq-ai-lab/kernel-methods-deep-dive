# 05. SMO (Sequential Minimal Optimization)

## 🎯 핵심 질문

- 왜 SVM dual을 **2개의 $\alpha$씩만** 업데이트하는가? 1개로는 왜 불가능하고 3개 이상이면 왜 과잉인가?
- 2-변수 sub-problem의 **해석적 해**는 어떻게 유도되는가?
- Working set 선택에 쓰이는 **KKT violation**은 어떻게 정의되고 왜 의미 있는가?
- SMO의 수렴성·계산 복잡도는? 대규모 데이터에 왜 효과적인가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Standard QP solver로 SVM dual을 풀면 $O(n^3)$ 메모리·시간이 들어 $n \geq 10^4$에서 실용성이 사라진다. SMO (Platt 1998)은 **두 변수씩 해석적으로** 업데이트하는 coordinate-ascent-type 알고리즘으로, **메모리 $O(n)$** (Gram은 필요할 때 계산)과 **실용적 속도 $O(n^2)$ ~ $O(n^{2.3})$** 를 달성했다. 이것이 LIBSVM, sklearn, LIBLINEAR 등 **모든 주요 SVM 구현의 핵심**. SMO를 이해하면 kernel method scaling의 "empirical sweet spot"을 파악하고, 왜 SVM이 1998~2010년대 산업 표준이 되었는지 알 수 있다.

---

## 📐 수학적 선행 조건

- [Ch3-02 Lagrange dual](./02-lagrange-dual.md), [Ch3-04 Soft-margin](./04-soft-margin-hinge.md)
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Coordinate descent/ascent, KKT conditions
- 기초: 2차 함수의 극값, 제약 하 최적화

---

## 📖 직관적 이해

### 왜 "2개씩"인가 — $\sum \alpha_i y_i = 0$ 제약 때문

Dual 제약 $\sum_i \alpha_i y_i = 0$은 모든 $\alpha_i$를 묶는다. $\alpha_1$ 하나만 변경하면 이 제약이 깨짐 → 1-변수 업데이트 불가.

**해결**: $\alpha_i, \alpha_j$ 두 변수를 **동시에** 변경, 제약 유지.

$$\alpha_i^{\text{new}} y_i + \alpha_j^{\text{new}} y_j = \alpha_i^{\text{old}} y_i + \alpha_j^{\text{old}} y_j \equiv \text{const}.$$

이 linear constraint + box $\alpha \in [0, C]^2$ 하에 2차 objective 최소화 → **1차원 해석적 해**.

### Working Set 선택 — KKT Violation

KKT 조건이 실제로 얼마나 깨졌는지 측정해 **위반이 큰 pair**를 선택.

"KKT violator"의 관찰:
- $\alpha_i = 0$이면 $y_i f(x_i) \geq 1$이어야 optimal.
- $\alpha_i = C$이면 $y_i f(x_i) \leq 1$이어야 optimal.
- $0 < \alpha_i < C$이면 $y_i f(x_i) = 1$.

KKT 위반: 위 조건 중 어느 것이 깨진 $i$.

**Platt의 heuristic**: 가장 큰 KKT 위반 $(i, j)$ pair 선택 → **dual 증가량 최대화**.

### 해석적 업데이트 — "Newton step in 1D"

2-변수 sub-problem을 linear constraint로 1-변수로 축소. 2차 함수이므로 **해석적 minimum** 존재.

- 2차 계수: $\eta := 2 k(x_i, x_j) - k(x_i, x_i) - k(x_j, x_j)$.
- Newton step: $\alpha_j^{\text{new, unclipped}} = \alpha_j^{\text{old}} + y_j (E_i - E_j) / \eta$, $E_k := f(x_k) - y_k$ (error).
- Box constraint로 **clip**: $\alpha_j^{\text{new}} = \max(L, \min(H, \alpha_j^{\text{new, unclipped}}))$.
- $\alpha_i^{\text{new}}$는 linear constraint로 복구.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Dual Sub-problem (2-변수)

고정된 $i, j$와 나머지 $\alpha_{k \ne i, j}$에 대해:

$$\max_{\alpha_i, \alpha_j} \quad W(\alpha_i, \alpha_j) = \alpha_i + \alpha_j - \frac{1}{2} \sum_{k, l} \alpha_k \alpha_l y_k y_l k(x_k, x_l)$$

s.t. $0 \leq \alpha_i, \alpha_j \leq C$, $\alpha_i y_i + \alpha_j y_j = \text{const}$.

### 정의 5.2 — KKT Violation

각 $i$에 대해:

- $\alpha_i = 0$: 위반량 = $\max(0, 1 - y_i f(x_i))$.
- $\alpha_i = C$: 위반량 = $\max(0, y_i f(x_i) - 1)$.
- $0 < \alpha_i < C$: 위반량 = $|y_i f(x_i) - 1|$.

### 정의 5.3 — Error Vector

$$E_i := f(x_i) - y_i = \sum_k \alpha_k y_k k(x_k, x_i) + b - y_i.$$

---

## 🔬 정리와 증명

### 정리 5.1 — 2-변수 Sub-problem 해석적 해

**명제**: 정의 5.1의 sub-problem에서 $\alpha_j^{\text{new}}$의 unclipped Newton step:

$$\alpha_j^{\text{new, unc}} = \alpha_j^{\text{old}} + \frac{y_j (E_i - E_j)}{\eta}, \quad \eta := 2 k(x_i, x_j) - k(x_i, x_i) - k(x_j, x_j).$$

그 다음 box constraint로 **clip**:

$$\alpha_j^{\text{new}} = \max(L, \min(H, \alpha_j^{\text{new, unc}})).$$

Box의 경계 $L, H$:

- $y_i \ne y_j$ (다른 class): $L = \max(0, \alpha_j - \alpha_i)$, $H = \min(C, C + \alpha_j - \alpha_i)$.
- $y_i = y_j$ (같은 class): $L = \max(0, \alpha_i + \alpha_j - C)$, $H = \min(C, \alpha_i + \alpha_j)$.

마지막으로 $\alpha_i^{\text{new}} = \alpha_i^{\text{old}} + y_i y_j (\alpha_j^{\text{old}} - \alpha_j^{\text{new}})$.

**증명 스케치**:

Linear constraint $\alpha_i y_i + \alpha_j y_j = s$ (상수)로 $\alpha_i = y_i (s - \alpha_j y_j) = y_i s - y_i y_j \alpha_j$. Objective를 $\alpha_j$만의 함수로:

$$W(\alpha_j) = C_1 + C_2 \alpha_j - \frac{1}{2} (-\eta) \alpha_j^2$$

(여기서 $C_1, C_2$는 상수; $\eta$는 위 정의).

$\frac{dW}{d\alpha_j} = C_2 + \eta \alpha_j = 0 \Rightarrow \alpha_j = -C_2 / \eta$. $C_2$를 계산하면 $y_j(E_i - E_j)$ 형태 (원식 전개; Platt 1998 상세).

Box constraint로 clipping은 standard projection. $\square$

### 정리 5.2 — $\eta < 0$ 조건

**명제**: $\eta = 2 k(x_i, x_j) - k(x_i, x_i) - k(x_j, x_j) \leq 0$. 특히 strict PD kernel에서 $x_i \ne x_j$이면 $\eta < 0$.

**증명**: $\phi(x_i) - \phi(x_j)$의 norm 제곱:

$$\|\phi(x_i) - \phi(x_j)\|^2 = k(x_i, x_i) - 2 k(x_i, x_j) + k(x_j, x_j) = -\eta \geq 0.$$

Strict PD: $x_i \ne x_j$이면 $\phi(x_i) \ne \phi(x_j)$ → $-\eta > 0 \Rightarrow \eta < 0$. $\square$

**해석**: $\eta < 0 \Rightarrow $ objective가 $\alpha_j$에 대해 **strictly concave** (maximization이므로 굿). Newton step이 잘 정의됨.

**수치적 주의**: $\eta = 0$ (duplicate 점)이면 업데이트 불가 → skip 또는 별도 처리.

### 정리 5.3 — SMO 알고리즘 (간이 버전)

**알고리즘**:

1. 초기화 $\alpha^{(0)} = 0$, $b^{(0)} = 0$.
2. 반복:
   - **Outer loop**: KKT 위반자를 돌아가며 $i$ 선택.
   - **Inner loop**: $i$에 대한 "second choice" $j$ 선택 — $\max_j |E_i - E_j|$ (Platt heuristic).
   - **Update**: 정리 5.1로 $\alpha_i, \alpha_j$ 업데이트.
   - $b$ 업데이트: 새 free SV의 functional margin = 1 조건.
   - $E$ 캐시 업데이트.
3. 모든 $i$의 KKT 위반이 tolerance 이하면 종료.

### 정리 5.4 — SMO의 수렴성

**명제**: SMO의 반복은 dual objective를 **단조 증가**시키고, SVM dual의 unique 최적해로 수렴한다 (strict PD kernel 가정).

**증명 아이디어**:
- 각 SMO step은 strict improvement (정리 5.2의 $\eta < 0$).
- 실행 가능 집합 (compact + convex)에서 단조 수렴 objective → 수렴.
- Unique optimum은 strict convexity (of $-W$ over 2D subspace).

**수렴 속도**: 이론적 linear rate, 실용적으로 $O(n^2)$ ~ $O(n^{2.3})$ 전체 복잡도 (data-dependent).

### 정리 5.5 — $b$ 업데이트 규칙

**명제**: 업데이트된 $\alpha_i^{\text{new}}, \alpha_j^{\text{new}}$에서 $b$는 다음과 같이 유지 또는 업데이트:

- **Case 1** ($0 < \alpha_i^{\text{new}} < C$, i.e., free SV): $b_{\text{new}} = y_i - f(x_i)$-like 식, 정확한 수식:

$$b_1 = E_i + y_i (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) k(x_i, x_i) + y_j (\alpha_j^{\text{new}} - \alpha_j^{\text{old}}) k(x_i, x_j) + b_{\text{old}}.$$

- **Case 2** ($0 < \alpha_j^{\text{new}} < C$): 유사하게 $b_2$.
- **Case 3** (둘 다 $(0, C)$): $b_{\text{new}} = (b_1 + b_2) / 2$.
- **Case 4** (둘 다 bounded): $b$는 $[b_1, b_2]$ 범위 내 아무 값 (관례: 중간값).

(Platt 1998 §2.3)

---

## 💻 NumPy로 검증

```python
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)

# 데이터
X, y01 = make_moons(n_samples=300, noise=0.2, random_state=0)
y = 2 * y01 - 1
n = len(y)

def rbf(X, Y, s=0.5):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X @ Y.T
    return np.exp(-d2 / (2 * s**2))

C = 1.0

# ─────────────────────────────────────────────
# SMO (simplified — Platt 1998의 교육적 버전)
# ─────────────────────────────────────────────
def smo_simple(X, y, C, kernel_fn, tol=1e-3, max_iter=100):
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    K_cache = kernel_fn(X, X)

    def f(i):
        return np.sum(alpha * y * K_cache[:, i]) + b

    for it in range(max_iter):
        changed = 0
        for i in range(n):
            E_i = f(i) - y[i]
            # KKT violation check
            if (y[i] * E_i < -tol and alpha[i] < C) or \
               (y[i] * E_i > tol and alpha[i] > 0):
                # 2nd choice: j with max |E_i - E_j|
                E_all = np.array([f(k) - y[k] for k in range(n)])
                j = np.argmax(np.abs(E_all - E_i))
                if j == i:
                    j = rng.integers(n)
                    while j == i:
                        j = rng.integers(n)

                E_j = E_all[j]
                a_i_old, a_j_old = alpha[i], alpha[j]

                # L, H
                if y[i] != y[j]:
                    L = max(0, a_j_old - a_i_old)
                    H = min(C, C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - C)
                    H = min(C, a_i_old + a_j_old)
                if L == H:
                    continue

                eta = 2 * K_cache[i, j] - K_cache[i, i] - K_cache[j, j]
                if eta >= 0:
                    continue

                a_j_new = a_j_old - y[j] * (E_i - E_j) / eta
                a_j_new = max(L, min(H, a_j_new))

                if abs(a_j_new - a_j_old) < 1e-5:
                    continue

                a_i_new = a_i_old + y[i] * y[j] * (a_j_old - a_j_new)

                # b 업데이트
                b1 = b - E_i - y[i] * (a_i_new - a_i_old) * K_cache[i, i] \
                     - y[j] * (a_j_new - a_j_old) * K_cache[i, j]
                b2 = b - E_j - y[i] * (a_i_new - a_i_old) * K_cache[i, j] \
                     - y[j] * (a_j_new - a_j_old) * K_cache[j, j]

                if 0 < a_i_new < C:
                    b = b1
                elif 0 < a_j_new < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alpha[i] = a_i_new
                alpha[j] = a_j_new
                changed += 1
        if changed == 0:
            print(f'SMO 수렴: iter {it}')
            break
    return alpha, b

t0 = time.time()
alpha_smo, b_smo = smo_simple(X, y, C, rbf, tol=1e-3, max_iter=200)
t_smo = time.time() - t0
print(f'SMO 소요 시간: {t_smo:.2f}s')
print(f'SVs: {(alpha_smo > 1e-4).sum()} / {n}')

# ─────────────────────────────────────────────
# 검증: sklearn과 비교
# ─────────────────────────────────────────────
t0 = time.time()
svc = SVC(kernel='rbf', C=C, gamma=1/(2*0.5**2))
svc.fit(X, y)
t_sk = time.time() - t0
print(f'sklearn 소요 시간: {t_sk:.2f}s')
print(f'sklearn SVs: {len(svc.support_)}')

# 결정 경계 일치 확인
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

K_grid = rbf(grid, X)
f_smo = K_grid @ (alpha_smo * y) + b_smo
pred_smo = np.sign(f_smo)
pred_sk = svc.predict(grid)
agree = (pred_smo == pred_sk).mean()
print(f'SMO vs sklearn 결정 일치율: {agree:.4f}')

# ─────────────────────────────────────────────
# Scaling 비교 — n 증가에 따른 시간
# ─────────────────────────────────────────────
import pandas as pd
results = []
for n_size in [100, 300, 1000]:
    X_s, y01_s = make_moons(n_samples=n_size, noise=0.2, random_state=0)
    y_s = 2 * y01_s - 1
    
    t0 = time.time()
    _, _ = smo_simple(X_s, y_s, C, rbf, max_iter=100)
    t_smo_s = time.time() - t0
    
    t0 = time.time()
    SVC(kernel='rbf', C=C, gamma=2).fit(X_s, y_s)
    t_sk_s = time.time() - t0
    
    results.append({'n': n_size, 'SMO simple (s)': t_smo_s, 'sklearn (s)': t_sk_s})

print(pd.DataFrame(results).to_string(index=False))
```

**출력 예시**:
```
SMO 수렴: iter 12
SMO 소요 시간: 1.45s
SVs: 76 / 300
sklearn 소요 시간: 0.03s
sklearn SVs: 75
SMO vs sklearn 결정 일치율: 0.9967

  n  SMO simple (s)  sklearn (s)
100           0.18         0.01
300           1.45         0.03
1000          12.34        0.21
```

→ Simplified SMO가 sklearn의 더 최적화된 LIBSVM과 정확도 거의 일치. 단, sklearn은 훨씬 빠름 (C 구현 + shrinking heuristic).

---

## 🔗 실전 활용

- **sklearn**: `SVC` 내부에 LIBSVM 사용 — SMO + shrinking + working set selection (WSS2 또는 WSS3).
- **LIBLINEAR**: Linear SVM 전용, 다른 알고리즘 (coordinate descent on primal). $n \gg d$에 최적.
- **Pegasos**: Primal hinge에 SGD. $n$ 매우 큰 경우 ($n > 10^6$) SMO보다 빠를 수 있음.
- **LASVM**: Online SMO, streaming data에 적합.
- **Shrinking heuristic**: "이미 optimum에 도달한 $\alpha$"를 inactive set으로 보내 sub-problem 크기 줄임 — sklearn의 핵심 최적화.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Memory $O(n^2)$ (Gram) | On-the-fly Gram 계산하면 시간 trade-off |
| Cache 관리 | SMO 가속의 큰 부분 — Gram 원소 중 자주 쓰이는 것 캐시 |
| **$n > 10^5$** | SMO도 느려짐 → LIBLINEAR (linear), Random Features, Nyström |
| Tolerance-dependent | 작은 tol (높은 정확도)은 수렴 매우 느림 |
| Non-PD kernel | $\eta \geq 0$ 가능 → skip 필요, 수렴 보장 약함 |
| Bias 중복 | 여러 $b$ estimate 평균 — 수치 노이즈 |

---

## 📌 핵심 정리

$$\boxed{\alpha_j^{\text{new, unc}} = \alpha_j^{\text{old}} + \frac{y_j (E_i - E_j)}{\eta}, \quad \eta = 2 k(x_i, x_j) - k(x_i, x_i) - k(x_j, x_j) < 0}$$

$$\boxed{\text{SMO: 2-변수씩 해석적 업데이트 + KKT violation으로 working set 선택}}$$

| 요소 | 역할 |
|------|------|
| **2-변수 choice** | $\sum \alpha_i y_i = 0$ 제약 유지하며 업데이트 |
| **해석적 해** | 2차 sub-problem의 Newton step (clip) |
| **$\eta < 0$** | Objective가 strictly concave in sub-problem |
| **KKT violation** | Working set 선택 기준 — 가장 큰 위반자 pair |
| **수렴성** | 단조 증가 objective → 유한 step 수렴 |
| **복잡도** | $O(n^2)$ ~ $O(n^{2.3})$, 메모리 $O(n)$ (with Gram caching strategy) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 1-변수 coordinate ascent는 SVM dual에 부적합한가?

<details>
<summary>힌트 및 해설</summary>

Dual 제약 $\sum_i \alpha_i y_i = 0$. $\alpha_1$ 하나만 변경하면 제약이 깨짐 → $\alpha_1^{\text{new}}$은 제약을 만족하지 못하는 값이 됨. 

**해결책**:
1. 2-변수: 한 쌍을 동시에 업데이트해 제약 유지 (SMO).
2. 또는: 변수 변환으로 제약 제거 (예: $\alpha_n$을 다른 $\alpha_1, \ldots, \alpha_{n-1}$의 함수로 표현) — 하지만 numerical 문제.

SMO의 2-변수 choice가 "minimal" working set 크기. 3-변수는 해석적 해가 일반적으로 없어 iterative sub-solver 필요 → 덜 효율적.

</details>

**문제 2** (심화): KKT violation 기반 working set selection이 왜 **dual objective increase**를 보장하는가?

<details>
<summary>힌트 및 해설</summary>

KKT 조건이 성립하면 $\alpha$는 optimal, objective의 gradient가 feasible direction에서 0. 따라서 **KKT 위반자** = "gradient에 non-zero ascent direction 있는 $i$".

Sub-problem에서 두 변수 업데이트 후 objective $W$가 증가하려면 $\nabla W$와 이동 방향이 양의 내적을 가져야. KKT 위반 $i, j$ pair는 정확히 그런 경우:

$y_i E_i$ 큼 + $y_j E_j$ 작음 ⇒ $y_i E_i - y_j E_j$ 큼 ⇒ newton step 크기 큼 ⇒ objective 증가 큼.

Platt (1998): Maximum violating pair (MVP) 선택이 "greedy ascent" 보장.

**Working Set Selection 2 (Fan, Chen, Lin 2005)**: MVP보다 개선된 selection으로 LIBSVM의 핵심 개선.

</details>

**문제 3** (ML 연결): Deep Learning의 SGD와 SMO의 본질적 차이는?

<details>
<summary>힌트 및 해설</summary>

- **SGD on NN**: Parametric model, non-convex objective, 확률적 batch gradient, 고정 $|\theta|$.
- **SMO on SVM**: Non-parametric (kernel), convex objective (quadratic), deterministic working set, data-dependent $n$.

| 측면 | SGD | SMO |
|------|-----|-----|
| Objective | Non-convex | Strictly convex QP |
| 수렴 | Local minimum | Global minimum |
| Iteration cost | $O(\|\theta\|)$ per mini-batch | $O(1)$ sub-problem + $O(n)$ error update |
| 메모리 | $O(\|\theta\|)$ | $O(n)$ (Gram cache 포함 시 $O(n^2)$) |
| Scaling | $n$ 매우 큼 OK | $n \leq 10^5$ 권장 |

**통합 관점** (NTK, Ch7-04): "무한폭 NN의 SGD" = "NTK kernel의 SMO-like solver". 두 패러다임이 실은 같은 문제의 다른 접근.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Soft-margin SVM과 Hinge Loss](./04-soft-margin-hinge.md) | [06. SVM Regression (SVR) ▶](./06-svr.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
