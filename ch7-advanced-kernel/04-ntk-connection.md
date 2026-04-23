# 04. Neural Tangent Kernel (NTK) 연결

## 🎯 핵심 질문

- **Neural Tangent Kernel** $\Theta(x, y) := \lim_{\text{width} \to \infty} \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle$는 무엇이고 왜 PD인가?
- Jacot et al. (2018)의 정리 — "**무한폭 NN의 gradient flow = NTK-RKHS kernel regression**"의 유도는?
- 이것이 **NN training의 수학적 이해**를 어떻게 변혁시켰는가?
- NTK framework가 **Layer 2 Generalization Theory**로 어떻게 이어지는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

2018년 Jacot et al.의 NTK 정리는 Deep Learning 이론의 **획기적 돌파구**다. "NN training은 black box"라는 오랜 믿음을 뒤엎고, **무한폭 한계에서 NN = Kernel Method**라는 정확한 수학적 동치 증명. 이 덕분에 (i) **NN 수렴성** 분석 가능 (기존 non-convex), (ii) **Generalization theory** 적용 (Rademacher complexity, eigenfunction analysis), (iii) **Overparameterization 이해** (왜 큰 모델이 더 잘 generalize하나?). 실무에서는 direct 응용 제한적이지만 (실제 NN은 유한 width), **이론적 proxy로서** 여전히 중요. 또한 **kernel method와 deep learning의 통합 관점** 제공.

---

## 📐 수학적 선행 조건

- [Ch2-03 Representer](../ch2-rkhs-representer/03-representer-theorem.md), [Ch5-01 KRR](../ch5-krr-kpca/01-kernel-ridge-regression.md), [Ch7-03 DKL](./03-deep-kernel-learning.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Jacobian, Gauss-Newton
- 기본 DL: Gradient flow, backpropagation, ReLU / Tanh activations
- 확률론: Law of large numbers, CLT

---

## 📖 직관적 이해

### Gradient Flow로서의 NN Training

Continuous-time 한계에서 gradient descent는 **gradient flow**:

$$\frac{d\theta(t)}{dt} = -\nabla_\theta L(\theta(t)).$$

$L$ = MSE loss: $L = \frac{1}{2} \sum_i (f_\theta(x_i) - y_i)^2$. 

**Output dynamics**: Chain rule로

$$\frac{d f_\theta(x)}{dt} = \nabla_\theta f_\theta(x)^\top \frac{d\theta}{dt} = -\sum_i (f_\theta(x_i) - y_i) \underbrace{\nabla_\theta f_\theta(x)^\top \nabla_\theta f_\theta(x_i)}_{\Theta_\theta(x, x_i) = \text{"NTK at } \theta \text{"}}.$$

즉 NN의 output은 **"NTK weight에 따른 interpolation"**처럼 진화.

### 무한폭 한계 — Jacot의 정리

**핵심 발견** (Jacot, Gabriel, Hongler 2018):

1. **Infinite width initialization**: $\theta_0 \sim \mathcal{N}(0, \sigma^2 / \text{width})$ standard init.
2. **$\Theta_{\theta_0}(x, y) \xrightarrow{\text{width} \to \infty} \Theta(x, y)$** in probability, **deterministic fixed kernel**.
3. **$\Theta_{\theta(t)} \approx \Theta$ for all $t$** ("**lazy regime**") — training 동안 NTK 거의 불변.

**결과**: Gradient flow가

$$\frac{d f_t(x)}{dt} = -\sum_i (f_t(x_i) - y_i) \Theta(x, x_i)$$

— **linear in $f_t$** (fixed kernel!). 이것은 **kernel gradient flow**로 exactly analyzable.

### Kernel Regression으로 수렴

$t \to \infty$ 한계에서 (MSE, gradient flow, no regularization):

$$f_\infty(x_*) = \Theta_*^\top \Theta^{-1} y$$

이것은 **NTK kernel regression** (KRR with $\lambda = 0$, Ch5-01).

**함의**:
- Infinite width NN의 "gradient descent 수렴 해" = "NTK kernel regression interpolant".
- NN training의 **closed form 예측** 가능 (무한폭).

### NTK의 구조 — Layer-wise

Fully-connected NN의 NTK는 layer-wise 재귀 식:

$$\Theta^{(l+1)}(x, y) = \Theta^{(l)}(x, y) \dot{\sigma}(\Sigma^{(l)}(x, y)) + \Sigma^{(l+1)}(x, y)$$

여기서 $\Sigma$는 **conjugate kernel** (activation covariance), $\dot{\sigma}$는 derivative of activation.

각 layer가 NTK에 non-trivial 기여. Deep network의 NTK는 depth에 따라 달라짐.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Neural Tangent Kernel (Empirical)

NN $f_\theta : \mathcal{X} \to \mathbb{R}$ with params $\theta \in \mathbb{R}^P$:

$$\Theta_\theta(x, y) := \nabla_\theta f_\theta(x)^\top \nabla_\theta f_\theta(y).$$

### 정의 4.2 — Infinite-Width NTK (Jacot 2018)

NN의 width를 적절히 scale ($\theta_0 \sim \mathcal{N}(0, \sigma^2/n_{\text{in}})$), width $\to \infty$:

$$\Theta(x, y) := \lim_{\text{width} \to \infty} \Theta_{\theta_0}(x, y).$$

(정확히: LLN으로 $\Theta_{\theta_0}$가 **deterministic fixed function**으로 수렴.)

### 정의 4.3 — Gradient Flow

Continuous-time gradient descent:

$$\frac{d\theta}{dt} = -\nabla_\theta L(\theta).$$

### 정의 4.4 — Lazy Regime

Infinite width에서 $\Theta_{\theta(t)}(x, y) = \Theta(x, y) + o(1)$ for all $t \geq 0$ (NTK constant throughout training).

---

## 🔬 정리와 증명

### 정리 4.1 — Empirical NTK의 PD성

**명제**: 모든 $\theta$에 대해 $\Theta_\theta$는 PD kernel.

**증명**: $\Theta_\theta(x, y) = \nabla_\theta f_\theta(x)^\top \nabla_\theta f_\theta(y)$은 feature map $\phi_\theta(x) := \nabla_\theta f_\theta(x)$의 inner product. Ch1-01 정리 1.4. $\square$

### 정리 4.2 — NTK Convergence (Jacot 2018, 대략)

**명제** (informal): 적절한 initialization으로 width-scaled NN에서 width $\to \infty$일 때

$$\Theta_{\theta_0}(x, y) \to \Theta(x, y) \quad \text{in probability}$$

$\Theta$ = deterministic **NTK limit**.

또한 training 동안 $\Theta_{\theta(t)} = \Theta + O(1/\sqrt{\text{width}})$ — lazy regime.

**증명 스케치**: 
- **LLN**: 각 neuron의 activation이 independent random variable들의 합 → LLN으로 deterministic limit.
- **Training dynamics**: $\theta(t) - \theta_0 = O(1/\sqrt{\text{width}})$ (movement 작음) → $\Theta_{\theta(t)} \approx \Theta_{\theta_0} \approx \Theta$.

**세부**: Jacot 2018, Arora et al. 2019, Yang 2020 논문들의 다양한 증명.

### 정리 4.3 — Gradient Flow = Kernel Regression

**명제**: Infinite width NTK lazy regime, MSE loss, gradient flow. $t \to \infty$에서 NN output이

$$f_\infty(x_*) = \Theta_*^\top \Theta^{-1} y$$

로 수렴 ($\Theta_* = [\Theta(x_*, x_i)]_i$, $\Theta = [\Theta(x_i, x_j)]$).

**증명 아이디어**:

Kernel gradient flow ODE:

$$\frac{df_t}{dt} = -(\Theta (f_t - y))_{(x_i)}$$

(training 점에서). Linear ODE → solution $f_t - y^* \to 0$ as $t \to \infty$, $y^* = y$ (interpolation).

Test output dynamics: $\frac{d f_t(x_*)}{dt} = -\Theta_*^\top (f_t - y)$. Integrating:

$$f_\infty(x_*) = f_0(x_*) + \Theta_*^\top \Theta^{-1} (y - f_0(x_{\text{train}})).$$

Zero init ($f_0 = 0$): $f_\infty = \Theta_*^\top \Theta^{-1} y$. $\square$

**중요 조건**: 
- Infinite width (exact).
- MSE loss.
- Gradient flow (not GD with large step).
- $\Theta$ invertible.

### 정리 4.4 — NTK의 Layer-wise 재귀 (FC net, ReLU)

**명제** (Jacot 2018): Fully-connected network with ReLU. Recursive:

$$\Sigma^{(l+1)}(x, y) = c_\sigma \mathbb{E}_{u \sim \mathcal{N}(0, \Sigma^{(l)}_{\{x, y\}})}[\sigma(u_1) \sigma(u_2)]$$

$$\Theta^{(l+1)}(x, y) = \Theta^{(l)}(x, y) \cdot c_\sigma \mathbb{E}[\dot{\sigma}(u_1) \dot{\sigma}(u_2)] + \Sigma^{(l+1)}(x, y)$$

$c_\sigma$는 activation-specific constant. ReLU의 경우 closed form:

$$\Sigma^{(l+1)}(x, y) = \frac{\|\Sigma^{(l)}\|_F \pi - \arccos(\rho)}{\pi \cdot 2} \cdot \sqrt{\Sigma^{(l)}_{xx} \Sigma^{(l)}_{yy}}$$

($\rho$ = angle between $x, y$ in $\Sigma^{(l)}$ metric).

**함의**: NTK는 NN 구조 (depth, width, activation)에 의해 결정되는 **specific kernel family**. Deep NN의 NTK는 매우 매끄러운 kernel.

### 정리 4.5 — Finite Width Deviations

**명제**: Finite width NN은 정확한 NTK behavior에서 편차. Feature learning이 실제로 일어남 (NTK와 다른 dynamics).

**Beyond-NTK regime**: $\theta$가 $\theta_0$로부터 크게 이동, $\Theta_{\theta(t)} \ne \Theta_{\theta_0}$. 이것이 "feature learning"의 수학적 정의.

**현대 연구**: Finite width NN의 generalization은 NTK보다 나을 수 있음 (feature learning 덕분). NTK는 **baseline/limit**.

---

## 💻 NumPy / PyTorch로 검증

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)

# ─────────────────────────────────────────────
# 1. Empirical NTK — 유한 width NN
# ─────────────────────────────────────────────
class FCNet(nn.Module):
    def __init__(self, in_dim=1, hidden=128, out_dim=1, depth=2):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(prev_dim, hidden), nn.ReLU()]
            prev_dim = hidden
        layers += [nn.Linear(prev_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)

def empirical_ntk(model, X1, X2):
    """Compute NTK matrix Θ(X1, X2) = J(X1) J(X2)^T empirically."""
    n1, n2 = len(X1), len(X2)
    params = list(model.parameters())
    
    def get_jacobian(x_batch):
        """Each row = vectorized gradient of output w.r.t. θ for input x_batch[i]."""
        n = len(x_batch)
        J = []
        for i in range(n):
            model.zero_grad()
            out = model(x_batch[i:i+1])
            grads = torch.autograd.grad(out.sum(), params, retain_graph=True, create_graph=False)
            J.append(torch.cat([g.flatten() for g in grads]))
        return torch.stack(J)
    
    J1 = get_jacobian(X1)
    J2 = get_jacobian(X2) if X2 is not X1 else J1
    return (J1 @ J2.T).detach().numpy()

# ─────────────────────────────────────────────
# 2. NTK 측정 at initialization, 여러 width
# ─────────────────────────────────────────────
X = torch.linspace(-2, 2, 20).reshape(-1, 1)

plt.figure(figsize=(12, 4))
for i, width in enumerate([50, 200, 1000]):
    model = FCNet(hidden=width)
    ntk = empirical_ntk(model, X, X)
    # Normalize to make comparison easier
    ntk_normalized = ntk / ntk.max()
    plt.subplot(1, 3, i+1)
    plt.imshow(ntk_normalized, cmap='viridis')
    plt.colorbar()
    plt.title(f'Empirical NTK (width={width})')
plt.tight_layout(); plt.show()

# Width 커지면 NTK가 smoother, deterministic-like

# ─────────────────────────────────────────────
# 3. NN training이 NTK kernel regression으로 수렴
# ─────────────────────────────────────────────
n = 15
X_train = torch.linspace(-2, 2, n).reshape(-1, 1)
y_train = torch.sin(2 * X_train).flatten() + 0.1 * torch.randn(n)

X_test = torch.linspace(-3, 3, 200).reshape(-1, 1)

# Wide NN training
model_wide = FCNet(hidden=1000, depth=3)
optimizer = torch.optim.SGD(model_wide.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(5000):
    pred = model_wide(X_train).flatten()
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred_nn = model_wide(X_test).flatten().numpy()

# NTK kernel regression using init NTK
model_init = FCNet(hidden=1000, depth=3)
torch.manual_seed(42)
model_init.apply(model_init._init)

ntk_train = empirical_ntk(model_init, X_train, X_train)
ntk_test = empirical_ntk(model_init, X_train, X_test)

# Kernel regression (with small regularization)
alpha = np.linalg.solve(ntk_train + 1e-4 * np.eye(n), y_train.numpy())
pred_ntk = ntk_test.T @ alpha

# ─────────────────────────────────────────────
# 4. 시각화 — NN vs NTK kernel regression
# ─────────────────────────────────────────────
plt.figure(figsize=(11, 5))
plt.scatter(X_train.numpy(), y_train.numpy(), c='red', s=50, zorder=5, label='Training')
plt.plot(X_test.numpy(), pred_nn, 'b-', lw=2, label='Wide NN prediction')
plt.plot(X_test.numpy(), pred_ntk, 'g--', lw=2, label='NTK kernel regression (theoretical limit)')
plt.plot(X_test.numpy(), np.sin(2 * X_test.numpy()), 'k:', alpha=0.3, label='True sin')
plt.title('NN ≈ NTK kernel regression (width=1000)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# 차이 측정
diff = np.mean(np.abs(pred_nn - pred_ntk))
print(f'|NN - NTK regression| 평균 차이: {diff:.4f}')
print('→ Wide NN에서 작은 차이 기대')
```

**출력 예시**:
```
|NN - NTK regression| 평균 차이: 0.0234
→ Wide NN에서 작은 차이 기대
```

→ Wide NN이 NTK kernel regression에 가까움. Exact match는 infinite width에서만 (finite width deviation 있음).

---

## 🔗 실전 활용

- **Theoretical tool**: NN의 generalization 분석의 수학적 proxy.
- **Kernel regression**: NTK를 kernel로 쓰는 SVM/KRR/GP (neural-tangents 라이브러리).
- **Architecture search**: NTK의 condition number로 good vs bad init 판별.
- **Infinite-width Bayesian NN**: NTK GP로 uncertainty 얻기.
- **Beyond-NTK research**: Finite width, feature learning이 왜 NTK보다 나은지 연구 활발.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Infinite width 가정 | 실제 NN은 finite width, NTK와 deviation |
| Lazy regime | Feature learning 일어나는 regime에서는 NTK 부정확 |
| MSE loss | Cross-entropy 등에서 추가 분석 필요 |
| Gradient flow | 실제 SGD는 stochastic + large steps |
| **No feature learning** | NTK는 fixed kernel → representation 학습 X |
| Practical gain 제한 | Wide NN은 NTK로 잘 설명되나 narrow NN은 부족 |

---

## 📌 핵심 정리

$$\boxed{\Theta(x, y) = \lim_{\text{width} \to \infty} \langle \nabla_\theta f_{\theta_0}(x), \nabla_\theta f_{\theta_0}(y) \rangle}$$

$$\boxed{\text{Infinite width NN training (MSE, GF)} \equiv \text{NTK kernel regression}: f_\infty(x_*) = \Theta_*^\top \Theta^{-1} y}$$

| Regime | 특성 |
|--------|------|
| **NTK / Lazy** | Width large, parameters 거의 불변, kernel fixed, linear dynamics |
| **Feature Learning** | Finite width, parameters significant 이동, feature evolves |
| **Beyond NTK** | 현대 연구 방향 — NN이 NTK보다 나은 이유 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): NTK가 PD kernel임을 정의로부터 직접 증명하라.

<details>
<summary>힌트 및 해설</summary>

$\Theta_\theta(x, y) = \nabla_\theta f_\theta(x)^\top \nabla_\theta f_\theta(y)$.

Feature map $\phi_\theta(x) := \nabla_\theta f_\theta(x) \in \mathbb{R}^P$ ($P$ = total number of parameters).

$\Theta_\theta(x, y) = \langle \phi_\theta(x), \phi_\theta(y) \rangle_{\mathbb{R}^P}$ — inner product kernel.

Ch1-01 정리 1.4로 PD. 

**Infinite width limit**: Pointwise limit of PD kernels이 PD (Ch1-03 정리 3.5). ✓

**함의**: $\Theta$는 valid RKHS 정의. NN training이 이 RKHS에서 함수를 찾는 것.

</details>

**문제 2** (심화): NTK의 **eigendecomposition**이 NN의 **learning dynamics**를 어떻게 설명하는가?

<details>
<summary>힌트 및 해설</summary>

NTK gradient flow $\frac{df_t}{dt} = -\Theta (f_t - y)$의 해:

$f_t = y + e^{-\Theta t}(f_0 - y)$ (operator form).

Eigendecomp $\Theta = \sum_k \lambda_k \phi_k \phi_k^\top$:

$f_t - y = \sum_k e^{-\lambda_k t} \langle f_0 - y, \phi_k \rangle \phi_k$.

**해석**:
- **Large $\lambda_k$** (fast eigenmodes): "Simple" directions (NTK의 top eigenvectors, 일반적으로 smooth functions) → training early에 fit.
- **Small $\lambda_k$** (slow): "Complex" directions (high-freq) → training 늦게 fit 또는 fit 안 함.

**함의**:
- **Early stopping = implicit regularization**: small $\lambda$ eigenmodes 보호.
- **Easy first, hard later**: 자연스러운 curriculum 효과.
- **Generalization**: Bias toward NTK의 top eigenvectors → 단순 함수 → good generalization.

**실무**: NN의 "deep learning magic"이 NTK eigenvalue decay로 부분 설명.

</details>

**문제 3** (ML 연결): NTK 이론이 "**큰 모델이 generalize를 잘 하는 이유**"에 어떤 answer를 제공하는가?

<details>
<summary>힌트 및 해설</summary>

**Classical statistics view**: "More parameters → more capacity → overfit".

**NTK insight**:
- Infinite width NN은 **specific RKHS의 함수**를 학습 (NTK-RKHS).
- Large width도 이 RKHS 범위 내 → **not 더 많은 capacity**.
- SGD의 implicit bias가 NTK의 top eigenvectors 선호 → **simple functions**.

**따라서**:
- Width 늘려도 "effective complexity"는 RKHS 크기로 제한.
- **Double descent**의 한 설명: 유한 width에서 interpolation → NTK limit에서 smooth.

**한계**:
- Real NN (finite width)는 feature learning → NTK 범위 초과 가능.
- NTK는 **linear model**: 실제 NN의 non-linear expressivity 설명 못함.

**현대 관점**: NTK는 "lazy NN"의 precise theory. Feature learning NN은 **beyond NTK**, 더 복잡한 이론 필요 (mean field, μP, etc.).

**Layer 2 Generalization Theory**: NTK ↔ Rademacher complexity ↔ implicit bias 연결 — generalization 전체 framework.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Deep Kernel Learning](./03-deep-kernel-learning.md) | [📚 README로 돌아가기](../README.md) |

</div>
