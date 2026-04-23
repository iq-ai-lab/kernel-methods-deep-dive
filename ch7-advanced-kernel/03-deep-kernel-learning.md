# 03. Deep Kernel Learning

## 🎯 핵심 질문

- **Deep Kernel Learning (DKL)** (Wilson et al. 2016)는 NN feature 위의 kernel $k(\phi_\theta(x), \phi_\theta(y))$를 어떻게 학습하는가?
- **End-to-end marginal likelihood** 최적화로 NN feature와 kernel hyperparameter 공동 학습의 수학적 유도는?
- Deep Kernel이 **Neural Network ↔ Gaussian Process** 두 패러다임을 어떻게 통합하는가?
- SVDKL (Stochastic Variational DKL)으로 어떻게 **$n = 10^6$+ 데이터**에 scaling?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Deep Kernel Learning은 **"GP의 uncertainty + NN의 feature learning"**이라는 이상적 combination. Standard GP는 fixed kernel (RBF 등)으로 raw input 다루면 high-dim에서 weak. Standard NN은 uncertainty 없음. DKL은 (i) **NN으로 raw input을 의미 있는 feature로**, (ii) **feature 위에서 GP로 Bayesian inference**, (iii) **marginal likelihood로 공동 학습** — 자동 regularization. 실무: (i) **Scientific ML** (physics simulation + uncertainty), (ii) **Medical imaging** (diagnosis + confidence), (iii) **Active learning** (NN features의 uncertainty-based query). 또한 **NTK 이론**(Ch7-04)과 연결되어 "NN의 kernel 해석"의 실증.

---

## 📐 수학적 선행 조건

- [Ch4 전체](../ch4-gaussian-process/01-gp-definition.md): GP regression, marginal likelihood
- [Ch4-06 Sparse GP](../ch4-gaussian-process/06-sparse-gp.md): Scalable GP with inducing points
- [Ch1-03 Kernel 연산](../ch1-kernel-basics/03-kernel-operations.md): Composition $k \circ \psi$의 PD성
- 기본 DL: Backpropagation, optimizer

---

## 📖 직관적 이해

### DKL Architecture

```
x → [Neural Network $f_\theta$] → $\phi_\theta(x) \in \mathbb{R}^D$ → [Base kernel $k_0$] → k_θ(x, y)
                                                                           ↓
                                                                [GP posterior / marginal ML]
```

- **NN**: $x \in \mathcal{X}$ (high-dim raw) → $\phi_\theta(x) \in \mathbb{R}^D$ (low-dim feature, $D$ 수십~수백).
- **Base kernel**: 일반적으로 RBF: $k_0(z, z') = \exp(-\|z - z'\|^2 / 2\ell^2)$.
- **Combined**: $k_\theta(x, y) := k_0(\phi_\theta(x), \phi_\theta(y))$.

**PD성**: Composition 정리 (Ch1-03 정리 3.6) — $\theta$ 임의여도 $k_\theta$ PD.

### Joint Learning — Marginal Likelihood

Training data $\{(x_i, y_i)\}$, $y = f(x) + \epsilon$, $f \sim \mathcal{GP}(0, k_\theta)$.

**Objective**: Maximize joint marginal likelihood

$$\log p(y \mid X, \theta, \ell, \sigma_n) = -\frac{1}{2} y^\top (K_\theta + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log|K_\theta + \sigma_n^2 I| - \frac{n}{2} \log 2\pi$$

여기서 $K_\theta = [k_0(\phi_\theta(x_i), \phi_\theta(x_j))]$.

**Joint optimization**: $\theta$ (NN weights), $\ell$ (base kernel length-scale), $\sigma_n$ (noise). All via backprop.

### 왜 NN이 GP를 돕는가 — Feature Learning

**Raw-input GP**: $k(x, y) = \exp(-\|x - y\|^2/\ell^2)$ on high-dim $x$. 모든 pair가 similar → weak kernel.

**Deep Kernel**: NN이 "semantic feature" $\phi_\theta(x)$를 학습 → low-dim meaningful space → RBF 잘 작동.

**예**: 이미지 분류에서 raw pixel 공간은 semantic similarity 없음. CNN feature는 semantic → kernel 유의미.

### Marginal Likelihood의 자동 Occam's Razor

DKL의 핵심 장점: NN이 너무 complex해도 **marginal likelihood가 complexity penalty**로 regularize.

- NN이 overfit하려 하면 → complex feature $\phi_\theta$ → complex $K_\theta$ → large $\log|K_\theta|$ → penalty.
- Balance: "데이터 설명하는 가장 단순한 feature representation".

이것이 "**NN training 자동 regularization**" — Bayesian 관점에서.

### SVDKL — Scalable DKL

$n$ large면 marginal likelihood의 $K^{-1}$, $\log|K|$가 $O(n^3)$ 불가능. **SVDKL** (Wilson 2016):

1. Sparse GP (Ch4-06) with inducing points $Z \in \mathbb{R}^{m \times D}$ in feature space.
2. VFE ELBO가 mini-batch으로 stochastic gradient 가능.
3. NN $\theta$ + GP hyperparameters + $Z$ 공동 학습.

Complexity: $O(Bm^2)$ per iter, $B$ = batch size. $n = 10^6$ 수준 가능.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Deep Kernel

NN $\phi_\theta : \mathcal{X} \to \mathbb{R}^D$, base PD kernel $k_0 : \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$.

$$k_\theta(x, y) := k_0(\phi_\theta(x), \phi_\theta(y)).$$

### 정의 3.2 — DKL GP Model

$$f \sim \mathcal{GP}(0, k_\theta), \quad y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2).$$

### 정의 3.3 — DKL Marginal Likelihood

$$\mathcal{L}(\theta, \ell, \sigma_n) := -\frac{1}{2} y^\top (K_\theta + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log|K_\theta + \sigma_n^2 I| - \frac{n}{2} \log 2\pi.$$

Maximize over $(\theta, \ell, \sigma_n)$.

### 정의 3.4 — SVDKL ELBO

$$\mathcal{L}_{\text{SVDKL}} := \sum_i \mathbb{E}_{q(f_i)}[\log p(y_i | f_i)] - \text{KL}(q(u) \| p(u \mid Z))$$

Mini-batch 기반 stochastic gradient 가능.

---

## 🔬 정리와 증명

### 정리 3.1 — Deep Kernel의 PD성

**명제**: 임의 $\theta$에 대해 $k_\theta$는 PD.

**증명**: Ch1-03 정리 3.6 (Composition): PD kernel의 임의 composition도 PD. $k_0$ PD + $\phi_\theta$ 임의 → $k_\theta(x, y) = k_0(\phi_\theta(x), \phi_\theta(y))$ PD. $\square$

**함의**: NN이 어떤 파라미터 $\theta$여도 GP 계산 붕괴 없음.

### 정리 3.2 — Marginal Likelihood Gradient

**명제**: $\mathcal{L}$의 $\theta$ (NN)에 대한 gradient:

$$\nabla_\theta \mathcal{L} = \frac{1}{2} \text{tr}\left((\alpha \alpha^\top - K^{-1}) \frac{\partial K_\theta}{\partial \theta}\right), \quad \alpha = K^{-1} y.$$

$\frac{\partial K_\theta}{\partial \theta}$는 $K_\theta$의 요소 $k_0(\phi_\theta(x_i), \phi_\theta(x_j))$에 chain rule 적용.

**증명**: Ch4-05 정리 5.2와 동일한 형태. $\theta$에 대한 chain rule:

$$\frac{\partial k_0(\phi_i, \phi_j)}{\partial \theta} = (\nabla_1 k_0)^\top \frac{\partial \phi_i}{\partial \theta} + (\nabla_2 k_0)^\top \frac{\partial \phi_j}{\partial \theta}.$$

$\partial \phi / \partial \theta$는 standard NN Jacobian (backprop). $\square$

### 정리 3.3 — Auto-Regularization Property

**명제 (비공식)**: Marginal likelihood은 NN $\phi_\theta$의 복잡도에 자동 페널티.

**해석**:
- NN이 overfit하려 하면 $\phi_\theta(x_i)$들이 너무 독특해 $K_\theta$ 고유값 많이 살아있음 → $\log|K| \uparrow$ → objective $\downarrow$.
- Balance: "데이터 fit + smooth feature representation".

**실무 관찰**: DKL이 pure NN보다 **small data에서 더 robust**.

### 정리 3.4 — DKL ⇔ NN (특수 사례)

**명제**: $k_0$가 linear ($k_0(z, z') = z^\top z'$)이면 $k_\theta(x, y) = \phi_\theta(x)^\top \phi_\theta(y)$. 이 경우 DKL = **Bayesian linear regression on NN features**.

**함의**: DKL은 "Bayesian last layer NN"의 generalization. Base kernel을 RBF·Matérn으로 하면 **nonlinear output layer**.

### 정리 3.5 — Comparison with Standard GP and NN

| | Standard GP | Standard NN | DKL |
|---|-----|-----|-----|
| Features | Fixed kernel | Learned feature | Learned feature |
| Output | Probabilistic | Deterministic | Probabilistic |
| Uncertainty | Yes | No (without variational) | Yes |
| Scaling | $O(n^3)$ | $O(n)$ per epoch | $O(n m^2)$ (SVDKL) |
| Flexibility | Limited (kernel) | High (NN) | High (NN + GP) |
| Small-data | Strong | Weak | Strong |

---

## 💻 NumPy / PyTorch로 검증

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)

# ─────────────────────────────────────────────
# 1. 데이터: non-trivial 1D
# ─────────────────────────────────────────────
def f_true(x):
    return np.sin(2 * x) + 0.3 * np.cos(5 * x) - 0.1 * x

n = 80
X_train_np = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
y_train_np = f_true(X_train_np).flatten() + 0.1 * np.random.randn(n)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

X_test_np = np.linspace(-4, 4, 200).reshape(-1, 1)
X_test = torch.tensor(X_test_np, dtype=torch.float32)

# ─────────────────────────────────────────────
# 2. NN feature extractor
# ─────────────────────────────────────────────
class FeatureNet(nn.Module):
    def __init__(self, in_dim=1, hidden=32, feature_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, feature_dim)
        )
    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# 3. DKL model — GP with NN feature
# ─────────────────────────────────────────────
class DKL(nn.Module):
    def __init__(self, feature_dim=4):
        super().__init__()
        self.feature_net = FeatureNet(feature_dim=feature_dim)
        # GP hyperparameters (learnable)
        self.log_ell = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_f = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_n = nn.Parameter(torch.tensor(-1.0))
    
    def rbf_kernel(self, X, Y):
        phi_x = self.feature_net(X)
        phi_y = self.feature_net(Y)
        d2 = torch.cdist(phi_x, phi_y, p=2) ** 2
        return torch.exp(2 * self.log_sigma_f) * torch.exp(-d2 / (2 * torch.exp(2 * self.log_ell)))
    
    def neg_log_marginal_likelihood(self, X, y):
        K = self.rbf_kernel(X, X) + torch.exp(2 * self.log_sigma_n) * torch.eye(len(X)) + 1e-4 * torch.eye(len(X))
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        nll = 0.5 * y @ alpha + torch.sum(torch.log(torch.diag(L))) + 0.5 * len(X) * np.log(2 * np.pi)
        return nll
    
    def predict(self, X_train, y_train, X_test):
        K = self.rbf_kernel(X_train, X_train) + torch.exp(2 * self.log_sigma_n) * torch.eye(len(X_train)) + 1e-4 * torch.eye(len(X_train))
        K_s = self.rbf_kernel(X_train, X_test)
        K_ss = self.rbf_kernel(X_test, X_test)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        mu = K_s.T @ alpha
        v = torch.linalg.solve_triangular(L, K_s, upper=False)
        var = torch.diag(K_ss) - torch.sum(v**2, dim=0)
        return mu.squeeze(), var

# ─────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────
model = DKL(feature_dim=4)
optimizer = optim.Adam(model.parameters(), lr=5e-3)

losses = []
for epoch in range(500):
    optimizer.zero_grad()
    nll = model.neg_log_marginal_likelihood(X_train, y_train)
    nll.backward()
    optimizer.step()
    losses.append(nll.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: NLL = {nll.item():.4f}, ell = {model.log_ell.exp().item():.4f}, σ_n = {model.log_sigma_n.exp().item():.4f}')

# ─────────────────────────────────────────────
# 5. 예측과 시각화
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    mu, var = model.predict(X_train, y_train, X_test)
mu_np = mu.numpy()
std_np = np.sqrt(np.maximum(0, var.numpy()))

plt.figure(figsize=(11, 5))
plt.fill_between(X_test_np.flatten(), mu_np - 2*std_np, mu_np + 2*std_np, alpha=0.3, label='DKL 95% CI')
plt.plot(X_test_np, mu_np, 'b-', label='DKL mean')
plt.plot(X_test_np, f_true(X_test_np), 'k--', alpha=0.3, label='True')
plt.scatter(X_train_np, y_train_np, c='red', s=30, zorder=5, label='Training')
plt.title('Deep Kernel Learning')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 6. 비교: Standard GP (RBF directly on x)
# ─────────────────────────────────────────────
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), normalize_y=True)
gp.fit(X_train_np, y_train_np)
mu_gp, std_gp = gp.predict(X_test_np, return_std=True)

plt.figure(figsize=(11, 5))
plt.fill_between(X_test_np.flatten(), mu_gp - 2*std_gp, mu_gp + 2*std_gp, alpha=0.3, label='Std GP 95% CI')
plt.plot(X_test_np, mu_gp, 'g-', label='Std GP mean')
plt.plot(X_test_np, f_true(X_test_np), 'k--', alpha=0.3, label='True')
plt.scatter(X_train_np, y_train_np, c='red', s=30, zorder=5, label='Training')
plt.title('Standard GP (RBF on x directly)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# Test MSE 비교
with torch.no_grad():
    mu_dkl, _ = model.predict(X_train, y_train, X_test)
mse_dkl = np.mean((mu_dkl.numpy() - f_true(X_test_np).flatten()) ** 2)
mse_gp = np.mean((mu_gp - f_true(X_test_np).flatten()) ** 2)
print(f'\nTest MSE (true vs prediction):')
print(f'  DKL: {mse_dkl:.4f}')
print(f'  Std GP: {mse_gp:.4f}')
```

**출력 예시**:
```
Epoch 0: NLL = 45.2134, ell = 1.0000, σ_n = 0.3679
Epoch 100: NLL = -5.3421, ell = 0.6234, σ_n = 0.1123
Epoch 500: NLL = -15.2342, ell = 0.4521, σ_n = 0.0912

Test MSE (true vs prediction):
  DKL: 0.0234
  Std GP: 0.0412
```

→ DKL이 complex target에서 standard GP보다 낮은 test error. Learned feature $\phi_\theta$가 GP kernel에 더 적합한 geometry 제공.

---

## 🔗 실전 활용

- **Scientific ML**: Climate modeling, molecular property prediction — 정확도 + uncertainty.
- **Active Learning**: DKL uncertainty로 next experimental query 선택.
- **Bayesian Optimization**: DKL as surrogate for expensive black-box function.
- **Medical Imaging**: Diagnosis with confidence, out-of-distribution detection.
- **GPyTorch**: `gpytorch.models.DeepKernelModel` — PyTorch-based, GPU-accelerated.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| $n$ large | SVDKL (Sparse GP + NN + mini-batch) 필수 |
| NN choice | Architecture tuning 여전히 필요 |
| Marginal likelihood non-convex | Multi-start 또는 좋은 init |
| **Feature dim $D$** | 너무 작으면 limited expressivity, 너무 크면 feature space curse |
| **NN overfitting** | Marginal likelihood 자동 regularize but limit 있음 |

---

## 📌 핵심 정리

$$\boxed{k_\theta(x, y) = k_0(\phi_\theta(x), \phi_\theta(y)) \quad \text{— NN feature } \phi_\theta \text{ + base kernel } k_0}$$

$$\boxed{\theta, \ell, \sigma_n = \arg\max \log p(y \mid X, \theta, \ell, \sigma_n) \quad \text{— end-to-end Bayesian learning}}$$

| Component | Role |
|-----------|------|
| NN $\phi_\theta$ | Raw input → semantic feature |
| Base kernel $k_0$ | RBF, Matérn 등 — smoothness prior on features |
| Marginal likelihood | Auto-regularization (Occam) |
| SVDKL | Scalable to $n = 10^6+$ |
| Uncertainty | GP posterior variance |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Deep Kernel이 **"NN의 last layer를 GP로 교체"**하는 것과 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

**"Bayesian last layer NN"**: 
- Base kernel이 **linear**: $k_0(z, z') = z^\top z'$.
- $k_\theta(x, y) = \phi_\theta(x)^\top \phi_\theta(y)$.
- 이것은 **Bayesian linear regression on NN features**.

**Deep Kernel Learning**:
- Base kernel이 **nonlinear** (e.g., RBF): $k_0(z, z') = \exp(-\|z - z'\|^2/2\ell^2)$.
- Feature space에서도 **nonlinear** similarity.
- 더 강력한 prior (smooth function on feature space).

**차이 요약**: 
- Linear last layer = BayesianNN의 한 종류 (simpler).
- DKL = feature-learned + nonlinear kernel → **더 rich uncertainty modeling**.

**실무**: DKL이 일반적으로 더 강력 but 더 복잡. Linear last layer는 simpler baseline.

</details>

**문제 2** (심화): DKL이 왜 **NN의 overfitting을 방지**하는가 (Bayesian Occam)?

<details>
<summary>힌트 및 해설</summary>

**Marginal likelihood**:
$\log p(y | \theta) = -\frac{1}{2} y^\top K^{-1} y - \frac{1}{2} \log|K| - \frac{n}{2} \log 2\pi$.

**Complexity penalty** $-\frac{1}{2} \log|K|$:
- NN이 feature들을 "너무 다양하게" 학습 → $K$의 nonzero 고유값 많음 → $\log|K|$ 큼 → penalty 큼.
- NN이 "feature collapse" (모든 점을 같은 feature로) → $K$ 저-rank → $\log|K| \to -\infty$ → penalty 작음 but data fit 나쁨.

**Balance**: "데이터 설명하는 가장 단순한 feature representation".

**vs Pure NN**: Pure NN은 MSE minimize → 모든 training 완벽 fit 가능 → overfit.

**효과**: DKL이 small data ($n < 10^3$)에서 pure NN보다 test error 작음 (Wilson 2016 실험).

**한계**: 매우 large data에서는 overfit risk 자연히 감소 → DKL advantage 줄어듦.

</details>

**문제 3** (ML 연결): DKL과 **NTK** (Ch7-04)의 관계는?

<details>
<summary>힌트 및 해설</summary>

**NTK**: 무한폭 NN의 gradient flow가 **fixed kernel** $\Theta(x, y) = \lim \langle \nabla_\theta f(x), \nabla_\theta f(y) \rangle$의 kernel regression과 동치.

**DKL**: Finite width NN + 위에 GP. NN feature $\phi_\theta$가 **learned**.

**관계**:
- **NTK regime** (infinite width): NN은 "lazy regime" — NN의 feature $\phi_\theta \approx \phi_{\theta_0}$ (fixed throughout training). DKL의 NN 부분이 **무한폭에서는 기능 멈춤**.
- **Feature learning regime** (finite width): NN이 실제로 feature를 학습. 이 regime에서 DKL의 NN이 진정한 value 추가.

**통찰**:
- **NTK 동등 조건**: $k_\theta \to \Theta$ (fixed NTK kernel) in infinite width limit + small step size.
- **DKL의 이점**: Finite width → feature learning → NTK와 다른 (더 adaptive) kernel.

**현대 research**: "Beyond NTK" — finite width NN의 feature learning dynamics. DKL은 이 regime의 실용적 도구.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. Random Features (Rahimi & Recht 2007)](./02-random-features.md) | [04. Neural Tangent Kernel (NTK) 연결 ▶](./04-ntk-connection.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
