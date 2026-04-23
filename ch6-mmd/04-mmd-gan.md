# 04. MMD-GAN과 생성모델

## 🎯 핵심 질문

- **MMD-GAN** (Li et al. 2015)의 objective — generator $G$의 출력 분포 $q_\theta$와 데이터 분포 $p$의 **MMD 최소화** — 는 standard GAN과 어떻게 다른가?
- MMD-based 생성모델이 **adversarial training**보다 안정적인 이유는?
- **Kernel 선택**이 MMD-GAN의 성능에 어떻게 영향을 미치는가?
- **Deep features + MMD** 접근 vs pure pixel MMD의 차이는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

GAN은 강력하지만 **unstable training** (mode collapse, vanishing gradient, discriminator overpowering)으로 악명. MMD-GAN은 "kernel-based non-adversarial loss"로 이 문제 우회. 이론적으로 **characteristic kernel** 하에서 $\text{MMD} = 0 \iff p = q$ → generator의 "실패 모드" 수학적으로 제어 가능. 실무적으로 (i) **training stability** (single objective), (ii) **interpretable loss** (kernel value로 해석), (iii) **small-data에서 경쟁력** (low-data regime). 다만 high-resolution 이미지에서는 (deep-feature GAN에 밀려) 점유율 감소, 하지만 **generator evaluation metric으로는 여전히 사용** (FID와 보완적).

---

## 📐 수학적 선행 조건

- [Ch6-01~03](./01-mmd-definition.md): MMD 이론
- GAN 기본: Generator, discriminator, minimax objective
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Gradient of kernel with respect to input

---

## 📖 직관적 이해

### GAN Review

Standard GAN (Goodfellow 2014): Generator $G_\theta : \mathcal{Z} \to \mathcal{X}$ (noise → data), discriminator $D_\phi : \mathcal{X} \to [0, 1]$ (real vs fake). Minimax:

$$\min_\theta \max_\phi \mathbb{E}_{x \sim p}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].$$

**문제**: $D$가 perfect이면 gradient vanishes; $G$가 너무 좋으면 $D$가 confused. Equilibrium 찾기 어려움.

### MMD-GAN: Kernel-based Single Objective

MMD-GAN (Li et al. 2015, Dziugaite et al. 2015):

$$\min_\theta \text{MMD}^2(p, q_\theta)$$

$q_\theta = G_\theta(\cdot)$의 출력 분포. Gradient:

$$\nabla_\theta \text{MMD}^2 = \nabla_\theta \left(\frac{1}{m^2} \sum_{i, j} k(G(z_i), G(z_j)) - \frac{2}{nm} \sum k(x_i, G(z_j)) + \text{const}\right).$$

Kernel $k$가 differentiable이면 (RBF) direct backprop.

### 왜 "Non-adversarial"인가

MMD-GAN은 **단일 objective** (Generator만 학습, discriminator 없음).

- **단순**: Minimax 아닌 minimization.
- **안정**: Gradient signal이 항상 informative (MMD > 0이면 nonzero gradient).
- **Mode collapse 완화**: $q_\theta$가 $p$의 일부만 cover하면 MMD > 0 → 계속 signal.

### Kernel 선택

**Fixed kernel (Gaussian)**: Simple, interpretable. 하지만 pixel-level에서 weak (high-dim에서 모든 pair가 similar 또는 dissimilar).

**Multi-scale kernel**: $k = \sum_l \exp(-\|x-y\|^2 / 2\sigma_l^2)$, 여러 $\sigma$. 다양한 granularity 포착.

**Learned kernel**: "Critic network" $\psi_\phi$ 학습해 $k(x, y) = k_0(\psi_\phi(x), \psi_\phi(y))$. **Feature space MMD** → adversarial-like training ("MMD critic"). Li et al. 2017 "MMD GAN".

### Deep-Feature MMD

Raw pixel MMD는 RBF bandwidth가 pixel space scale과 맞지 않음 — image similarity가 semantic이 아닌 pixel-level만.

**해결**: **Pre-trained CNN** (e.g., VGG, Inception) 의 intermediate layer feature 위에서 MMD. This is the basis of **FID (Frechet Inception Distance)** — Inception feature의 Gaussian 근사 간 Fréchet distance = MMD with linear kernel.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — MMD-GAN Objective

$$\mathcal{L}_{\text{MMD}}(\theta) := \text{MMD}^2(p, q_\theta) = \mathbb{E}_{x, x' \sim p}[k(x, x')] + \mathbb{E}_{\tilde{x}, \tilde{x}' \sim q_\theta}[k(\tilde{x}, \tilde{x}')] - 2 \mathbb{E}_{x \sim p, \tilde{x} \sim q_\theta}[k(x, \tilde{x})].$$

첫 항 $\mathbb{E}_{p, p}[k]$는 $\theta$-independent → 무시 가능.

**Practical loss**:

$$\mathcal{L}(\theta) = \mathbb{E}_{\tilde{x}, \tilde{x}' \sim q_\theta}[k(\tilde{x}, \tilde{x}')] - 2 \mathbb{E}_{x \sim p, \tilde{x} \sim q_\theta}[k(x, \tilde{x})].$$

### 정의 4.2 — MMD-GAN with Learned Features (Li et al. 2017)

$$\mathcal{L}(\theta, \phi) := \text{MMD}^2_{k_\phi}(p, q_\theta), \quad k_\phi(x, y) = k_0(\psi_\phi(x), \psi_\phi(y)).$$

Generator $G_\theta$ **minimize**, critic $\psi_\phi$ **maximize** (encoder → large MMD when distributions differ).

Minimax + constraints (injective $\psi$, PD kernel preserved).

---

## 🔬 정리와 증명

### 정리 4.1 — MMD-GAN의 일관성

**명제**: Characteristic kernel $k$, infinite data, infinite optimization power 하에서 $\mathcal{L}_{\text{MMD}} = 0 \iff q_\theta = p$.

**증명**: Ch6-01 정리 1.4 직접 따름. Characteristic → MMD = 0 iff distributions match. $\square$

**함의**: Mode-collapse 없음 (이론적으로). 실무적으로는 finite data, limited optimization으로 mode-collapse 발생 가능하지만 덜 심각.

### 정리 4.2 — Gradient 공식

**명제**: $\mathcal{L}(\theta)$의 $\theta$에 대한 gradient:

$$\nabla_\theta \mathcal{L} = \frac{2}{m} \sum_j \left[\frac{1}{m} \sum_{j'} \nabla_\theta G(z_j) \nabla_1 k(G(z_j), G(z_{j'})) - \frac{1}{n} \sum_i \nabla_\theta G(z_j) \nabla_1 k(G(z_j), x_i)\right].$$

($\nabla_1 k$는 첫 번째 argument에 대한 kernel gradient.)

RBF kernel에서 $\nabla_1 k(x, y) = -(x - y)/\sigma^2 \cdot k(x, y)$.

**증명**: Chain rule + linear expectation. $\square$

### 정리 4.3 — Adversarial Perspective (Learned Kernel)

**명제**: Learned feature MMD-GAN은 다음과 동치:

$$\min_G \max_\psi \|\mathbb{E}_p[\psi(x)] - \mathbb{E}_q[\psi(x)]\|_{\mathcal{H}_k}$$

(feature space에서의 moment difference, RKHS norm으로).

이것은 **WGAN**의 MMD version — Kantorovich-Rubinstein (Lipschitz test function) 대신 **RKHS unit ball** test function.

**증명**: IPM 관점 (Ch6-01 정리 1.5) + feature space. $\square$

### 정리 4.4 — Training Stability (Empirical)

**경험적 관찰**:

1. **Fixed kernel MMD-GAN**: 매우 안정적 (minimax 아님). 그러나 capability limited (kernel이 expressive 충분 해야).

2. **Learned kernel MMD-GAN**: 좀 덜 안정적 (adversarial component) but 더 expressive.

3. **Mode coverage**: MMD-GAN 기본이 standard GAN보다 **더 다양한 mode 포착** — characteristic kernel 덕분.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────
# 1. 2D 장난감 예시: Gaussian mixture 생성
# ─────────────────────────────────────────────
# Target: 4-mode Gaussian mixture
def sample_target(n):
    modes = np.array([[2, 2], [-2, 2], [2, -2], [-2, -2]])
    mode_idx = np.random.choice(4, n)
    return modes[mode_idx] + 0.3 * np.random.randn(n, 2)

# Generator: latent z → 2D output
class Generator(nn.Module):
    def __init__(self, z_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, z):
        return self.net(z)

G = Generator()
optimizer = optim.Adam(G.parameters(), lr=1e-3)

# ─────────────────────────────────────────────
# 2. MMD² loss (multi-scale RBF)
# ─────────────────────────────────────────────
def mmd2_loss(x, y, sigmas=[0.1, 0.5, 1.0, 2.0]):
    def k(X, Y, s):
        d2 = torch.sum((X.unsqueeze(1) - Y.unsqueeze(0))**2, dim=-1)
        return torch.exp(-d2 / (2 * s**2))
    
    loss = 0
    for s in sigmas:
        Kxx = k(x, x, s)
        Kyy = k(y, y, s)
        Kxy = k(x, y, s)
        loss = loss + Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return loss / len(sigmas)

# ─────────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────────
batch_size = 256
losses = []

for step in range(3000):
    # Real samples
    x_real = torch.tensor(sample_target(batch_size), dtype=torch.float32)
    # Generate
    z = torch.randn(batch_size, 4)
    x_fake = G(z)
    
    # MMD loss (only Kyy and Kxy terms matter for gradient)
    loss = mmd2_loss(x_real, x_fake)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Step'); plt.ylabel('MMD² loss')
plt.title('MMD-GAN training — stable loss decrease')
plt.grid(True, alpha=0.3); plt.show()

# ─────────────────────────────────────────────
# 4. Generated samples 시각화
# ─────────────────────────────────────────────
with torch.no_grad():
    z_test = torch.randn(1000, 4)
    x_gen = G(z_test).numpy()

x_real_large = sample_target(1000)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(x_real_large[:, 0], x_real_large[:, 1], s=5, alpha=0.5)
axes[0].set_title('Target: 4-mode GMM')
axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)

axes[1].scatter(x_gen[:, 0], x_gen[:, 1], s=5, alpha=0.5, c='red')
axes[1].set_title('MMD-GAN generated')
axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 5. Mode coverage 검증
# ─────────────────────────────────────────────
def count_modes(samples, centers, threshold=1.0):
    covered = 0
    for c in centers:
        if np.any(np.linalg.norm(samples - c, axis=1) < threshold):
            covered += 1
    return covered

centers = np.array([[2, 2], [-2, 2], [2, -2], [-2, -2]])
n_modes_covered = count_modes(x_gen, centers)
print(f'Modes covered: {n_modes_covered} / 4')

# ─────────────────────────────────────────────
# 6. Comparison with vanilla GAN (brief)
# ─────────────────────────────────────────────
# Note: 실제 vanilla GAN은 discriminator 포함, 길어져서 여기서는 개념 설명만
# MMD-GAN의 장점: single loss, mode coverage 양호
# Vanilla GAN: discriminator와 adversarial, mode collapse 빈번
```

**출력 예시**:
```
Modes covered: 4 / 4
```

→ MMD-GAN이 4 mode 모두 포착 (multi-scale kernel 덕분).

---

## 🔗 실전 활용

- **Small-data generation**: MMD-GAN이 $n < 10^4$ 데이터에서 stable. 의료 영상, 과학 simulation.
- **Evaluation metric**: FID (MMD with Inception features, linear kernel)는 GAN 평가 표준.
- **Kernel choice**: Multi-scale RBF (Dziugaite) 또는 deep critic (Li et al. 2017).
- **Domain adaptation**: Source와 target domain의 MMD를 encoder가 minimize → alignment.
- **비교**: Standard GAN이 high-resolution에서 우월 (learned discriminator expressive). MMD-GAN은 low-resolution·small-data niche.

---

## ⚖️ 가정과 한계

| 측면 | 한계 |
|------|------|
| Kernel bandwidth 선택 | Critical — multi-scale로 완화 |
| Pixel-level MMD | High-dim에서 weak; deep feature 필요 |
| Computational | $O(m^2)$ per batch (MMD) — mini-batch size 제한 |
| **Large-scale**: | Standard GAN이 more scalable |
| **Quality**: | StyleGAN 등 현대 GAN이 image quality 우월 |

---

## 📌 핵심 정리

$$\boxed{\min_\theta \mathcal{L}_{\text{MMD}}(\theta) = \text{MMD}^2(p, q_\theta) = \mathbb{E}[k(X, X')] - 2\mathbb{E}[k(X, G(Z))] + \mathbb{E}[k(G(Z), G(Z'))]}$$

$$\boxed{\text{Characteristic kernel} \implies \mathcal{L} = 0 \iff q_\theta = p}$$

| | Standard GAN | MMD-GAN (fixed kernel) | MMD-GAN (learned kernel) |
|--|---------|-----------|-----------|
| Objective | Minimax | Min (single) | Minimax |
| Stability | Low | High | Medium |
| Mode coverage | Poor often | Good | Medium |
| Expressiveness | High | Limited | High |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MMD-GAN training에서 **discriminator가 없는** 이유는?

<details>
<summary>힌트 및 해설</summary>

Standard GAN에서 discriminator는 "distribution 차이를 측정하는 learned metric"을 제공. Generator는 이 metric을 minimize.

MMD는 **fixed kernel로 이미 distribution metric** 제공. 따라서:
- Discriminator 역할 = MMD kernel이 대신.
- Generator만 학습 = single loss minimization.

**이득**:
1. **Stability**: Minimax vs min, 후자가 수렴 guarantee 용이.
2. **Single objective**: 두 networks 균형 걱정 없음.

**손실**:
1. **Expressiveness**: Fixed kernel이 high-dim data의 semantic 차이를 포착 못함 (pixel-level).
2. **Learned kernel variant**: $k_\phi$ 학습으로 이 문제 부분 해결, 하지만 다시 minimax.

</details>

**문제 2** (심화): Multi-scale kernel $k = \sum_l e^{-\|x-y\|^2 / 2\sigma_l^2}$이 single-scale보다 좋은 이유는?

<details>
<summary>힌트 및 해설</summary>

**이유 1 - Scale-invariance**: 데이터의 어떤 scale에서 차이가 있는지 모름. Single $\sigma$는 그 scale만 감지.

**이유 2 - Characteristic robustness**: 여러 scale의 RBF 혼합도 characteristic (characteristic의 sum).

**이유 3 - Gradient signal**: 
- Generator가 $p$에 가까워지면 coarse scale ($\sigma$ 큼)에서 MMD → 0.
- 미세한 차이는 fine scale ($\sigma$ 작음)에서 포착.
- 여러 scale이 combined gradient signal을 **continuously** 제공 → vanishing gradient 완화.

**예시**: Bi-modal data $p$, $\sigma = 0.1$ (좁음)만 쓰면 각 mode 독립 처리, mode 간 차이 포착 못함. $\sigma = 5$ (넓음)만 쓰면 mode 구별 못함. 혼합이 필요.

**실무**: $\sigma \in \{1, 2, 4, 8, 16\}$ 같은 dyadic scale 또는 median heuristic based multi-scale.

</details>

**문제 3** (ML 연결): FID (Frechet Inception Distance)가 **MMD의 한 형태**임을 설명하라.

<details>
<summary>힌트 및 해설</summary>

**FID**: $\text{FID}(p, q) = \|\mu_p - \mu_q\|^2 + \text{tr}(\Sigma_p + \Sigma_q - 2 (\Sigma_p \Sigma_q)^{1/2})$.

이것은 **Inception V3의 pool3 feature** 위에서 **Gaussian 근사된 분포**의 Fréchet distance.

**MMD 연결**:
- Linear kernel on Inception features: $k(x, y) = \psi(x)^\top \psi(y)$ ($\psi$ = Inception feature).
- MMD² with linear kernel = $\|\mathbb{E}_p[\psi] - \mathbb{E}_q[\psi]\|^2$ = first term of FID.
- FID 추가로 covariance 차이 포함 — "MMD with polynomial kernel degree 2 on Inception features"와 근사.

**해석**: FID = "Inception feature space에서의 MMD-like metric". Raw pixel MMD 대신 deep feature MMD가 **perceptual similarity**와 잘 맞음.

**실무**: FID가 GAN evaluation의 standard. MMD는 더 일반적 (any kernel, any feature space) but FID만큼 잘 calibrated된 metric은 드묾.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Two-Sample Test (Gretton et al. 2012)](./03-two-sample-test.md) | [05. Kernel Embedding 일반화 — HSIC·Distribution Regression ▶](./05-kernel-embedding-generalizations.md) |
| [📚 README로 돌아가기](../README.md) | |

</div>
