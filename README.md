<div align="center">

# 🧬 Kernel Methods Deep Dive

### `sklearn.svm.SVC(kernel='rbf')` 를 쓰는 것과,

### **kernel trick** 의 정당성이 Mercer 정리에서 나오고

$$k(x, y) = \sum_n \lambda_n \phi_n(x)\,\phi_n(y)$$

### 의 **implicit feature map 이 무한차원** 임을 증명할 수 있는 것은 **다르다.**

<br/>

> *Gaussian Process 를 **사용하는 것** 과, GP posterior mean 이 Kernel Ridge Regression 과 **완전히 동치** 이고 예측 분산이*
>
> $$k(x_*, x_*) - k_*^\top (K + \sigma^2 I)^{-1} k_*$$
>
> *로 shrink 하는 메커니즘을 유도할 수 있는 것은 다르다.*
>
> *MMD 를 **정의하는 것** 과, characteristic kernel 하에서*
>
> $$\text{MMD}(p, q) = 0 \;\iff\; p = q$$
>
> *이고 이것이 MMD-GAN 의 이론적 근거임을 증명할 수 있는 것은 다르다.*

<br/>

**다루는 정리·기법 (시간순)**

Mercer 1909 *Mercer 정리* · Aronszajn 1950 *RKHS* · Vapnik 1995 *SVM dual* · Schölkopf 2001 *Representer 정리* · Schölkopf 1998 *Kernel PCA* · Rasmussen–Williams 2006 *Gaussian Process* · Gretton 2012 *MMD + characteristic kernel* · Rahimi–Recht 2007 *Random Features* · Jacot 2018 *Neural Tangent Kernel*

<br/>

**핵심 질문**

> **왜 kernel 이 비선형성을 선형 방법에 주입하는 엔진인가** — Positive definite kernel 부터 Representer 정리 · SVM dual · GP posterior · MMD · NTK 까지, SVM · GP · Kernel Ridge · Kernel PCA · MMD-GAN 의 수학적 기반을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![CVXPY](https://img.shields.io/badge/CVXPY-1.4-3C7A89?style=flat-square)](https://www.cvxpy.org/)
[![Docs](https://img.shields.io/badge/Docs-35개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-14k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-200개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-105개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Kernel method에 관한 자료는 대부분 **"RBF kernel을 쓰면 비선형 분류가 됩니다"** 에서 멈춥니다. 하지만 왜 $k(x, y) = \phi(x)^\top \phi(y)$에서 $\phi$가 무한차원일 수 있는지, Representer 정리가 왜 **SVM·KRR·KPCA·GP를 하나의 형태 $\sum \alpha_i k(\cdot, x_i)$로 묶는지**, GP의 marginal likelihood가 어떻게 자동으로 Occam's razor 역할을 하는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "RBF kernel은 비선형 분류에 좋습니다" | **Mercer 정리** $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$로 implicit feature map이 무한차원임을 증명, Gaussian kernel의 RKHS가 Sobolev 공간과 어떻게 관련되는지 유도 |
| "SVM은 margin을 최대화합니다" | Primal $\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$부터 라그랑주 쌍대 $\max \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j k(x_i, x_j)$까지 **KKT로 한 줄씩 유도**, support vector의 기하적 의미 |
| "GP는 함수에 대한 확률분포입니다" | Prior $f \sim \mathcal{GP}(0, k)$에서 joint Gaussian의 조건부로 posterior $\mathcal{N}(k_*^\top (K + \sigma^2 I)^{-1} y,\ k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*)$를 **완전 유도**, covariance function이 왜 함수 prior인가 |
| "Kernel Ridge는 Ridge에 kernel을 끼운 것" | Representer 정리로 $\alpha = (K + \lambda I)^{-1} y$ 유도, **GP posterior mean과 수치적으로 정확히 일치함**을 증명 — 같은 식의 Bayesian 해석과 regularized risk 해석 |
| "MMD는 분포 차이 측정" | $\mu_p = \mathbb{E}_{X \sim p}[k(\cdot, X)] \in \mathcal{H}_k$의 RKHS norm으로 $\text{MMD}(p,q) = \|\mu_p - \mu_q\|_{\mathcal{H}_k}$, **characteristic kernel** 하에서 $\text{MMD} = 0 \iff p = q$의 완전 증명 (Gretton 2012) |
| "Kernel PCA는 특성공간에서 PCA" | Centered Gram $\tilde{K} = K - \mathbf{1}K/n - K\mathbf{1}/n + \mathbf{1}K\mathbf{1}/n^2$의 고유분해, projection $\phi(x) \cdot v_k = \sum \alpha_k^i k(x_i, x)$를 명시적 $\phi$ 없이 계산하는 수학적 근거 |
| "Random Features는 계산 트릭" | **Bochner 정리**로 shift-invariant kernel의 Fourier 표현, $\phi(x) = \sqrt{2/D}(\cos(\omega_i^\top x + b_i))_i$가 **왜 RBF를 근사**하는지 $\mathbb{E}_\omega[\phi(x)^\top \phi(y)] = k(x, y)$로 증명 |
| "NTK는 무한폭 신경망의 kernel" | $\Theta(x, y) = \lim_{\text{width} \to \infty} \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle$가 **PD kernel임을 증명**, NN 훈련이 NTK-RKHS kernel regression과 동치(Jacot 2018)가 되는 과정 |
| 공식 나열 | NumPy + cvxpy로 SVM dual 바닥 구현, GP regression 바닥 구현, Random Features로 $n = 10^5$ 스케일링, **sklearn/GPy 결과와 값 단위로 일치 검증** |

---

## 📌 선행 레포 & 후속 레포

```
[Functional Analysis]  ──►  [Convex Optimization]  ──►  이 레포  ──►  [Bayesian ML Deep Dive]
  RKHS·Mercer·                 Lagrangian 쌍대·KKT·        Kernel Methods         GP 확장·BO·BNN
  Moore-Aronszajn              Slater 조건                 (SVM·GP·MMD·KPCA)
                                                               │
                                                               ▼
                                               [Generalization Theory Deep Dive]
                                                 Rademacher·NTK·이중 강하
  ▲                    ▲                    ▲
  │                    │                    │
[Linear Algebra]  [Probability Theory]  [Mathematical Statistics]
 스펙트럴·PD 행렬   다변수정규·조건부        Bayesian 추론
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Functional Analysis Deep Dive**(RKHS·Mercer·Moore-Aronszajn)와 **Convex Optimization Deep Dive**(Lagrangian 쌍대·KKT)를 선행 지식으로 전제합니다. RKHS를 처음 접한다면 [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) Ch5(RKHS)부터 학습하세요.

> 💡 **권장 선행**: 다변수정규분포와 조건부 기댓값은 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive), Bayesian 추론의 기본은 [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) Ch6에서 학습할 수 있습니다. 스펙트럴 분해와 양정치 행렬은 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)에서.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Kernel의_기초·Mercer-7A4ED9?style=for-the-badge)](./ch1-kernel-basics/01-positive-definite-kernel.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-RKHS·Representer-7A4ED9?style=for-the-badge)](./ch2-rkhs-representer/01-moore-aronszajn.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Support_Vector_Machine-7A4ED9?style=for-the-badge)](./ch3-svm/01-hard-margin-svm.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Gaussian_Process-7A4ED9?style=for-the-badge)](./ch4-gaussian-process/01-gp-definition.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-KRR·Kernel_PCA-7A4ED9?style=for-the-badge)](./ch5-krr-kpca/01-kernel-ridge-regression.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-MMD·Two--Sample_Test-7A4ED9?style=for-the-badge)](./ch6-mmd/01-mmd-definition.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-MKL·Random_Features·NTK-7A4ED9?style=for-the-badge)](./ch7-advanced-kernel/01-multiple-kernel-learning.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Kernel의 기초와 Mercer 정리

> **핵심 질문:** Positive definite kernel은 어떻게 정의되는가? RBF/Polynomial/Laplace가 왜 PD이고 Sigmoid tanh는 왜 아닌가? Mercer 정리의 고유함수 전개는 어떻게 implicit feature map이 되는가? Characteristic kernel은 무엇이 다른가?

<details>
<summary><b>PD kernel 정의부터 Characteristic kernel까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Positive Definite Kernel의 정의](./ch1-kernel-basics/01-positive-definite-kernel.md) | $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 대칭 + $\sum \alpha_i \alpha_j k(x_i, x_j) \geq 0$의 엄밀한 정의, 그람 행렬 $K = [k(x_i, x_j)]$의 **양정치성**과의 동치, conditionally PD와의 차이 |
| [02. 대표 커널의 목록과 PD 증명](./ch1-kernel-basics/02-kernel-zoo.md) | **Linear** $x^\top y$, **Polynomial** $(x^\top y + c)^d$, **Gaussian/RBF** $\exp(-\|x-y\|^2 / 2\sigma^2)$, **Laplace** $\exp(-\|x-y\|/\sigma)$ 각각의 PD 증명, **Sigmoid** $\tanh$가 PD가 아닌 반례 |
| [03. Kernel 연산 — Sum·Product·Composition](./ch1-kernel-basics/03-kernel-operations.md) | $k_1 + k_2$, $k_1 \cdot k_2$, $f(x) k(x, y) f(y)$가 모두 PD임을 직접 증명(그람 행렬 관점), **새 kernel을 기존 kernel로 설계**하는 방법론, ANOVA kernel·tensor product kernel |
| [04. Mercer 정리의 서술과 해석](./ch1-kernel-basics/04-mercer-theorem.md) | 컴팩트 집합 위 연속 PD kernel은 **$k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$** 고유함수 전개, feature map $\phi(x) = (\sqrt{\lambda_n} \phi_n(x))_n$이 $\ell^2$에 속함, kernel trick의 **수학적 정당성** |
| [05. Characteristic Kernel과 Universal Kernel](./ch1-kernel-basics/05-characteristic-universal.md) | **Characteristic**: mean embedding $\mu_p = \mathbb{E}_p[k(\cdot, X)]$가 $p \mapsto \mu_p$로 단사 → MMD의 근거, **Universal**: $C(\mathcal{X})$에서 dense(Stone-Weierstrass), Gaussian RBF가 두 성질 모두 만족함의 증명 |

</details>

<br/>

### 🔹 Chapter 2: RKHS와 Representer 정리

> **핵심 질문:** PD kernel 하나로 어떻게 Hilbert 공간이 "유일하게" 구성되는가(Moore-Aronszajn)? 재생성질 $f(x) = \langle f, k_x \rangle$가 왜 kernel method의 중심 도구인가? Representer 정리가 왜 무한차원 최적화를 유한차원으로 환원시키는가?

<details>
<summary><b>RKHS 구성부터 Representer 정리까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. RKHS 구성 (Moore-Aronszajn)](./ch2-rkhs-representer/01-moore-aronszajn.md) | PD kernel $k$ 주어지면 **유일한 RKHS $\mathcal{H}_k$ 존재**, 구성 단계: $\text{span}\{k_x\}$에 내적 $\langle k_x, k_y \rangle := k(x, y)$ 정의 → well-defined 검증 → 완비화, 유일성 증명 |
| [02. 재생성질과 평가범함수](./ch2-rkhs-representer/02-reproducing-property.md) | **재생성질** $f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}$ 증명, 평가 $\delta_x : f \mapsto f(x)$가 **유계 선형범함수**(Riesz 표현 정리로 유도), $\mathcal{H}_k$의 노름이 함수 매끄러움을 제어하는 의미 |
| [03. Representer 정리 완전 증명](./ch2-rkhs-representer/03-representer-theorem.md) | $\min_{f \in \mathcal{H}_k} \sum L(y_i, f(x_i)) + \Omega(\|f\|_{\mathcal{H}_k})$의 **최적해는 $f^* = \sum_{i=1}^n \alpha_i k(\cdot, x_i)$**, 증명: $f = f_\parallel + f_\perp$ 분해 → $f_\perp$가 손실에 영향 없고 노름만 증가 |
| [04. Representer 정리의 계산적 의미](./ch2-rkhs-representer/04-computational-reduction.md) | 무한차원 $\mathcal{H}_k$의 최적화가 **유한차원 $\alpha \in \mathbb{R}^n$ 최적화로 환원** → kernel trick의 근거, SVM·KRR·KPCA·GP가 **모두 같은 형태**를 갖는 이유의 통합 관점 |
| [05. $\mathcal{H}_k$의 함수 공간적 성질](./ch2-rkhs-representer/05-rkhs-function-spaces.md) | Gaussian RBF의 $\mathcal{H}_k$가 **Sobolev 공간 $H^s$**와 어떻게 관련되는지, 차원별 근사 오차, 어떤 함수가 RKHS에 속하고 어떤 것이 안 속하는가(예: 불연속 함수), $\mathcal{H}_k$의 크기와 kernel 선택의 trade-off |

</details>

<br/>

### 🔹 Chapter 3: Support Vector Machine (SVM)

> **핵심 질문:** Margin 최대화는 왜 $\min \frac{1}{2}\|w\|^2$가 되는가? 쌍대 문제가 왜 $\max \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j k(x_i, x_j)$ 형태인가? Support vector의 KKT 해석, soft-margin의 hinge loss 재작성은?

<details>
<summary><b>Hard-margin부터 SVR까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Margin 최대화와 Hard-margin SVM](./ch3-svm/01-hard-margin-svm.md) | 선형 분리 가능 데이터에서 margin $= 2/\|w\|$ 최대화 $\iff \min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$의 **기하학적 유도**, 분리 초평면의 일의성 |
| [02. 라그랑주 쌍대와 Dual Form](./ch3-svm/02-lagrange-dual.md) | Lagrangian $L = \frac{1}{2}\|w\|^2 - \sum \alpha_i(y_i(w^\top x_i + b) - 1)$, KKT로 $w = \sum \alpha_i y_i x_i$, dual $\max \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j x_i^\top x_j$의 **완전 유도** |
| [03. Kernel SVM](./ch3-svm/03-kernel-svm.md) | Dual에서 $x^\top y \to k(x, y)$ 치환의 **Representer 정리 정당화**, 예측 $\hat{y}(x) = \text{sign}(\sum \alpha_i y_i k(x_i, x) + b)$, RBF·Polynomial kernel 하의 decision boundary 시각화 |
| [04. Soft-margin SVM과 Hinge Loss](./ch3-svm/04-soft-margin-hinge.md) | Slack $\xi_i \geq 0$로 분리 불가능 데이터 처리, $\min \frac{1}{2}\|w\|^2 + C \sum \xi_i$, **Hinge loss** $\max(0, 1 - y_i f(x_i))$로 재작성, $C$의 bias-variance trade-off |
| [05. SMO (Sequential Minimal Optimization)](./ch3-svm/05-smo.md) | 한 번에 $\alpha_i, \alpha_j$ **두 개만 업데이트**하는 해석적 공식 유도(Platt 1998), KKT violation 기반 working set 선택, 수렴성, cvxpy 구현과 성능 비교 |
| [06. SVM Regression (SVR)](./ch3-svm/06-svr.md) | **$\epsilon$-insensitive loss** $\max(0, \|y - f(x)\| - \epsilon)$, primal/dual 유도, $\epsilon$-tube의 기하학적 해석, $\epsilon$과 $C$의 상호작용 |

</details>

<br/>

### 🔹 Chapter 4: Gaussian Process

> **핵심 질문:** GP는 왜 "함수에 대한 정규분포"인가? Posterior mean과 variance는 어떻게 유도되는가? 왜 GP posterior mean이 Kernel Ridge Regression과 완전히 같은가? Marginal likelihood가 왜 자동 Occam's razor인가?

<details>
<summary><b>GP 정의부터 Sparse GP까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. GP의 정의와 공분산 함수](./ch4-gaussian-process/01-gp-definition.md) | GP = "**임의 유한집합** $\{x_1, \dots, x_n\}$에 대해 $(f(x_1), \dots, f(x_n))$이 다변수정규분포", mean function $m(x)$·covariance function $k(x, y)$로 완전 특징화, Kolmogorov 확장 정리 |
| [02. GP Regression — Posterior 유도](./ch4-gaussian-process/02-gp-posterior.md) | Prior $f \sim \mathcal{GP}(0, k)$, 관측 $y = f(x) + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$, **joint Gaussian 조건부**로 posterior $f(x_*) \| y \sim \mathcal{N}(m_*, \sigma_*^2)$ 공식 유도 |
| [03. GP ⇔ Kernel Ridge Regression 동치](./ch4-gaussian-process/03-gp-equals-krr.md) | GP posterior mean $m_* = k_*^\top (K + \sigma^2 I)^{-1} y$가 **KRR 해와 정확히 일치**함을 증명, 차이는 GP가 분산(uncertainty)을 **추가로** 제공하는 것뿐, 수치 실험으로 값 단위 일치 확인 |
| [04. GP Classification과 Laplace Approximation](./ch4-gaussian-process/04-gp-classification.md) | Bernoulli likelihood $p(y \| f) = \sigma(yf)$의 **non-Gaussian posterior**, Laplace approximation으로 Gaussian 근사, Newton-Raphson 알고리즘, Expectation Propagation 대안 비교 |
| [05. Hyperparameter Learning — Marginal Likelihood](./ch4-gaussian-process/05-marginal-likelihood.md) | $\log p(y \| \theta) = -\frac{1}{2} y^\top K_\theta^{-1} y - \frac{1}{2} \log\|K_\theta\| - \frac{n}{2}\log 2\pi$ 최대화, **자동 Occam's razor**(복잡도 페널티 $-\frac{1}{2}\log\|K\|$의 의미), 경사법으로 length-scale 학습 |
| [06. Sparse GP와 Inducing Points](./ch4-gaussian-process/06-sparse-gp.md) | 풀 GP의 $O(n^3)$ 비용, **Inducing points** $Z \subset \mathcal{X}$로 FITC/VFE 근사, $O(n m^2)$ 계산 비용, **Titsias 2009**의 변분 근사 하한 유도 |

</details>

<br/>

### 🔹 Chapter 5: Kernel Ridge Regression과 Kernel PCA

> **핵심 질문:** Kernel Ridge는 어떻게 closed-form $(K + \lambda I)^{-1} y$를 얻는가? Kernel PCA의 centering은 왜 $\tilde{K} = K - \mathbf{1}K/n - K\mathbf{1}/n + \mathbf{1}K\mathbf{1}/n^2$인가? Spectral clustering이 왜 Kernel PCA의 특수 사례인가?

<details>
<summary><b>KRR 유도부터 Kernel k-Means까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Kernel Ridge Regression 완전 유도](./ch5-krr-kpca/01-kernel-ridge-regression.md) | Ridge의 dual 형태, Representer 정리 적용, **closed-form** $\alpha = (K + \lambda I)^{-1} y$, 예측 $f(x) = k(x)^\top (K + \lambda I)^{-1} y$, $\lambda$와 predictive smoothness의 관계 |
| [02. Kernel PCA의 수학](./ch5-krr-kpca/02-kernel-pca.md) | 특성공간에서 PCA, **centered Gram 행렬** $\tilde{K} = K - \mathbf{1}K/n - K\mathbf{1}/n + \mathbf{1}K\mathbf{1}/n^2$의 고유값 분해, projection $\phi(x) \cdot v_k = \sum \alpha_k^i k(x_i, x)$, 비선형 차원축소 |
| [03. Spectral Clustering — Graph Laplacian](./ch5-krr-kpca/03-spectral-clustering.md) | Similarity matrix로 graph 구성, **$L = D - W$의 고유벡터로 clustering**, Kernel PCA의 특수 사례임을 증명, Normalized cut과의 관계(Shi & Malik) |
| [04. Kernel k-Means](./ch5-krr-kpca/04-kernel-kmeans.md) | Feature space에서 k-means, **명시적 $\phi$ 없이 Gram 행렬로** 계산 — 거리 $\|\phi(x_i) - \mu_k\|^2 = k(x_i, x_i) - \frac{2}{\|C_k\|}\sum_{j \in C_k} k(x_i, x_j) + \frac{1}{\|C_k\|^2}\sum_{i, j \in C_k} k(x_i, x_j)$, 임의 모양 클러스터 탐지 |

</details>

<br/>

### 🔹 Chapter 6: Maximum Mean Discrepancy (MMD)와 Two-Sample Test

> **핵심 질문:** Mean embedding $\mu_p = \mathbb{E}_p[k(\cdot, X)]$는 왜 distribution을 유일하게 특징짓는가? MMD의 샘플 추정량은 어떻게 편향되지 않게 되는가? MMD-GAN은 왜 adversarial GAN보다 안정적인가?

<details>
<summary><b>MMD 정의부터 HSIC까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. MMD의 정의와 RKHS 해석](./ch6-mmd/01-mmd-definition.md) | $\text{MMD}(p, q; \mathcal{H}) = \|\mu_p - \mu_q\|_{\mathcal{H}}$, mean embedding $\mu_p = \mathbb{E}_{X \sim p}[k(\cdot, X)]$, **characteristic kernel 하에서 $\text{MMD} = 0 \iff p = q$** 완전 증명 (Gretton 2012) |
| [02. MMD의 샘플 추정량](./ch6-mmd/02-mmd-estimator.md) | $\widehat{\text{MMD}}^2 = \frac{1}{n^2}\sum k(x_i, x_j) - \frac{2}{nm}\sum k(x_i, y_j) + \frac{1}{m^2}\sum k(y_i, y_j)$의 biased/unbiased 버전, **U-statistic 분해**, 수렴률 $O(1/\sqrt{n})$ |
| [03. Two-Sample Test (Gretton et al. 2012)](./ch6-mmd/03-two-sample-test.md) | Null $H_0: p = q$, **MMD² 분포**를 permutation test나 점근분포로 critical value 설정, 고차원·구조화 데이터에서 왜 강력한가, Kolmogorov-Smirnov/energy distance와의 비교 |
| [04. MMD-GAN과 생성모델](./ch6-mmd/04-mmd-gan.md) | Generator 출력 분포와 데이터 분포의 **MMD 최소화**, adversarial 훈련보다 안정적인 이유, kernel 선택의 중요성, MMD-GAN(Li et al. 2015) 아키텍처, Wasserstein GAN과의 비교 |
| [05. Kernel Embedding 일반화 — HSIC·Distribution Regression](./ch6-mmd/05-kernel-embedding-generalizations.md) | **Hilbert-Schmidt Independence Criterion**(HSIC)로 독립성 검정, Distribution regression ($x \mapsto p_x$의 regression), conditional mean embedding, 인과추론에서의 응용 |

</details>

<br/>

### 🔹 Chapter 7: Kernel Method 심화 주제

> **핵심 질문:** 여러 kernel을 어떻게 결합하는가(MKL)? Bochner 정리로 왜 RBF가 유한차원 Random Feature로 근사되는가? NN과 GP의 하이브리드(Deep Kernel)? 무한폭 NN의 NTK가 왜 RKHS 재생핵인가?

<details>
<summary><b>MKL부터 NTK까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Multiple Kernel Learning (MKL)](./ch7-advanced-kernel/01-multiple-kernel-learning.md) | 여러 kernel의 **볼록 결합** $k = \sum \beta_i k_i$, $\beta$ 학습을 SDP 또는 경사법으로, **automatic kernel selection**, SimpleMKL 알고리즘, $\ell_p$-norm MKL |
| [02. Random Features (Rahimi & Recht 2007)](./ch7-advanced-kernel/02-random-features.md) | **Bochner 정리**로 shift-invariant kernel의 Fourier 표현 $k(x - y) = \int e^{i\omega^\top(x - y)} p(\omega) d\omega$, 유한차원 $\phi(x) = \sqrt{2/D}(\cos(\omega_i^\top x + b_i))_i$, 빅데이터 대응 $O(n D^2)$ |
| [03. Deep Kernel Learning](./ch7-advanced-kernel/03-deep-kernel-learning.md) | Neural Network feature $\phi_\theta(x)$ 위에 kernel $k(\phi_\theta(x), \phi_\theta(y))$, **GP와 NN의 하이브리드**, Wilson et al. 2016, end-to-end marginal likelihood 학습 |
| [04. Neural Tangent Kernel (NTK) 연결](./ch7-advanced-kernel/04-ntk-connection.md) | 무한폭 NN의 **$\Theta(x, y) = \lim_{\text{width} \to \infty} \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle$**이 PD → RKHS 존재, **NN 훈련 = NTK-RKHS kernel regression**(Jacot et al. 2018), Layer 2 Generalization Theory로의 다리 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 200+ 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **PD kernel ⇔ Gram 행렬 양정치** | $k$가 PD $\iff$ 모든 유한 집합에 대해 $K = [k(x_i, x_j)] \succeq 0$ | [Ch1-01](./ch1-kernel-basics/01-positive-definite-kernel.md) |
| **Kernel 연산 보존** | $k_1 + k_2$, $k_1 \cdot k_2$, $f(x) k(x, y) f(y)$가 모두 PD | [Ch1-03](./ch1-kernel-basics/03-kernel-operations.md) |
| **Mercer 정리** | 연속 PD kernel은 $k(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$ — implicit feature map이 무한차원 | [Ch1-04](./ch1-kernel-basics/04-mercer-theorem.md) |
| **Characteristic ⇒ 분포 구분** | $k$가 characteristic $\Rightarrow$ $p \mapsto \mu_p$ 단사 $\Rightarrow$ $\text{MMD} = 0 \iff p = q$ | [Ch1-05](./ch1-kernel-basics/05-characteristic-universal.md) |
| **Moore-Aronszajn** | 각 PD kernel에 대해 **유일한 RKHS**가 존재 | [Ch2-01](./ch2-rkhs-representer/01-moore-aronszajn.md) |
| **재생성질** | $f(x) = \langle f, k_x \rangle_{\mathcal{H}_k}$ — 평가는 $k_x$와의 내적 | [Ch2-02](./ch2-rkhs-representer/02-reproducing-property.md) |
| **Representer 정리** | $\min_{f \in \mathcal{H}_k} \sum L(y_i, f(x_i)) + \Omega(\|f\|)$의 해는 $f^* = \sum_i \alpha_i k(\cdot, x_i)$ | [Ch2-03](./ch2-rkhs-representer/03-representer-theorem.md) |
| **SVM 쌍대** | Primal $\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$의 dual이 $\max \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j k(x_i, x_j)$ | [Ch3-02](./ch3-svm/02-lagrange-dual.md) |
| **KKT로 Support Vector 특성화** | $\alpha_i > 0 \iff$ $x_i$는 support vector(경계 위 또는 margin 위반) | [Ch3-02](./ch3-svm/02-lagrange-dual.md) |
| **GP Posterior 공식** | $f(x_*) \| y \sim \mathcal{N}(k_*^\top (K + \sigma^2 I)^{-1} y,\ k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*)$ | [Ch4-02](./ch4-gaussian-process/02-gp-posterior.md) |
| **GP mean = KRR 해** | GP posterior mean $m_* = k_*^\top (K + \sigma^2 I)^{-1} y$가 KRR closed-form과 **값 단위 일치** | [Ch4-03](./ch4-gaussian-process/03-gp-equals-krr.md) |
| **Marginal likelihood 분해** | $\log p(y\|\theta) = -\frac{1}{2}y^\top K_\theta^{-1} y - \frac{1}{2}\log\|K_\theta\| - \frac{n}{2}\log 2\pi$ — data fit + Occam's razor | [Ch4-05](./ch4-gaussian-process/05-marginal-likelihood.md) |
| **KPCA centering 공식** | $\tilde{K} = K - \mathbf{1}K/n - K\mathbf{1}/n + \mathbf{1}K\mathbf{1}/n^2$이 특성공간의 평균을 원점으로 맞춤 | [Ch5-02](./ch5-krr-kpca/02-kernel-pca.md) |
| **Characteristic ⇒ MMD=0 iff p=q** | Gretton et al. (2012) — 분포 구분 기준으로서의 MMD | [Ch6-01](./ch6-mmd/01-mmd-definition.md) |
| **MMD 불편 추정량** | U-statistic 형태의 $\widehat{\text{MMD}}_u^2$는 $\mathbb{E}[\widehat{\text{MMD}}_u^2] = \text{MMD}^2$ | [Ch6-02](./ch6-mmd/02-mmd-estimator.md) |
| **Bochner 정리** | Shift-invariant PD kernel $\iff$ 음이 아닌 유한 측도의 Fourier 변환 | [Ch7-02](./ch7-advanced-kernel/02-random-features.md) |
| **Random Features 근사** | $\phi(x) = \sqrt{2/D}(\cos(\omega_i^\top x + b_i))_i \Rightarrow \mathbb{E}[\phi(x)^\top \phi(y)] = k(x, y)$ | [Ch7-02](./ch7-advanced-kernel/02-random-features.md) |
| **NTK의 PD성·NN 훈련과 kernel regression 동치** | Jacot et al. (2018) — 무한폭 NN의 gradient flow = NTK-RKHS kernel regression | [Ch7-04](./ch7-advanced-kernel/04-ntk-connection.md) |

> 💡 **챕터별 총 정리 수**: Ch1(31) · Ch2(28) · Ch3(37) · Ch4(34) · Ch5(22) · Ch6(27) · Ch7(21) — 합계 **200개 정리 + 증명**, 약 **13,700+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
cvxpy==1.4.0           # SVM dual QP 직접 풀기
scikit-learn==1.3.0    # sklearn 결과와 값 단위 비교
GPy==1.13.0            # Sparse GP · marginal likelihood 비교
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            cvxpy==1.4.0 scikit-learn==1.3.0 GPy==1.13.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — GP Regression 바닥부터 + KRR 동치 검증 + MMD로 분포 구분
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X, Y, sigma=1.0):
    d = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return np.exp(-d / (2 * sigma ** 2))

# ─────────────────────────────────────────────
# 1. GP Regression — Rasmussen & Williams (2.2)
# ─────────────────────────────────────────────
rng = np.random.default_rng(42)
X_train = rng.uniform(-3, 3, 20).reshape(-1, 1)
y_train = np.sin(X_train).flatten() + 0.1 * rng.standard_normal(20)

sigma_k, sigma_n = 1.0, 0.1
X_test = np.linspace(-5, 5, 200).reshape(-1, 1)

K    = rbf_kernel(X_train, X_train, sigma_k)
K_s  = rbf_kernel(X_train, X_test,  sigma_k)
K_ss = rbf_kernel(X_test,  X_test,  sigma_k)

L     = np.linalg.cholesky(K + sigma_n ** 2 * np.eye(len(X_train)))
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
mu    = K_s.T @ alpha                        # posterior mean
v     = np.linalg.solve(L, K_s)
std   = np.sqrt(np.diag(K_ss - v.T @ v))     # posterior std

plt.figure(figsize=(10, 4))
plt.fill_between(X_test.flatten(), mu - 2 * std, mu + 2 * std, alpha=0.3, label='95% CI')
plt.plot(X_test, mu, 'b-', label='Posterior mean')
plt.scatter(X_train, y_train, c='r', zorder=5, label='train')
plt.legend(); plt.title('GP Regression — 데이터에서 멀어지면 분산 증가')
plt.show()

# ─────────────────────────────────────────────
# 2. KRR과 GP posterior mean의 동치 검증
# ─────────────────────────────────────────────
lam       = sigma_n ** 2
alpha_krr = np.linalg.solve(K + lam * np.eye(len(X_train)), y_train)
mu_krr    = K_s.T @ alpha_krr
print(f'GP mean vs KRR mean 최대 차이: {np.max(np.abs(mu - mu_krr)):.2e}')
# → 1e-13 수준. 같은 식의 Bayesian/regularized risk 두 해석.

# ─────────────────────────────────────────────
# 3. MMD — 두 분포 구분
# ─────────────────────────────────────────────
def mmd2_biased(X, Y, sigma=1.0):
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

X = rng.standard_normal((100, 2))
Y = rng.standard_normal((100, 2)) + 0.5           # shifted
print(f'MMD² (같은 분포): {mmd2_biased(X, X):.4f}')  # ~0
print(f'MMD² (다른 분포): {mmd2_biased(X, Y):.4f}')  # 유의하게 큼
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 방법이 ML에서 중요한가** | SVM·GP·MMD·Random Features의 실전 의의 |
| 3 | 📐 **수학적 선행 조건** | Functional Analysis·Convex Opt·Linear Algebra·Probability 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | "kernel이 비선형성을 선형 방법에 주입"의 반복 비유 |
| 5 | ✏️ **엄밀한 정의** | PD kernel·RKHS·SVM primal·GP prior의 엄밀한 수식 |
| 6 | 🔬 **정리와 증명** | Mercer·Representer·GP posterior·MMD=0 ⟺ p=q — "자명하다" 없이 |
| 7 | 💻 **NumPy 구현 검증** | SVM dual(cvxpy)·GP 바닥 구현·Random Features 스케일링 |
| 8 | 🔗 **실전 활용** | 언제 kernel이 유리한가, 언제 딥러닝이 이기는가 |
| 9 | ⚖️ **가정과 한계** | $O(n^2)$ 메모리, $O(n^3)$ 계산, kernel 선택 감도 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현 문제 |

> 📚 **연습문제 총 105개**: 35문서 × 문서당 3문제(기초/심화/ML 연결), 모든 문제에 `<details>` 펼침 해설 포함. SVM dual 재유도부터 NTK-RKHS 해석까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되므로 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 390줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 35문서는 약 **40~50시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "SVM을 쓰지만 왜 쌍대 문제가 그 형태인지 모른다" — SVM 집중 (5일, 약 10~13시간)</b></summary>

<br/>

```
Day 1  Ch1-01~02  PD kernel 정의와 대표 kernel zoo
Day 2  Ch1-04     Mercer 정리 — implicit feature map이 무한차원
Day 3  Ch2-03     Representer 정리 — SVM이 왜 $\sum \alpha_i k$ 형태
Day 4  Ch3-01~02  Hard-margin SVM과 라그랑주 쌍대 완전 유도
Day 5  Ch3-03~04  Kernel SVM과 Soft-margin/Hinge loss
```

</details>

<details>
<summary><b>🟡 "GP를 쓰지만 covariance 선택이 왜 prior 선택인지 모른다" — GP 집중 (1주, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch1-01, Ch1-04  PD kernel과 Mercer
Day 2  Ch2-01~03       RKHS 구성과 Representer 정리
Day 3  Ch4-01          GP 정의 — 함수에 대한 정규분포
Day 4  Ch4-02          GP posterior 유도 (joint Gaussian 조건부)
Day 5  Ch4-03          GP ⇔ KRR 동치 — 값 단위 실험 일치 확인
Day 6  Ch4-05          Marginal likelihood — 자동 Occam's razor
Day 7  Ch4-06          Sparse GP·Inducing Points — 스케일링
```

</details>

<details>
<summary><b>🔴 "Kernel Method의 수학적 기반을 완전 정복한다" — 전체 정복 (7주, 약 40~50시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — Kernel 기초와 Mercer
        → PD 정의부터 characteristic/universal까지
        → RBF·Polynomial·Laplace의 PD 증명 직접 재구성

2주차  Chapter 2 전체 — RKHS와 Representer 정리
        → Moore-Aronszajn 구성의 매 단계 이해
        → Representer 정리로 "모든 kernel method가 같은 형태" 통찰

3주차  Chapter 3 전체 — SVM
        → Primal ↔ Dual 변환을 KKT로 한 줄씩
        → cvxpy로 SVM dual QP 바닥 구현, sklearn 결과와 비교

4주차  Chapter 4 전체 — Gaussian Process
        → GP posterior 공식 직접 유도
        → KRR과의 값 단위 일치 실험
        → Marginal likelihood로 length-scale 학습

5주차  Chapter 5 전체 — KRR·Kernel PCA·Spectral Clustering
        → KPCA centering 공식의 의미
        → Spectral clustering이 KPCA 특수 사례임을 확인

6주차  Chapter 6 전체 — MMD
        → Characteristic kernel로 MMD=0 iff p=q 증명 재구성
        → Two-sample test 구현 (permutation)
        → MMD-GAN 토이 실험 (2D mixture)

7주차  Chapter 7 전체 — Random Features·Deep Kernel·NTK
        → Bochner 정리로 RBF 근사
        → Random Features로 n = 10^5 KRR 스케일링
        → NTK를 초기화 NN으로 수치 추정
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | Hilbert 공간, Riesz 표현, RKHS, Mercer, Moore-Aronszajn | Ch1-04(Mercer), Ch2-01~02(RKHS 구성과 재생성질) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 양정치 행렬, 스펙트럴 분해, 고유값 | Ch1-01(Gram 양정치), Ch5-02(KPCA 고유분해) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 다변수정규분포, 조건부 기댓값, 특성함수 | Ch4-02(GP posterior의 joint Gaussian 조건부), Ch7-02(Bochner) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | Lagrangian 쌍대, KKT, QP | Ch3-02(SVM dual), Ch3-05(SMO), Ch7-01(MKL의 SDP) |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | Bayesian 추론, posterior·marginal likelihood | Ch4 전체(GP의 Bayesian 해석) |
| [bayesian-ml-deep-dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive) | Variational inference, BNN, Bayesian Optimization | Ch4-05~06(VFE 변분 근사), Ch7-03(Deep Kernel) |
| [generalization-theory-deep-dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) | Rademacher complexity, 이중 강하, NTK | Ch7-04(NTK-RKHS 연결) — 이 레포의 후속 |

> 💡 이 레포는 **kernel method의 통일 프레임워크**에 집중합니다. Functional Analysis의 RKHS·Mercer 이론을 선행하면 Ch1~2가 훨씬 자연스럽고, Convex Optimization의 쌍대 이론을 선행하면 Ch3(SVM)가 "왜 그 형태인지" 완전 투명해집니다. Ch7의 NTK는 **Generalization Theory** 레포에서 더 깊이 다룹니다.

---

## 📖 Reference

### 🏛️ Kernel Method 표준 교재
- **Learning with Kernels** (Schölkopf & Smola, 2002) — 현대 kernel method 표준 교재
- **Kernel Methods for Pattern Analysis** (Shawe-Taylor & Cristianini, 2004) — SVM·KPCA의 표준 레퍼런스
- **Gaussian Processes for Machine Learning** (Rasmussen & Williams, 2006) — **GP의 바이블**
- **Reproducing Kernel Hilbert Spaces in Probability and Statistics** (Berlinet & Thomas-Agnan, 2004) — RKHS 이론 심화

### ⚙️ SVM 원전과 SMO
- **Support-Vector Networks** (Cortes & Vapnik, 1995) — **SVM 원전**
- **Statistical Learning Theory** (Vapnik, 1998) — VC 이론과 SVM의 기반
- **Training Support Vector Machines: Sequential Minimal Optimization** (Platt, 1998) — **SMO 원전**
- **A Tutorial on Support Vector Regression** (Smola & Schölkopf, 2004) — SVR 표준 리뷰

### 🌀 MMD · Two-Sample Test · 생성모델
- **A Kernel Two-Sample Test** (Gretton et al., 2012) — **MMD 원전**
- **Kernel Mean Embedding of Distributions: A Review and Beyond** (Muandet et al., 2017) — MMD 리뷰
- **Generative Moment Matching Networks** (Li et al., 2015) — **MMD-GAN 계열**
- **Training Generative Neural Networks via Maximum Mean Discrepancy Optimization** (Dziugaite et al., 2015)

### 🚀 Scaling · Random Features · Deep Kernel
- **Random Features for Large-Scale Kernel Machines** (Rahimi & Recht, 2007) — **Random Features 원전**
- **Weighted Sums of Random Kitchen Sinks** (Rahimi & Recht, 2008) — 후속
- **Deep Kernel Learning** (Wilson et al., 2016) — NN + GP 하이브리드
- **Variational Learning of Inducing Variables in Sparse Gaussian Processes** (Titsias, 2009) — Sparse GP VFE

### 🧠 NTK · Kernel-NN 연결
- **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** (Jacot, Gabriel & Hongler, 2018) — **NTK 원전**
- **On Exact Computation with an Infinitely Wide Neural Net** (Arora et al., 2019) — NTK의 정확 계산
- **Gradient Descent as a Kernel Method** (Yang, 2020) — 무한폭 NN의 kernel 해석

### 📐 Functional Analysis 기반
- **A Hilbert Space Embedding for Distributions** (Smola, Gretton, Song & Schölkopf, 2007) — mean embedding
- **Characteristic Kernels on Groups and Semigroups** (Fukumizu et al., 2008) — characteristic kernel 이론
- **Universality, Characteristic Kernels and RKHS Embedding of Measures** (Sriperumbudur, Fukumizu & Lanckriet, 2011)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"Kernel을 사용하는 것과, 왜 하나의 PD 함수 $k(x, y)$가 Hilbert 공간·implicit feature map·SVM 쌍대·GP prior·MMD를 모두 결정하는지를 증명할 수 있는 것은 다르다"*

</div>
