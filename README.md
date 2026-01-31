# Empirical Influence Function — PyTorch Implementation

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PyTorch implementation of influence function methods** for understanding how individual training samples affect model predictions. Includes the classic ICML 2017 influence function, **TracIn** (NeurIPS 2020), and **EmpiricalIF** (NeurIPS 2022) for fast, single-checkpoint influence estimation without inverse Hessian.

---

## Table of Contents

- [What is an Influence Function?](#-what-is-an-influence-function)
- [How It Works](#-how-influence-function-works)
- [Methods Included](#-methods-included)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Citation](#-citation)
- [Author & Contact](#-author--contact)

---

## 🧠 What is an Influence Function?

The **influence function** (*Understanding Black-box Predictions via Influence Functions*, ICML 2017) answers:

> **How much does a single training point affect the model's prediction or loss on a specific test point?**

Instead of retraining after removing or perturbing a training sample, the influence function **estimates** the change in the model’s prediction (or loss) on a test point using gradient and Hessian approximations — **no retraining required**.

---

## 🧪 How Influence Function Works

<!---
$$
\text{Influence}(z_i, z_{\text{test}}) = - \nabla_\theta \mathcal{L}(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_i, \hat{\theta})
$$
!--->
<img width="450" alt="Classic influence function formula: gradient of test loss times inverse Hessian times gradient of training loss" src="https://github.com/user-attachments/assets/2b51e430-b063-4cf3-abde-1a7e970401c2" />

Where $z_i$ is a training sample, $z_{\text{test}}$ is the test sample, $\hat{\theta}$ are the trained model parameters, $\mathcal{L}$ is the loss function.  
$H_{\hat{\theta}}$ is the Hessian of the **total** training loss at $\hat{\theta}$, i.e. $H_{\hat{\theta}} = \frac{1}{n} \sum_{i=1}^{n} \nabla^2_\theta \mathcal{L}(z_i, \theta) \bigg|_{\theta = \hat{\theta}}$

- **Positive influence** → keeping this training point increases test loss → **harmful** for this test point  
- **Negative influence** → keeping this point decreases test loss → **helpful**

---

## ❗ Limitations of the Original Influence Function

The main bottleneck is **estimating the inverse Hessian**. Conjugate gradient or damping can be expensive and unstable. This repo provides two **lightweight alternatives** that avoid the full inverse Hessian:

| Paper | Method | Venue |
| --- | --- | --- |
| Estimating Training Data Influence by Tracing Gradient Descent | **TracIn** | NeurIPS 2020 |
| Debugging and Explaining Metric Learning Approach: An Influence Function Perspective | **EmpiricalIF** | NeurIPS 2022 |

### Intuition: TracIn

<img width="464" alt="TracIn formula: average over checkpoints of gradient dot product between test and training sample" src="https://github.com/user-attachments/assets/342a758b-1c37-422e-8ef2-af95533f55a5" />

With $T$ checkpoints from training, TracIn measures **gradient alignment** (dot product) at each checkpoint:

- **Positive** → $z_i$ helped reduce test loss  
- **Negative** → $z_i$ hurt test performance (possibly harmful)

TracIn is **first-order** (no Hessian); it needs **multiple checkpoints** for good estimates.

### Intuition: EmpiricalIF

<div style="text-align: center;">
<img width="609" alt="EmpiricalIF formula: expectation of loss change alignment under parameter perturbation" src="https://github.com/user-attachments/assets/c8a46a8a-9a54-43fa-9df3-d076d2be8575" />
</div>

EmpiricalIF uses the **final checkpoint** only. It perturbs $\hat{\theta}$ with $\delta$ (e.g. on a sphere of radius $r$) and measures **loss-change alignment**:

- **Positive** → $z_i$ and test $z_{\text{test}}$ co-evolve → helpful  
- **Negative** → $z_i$ conflicts with test → harmful  

EmpiricalIF is a **single-checkpoint** variant of TracIn. In practice, using the steepest descent and ascent directions for the test loss is enough to compute it.

---

## 🛠️ Installation

1. **Install PyTorch** (match your CUDA version):  
   [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Quick Start

**Inputs:**

- `dl_train`: `torch.utils.data.DataLoader` for training data  
- `model`: `nn.Module`  
- `param_filter_fn`: which parameters to use (e.g. last layer only)  
- `criterion`: loss with `reduction="none"`  

**Output:**

- `IF.query_influence(test_input, test_target)` returns a list of influence scores of length `|dl_train|`, one per training sample.

---

## 📖 Usage Examples

### Empirical IF (recommended for speed)

```python
from src.IF import EmpiricalIF

IF = EmpiricalIF(dl_train=trainloader,
                 model=resnet18,
                 param_filter_fn=lambda name, param: 'fc' in name,
                 criterion=nn.CrossEntropyLoss(reduction="none"))

for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)  # shape: (|dl_train|,)
```

**Reverse check** (perturb top/bottom influence samples and compare):

```python
most_inf, least_inf = IF.reverse_check(
    query_input=test_input,
    query_target=test_target,
    influence_values=IF_scores,
    check_ratio=0.01  # top and bottom 1%
)

for idx, orig_if, rev_if in most_inf:
    print(f"Top IF sample {idx}: IF={orig_if:.4f}, Reverse IF={rev_if:.4f}")

for idx, orig_if, rev_if in least_inf:
    print(f"Bottom IF sample {idx}: IF={orig_if:.4f}, Reverse IF={rev_if:.4f}")
```

### Original Influence Function (ICML 2017)

```python
from src.IF import BaseInfluenceFunction

IF = BaseInfluenceFunction(dl_train=trainloader,
                           model=resnet18,
                           param_filter_fn=lambda name, param: 'fc' in name,
                           criterion=nn.CrossEntropyLoss(reduction="none"))

for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)
```

### TracIn

```python
from src.IF import TracIn

IF = TracIn(dl_train=trainloader,
            model=resnet18,
            param_filter_fn=lambda name, param: 'fc' in name,
            criterion=nn.CrossEntropyLoss(reduction="none"))

for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)
```

---

## 📚 Citation

If you use this repository, please cite the EmpiricalIF paper:

```bibtex
@article{liu2022debugging,
  title={Debugging and Explaining Metric Learning Approaches: An Influence Function Based Perspective},
  author={Liu, Ruofan and Lin, Yun and Yang, Xianglin and Dong, Jin Song},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={7824--7837},
  year={2022}
}
```

---

## 👤 Author & Contact

**KuchikiRenji**

| | |
| --- | --- |
| **GitHub** | [github.com/KuchikiRenji](https://github.com/KuchikiRenji) |
| **Email** | [KuchikiRenji@outlook.com](mailto:KuchikiRenji@outlook.com) |
| **Discord** | `kuchiki_renji` |

For questions, collaborations, or feedback about this implementation, reach out via the links above.
