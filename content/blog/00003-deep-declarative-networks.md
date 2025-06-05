---
title: Deep Declarative Networks
description: A New Hope — Paper Review
type: article
date: 2025-05-25
categories:
  - Optimization
  - Machine Learning
  - Paper Review
tags:
  - Smooth Optimization
  - Neural Networks
  - Lagrange Multipliers
  - Linear Algebra
  - Calculus
slug: deep-declarative-networks
---

The paper <a href="https://arxiv.org/abs/1909.04866" rel="noreferrer external" target="_blank">“Deep Declarative Networks: A New Hope” by Gould et al.</a> proposes a novel architectural paradigm that blends optimization layers directly into deep learning models. Rather than just stacking layers of differentiable functions, Deep Declarative Networks (DDNs) allow certain layers to implicitly solve optimization problems — and do so in a way that’s fully differentiable and compatible with backpropagation.

# 🔍 What’s the big idea?

At the core of DDNs is the insight that many useful computations in machine learning (e.g., projections onto constraints, solving small QPs, estimating parameters under constraints) are more naturally posed as optimization problems rather than function approximations. Instead of approximating such operations with trainable neural nets, DDNs define them declaratively — via an optimization problem — and then embed them into the network as a layer.

These “declarative nodes” return the solution to an optimization problem, and the authors show how to compute gradients through these nodes using the implicit function theorem — a classic tool in applied math that’s having a bit of a renaissance in differentiable programming.

# 🧠 Why does this matter?

This approach gives models inductive bias grounded in optimization structure. For example, a model can directly project predictions onto a feasible region, or perform MAP estimation over structured variables as part of the forward pass. Instead of learning this structure from scratch, you just declare it.

DDNs thus combine the flexibility of deep learning with the precision and control of optimization. This opens up powerful hybrid modeling options, especially in settings like:
* Structured prediction
* Robotics and control
* Computer vision (e.g. pose estimation)
* Probabilistic programming

# 🛠️ Technical highlights
* The authors provide an elegant formulation of declarative nodes, and a recipe for computing backward gradients using the implicit function theorem.
* They show that this method generalizes standard differentiable operations, but is especially useful when constraints or structure are critical.
* Experiments demonstrate improved performance in structured tasks, particularly where constraints matter.

# 🤔 Final thoughts

This paper is an excellent example of the growing trend toward differentiable programming and hybrid learning systems. It offers a clean, elegant way to incorporate domain knowledge (via optimization problems) into deep networks without sacrificing end-to-end training.

If you’re working on models that require structured outputs, constrained inference, or optimization-in-the-loop, Deep Declarative Networks deserve a close look. They might just be the right abstraction layer between deep learning and mathematical modeling.

**Recommended for**: ML researchers, applied scientists in control/vision, and anyone interested in the convergence of optimization and learning.

# 📚 References
1. Gould, S., Hartley, R., & Campbell, D. (2020).

    **Deep Declarative Networks: A New Hope**

    <a href="https://arxiv.org/abs/1909.04866" rel="noreferrer external" target="_blank">arXiv:1909.04866</a>

## 📜 Related Worth-to-Read Papers
1. Gould, S. et al. (2016).

    **On Differentiating Parameterized Argmin and Argmax Problems with Application to Bi-level Optimization**

    <a href="https://arxiv.org/abs/1607.05447" rel="noreferrer external" target="_blank">arXiv:1607.05447</a>

2. Agrawal, A., Amos, B., Barratt, S., Boyd, S., & Kolter, J. Z. (2019).

    **Differentiable Convex Optimization Layers**

    <a href="https://arxiv.org/abs/1910.12430" rel="noreferrer external" target="_blank">arXiv:1910.12430</a>
