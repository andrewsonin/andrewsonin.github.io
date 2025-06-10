---
title: Differentiable Decision Makers
description: Learning Structured Decision Procedures with Gradients and Networks
type: article
date: 2025-06-01
categories:
  - Optimization
  - Machine Learning
tags:
  - Smooth Optimization
  - Neural Networks
  - Lagrange Multipliers
  - Linear Algebra
  - Calculus
slug: optimize-decision-makers
---

In many modern optimization pipelines — from classical machine learning models to cutting-edge AI systems — the process of tuning parameters and hyperparameters remains stubbornly discrete, heuristic, and slow. What if, instead of treating entire algorithms as black boxes, we could differentiate through them just like we do through neural networks? What if the logic behind decision-making could itself become part of a smooth, trainable computational graph? This article introduces the concept of **differentiable algorithms** — a class of computational procedures where every operation, even those involving optimization problems, supports gradient propagation. The implications are profound: from faster training to deeper algorithm integration, this paradigm opens a new frontier in algorithm design and learning.
## **🔥 Why Differentiable Algorithms Might Be the Future of Optimization**

In the course of my recent research, I stumbled upon an intriguing pattern:

1. If an algorithm contains a logical decision-making block, it can often — under reasonable assumptions — be reformulated as the **optimizer’s argument** for a smooth function.

2. A surprisingly wide class of algorithms, much larger than you’d initially expect, can be expressed as **compositions** of the following building blocks:

    - Smooth functions of input data and hyperparameters,

    - Optimization arguments (e.g., `argmin` or `argmax`) of smooth functions over the same.

If every component in this composition is smooth, then the entire construction is also smooth. We’ll refer to this function as \(w\), and such an algorithm as a **differentiable algorithm**. If the representation is exact rather than approximate, we call it a **strictly differentiable algorithm**.

Let’s dig into why this perspective is so powerful.

## **🎯 Why Should You Care?**

The main appeal of differentiable algorithms is this:

**They allow you to optimize hyperparameters using gradient descent instead of costly grid or random search** — often with dramatically lower time and memory complexity.  

Here’s a conceptual diagram illustrating how such an algorithm flows:

![Differentiable Algorithm Graph](../../blog/dag.svg)
In this diagram:

- **Imperative tensors** represent blocks defined by _explicit_ smooth functions.

- **Declarative tensors** represent _implicit_ functions — outputs of optimization problems over smooth objectives.

This terminology is based on the <a href={{< ref "./00003-deep-declarative-networks.md" >}} target="_blank">Deep Declarative Networks</a> framework we’ve discussed before.
## **🚀 Gradient-Based Hyperparameter Tuning**

Let \(w:\Theta\times\mathcal X\to\mathcal W\), where:
- \(\Theta\subseteq\mathbb R^P\): the hyperparameter space,
- \(\mathcal X\subseteq\mathbb R^N\): the input space,
- \(\mathcal W\subseteq\mathbb R^W\): the output space of the algorithm.

Given a dataset \(\bold X = [\boldsymbol x_1, \dots, \boldsymbol x_S]\), define:

$$w\left(\boldsymbol\theta,\bold X\right) = \left[w\left(\boldsymbol\theta,\boldsymbol x_1\right),\dots,w\left(\boldsymbol\theta,\boldsymbol x_S\right)\right]$$
  
If we can evaluate performance using a smooth loss function \(\mathrm{Loss}:\mathcal W^S\to\mathbb R\), then finding the optimal hyperparameters becomes a straightforward gradient-based optimization:

$$\widehat{\boldsymbol\theta}\left(\bold X\right)=\argmin_{\boldsymbol\theta}\mathrm{Loss}\left(w\left(\boldsymbol\theta,\bold X\right)\right)$$

This can be solved using backpropagation or similar gradient-based methods.

This problem can generally be solved using <a href={{< ref "./00001-from-discrete-to-smooth.md" >}} target=”_blank”>gradient descent or its variations</a>.

At each step, we need to compute the tensor of partial derivatives \(\partial_{\boldsymbol\theta}w\left(\boldsymbol\theta,\bold X\right)\), which can be done via the chain rule — or, as it is more commonly known in the AI community, **backpropagation**. This method expresses each element of \(\partial_{\boldsymbol\theta}w\left(\boldsymbol\theta,\bold X\right)\) as a product of partial derivatives of all logical blocks in the algorithm.

The tuning process for hyperparameters over a dataset is illustrated below:

![Differentiable Algorithm Training](../../blog/diffalg.svg)
## **🧠 Integration with Neural Networks**

What if we want to _predict_ good hyperparameters on the fly?

Because differentiable algorithms are, well, differentiable, we can integrate them directly into a machine learning pipeline — **including neural networks** — without:
- Designing special loss functions for training,

- Losing accuracy through output calibration tricks.

Here’s how such an integration might look in practice:

![Differentiable Algorithm — Neural Network Training](../../blog/diffalg-neural.svg)

## **⚠️ When Differentiation Gets Hard: The Argmin Problem**

Of course, not everything is smooth sailing. The challenge comes from **decision-making blocks** that represent optimization problems — i.e., when a part of your algorithm is an `argmin` or `argmax` over a smooth function.

The result of such a block is _implicitly_ defined with respect to inputs and hyperparameters, so we can’t just take its derivative in the usual way.

Let’s walk through a concrete example.

## **📐 Differentiating Through an Argmin with Constraints**

> _Differentiating Parameterized Argmin and Argmax Solutions from Optimization Problems with Linear Equality Constraints Having Parameter-Dependent Constant Terms_

We consider a constrained optimization problem where the decision is the solution to a minimization problem under parameterized linear equality constraints.

Let
* \(\mathcal\Theta\subseteq\mathbb R^P,\mathcal X\subseteq\mathbb R^N,\mathcal G\subseteq \mathbb R^M\times\mathbb R^N,\mathcal B\subseteq\mathbb R^M,\)
* \(f:\mathcal\Theta\times\mathcal X\to\mathbb R\) be twice continuously differentiable function,
* \(G\in\mathcal G\) be matrix such that \(\mathrm{rank}\left(G\right)=M,\)
* \(\boldsymbol b:\mathcal\Theta\to\mathcal B\) be vector-function with first derivatives.

Let

$$
\begin{cases}
  \boldsymbol{\widehat{x}}\left(\boldsymbol\theta\right)=\underset{\boldsymbol x}{\mathrm{argmin}}\left(f\left(\boldsymbol\theta,\boldsymbol x\right)\right),\\
  \mathrm{subject~to}:G\cdot\boldsymbol x\left(\boldsymbol\theta\right)+\boldsymbol{b}\left(\boldsymbol\theta\right)=\boldsymbol 0.
\end{cases}
$$

Then

$$\boldsymbol{\widehat{x}_\theta}\left(\boldsymbol\theta\right)=\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot\left(G^T\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot G^T\right)^{-1}\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot\boldsymbol{f_{x\theta}}\left(\boldsymbol\theta\right)-\boldsymbol{b_\theta}\left(\boldsymbol\theta\right)\right)-\boldsymbol{f_{x\theta}}\left(\boldsymbol\theta\right)\right),$$

where
   * \(f_{\boldsymbol{x\theta}}:\mathcal\Theta\to\mathbb R^N\times\mathbb R^P\) be matrix-function of all partial second-order derivatives,
   * \(f_{\boldsymbol{xx}}:\mathcal\Theta\to\mathbb R^N\times\mathbb R^N\) be **invertible** matrix-function of all partial second-order derivatives such that \(\\G\cdot\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot G^T\) is also invertible,
   * \(\boldsymbol{\widehat x_\theta}:\mathcal\Theta\to\mathbb R^N\times\mathbb R^P\) be matrix-function of all full first-order derivatives,
   * \(\boldsymbol{b_\theta}:\mathcal\Theta\to\mathbb R^M\times\mathbb R^P\) be matrix-function of all full first-order derivatives.

   All these conditions must be met at the \(\boldsymbol{\widehat{x}}\left(\boldsymbol\theta\right)\).
## Proof
Consider the Lagrangian for the optimization problem above:
$$\mathcal L\left(\boldsymbol\theta,\boldsymbol x,\boldsymbol\lambda\right)=f\left(\boldsymbol\theta,\boldsymbol x\right)+\lambda_\alpha\cdot\left(G_{\alpha\beta}\cdot x_\beta\left(\boldsymbol\theta\right)+b_\alpha\left(\boldsymbol\theta\right)\right).$$
> Here we use **Einstein notation** with greek letters indicating summation indices.

Assume that \(\left(\boldsymbol{x^*}\left(\boldsymbol\theta\right),\boldsymbol{\lambda^*}\left(\boldsymbol\theta\right)\right)\) is a saddle point of the Lagrangian. Then taking partial derivatives over \(\boldsymbol x\) and \(\boldsymbol\lambda\) in this point gives us the following:
$$\begin{cases}
  \partial^x_i\mathcal L\left(\boldsymbol\theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right),\boldsymbol{\lambda^*}\left(\boldsymbol\theta\right)\right)=\partial^x_if\left(\boldsymbol\theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right)\right)+\lambda^*_\alpha\left(\boldsymbol\theta\right)\cdot G_{\alpha i}=0,\\
  \partial^\lambda_j\mathcal L\left(\boldsymbol\Theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right),\boldsymbol{\lambda^*}\left(\boldsymbol\theta\right)\right)=G_{j\beta}\cdot x^*_\beta\left(\boldsymbol\theta\right)+b_j\left(\boldsymbol\theta\right)=0.
\end{cases}$$
Let's take the full derivative with respect to \(\theta_k\):
$$\begin{cases}
  \mathrm d^\theta_k\partial^x_i\mathcal L\left(\boldsymbol\theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right),\boldsymbol{\lambda^*}\left(\boldsymbol\theta\right)\right)=\partial^{x\theta}_{ik}f\left(\boldsymbol\theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right)\right)+\partial^{xx}_{i\gamma}f\left(\boldsymbol\theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right)\right)\cdot\mathrm d^\theta_kx^*_\gamma\left(\boldsymbol\theta\right)+\mathrm d^\theta_k\lambda^*_\alpha\left(\boldsymbol\theta\right)\cdot G_{\alpha i}=0,\\
  \mathrm d^\theta_k\partial^\lambda_j\mathcal L\left(\boldsymbol\Theta,\boldsymbol{x^*}\left(\boldsymbol\theta\right),\boldsymbol{\lambda^*}\left(\boldsymbol\theta\right)\right)=G_{j\beta}\cdot \mathrm d^\theta_kx^*_\beta\left(\boldsymbol\theta\right)+\mathrm d^\theta_kb_j\left(\boldsymbol\theta\right)=0.
\end{cases}$$

Let's rewrite the resulting equations collapsing them to the matrix notation:
$$\begin{cases}
  f_{\boldsymbol{x\theta}}+f_{\boldsymbol{xx}}\cdot\boldsymbol{x_\theta}+G^T\cdot\boldsymbol{\lambda_\theta}=\boldsymbol 0,\\
  G\cdot\boldsymbol{x_\theta}+\boldsymbol{b_\theta}=\boldsymbol 0.
\end{cases}$$From the first equation, we have:
$$\boldsymbol{x_\theta}=-\boldsymbol{f_{xx}^{-1}}\cdot\left(\boldsymbol{f_{x\theta}}+G^T\cdot\boldsymbol{\lambda_\theta}\right).$$
Substituting into the second equation gives:
$$
-G\cdot\boldsymbol{f^{-1}_{xx}}\cdot\boldsymbol{f_{x\theta}}
-G\cdot\boldsymbol{f^{-1}_{xx}}\cdot G^T\cdot\boldsymbol{\lambda_\theta}+\boldsymbol{b_\theta}=\boldsymbol 0.
$$
$$\boldsymbol{\lambda_\theta}=\left(G\cdot\boldsymbol{f^{-1}_{xx}}\cdot G^T\right)^{-1}\cdot\left(\boldsymbol{b_\theta}-G\cdot\boldsymbol{f^{-1}_{xx}}\cdot\boldsymbol{f_{x\theta}}\right).$$
Substituting it back into the first row gives us the following:
$$\boldsymbol{x_\theta}=\boldsymbol{f^{-1}_{xx}}\cdot\left(G^T\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\cdot G^T\right)^{-1}\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\cdot\boldsymbol{f_{x\theta}}-\boldsymbol{b_\theta}\right)-\boldsymbol{f_{x\theta}}\right).$$

The derivation for more complex cases of boundary conditions is given in the paper <a href={{< ref "./00003-deep-declarative-networks.md" >}} target="_blank">Deep Declarative Networks</a>.

## **✅ Key Takeaways**

- Differentiable algorithms let you optimize hyperparameters _analytically_, using gradients — no brute force.

- You can compute gradients even through `argmin` and `argmax` by applying implicit differentiation.

- This makes it possible to integrate classical algorithms (e.g., solvers, optimizers) into neural networks.

- The resulting system is fully end-to-end trainable — blending symbolic reasoning with deep learning.