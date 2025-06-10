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
slug: diff-decision-makers
---

In many optimisation tasks — including hyperparameter search, meta-learning, and algorithmic tuning — black-box heuristics are still the default. When every component of a pipeline is differentiable, gradients provide a direct, low-variance learning signal that turns these outer loops into first-class optimisation problems. A **differentiable algorithm** is a procedure in which each operation — including embedded optimisation sub-problems — admits derivatives with respect to its inputs and hyperparameters. This perspective converts formerly discrete search into continuous, gradient-driven updates that integrate naturally with modern autodiff libraries such as `PyTorch`, `JAX`, etc.

## Motivation: Where Differentiable Algorithms Fit in Optimization

<!-- streamlined wording to better match an ML-savvy audience -->

### What do we call a *Decision Maker*?

In this article a **decision maker** is any algorithmic routine whose primary output is a concrete action or discrete choice — approve a loan, hedge a portfolio, pick a route, allocate bandwidth. Formally we view it as a mapping
\[\mathrm{DM}:\; (\text{data},\; \text{hyper-parameters})\;\mapsto\; \text{decision}.\]
Making the routine differentiable lets us train those hyper-parameters (or even the structure of the routine itself) with gradient methods rather than grid search or handcrafted rules. The examples below — credit approval and risk-aware hedging — illustrate, respectively, an **imperative** and a **mixed imperative / declarative** decision maker.

## Key Benefits

In practice, differentiable algorithms unlock gradient-based hyperparameter tuning — often far cheaper and more precise than exhaustive search:

Here’s a conceptual diagram illustrating how such an algorithm flows:

![Differentiable Algorithm Graph](../../blog/dag.svg)
In this diagram:

Each tensor label encodes three things:
- The input dimension (if any),
- The output dimension, and
- The activation function (if any) applied to that output.

An arrow from tensor X to tensor Y indicates that at least part of X’s output is consumed as (at least part of) Y’s input.

Dashed arrows show the backward pass — how gradients propagate during hyperparameter tuning (see the “Gradient-Based Hyperparameter Tuning” section below).

We distinguish two classes of tensors in the diagram:

- **Imperative tensors** represent blocks defined by _explicit_ smooth functions.

- **Declarative tensors** represent _implicit_ functions — outputs of optimization problems over smooth objectives.

> This terminology is based on the <a href={{< ref "./00003-deep-declarative-networks.md" >}} target="_blank">Deep Declarative Networks</a> framework we’ve discussed before.

## 🚀 Gradient-Based Hyperparameter Tuning

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
## 🧠 Integration with Neural Networks

What if we want to _predict_ good hyperparameters on the fly?

Because differentiable algorithms are, well, differentiable, we can integrate them directly into a machine learning pipeline — **including neural networks** — without:
- Designing special loss functions for training,

- Losing accuracy through output calibration tricks.

Here’s how such an integration might look in practice:

![Differentiable Algorithm — Neural Network Training](../../blog/diffalg-neural.svg)

## 🛠️ Practical Walk-Throughs

Below are two end-to-end mini-projects that show how differentiable algorithms feel in practice. Both can be run in a modern autodiff framework (PyTorch, JAX, TensorFlow) with fewer than 50 lines of code.

### 1️⃣ Purely Imperative: Credit-Approval Classifier

Consider a bank that wants to automate the decision “Should we approve this consumer loan?”. The pipeline can be built **entirely out of smooth, explicit functions**:

1. **Feature engineering** – normalise income, debt-to-income ratio, length of credit history, etc. using differentiable transforms such as \(z\)-scores.
2. **Neural scoring model** – a two-layer perceptron outputs the approval probability
   \[p=\sigma\bigl(W_2\,\mathrm{ReLU}(W_1 x + b_1)+b_2\bigr).\]
3. **Decision rule** – \(\text{approve} \Leftrightarrow p>0.5\).

Because every block is explicit, the whole DAG is coloured **imperative** (yellow in the original figure) and ordinary back-prop works out of the box.

```python
# toy implementation – PyTorch / JAX pseudo-code

def credit_score(x, theta):
    z1 = jax.nn.relu(x @ theta["W1"] + theta["b1"])
    p  = jax.nn.sigmoid(z1 @ theta["W2"] + theta["b2"])
    return p  # probability of approval
```

Training the parameters \(\theta\) is a vanilla supervised-learning loop – no surprises, but the example firmly grounds the idea of an **imperative differentiable algorithm**.

### 2️⃣ Imperative + Declarative: Risk-Aware Portfolio Hedging

Financial institutions rarely optimise just one metric. A trading desk, for example, wants to **maximise expected PnL _and_ keep correlated risk under control**. That second requirement introduces a **declarative node** – an inner optimisation problem.

In this setup we

*let \(\theta=(\phi,\lambda)\) — where \(\phi\) are the neural-network parameters of the utility surrogate and \(\lambda\) is the risk-aversion scalar that penalises covariance risk;*

* predict the instantaneous utility coefficients \(\hat u_\phi(x)\) with a small network (imperative);
* choose hedge weights \(w\) by solving
$$
\begin{cases}
  w^*(\theta,x)=\underset{w}{\argmax}\left(\hat u_\phi(x, w) - \lambda\,w^\top \Sigma(x) w\right),\\
  \text{subject to}: \mathbf 1^\top w = 0,
\end{cases}
$$
  where \(\Sigma(x)\) is the factor-covariance matrix and \(\lambda\) is a risk-aversion hyper-parameter.

This quadratic programme has a closed-form solution \(w^* = A^{-1}b\) with
- \(A = 2\lambda\Sigma\), the risk-weighted covariance matrix, and
- \(b = \hat u_{\phi}(x)\), the utility vector predicted by the neural network that sums to 1.

Calling `jax.numpy.linalg.solve` (or the equivalent in PyTorch) gives us both the answer **and** its gradients via implicit differentiation.

> **Note:** A closed-form solution is not required for a declarative node; any differentiable numerical solver (e.g., an iterative QP solver) will work as long as gradients can be obtained.

```python
# pseudo-code – JAX flavour

def hedge_weights(x, theta):
    b      = pnl_net(x, theta["phi"])      # imperative
    Sigma  = cov_estimator(x)              # imperative
    A      = 2 * theta["lambda"] * Sigma   # risk-weighted covariance
    w_star = jax.numpy.linalg.solve(A, b)  # declarative argmax
    return w_star
```

In the overall computational graph we now have

* yellow imperative nodes (`pnl_net`, `cov_estimator`), and
* a purple declarative node (`jax.numpy.linalg.solve`).

Because the declarative node is differentiable, we can jointly learn \(\phi\) _and_ tune \(\lambda\) on historical data by gradient descent instead of costly grid searches.

These two walkthroughs should make the mental model concrete before we dive into the mathematical details of differentiating through the inner optimisation.

## ⚠️ When Differentiation Gets Hard: The Argmin Problem

Of course, not everything is smooth sailing. The challenge comes from **decision-making blocks** that represent optimization problems — i.e., when a part of your algorithm is an `argmin` or `argmax` over a smooth function.

The result of such a block is _implicitly_ defined with respect to inputs and hyperparameters, so we can’t just take its derivative in the usual way.

Let’s walk through a concrete example.

### 📐 Differentiating Through an Argmin with Constraints

> _Differentiating Parameterized Argmin and Argmax Solutions from Optimization Problems with Linear Equality Constraints Having Parameter-Dependent Constant Terms_

We consider a constrained optimization problem where the decision is the solution to a minimization problem under parameterized linear equality constraints.

_My thanks go to Semyon Semenov for reviewing the mathematical parts of this work and suggesting improvements._

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

$$\boldsymbol{\widehat{x}_\theta}\left(\boldsymbol\theta\right)=\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot\left(G^\top\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot G^\top\right)^{-1}\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot\boldsymbol{f_{x\theta}}\left(\boldsymbol\theta\right)-\boldsymbol{b_\theta}\left(\boldsymbol\theta\right)\right)-\boldsymbol{f_{x\theta}}\left(\boldsymbol\theta\right)\right),$$

where
   * \(f_{\boldsymbol{x\theta}}:\mathcal\Theta\to\mathbb R^N\times\mathbb R^P\) be matrix-function of all partial second-order derivatives,
   * \(f_{\boldsymbol{xx}}:\mathcal\Theta\to\mathbb R^N\times\mathbb R^N\) be **invertible** matrix-function of all partial second-order derivatives such that \(\\G\cdot\boldsymbol{f^{-1}_{xx}}\left(\boldsymbol\theta\right)\cdot G^\top\) is also invertible,
   * \(\boldsymbol{\widehat x_\theta}:\mathcal\Theta\to\mathbb R^N\times\mathbb R^P\) be matrix-function of all full first-order derivatives,
   * \(\boldsymbol{b_\theta}:\mathcal\Theta\to\mathbb R^M\times\mathbb R^P\) be matrix-function of all full first-order derivatives.

   All these conditions must be met at the \(\boldsymbol{\widehat{x}}\left(\boldsymbol\theta\right)\).

### Derivation road-map

Before diving into the algebra, here is the flow of the argument in one glance:

1. **Lagrangian set-up** – introduce multipliers \(\lambda\) and write the Lagrangian \(\mathcal L(\theta,x,\lambda)\).
2. **KKT conditions** – impose stationarity in \(x\) and feasibility of the linear constraints to obtain a block system in \((x^*,\lambda^*)\).
3. **Differentiate KKT system** – take total derivatives w.r.t. \(\theta\) to get a linear system in the unknown Jacobians \(x_{\theta}\) and \(\lambda_{\theta}\).
4. **Block elimination** – solve the linear system explicitly; first eliminate \(\lambda_{\theta}\), then back-substitute for \(x_{\theta}\).
5. **Compact form** – rearrange the result to the closed-form expression shown below.

Readers who only need the final formula can skip to Step 5; the remainder of this section walks through Steps 1-4 in detail.

### Proof
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
  f_{\boldsymbol{x\theta}}+f_{\boldsymbol{xx}}\cdot\boldsymbol{x_\theta}+G^\top\cdot\boldsymbol{\lambda_\theta}=\boldsymbol 0,\\
  G\cdot\boldsymbol{x_\theta}+\boldsymbol{b_\theta}=\boldsymbol 0.
\end{cases}$$From the first equation, we have:
$$\boldsymbol{x_\theta}=-\boldsymbol{f_{xx}^{-1}}\cdot\left(\boldsymbol{f_{x\theta}}+G^\top\cdot\boldsymbol{\lambda_\theta}\right).$$
Substituting into the second equation gives:
$$
-G\cdot\boldsymbol{f^{-1}_{xx}}\cdot\boldsymbol{f_{x\theta}}
-G\cdot\boldsymbol{f^{-1}_{xx}}\cdot G^\top\cdot\boldsymbol{\lambda_\theta}+\boldsymbol{b_\theta}=\boldsymbol 0.
$$
$$\boldsymbol{\lambda_\theta}=\left(G\cdot\boldsymbol{f^{-1}_{xx}}\cdot G^\top\right)^{-1}\cdot\left(\boldsymbol{b_\theta}-G\cdot\boldsymbol{f^{-1}_{xx}}\cdot\boldsymbol{f_{x\theta}}\right).$$
Substituting it back into the first row gives us the following:
$$\boldsymbol{x_\theta}=\boldsymbol{f^{-1}_{xx}}\cdot\left(G^\top\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\cdot G^\top\right)^{-1}\cdot\left(G\cdot\boldsymbol{f^{-1}_{xx}}\cdot\boldsymbol{f_{x\theta}}-\boldsymbol{b_\theta}\right)-\boldsymbol{f_{x\theta}}\right).$$

The derivation for more complex cases of boundary conditions is given in the paper <a href={{< ref "./00003-deep-declarative-networks.md" >}} target="_blank">Deep Declarative Networks</a>.

## ✅ Key Takeaways

- Differentiable algorithms let you optimize hyperparameters _analytically_, using gradients — no brute force.

- You can compute gradients even through `argmin` and `argmax` by applying implicit differentiation.

- This makes it possible to integrate classical algorithms (e.g., solvers, optimizers) into neural networks.

- The resulting system is fully end-to-end trainable — blending symbolic reasoning with deep learning.