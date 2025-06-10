---
title: From Discrete to Smooth
description: A Practical Guide to Optimization Problems
type: article
date: 2025-05-16
categories:
  - Optimization
tags:
  - Smooth Optimization
  - Convex Optimization
  - Lagrange Multipliers
  - Karush–Kuhn–Tucker Conditions
slug: from-discrete-to-smooth
---

Many real-world tasks can be posed as optimization problems — from resource allocation to machine learning, signal processing, or operations research. The general form is:

$$
\min_{x} \; f(x) \quad \text{subject to} \quad x \in \mathcal{C}
$$

where \( f(x) \) is the objective function and \( \mathcal{C} \) is the feasible region.

In practice, especially in fields like high-frequency trading and quantitative finance, it's common to see practitioners get stuck in discrete formulations of optimization problems. While discrete methods have their place, there are many cases where insisting on a discrete approach leads to significant inefficiencies — both computationally and financially. I've often observed situations where a problem that naturally lends itself to smooth optimization techniques is instead attacked with brute-force grid search or other combinatorial methods.

This article is a response to that pattern — it’s a guide to recognizing when and how a discrete problem can (and should) be relaxed into a smooth one, saving time, money, and complexity. It outlines a pragmatic strategy: **reduce complexity step by step, ideally toward smooth and convex formulations**.

---

## 1. Discrete vs. Smooth Optimization

### 🚫 Discrete Optimization

When some variables must be integers (e.g., \( x \in \{0,1\}^n \)), the problem becomes discrete or combinatorial. These problems are notoriously hard:

- They often lack general theory or closed-form solutions.
- Solvers are slow and may rely on heuristics or exhaustive search.
- Even toy examples can be computationally expensive.

> ⚠️ **Avoid discrete optimization if possible**. Instead, try **relaxing the problem** into a continuous one.

### ✅ Smooth Optimization

If variables are allowed to take real values (e.g., \( x \in [0,1]^n \)), then **gradient-based methods** and other powerful tools become available:

- Easier to analyze mathematically.
- Amenable to standard solvers (e.g., CVXPY, SciPy, PyTorch optimizers).
- Scalable to large problems.

---

## 2. A Practical Strategy: Reduce to Smooth Optimization

Whenever you're facing a hard problem — especially one involving integer variables or non-differentiable objectives — follow this pragmatic funnel:

---

### ✅ Step 1: Try Convex Optimization

If you can reformulate the problem so that:

- The objective \( f(x) \) is **convex** (i.e., Hessian \( \nabla^2 f(x) \succeq 0 \)), and
- The feasible set \( \mathcal{C} \) is **convex** (e.g., box constraints, linear inequalities),

then **convex optimization solvers** will likely give reliable global minima efficiently.

> 🧠 You can verify convexity by checking that the Hessian is positive semi-definite, or by using composition rules (see <a href="https://web.stanford.edu/~boyd/cvxbook/" target="_blank" rel="noreferrer external">Boyd & Vandenberghe</a>).

If convexity cannot be guaranteed — say, due to regularization or relaxations — the solver may fail or return local minima.

---

### 🧮 Step 2: Use the Lagrangian Method with Quadratic Forms

If convexity fails, keep things **quadratic and smooth**. That is:

- Use only terms like \( x^2 \), \( x_i x_j \), and linear terms in the objective.
- Stick to **linear** equality and inequality constraints.
- Do not use modulus in linear terms.

Then solve using the **method of Lagrange multipliers**. Construct:

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda^\top (Ax - b)
$$

Solving the KKT (Karush-Kuhn-Tucker) conditions gives:

- A **system of linear equations** if everything is quadratic/linear.
- Either an exact solution or a singular matrix (which signals a poorly posed problem).

> ⚠️ This works only if the constraint matrix has full rank and satisfies regularity conditions (e.g., constraint qualification). Otherwise, numerical instabilities may arise.

---

### 🔧 Step 3: Use Gradient-Based Optimization

If all else fails and the objective is **non-convex but differentiable**, use **gradient descent** or variants like **Adam**. Key tools:

- **Variable substitution** for handling linear equality constraints:  
  If \( x_1 + x_2 + x_3 = 1 \), express \( x_3 = 1 - x_1 - x_2 \) and optimize over the remaining variables.

- **Projection** for handling box constraints:  
  After each gradient step, clip \( x_i \leftarrow \min(\max(x_i, 0), 1) \) to stay in \([0, 1]^n\).

- **Regularization** to encourage binary solutions in relaxed problems:  
  Add a term like \( \lambda \sum_i x_i(1 - x_i) \), which is minimized at \( x_i \in \{0,1\} \).

---

## 3. Illustrative Example: Relaxing a Binary Problem

Suppose we are tasked with choosing a subset of assets to include in a portfolio. Each asset either is included (1) or not (0). The goal is to minimize risk while satisfying certain constraints on return, sector exposure, or liquidity. Such selection naturally leads to a binary optimization problem. Let’s formalize a simplified version of this situation:

Given:
- A cost vector \( c \in \mathbb{R}^n \)
- A constraint matrix \( A \in \mathbb{R}^{m \times n} \)
- A right-hand-side vector \( b \in \mathbb{R}^m \)

The problem is:

$$
\min_{x \in \{0,1\}^n} \; c^\top x \quad \text{s.t.} \quad Ax = b
$$

This is discrete and hard. Let’s relax it:

1. Allow \( x \in [0,1]^n \) instead of binary.
2. Add a regularization term that encourages binary-like solutions.

New problem:

$$
\min_{x \in [0,1]^n} \; c^\top x + \lambda \sum_{i=1}^n x_i(1 - x_i) \quad \text{s.t.} \quad Ax = b
$$

Here, \(\lambda>0\) is a regularization parameter — not a Lagrange multiplier — that balances the original objective and the penalty term encouraging binary solutions.

This objective is differentiable but not convex due to the \( x_i(1 - x_i) \) terms. You can now:

- Eliminate as many variables as the number of linearly independent equality constraints in \(Ax=b\). This reduces a constrained optimization problem to a lower-dimensional unconstrained problem.
- Apply projected gradient descent or Adam optimizer.
- Use clipping to enforce \( x_i \in [0,1] \).

---

## 4. Summary: A Decision Table for Optimization

| Problem Type         | Feasibility   | Theory       | Recommended Method        |
|----------------------|---------------|--------------|---------------------------|
| Discrete / Integer   | Very difficult| Weak         | Avoid or relax            |
| Smooth, non-convex   | Moderate      | Limited      | Gradient descent (Adam)   |
| Quadratic + linear   | Solvable      | Strong       | Lagrange multipliers      |
| Smooth & convex      | Easy          | Strong       | Convex solvers / CVXPY    |

> ✅ **Golden Rule**: When in doubt, reduce your problem to something **smooth, quadratic, and convex** — in that order.

---

## References

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. <a href="https://web.stanford.edu/~boyd/cvxbook/" target="_blank" rel="noreferrer external">Fee online</a>