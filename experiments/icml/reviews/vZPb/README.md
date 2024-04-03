Comments for reviewer vZPb
--------------------------

We provide an additional experiment that assesses the statistical relevance of problem $(\mathcal{P})$ compared to other sparse regression methods as well as the numerical performance of the *simultaneous pruning* strategy proposed in our paper to address $(\mathcal{P})$ exactly via a BnB algorithm.
In short, it outlines working regimes where both:

- Solving $(\mathcal{P})$ is relevant from a statistical point of view.
- Our simultaneous pruning strategy provides significant gains in solving time.

### Data generation

**File `statistics_synthetic.pdf`**: 
We generate data $(\mathbf{x^{\dagger}},\mathbf{y},\mathbf{A})$ as specified in Section 4.1 with the parameters $k=10$, $m=150$, $n=200$, $\rho=0.9$ and $\tau=10$.
This data is then randomly split into a training set $(\mathbf{y}_{\mathrm{train}},\mathbf{A}_{\mathrm{train}})$ and a testing set $(\mathbf{y}_{\mathrm{test}},\mathbf{A}_{\mathrm{test}})$.
The training set contains $m_{\mathrm{train}}=100$ samples and the testing set contains $m_{\mathrm{test}}=50$ samples out of the $m = 150$ ones generated.
The results presented in our graphics are averaged over $100$ independent realizations of this data generation process.

**File `statistics_riboflavin.pdf`**: 
The $(\mathbf{y},\mathbf{A})$ corresponds to the dataset *riboflavin* used in the Section 4.2 of our paper.
This data is then randomly split into a training set $(\mathbf{y}_{\mathrm{train}},\mathbf{A}_{\mathrm{train}})$ and a testing set $(\mathbf{y}_{\mathrm{test}},\mathbf{A}_{\mathrm{test}})$.
The training set contains $m_{\mathrm{train}}=48$ samples and the testing set contains $m_{\mathrm{test}}=23$ samples out of the $m = 71$ ones generated.
The results presented in our graphics are averaged over $100$ independent realizations of this data generation process.

### Sparse regression methods

We consider different sparse regression methods.
Each of them construct a sequence of vectors $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ verifying $\|\mathbf{x}^{(s)}\|_0 \leq s$ from the training data $(\mathbf{y}_{\mathrm{train}},\mathbf{A}_{\mathrm{train}})$.
As the sparsity level $s$ increases, they are intended to better fit the underlying model.
First, we run the OMP algorithm which directly constructs the sequence $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ by selecting one new non-zero entry at a time in a greedy fashion.
We refer the reader to 

> Tropp, J. A. and Wright, S. J. (2010). Computational methods for sparse solution of linear inverse problems. Proceedings of the IEEE, 98(6), 948-958.

for a complete description of this algorithm.
Then, we solve the Lasso, Elastic-Net and L0-regularized problems respectively defined as

$$
\begin{align}
    \tag{$\mathcal{P}_{\ell_1}$}
    \textstyle \min_{\mathbf{x}} &\ \frac{1}{2} \|\mathbf{y}_{\mathrm{train}} - \mathbf{A}_{\mathrm{train}}\mathbf{x}\|_2^2 + \lambda_1 \|\mathbf{x}\|_1 \\
    \tag{$\mathcal{P}_{\ell_1\ell_2}$}
    \textstyle \min_{\mathbf{x}} &\ \frac{1}{2} \|\mathbf{y}_{\mathrm{train}} - \mathbf{A}_{\mathrm{train}}\mathbf{x}\|_2^2 + \lambda_1(\sigma \|\mathbf{x}\|_1 + \tfrac{1-\sigma}{2}\|\mathbf{x}\|_2^2) \\
    \tag{$\mathcal{P}_{\mathrm{\ell_0\ell_2}}$}
    \textstyle \min_{\mathbf{x}} & \ \frac{1}{2} \|\mathbf{y}_{\mathrm{train}} - \mathbf{A}_{\mathrm{train}}\mathbf{x}\|_2^2 + \lambda_0\|\mathbf{x}\|_0 + \tfrac{\gamma}{2}\|\mathbf{x}\|_2^2
\end{align}$$

for a decreasing sequence of $\lambda_1$ and $\lambda_0$ values.
Each method then outputs a pool of solutions $\{\mathbf{x}^{(p)}\}_{p \in P}$ with varying sparsity levels.
In problem $(\mathcal{P}_{\ell_1\ell_2})$, we calibrate $\sigma$ using `scikit-learn` and in problem $(\mathcal{P}_{\ell_0\ell_2})$, we calibrate $\gamma$ as specified in our Appendix B.3.
For each of these three problems, we construct the sequence $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ from the pool of solutions $\{\mathbf{x}^{(p)}\}_{p \in P}$ obtained where 

$$\textstyle\mathbf{x}^{(s)} = \arg\min_{\|\mathbf{x}^{(p)}\|_0 \leq s, \ p \in P} \tfrac{1}{m_{\mathrm{train}}}\|\mathbf{y}_{\mathrm{train}} - \mathbf{A}_{\mathrm{train}}\mathbf{x}^{(p)}\|_2^2$$

for all $s \in \{0,\ldots,S\}$.
Stated otherwise, we select the solution in the pool with the best least training error among those verifying the sparsity criterion.

### Statistical performance

We plot different performance metrics for the sequence of vectors $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ obtained with the OMP algorithm, the Lasso, the Elastic-Net and the L0-regularized problems.
First, when the ground truth vector $\mathbf{x^{\dagger}}$ is available, we represent the F1 support recovery score of the vectors $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ with respect to $\mathbf{x^{\dagger}}$.
The closer the F1-score to $1$, the better.
We refer to Section 5.1 in 

> Dedieu, A., Hazimeh, H., & Mazumder, R. (2021). Learning sparse classifiers: Continuous and mixed integer optimization perspectives. Journal of Machine Learning Research, 22(135), 1-47.

for a precise definition of this metric.
Moreover, we represent the training error

$$\textstyle\frac{1}{m_{\mathrm{train}}} \|\mathbf{y}_{\mathrm{train}} - \mathbf{A}_{\mathrm{train}}\mathbf{x}^{(s)}\|_2^2$$

and the testing error

$$\textstyle\frac{1}{m_{\mathrm{test}}} \|\mathbf{y}_{\mathrm{test}} - \mathbf{A}_{\mathrm{test}}\mathbf{x}^{(s)}\|_2^2$$

of the sequences $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ generated by the different sparse regression method.
Finally, we represent the solving time to construct each vector of the sequence $\{\mathbf{x}^{(s)}\}_{s=0}^{S}$ obtained with the different sparse regression methods considered.
We implement our own OMP algorithm and use the `scikit-learn` package to solve the Lasso and Elastic-Net problems.
Besides, problem $(\mathcal{P}_{\ell_0\ell_2})$ is addressed using two of the solvers considered in our paper:
- `l0bnb` which is considered as the state-of-the-art BnB algorithm to solve $(\mathcal{P}_{\ell_0\ell_2})$ and available [here](https://github.com/alisaab/l0bnb).
- `proposed` which corresponds to the BnB algorithm presented in Section 2 of our paper, enhanced with the *simultaneous pruning* strategy proposed in Section 3.

### Observations

We remark that problem $(\mathcal{P}_{\ell_0\ell_2})$ leads to better support recovery and testing performances than the other sparse regression methods.
With synthetic data where $\mathbf{x^{\dagger}}$ is available, this is especially in working regimes of interest where $\|\mathbf{x}^{(s)}\|_0 \simeq \|\mathbf{x^{\dagger}}\|_0 = k$.
Although solving $(\mathcal{P}_{\ell_0\ell_2})$ is computationally more expensive than the other sparse regression methods, we note a significant decrease in the solution time using our *simultaneous pruning* strategy compared to other methods addressing $(\mathcal{P}_{\ell_0\ell_2})$ exactly.