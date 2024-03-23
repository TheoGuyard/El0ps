Comments for reviewer 34MP
--------------------------

The file `sensibility_reltol.pdf` presents a similar experiment to that in Figure 4 of our paper, but where we vary the accuracy targeted while solving the relaxation across the BnB tree.
Instances are generated as in Section 4.1 with the parameters $k=5$, $m=500$,$n=1000$, $\rho=0.9$ and $\tau=10$.
We solve the problem for different values of $\epsilon$, which represents the relative optimality tolerance while solving the relaxations $(\mathcal{R}^{\nu})$ during the BnB algorithm.
The figure represents the gains in terms of solving time obtained by our simultaneous pruning strategy as compared to the standard one.