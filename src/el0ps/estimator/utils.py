import numpy as np
from el0ps.penalty import (
    Bigm,
    BigmL1norm,
    BigmL2norm,
    L1L2norm,
    L1norm,
    L2norm,
)


def select_bigml1l2_penalty(
    alpha: float = 0.0, beta: float = 0.0, M: float = np.inf
):
    if alpha == 0.0 and beta == 0.0 and M != np.inf:
        penalty = Bigm(M)
    elif alpha != 0.0 and beta == 0.0 and M == np.inf:
        penalty = L1norm(alpha)
    elif alpha != 0.0 and beta == 0.0 and M != np.inf:
        penalty = BigmL1norm(M, alpha)
    elif alpha == 0.0 and beta != 0.0 and M == np.inf:
        penalty = L2norm(beta)
    elif alpha == 0.0 and beta != 0.0 and M != np.inf:
        penalty = BigmL2norm(M, beta)
    elif alpha != 0.0 and beta != 0.0 and M == np.inf:
        penalty = L1L2norm(alpha, beta)
    else:
        raise ValueError(
            "Setting `alpha=0`, `beta=0` and `M=np.inf` simultaneously is not "
            "allowed."
        )
    return penalty
