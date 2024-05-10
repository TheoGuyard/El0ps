import numpy as np
from abc import abstractmethod
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, _fit_context
from sklearn.multiclass import check_classification_targets
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import LabelEncoder
from el0ps.solvers import BaseSolver, BnbSolver
from el0ps.datafits import BaseDatafit
from el0ps.penalties import (
    BasePenalty,
    Bigm,
    BigmL1norm,
    BigmL2norm,
    BigmL1L2norm,
    L1L2norm,
    L1norm,
    L2norm,
)


def _fit(
    estimator,
    datafit: BaseDatafit,
    penalty: BasePenalty,
    X: ArrayLike,
    lmbd: float,
    solver: BaseSolver,
    skip_validate_data: bool = False,
):
    if not skip_validate_data:
        if isinstance(estimator, ClassifierMixin):
            check_classification_targets(datafit.y)
            enc = LabelEncoder()
            datafit.y = enc.fit_transform(datafit.y)
            datafit.y = 2.0 * datafit.y - 1.0
            if len(enc.classes_) > 2:
                raise ValueError("Only binary classification is supported")
            estimator.classes_ = np.unique(datafit.y)
        else:
            estimator.classes_ = None
        check_X_params = dict(dtype=np.float64, order="F")
        check_y_params = dict(dtype=np.float64, order="F", ensure_2d=False)
        X, datafit.y = estimator._validate_data(
            X, datafit.y, validate_separately=(check_X_params, check_y_params)
        )
        assert datafit.y.ndim == 1

    # Initialize estimator.coef_
    if not hasattr(estimator, "coef_"):
        estimator.coef_ = None
    if estimator.coef_ is None:
        estimator.coef_ = np.zeros(X.shape[1])

    # Initialize estimator.intercept_
    if not hasattr(estimator, "intercept_"):
        estimator.intercept_ = None
    if estimator.intercept_ is None:
        estimator.intercept_ = 0.0

    # Solve the estimator optimization problem
    result = solver.solve(datafit, penalty, X, lmbd, x_init=estimator.coef_)

    # Recover the results
    estimator.is_fitted_ = True
    estimator.fit_result_ = result
    estimator.coef_ = np.copy(result.x)
    estimator.n_iter_ = result.iter_count
    # TODO: Recover intercept when supported

    return estimator


class BaseL0Estimator(LinearModel):
    """Base class for L0-norm estimators.

    The optimization problem solved is

    .. math:: \min f(Xw) + \lambda \|w\|_0 + h(w)

    where :math:`f` is a datafit term, :math:`h` is a penalty term and
    :math:`lmbd` is the L0-norm weight. The derived classes implement how
    the datafit and penalty terms are defined.
    """  # noqa: W605

    def __init__(
        self,
        lmbd: float = 1.0,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ) -> None:

        if fit_intercept:
            raise NotImplementedError("Fit intercept not implemented yet")
        if lmbd <= 0.0:
            raise ValueError("Parameter `lmbd` must be non-negative")

        self.lmbd = lmbd
        self.fit_intercept = fit_intercept
        self.solver = solver

        self.is_fitted_ = False
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

    @_fit_context(prefer_skip_nested_validation=True)
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the estimator.

        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Data matrix.

        y : ArrayLike, shape (n_samples,)
            Target vector.
        """
        ...


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
    elif alpha != 0.0 and beta != 0.0 and M != np.inf:
        penalty = BigmL1L2norm(M, alpha, beta)
    else:
        raise ValueError(
            "Setting `alpha=0`, `beta=0` and `M=np.inf` simulteanously is not "
            "allowed."
        )
    return penalty
