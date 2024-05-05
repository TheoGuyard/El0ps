import numpy as np
from abc import abstractmethod
from enum import Enum
from numpy.typing import ArrayLike
from sklearn.multiclass import check_classification_targets
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_consistent_length
from el0ps.solvers import BaseSolver, BnbSolver
from el0ps.datafits import BaseDatafit
from el0ps.penalties import BasePenalty


class TargetClass(Enum):
    BINARY = "binary"
    REGRESSION = "regression"


def _fit(
    estimator,
    datafit: BaseDatafit,
    penalty: BasePenalty,
    X: ArrayLike,
    lmbd: float,
    solver: BaseSolver,
):

    # Encode classification targets
    if estimator.target_class == TargetClass.BINARY:
        check_classification_targets(datafit.y)
        enc = LabelEncoder()
        datafit.y = enc.fit_transform(datafit.y)
        datafit.y = 2.0 * datafit.y - 1.0
        if enc.classes_ > 2:
            raise ValueError("Only binary classification is supported")

    # Validate X and y data types and shapes
    X = check_array(X, dtype=np.float64, order="F")
    datafit.y = check_array(
        datafit.y, dtype=np.float64, order="F", ensure_2d=False
    )
    check_consistent_length(X, datafit.y)
    assert datafit.y.ndim == 1

    # Initialize estimator.coef_ attribute
    if not hasattr(estimator, "coef_"):
        estimator.coef_ = None
    if estimator.coef_ is None:
        estimator.coef_ = np.zeros(X.shape[1])

    # Initialize estimator.intercept_ attribute
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

    .. math::
        min     f(X @ w) + lmbd ||w||_0 + g(w)

    where :math:`f` is a datafit term, :math:`g` is a penalty term and
    :math:`lmbd` is the L0-norm weight. The derived classes implement how
    the datafit and penalty terms are defined.
    """

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

    @property
    @abstractmethod
    def target_class(self) -> TargetClass:
        """Return the target class of the estimator."""
        ...
