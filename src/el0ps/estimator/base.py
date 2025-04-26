import numpy as np
from numpy.typing import NDArray
from sklearn.multiclass import check_classification_targets
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import validate_data

from el0ps.solver import BaseSolver, BnbSolver
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


class L0Estimator(LinearModel, BaseEstimator):
    r"""Scikit-learn-compatible `linear model <https://scikit-learn.org/stable/api/sklearn.linear_model.html>`_
    estimators based on L0-regularized problems.

    The estimator corresponds to a solution of the problem

    .. math::

        \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

    where :math:`f` is a datafit function,
    :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix,
    :math:`\lambda > 0` is a parameter, the L0-norm :math:`\|\cdot\|_0` counts
    the number of non-zero entries in its input, and :math:`h` is a penalty
    function.

    Parameters
    ----------
    datafit: BaseDatafit
        Datafit function.
    penalty: BasePenalty
        Penalty function.
    lmbd: float
        L0-norm weight.
    fit_intercept: bool, default=False
        Whether to fit an intercept term.
    solver: BaseSolver, default=BnbSolver()
        Solver for the estimator associated problem.
    """  # noqa: E501

    def __init__(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        lmbd: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ) -> None:

        if fit_intercept:
            raise NotImplementedError("Fit intercept not implemented yet")

        if not hasattr(datafit, "y"):
            raise ValueError(
                "The datafit object must have an attribute `y` to be used in "
                "an `L0Estimator`."
            )

        self.datafit = datafit
        self.penalty = penalty
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept
        self.solver = solver

        self.is_fitted_ = False
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

    def fit(self, X: NDArray, y: NDArray):

        # Sanity checks
        if isinstance(self, LinearClassifierMixin):
            check_classification_targets(y)
            enc = LabelEncoder()
            y = enc.fit_transform(y)
            y = 2.0 * y - 1.0
            if len(enc.classes_) > 2:
                raise ValueError("Only binary classification is supported.")
            self.classes_ = np.unique(y)
        else:
            self.classes_ = None
        check_X_params = dict(dtype=np.float64, order="F")
        check_y_params = dict(dtype=np.float64, order="F", ensure_2d=False)
        X, y = validate_data(
            self, X, y, validate_separately=(check_X_params, check_y_params)
        )
        assert y.ndim == 1

        # Initialize estimator.coef_
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])

        # Initialize estimator.intercept_
        if self.intercept_ is None:
            self.intercept_ = 0.0

        # Update datafit target vector
        y_old = np.copy(self.datafit.y)
        self.datafit.y = y

        # Solve the estimator optimization problem
        result = self.solver.solve(
            self.datafit, self.penalty, X, self.lmbd, x_init=self.coef_
        )

        # Reset the datafit target vector
        self.datafit.y = y_old

        # Recover the results
        self.is_fitted_ = True
        self.fit_result_ = result
        self.coef_ = np.copy(result.x)
        self.n_iter_ = result.iter_count
        # TODO: Recover intercept when supported
        self.intercept_ = 0.0

        return self

    def predict(self, X: NDArray) -> NDArray:
        if not self.is_fitted_:
            raise ValueError("The estimator is not fitted yet.")
        decision = X @ self.coef_ + self.intercept_
        if isinstance(self, LinearClassifierMixin):
            binary_preds = (decision > 0).astype(int)
            decision = self.classes_[binary_preds]
        return decision
