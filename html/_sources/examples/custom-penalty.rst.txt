.. _examples-custom-penalty:

Custom penalty
--------------

The following example shows how to implement a custom penalty function in ``el0ps``, as explained in the :ref:`custom-penalty` section.
The latter also allows for just-in-time compilation as explained in the :ref:`custom-compilation` section.

.. code-block:: python
    
    import numpy as np
    from numba import float64
    from numpy.typing import NDArray
    from el0ps.penalty import BasePenalty, SymmetricPenalty
    from el0ps.compilation import CompilableClass

    class ReverseHuber(BasePenalty, SymmetricPenalty, CompilableClass):
        """
        Reverse Huber penalty function defined as

            h(x) = d * |x|              if |x| <= 1 
            h(x) = d * (x^2 + 1) / 2    otherwise

        for some d > 0.
        """

        def __init__(self, d: float) -> None:
            self.d = d
        

        # ----- Functions required when deriving from CompilableClass ----- #

        def get_spec(self) -> tuple:
            return (('d', float64),)

        def params_to_dict(self) -> dict:
            return {'d': self.d}


        # ----- Functions required when deriving from BaseDatafit ----- #

        def value(self, x: float, i: int) -> float:
            if np.abs(x) <= 1.:
                return self.d * np.abs(x)
            else:
                return self.d * (x ** 2 + 1.) / 2.
        
        def conjugate(self, x: float, i: int) -> float:
            return 0.5 * np.maximum(0., x ** 2 - self.d ** 2) / self.d

        def prox(self, x: float, i: int, eta: float) -> float:
            if np.abs(x) <= eta * self.d + 1.:
                return np.sign(x) * max(np.abs(x) - eta * self.d, 0.)
            else:
                return x / (1. + eta * self.d)
        
        def subdiff(self, x: float, i: int) -> NDArray:
            if x == 0.:
                return [-self.d, self.d]
            elif np.abs(x) <= 1.:
                return 2 * [self.d * np.sign(x)]
            else:
                return 2 * [self.d * x]

         def conjugate_subdiff(self, x: float, i: int) -> NDArray:
            if np.abs(x) < self.d:
                return [0., 0.]
            elif x == -self.d:
                return [-1., 0.]
            elif x == self.d:
                return [0., 1.]
            else:
                return [x / self.d, x / self.d]

        
        # ----- Functions required when deriving from SymmetricPenalty ----- #

        def param_slope(self, lmbd:float, i: int) -> float:
            return np.sqrt(self.d ** 2 + 2. * lmbd * self.d)
