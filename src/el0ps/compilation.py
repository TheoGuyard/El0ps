"""Compilation utilities."""

from abc import abstractmethod
from functools import lru_cache
from numba.experimental import jitclass


class CompilableClass:
    """Template for classes that can be compiled with Numba."""

    @abstractmethod
    def get_spec(self) -> tuple:
        """Specify the numba types of the class attributes.

        Returns
        -------
        spec: Tuple of (attr_name, dtype)
            Specs to be passed to Numba jitclass to compile the class.
        """
        ...

    @abstractmethod
    def params_to_dict(self) -> dict:
        """Returns the parameters name and value used to initialize the class
        instance.

        Returns
        -------
        dict_of_params: dict
            The parameters name and value used to initialize the class
            instance.
        """
        ...


@lru_cache()
def compiled_clone(instance: CompilableClass):
    r"""Compile a class instance to a
    `jitclass <https://numba.readthedocs.io/en/stable/user/jitclass.html>`_.
    This function is inspired from a similar one in the
    `skglm <https://github.com/scikit-learn-contrib/skglm/tree/main>`_ package.

    Parameters
    ----------
    instance: object
        Instance to compile.

    Returns
    -------
    compiled_instance: jitclass
        Compiled instance.
    """
    cls = instance.__class__
    spec = instance.get_spec()
    params = instance.params_to_dict()
    return jitclass(spec)(cls)(**params)
