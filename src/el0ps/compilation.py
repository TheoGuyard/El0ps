"""Compilation utilities."""

from abc import abstractmethod
from functools import lru_cache
from numba.experimental import jitclass


class CompilableClass:

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
        """Get the parameters to initialize an instance of the class.

        Returns
        -------
        dict_of_params: dict
            The parameters to instantiate an object of the class.
        """
        ...


@lru_cache()
def compiled_clone(instance: CompilableClass):
    """Compile a class instance to a ``jitclass``. Credits: ``skglm`` package.

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
