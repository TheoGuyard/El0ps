.. _news:

==========
What's new
==========

Version 0.0.4
-------------

* Add ``BigmL1L2norm`` penalty
* Update ``MipSolver`` interface
* Fix KL penalty definition
* Fix bugs in numba compilation

Version 0.0.3
-------------

* Revised documentation
* Revised workslow for penalty definitions, with only the ``param_slope_pos`` and ``param_slope_neg`` methods needed
* Revised MIP model in ``MipSolver``

Version 0.0.2
-------------

* Parallelize the BnB code using ``pybnb``
* Compiled ``BoundingSolver`` using ``numba``
* Add ``MipSolver`` and ``OaSolver`` solvers
* Make penalties unsymmetrical by default and add ``SymmetricPenalty`` class
* Remove experiments from this repo

Version 0.0.1
-------------

* Initial release of the package.
* Features currently implemented are suggested to modification in the future.