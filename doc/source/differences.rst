Differences between sympy_equation and algebra_with_sympy
---------------------------------------------------------

``sympy_equation`` is a fork of `algebra_with_sympy <https://github.com/gutow/Algebra_with_Sympy>`_,
starting from the version 1.0.2. While there are a lot of similarities,
these packages are not interchangeable. The main differences are:

* ``algebra_with_sympy`` installs a custom version of SymPy, which exposes
  the ``Equation`` class. The basic idea is to better integrate the ``Equation``
  class with other SymPy functionalities. The downside is that as new releases
  of SymPy are available, the users of ``algebra_with_sympy`` must wait for a
  new version of the package to be released as well.
  Differently, ``sympy_equation`` is an external package that only depends on
  SymPy. The ``Equation`` class is implemente into ``sympy_equation``.
  As new releases of SymPy are available, ``sympy_equation`` will work
  with them right away. The downside is that it might not be as integrated with
  SymPy's functionalities as one would like it to be.
* ``algebra_with_sympy`` exposes the ``algwsym_config`` object to customize
  the behaviour of the module. Similarly, ``sympy_equation`` exposes the
  ``equation_config``. The available options are very similar, but their
  default values are often different.
* ``algebra_with_sympy`` overwrites the default behaviour of SymPy's
  ``solve()`` and ``solveset()`` in order for them to return sets of solutions.
  This can be annoying if you are used to the SymPy's way of doing things.
  Differently, ``sympy_equation`` doesn't change the behaviour of these
  functions, rather it extends it in order to deal with the ``Equation`` class.
