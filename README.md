# sympy_equation

[![Documentation Status](https://readthedocs.org/projects/sympy-equation/badge/?version=latest)](http://sympy-plot-backends.readthedocs.io/)
[![](https://img.shields.io/static/v1?label=Github%20Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/Davide-sd)

This package defines relations that all high school and college students would
recognize as mathematical equations, consisting of a left hand side (lhs) and
a right hand side (rhs) connected by the relation operator "=". This is
implemented by the ``Equation`` class, which also supports mathematical
operations applied to both sides simultaneously, just as students are taught
to do when  attempting to isolate (solve for) a variable. Thus the statement
``Equation/b`` yields a new equation ``Equation.lhs/b = Equation.rhs/b``.

The intent is to allow using the mathematical tools in SymPy to rearrange
equations and perform algebra in a stepwise fashion using as close to standard
mathematical notation as  possible. In this way more people can successfully
perform  algebraic rearrangements without stumbling over missed details such
as a negative sign.

A simple example as it would appear in a [Jupyter](https://jupyter.org)
notebook is shown immediately below:

![screenshot of simple example](https://raw.githubusercontent.com/Davide-sd/sympy-equation/master/assets/simple_example.png)

In IPython environments (IPython, Jupyter, Google  Colab, etc...) there is
also a shorthand syntax for entering equations provided through the IPython
preparser. An equation can be specified as ``eq1 =@ a/b = c/d``.


![screenshot of short syntax](https://raw.githubusercontent.com/Davide-sd/sympy-equation/master/assets/short_syntax.png)

If no Python name is specified for the equation (no ``eq_name`` to the left of ``=@``), the equation will still be defined, but will not be easily accessible
for further computation. The ``=@`` symbol combination was chosen to avoid
conflicts with reserved python  symbols while minimizing impacts on syntax
highlighting and autoformatting.

[More examples of the capabilities of Algebra with Sympy are
here](https://sympy-equation.readthedocs.io/en/latest/tutorial.html).


## Development and Support

If you feel like a feature could be implemented, open an issue or create a PR. Implementing new features and fixing bugs requires time and energy too. If you found this module useful and would like to show your appreciation, please consider sponsoring this project with either one of these options:

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/davide_sd)
or
[![](https://img.shields.io/static/v1?label=Github%20Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/Davide-sd)



## Installation


``sympy_equation`` can be installed with ``pip`` or ``conda``.

```
pip install sympy_equation
```

Or

```
conda install -c conda-forge sympy_equation
```


## Customizing the module


``equation_config`` is an object containing a few properties to customize
the behaviour of the module:

```py
from sympy_equation import equation_config
```

Arguably the most useful options are  :

* ``equation_config.integers_as_exact`` (default is False).
  When it's True and we are running in an IPython/Jupyter environment,
  it preparses the content of a code line in order to convert integer numbers
  to sympy's Integer. In doing so, we can write 2/3, which will be
  converted to Integer(2)/Integer(3), which then SymPy converts
  to Rational(2, 3). If False, no preparsing is done, and Python evaluates
  2/3 to 0.6666667, which will then be converted by SymPy to a Float.
* ``equation_config.show_label`` (default is False). When it's True, a label
  with the name of the equation in the python environment will be shown on
  the screen.

Check out the documentation to read more about these and other options.


## Differences between sympy_equation and algebra_with_sympy


* ``sympy_equation`` is a fork of [algebra_with_sympy](https://github.com/gutow/Algebra_with_Sympy), starting from the version 1.0.2.
* ``algebra_with_sympy`` installs a custom version of SymPy, which exposes
  the ``Equation`` class. The basic idea is to better integrate the ``Equation``
  class with other SymPy functionalities. The downside is that as new releases
  of SymPy are available, the users of ``algebra_with_sympy`` must wait for a
  new version of the package to be released as well.
  Differently, ``sympy_equation`` is an external package that only depends on
  SymPy: as new releases of SymPy are available, ``sympy_equation`` will work
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
