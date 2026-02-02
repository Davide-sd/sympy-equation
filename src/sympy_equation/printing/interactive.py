from sympy_equation.printing.extended_latex import ExtendedLatexPrinter
from sympy_equation.doc_utils import add_parameters_to_docstring
from sympy.printing.defaults import Printable
from sympy.printing.printer import print_function


def _can_print(o):
    """Return True if type o can be printed with one of the SymPy printers.

    If o is a container type, this is True if and only if every element of
    o can be printed in this way.
    """

    try:
        # If you're adding another type, make sure you add it to printable_types
        # later in this file as well

        builtin_types = (list, tuple, set, frozenset)
        if isinstance(o, builtin_types):
            # If the object is a custom subclass with a custom str or
            # repr, use that instead.
            if (type(o).__str__ not in (i.__str__ for i in builtin_types) or
                type(o).__repr__ not in (i.__repr__ for i in builtin_types)):
                return False
            return all(_can_print(i) for i in o)
        elif isinstance(o, dict):
            return all(_can_print(i) and _can_print(o[i]) for i in o)
        elif isinstance(o, bool):
            return False
        elif isinstance(o, Printable):
            # types known to SymPy
            return True
        elif hasattr(o, "_latex"):
            # types which add support themselves
            return True
        return False
    except RuntimeError:
        return False
        # This is in case maximum recursion depth is reached.
        # Since RecursionError is for versions of Python 3.5+
            # so this is to guard against RecursionError for older versions.


@add_parameters_to_docstring(ExtendedLatexPrinter)
@print_function(ExtendedLatexPrinter)
def init_latex_printing(**settings):
    r"""
    Initialize an ``ExtendedLatexPrinter`` on this interactive shell.

    Returns
    =======
    latex_printer : ExtendedLatexPrinter

    Examples
    ========

    >>> from sympy import *
    >>> from sympy_equation import init_latex_printing
    >>> x, y, z, t = symbols("x:z t")
    >>> f = Function("f")(x, y, z)
    >>> g = Function("g")(t)
    >>> expr = f.diff(x, 2) + g.diff(t)
    >>> printer = init_latex_printing()

    Use the ``doprint`` method in order to generate Latex code of a symbolic
    expressions:

    >>> print(printer.doprint(expr))
    \frac{\partial^{2}}{\partial x^{2}} f{\left(x,y,z \right)} + \frac{d}{d t} g{\left(t \right)}

    Adding a rule to print derivatives of ``f`` using shorter partial notation:

    >>> printer.add_rule(f, derivative="d-notation")
    >>> print(printer.doprint(expr))
    \partial_{xx}f{\left(x,y,z \right)} + \frac{d}{d t} g{\left(t \right)}

    Adding a rule to print derivates of ``g`` using dot notation, while hiding
    the arguments of ``g``:

    >>> printer.add_rule(g, derivative="dot", applied_undef_args=None)
    >>> print(printer.doprint(expr))
    \partial_{xx}f{\left(x,y,z \right)} + \dot{g}

    Customizing vectors from the sympy.vector module. For a Cartesian system
    the default output looks like:

    >>> from sympy.vector import CoordSys3D
    >>> C = CoordSys3D("C")
    >>> x, y, z = C.base_scalars()
    >>> i, j, k = C.base_vectors()
    >>> f1, f2, f3 = [Function(k)(x, y, z) for k in ["f1", "f2", "f3"]]
    >>> v1 = f1 * i + f2 * j + f3 * k
    >>> print(printer.doprint(v1))     # doctest: +NORMALIZE_WHITESPACE
    f_{1}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{i}_{C}}
    + f_{2}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{j}_{C}}
    + f_{3}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{k}_{C}}

    Hide arguments of applied undefined functions:

    >>> for f in [f1, f2, f3]:
    ...     printer.add_rule(f, applied_undef_args=None)
    >>> print(printer.doprint(v1))
    f_{1}\,\mathbf{\hat{i}_{C}} + f_{2}\,\mathbf{\hat{j}_{C}} + f_{3}\,\mathbf{\hat{k}_{C}}

    By default, base vectors uses the i, j, k notation (for curvilinear systems
    too), and base scalars shows the systems they are associated to:

    >>> S = C.create_new("S", transformation="spherical")
    >>> r, theta, phi = S.base_scalars()
    >>> e_r, e_theta, e_phi = S.base_vectors()
    >>> f4, f5, f6 = [Function(k)(r, theta, phi) for k in ["f4", "f5", "f6"]]
    >>> v2 = f4 * e_r + f5 * e_theta + f6 * e_phi
    >>> print(printer.doprint(v2))     # doctest: +NORMALIZE_WHITESPACE
    f_{4}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}}
    + f_{5}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\theta}}
    + f_{6}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\phi}}

    Set base vectors to be rendered with the '\hat{e}_{direction}' notation,
    and base scalars to hide the system they are associated to:

    >>> printer.add_rule(S, base_vector_style="e", base_scalar_style="normal-ns")
    >>> print(printer.doprint(v2))     # doctest: +NORMALIZE_WHITESPACE
    f_{4}{\left(r,\theta,\phi \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}}
    + f_{5}{\left(r,\theta,\phi \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\theta}}
    + f_{6}{\left(r,\theta,\phi \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\phi}}

    Hide arguments of applied undefined functions:

    >>> for f in [f4, f5, f6]:
    ...     printer.add_rule(f, applied_undef_args=None)
    >>> print(printer.doprint(v2))     # doctest: +NORMALIZE_WHITESPACE
    f_{4}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}}
    + f_{5}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\theta}}
    + f_{6}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\phi}}

    Apply colors to specified sub-expressions:

    >>> printer.colorize[f4] = "red"
    >>> printer.colorize[f5] = "green"
    >>> print(printer.doprint(f4 + f5 + f6))
    \textcolor{red}{f_{4}} + \textcolor{green}{f_{5}} + f_{6}
    """

    latex_mode = settings.get("latex_mode", "plain")
    p = ExtendedLatexPrinter(**settings)

    def _print_latex_text(expr):
        """
        A function to generate the latex representation of SymPy expressions.
        """
        if _can_print(expr):
            s = p.doprint(expr)
            if latex_mode == 'plain':
                return '$\\displaystyle %s$' % s
            return s

    in_ipython = False
    try:
        ip = get_ipython()
    except NameError:
        pass
    else:
        in_ipython = (ip is not None)

    if in_ipython:
        latex_formatter = ip.display_formatter.formatters['text/latex']
        printable_types = [float, tuple, list, set, frozenset, dict, int]
        for cls in printable_types:
            latex_formatter.for_type(cls, _print_latex_text)
        Printable._repr_latex_ = _print_latex_text

    return p
