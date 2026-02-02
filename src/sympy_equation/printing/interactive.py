from sympy_equation.printing.extended_latex import ExtendedLatexPrinter
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


@print_function(ExtendedLatexPrinter)
def init_latex_printing(**settings):
    """Initialize an ``ExtendedLatexPrinter`` on this interactive shell.

    Read ``extended_latex`` for the documentation with all available settings.

    Returns
    -------
    latex_printer : ExtendedLatexPrinter
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
