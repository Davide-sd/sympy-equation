from sympy_equation.printing.extended_latex import ExtendedLatexPrinter
from sympy.printing.defaults import Printable
from sympy.printing.printer import print_function


@print_function(ExtendedLatexPrinter)
def init_latex_printing(**settings):
    """Initialize an ``ExtendedLatexPrinter`` on this interactive shell.

    Read ``extended_latex`` for the documentation with all available settings.

    Returns
    -------
    latex_printer : ExtendedLatexPrinter
    """

    latex_mode = settings.get("latex_mode", "plain")
    p = ExtendedLatexPrinter(settings)

    def _print_latex_text(expr):
        """
        A function to generate the latex representation of SymPy expressions.
        """
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
