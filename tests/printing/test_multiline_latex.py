from sympy_equation import multiline_latex
from sympy import symbols, Function, sin, exp, cos, log, I


def test_multiline_latex_custom_printer_options():
    # verify that multiline_latex is using extended_latex

    x, y, alpha = symbols('x y alpha')
    f = Function("f")(x, y)
    expr = sin(alpha*y) + exp(I*alpha) - cos(log(y)) + f.diff(y)
    assert multiline_latex(x, expr) == r"""\begin{eqnarray}
x & = & e^{i \alpha} \nonumber\\
& & + \sin{\left(\alpha y \right)} \nonumber\\
& & - \cos{\left(\log{\left(y \right)} \right)} \nonumber\\
& & + \frac{\partial}{\partial y} f{\left(x,y \right)} 
\end{eqnarray}"""

    res = multiline_latex(x, expr, applied_undef_args=None, derivative="subscript")
    assert res == r"""\begin{eqnarray}
x & = & e^{i \alpha} \nonumber\\
& & + \sin{\left(\alpha y \right)} \nonumber\\
& & - \cos{\left(\log{\left(y \right)} \right)} \nonumber\\
& & + f_{y} 
\end{eqnarray}"""
