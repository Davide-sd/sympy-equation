#!ipython
from sympy_equation.preparser import (
    sympy_equation_preparser as parser,
    integers_as_exact,
)
from sympy_equation import Equation, equation_config
from sympy.abc import a, b, c, d
from sympy import Float, Rational
import pytest


def test_integers_as_exact():
    lines = []
    lines.append('1/2*x + 0.333*x')
    lines.append('2/3*z + 2.0*y + ln(3*x)')
    result = integers_as_exact(lines)
    splitlines = result.split('\n')
    expectedlines = ['Integer (1 )/Integer (2 )*x +0.333 *x ',
            'Integer (2 )/Integer (3 )*z +2.0 *y +ln (Integer (3 )*x )']
    for k in range(len(splitlines)):
        assert splitlines[k] == expectedlines[k]


def test_set_integers_as_exact(ipython_shell):
    equation_config.integers_as_exact = True
    assert integers_as_exact in ipython_shell.input_transformers_post


def test_unset_integers_as_exact(ipython_shell):
    equation_config.integers_as_exact = False
    assert integers_as_exact not in ipython_shell.input_transformers_post


def test_sympy_equation_preparser():
    lines = []
    expected_out = []
    lines.append('# A comment.\n')
    expected_out.append('# A comment.\n')
    assert parser(lines) == expected_out
    lines.append('eq1 =@ a + b = c/d\n')
    expected_out.append('eq1 = Eqn( a + b , c/d)\n')
    assert parser(lines) == expected_out
    lines.append('obj?\n')
    expected_out.append('obj?\n')
    assert parser(lines) == expected_out
    lines.append('eq1 =@a + b=c/d\n')
    expected_out.append('eq1 = Eqn(a + b,c/d)\n')
    assert parser(lines) == expected_out
    lines.append('tst = (a\n')
    expected_out.append('tst = (a\n')
    lines.append('      +b)\n')
    expected_out.append('      +b)\n')
    assert parser(lines) == expected_out
    lines.append('@property\n')
    expected_out.append('@property\n')
    assert parser(lines) == expected_out
    lines.append('\n')
    expected_out.append('\n')
    assert parser(lines) == expected_out
    lines.append('eq1 =@ a + b = c/d # A trailing comment\n')
    expected_out.append('eq1 = Eqn( a + b , c/d )\n')
    assert parser(lines) == expected_out


def test_sympy_equation_preparser_errors():
    lines = []
    expected_out = []
    lines.append('# A comment.\n')
    expected_out.append('# A comment.\n')
    assert parser(lines) == expected_out
    lines.append('eq1 =@ a + b > c/d\n')
    pytest.raises(ValueError, lambda: parser(lines))


@pytest.mark.parametrize("eq_code, lhs, rhs", [
    ("e = Eqn(a/b, c/d)", a / b, c / d),
    ("e =@ a+b = c+d", a + b, c + d),
])
def test_sympy_equation_preparser_in_ipython(eq_code, lhs, rhs, ipython_shell):
    ipython_shell.run_cell(
        "from sympy import Integer\n"
        "from sympy_equation import Eqn\n"
        "from sympy.abc import a, b, c, d"
    )
    ipython_shell.run_cell(eq_code)
    assert isinstance(ipython_shell.user_ns["e"], Equation)
    assert ipython_shell.user_ns["e"].lhs == lhs
    assert ipython_shell.user_ns["e"].rhs == rhs


@pytest.mark.parametrize("integers_as_exact, eq_code, lhs, rhs", [
    (False, "e = Eqn(a+1/2, c/d+3/4)", a + Float("0.5"), c / d + Float("0.75")),
    (False, "e =@ a+1/2 = c/d+3/4", a + Float("0.5"), c / d + Float("0.75")),
    (True, "e = Eqn(a+1/2, c/d+3/4)", a + Rational(1, 2), c / d + Rational(3, 4)),
    (True, "e =@ a+1/2 = c/d+3/4", a + Rational(1, 2), c / d + Rational(3, 4)),
])
def test_sympy_equation_preparser_with_integers_as_exact_in_ipython(
    integers_as_exact, eq_code, lhs, rhs, ipython_shell
):
    ipython_shell.run_cell(
        "from sympy import Integer\n"
        "from sympy_equation import Eqn, equation_config\n"
        "from sympy.abc import a, b, c, d"
    )
    ipython_shell.run_cell(
        f"equation_config.integers_as_exact = {integers_as_exact}"
    )
    ipython_shell.run_cell(eq_code)
    assert ipython_shell.user_ns["e"].lhs == lhs
    assert ipython_shell.user_ns["e"].rhs == rhs
