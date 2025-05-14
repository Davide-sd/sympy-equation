from IPython.lib.pretty import pretty
from sympy import (
    symbols, integrate, simplify, expand, factor, Integral, Add,
    diff, FiniteSet, Function, Matrix, S, Eq, Equality,
    sin, cos, log, exp, latex, Symbol, I, pi, Float, Derivative, Rational,
    oo, Piecewise, gamma, sign, re, sqrt, root, Heaviside,
    solve as sympy_solve
)
from sympy.core.function import AppliedUndef
from sympy.printing.latex import LatexPrinter
from sympy_equation.algebraic_equation import (
    Eqn,
    Equation,
    solve,
    collect,
    equation_config,
)
import sympy_equation as se
import pytest


equation_config.show_label = True
equation_config.solve_to_list = False
a, b, c, d = symbols("a, b, c, d")


class CustomLatexPrinter(LatexPrinter):
    """Print undefined applied functions without arguments"""
    def _print_Function(self, expr, exp=None):
        if isinstance(expr, AppliedUndef):
            return self._print(Symbol(expr.func.__name__))
        return super()._print_Function(expr, exp)


def my_latex(expr, **settings):
    """Mimic latex()"""
    return CustomLatexPrinter(settings).doprint(expr)


def test_define_equation():
    a, b, c = symbols('a b c')
    pytest.raises(TypeError, lambda: Equation(FiniteSet(a), FiniteSet(b, c)))
    assert not Equation(1, 0).check()
    assert Eqn(1, 0) == Equation(1, 0)
    tsteqn = Equation(a, b/c)
    assert tsteqn.args == (a, b/c)
    assert tsteqn.lhs == a
    assert tsteqn.rhs == b/c
    assert tsteqn.free_symbols == {a, b, c}


def test_convert_equation():
    a, b, c = symbols('a b c')
    tsteqn = Equation(a, b/c)
    assert tsteqn.as_Boolean() == Eq(a, b/c)
    assert tsteqn.reversed == Equation(b/c, a)
    assert tsteqn.swap == Equation(b/c, a)


def test_binary_op():
    a, b, c = symbols('a b c')
    tsteqn = Equation(a, b/c)
    assert tsteqn + c == Equation(a + c, b/c + c)
    assert c + tsteqn == Equation(c + a, c + b/c)
    assert tsteqn*c == Equation(a*c, b)
    assert c*tsteqn == Equation(c*a, b)
    assert tsteqn - c == Equation(a - c, b/c - c)
    assert c - tsteqn == Equation(c - a, c - b/c)
    assert tsteqn/ c == Equation(a/c, b/c**2)
    assert c/tsteqn == Equation(c/a, c**2/b)
    assert tsteqn % c == Equation(a % c, (b/c) % c)
    assert c % tsteqn == Equation(c % a, c % (b/c))
    assert tsteqn**c == Equation(a**c, (b/c)**c)
    assert c**tsteqn == Equation(c**a, c**(b/c))
    assert tsteqn + tsteqn == Equation(2*a, 2*b/c)
    assert tsteqn*tsteqn == Equation(a**2, b**2/c**2)
    assert tsteqn - tsteqn == Equation(0, 0)
    assert tsteqn/tsteqn == Equation(1, 1)
    assert tsteqn % tsteqn == Equation(0, 0)
    assert tsteqn**tsteqn == Equation(a**a, (b/c)**(b/c))
    assert tsteqn**a == Equation(a**a, (b/c)**a)
    assert tsteqn._eval_power(tsteqn) == Equation(a**a, (b/c)**(b/c))
    assert tsteqn._eval_power(a) == Equation(a**a, (b/c)**a)
    pytest.raises(TypeError, lambda: tsteqn & tsteqn)
    pytest.raises(TypeError, lambda: 2 & tsteqn)
    pytest.raises(TypeError, lambda: tsteqn & 2)
    pytest.raises(TypeError, lambda: tsteqn ^ tsteqn)
    pytest.raises(TypeError, lambda: 2 ^ tsteqn)
    pytest.raises(TypeError, lambda: tsteqn ^ 2)


@pytest.mark.parametrize("show_label, human_text, output_txt", [
    ( True, True, 'a = b/c          (tsteqn)' ),
    ( True, False, 'Equation(a, b/c)' ),
    ( False, True, 'a = b/c' ),
    ( False, False, 'Equation(a, b/c)' ),
])
def test_output_pretty_print_show_label(show_label, human_text, output_txt):
    # verify that the output of a textual-based ipython cell
    # shows the correct results depending on the configuration values.
    a, b, c = symbols('a b c')
    tsteqn = Eqn(a, b / c)

    import __main__ as gs
    vars(gs)['tsteqn'] = tsteqn
    assert tsteqn._get_eqn_name() == 'tsteqn'

    equation_config.show_label = show_label
    equation_config.human_text = human_text
    assert pretty(tsteqn) == output_txt


@pytest.mark.parametrize("show_label, output_latex", [
    ( True, 'a = \\frac{b}{c}\\qquad (tsteqn)' ),
    ( True, 'a = \\frac{b}{c}\\qquad (tsteqn)' ),
    ( False, 'a = \\frac{b}{c}' ),
    ( False, 'a = \\frac{b}{c}' ),
])
def test_output_latex_show_label(
    show_label, output_latex, ipython_shell
):
    a, b, c = symbols('a b c')
    tsteqn = Eqn(a, b / c)

    import __main__ as gs
    vars(gs)['tsteqn'] = tsteqn
    assert tsteqn._get_eqn_name() == 'tsteqn'

    equation_config.show_label = show_label
    assert latex(tsteqn) == output_latex


def test_output_latex_custom_printer():
    f = Function("f")(a, b, c)
    eq = Eqn(f, 2 + sin(a))
    assert latex(eq) == "f{\\left(a,b,c \\right)} = \\sin{\\left(a \\right)} + 2"
    # use custom printer
    assert my_latex(eq) == "f = \\sin{\\left(a \\right)} + 2"


def test_outputs(capsys):
    a, b, c = symbols('a b c')
    tsteqn = Eqn(a, b/c)
    assert tsteqn.__str__() == 'a = b/c'
    assert tsteqn.__repr__() == 'Equation(a, b/c)'
    assert latex(tsteqn) == 'a = \\frac{b}{c}'


def test_outputs_solve(capsys):
    equation_config.human_text = True

    x, y = symbols('x y', real=True)
    eq1 = Eqn(abs(2*x + y),3)
    eq2 = Eqn(abs(x + 2*y),3)
    B = solve([eq1,eq2],x,y)
    assert B.__repr__() == 'FiniteSet(FiniteSet(Equation(x, -3), ' \
        'Equation(y, 3)), FiniteSet(Equation(x, -1), ' \
        'Equation(y, -1)), FiniteSet(Equation(x, 1), ' \
        'Equation(y, 1)), FiniteSet(Equation(x, 3),' \
        ' Equation(y, -3)))'
    assert B.__str__() == '{{x = -3, y = 3}, {x = -1, y = -1}, ' \
                                   '{x = 1, y = 1}, {x = 3, y = -3}}'

def test_sympy_functions():
    a, b, c = symbols('a b c')
    tsteqn = Equation(a, b/c)
    assert tsteqn.apply(sin) == Equation(sin(a),sin(b/c))
    assert tsteqn.apply(log) == Equation(log(a),log(b/c))
    # Check matrix exponentiation is not overridden.
    assert tsteqn.apply(exp) == Equation(exp(tsteqn.lhs),exp(tsteqn.rhs))
    tsteqn5 = Equation(a, Matrix([[1, 1], [1, 1]]))
    assert tsteqn5.apply(exp).lhs == exp(a)
    assert tsteqn5.apply(exp).rhs == exp(Matrix([[1, 1], [1, 1]]))


@pytest.mark.parametrize("eqn, args, kwargs, res", [
    (Equation(a, a**2*b/c), (a, ), {}, Equation(1, 2*a*b/c)),
    (Equation(a, a**2*b/c), (a, ), {"evaluate": True}, Equation(1, 2*a*b/c)),
    (Equation(a, a**2*b/c), (a, ), {"evaluate": False}, Equation(Derivative(a, a), Derivative(a**2*b/c, a))),
    (Equation(a, a**2*b/c), (a, 2), {}, Equation(0, 2*b/c)),
])
def test_diff(eqn, args, kwargs, res):
    assert diff(eqn, *args, **kwargs) == res
    assert eqn.diff(*args, **kwargs) == res


def test_derivative():
    e = Equation(a, a**2*b/c)
    assert Derivative(e, a).doit() == Equation(1, 2*a*b/c)


@pytest.mark.parametrize("eqn, args, kwargs, res", [
    (Equation(a, a**2*b/c), (a, ), {}, Equation(a**2/2, a**3*b/(3*c))),
    (Equation(a, a**2*b/c), ((a, 1, 2), ), {}, Equation(Rational(3, 2), 7*b/(3*c))),
    (Equation(a, b**a*exp(-b)), ((b, 0, oo), ), {}, Equation(oo*sign(a), Piecewise((gamma(a + 1), re(a) > -1), (Integral(b**a*exp(-b), (b, 0, oo)), True)))),
    (Equation(a, b**a*exp(-b)), ((b, 0, oo), ), {"conds": "none"}, Equation(oo*sign(a), gamma(a + 1))),
])
def test_integrate(eqn, args, kwargs, res):
    assert integrate(eqn, *args, **kwargs) == res
    assert eqn.integrate(*args, **kwargs) == res


def test_helper_functions():
    a, b, c, x= symbols('a b c x')
    tsteqn = Equation(a, b/c)
    assert integrate(tsteqn, c) == Equation(a*c, b*log(c))
    assert tsteqn.integrate(c) == Equation(a*c, b*log(c))
    assert tsteqn.evalf(4, {b: 2.0, c: 4}) == Equation(a, Float("0.5", dps=4))
    assert diff(tsteqn, c) == Equation(0, -b/c**2)
    assert tsteqn.diff(c) == Equation(0, -b/c**2)
    tsteqn = Equation(a*c, b/c)
    assert diff(tsteqn, c) == Equation(a, -b/c**2)
    assert tsteqn.diff(c) == Equation(a, -b/c**2)

    def adsq(eqn):
        # Arbitrary python function
        return eqn + eqn**2

    assert adsq(Equation(a*c, b/c)) == Equation(a**2*c**2 + a*c, b**2/c**2 +
                                                b/c)
    assert Equation((a - 1)*(a + 1), (2*b + c)**2).expand() == Equation(
        a**2 - 1, 4*b**2 + 4*b*c + c**2)
    assert expand(Equation((a - 1)*(a + 1), (2*b + c)**2)) == Equation(
        a**2 - 1, 4*b**2 + 4*b*c + c**2)
    assert Equation(a**2 - 1, 4*b**2 + 4*b*c + c**2).factor() == Equation(
        (a - 1)*(a + 1), (2*b + c)**2)
    assert factor(Equation(a**2 - 1, 4*b**2 + 4*b*c + c**2)) == Equation(
        (a - 1)*(a + 1), (2*b + c)**2)
    assert Equation(a**2 - 1, 4*b**2 + 4*b*c + c*a).collect(c) == Equation(
        a**2- 1, 4*b**2 + c*(a + 4*b))
    # assert collect(Equation(a**2 - 1, 4*b**2 + 4*b*c + c*a), c) == Equation(
    #     a**2- 1, 4*b**2 + c*(a + 4*b))
    assert Equation((a + 1)**2/(a + 1), exp(log(c))).simplify() == Equation(
        a + 1, c)
    assert simplify(Equation((a + 1)**2/(a + 1), exp(log(c)))) == Equation(
        a + 1, c)
    assert Eqn(a,b/c).apply(lambda e: root(e, 3)) == Equation(a**(S(1)/S(3)), (b/c)**(S(1)/S(3)))
    assert root(b/c,3) == (b/c)**(S(1)/S(3))
    assert Eqn(a,b/c).apply(sqrt) == Equation(sqrt(a), sqrt(b/c))


def test_solve_1():
    a, b, c, x = symbols('a b c x')
    eq = Equation(a * x**2, b * x + c)
    computed_sol = solve(eq, x)
    sol1 = Equation(x, ((b - sqrt(4*a*c + b**2))/(2*a)).expand())
    sol2 = Equation(x, ((b + sqrt(4*a*c + b**2))/(2*a)).expand())
    assert len(computed_sol) == 2
    assert sol1 in computed_sol
    assert sol2 in computed_sol

    # verify sympy_equation.solve is a wrapper of sympy.solve
    res1a = solve(eq.to_expr(), x)
    assert isinstance(res1a, list) and (len(res1a) == 2)
    assert sol1.rhs.simplify() in res1a
    assert sol2.rhs.simplify() in res1a

    res1b = sympy_solve(eq.to_expr(), x)
    assert res1a == res1b

    res2a = solve([eq.to_expr()], x)
    assert isinstance(res2a, list) and (len(res2a) == 2)
    assert all(isinstance(t, tuple) for t in res2a)
    assert all(len(t) == 1 for t in res2a)
    res2a = [res2a[0][0], res2a[1][0]]
    assert sol1.rhs in res2a
    assert sol2.rhs in res2a

    res2b = sympy_solve([eq.to_expr()], x)
    assert isinstance(res2b, list) and (len(res2b) == 2)
    assert all(isinstance(t, tuple) for t in res2b)
    assert all(len(t) == 1 for t in res2b)
    res2b = [res2b[0][0], res2b[1][0]]
    assert res2a == res2b


def test_solve_2():
    x, y = symbols('x y', real = True)
    eq1 = Eqn(abs(2*x + y), 3)
    eq2 = Eqn(abs(x + 2*y), 3)

    assert solve([eq1,eq2], x, y) == FiniteSet(
        FiniteSet(Equation(x, -3), Equation(y, 3)),
        FiniteSet(Equation(x, -1), Equation(y, -1)),
        FiniteSet(Equation(x, 1), Equation(y, 1)),
        FiniteSet(Equation(x, 3), Equation(y, -3))
    )

    equation_config.solve_to_list = True
    assert solve([eq1,eq2], x, y) == [
        [Equation(x, -3), Equation(y, 3)],
        [Equation(x, -1), Equation(y, -1)],
        [Equation(x, 1), Equation(y, 1)],
        [Equation(x, 3), Equation(y, -3)]
    ]

    # verify sympy_equation.solve is a wrapper of sympy.solve
    exprs = [e.as_expr() for e in [eq1, eq2]]
    res1 = solve(exprs, x, y, dict=False)
    res2 = solve(exprs, x, y, dict=True)
    res3 = sympy_solve(exprs, x, y, dict=False)
    res4 = sympy_solve(exprs, x, y, dict=True)
    assert res1 == res3
    assert res2 == res4


def test_solve_3():
    xi, wn = symbols("xi omega_n", real=True, positive=True)
    Tp, Ts = symbols("T_p, T_s", real=True, positive=True)
    e1 = Eqn(Tp, pi / (wn*sqrt(1 - xi**2)))
    e2 = Eqn(Ts, 4 / (wn*xi))

    equation_config.solve_to_list = False
    assert solve([e1, e2], [xi, wn]) == FiniteSet(
        Eqn(xi, 4*Tp/sqrt(16*Tp**2 + pi**2*Ts**2)),
        Eqn(wn, sqrt(16*Tp**2 + pi**2*Ts**2)/(Tp*Ts)))

    equation_config.solve_to_list = True
    assert solve([e1, e2], [xi, wn]) == [
        Eqn(xi, 4*Tp/sqrt(16*Tp**2 + pi**2*Ts**2)),
        Eqn(wn, sqrt(16*Tp**2 + pi**2*Ts**2)/(Tp*Ts))
    ]

    # order of symbols are swapped -> results are swapped as well
    assert solve([e1, e2], [wn, xi]) == [
        Eqn(wn, sqrt(16*Tp**2 + pi**2*Ts**2)/(Tp*Ts)),
        Eqn(xi, 4*Tp/sqrt(16*Tp**2 + pi**2*Ts**2))
    ]


def test_Heaviside():
    a, b, c, x = symbols('a b c x')
    tsteqn = Equation(a, b / c)
    assert (tsteqn.apply(Heaviside) ==
            Equation(Heaviside(tsteqn.lhs), Heaviside(tsteqn.rhs)))
    assert Heaviside(0) == S(1)/S(2)


def test_equality_extension():
    a, b, c, x = symbols('a b c x')
    tstequal = Equality(a, b / c)
    assert(tstequal.to_Equation() == Equation(a, b / c))


def test_apply_syntax():
    a, b, c, x = symbols('a b c x')
    tsteqn = Equation(a, b/c)
    assert tsteqn.apply(log) == Equation(log(a), log(b/c))
    assert tsteqn.applylhs(log) == Equation(log(a), b / c)
    assert tsteqn.applyrhs(log) == Equation(a, log(b / c))
    poly = Equation(a*x**2 + b*x + c*x**2, a*x**3 + b*x**3 + c*x)
    assert poly.applyrhs(collect, x) == Equation(poly.lhs, poly.rhs.collect(x))


def test_do_syntax():
    a, b, c, x = symbols('a b c x')
    tsteqn = Equation(a, b/c)
    pytest.raises(AttributeError, lambda: tsteqn.do.log())
    poly = Equation(a*x**2 + b*x + c*x**2, a*x**3 + b*x**3 + c*x)
    assert poly.dorhs.collect(x) == Eqn(poly.lhs, poly.rhs.collect(x))
    assert poly.dolhs.collect(x) == Eqn(poly.lhs.collect(x), poly.rhs)
    assert poly.do.collect(x) == Eqn(poly.lhs.collect(x), poly.rhs.collect(x))


def test_rewrite_add():
    b, x = symbols("x, b")
    eq = Equation(x + b, x - b)
    assert eq.rewrite(Add) == Equation(2 * b, 0)
    assert set(eq.rewrite(Add, evaluate=None).lhs.args) == set((b, x, b, -x))
    assert set(eq.rewrite(Add, evaluate=False).lhs.args) == set((b, x, b, -x))
    assert eq.rewrite(Add, eqn=False) == 2 * b
    assert set(eq.rewrite(Add, eqn=False, evaluate=False).args) == set((b, x, b, -x))


def test_rewrite():
    x = symbols("x")
    eq = Equation(exp(I*x),cos(x) + I*sin(x))

    # NOTE: Must use `sexp` otherwise the test is going to fail.
    # This reflects the fact that rewrite pulls the fuction exp internally
    # from the definitions of functions in sympy and not from the globally
    # redefined functions that are Equation aware.
    from sympy import exp as sexp
    assert eq.rewrite(exp) == Equation(exp(I*x), sexp(I*x))
    assert eq.rewrite(Add) == Equation(exp(I*x) - I*sin(x) - cos(x), 0)


def test_subs():
    a, b, c, x = symbols('a b c x')
    eq1 = Equation(x + a + b + c, x * a * b * c)
    eq2 = Equation(x + a, 4)
    assert eq1.subs(a, 2) == Equation(x + b + c + 2, 2 * x * b * c)
    assert eq1.subs([(a, 2), (b, 3)]) == Equation(x + c + 5, 6 * x * c)
    assert eq1.subs({a: 2, b: 3}) == Equation(x + c + 5, 6 * x * c)
    assert eq1.subs(eq2) == Equation(4 + b + c, x * a * b * c)

    # verify that proper errors are raised
    eq3 = Equation(b, 5)
    pytest.raises(TypeError, lambda: eq1.subs([eq2, eq3]))
    pytest.raises(ValueError, lambda: eq1.subs(eq2, {b: 5}))

    # verify that substituting an Equation into an expression is not supported
    pytest.raises(ValueError, lambda: eq1.dolhs.subs(eq2))
    pytest.raises(ValueError, lambda: eq1.dorhs.subs(eq2))
    pytest.raises(ValueError, lambda: (x + a + b + c).subs(eq2))

    # verify the effectiveness of `simultaneous`
    eq = Equation((x + a) / a, b * c)
    sd = {x + a: a, a: x + a}
    assert eq.subs(sd) == Equation(1, b * c)
    assert eq.subs(sd, simultaneous=True) == Equation(a / (x + a), b * c)


def test_issue_23():
    # This gave a key error
    a, t = symbols('a t')
    assert simplify(a * cos(t) + sin(t)) == a * cos(t) + sin(t)


@pytest.mark.parametrize("eqn, expected", [
    (Eqn(a / b, c / d), Eqn(a * d, c * b)),
    (Eqn(a / b, 1), Eqn(a, b)),
    (Eqn(a / b, -1), Eqn(a, -b)),
])
def test_cross_multiply(eqn, expected):
    assert eqn.cross_multiply() == expected


@pytest.mark.parametrize("eqn, expr", [
    (Eqn(a, b), a - b),
    (Eqn(a / b, c / d), a / b - c / d),
])
def test_as_expr_to_expr(eqn, expr):
    assert eqn.as_expr() == expr
    assert eqn.to_expr() == expr


@pytest.mark.parametrize("eqn, kwargs, res", [
    (Eqn(pi, b * c * 0.5), {}, Equation(pi, b*c/2)),
    (Eqn(pi, b * c * 0.5), {"tolerance": 0.1}, Equation(Rational(22, 7), b*c/2)),
])
def test_nsimplify(eqn, kwargs, res):
    assert eqn.nsimplify(**kwargs) == res


def test_version():
    assert hasattr(se, "__version__")
