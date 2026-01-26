from sympy_equation import Eqn, extended_latex, multiline_latex
from sympy_equation.printing.extended_latex import ExtendedLatexPrinter
from sympy import (
    symbols, Function, asin, cos, sin, sqrt, Derivative, exp, log, I,
    Rational, Pow
)
from sympy.vector import CoordSys3D
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
import pytest


t, x = symbols("t, x")
f = Function("f")
g = Function("rho_1__0")


def test_applied_undef_args():
    f = Function("f")
    g = Function("g")
    x, y = symbols("x, y")
    expr = f(x, g(x, y))

    assert extended_latex(expr) == r"f{\left(x,g{\left(x,y \right)} \right)}"
    assert extended_latex(expr, applied_undef_args=True) == r"f{\left(x,g{\left(x,y \right)} \right)}"
    assert extended_latex(expr, applied_undef_args="all") == r"f{\left(x,g{\left(x,y \right)} \right)}"
    assert extended_latex(expr, applied_undef_args="first-level") == r"f{\left(x,g\right)}"
    assert extended_latex(expr, applied_undef_args=False) == r"f"
    assert extended_latex(expr, applied_undef_args=None) == r"f"


def test_print_func_no_args_with_exponent():
    x, y = symbols("x, y")
    gamma = symbols("gamma")
    f = Function("f")

    expr = f(x, y)
    assert extended_latex(expr, applied_undef_args=True) == r"f{\left(x,y \right)}"
    assert extended_latex(expr, applied_undef_args=False) == r"f"

    expr = f(x, y)**(2 + x)
    assert extended_latex(expr, applied_undef_args=True) == r"f^{x + 2}{\left(x,y \right)}"
    assert extended_latex(expr, applied_undef_args=False) == r"f^{x + 2}"

    expr = f(x, y)**gamma
    assert extended_latex(expr, applied_undef_args=True) == r"f^{\gamma}{\left(x,y \right)}"
    assert extended_latex(expr, applied_undef_args=False) == r"f^{\gamma}"


@pytest.mark.parametrize("expr, derivative, applied_undef_args, expected", [
    (f(t).diff(t), None, True, r"\frac{d}{d t} f{\left(t \right)}"),
    (f(t).diff(t), None, False, r"\frac{d f}{d t}"),
    (f(t).diff(t), "dot", True, r"\dot{f}{\left(t \right)}"),
    (f(t).diff(t), "dot", False, r"\dot{f}"),
    (f(t, x).diff(t), None, True, r"\frac{\partial}{\partial t} f{\left(t,x \right)}"),
    (f(t, x).diff(t), None, False, r"\frac{\partial f}{\partial t}"),
    (f(t, x).diff(t), "dot", True, r"\dot{f}{\left(t,x \right)}"),
    (f(t, x).diff(t), "dot", False, r"\dot{f}"),
    (g(t).diff(t), "dot", True, r"\dot{\rho}_{1}^{0}{\left(t \right)}"),
    (g(t).diff(t), "dot", False, r"\dot{\rho}_{1}^{0}"),
    (f(t, x).diff(x), "dot", True, r"\frac{\partial}{\partial x} f{\left(t,x \right)}"),
    (f(t, x).diff(x), "dot", False, r"\frac{\partial f}{\partial x}"),
    (f(t, x).diff(t), "subscript", True, r"f_{t}{\left(t,x \right)}"),
    (f(t, x).diff(t), "subscript", False, r"f_{t}"),
    (f(t, x).diff(t, x), "subscript", True, r"f_{tx}{\left(t,x \right)}"),
    (f(t, x).diff(t, x), "subscript", False, r"f_{tx}"),
    (f(t, x).diff(x, t), "subscript", True, r"f_{tx}{\left(t,x \right)}"),
    (f(t, x).diff(x, t), "subscript", False, r"f_{tx}"),
    (f(t, x).diff(t, 2), "subscript", True, r"f_{tt}{\left(t,x \right)}"),
    (f(t, x).diff(t, 2), "subscript", False, r"f_{tt}"),
    (f(t, x).diff(t, 2, x), "subscript", True, r"f_{ttx}{\left(t,x \right)}"),
    (f(t, x).diff(t, 2, x), "subscript", False, r"f_{ttx}"),
    (g(t).diff(t), "subscript", True, r"\rho_{1,t}^{0}{\left(t \right)}"),
    (g(t).diff(t), "subscript", False, r"\rho_{1,t}^{0}"),
    (g(t, x).diff(t, x, x), "subscript", True, r"\rho_{1,txx}^{0}{\left(t,x \right)}"),
    (g(t, x).diff(t, x, x), "subscript", False, r"\rho_{1,txx}^{0}"),
    (f(t).diff(t), "prime-arabic", True, r"f^{\prime}{\left(t \right)}"),
    (f(t).diff(t), "prime-arabic", False, r"f^{\prime}"),
    (f(t).diff(t, 2), "prime-arabic", True, r"f^{\prime\prime}{\left(t \right)}"),
    (f(t).diff(t, 2), "prime-arabic", False, r"f^{\prime\prime}"),
    (f(t).diff(t, 3), "prime-arabic", True, r"f^{\prime\prime\prime}{\left(t \right)}"),
    (f(t).diff(t, 3), "prime-arabic", False, r"f^{\prime\prime\prime}"),
    (f(t).diff(t, 4), "prime-arabic", True, r"f^{\left(4\right)}{\left(t \right)}"),
    (f(t).diff(t, 4), "prime-arabic", False, r"f^{\left(4\right)}"),
    (g(t).diff(t), "prime-arabic", True, r"\rho_{1}^{0,\prime}{\left(t \right)}"),
    (g(t).diff(t), "prime-arabic", False, r"\rho_{1}^{0,\prime}"),
    (f(t).diff(t), "prime-roman", True, r"f^{\prime}{\left(t \right)}"),
    (f(t).diff(t), "prime-roman", False, r"f^{\prime}"),
    (f(t).diff(t, 2), "prime-roman", True, r"f^{\prime\prime}{\left(t \right)}"),
    (f(t).diff(t, 2), "prime-roman", False, r"f^{\prime\prime}"),
    (f(t).diff(t, 3), "prime-roman", True, r"f^{\prime\prime\prime}{\left(t \right)}"),
    (f(t).diff(t, 3), "prime-roman", False, r"f^{\prime\prime\prime}"),
    (f(t).diff(t, 4), "prime-roman", True, r"f^{\left(iv\right)}{\left(t \right)}"),
    (f(t).diff(t, 4), "prime-roman", False, r"f^{\left(iv\right)}"),
    (f(t, x).diff(t), "prime-arabic", True, r"\frac{\partial}{\partial t} f{\left(t,x \right)}"),
    (f(t, x).diff(t), "prime-arabic", False, r"\frac{\partial f}{\partial t}"),
    (f(t, x).diff(t), "prime-roman", True, r"\frac{\partial}{\partial t} f{\left(t,x \right)}"),
    (f(t, x).diff(t), "prime-roman", False, r"\frac{\partial f}{\partial t}"),
    (g(t).diff(t, 4), "prime-roman", True, r"\rho_{1}^{0,\left(iv\right)}{\left(t \right)}"),
    (g(t).diff(t, 4), "prime-roman", False, r"\rho_{1}^{0,\left(iv\right)}"),
    (f(t).diff(t), "d-notation", True, r"D_{t}f{\left(t \right)}"),
    (f(t).diff(t), "d-notation", False, r"D_{t}f"),
    (f(t).diff(t, 3), "d-notation", True, r"D_{t}^{3}f{\left(t \right)}"),
    (f(t).diff(t, 3), "d-notation", False, r"D_{t}^{3}f"),
    (f(t, x).diff(t), "d-notation", True, r"\partial_{t}f{\left(t,x \right)}"),
    (f(t, x).diff(t), "d-notation", False, r"\partial_{t}f"),
    (f(t, x).diff(t, x), "d-notation", True, r"\partial_{tx}f{\left(t,x \right)}"),
    (f(t, x).diff(t, x), "d-notation", False, r"\partial_{tx}f"),
    (f(t, x).diff(t, 2, x), "d-notation", True, r"\partial_{ttx}f{\left(t,x \right)}"),
    (f(t, x).diff(t, 2, x), "d-notation", False, r"\partial_{ttx}f"),
    (g(t).diff(t, 4), "d-notation", True, r"D_{t}^{4}\rho^{0}_{1}{\left(t \right)}"),
    (g(t).diff(t, 4), "d-notation", False, r"D_{t}^{4}\rho^{0}_{1}"),
])
def test_derivatives_applied_undef_1(expr, derivative, applied_undef_args, expected):
    settings = dict(derivative=derivative, applied_undef_args=applied_undef_args)
    assert extended_latex(expr, **settings) == expected


def test_derivative_applied_undef_2():
    # here, the name of the function tb is wrapped in \bar{}
    # test that the printer is able to deal with it

    r, x, tau, rb, tbs, xb = symbols(r"r x tau \bar{r} \bar{\theta} \bar{x}")
    tb = Function(r"\bar{\theta}")(xb, rb)
    tb_eq = Eqn(tb, rb / xb)
    psi = Function("psi")(xb, tb)
    f = Function("f")
    e = Eqn(psi, xb**2 * f(tb))

    assert extended_latex(e, applied_undef_args=False) == r"\psi = \bar{x}^{2} f"

    e_rb = e.apply(Derivative, rb).dorhs.doit()
    assert extended_latex(e_rb, applied_undef_args=False, derivative=None) == r"\frac{\partial \psi}{\partial \bar{r}} = \bar{x}^{2} \frac{\partial \bar{\theta}}{\partial \bar{r}} \frac{\partial f}{\partial \bar{\theta}}"
    assert extended_latex(e_rb, applied_undef_args=False, derivative="dot") == r"\frac{\partial \psi}{\partial \bar{r}} = \bar{x}^{2} \frac{\partial \bar{\theta}}{\partial \bar{r}} \frac{\partial f}{\partial \bar{\theta}}"
    assert extended_latex(e_rb, applied_undef_args=False, derivative="subscript") == r"\psi_{\bar{r}} = \bar{x}^{2} \bar{\theta}_{\bar{r}} f_{\bar{\theta}}"
    assert extended_latex(e_rb, applied_undef_args=False, derivative="d-notation") == r"\partial_{\bar{r}}\psi = \bar{x}^{2} \partial_{\bar{r}}\bar{\theta} D_{\bar{\theta}}f"
    assert extended_latex(e_rb, applied_undef_args=False, derivative="prime-arabic") == r"\frac{\partial \psi}{\partial \bar{r}} = \bar{x}^{2} \frac{\partial \bar{\theta}}{\partial \bar{r}} f^{\prime}"
    assert extended_latex(e_rb, applied_undef_args=False, derivative="prime-roman") == r"\frac{\partial \psi}{\partial \bar{r}} = \bar{x}^{2} \frac{\partial \bar{\theta}}{\partial \bar{r}} f^{\prime}"


@pytest.mark.parametrize("expr, derivative, applied_undef_args, expected", [
    (f(t).diff(t)**2, None, True, r"\left(\frac{d}{d t} f{\left(t \right)}\right)^{2}"),
    (f(t).diff(t)**2, None, False, r"\left(\frac{d f}{d t}\right)^{2}"),
    (f(t).diff(t)**2, "dot", True, r"\dot{f}{\left(t \right)}^{2}"),
    (f(t).diff(t)**2, "dot", False, r"\dot{f}^{2}"),
    (f(t).diff(t)**2, "subscript", True, r"f_{t}^{2}{\left(t \right)}"),
    (f(t).diff(t)**2, "subscript", False, r"f_{t}^{2}"),
    (f(t).diff(t, 4)**2, "prime-arabic", True, r"\left(f^{\left(4\right)}{\left(t \right)}\right)^{2}"),
    (f(t).diff(t, 4)**2, "prime-arabic", False, r"\left(f^{\left(4\right)}\right)^{2}"),
    (f(t).diff(t, 4)**2, "prime-roman", True, r"\left(f^{\left(iv\right)}{\left(t \right)}\right)^{2}"),
    (f(t).diff(t, 4)**2, "prime-roman", False, r"\left(f^{\left(iv\right)}\right)^{2}"),
    (f(t).diff(t)**2, "d-notation", True, r"\left(D_{t}f{\left(t \right)}\right)^{2}"),
    (f(t).diff(t)**2, "d-notation", False, r"\left(D_{t}f\right)^{2}"),
    (g(t).diff(t)**2, "subscript", True, r"\left(\rho_{1,t}^{0}{\left(t \right)}\right)^{2}"),
    (g(t).diff(t)**2, "subscript", False, r"\left(\rho_{1,t}^{0}\right)^{2}"),
    (f(t).diff(t)**symbols("gamma"), "subscript", True, r"f_{t}^{\gamma}{\left(t \right)}"),
    (f(t).diff(t)**symbols("gamma"), "subscript", False, r"f_{t}^{\gamma}"),
])
def test_power_of_derivatives_applied_undef(expr, derivative, applied_undef_args, expected):
    settings = dict(derivative=derivative, applied_undef_args=applied_undef_args)
    assert extended_latex(expr, **settings) == expected


def test_unmodified_expr():
    # sympy.physics.vector.printing.vlatex does something silly:
    # it executes the `.doit()` method of a derivative before printing it.
    # this leads to unpredictable and difficult to debug situations,
    # because the expression being visualized on the screen is not the
    # actual expression given to the printer.
    # see: https://github.com/sympy/sympy/issues/25415
    # Instead, extended_latex doesn't execute the doit method.

    R, L, t = symbols('R, L, t')
    theta = Function('theta')
    e = -R*sin(theta(t))*Derivative(theta(t), t) - R*sin(theta(t))*Derivative(asin(R*sin(theta(t))/L), t)

    # NOTE: here you can see a \partial/\partial t, which would not be visible
    # if using sympy.physics.vector.printing.vlatex
    assert extended_latex(e) == r"- R \sin{\left(\theta{\left(t \right)} \right)} \frac{d}{d t} \theta{\left(t \right)} - R \sin{\left(\theta{\left(t \right)} \right)} \frac{\partial}{\partial t} \operatorname{asin}{\left(\frac{R \sin{\left(\theta{\left(t \right)} \right)}}{L} \right)}"


def test_add_rule_1():
    x, y, z = symbols("x:z")
    f = Function("f")(x, y, z)
    g = Function("g")(x, y, z)

    p = ExtendedLatexPrinter()
    assert len(p.override_rules) == 0
    assert p.doprint(f + g) == r"f{\left(x,y,z \right)} + g{\left(x,y,z \right)}"
    assert p.doprint(f.diff(x) + g.diff(y)) == r"\frac{\partial}{\partial x} f{\left(x,y,z \right)} + \frac{\partial}{\partial y} g{\left(x,y,z \right)}"

    p.add_rule(g, applied_undef_args=False)
    assert len(p.override_rules) == 1
    assert p.doprint(f + g) == r"f{\left(x,y,z \right)} + g"
    assert p.doprint(f.diff(x) + g.diff(y)) == r"\frac{\partial}{\partial x} f{\left(x,y,z \right)} + \frac{\partial g}{\partial y}"

    p.add_rule(g, derivative="d-notation")
    assert len(p.override_rules) == 2
    assert p.doprint(f.diff(x) + g.diff(y)) == r"\frac{\partial}{\partial x} f{\left(x,y,z \right)} + \partial_{y}g"


def test_add_rule_2():
    x, y, z = symbols("x:z")
    f = Function("f")(x, y, z)
    g = Function("g")(x)

    p = ExtendedLatexPrinter(
        dict(applied_undef_args=None, derivative="subscript"))
    p.add_rule(g, derivative="prime-arabic")
    assert p.doprint(f.diff(x) + g.diff(x, 2)) == r"f_{x} + g^{\prime\prime}"


def test_add_rule_3():
    C = CoordSys3D("C")
    x, y, z = C.base_scalars()
    i, j, k = C.base_vectors()
    S = C.create_new("S", transformation="spherical")
    r, theta, phi = S.base_scalars()
    e_r, e_theta, e_phi = S.base_vectors()

    f1 = Function("f1")(x, y, z)
    f2 = Function("f2")(x, y, z)
    f3 = Function("f3")(x, y, z)
    v1 = f1 * i + f2 * j + f3 * k

    f4 = Function("f4")(r, theta, phi)
    f5 = Function("f5")(r, theta, phi)
    f6 = Function("f6")(r, theta, phi)
    v2 = f4 * e_r + f5 * e_theta + f6 * e_phi

    p = ExtendedLatexPrinter()
    assert p.doprint(v1) == r"f_{1}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{i}_{C}} + f_{2}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{j}_{C}} + f_{3}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{k}_{C}}"
    assert p.doprint(v2) == r"f_{4}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{i}_{S}} + f_{5}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{j}_{S}} + f_{6}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{k}_{S}}"

    p.add_rule(S, base_vector_style="e")
    assert p.doprint(v2) == r"f_{4}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}} + f_{5}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\theta}} + f_{6}{\left(r_{\text{S}},\theta_{\text{S}},\phi_{\text{S}} \right)}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\phi}}"

    p.add_rule(f4, applied_undef_args=None)
    p.add_rule(f5, applied_undef_args=None)
    p.add_rule(f6, applied_undef_args=None)
    assert p.doprint(v1) == r"f_{1}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{i}_{C}} + f_{2}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{j}_{C}} + f_{3}{\left(x_{\text{C}},y_{\text{C}},z_{\text{C}} \right)}\,\mathbf{\hat{k}_{C}}"
    assert p.doprint(v2) == r"f_{4}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}} + f_{5}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\theta}} + f_{6}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\phi}}"

    p.add_rule(C, base_scalar_style="normal-ns")
    assert p.doprint(v1) == r"f_{1}{\left(x,y,z \right)}\,\mathbf{\hat{i}_{C}} + f_{2}{\left(x,y,z \right)}\,\mathbf{\hat{j}_{C}} + f_{3}{\left(x,y,z \right)}\,\mathbf{\hat{k}_{C}}"
    assert p.doprint(x*y*z) == "x y z"
    assert p.doprint(r * theta * phi) == r"\phi_{\text{S}} r_{\text{S}} \theta_{\text{S}}"


def test_vector_dyadics():
    C = CoordSys3D("C")
    assert extended_latex(C.i | C.i) == r"\left(\mathbf{\hat{i}_{C}}\otimes\mathbf{\hat{i}_{C}}\right)"
    assert extended_latex(C.i | C.j) == r"\left(\mathbf{\hat{i}_{C}}\otimes\mathbf{\hat{j}_{C}}\right)"
    assert extended_latex(C.j | C.i) == r"\left(\mathbf{\hat{j}_{C}}\otimes\mathbf{\hat{i}_{C}}\right)"
    assert extended_latex(C.j | C.i, dyadic_style="otimes") == r"\left(\mathbf{\hat{j}_{C}}\otimes\mathbf{\hat{i}_{C}}\right)"
    assert extended_latex(C.j | C.i, dyadic_style="vline") == r"\left(\mathbf{\hat{j}_{C}}\middle|\mathbf{\hat{i}_{C}}\right)"
    assert extended_latex(C.j | C.i, dyadic_style="none") == r"\left(\mathbf{\hat{j}_{C}}\mathbf{\hat{i}_{C}}\right)"

    d = C.i | C.j
    assert extended_latex(2 * d) == r"2\left(\mathbf{\hat{i}_{C}}\otimes\mathbf{\hat{j}_{C}}\right)"
    assert extended_latex(2**C.x * d) == r"2^{x_{\text{C}}}\left(\mathbf{\hat{i}_{C}}\otimes\mathbf{\hat{j}_{C}}\right)"
    assert extended_latex((C.x + 2) * d) == r"\left(x_{\text{C}} + 2\right)\left(\mathbf{\hat{i}_{C}}\otimes\mathbf{\hat{j}_{C}}\right)"


def test_vector_base_scalar():
    C = CoordSys3D("C")
    S = C.create_new("Sph", transformation="spherical")

    assert extended_latex(C.x, base_scalar_style="legacy") == r"\mathbf{{x}_{C}}"
    assert extended_latex(C.x, base_scalar_style="normal") == r"x_{\text{C}}"
    assert extended_latex(C.x, base_scalar_style="bold") == r"\boldsymbol{x}_{\textbf{C}}"

    assert extended_latex(S.theta, base_scalar_style="legacy") == r"\mathbf{{theta}_{Sph}}"
    assert extended_latex(S.theta, base_scalar_style="normal") == r"\theta_{\text{Sph}}"
    assert extended_latex(S.theta, base_scalar_style="bold") == r"\boldsymbol{\theta}_{\textbf{Sph}}"


def test_vector_base_vector():
    C = CoordSys3D("C")
    S = C.create_new("S", transformation="spherical")

    assert extended_latex(C.i, base_vector_style="legacy") == r"\mathbf{\hat{i}_{C}}"
    assert extended_latex(C.i, base_vector_style="ijk") == r"\mathbf{\hat{i}_{C}}"
    assert extended_latex(C.i, base_vector_style="ijk-ns") == r"\mathbf{\hat{i}}"
    assert extended_latex(C.i, base_vector_style="e") == r"\mathbf{\hat{e}}^{\left(\text{C}\right)}_{\boldsymbol{x}}"
    assert extended_latex(C.i, base_vector_style="e-ns") == r"\mathbf{\hat{e}}_{\boldsymbol{x}}"
    assert extended_latex(C.i, base_vector_style="system") == r"\mathbf{\hat{c}}_{\boldsymbol{x}}"
    assert extended_latex(C.j, base_vector_style="system") == r"\mathbf{\hat{c}}_{\boldsymbol{y}}"
    assert extended_latex(C.k, base_vector_style="system") == r"\mathbf{\hat{c}}_{\boldsymbol{z}}"

    assert extended_latex(S.i, base_vector_style="legacy") == r"\mathbf{\hat{i}_{S}}"
    assert extended_latex(S.i, base_vector_style="ijk") == r"\mathbf{\hat{i}_{S}}"
    assert extended_latex(S.i, base_vector_style="ijk-ns") == r"\mathbf{\hat{i}}"
    assert extended_latex(S.i, base_vector_style="e") == r"\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}}"
    assert extended_latex(S.i, base_vector_style="e-ns") == r"\mathbf{\hat{e}}_{\boldsymbol{r}}"
    assert extended_latex(S.i, base_vector_style="system") == r"\mathbf{\hat{s}}_{\boldsymbol{r}}"


def test_vector_add_mul():
    C = CoordSys3D("C")
    S = C.create_new("S", transformation="spherical")
    v = C.i + 2*C.j + 3*C.k - S.i + S.r**2 * S.j - (2 - S.r) / (S.r) * S.k

    assert extended_latex(v, base_vector_style="legacy") == r"\mathbf{\hat{i}_{C}} + \left(2\right)\mathbf{\hat{j}_{C}} + \left(3\right)\mathbf{\hat{k}_{C}} - \mathbf{\hat{i}_{S}} + \left(r_{\text{S}}^{2}\right)\mathbf{\hat{j}_{S}} + \left(- \frac{2 - r_{\text{S}}}{r_{\text{S}}}\right)\mathbf{\hat{k}_{S}}"
    assert extended_latex(v, base_vector_style="ijk") == r"\mathbf{\hat{i}_{C}} + 2\,\mathbf{\hat{j}_{C}} + 3\,\mathbf{\hat{k}_{C}} - \mathbf{\hat{i}_{S}} + r_{\text{S}}^{2}\,\mathbf{\hat{j}_{S}} -  \frac{2 - r_{\text{S}}}{r_{\text{S}}}\,\mathbf{\hat{k}_{S}}"
    assert extended_latex(v, base_vector_style="e") == r"\mathbf{\hat{e}}^{\left(\text{C}\right)}_{\boldsymbol{x}} + 2\,\mathbf{\hat{e}}^{\left(\text{C}\right)}_{\boldsymbol{y}} + 3\,\mathbf{\hat{e}}^{\left(\text{C}\right)}_{\boldsymbol{z}} - \mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{r}} + r_{\text{S}}^{2}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\theta}} -  \frac{2 - r_{\text{S}}}{r_{\text{S}}}\,\mathbf{\hat{e}}^{\left(\text{S}\right)}_{\boldsymbol{\phi}}"
    assert extended_latex(v, base_vector_style="system") == r"\mathbf{\hat{c}}_{\boldsymbol{x}} + 2\,\mathbf{\hat{c}}_{\boldsymbol{y}} + 3\,\mathbf{\hat{c}}_{\boldsymbol{z}} - \mathbf{\hat{s}}_{\boldsymbol{r}} + r_{\text{S}}^{2}\,\mathbf{\hat{s}}_{\boldsymbol{\theta}} -  \frac{2 - r_{\text{S}}}{r_{\text{S}}}\,\mathbf{\hat{s}}_{\boldsymbol{\phi}}"
    assert extended_latex(v, vector="matrix") == r"\begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix}_{\text{C}} + \begin{bmatrix}-1 \\ r_{\text{S}}^{2} \\ - \frac{2 - r_{\text{S}}}{r_{\text{S}}}\end{bmatrix}_{\text{S}}"
    assert extended_latex(v, vector="matrix-ns") == r"\begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix} + \begin{bmatrix}-1 \\ r_{\text{S}}^{2} \\ - \frac{2 - r_{\text{S}}}{r_{\text{S}}}\end{bmatrix}"
    assert extended_latex(3 * C.j, vector="matrix") == r"\begin{bmatrix}0 \\ 3 \\ 0\end{bmatrix}_{\text{C}}"
    assert extended_latex(3 * C.j, vector="matrix-ns") == r"\begin{bmatrix}0 \\ 3 \\ 0\end{bmatrix}"


@pytest.mark.parametrize("setting, value", [
    ("applied_undef_args", "asd"),
    ("derivative", "asd"),
    ("base_scalar_style", "asd"),
    ("base_vector_style", "asd"),
    ("vector", "asd"),
    ("dyadic_style", "asd"),
])
def test_extended_latex_wrong_settings_values(setting, value):
    kwargs = {setting: value}
    pytest.raises(ValueError, lambda: extended_latex(1, **kwargs))



###############################################################################
# THE FOLLOWING TESTS COMES FROM sympy.physics.vector.tests
###############################################################################


a, b, c, t = symbols('a, b, c, t')
alpha, omega, beta = dynamicsymbols('alpha, omega, beta')

A = ReferenceFrame('A')
N = ReferenceFrame('N')

v = a ** 2 * N.x + b * N.y + c * sin(alpha) * N.z
w = alpha * N.x + sin(omega) * N.y + alpha * beta * N.z
ww = alpha * N.x + asin(omega) * N.y - alpha.diff(t) * beta * N.z
o = a/b * N.x + (c+b)/a * N.y + c**2/b * N.z

y = a ** 2 * (N.x | N.y) + b * (N.y | N.y) + c * sin(alpha) * (N.z | N.y)
x = alpha * (N.x | N.x) + sin(omega) * (N.y | N.z) + alpha * beta * (N.z | N.x)
xx = N.x | (-N.y - N.z)
xx2 = N.x | (N.y + N.z)


def vlatex(expr, **settings):
    settings.setdefault("applied_undef_args", None)
    settings.setdefault("derivative", "dot")
    return extended_latex(expr, **settings)


def test_vlatex_printer():
    t = symbols("t")
    r = Function('r')(t)
    assert vlatex(r ** 2) == "r^{2}"
    r2 = Function('r^2')(t)
    assert vlatex(r2.diff(t, 2)) == r'\ddot{r}^{2}'
    ra = Function('r_a')(t)
    assert vlatex(ra.diff(t, 2)) == r'\ddot{r}_{a}'


def test_vector_latex():

    a, b, c, d, omega = symbols('a, b, c, d, omega')

    v = (a ** 2 + b / c) * A.x + sqrt(d) * A.y + cos(omega) * A.z

    assert vlatex(v) == (r'(a^{2} + \frac{b}{c})\mathbf{\hat{a}_x} + '
                         r'\sqrt{d}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\omega \right)}'
                         r'\mathbf{\hat{a}_z}')

    theta, omega, alpha, q = dynamicsymbols('theta, omega, alpha, q')

    v = theta * A.x + omega * omega * A.y + (q * alpha) * A.z

    assert vlatex(v) == (r'\theta\mathbf{\hat{a}_x} + '
                         r'\omega^{2}\mathbf{\hat{a}_y} + '
                         r'\alpha q\mathbf{\hat{a}_z}')

    phi1, phi2, phi3 = dynamicsymbols('phi1, phi2, phi3')
    theta1, theta2, theta3 = symbols('theta1, theta2, theta3')

    v = (sin(theta1) * A.x +
         cos(phi1) * cos(phi2) * A.y +
         cos(theta1 + phi3) * A.z)

    assert vlatex(v) == (r'\sin{\left(\theta_{1} \right)}'
                         r'\mathbf{\hat{a}_x} + \cos{'
                         r'\left(\phi_{1} \right)} \cos{'
                         r'\left(\phi_{2} \right)}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\theta_{1} + '
                         r'\phi_{3} \right)}\mathbf{\hat{a}_z}')

    N = ReferenceFrame('N')

    a, b, c, d, omega = symbols('a, b, c, d, omega')

    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z

    expected = (r'(a^{2} + \frac{b}{c})\mathbf{\hat{n}_x} + '
                r'\sqrt{d}\mathbf{\hat{n}_y} + '
                r'\cos{\left(\omega \right)}'
                r'\mathbf{\hat{n}_z}')

    assert vlatex(v) == expected

    # Try custom unit vectors.

    N = ReferenceFrame('N', latexs=(r'\hat{i}', r'\hat{j}', r'\hat{k}'))

    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z

    expected = (r'(a^{2} + \frac{b}{c})\hat{i} + '
                r'\sqrt{d}\hat{j} + '
                r'\cos{\left(\omega \right)}\hat{k}')
    assert vlatex(v) == expected

    expected = r'\alpha\mathbf{\hat{n}_x} + \operatorname{asin}{\left(\omega ' \
        r'\right)}\mathbf{\hat{n}_y} -  \beta \dot{\alpha}\mathbf{\hat{n}_z}'
    assert vlatex(ww) == expected

    expected = r'- \mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} - ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    assert vlatex(xx) == expected

    expected = r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    assert vlatex(xx2) == expected


def test_vector_latex_arguments():
    assert vlatex(N.x * 3.0, full_prec=False) == r'3.0\mathbf{\hat{n}_x}'
    assert vlatex(N.x * 3.0, full_prec=True) == r'3.00000000000000\mathbf{\hat{n}_x}'


def test_vector_latex_with_functions():

    N = ReferenceFrame('N')

    omega, alpha = dynamicsymbols('omega, alpha')

    v = omega.diff(t) * N.x

    assert vlatex(v) == r'\dot{\omega}\mathbf{\hat{n}_x}'

    v = omega.diff(t) ** alpha * N.x

    assert vlatex(v) == (r'\dot{\omega}^{\alpha}'
                          r'\mathbf{\hat{n}_x}')


def test_dyadic_latex():

    expected = (r'a^{2}\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + '
                r'b\mathbf{\hat{n}_y}\otimes \mathbf{\hat{n}_y} + '
                r'c \sin{\left(\alpha \right)}'
                r'\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_y}')

    assert vlatex(y) == expected

    expected = (r'\alpha\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_x} + '
                r'\sin{\left(\omega \right)}\mathbf{\hat{n}_y}'
                r'\otimes \mathbf{\hat{n}_z} + '
                r'\alpha \beta\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_x}')

    assert vlatex(x) == expected

    assert vlatex(Dyadic([])) == '0'


def test_issue_12078():
    x = symbols('x')
    J = symbols('J')

    f = Function('f')
    g = Function('g')
    h = Function('h')

    # NOTE: this test changed in comparison to sympy.physics.vector.printing
    # because that printer is going to print: J \dot{d} - J \dot{h}
    expected = r'J \frac{d}{d x} \left(g - h\right)'

    expr = J*f(x).diff(x).subs(f(x), g(x)-h(x))

    assert vlatex(expr) == expected



def test_vector_derivative_printing():
    # First order
    v = omega.diff(t) * N.x
    assert vlatex(v) == r'\dot{\omega}\mathbf{\hat{n}_x}'

    # Second order
    v = omega.diff(t, 2) * N.x
    assert vlatex(v) == r'\ddot{\omega}\mathbf{\hat{n}_x}'

    # Third order
    v = omega.diff(t, 3) * N.x
    assert vlatex(v) == r'\dddot{\omega}\mathbf{\hat{n}_x}'

    # Fourth order
    v = omega.diff(t, 4) * N.x
    assert vlatex(v) == r'\ddddot{\omega}\mathbf{\hat{n}_x}'

    # Fifth order
    v = omega.diff(t, 5) * N.x
    assert vlatex(v) == r'\frac{d^{5} \omega}{d t^{5}}\mathbf{\hat{n}_x}'


def test_issue_14041():
    A_frame = ReferenceFrame('A')
    thetad, phid = dynamicsymbols('theta, phi', 1)
    L = symbols('L')

    assert vlatex(L*(phid + thetad)**2*A_frame.x) == \
        r"L \left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    assert vlatex((phid + thetad)**2*A_frame.x) == \
        r"\left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    assert vlatex((phid*thetad)**a*A_frame.x) == \
        r"\left(\dot{\phi} \dot{\theta}\right)^{a}\mathbf{\hat{a}_x}"
