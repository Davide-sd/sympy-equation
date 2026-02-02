import param
from dataclasses import dataclass
from sympy.abc import greeks
from sympy.printing.latex import LatexPrinter
from sympy.core.function import AppliedUndef
from sympy.printing.conventions import requires_partial
from sympy.printing.latex import tex_greek_dictionary
from sympy import Symbol, Derivative, Pow, Expr, Mul, Basic
from sympy_equation.printing.utils import print_function
from sympy_equation._doc_utils import add_parameters_to_docstring
from typing import Callable, Any, Mapping
import re


greek_letters_set = frozenset(greeks)


_function_pattern = re.compile(
    r'''
    (?P<name> # function name, like rho, f, \bar{\theta}
        \\[A-Za-z]+(?:\{[^{}]*\})?    # LaTeX macro, optionally with one {arg}
        | [A-Za-z]                    # OR a single Latin letter
    )

    ( # optional (sub,sup) pair in any order
        (?:_\{(?P<sub1>.*?)\}\^\{(?P<sup1>.*?)\})   # _{sub} ^{sup}
      | (?:\^\{(?P<sup2>.*?)\}_\{(?P<sub2>.*?)\})   # ^{sup} _{sub}
      | (?:_\{(?P<sub3>.*?)\})                      # _{sub}
      | (?:\^\{(?P<sup3>.*?)\})                     # ^{sup}
    )?

    (?: (?P<args>\{.*\}) )?               # optional argument block
    ''',
    re.VERBOSE
)


def _extract_function_components(match):
    name, sub, sup, args = None, None, None, None
    sub_groups = [f"sub{i}" for i in range(1, 4)]
    sup_groups = [f"sup{i}" for i in range(1, 4)]

    def pick_group(names):
        for name in names:
            v = match.group(name)
            if v is not None:
                return v
        return None

    name = match.group("name")
    sub = pick_group(sub_groups)
    sup = pick_group(sup_groups)
    args = match.group("args")

    return name, sub, sup, args


def _int_to_roman(num):
    roman_map = {
        1000: "M",
        900:  "CM",
        500:  "D",
        400:  "CD",
        100:  "C",
        90:   "XC",
        50:   "L",
        40:   "XL",
        10:   "X",
        9:    "IX",
        5:    "V",
        4:    "IV",
        1:    "I"
    }

    result = ""
    for value, symbol in roman_map.items():
        if num == 0:
            break
        count, num = divmod(num, value)
        result += symbol * count
    return result


@dataclass
class OverrideRule:
    matches: Callable[[Expr], bool]
    settings: Mapping[str, Any]
    applied_to: Expr


_new_default_settings = LatexPrinter._get_initial_settings()
_new_default_settings.update({
    "applied_undef_args": "all",
    "derivative": None,
    "base_scalar_style": "normal",
    "base_vector_style": "ijk",
    "dyadic_style": "otimes",
    "vector": "legacy",
})


def _validate_setting(settings, name, allowed_values):
    if name in settings:
        if settings[name] not in allowed_values:
            raise ValueError(
                "f`{name}` must be one of the following values:"
                f" {allowed_values}. Instead,"
                f" '{settings[name]}' was received."
            )


class _PrinterSettings(param.Parameterized):
    applied_undef_args = param.Selector(
        default="all",
        objects=[None, True, False, "all", "first-level"], doc="""
        Strategy to represent the arguments of applied undefined functions.
        It can be:

        * ``"all"`` or ``True``: all arguments will be shown.
        * ``"first-level"``: consider f(x, g(x, y)). When this option is set,
            the rendered function will look like f(x, g).
        * ``False`` or ``None``: no arguments will be shown.""")
    base_scalar_style = param.Selector(
        default="normal",
        objects=["legacy", "normal", "normal-ns", "bold", "bold-ns"], doc="""
        Controls how to render base scalars from the sympy.vector module.
        It can be:

        * ``"legacy"``: use standard SymPy's latex printer. Base scalars
          are rendered in bold font.
        * ``"normal"`` (default): rendered as 'symbol_{system}'. No bold font.
        * ``"normal-ns"``: rendered as  'symbol'. No bold font, no system.
        * ``"bold"``: rendered as 'symbol_{system}' using bold font.
        * ``"bold-ns"``: rendered as 'symbol' using bold font.""")
    base_vector_style = param.Selector(
        default="auto",
        objects=["legacy", "ijk", "ijk-ns", "e", "e-ns", "system", "auto"],
        doc=r"""
        Controls how to render base vectors and vectors from the
        sympy.vector module, when the option ``vector="legacy"``. It can be:

        * ``"auto"``: uses 'ijk' for Cartesian systems and 'e' for the
          other systems.
        * ``"legacy"``: use standard SymPy's latex printer. Base vectors are
          rendered as '\hat{i}_{system}, \hat{j}_{system}, \hat{k}_{system}'.
          Any coefficient to these base vectors are wrapped in parenthesis.
        * ``"ijk"``: similar to ``"legacy"``, but the coefficients
          won't be wrapped in parenthesis, unless strictly required.
        * ``"ijk-ns"``: no system is shown, '\hat{i}, \hat{j}, \hat{k}'.
          This is useful if we are working with only one cartesian system.
        * ``"e"``: '\hat{e}_{system, base scalar}'.
        * ``"e-ns"``: no system is shown, '\hat{e}_{base scalar}'. This is
          useful if we are working with only one curvilinear system.
        * ``"system"``: `\hat{system}_{base scalar}`.""")
    derivative = param.Selector(
        default=None,
        objects = [
            None, "subscript", "prime-arabic", "prime-roman", "dot",
            "d-notation"], doc=r"""
        Stategy to represent derivatives of applied undefined functions.
        It can be:

        * None (default value): standard notation, for example df/dx.
        * ``"prime-arabic"``: Lagrange's notation for derivatives, which only
          works for functions with one argument. For example:

          * $\frac{d f}{d x} \rightarrow f'$
          * $\frac{d^{2} f}{d x^2} \rightarrow f''$
          * $\frac{d^{3} f}{d x^3} \rightarrow f'''$
          * $\frac{d^{4} f}{d x^4} \rightarrow f^{(4)}$

        * ``"prime-roman"``: Lagrange's notation for derivatives, which only 
          works for functions with one argument. For example:

          * $\frac{d f}{d x} \rightarrow f'$
          * $\frac{d^{2} f}{d x^2} \rightarrow f''$
          * $\frac{d^{3} f}{d x^3} \rightarrow f'''$
          * $\frac{d^{4} f}{d x^4} \rightarrow f^{iv}$

        * ``"dot"``: Newton's notation for time derivatives using dots, 
          for example $\frac{d f}{d t} \rightarrow \dot{f}$. It only works if
          the time symbol is constructed with no assumptions.

        * ``"subscript"``: add subscripts on the right side of the function
          name. For example:

          * $\frac{d f}{d x} \rightarrow (f)_{x}$
          * $\frac{d^{2} f}{d x^2} \rightarrow (f)_{xx}$
          * $\frac{\partial^{2} f}{\partial x \partial y} \rightarrow (f)_{xy}$

        * ``"d-notation"``: add subscripts on the right side of derivative
          short syntax. For example:

          * $\frac{d f(x)}{d x} \rightarrow  D_{x} f$
          * $\frac{d^{2} f(x)}{d x^2} \rightarrow D_{x}^{2} f$
          * $\frac{\partial^{2} f(x, y)}{\partial x \partial y} \rightarrow \partial_{xy} f$""")
    full_prec = param.Boolean(False, doc="""
        If set to True, a floating point number is printed with
        full precision.""")
    fold_frac_powers = param.Boolean(False, doc=r"""
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional
        powers.""")
    fold_func_brackets = param.Boolean(False, doc="""
        Fold function brackets where applicable.""")
    fold_short_frac = param.Boolean(False, doc=r"""
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.""")
    inv_trig_style = param.Selector(default="abbreviated", objects=[
        "abbreviated", "full", "power"], doc="""
        How inverse trig functions should be displayed.""")
    itex = param.Boolean(False, doc="""
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.""")
    ln_notation = param.Boolean(False, doc=r"""
        If set to ``True``, ``\ln`` is used instead of default ``\log``.""")
    long_frac_ratio = param.Integer(None, bounds=(0, None), doc="""
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.""")
    mat_delim = param.Selector(default="[", objects=["[", "(", "", None], doc="""
        The delimiter to wrap around matrices.""")
    mat_str = param.Selector(default=None, objects=[
        "smallmatrix", "matrix", "bmatrix", "array"], doc="""
        Which matrix environment string to emit. Defaults to ``'smallmatrix'``
        for inline mode, ``'matrix'`` for matrices of no more than 10 columns,
        and ``'array'`` otherwise.""")
    mode = param.Selector(default="plain", objects=[
        "plain", "inline", "equation", "equation*"], doc="""
        Specifies how the generated code will be delimited. 
        If ``mode='plain'``, then the resulting code will not be
        delimited at all (this is the default). If ``mode='inline'`` then
        inline LaTeX ``$...$`` will be used. If ``mode='equation'`` or
        ``mode='equation*'``, the resulting code will be enclosed in the
        ``equation`` or ``equation*`` environment (remember to import
        ``amsmath`` for ``equation*``), unless the ``itex`` option is
        set. In the latter case, the ``$$...$$`` syntax is used.""")
    mul_symbol = param.String(
        default=None, doc="""
        The symbol to use for multiplication. Possible options are:
        None, 'ldot', 'dot', 'times'.""")
    order = param.Selector(default=None, objects=[
        None, "lex", "grlex", "grevlex", "rev-lex", "old", "none"], doc="""
        This parameter does nothing for `~.Mul` objects, but it reorders
        monomials in `~.Add`. Setting ``order='old'`` uses the compatibility
        ordering for ``~.Add`` defined in Printer. For very large expressions,
        set the ``order`` keyword to ``'none'`` if speed is a concern.""")
    symbol_names = param.Dict(default={}, doc="""
        Dictionary of symbols and the custom strings they should
        be emitted as.""")
    root_notation = param.Boolean(True, doc="""
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.""")
    mat_symbol_style = param.Selector(default="plain",
        objects=["plain", "bold"], doc=r"""
        Can be either ``'plain'`` (default) or ``'bold'``. If set to
        ``'bold'``, a `~.MatrixSymbol` A will be printed as ``\mathbf{A}``,
        otherwise as ``A``.""")
    imaginary_unit = param.String(
        default="i", doc=r"""
        String to use for the imaginary unit. Defined options are ``'i'``
        (default) and ``'j'``. Adding ``r`` or ``t`` in front gives ``\mathrm``
        or ``\text``, so ``'ri'`` leads to ``\mathrm{i}``.""")
    gothic_re_im = param.Boolean(False, doc=r"""
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``,
        respectively. The default is ``False`` leading to `\operatorname{re}`
        and `\operatorname{im}`.""")
    decimal_separator = param.Selector(default="period", objects=[
        "period", "comma"], doc="""
        Specifies what separator to use to separate the whole and fractional
        parts of a floating point number as in `2.5` for the default,
        ``period`` or `2{,}5` when ``comma`` is specified. Lists, sets, and
        tuple are printed with semicolon separating the elements when ``comma``
        is chosen. For example, [1; 2; 3] when ``comma`` is chosen and [1,2,3]
        for when ``period`` is chosen.""")
    perm_cyclic = param.Boolean(True)
    min = param.Integer(None, doc="""
        Sets the lower bound for the exponent to print floating point numbers
        in fixed-point format.""")
    max = param.Integer(None, doc="""
        Sets the upper bound for the exponent to print floating point numbers
        in fixed-point format.""")
    diff_operator = param.Selector(default="d", objects=["d", "rd", "td"], doc=r"""
        String to use for differential operator. Default is ``'d'``, to print
        in italic form. ``'rd'``, ``'td'`` are shortcuts for ``\mathrm{d}`` and ``\text{d}``.""")
    adjoint_style = param.Selector(default="dagger", objects=[
        "dagger", "star", "hermitian"], doc="""
        String to use for the adjoint symbol.""")
    disable_split_super_sub = param.Boolean(False)
    vector = param.Selector(
        default="legacy", objects=["legacy", "matrix", "matrix-ns"], doc="""
        Controls how to render vectors from the sympy.vector module.
        It can be:

        * ``"legacy"``: vectors are rendered as linear combination between
          terms and base vectors.
        * ``"matrix"``: vectors are rendered as 3x1 matrices.
        * ``"matrix-ns"``: vectors are rendered as 3x1 matrices without the
          label indicating the system.""")
    dyadic_style = param.Selector(
        default="otimes", objects=["none", "vline", "otimes"], doc="""
        The symbol denoting a dyadic.""")
    colorize = param.Dict(default={}, doc="""
        Map sub-expressions to specific latex colors.""")
    print_builtin = param.Boolean(True, doc="""
        If ``True`` then floats and integers will be printed. If ``False`` the
        printer will only print SymPy types.""")

    def __init__(self, **params):
        # sadly, there is a method called 'parenthesize_super', hence
        # I can't define a parameter for this name
        parenthesize_super = params.pop("parenthesize_super", True)
        super().__init__(**params)
        self._further_settings = {
            "parenthesize_super": parenthesize_super,
            # these values will be set later
            "mul_symbol_latex": None,
            "mul_symbol_latex_numbers": None,
            "imaginary_unit_latex": None,
            "diff_operator_latex": None,
        }
        self._update_mul_symbols()
        self._update_imaginary_unit()
        self._update_diff_operator()

    @property
    def _settings(self):
        d = self.param.values().copy()
        d.update(self._further_settings)
        return d

    @param.depends("mul_symbol", watch=True)
    def _update_mul_symbols(self):
        mul_symbol_table = {
            None: r" ",
            "ldot": r" \,.\, ",
            "dot": r" \cdot ",
            "times": r" \times "
        }
        self._further_settings["mul_symbol_latex"] = mul_symbol_table.get(
            self.mul_symbol, self.mul_symbol)

        if self.mul_symbol in mul_symbol_table:
            val = mul_symbol_table[self.mul_symbol or 'dot']
        elif (
            self.mul_symbol.strip() in [
                '', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']
        ):
            val = mul_symbol_table['dot']
        else:
            val = self.mul_symbol
        self._further_settings['mul_symbol_latex_numbers'] = val

    @param.depends("imaginary_unit", watch=True)
    def _update_imaginary_unit(self):
        imaginary_unit_table = {
            None: r"i",
            "i": r"i",
            "ri": r"\mathrm{i}",
            "ti": r"\text{i}",
            "j": r"j",
            "rj": r"\mathrm{j}",
            "tj": r"\text{j}",
        }
        val = imaginary_unit_table.get(self.imaginary_unit, self.imaginary_unit)
        self._further_settings['imaginary_unit_latex'] = val

    @param.depends("diff_operator", watch=True)
    def _update_diff_operator(self):
        diff_operator_table = {
            None: r"d",
            "d": r"d",
            "rd": r"\mathrm{d}",
            "td": r"\text{d}",
        }
        diff_operator = self.diff_operator
        self._further_settings["diff_operator_latex"] = diff_operator_table.get(diff_operator, diff_operator)


class ExtendedLatexPrinter(_PrinterSettings, LatexPrinter):
    r""" 
    Extended Latex printer with new options and ability to set customization
    rules for selected types of symbolic expressions.

    References
    ----------

    * https://en.wikipedia.org/wiki/Notation_for_differentiation
    * https://en.wikibooks.org/wiki/LaTeX/Colors
    """

    def __init__(self, **params):
        super().__init__(**params)

        if (self.fold_short_frac is None) and (self.mode == 'inline'):
            self.fold_short_frac = True

        self.override_rules = []
        self._str = str
        self._delim_dict = {'(': ')', '[': ']'}
        # _print_level is the number of times self._print() was recursively
        # called. See StrPrinter._print_Float() for an example of usage
        self._print_level = 0

    def _print(self, expr, **kwargs) -> str:
        if isinstance(expr, (int, float)) and (not self.print_builtin):
            return ""

        res = super()._print(expr, **kwargs)
        if isinstance(expr, Expr) and (expr in self.colorize):
            res = r"\textcolor{%s}{%s}" % (self.colorize[expr], res)
        return res

    def add_rule(self, expr, **settings):
        r"""Add a customization rule acting on the provided symbolic expression.

        See Also
        --------
        remove_rule, show_rules

        Examples
        --------

        >>> from sympy import *
        >>> from sympy_equation.printing.extended_latex import ExtendedLatexPrinter
        >>> x, y, t = symbols("x y t")
        >>> f = Function("f")(x, y)
        >>> g = Function("g")(t)
        >>> expr = f.diff(x, 2) + g.diff(t)
        >>> printer = ExtendedLatexPrinter()
        >>> print(printer.doprint(expr))
        \frac{\partial^{2}}{\partial x^{2}} f{\left(x,y \right)} + \frac{d}{d t} g{\left(t \right)}

        ``g`` is a function of time. Let's hide the argument, and set the time
        derivative to be shown with dot notation:

        >>> printer.add_rule(g, derivative="dot", applied_undef_args=None)
        >>> print(printer.doprint(expr))
        \frac{\partial^{2}}{\partial x^{2}} f{\left(x,y \right)} + \dot{g}

        >>> printer.show_rules()
        [0] g(t)     {'derivative': 'dot'}
        [1] g(t)     {'applied_undef_args': None}

        """
        if not isinstance(expr, Basic):
            raise TypeError("`expr` must be a symbolic expression.")

        # remove old rules that are going to be overwritten
        rules_to_remove = []
        for r in self.override_rules:
            if r.applied_to == expr:
                if any(k in settings for k in r.settings):
                    rules_to_remove.append(r)

        for r in rules_to_remove:
            self.override_rules.remove(r)

        new_rules = []
        if isinstance(expr, AppliedUndef) and ("derivative" in settings):
            derivative_pattern = lambda current_expr: (
                isinstance(current_expr, Derivative)
                and (current_expr.expr == expr)
            )
            new_rules.append(OverrideRule(
                matches=derivative_pattern,
                settings={"derivative": settings.pop("derivative")},
                applied_to=expr
            ))
        elif type(expr).__name__ == "CoordSys3D":
            from sympy.vector import BaseVector, BaseScalar
            base_pattern = lambda current_expr: (
                isinstance(current_expr, (BaseVector, BaseScalar))
                and (current_expr.args[1] == expr)
            )
            new_rules.append(OverrideRule(
                matches=base_pattern,
                settings=settings,
                applied_to=expr
            ))
            settings = {}

        if len(settings) > 0:
            expr_pattern = lambda current_expr: current_expr == expr
            new_rules.append(OverrideRule(
                matches=expr_pattern, settings=settings, applied_to=expr))

        self.override_rules.extend(new_rules)

    def remove_rule(self, rule):
        """Remove a customization rule, specified by an index.

        See Also
        --------
        show_rules, add_rule
        """
        if not isinstance(rule, list):
            rule = [rule]

        rules_to_remove = []
        for r in rule:
            if not isinstance(r, (int, OverrideRule)):
                raise TypeError(
                    "`rule` must be an instance of `int` or `OverrideRule`."
                    f" Instead, type {type(rule).__name__} was received."
                )

            if isinstance(r, int):
                if (r >= 0) and (r < len(self.override_rules)):
                    rules_to_remove.append(self.override_rules[r])
            elif r in self.override_rules:
                rules_to_remove.append(r)

        for r in rules_to_remove:
            self.override_rules.remove(r)

    def show_rules(self):
        """Show all customization rules applied to target symbolic
        expressions.

        See Also
        --------
        add_rule, remove_rule
        """
        if len(self.override_rules) == 0:
            print("No rules yet.")
            return

        index_width = len(str(len(self.override_rules) - 1))
        applied_to_width = max(
            len(str(r.applied_to)) for r in self.override_rules) + 4
        for i, rule in enumerate(self.override_rules):
            idx = f"[{i:^{index_width}}]"
            a = f"{str(rule.applied_to):<{applied_to_width}}"
            b = f"{rule.settings}"
            print(f"{idx} {a} {b}")

    def _print_Pow(self, expr: Pow):
        if (
            isinstance(expr.base, Derivative)
            and isinstance(expr.base.expr, AppliedUndef)
            and (self._settings["derivative"] == "subscript")
        ):
            func = self._print(expr.base)
            match = _function_pattern.fullmatch(func)
            if match:
                name, sub, sup, args = _extract_function_components(match)
                # NOTE: consider f(t). Then
                # * df/dt=f_t
                # * (df/dt)**2 = f_t^2
                # This looks good and it's readable.
                # Now consider p_1^0(t). Then:
                # * dp_1^0/dt=p_{1,t}^0
                # * (dp_1^0/dt)**2=p_{1,t}^{0, 2}
                # The exponent looks rather confusing. It is easier to read:
                # (p_{1,t}^0)**2
                if not sup:
                    if sub:
                        name += "_{%s}" % sub
                    name += "^{%s}" % self._print(expr.exp)
                    if args:
                        name += args
                    return name

        return super()._print_Pow(expr)

    def _print_Function(self, expr, exp=None):
        if isinstance(expr, AppliedUndef):
            applied_undef_args = self._get_setting_for(
                expr, "applied_undef_args")

            if applied_undef_args in [False, None, "first-level"]:
                new_f = self._print(Symbol(expr.func.__name__))
                if exp:
                    new_f = f"{new_f}^{{{exp}}}"

                if not applied_undef_args:
                    return new_f
                elif (applied_undef_args == "first-level"):
                    args = []
                    for a in expr.args:
                        if isinstance(a, AppliedUndef):
                            a = Symbol(a.func.__name__)
                        args.append(a)

                    args = [self._print(a) for a in args]
                    return new_f + fr"{{\left({','.join(args)}\right)}}"

        return super()._print_Function(expr, exp)

    def _print_Derivative(self, expr):
        if isinstance(expr.expr, AppliedUndef):
            mapping = {
                "subscript": self._subscript_derivative,
                "dot": self._dot_derivative,
                "d-notation": self._d_notation_derivative,
                "prime-arabic": lambda e: self._prime_derivative(e, "prime-arabic"),
                "prime-roman": lambda e: self._prime_derivative(e, "prime-roman"),
            }

            derivative_style = self._get_setting_for(expr, "derivative")
            if derivative_style:
                func = mapping[derivative_style]
                res = func(expr)
                if res:
                    return res

        if requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = []
        dim = 0
        for x, num in expr.variable_count:
            dim += num
            if num == 1:
                tex.append(r"%s" % self._print(x))
            else:
                tex.append(r"%s^{%s}" % (self.parenthesize_super(self._print(x)),
                                        self._print(num)))

        tex = diff_symbol + " " + (diff_symbol + " ").join(reversed(tex))

        if isinstance(expr.expr, AppliedUndef):
            applied_undef_args = self._get_setting_for(
                expr.expr, "applied_undef_args")
            if not applied_undef_args:
                if dim == 1:
                    return r"\frac{%s %s}{%s}" % (diff_symbol, self._print(expr.expr), tex)
                else:
                    return r"\frac{%s^{%s} %s}{%s}" % (diff_symbol, self._print(dim), self._print(expr.expr), tex)

        return super()._print_Derivative(expr)

    def _subscript_derivative(self, der_expr):
        base = self._print(der_expr.expr)
        match = _function_pattern.fullmatch(base)
        if not match:
            return None

        f, sub, sup, args = _extract_function_components(match)
        subscripts = "".join([
            self._print(symb) * n for (symb, n) in der_expr.args[1:]
        ])

        if sub:
            f += "_{%s,%s}" % (sub, subscripts)
        else:
            f += "_{%s}" % subscripts
        if sup:
            f += "^{%s}" % sup
        if args:
            f += args

        return f

    def _d_notation_derivative(self, der_expr):
        if (
            (len(der_expr.expr.args) == 1)
            and (len(set(der_expr.variables)) == 1)
        ):
            symb, n = der_expr.args[1]
            d_symbol = "D_{%s}" % self._print(symb)
            if n > 1:
                d_symbol += "^{%s}" % n
        else:
            subscripts = "".join([
                self._print(symb) * n for (symb, n) in der_expr.args[1:]
            ])
            d_symbol = r"\partial_{%s}" % subscripts

        return d_symbol + self._print(der_expr.expr)

    def _dot_derivative(self, der_expr):
        # check if expr is a dynamicsymbol
        t = Symbol("t")
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        test1 = not all(True for i in red if i.free_symbols == {t})
        test2 = not all(t == i for i in syms)
        if test1 or test2:
            return None

        # done checking
        dots = len(syms)
        base = self._print(expr)
        match = _function_pattern.fullmatch(base)
        if not match:
            return None

        base, sub, sup, args = _extract_function_components(match)

        if dots == 1:
            base = r"\dot{%s}" % base
        elif dots == 2:
            base = r"\ddot{%s}" % base
        elif dots == 3:
            base = r"\dddot{%s}" % base
        elif dots == 4:
            base = r"\ddddot{%s}" % base
        else:  # Fallback to standard printing
            return None

        if sub:
            base += "_{%s}" % sub
        if sup:
            base += "^{%s}" % sup
        if args:
            base += args

        return base

    def _prime_derivative(self, der_expr, user_pref):
        f = der_expr.expr
        if not isinstance(f, AppliedUndef):
            return None
        if len(f.args) > 1:
            return None

        base = self._print(f)
        match = _function_pattern.fullmatch(base)
        if not match:
            return None

        base, sub, sup, args = _extract_function_components(match)

        n = len(der_expr.variables)
        if n <= 3:
            exp = r"\prime" * n
        else:
            n = n if user_pref == "prime-arabic" else _int_to_roman(n).lower()
            exp = r"\left(%s\right)" % n

        if sub:
            base += "_{%s}" % sub
        if sup:
            base += "^{%s,%s}" % (sup, exp)
        else:
            base += f"^{{{exp}}}"
        if args:
            base += args

        return base

    def _print_BaseDyadic(self, expr):
        a1, a2 = [self._print(a) for a in expr.args]
        if self._settings["dyadic_style"] == "otimes":
            symbol = r"\otimes"
        elif self._settings["dyadic_style"] == "vline":
            symbol = r"\middle|"
        else:
            symbol = ""
        return fr"\left({a1}{symbol}{a2}\right)"

    def _print_DyadicMul(self, expr):
        s, d = Mul(*expr.args[:-1]), expr.args[-1]
        s_latex = self._print(s)
        d_latex = self._print(d)
        if s.is_Add:
            s_latex = fr"\left({s_latex}\right)"
        return f"{s_latex}{d_latex}"

    def _print_BaseScalar(self, expr):
        base_scalar_style = self._get_setting_for(expr, "base_scalar_style")
        if base_scalar_style == "legacy":
            return expr._latex_form

        coord_sys = expr._system._name
        name = expr.name.split(".")[1]
        if name in greek_letters_set:
            name = r"\%s" % name
        elif name in tex_greek_dictionary:
            name = tex_greek_dictionary[name]

        if base_scalar_style == "normal":
            return r"%s_{\text{%s}}" % (name, coord_sys)
        elif base_scalar_style == "normal-ns":
            return r"%s" % name
        elif base_scalar_style == "bold":
            return r"\boldsymbol{%s}_{\textbf{%s}}" % (name, coord_sys)
        return r"\boldsymbol{%s}" % name

    def _get_setting_for(self, expr, k):
        i = 0
        while i < len(self.override_rules):
            if (
                (k in self.override_rules[i].settings)
                and self.override_rules[i].matches(expr)
            ):
                return self.override_rules[i].settings[k]
            i += 1

        return self._settings[k]

    def _print_BaseVector(self, expr):
        base_vector_style = self._get_setting_for(expr, "base_vector_style")
        idx, sys = expr.args

        if base_vector_style == "auto":
            if sys._is_cartesian:
                base_vector_style = "ijk"
            else:
                base_vector_style = "e"
        if base_vector_style in ["legacy", "ijk"]:
            return expr._latex_form

        system_name = sys._name
        vector_name = sys.base_vectors()[idx]._name.split(".")[1]
        scalar_name = sys.base_scalars()[idx].name.split(".")[1]
        if scalar_name in greek_letters_set:
            scalar_name = r"\%s" % scalar_name
        elif scalar_name in tex_greek_dictionary:
            scalar_name = tex_greek_dictionary[scalar_name]

        if base_vector_style == "ijk-ns":
            return r"\mathbf{\hat{%s}}" % vector_name
        if base_vector_style == "e":
            return r"\mathbf{\hat{e}}^{\left(\text{%s}\right)}_{\boldsymbol{%s}}" % (system_name, scalar_name)
        if base_vector_style == "e-ns":
            return r"\mathbf{\hat{e}}_{\boldsymbol{%s}}" % scalar_name
        return r"\mathbf{\hat{%s}}_{\boldsymbol{%s}}" % (system_name.lower(), scalar_name)

    def _print_BasisDependent(self, expr):
        if self._settings["vector"] in ["matrix", "matrix-ns"]:
            template = r"\begin{bmatrix}%s \\ %s \\ %s\end{bmatrix}_{%s}"
            template_ns = r"\begin{bmatrix}%s \\ %s \\ %s\end{bmatrix}"
            result = ""

            parts = expr.separate()
            for i, (sys, v) in enumerate(parts.items()):
                c1, c2, c3 = [self._print(v & d) for d in sys.base_vectors()]

                if self._settings["vector"] == "matrix-ns":
                    result += template_ns % (c1, c2, c3)
                else:
                    system_name = r"\text{%s}" % sys._name
                    result += template % (c1, c2, c3, system_name)

                if i != len(parts) - 1:
                    result += " + "
            return result

        if self._settings["base_vector_style"] == "legacy":
            return super()._print_BasisDependent(expr)

        from sympy.vector import Vector

        o1: list[str] = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + self._print(k))
                elif v == -1:
                    o1.append(' - ' + self._print(k))
                else:
                    arg_str = self._print(v)
                    if v.is_Add:
                        sign = ' + '
                        arg_str = r"\left(" + arg_str + r"\right)"
                    else:
                        sign = ' + ' if arg_str[0] != "-" else ' - '
                        arg_str = arg_str if arg_str[0] != "-" else arg_str[1:]
                    o1.append(sign + arg_str + r"\," + self._print(k))

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr


@add_parameters_to_docstring(ExtendedLatexPrinter)
@print_function(ExtendedLatexPrinter)
def extended_latex(expr, **settings):
    """Wrapper function to ``ExtendedLatexPrinter``.

    Returns
    -------
    printer : ExtendedLatexPrinter
    """
    return ExtendedLatexPrinter(**settings).doprint(expr)
