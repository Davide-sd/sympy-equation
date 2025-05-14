import param
import sympy
from  sympy import (
    Expr, Basic, Equality, Add,
    Derivative, Integral, fraction
)
from sympy.core.add import _unevaluated_Add
from sympy.core.evalf import EvalfMixin
from sympy.core.sympify import _sympify
from sympy.simplify.radsimp import collect
# NOTE: by executing the following import, the module will automatically
# load the preparser that allows the syntax my_eq =@ lhs = rhs.
from sympy_equation.preparser import integers_as_exact



class Equation(Basic, EvalfMixin):
    """
    This class defines an equation with a left-hand-side (lhs) and a right-
    hand-side (rhs) connected by the "=" operator (e.g. `p*V = n*R*T`),
    which supports mathematical operations like addition, subtraction,
    multiplication, divison and exponentiation.

    In particular, this class is intended to allow using the mathematical
    tools in SymPy to rearrange equations and perform algebra in a stepwise
    fashion.

    Create an equation with the call ``Equation(lhs,rhs)``, where ``lhs`` and
    ``rhs`` are any valid Sympy expression. ``Eqn(...)`` is a synonym for
    ``Equation(...)``.

    Parameters
    ==========
    lhs : ``Expr``
    rhs : ``Expr``

    Attributes
    ==========
    lhs : ``Expr``
        Return the left-hand side of the equation.
    rhs : ``Expr``
        Return the right-hand side of the equation.
    swap : Equation
        Swap the lhs and rhs sides.

    Methods
    =======
    apply(func, *args)
        Apply a function ``func`` with arguments ``*args`` to both sides
        of the equation.
    applylhs(func, *args)
        Apply a function ``func`` with arguments ``*args`` to the lhs of
        the equation.
    applyrhs(func, *args)
        Apply a function ``func`` with arguments ``*args`` to the rhs of
        the equation.
    as_Boolean()
        Convert the ``Equation`` to an ``Equality``.
    as_expr()
        Convert the ``Equation`` to a symbolic expression of the form 'lhs - rhs'.
    check(**kwargs)
        Forces simplification and casts as ``Equality`` to check validity.
    to_expr()
        Alias of ``as_expr()``.
    cross_multiply()
        Given and equation ``Equation(a/b, c/d)``, cross-multiply
        the members in order to get a new ``Equation(a*d, b*c)``.


    Examples
    ========

    >>> from sympy import *
    >>> from sympy_equation import Eqn, Equation
    >>> a, b, c, d, e, x = symbols('a, b, c, d, e, x')

    Mathematical operations between an equation and a scalar value:

    >>> eq = Eqn(a, b/c)
    >>> eq
    Equation(a, b/c)
    >>> eq + d
    Equation(a + d, b/c + d)
    >>> eq - d
    Equation(a - d, b/c - d)
    >>> eq * c
    Equation(a*c, b)
    >>> c * eq
    Equation(a*c, b)
    >>> eq / c
    Equation(a/c, b/c**2)
    >>> c / eq
    Equation(c/a, c**2/b)
    >>> eq ** d
    Equation(a**d, (b/c)**d)
    >>> d ** eq
    Equation(d**a, d**(b/c))

    Mathematical operations between two equations:

    >>> e1 = Eqn(a, b + c)
    >>> e2 = Eqn(e, d - a)
    >>> e1 + e2
    Equation(a + e, -a + b + c + d)
    >>> e1 - e2
    Equation(a - e, a + b + c - d)
    >>> e1 * e2
    Equation(a*e, (-a + d)*(b + c))
    >>> e1 / e2
    Equation(a/e, (b + c)/(-a + d))
    >>> e1 ** e2
    Equation(a**e, (b + c)**(-a + d))

    Apply mathematical functions to the equation:

    >>> eq.apply(exp)
    Equation(exp(a), exp(b/c))
    >>> def add_square(eqn):
    ...     return eqn+eqn**2
    ...
    >>> add_square(eq)
    Equation(a**2 + a, b**2/c**2 + b/c)
    >>> eq.apply(add_square)
    Equation(a**2 + a, b**2/c**2 + b/c)
    >>> eq.applylhs(add_square)
    Equation(a**2 + a, b/c)
    >>> eq.applyrhs(add_square)
    Equation(a, b**2/c**2 + b/c)

    Expression manipulation:

    >>> f = Eqn(x**2 - 1, c)
    >>> f
    Equation(x**2 - 1, c)
    >>> f/(x+1)
    Equation((x**2 - 1)/(x + 1), c/(x + 1))
    >>> (f/(x+1)).simplify()
    Equation(x - 1, c/(x + 1))
    >>> simplify(f/(x+1))
    Equation(x - 1, c/(x + 1))
    >>> (f/(x+1)).expand()
    Equation(x**2/(x + 1) - 1/(x + 1), c/(x + 1))
    >>> expand(f/(x+1))
    Equation(x**2/(x + 1) - 1/(x + 1), c/(x + 1))
    >>> factor(f)
    Equation((x - 1)*(x + 1), c)
    >>> f.factor()
    Equation((x - 1)*(x + 1), c)
    >>> eq3 = Eqn(2 * b + 2 * c, d - b)
    >>> eq3
    Equation(2*b + 2*c, -b + d)
    >>> eq3.applylhs(collect_const, 2)
    Equation(2*(b + c), -b + d)

    In addition to ``.apply...`` there is also the less general ``.do``,
    ``.dolhs``, ``.dorhs``, which only works for operations defined on the
    ``Expr`` class (e.g.``.collect(), .factor(), .expand()``, etc...):

    >>> poly = Eqn(a*x**2 + b*x + c*x**2, a*x**3 + b*x**3 + c*x)
    >>> poly.dolhs.collect(x)
    Equation(b*x + x**2*(a + c), a*x**3 + b*x**3 + c*x)
    >>> poly.dorhs.collect(x)
    Equation(a*x**2 + b*x + c*x**2, c*x + x**3*(a + b))
    >>> poly.do.collect(x)
    Equation(b*x + x**2*(a + c), c*x + x**3*(a + b))
    >>> poly.dorhs.factor()
    Equation(a*x**2 + b*x + c*x**2, x*(a*x**2 + b*x**2 + c))

    Substitutions and numerical evaluation:

    >>> p, V, n, R, T = var('p V n R T')
    >>> L, atm, mol, K = var('L atm mol K', positive=True, real=True) # units
    >>> ideal_gas_law = Eqn(p * V, n * R * T)
    >>> pressure = ideal_gas_law / V
    >>> pressure.subs({R:0.08206*L*atm/mol/K,T:273*K,n:1.00*mol,V:24.0*L})
    Equation(p, 0.9334325*atm)

    Evaluate up to n-digits:

    >>> pressure.evalf(subs={R:0.08206*L*atm/mol/K,T:273*K,n:1.00*mol,V:24.0*L}, n=2)
    Equation(p, 0.93*atm)

    Substituting an equation into another equation:

    >>> e3 = Eqn(a/b, c/d)
    >>> e4 = Eqn(d, (a + b) / c)
    >>> e3.subs(e4)
    Equation(a/b, c**2/(a + b))

    Utility operations:

    >>> eq
    Equation(a, b/c)
    >>> eq.reversed  # or t.swap
    Equation(b/c, a)
    >>> eq.lhs
    a
    >>> eq.rhs
    b/c
    >>> eq.as_Boolean()
    Eq(a, b/c)

    ``.check()`` to verify if the lhs is equal to the rhs. It is a
    convenience method for ``.as_Boolean().simplify()``:

    >>> Equation(pi*(I+2), pi*I+2*pi).check()
    True
    >>> Eqn(a,a+1).check()
    False

    Convert an Equation to an expression of the form ``lhs - rhs``:

    >>> eq.to_expr()
    a - b/c

    Cross multiply members of an equation:

    >>> e5 = Eqn(a/b, c/d)
    >>> e5.cross_multiply()
    Equation(a*d, b*c)
    >>> e6 = Eqn(a/(b+c), 1)
    >>> e6.cross_multiply()
    Equation(a, b + c)

    Differentiation is applied to both sides:

    >>> q=Eqn(a*b, b**2/c**2)
    >>> q
    Equation(a*b, b**2/c**2)
    >>> diff(q,b)
    Equation(a, 2*b/c**2)
    >>> q.diff(c)
    Equation(0, -2*b**2/c**3)

    Integration is applied to both sides:

    >>> q=Eqn(a*c,b/c)
    >>> integrate(q,b)
    Equation(a*b*c, b**2/(2*c))

    Integration of each side with respect to different variables:

    >>> q.dorhs.integrate(b).dolhs.integrate(a)
    Equation(a**2*c/2, b**2/(2*c))

    Solving equations:

    >>> from sympy_equation import solve
    >>> eq = Eqn(a - b, c/a)
    >>> solve(eq,a)
    [Equation(a, b/2 - sqrt(b**2 + 4*c)/2), Equation(a, b/2 + sqrt(b**2 + 4*c)/2)]
    >>> solve(eq, b)
    [Equation(b, (a**2 - c)/a)]
    >>> solve(eq, c)
    [Equation(c, a**2 - a*b)]

    """

    def __new__(cls, lhs, rhs, **kwargs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        if not isinstance(lhs, Expr) or not isinstance(rhs, Expr):
            raise TypeError('lhs and rhs must be valid sympy expressions.')
        return super().__new__(cls, lhs, rhs)

    def _get_eqn_name(self):
        """
        Tries to find the python string name that refers to the equation. In
        IPython environments (IPython, Jupyter, etc...) looks in the user_ns.
        If not in an IPython environment looks in __main__.
        :return: string value if found or empty string.
        """
        import __main__ as shell
        for k in dir(shell):
            item = getattr(shell, k)
            if isinstance(item, Equation):
                if item == self and not k.startswith('_'):
                    return k
        return ''

    @property
    def lhs(self):
        """
        Returns the lhs of the equation.
        """
        return self.args[0]

    @property
    def rhs(self):
        """
        Returns the rhs of the equation.
        """
        return self.args[1]

    def as_Boolean(self):
        """
        Converts the equation to an Equality.
        """
        return Equality(self.lhs, self.rhs)

    def check(self, **kwargs):
        """
        Forces simplification and casts as `Equality` to check validity.
        Parameters
        ----------
        kwargs any appropriate for `Equality`.

        Returns
        -------
        True, False or an unevaluated `Equality` if truth cannot be determined.
        """
        return Equality(self.lhs, self.rhs, **kwargs).simplify()

    @property
    def reversed(self):
        """
        Swaps the lhs and the rhs.
        """
        return Equation(self.rhs, self.lhs)

    swap = reversed

    def _applytoexpr(self, expr, func, *args, **kwargs):
        # Applies a function to an expression checking whether there
        # is a specialized version associated with the particular type of
        # expression. Errors will be raised if the function cannot be
        # applied to an expression.
        funcname = getattr(func, '__name__', None)
        if funcname is not None:
            localfunc = getattr(expr, funcname, None)
            if localfunc is not None:
                return localfunc(*args, **kwargs)
        return func(expr, *args, **kwargs)

    def apply(self, func, *args, side='both', **kwargs):
        """
        Apply an operation/function/method to the equation returning the
        resulting equation.

        Parameters
        ==========

        func : object
            object to apply to the equation, usually a function.

        *args :
            Arguments passed to the function.

        side : str, optional
            Specifies which side of the equation the operation will be applied
            to. Default is 'both'. Possible options are 'both', 'lhs', 'rhs'.

        **kwargs :
            Keyword arguments passed to the function.
         """
        lhs = self.lhs
        rhs = self.rhs
        if side in ('both', 'lhs'):
            lhs = self._applytoexpr(self.lhs, func, *args, **kwargs)
        if side in ('both', 'rhs'):
            rhs = self._applytoexpr(self.rhs, func, *args, **kwargs)
        return Equation(lhs, rhs)

    def applylhs(self, func, *args, **kwargs):
        """
        If lhs side of the equation has a defined subfunction (attribute) of
        name ``func``, that will be applied instead of the global function.
        The operation is applied to only the lhs.
        """
        return self.apply(func, *args, **kwargs, side='lhs')

    def applyrhs(self, func, *args, **kwargs):
        """
        If rhs side of the equation has a defined subfunction (attribute) of
        name ``func``, that will be applied instead of the global function.
        The operation is applied to only the rhs.
        """
        return self.apply(func, *args, **kwargs, side='rhs')

    class _sides:
        """
        Helper class for the `.do.`, `.dolhs.`, `.dorhs.` syntax for applying
        submethods of expressions.
        """

        def __init__(self, eqn, side='both'):
            self.eqn = eqn
            self.side = side

        def __getattr__(self, name):
            import functools
            func = None
            if self.side in ('rhs', 'both'):
                func = getattr(self.eqn.rhs, name, None)
            else:
                func = getattr(self.eqn.lhs, name, None)
            if func is None:
                raise AttributeError(
                    f'Expressions in the equation have no attribute `{name}`.'
                    f' Try `.apply({name}, *args)` or pass the equation as'
                    f' a parameter to `{name}()`.')
            return functools.partial(self.eqn.apply, func, side=self.side)

    @property
    def do(self):
        return self._sides(self, side='both')

    @property
    def dolhs(self):
        return self._sides(self, side='lhs')

    @property
    def dorhs(self):
        return self._sides(self, side='rhs')

    def _eval_rewrite(self, rule, args, **kwargs):
        """Return Equation(L, R) as Equation(L - R, 0) or as L - R.

        Parameters
        ==========

        evaluate : bool, optional
            Control the evaluation of the result. If `evaluate=None` then
            terms in L and R will not cancel but they will be listed in
            canonical order; otherwise non-canonical args will be returned.
            Default to True.

        eqn : bool, optional
            Control the returned type. If `eqn=True`, then Equation(L - R, 0)
            is returned. Otherwise, the L - R symbolic expression is returned.
            Default to True.

        Examples
        ========
        >>> from sympy import Add
        >>> from sympy.abc import b, x
        >>> from sympy_equation import Equation
        >>> eq = Equation(x + b, x - b)
        >>> eq.rewrite(Add)
        Equation(2*b, 0)
        >>> eq.rewrite(Add, evaluate=None).lhs.args
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).lhs.args
        (b, x, b, -x)
        >>> eq.rewrite(Add, eqn=False)
        2*b
        >>> eq.rewrite(Add, eqn=False, evaluate=False).args
        (b, x, b, -x)
        """
        if rule == Add:
            # NOTE: the code about `evaluate` is very similar to
            # sympy.core.relational.Equality._eval_rewrite_as_Add
            eqn = kwargs.pop("eqn", True)
            evaluate = kwargs.get('evaluate', True)
            L, R = args
            if evaluate:
                # allow cancellation of args
                expr = L - R
            else:
                args = Add.make_args(L) + Add.make_args(-R)
                if evaluate is None:
                    # no cancellation, but canonical
                    expr = _unevaluated_Add(*args)
                else:
                    # no cancellation, not canonical
                    expr = Add._from_args(args)
            if eqn:
                return self.func(expr, 0)
            return expr

    def subs(self, *args, **kwargs):
        """Substitutes old for new in an equation after sympifying args.

        `args` is either:

        * one or more arguments of type `Equation(old, new)`.
        * two arguments, e.g. foo.subs(old, new)
        * one iterable argument, e.g. foo.subs(iterable). The iterable may be:

            - an iterable container with (old, new) pairs. In this case the
              replacements are processed in the order given with successive
              patterns possibly affecting replacements already made.
            - a dict or set whose key/value items correspond to old/new pairs.
              In this case the old/new pairs will be sorted by op count and in
              case of a tie, by number of args and the default_sort_key. The
              resulting sorted list is then processed as an iterable container
              (see previous).

        If the keyword ``simultaneous`` is True, the subexpressions will not be
        evaluated until all the substitutions have been made.

        Please, read ``help(Expr.subs)`` for more examples.

        Examples
        ========

        >>> from sympy.abc import a, b, c, x
        >>> from sympy_equation import Equation
        >>> eq = Equation(x + a, b * c)

        Substitute a single value:

        >>> eq.subs(b, 4)
        Equation(a + x, 4*c)

        Substitute a multiple values:

        >>> eq.subs([(a, 2), (b, 4)])
        Equation(x + 2, 4*c)
        >>> eq.subs({a: 2, b: 4})
        Equation(x + 2, 4*c)

        Substitute an equation into another equation:

        >>> eq2 = Equation(x + a, 4)
        >>> eq.subs(eq2)
        Equation(4, b*c)

        Substitute multiple equations into another equation:

        >>> eq1 = Equation(x + a + b + c, x * a * b * c)
        >>> eq2 = Equation(x + a, 4)
        >>> eq3 = Equation(b, 5)
        >>> eq1.subs(eq2, eq3)
        Equation(c + 9, 5*a*c*x)

        """
        new_args = args
        if all(isinstance(a, self.func) for a in args):
            new_args = [{a.args[0]: a.args[1] for a in args}]
        elif (len(args) == 1) and all(isinstance(a, self.func) for a in
                                      args[0]):
            raise TypeError("You passed into `subs` a list of elements of "
                            "type `Equation`, but this is not supported. Please, consider "
                            "unpacking the list with `.subs(*eq_list)` or select your "
                            "equations from the list and use `.subs(eq_list[0], eq_list["
                            "2], ...)`.")
        elif any(isinstance(a, self.func) for a in args):
            raise ValueError("`args` contains one or more Equation and some "
                             "other data type. This mode of operation is not supported. "
                             "Please, read `subs` documentation to understand how to "
                             "use it.")
        return super().subs(*new_args, **kwargs)

    #####
    # Overrides of binary math operations
    #####

    @classmethod
    def _binary_op(cls, a, b, opfunc_ab):
        if isinstance(a, Equation) and not isinstance(b, Equation):
            return Equation(opfunc_ab(a.lhs, b), opfunc_ab(a.rhs, b))
        elif isinstance(b, Equation) and not isinstance(a, Equation):
            return Equation(opfunc_ab(a, b.lhs), opfunc_ab(a, b.rhs))
        elif isinstance(a, Equation) and isinstance(b, Equation):
            return Equation(opfunc_ab(a.lhs, b.lhs), opfunc_ab(a.rhs, b.rhs))
        else:
            return NotImplemented

    def __add__(self, other):
        return self._binary_op(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, self, lambda a, b: a + b)

    def __mul__(self, other):
        return self._binary_op(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, self, lambda a, b: a * b)

    def __sub__(self, other):
        return self._binary_op(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, self, lambda a, b: a - b)

    def __truediv__(self, other):
        return self._binary_op(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_op(other, self, lambda a, b: a / b)

    def __mod__(self, other):
        return self._binary_op(self, other, lambda a, b: a % b)

    def __rmod__(self, other):
        return self._binary_op(other, self, lambda a, b: a % b)

    def __pow__(self, other):
        return self._binary_op(self, other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return self._binary_op(other, self, lambda a, b: a ** b)

    def _eval_power(self, other):
        return self.__pow__(other)

    def expand(self, *args, **kwargs):
        return Equation(self.lhs.expand(*args, **kwargs), self.rhs.expand(
            *args, **kwargs))

    def simplify(self, *args, **kwargs):
        return self._eval_simplify(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        return Equation(self.lhs.simplify(*args, **kwargs), self.rhs.simplify(
            *args, **kwargs))

    def _eval_factor(self, *args, **kwargs):
        # TODO: cancel out factors common to both sides.
        return Equation(self.lhs.factor(*args, **kwargs), self.rhs.factor(
            *args, **kwargs))

    def factor(self, *args, **kwargs):
        return self._eval_factor(*args, **kwargs)

    def _eval_collect(self, *args, **kwargs):
        return Equation(collect(self.lhs, *args, **kwargs),
                        collect(self.rhs, *args, **kwargs))

    def collect(self, *args, **kwargs):
        return self._eval_collect(*args, **kwargs)

    def evalf(self, *args, **kwargs):
        return Equation(self.lhs.evalf(*args, **kwargs),
                        self.rhs.evalf(*args, **kwargs))

    n = evalf

    def nsimplify(self, **kwargs):
        """See the documentation of nsimplify function in sympy.simplify."""
        return self.func(*[t.nsimplify(**kwargs) for t in self.args])

    def _eval_derivative(self, *args, **kwargs):
        # NOTE: as of SymPy 1.14.0, this method is never called.
        # So, Derivative(Equation(...), variable) is only applied to the
        # LHS.
        return Equation(
            Derivative(self.lhs, *args, **kwargs),
            Derivative(self.rhs, *args, **kwargs)
        )

    def _eval_Integral(self, *args, **kwargs):
        return Equation(
            Integral(self.lhs, *args, **kwargs),
            Integral(self.rhs, *args, **kwargs)
        )

    def __repr__(self):
        return f"Equation({self.lhs.__repr__()}, {self.rhs.__repr__()})"

    def _repr_pretty_(self, p, cycle):
        # NOTE: https://ipython.readthedocs.io/en/stable/config/integrating.html
        if equation_config.human_text is False:
            p.text(self.__repr__())
            return

        labelstr = ""
        namestr = self._get_eqn_name()
        if namestr != '' and equation_config.show_label:
            labelstr += '          (' + namestr + ')'
        p.text(self.__str__() + labelstr)

    def _latex(self, printer):
        lhs = printer._print(self.lhs)
        rhs = printer._print(self.rhs)
        tempstr = f"{lhs} = {rhs}"

        show_label = equation_config.show_label
        latex_as_equations = equation_config.latex_as_equations

        if latex_as_equations:
            return r'\begin{equation}'+ tempstr +r'\end{equation}'
        else:
            namestr = self._get_eqn_name()
            if namestr != '' and show_label:
                tempstr += r'\qquad (' + namestr + ')'

            return tempstr

    def __str__(self):
        return str(self.lhs) + ' = ' + str(self.rhs)

    def cross_multiply(self):
        """Cross-multiply the members of the equation. For example:

        n1   n2
        -- = --
        d1   d2

        gives:

        n1 * d2 = n2 * d1

        """
        n1, d1 = fraction(self.lhs)
        n2, d2 = fraction(self.rhs)
        return self.func(n1 * d2, n2 * d1)

    def as_expr(self):
        "Return an expression of the form: LHS - RHS."
        return self.lhs - self.rhs

    def to_expr(self):
        return self.as_expr()

    def diff(self, *symbols, **kwargs):
        return self.func(
            self.lhs.diff(*symbols, **kwargs),
            self.rhs.diff(*symbols, **kwargs)
        )

    def integrate(
        self, *args, meijerg=None, conds='piecewise', risch=None,
        heurisch=None, manual=None, **kwargs
    ):
        return self.func(
            self.lhs.integrate(
                *args, meijerg=meijerg, conds=conds, risch=risch,
                heurisch=heurisch, manual=manual, **kwargs
            ),
            self.rhs.integrate(
                *args, meijerg=meijerg, conds=conds, risch=risch,
                heurisch=heurisch, manual=manual, **kwargs
            ),
        )


Eqn = Equation

class _equation_config(param.Parameterized):
    """This class implements the configuration options for the module.

    Do not instantiate it directly, instead import the following:

    .. code-block:: python

       from sympy_equation import equation_config

    Then, set the appropriate attribute to the intended value, for example:

    .. code-block:: python

       equation_config.integers_as_exact = True

    """

    show_label = param.Boolean(False, doc="""
        If `True` a label with the name of the equation in the python
        environment will be shown on the screen. Default to `False`.""")

    human_text = param.Boolean(False, doc="""
        For text-based interactive environments, if the last line of a cell
        is the name of some equation, its execution will show the textual
        representation of the equation in the output. If ``human_text=True``
        the equation will be shown as "lhs = rhs". If ``False``, it will be
        shown as ``Equation(lhs, rhs)``.""")

    solve_to_list = param.Boolean(True, doc="""
        If ``True``, the results of a call to
        ``solve([e1, e2, ...], v1, v2, ...)`` will return a Python
        ``list``, otherwise it returns a Sympy's ``FiniteSet``.

        Note: setting this `True` means that expressions within the
        returned solutions might not be pretty-printed in Jupyter and
        IPython.""")

    latex_as_equations = param.Boolean(False, doc="""
        If `True` any output that is returned as LaTex for
        pretty-printing will be wrapped in the formal Latex for an
        equation. For example rather than:

        ```
        \\frac{a}{b}=c
        ```

        the output will be:

        ```
        \\begin{equation}\\frac{a}{b}=c\\end{equation}
        ```

        In an interactive environment like Jupyter notebook, this effectively
        moves the equation horizontally to the center of the screen.""")

    integers_as_exact = param.Boolean(False, doc="""
        If running in an IPython/Jupyter environment, preparse the content
        of a code line in order to convert integer numbers to sympy's Integer.
        This can be handy when writing expressions containing rational number.
        For example, by settings this to ``True`` we can write 2/3 which will
        be automatically converted to Integer(2)/Integer(3) which than SymPy
        converts to Rational(2, 3). If ``False``, no preparsing  is done,
        and Python evaluates 2/3 to 0.6666667, which will then be converted
        by SymPy to a Float.

        Note: it is reccommended to set this options to ``True`` only when
        executing purely symbolic computations, not when using other numerical
        libraries, such as Numpy, because it will create hard to debug
        situations. Consider executing this: ``np.cos(np.pi / 4)``.
        If ``integers_as_exact = True``, this will raise an error because
        4 is first replaced with sympy's Integer(4), then
        ``np.pi / Integer(4)`` becomes a symbolic expression and ``np.cos``
        is unable to evaluate it.""")

    @param.depends("integers_as_exact", watch=True)
    def _update_integers_as_exact(self):
        _toggle_integers_as_exact(self.integers_as_exact)


equation_config = _equation_config()


def _toggle_integers_as_exact(value):
    # TODO: Currently, equation_config.integers_as_exact = True/False
    # must be executed on its own cell. If originally we have this situation:
    #       equation_config.integers_as_exact is False
    # and then we execute these on the same cell:
    #       equation_config.integers_as_exact = True
    #       Eqn(a+1/2, b-3/4)
    # the output will be:
    #       a + 0.5 = b - 0.75
    # Creating equation_config as a singleton might solve this problem.

    try:
        import IPython
        ipython_shell = IPython.get_ipython()

        if ipython_shell:
            input_tranformations = ipython_shell.input_transformers_post

            if value is True:
                input_tranformations.append(integers_as_exact)
            else:
                # The below looks excessively complicated, but more reliably finds
                # the transformer to remove across varying IPython environments.
                for k in input_tranformations:
                    if "integers_as_exact" in k.__name__:
                        input_tranformations.remove(k)

    except ModuleNotFoundError:
        pass


Eqn = Equation


def solve(f, *symbols, **flags):
    """
    Wrapper of ``sympy.solve()``.

    If passed an expression (or a list of expressions) and variable(s) to
    solve for it behaves exactly like ``sympy.solve``.

    If passed an equation (or equations) it returns solutions as equations.
    If multiple solutions are present, they will be contained either in a list
    or in a FiniteSet, depending on the value of
    ``equation_config.solve_to_list``.

    See ``sympy.solve`` for the full documentation.

    Examples
    --------

    >>> from sympy_equation import Eqn, solve, equation_config
    >>> from sympy import symbols
    >>> a, b, c, x, y = symbols('a b c x y', real=True)

    When an ``Equation`` is provided as an argument of ``solve``, the output
    will contain objects of type ``Equation``:

    >>> eq = Eqn(a * x**2, -b * x - c)
    >>> res = solve(eq, x)
    >>> res
    [Equation(x, -b/(2*a) - sqrt(-4*a*c + b**2)/(2*a)), Equation(x, -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a))]

    Request the output to be a FiniteSet:

    >>> equation_config.solve_to_list = False
    >>> res = solve(eq, x)
    >>> res
    FiniteSet(Equation(x, -b/(2*a) - sqrt(-4*a*c + b**2)/(2*a)), Equation(x, -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a)))

    Solve multiple equations:

    >>> equation_config.solve_to_list = True
    >>> eq1 = Eqn(abs(2*x+y), 3)
    >>> eq2 = Eqn(abs(x + 2*y), 3)
    >>> res = solve((eq1, eq2), x, y)
    >>> res
    [[Equation(x, -3), Equation(y, 3)], [Equation(x, -1), Equation(y, -1)], [Equation(x, 1), Equation(y, 1)], [Equation(x, 3), Equation(y, -3)]]

    Convert the equations to expressions. In this case, ``solve`` is just a
    wrapper to ``sympy.solve``:

    >>> expr = [e.as_expr() for e in [eq1, eq2]]
    >>> res = solve(expr, x, y, dict=False)
    >>> res
    [(-3, 3), (-1, -1), (1, 1), (3, -3)]
    >>> res = solve(expr, x, y, dict=True)
    >>> res
    [{x: -3, y: 3}, {x: -1, y: -1}, {x: 1, y: 1}, {x: 3, y: -3}]

    """
    from sympy.solvers.solvers import solve
    from sympy.sets.sets import FiniteSet

    is_f_iter = hasattr(f, '__iter__')
    if not is_f_iter:
        f = [f]

    newf = [e.as_expr() if isinstance(e, Equation) else e for e in f]
    contains_eqn = any(isinstance(e, Equation) for e in f)

    if not contains_eqn:
        # execute solve without any pre-post processing in order not to alter
        # the expected behaviour by the users
        f = f if is_f_iter else f[0]
        return solve(f, *symbols, **flags)

    solns = []
    flags['dict'] = True
    result = solve(newf, *symbols, **flags)

    if len(symbols) == 1 and hasattr(symbols[0], "__iter__"):
        symbols = symbols[0]

    if len(result[0]) == 1:
        for k in result:
            for key in k.keys():
                val = k[key]
                tempeqn = Eqn(key, val)
                solns.append(tempeqn)
        if len(solns) == len(symbols):
            # sort according to the user-provided symbols
            solns = sorted(solns, key=lambda x: symbols.index(x.lhs))
    else:
        for k in result:
            solnset = []
            for key in k.keys():
                val = k[key]
                tempeqn = Eqn(key, val)
                solnset.append(tempeqn)
            if not equation_config.solve_to_list:
                solnset = FiniteSet(*solnset)
            else:
                if len(solnset) == len(symbols):
                    # sort according to the user-provided symbols
                    solnset = sorted(solnset, key=lambda x: symbols.index(x.lhs))
            solns.append(solnset)

    if equation_config.solve_to_list:
        if len(solns) == 1 and hasattr(solns[0], "__iter__"):
            # no need to wrap a list of a single element inside another list
            return solns[0]
        return solns
    else:
        if len(solns) == 1:
            # do not wrap a singleton in FiniteSet if it already is
            for k in solns:
                if isinstance(k, FiniteSet):
                    return k
        return FiniteSet(*solns)


def _eq_to_eqn(self):
    """Convert the Equality as an Equation.
    """
    return Equation(self.lhs, self.rhs)


Equality.to_Equation = _eq_to_eqn


def __FiniteSet__repr__override__(self):
    """Override of the `FiniteSet.__repr__(self)` to overcome sympy's
    inconsistent wrapping of Finite Sets which prevents reliable use of
    copy and paste of the code representation.
    """
    insidestr = ""
    for k in self.args:
        insidestr += k.__repr__() +', '
    insidestr = insidestr[:-2]
    reprstr = "FiniteSet("+ insidestr + ")"
    return reprstr


sympy.sets.FiniteSet.__repr__ = __FiniteSet__repr__override__


def __FiniteSet__str__override__(self):
    """Override of the `FiniteSet.__str__(self)` to overcome sympy's
    inconsistent wrapping of Finite Sets which prevents reliable use of
    copy and paste of the code representation.
    """
    insidestr = ""
    for k in self.args:
        insidestr += str(k) + ', '
    insidestr = insidestr[:-2]
    strrep = "{"+ insidestr + "}"
    return strrep


sympy.sets.FiniteSet.__str__ = __FiniteSet__str__override__
