import sys
import param
import sympy
from algebra_with_sympy.preparser import integers_as_exact
from sympy import *

from sympy.core.basic import Basic
from sympy.core.evalf import EvalfMixin


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
        Left-hand side of the equation.
    rhs : ``Expr``
        Left-hand side of the equation.
    swap : Equation
        Swap the lhs and rhs sides.

    Methods
    =======
    as_Boolean()
        Convert the ``Equation`` to an ``Equality``.
    as_expr()
        Convert the ``Equation`` to a symbolic expression of the form 'lhs - rhs'.
    check(**kwargs)
        Forces simplification and casts as ``Equality`` to check validity.
    to_expr()
        Alias of ``as_expr()``.
    cm()
        Given and equation in the form ``Equation(a/b, c/d)``, cross-multiply
        the members in order to get a new ``Equation(a*d, b*c)``.


    Examples
    ========

    >>> from sympy import *
    >>> from algebra_with_sympy import Eqn, Equation
    >>> a, b, c, x = symbols('a b c x')
    >>> Equation(a,b/c)
    Equation(a, b/c)
    >>> t=Eqn(a,b/c)
    >>> t
    Equation(a, b/c)
    >>> t*c
    Equation(a*c, b)
    >>> c*t
    Equation(a*c, b)
    >>> t.apply(exp)
    Equation(exp(a), exp(b/c))
    >>> t.apply(lambda side: exp(log(side)))
    Equation(a, b/c)

    Simplification and Expansion:

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
    >>> f2 = f+a*x**2+b*x +c
    >>> f2
    Equation(a*x**2 + b*x + c + x**2 - 1, a*x**2 + b*x + 2*c)
    >>> collect(f2,x)
    Equation(b*x + c + x**2*(a + 1) - 1, a*x**2 + b*x + 2*c)

    Apply operation to only one side:

    >>> poly = Eqn(a*x**2 + b*x + c*x**2, a*x**3 + b*x**3 + c*x)
    >>> poly.applyrhs(factor,x)
    Equation(a*x**2 + b*x + c*x**2, x*(c + x**2*(a + b)))
    >>> poly.applylhs(factor)
    Equation(x*(a*x + b + c*x), a*x**3 + b*x**3 + c*x)
    >>> poly.applylhs(collect,x)
    Equation(b*x + x**2*(a + c), a*x**3 + b*x**3 + c*x)

    ``.apply...`` also works with user defined python functions:

    >>> def add_square(eqn):
    ...     return eqn+eqn**2
    ...
    >>> t.apply(add_square)
    Equation(a**2 + a, b**2/c**2 + b/c)
    >>> t.applyrhs(add_square)
    Equation(a, b**2/c**2 + b/c)
    >>> t.apply(add_square, side = 'rhs')
    Equation(a, b**2/c**2 + b/c)
    >>> t.applylhs(add_square)
    Equation(a**2 + a, b/c)
    >>> add_square(t)
    Equation(a**2 + a, b**2/c**2 + b/c)

    In addition to ``.apply...`` there is also the less general ``.do``,
    ``.dolhs``, ``.dorhs``, which only works for operations defined on the
    ``Expr`` class (e.g.``.collect(), .factor(), .expand()``, etc...):

    >>> poly.dolhs.collect(x)
    Equation(b*x + x**2*(a + c), a*x**3 + b*x**3 + c*x)
    >>> poly.dorhs.collect(x)
    Equation(a*x**2 + b*x + c*x**2, c*x + x**3*(a + b))
    >>> poly.do.collect(x)
    Equation(b*x + x**2*(a + c), c*x + x**3*(a + b))
    >>> poly.dorhs.factor()
    Equation(a*x**2 + b*x + c*x**2, x*(a*x**2 + b*x**2 + c))

    ``poly.do.exp()`` or other sympy math functions will raise an error.

    Rearranging an equation (simple example made complicated as illustration):

    >>> p, V, n, R, T = var('p V n R T')
    >>> eq1=Eqn(p*V,n*R*T)
    >>> eq1
    Equation(V*p, R*T*n)
    >>> eq2 =eq1/V
    >>> eq2
    Equation(p, R*T*n/V)
    >>> eq3 = eq2/R/T
    >>> eq3
    Equation(p/(R*T), n/V)
    >>> eq4 = eq3*R/p
    >>> eq4
    Equation(1/T, R*n/(V*p))
    >>> 1/eq4
    Equation(T, V*p/(R*n))
    >>> eq5 = 1/eq4 - T
    >>> eq5
    Equation(0, -T + V*p/(R*n))

    Substitution (#'s and units):

    >>> L, atm, mol, K = var('L atm mol K', positive=True, real=True) # units
    >>> eq2.subs({R:0.08206*L*atm/mol/K,T:273*K,n:1.00*mol,V:24.0*L})
    Equation(p, 0.9334325*atm)
    >>> eq2.subs({R:0.08206*L*atm/mol/K,T:273*K,n:1.00*mol,V:24.0*L}).evalf(4)
    Equation(p, 0.9334*atm)

    Substituting an equation into another equation:

    >>> P, P1, P2, A1, A2, E1, E2 = var("P, P1, P2, A1, A2, E1, E2")
    >>> eq1 = Eqn(P, P1 + P2)
    >>> eq2 = Eqn(P1 / (A1 * E1), P2 / (A2 * E2))
    >>> P1_val = (eq1 - P2).swap
    >>> P1_val
    Equation(P1, P - P2)
    >>> eq2 = eq2.subs(P1_val)
    >>> eq2
    Equation((P - P2)/(A1*E1), P2/(A2*E2))

    Combining equations (Math with equations: lhs with lhs and rhs with rhs)
    >>> q = Eqn(a*c, b/c**2)
    >>> q
    Equation(a*c, b/c**2)
    >>> t
    Equation(a, b/c)
    >>> q+t
    Equation(a*c + a, b/c + b/c**2)
    >>> q/t
    Equation(c, 1/c)
    >>> t**q
    Equation(a**(a*c), (b/c)**(b/c**2))

    Utility operations:

    >>> t.reversed
    Equation(b/c, a)
    >>> t.swap
    Equation(b/c, a)
    >>> t.lhs
    a
    >>> t.rhs
    b/c
    >>> t.as_Boolean()
    Eq(a, b/c)

    `.check()` convenience method for `.as_Boolean().simplify()`:

    >>> Equation(pi*(I+2), pi*I+2*pi).check()
    True
    >>> Eqn(a,a+1).check()
    False

    Differentiation is applied to both sides:

    >>> q=Eqn(a*b, b**2/c**2)
    >>> q
    Equation(a*b, b**2/c**2)
    >>> diff(q,b)
    Equation(a, 2*b/c**2)
    >>> diff(q,c)
    Equation(0, -2*b**2/c**3)

    Integration is applied to both sides:
    >>> q=Eqn(a*c,b/c)
    >>> integrate(q,b)
    Equation(a*b*c, b**2/(2*c))

    Integration of each side with respect to different variables
    >>> q.dorhs.integrate(b).dolhs.integrate(a)
    Equation(a**2*c/2, b**2/(2*c))

    Automatic solutions using sympy solvers. THIS IS EXPERIMENTAL. Please
    report issues at https://github.com/gutow/Algebra_with_Sympy/issues.
    >>> tosolv = Eqn(a - b, c/a)
    >>> solve(tosolv,a)
    [Equation(a, b/2 - sqrt(b**2 + 4*c)/2), Equation(a, b/2 + sqrt(b**2 + 4*c)/2)]
    >>> solve(tosolv, b)
    [Equation(b, (a**2 - c)/a)]
    >>> solve(tosolv, c)
    [Equation(c, a**2 - a*b)]
    """

    def __new__(cls, lhs, rhs, **kwargs):
        from sympy.core.sympify import _sympify
        from sympy.core.expr import Expr
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
        from sympy import Equality
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
        from sympy.core.relational import Equality
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

        func: object
            object to apply usually a function

        args: as necessary for the function

        side: 'both', 'lhs', 'rhs', optional
            Specifies which side of the equation the operation will be applied
            to. Default is 'both'.

        kwargs: as necessary for the function
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
        >>> from sympy import Equation
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
        from sympy import Add
        from sympy.core.add import _unevaluated_Add
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
        >>> from sympy import Equation
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

    #####
    # Operation helper functions
    #####
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
        from sympy.simplify.radsimp import collect
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

    #####
    # Output helper functions
    #####
    def __repr__(self):
        repstr = 'Equation(%s, %s)' % (
        self.lhs.__repr__(), self.rhs.__repr__())
        return repstr

    def _latex(self, printer):
        tempstr = ''
        tempstr += printer._print(self.lhs)
        tempstr += '='
        tempstr += printer._print(self.rhs)
        return tempstr

    def __str__(self):
        tempstr = ''
        tempstr += str(self.lhs) + ' = ' + str(self.rhs)
        return tempstr

    def cm(self):
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

class _algwsym_config(param.Parameterized):

    show_label = param.Boolean(False, doc="""
        If `True` a label with the name of the equation in the python
        environment will be shown on the screen. Default = `False`.""")

    human_text = param.Boolean(False, doc="""
        For text-based interactive environments, or for Python consoles,
        by entering the name of the equation and executing this line of code
        it will show the equation in textual format. If ``human_text=True``
        the equation will be shown as "lhs = rhs". If ``False``, it will be
        shown as ``Equation(lhs, rhs)``.""")

    solve_to_list = param.Boolean(True, doc="""
        If ``True`` the results of a call to ``solve(...)`` will return a
        Python ``list`` rather than a Sympy ``FiniteSet``.

        Note: setting this `True` means that expressions within the
        returned solutions might not be pretty-printed in Jupyter and
        IPython.""")

    latex_as_equations = param.Boolean(False, doc="""
        If `True` any output that is returned as LaTex for
        pretty-printing will be wrapped in the formal Latex for an
        equation. For example rather than:
        ```
        $\\frac{a}{b}=c$
        ```
        the output will be:
        ```
        \\begin{equation}\\frac{a}{b}=c\\end{equation}
        ```
        In an interactive environment like Jupyter notebook, this effectively
        moves the equation horizontally to the center of the screen.""")

    latex_printer = param.Callable(latex, doc="""
        The latex printer function which will be used to create
        the latex output to be shown on the screen.""")

    integers_as_exact = param.Boolean(False, doc="""
        If running in an IPython/Jupyter environment, preparse the content
        of a code line in order to convert integer numbers to sympy's Integer.
        This can be handy when writing expressions containing rational number:
        by settings this to ``True`` for example, we can write 2/3 which will
        be automatically converted to Integer(2)/Integer(3) which than SymPy
        would convert to Rational(2, 3). If ``False``, no preparsing would
        be done, and Python would be evaluated 2/3 to 0.6666667, which will
        then be converted by SymPy to a Float.

        However, it is reccommended to set this options to ``True`` if we
        are only using SymPy and symbolic operations, not when we are using
        other numerical libraries, such as Numpy, because it will create hard
        to debug situation. Consider this line of code: ``np.cos(np.pi / 4)``.
        This will raise an error, because 4 is first converted to
        sympy's Integer, then ``np.pi / 4`` becomes a symbolic expression
        and ``np.cos`` is unable to evaluate it.""")

    @param.depends("integers_as_exact", watch=True)
    def _update_integers_as_exact(self):
        if self.integers_as_exact:
            set_integers_as_exact()
        else:
            unset_integers_as_exact()


algwsym_config = _algwsym_config()


def __latex_override__(expr, *arg):
    algwsym_config = False
    ip = False
    try:
        from IPython import get_ipython
        if get_ipython():
            ip = True
    except ModuleNotFoundError:
        pass
    colab = False
    try:
        from google.colab import output
        colab = True
    except ModuleNotFoundError:
        pass
    show_label = False
    latex_as_equations = False
    latex_printer = latex
    if ip:
        algwsym_config = get_ipython().user_ns.get("algwsym_config", False)
    else:
        algwsym_config = globals()['algwsym_config']
    if algwsym_config:
        show_label = algwsym_config.show_label
        latex_as_equations = algwsym_config.latex_as_equations
        latex_printer = algwsym_config.latex_printer
    if latex_as_equations:
        return r'\begin{equation}'+latex_printer(expr)+r'\end{equation}'
    else:
        tempstr = ''
        namestr = ''
        if isinstance(expr, Equation):
            namestr = expr._get_eqn_name()

        if namestr != '' and algwsym_config and show_label:
            tempstr += r'$'+latex_printer(expr)
            # work around for colab's inconsistent handling of mixed latex and
            # plain strings.
            if colab:
                colabname = namestr.replace('_', r'\_')
                tempstr += r'\,\,\,\,\,\,\,\,\,\,(' + colabname + ')$'
            else:
                tempstr += r'\,\,\,\,\,\,\,\,\,\,$(' + namestr + ')'
            return tempstr
        else:
            return '$'+latex_printer(expr) + '$'

def __command_line_printing__(expr, *arg):
    # print('Entering __command_line_printing__')
    human_text = True
    if algwsym_config:
        human_text = algwsym_config.human_text
    tempstr = ''
    if not human_text:
        return print(tempstr + repr(expr))
    else:
        labelstr = ''
        namestr = ''
        if isinstance(expr, Equation):
            namestr = expr._get_eqn_name()
        if namestr != '' and algwsym_config.show_label:
            labelstr += '          (' + namestr + ')'
        return print(tempstr + str(expr) + labelstr)

# Now we inject the formatting override(s)
ip = None
try:
    from IPython import get_ipython
    ip = get_ipython()
except ModuleNotFoundError:
    ip = false
formatter = None
if ip:
    # In an environment that can display typeset latex
    formatter = ip.display_formatter
    old = formatter.formatters['text/latex'].for_type(Basic,
                                                      __latex_override__)
    # print("For type Basic overriding latex formatter = " + str(old))

    # For the terminal based IPython
    if "text/latex" not in formatter.active_types:
        old = formatter.formatters['text/plain'].for_type(tuple,
                                                    __command_line_printing__)
        # print("For type tuple overriding plain text formatter = " + str(old))
        for k in sympy.__all__:
            if k in globals() and not "Printer" in k:
                if isinstance(globals()[k], type):
                    old = formatter.formatters['text/plain'].\
                        for_type(globals()[k], __command_line_printing__)
                    # print("For type "+str(k)+
                    # " overriding plain text formatter = " + str(old))
else:
    # command line
    # print("Overriding command line printing of python.")
    sys.displayhook = __command_line_printing__

# Numerics controls
def set_integers_as_exact():
    """This operation uses `sympy.interactive.session.int_to_Integer`, which
    causes any number input without a decimal to be interpreted as a sympy
    integer, to pre-parse input cells. It also sets the flag
    `algwsym_config.integers_as_exact = True` This is the default
    mode of algebra_with_sympy. To turn this off call
    `unset_integers_as_exact()`.
    """
    ip = False
    try:
        from IPython import get_ipython
        ip = True
    except ModuleNotFoundError:
        ip = False
    if ip:
        if get_ipython():
            get_ipython().input_transformers_post.append(integers_as_exact)
            algwsym_config = get_ipython().user_ns.get("algwsym_config", False)
            if algwsym_config:
                algwsym_config.integers_as_exact = True
            else:
                raise ValueError("The algwsym_config object does not exist.")
    return

def unset_integers_as_exact():
    """This operation disables forcing of numbers input without
    decimals being interpreted as sympy integers. Numbers input without a
    decimal may be interpreted as floating point if they are part of an
    expression that undergoes python evaluation (e.g. 2/3 -> 0.6666...). It
    also sets the flag `algwsym_config.integers_as_exact = False`.
    Call `set_integers_as_exact()` to avoid this conversion of rational
    fractions and related expressions to floating point. Algebra_with_sympy
    starts with `set_integers_as_exact()` enabled (
    `algwsym_config.integers_as_exact = True`).
    """
    ip = False
    try:
        from IPython import get_ipython
        ip = True
    except ModuleNotFoundError:
        ip = False
    if ip:
        if get_ipython():
            pre = get_ipython().input_transformers_post
            # The below looks excessively complicated, but more reliably finds the
            # transformer to remove across varying IPython environments.
            for k in pre:
                if "integers_as_exact" in k.__name__:
                    pre.remove(k)
            algwsym_config = get_ipython().user_ns.get("algwsym_config", False)
            if algwsym_config:
                algwsym_config.integers_as_exact = False
            else:
                raise ValueError("The algwsym_config object does not exist.")

    return

Eqn = Equation
if ip and "text/latex" not in formatter.active_types:
    old = formatter.formatters['text/plain'].for_type(Eqn,
                                                __command_line_printing__)
    # print("For type Equation overriding plain text formatter = " + str(old))

def units(names):
    """
    This operation declares the symbols to be positive values, so that sympy
    will handle them properly when simplifying expressions containing units.
    Units defined this way are just unit symbols. If you want units that are
    aware of conversions see sympy.physics.units.


    :param string names: a string containing a space separated list of
    symbols to be treated as units.

    :return string list of defined units: calls `name = symbols(name,
    positive=True)` in the interactive namespace for each symbol name.
    """
    from sympy.core.symbol import symbols
    #import __main__ as shell
    user_namespace = None
    try:
        from IPython import get_ipython
        if get_ipython():
            user_namespace = get_ipython().user_ns
    except ModuleNotFoundError:
        pass
    syms = names.split(' ')
    retstr = ''

    if user_namespace==None:
        import sys
        frame_num = 0
        frame_name = None
        while frame_name != '__main__' and frame_num < 50:
            user_namespace = sys._getframe(frame_num).f_globals
            frame_num +=1
            frame_name = user_namespace['__name__']
    retstr +='('
    for k in syms:
        user_namespace[k] = symbols(k, positive = True)
        retstr += k + ','
    retstr = retstr[:-1] + ')'
    return retstr


def solve(f, *symbols, **flags):
    """
    Override of sympy `solve()`.

    If passed an expression and variable(s) to solve for it behaves
    almost the same as normal solve with `dict = True`, except that solutions
    are wrapped in a FiniteSet() to guarantee that the output will be pretty
    printed in Jupyter like environments.

    If passed an equation or equations it returns solutions as a
    `FiniteSet()` of solutions, where each solution is represented by an
    equation or set of equations.

    To get a Python `list` of solutions (pre-0.11.0 behavior) rather than a
    `FiniteSet` issue the command `algwsym_config.solve_to_list = True`.
    This also prevents pretty-printing in IPython and Jupyter.

    Examples
    --------
    >>> a, b, c, x, y = symbols('a b c x y', real = True)
    >>> import sys
    >>> sys.displayhook = __command_line_printing__ # set by default on normal initialization.
    >>> eq1 = Eqn(abs(2*x+y),3)
    >>> eq2 = Eqn(abs(x + 2*y),3)
    >>> B = solve((eq1,eq2))

    Default human readable output on command line
    >>> B
    {{x = -3, y = 3}, {x = -1, y = -1}, {x = 1, y = 1}, {x = 3, y = -3}}

    To get raw output turn off by setting
    >>> algwsym_config.human_text=False
    >>> B
    FiniteSet(FiniteSet(Equation(x, -3), Equation(y, 3)), FiniteSet(Equation(x, -1), Equation(y, -1)), FiniteSet(Equation(x, 1), Equation(y, 1)), FiniteSet(Equation(x, 3), Equation(y, -3)))

    Pre-0.11.0 behavior where a python list of solutions is returned
    >>> algwsym_config.solve_to_list = True
    >>> solve((eq1,eq2))
    [[Equation(x, -3), Equation(y, 3)], [Equation(x, -1), Equation(y, -1)], [Equation(x, 1), Equation(y, 1)], [Equation(x, 3), Equation(y, -3)]]
    >>> algwsym_config.solve_to_list = False # reset to default

    """
    from sympy.solvers.solvers import solve
    from sympy.sets.sets import FiniteSet
    newf =[]
    solns = []
    displaysolns = []
    contains_eqn = False
    if hasattr(f,'__iter__'):
        for k in f:
            if isinstance(k, Equation):
                newf.append(k.lhs-k.rhs)
                contains_eqn = True
            else:
                newf.append(k)
    else:
        if isinstance(f, Equation):
            newf.append(f.lhs - f.rhs)
            contains_eqn = True
        else:
            newf.append(f)
    flags['dict'] = True
    result = solve(newf, *symbols, **flags)
    if len(symbols) == 1 and hasattr(symbols[0], "__iter__"):
        symbols = symbols[0]
    if contains_eqn:
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
                if not algwsym_config.solve_to_list:
                    solnset = FiniteSet(*solnset)
                else:
                    if len(solnset) == len(symbols):
                        # sort according to the user-provided symbols
                        solnset = sorted(solnset, key=lambda x: symbols.index(x.lhs))
                solns.append(solnset)
    else:
        solns = result
    if algwsym_config.solve_to_list:
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

def solveset(f, symbols, domain=sympy.Complexes):
    """
    Very experimental override of sympy solveset, which we hope will replace
    solve. Much is not working. It is not clear how to input a system of
    equations unless you directly select `linsolve`, etc...
    """
    from sympy.solvers import solveset as solve
    newf = []
    solns = []
    displaysolns = []
    contains_eqn = False
    if hasattr(f, '__iter__'):
        for k in f:
            if isinstance(k, Equation):
                newf.append(k.lhs - k.rhs)
                contains_eqn = True
            else:
                newf.append(k)
    else:
        if isinstance(f, Equation):
            newf.append(f.lhs - f.rhs)
            contains_eqn = True
        else:
            newf.append(f)
    result = solve(*newf, symbols, domain=domain)
    # if contains_eqn:
    #     if len(result[0]) == 1:
    #         for k in result:
    #             for key in k.keys():
    #                 val = k[key]
    #                 tempeqn = Eqn(key, val)
    #                 solns.append(tempeqn)
    #         display(*solns)
    #     else:
    #         for k in result:
    #             solnset = []
    #             displayset = []
    #             for key in k.keys():
    #                 val = k[key]
    #                 tempeqn = Eqn(key, val)
    #                 solnset.append(tempeqn)
    #                 if algwsym_config.show_solve_output:
    #                     displayset.append(tempeqn)
    #             if algwsym_config.show_solve_output:
    #                 displayset.append('-----')
    #             solns.append(solnset)
    #             if algwsym_config.show_solve_output:
    #                 for k in displayset:
    #                     displaysolns.append(k)
    #         if algwsym_config.show_solve_output:
    #             display(*displaysolns)
    # else:
    solns = result
    return solns


class Equality(Equality):
    """
    Extension of Equality class to include the ability to convert it to an
    Equation.
    """
    def to_Equation(self):
        """
        Return: recasts the Equality as an Equation.
        """
        return Equation(self.lhs,self.rhs)

    def to_Eqn(self):
        """
        Synonym for to_Equation.
        Return: recasts the Equality as an Equation.
        """
        return self.to_Equation()

Eq = Equality

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

# Redirect python abs() to Abs()
abs = Abs