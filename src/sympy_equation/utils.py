import param
import warnings
from sympy import (
    Basic, Expr, latex, postorder_traversal, count_ops, Add, fraction,
    Equality
)
from sympy_equation.algebraic_equation import Equation
from numbers import Number as PythonNumber
from typing import Callable, List
from contextlib import contextmanager
import inspect


@contextmanager
def edit_readonly(parameterized):
    """
    Temporarily set parameters on Parameterized object to readonly=False,
    constant=False, to allow editing them.
    """
    params = parameterized.param.objects("existing").values()
    readonlys = [p.readonly for p in params]
    constants = [p.constant for p in params]
    for p in params:
        p.constant = False
        p.readonly = False
    try:
        yield
    except:
        raise
    finally:
        for p, ro, const in zip(params, readonlys, constants):
            p.constant = const
            p.readonly = ro


def _table_generator(
    expressions: dict[int, Expr],
    use_latex: bool=True,
    latex_printer: Callable[[Expr], str]=None,
    column_labels: List[str]=["idx", "expr"],
    title: str="",
) -> None:
    if latex_printer is None:
        latex_printer = latex
    if title is None:
        title = ""

    try:
        from IPython.display import Markdown, display
    except ImportError:
        display, Markdown = None, None
        if use_latex:
            warnings.warn(
                "You decided to show the table using Markdown+Latex, but this"
                " mode of operation requires IPython, which was not found."
                " Proceeding by showing a textual table.",
                stacklevel=1
            )

    if use_latex and Markdown:
        # Latex mode: just print markdown-compatible table
        header = f"| {column_labels[0]} | {column_labels[1]} |"
        sep = "|:-----:|:------|"

        if title:
            table = title + "\n" + header + "\n" + sep + "\n"
        else:
            table = header + "\n" + sep + "\n"

        for idx, expr in expressions.items():
            expr_str = str(expr) if not use_latex else f"${latex_printer(expr)}$"
            table += f"| {idx} | {expr_str} |\n"

        display(Markdown(table))

    else:
        rows = []
        for i, expr in expressions.items():
            expr_str = str(expr) if not use_latex else f"${latex_printer(expr)}$"
            rows.append((str(i), expr_str))

        # Text mode: compute column widths
        index_width = max(len(r[0]) for r in rows + [("index", "")])
        expr_width  = max(len(r[1]) for r in rows + [("", column_labels[1])])

        header = f"{column_labels[0].ljust(index_width)} | {column_labels[1].ljust(expr_width)}"
        sep = f"{'-'*index_width}-|-{'-'*expr_width}"
        print(header)
        print(sep)
        for idx, expr_str in rows:
            print(f"{idx.ljust(index_width)} | {expr_str.ljust(expr_width)}")


class table_of_expressions(param.Parameterized):
    """
    Nicely print the arguments of a symbolic expression as a table with two
    columns: an index, and the argument itself. The index can later be used
    to retrive the argument we are interested in, without having to resort
    to pattern matching operations.

    There are three modes of operation:

    1. ``t = table_of_expressions(list/set of expressions)``: sort the list/set
       of expressions in a deterministic way and visualize the table on the
       screen. This is useful when dealing with results of pattern 
       matching operations.
    2. ``t = table_of_expressions(expr, mode="args")``: visualize the arguments
       of a symbolic expressions. These arguments are not sorted, which means
       ``t[idx]`` returns the same expression as ``expr.args[idx]``.
    3. ``t = table_of_expressions(expr, mode="nodes")``: visualize the unique
       nodes of the expression tree, sorted in a deterministic way.
       This mode uses :py:func:`sympy.postorder_traversal` in order to
       retrieve all nodes of the expression tree. The larger the expression,
       the greater the number of nodes. For large expressions there are two
       disadvantages to be aware of:

       * screen space: if ``auto_show=True`` the table will be automatically
         visualized on the screen. The larger the expression tree, the more
         time to show it on the screen and the more space will be used.
         This can be mitigated by filtering the table with the ``select``
         keyword arguments (see examples below).
       * memory usage.

    Examples
    --------

    There are situations where we might be dealing with relatively complex and 
    large expressions. Suppose the following expression is the result of a 
    symbolic integration:

    >>> from sympy import symbols, Pow
    >>> from sympy_equation import table_of_expressions
    >>> L, mdot, q, T_in, c_p, n, xi = symbols("L, mdot, q, T_in, cp, n, xi")
    >>> expr = L**2*mdot**2*q**2*(L*q + T_in*mdot*c_p)**(n*xi) + 2*L*T_in*mdot**3*c_p*q*(L*q + T_in*mdot*c_p)**(n*xi) + T_in**2*mdot**4*c_p**2*(L*q + T_in*mdot*c_p)**(n*xi) - mdot**(n*xi + 4)*(T_in*c_p)**(n*xi + 2)

    This addition is composed of 4 terms. 3 of them share a common term
    that can be collected, ``(L*q + T_in*mdot*c_p)**(n*xi)``. 
    
    Let's explore the first mode of operation of this class. Instead of typing
    that term directly and risk inserting typing errors, we can extract it 
    with a pattern matching operation. For example, let's find all powers:

    >>> ton = table_of_expressions(expr.find(Pow), use_latex=False)
    idx   | exprs                       
    ------|-----------------------------
    0     | L**2                        
    1     | T_in**2                     
    2     | cp**2                       
    3     | mdot**2                     
    4     | mdot**3                     
    5     | mdot**4                     
    6     | q**2                        
    7     | mdot**(n*xi + 4)            
    8     | (T_in*cp)**(n*xi + 2)       
    9     | (L*q + T_in*cp*mdot)**(n*xi)

    The above output shows sub-expressions that are powers. The interested
    terms is located at index 9:

    >>> expr.collect(ton[9])
    -mdot**(n*xi + 4)*(T_in*cp)**(n*xi + 2) + (L*q + T_in*cp*mdot)**(n*xi)*(L**2*mdot**2*q**2 + 2*L*T_in*cp*mdot**3*q + T_in**2*cp**2*mdot**4)

    This is now an addition of 2 terms.

    Let's explore the second mode of operation, which shows the arguments of a 
    symbolic expression:

    >>> t = table_of_expressions(expr, mode="args", use_latex=False)
    idx   | args                                              
    ------|---------------------------------------------------
    0     | -mdot**(n*xi + 4)*(T_in*cp)**(n*xi + 2)           
    1     | L**2*mdot**2*q**2*(L*q + T_in*cp*mdot)**(n*xi)    
    2     | T_in**2*cp**2*mdot**4*(L*q + T_in*cp*mdot)**(n*xi)
    3     | 2*L*T_in*cp*mdot**3*q*(L*q + T_in*cp*mdot)**(n*xi)

    The above output shows the 4 sub-expressions that composes ``expr``.
    Again, the table can be indexed in order to retrieve the interested term,
    for example:

    >>> t[2]
    T_in**2*cp**2*mdot**4*(L*q + T_in*cp*mdot)**(n*xi)

    The third mode of operation shows a the unique nodes that are contained
    in the expression tree.

    >>> table_of_expressions(expr, mode="nodes", use_latex=False)   # doctest:+SKIP

    This mode of operation usually creates large tables. The can be filtered
    by terms contained in the nodes, using the ``select`` keyword argument.
    For example, let's visualize the node containing the term ``L*q``:

    >>> toe = table_of_expressions(expr, select=[L*q], mode="nodes", use_latex=False)
    idx   | nodes                                                                                                                                                                                            
    ------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    12    | L*q                                                                                                                                                                                              
    26    | L*q + T_in*cp*mdot                                                                                                                                                                               
    27    | (L*q + T_in*cp*mdot)**(n*xi)                                                                                                                                                                     
    29    | L**2*mdot**2*q**2*(L*q + T_in*cp*mdot)**(n*xi)                                                                                                                                                   
    30    | T_in**2*cp**2*mdot**4*(L*q + T_in*cp*mdot)**(n*xi)                                                                                                                                               
    31    | 2*L*T_in*cp*mdot**3*q*(L*q + T_in*cp*mdot)**(n*xi)                                                                                                                                               
    32    | L**2*mdot**2*q**2*(L*q + T_in*cp*mdot)**(n*xi) + 2*L*T_in*cp*mdot**3*q*(L*q + T_in*cp*mdot)**(n*xi) + T_in**2*cp**2*mdot**4*(L*q + T_in*cp*mdot)**(n*xi) - mdot**(n*xi + 4)*(T_in*cp)**(n*xi + 2)

    """

    expr = param.Parameter(doc="""
        The symbolic expression whose arguments or nodes are to be shown.""")
    mode = param.Selector(default="args", objects=["args", "nodes"], doc="""
        What to extract from the symbolic expression.""")
    select = param.List(default=[], item_type=(Expr, PythonNumber), doc="""
        List of targets used to filter the table. The table is constructed
        by looping over ``expressions``. If an expression contains any of
        the targets, it will be shown on the table.""")
    use_latex = param.Boolean(default=True, doc="""
        If True, a Markdown table containing latex expressions will
        be shown. Otherwise, a plain-text table will be shown.""")
    latex_printer = param.Callable(default=None, doc="""
        A function similar to sympy's ``latex`` that generates the appropriate
        Latex representation for symbolic expressions when ``use_latex=True``.
        If not provided, sympy's ``latex`` will be used.""")
    auto_show = param.Boolean(default=True, doc="""
        If True, the table will be shown on the screen automatically
        after instantiation, or after editing the `expr` and `has` attributes.
        Otherwise, the ``show()`` method must be executed manually in order
        to visualize the table.""")
    column_labels = param.List(default=["idx", "exprs"], bounds=(2, 2), doc="""
        Labels to be shown on the header of the table.""")
    # readonly attributes
    expressions = param.List(default=[], item_type=Basic, readonly=True, doc="""
        List of sub-expressions composing `expr`.""")
    selected_idx = param.List(default=[], item_type=int, readonly=True, doc="""
        Get the indices of the expressions that were filtered 
        by ``select``.""")

    def __init__(self, expr, **params):
        mode = params.get("mode", self.mode)
        params.setdefault(
            "column_labels",
            ["idx", mode if isinstance(expr, Basic) else "exprs"]
        )
        super().__init__(expr=expr, **params)
        self._extract_expressions_from_expr()
        self._select_expressions()
        if self.auto_show:
            self.show()

    @param.depends("expr", watch=True)
    def _extract_expressions_from_expr(self):
        expressions = None
        sort = lambda _list: sorted(
            set(_list),
            # NOTE: sort by operation count, then by string representation
            # for tie-breaking. This key should ensure deterministic ordering
            key=lambda expr: (count_ops(expr), str(expr))
        )

        if inspect.isgenerator(self.expr):
            expressions = sort(self.expr)
        elif isinstance(self.expr, Basic):
            if self.mode == "args":
                # do not sort the arguments, because the knowledge of their
                # actual indexes is very important when applying other
                # manipulations
                expressions = list(self.expr.args)
            else:
                # get a list of unique nodes, sorted by length
                # and alphabetically
                expressions = sort(postorder_traversal(self.expr))
        elif isinstance(self.expr, (tuple, set, list)):
            # self.expr is the results of some pattern-matching query
            expressions = sort(self.expr)

        if expressions is None:
            raise TypeError(
                "Could not extract a list of expressions from `expr`."
                f" This is likely due to a wrong type(expr)={type(self.expr)}."
                " Please, read the documentation to understand what"
                " `expr` should be."
            )

        with edit_readonly(self):
            self.expressions = expressions

    @param.depends("expr", "select", watch=True)
    def _select_expressions(self):
        indices = []
        for i, expr in enumerate(self.expressions):
            if expr.has(*self.select):
                indices.append(i)

        with edit_readonly(self):
            self.idx_selected_expressions = indices

    @param.depends("expr", "select", watch=True)
    def _trigger_show(self):
        if self.auto_show:
            self.show()

    def show(self):
        """Show the table on the screen."""
        indices = self.idx_selected_expressions
        if len(self.select) == 0:
            indices = range(len(self.expressions))

        expressions = {i: self.expressions[i] for i in indices}

        _table_generator(
            expressions,
            use_latex=self.use_latex,
            latex_printer=self.latex_printer,
            column_labels=self.column_labels,
        )

    def __getitem__(self, k):
        return self.expressions[k]

    def __len__(self):
        return len(self.expressions)

    def __repr__(self):
        return object.__repr__(self)

    def get_selected_expressions(self):
        """Returns the expressions filtered by ``select``."""
        return [
            node for i, node in enumerate(self.expressions)
            if i in self.idx_selected_expressions
        ]


def process_arguments_of_add(expr, indices_groups, func, check=True):
    """
    Given an addition composed of several terms, this function performs
    the following:
    
    1. select a group of terms.
    2. add them together to create.
    3. apply `func` to the result of step 2, which will compute a 
       new expression.
    4. replace the addition of step 2 with the new expression.

    the function ``table_of_expressions`` can be used to select the 
    appropriate terms to be modified.
    
    Parameters
    ----------
    expr : Add
        The addition to modify.
    indices_groups : list
        A list of lists of integer numbers. Each list contains 
        indices of arguments to be selected in step 1. In practice,
        each list represent a group of terms.
    func : callable
        A callable requiring one argument, the expression created
        at step 2, and returning a new expression.
    check : boolean, optional
        If True, verify that the new expression is mathematically
        equivalent to `expr`. If their are not, or the equivalency
        could not be established, a warning will be shown, but the
        function will returned the modified expression.
    
    Returns
    -------
    new_expr : Add

    Examples
    --------

    Consider the following addition. Modify it in order to collect
    terms containing ratios. 

    >>> from sympy import symbols, factor
    >>> from sympy_equation import process_arguments_of_add, table_of_expressions
    >>> gamma, v1, v2, p1, p2 = symbols("gamma, v1, v2, p1, p2")
    >>> expr = gamma - gamma*v2/v1 + gamma*p2/p1 + 1 + v2/v1 - p2/p1
    >>> toe = table_of_expressions(expr, use_latex=False)
    idx   | args        
    ------|-------------
    0     | 1           
    1     | gamma       
    2     | v2/v1       
    3     | -p2/p1      
    4     | gamma*p2/p1 
    5     | -gamma*v2/v1
    
    From the above table, we can see that terms 2 and 5 contains v2/v1,
    while terms 3 and 4 contains p2/p1. Sympy's `factor()` can be used for 
    this task:

    >>> new_expr = process_arguments_of_add(expr, [[2, 5], [3, 4]], factor)
    >>> new_expr
    gamma + 1 - v2*(gamma - 1)/v1 + p2*(gamma - 1)/p1

    See Also
    --------
    table_of_expressions
    """
    if not isinstance(expr, Expr):
        return expr
    if not expr.is_Add:
        return expr

    # make sure indices_groups is a list of lists
    if all(isinstance(i, PythonNumber) for i in indices_groups):
        indices_groups = [indices_groups]
    if (
        (not all(isinstance(i, (list, tuple)) for i in indices_groups))
        or (not all(all(isinstance(t, int) for t in _list) for _list in indices_groups))
    ):
        raise ValueError(
            "`indices_groups` must be a list of lists containing integer numbers"
            " representing the index of arguments of `expr`."
        )

    substitutions_dict = {}
    for indices in indices_groups:
        term = Add(*[a for i, a in enumerate(expr.args) if i in indices])
        new_term = func(term)
        substitutions_dict[term] = new_term

    new_expr = expr.subs(substitutions_dict)
    if check and (not new_expr.equals(expr)):
        warnings.warn(
            "The substitution created a new expression which is"
            " mathematically different from the original expression"
            " (or its equivalency could not be established)."
            " Watch out!",
            stacklevel=1
        )
    return new_expr


def divide_term_by_term(expr, denominator=None):
    """
    Consider a symbolic expression having the form `numerator / denominator`,
    where `numerator` is an addition. This function will divide each term
    of `numerator` by `denominator`.

    Parameters
    ----------
    expr : Add or Mul
        The symbolic expression to be modified. If `denominator=None`,
        then `expr` must be a fraction, where the numerator is an addition.
        If `denominator` is provided then `expr` must be an addition.
    denominator : Expr or None
        If None, the denominator will be extracted using sympy's `fraction()`.
        If an expression is provided, then all arguments of `expr` will be
        divided by `denominator`.

    Returns
    -------
    new_expr : Add

    Examples
    --------

    Consider an expression with the form numerator/denominator:

    >>> from sympy import symbols
    >>> from sympy_equation import divide_term_by_term
    >>> gamma, v1, v2, p1, p2 = symbols("gamma, v1, v2, p1, p2")
    >>> expr = (gamma + 1 - v2*(gamma - 1)/v1 + p2*(gamma - 1)/p1)/(gamma - 1)

    Note the denominator on the right `(gamma - 1)`. Let's divide term by term:
    
    >>> new_expr = divide_term_by_term(expr)
    >>> new_expr
    gamma/(gamma - 1) + 1/(gamma - 1) - v2/v1 + p2/p1

    Now, consider an addition of terms. We would like to divide all terms
    by the same denominator:

    >>> a, b, c, d, e = symbols("a:e")
    >>> expr = a + b - c / d
    >>> den = 2*a - e
    >>> new_expr = divide_term_by_term(expr, denominator=den)
    >>> new_expr
    a/(2*a - e) + b/(2*a - e) - c/(d*(2*a - e))

    """
    if not isinstance(expr, Expr):
        return expr

    if denominator is None:
        numerator, denominator = fraction(expr)
    else:
        numerator = expr

    if not numerator.is_Add:
        return expr
    if denominator == 1:
        return expr
    return Add(*[a / denominator for a in numerator.args])


def collect_reciprocal(expr, term_to_collect, check=True):
    """
    Given an addition, collect the specified term from the addends.
    This is different from sympy's ``collect``, in fact it doesn't use it
    at all. While's ``collect`` requires the term to be collected to be 
    contained by some two or more terms of an addition, this function 
    requires at does not.
    See examples below to understand the goal of this function.

    Parameters
    ----------
    expr : Expr
        The addition to modify.
    term_to_collect : Expr
    check : boolean, optional
        If True, verify that the new expression is mathematically
        equivalent to `expr`. If their are not, or the equivalency
        could not be established, a warning will be shown, but the
        function will returned the modified expression.
    
    Return
    ------
    new_expr : Mul

    Examples
    --------

    >>> from sympy import symbols
    >>> from sympy_equation import collect_reciprocal
    >>> a = symbols("a")
    >>> expr = a + 1
    >>> collect_reciprocal(expr, a)
    a*(1 + 1/a)

    >>> v1, v2, gamma = symbols("v1, v2, gamma")
    >>> expr = -1 + v2*(gamma + 1)/(v1*(gamma - 1))
    >>> collect_reciprocal(expr, v2/v1)
    v2*(-v1/v2 + (gamma + 1)/(gamma - 1))/v1
    """
    if not isinstance(expr, Expr):
        return
    if not expr.is_Add:
        return expr
    if not any(a.has(term_to_collect) for a in expr.args):
        return expr

    reciprocal = 1 / term_to_collect
    new_add = Add(*[a * reciprocal for a in expr.args])
    new_expr = term_to_collect * new_add
    if check and (not new_expr.equals(expr)):
        warnings.warn(
            "The new expression is mathematically different"
            " from the original expression (or its equivalency"
            " could not be established). Watch out!",
            stacklevel=1
        )
    return new_expr


def split_two_terms_add(eq):
    """
    Consider an equation having one of the following forms:

    * ``a + b = 0``: LHS is an addition of two terms.
    * ``0 = a + b``: RHS is an addition of two terms.

    This function splits the addition and places the terms
    on the two sides of the equation: ``a = -b``.

    If the equation doesn't have the expected form, the function 
    returns it unmodified.

    Parameters
    ----------
    eq : Expr, Equation, Equality

    Returns
    -------
    new_eq : Equation, Equality
    """
    if not isinstance(eq, (Expr, Equation, Equality)):
        return eq

    if isinstance(eq, Expr):
        addition = eq
    else:
        if not (eq.lhs.is_Add ^ eq.rhs.is_Add):
            # at most one side must be an addition
            return eq

        if not ((eq.lhs == 0) ^ (eq.rhs == 0)):
            # at most one side must be zero
            return eq

        addition = eq.lhs if eq.lhs.is_Add else eq.rhs

    if len(addition.args) != 2:
        return eq
    class_ = Equation if isinstance(eq, Expr) else eq.func
    return class_(addition.args[0], -addition.args[1])
