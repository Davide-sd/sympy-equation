import param
import warnings
from sympy import (
    Basic, Expr, latex, postorder_traversal, count_ops, Add, fraction
)
from numbers import Number as PythonNumber
from typing import Callable, List
from contextlib import contextmanager


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


class _TableCommon(param.Parameterized):
    expr = param.ClassSelector(class_=Basic, doc="""
        The symbolic expression whose nodes are to be shown""")
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
    expressions = param.List(default=[], item_type=Basic, readonly=True, doc="""
        List of sub-expressions composing `expr`.""")
    selected_idx = param.List(default=[], item_type=int, readonly=True, doc="""
        Get the indices of the expressions that were filtered 
        by ``select``.""")
    column_labels = param.List(default=["idx", "expr"], bounds=(2, 2), doc="""
        Labels to be shown on the header of the table.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._extract_expressions_from_expr()
        self._select_expressions()
        if self.auto_show:
            self.show()

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


class table_of_nodes(_TableCommon):
    """
    Nicely print the nodes of a symbolic expression as a table with two 
    columns: an index, and the node itself. The index can later be used 
    to retrive the node we are interested in, without having to resort
    to pattern matching operations.

    This class uses :py:func:`sympy.postorder_traversal` in order to
    retrieve all nodes of the expression tree. The larger the expression,
    the greater the number of nodes. There are two disadvantages in using
    this class for large or huge expressions:

    1. screen space: if ``auto_show=True`` the table will be automatically
       visualized on the screen. The larger the expression tree, the more
       time to show it on the screen and the more space will be used.
       This can be mitigated by filtering the table with the ``select``
       keyword arguments (see examples below).
    2. memory usage.

    Examples
    --------

    Let's consider a simple expression. By default, this class is going
    to visualize all nodes of the expression tree:

    >>> from sympy import symbols
    >>> from sympy_equation import table_of_nodes
    >>> a, b, c = symbols("a, b, c")
    >>> expr = c * (a - b)
    >>> ton = table_of_nodes(expr, use_latex=False)
    idx   | nodes    
    ------|----------
    0     | a        
    1     | b        
    2     | c        
    3     | -1       
    4     | -b       
    5     | a - b    
    6     | c*(a - b)

    Let's say we would like to extract the node ``a - b``. Then, we index
    the table:

    >>> ton[5]
    a - b

    Let's see a useful application of this class. There are situations where 
    we might be dealing with relatively complex and large expressions. 
    Suppose the following expression is the result of a symbolic integration:

    >>> L, mdot, q, T_in, c_p, n, xi = symbols("L, mdot, q, T_in, cp, n, xi")
    >>> expr = L**2*mdot**2*q**2*(L*q + T_in*mdot*c_p)**(n*xi) + 2*L*T_in*mdot**3*c_p*q*(L*q + T_in*mdot*c_p)**(n*xi) + T_in**2*mdot**4*c_p**2*(L*q + T_in*mdot*c_p)**(n*xi) - mdot**(n*xi + 4)*(T_in*c_p)**(n*xi + 2)

    This addition is composed of 4 terms. 3 of them share a common term
    that can be collected, ``(L*q + T_in*mdot*c_p)**(n*xi)``. Instead of typing
    it directly and risk inserting typing errors, we can extract it from
    the expression tree. However, in this case the expression tree is large.
    We can filter the table with the ``select`` keyword:

    >>> ton = table_of_nodes(expr, select=[L*q], use_latex=False)
    idx   | nodes                                                                                                                                                                                                  
    ------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    12    | L*q                                                                                                                                                                                                    
    26    | L*q + T_in*c_p*mdot                                                                                                                                                                                    
    27    | (L*q + T_in*c_p*mdot)**(n*xi)                                                                                                                                                                          
    29    | L**2*mdot**2*q**2*(L*q + T_in*c_p*mdot)**(n*xi)                                                                                                                                                        
    30    | T_in**2*c_p**2*mdot**4*(L*q + T_in*c_p*mdot)**(n*xi)                                                                                                                                                   
    31    | 2*L*T_in*c_p*mdot**3*q*(L*q + T_in*c_p*mdot)**(n*xi)                                                                                                                                                   
    32    | L**2*mdot**2*q**2*(L*q + T_in*c_p*mdot)**(n*xi) + 2*L*T_in*c_p*mdot**3*q*(L*q + T_in*c_p*mdot)**(n*xi) + T_in**2*c_p**2*mdot**4*(L*q + T_in*c_p*mdot)**(n*xi) - mdot**(n*xi + 4)*(T_in*c_p)**(n*xi + 2)

    The above output shows all nodes of the expression tree containing 
    the ``L*q`` term. From this table we can quickly select the target node:

    >>> e.collect(ton[27])
    -mdot**(n*xi + 4)*(T_in*c_p)**(n*xi + 2) + (L*q + T_in*c_p*mdot)**(n*xi)*(L**2*mdot**2*q**2 + 2*L*T_in*c_p*mdot**3*q + T_in**2*c_p**2*mdot**4)

    This is now an addition of 2 terms.

    See Also
    --------
    table_of_arguments
    """

    def __init__(self, expr, **params):
        params.setdefault("column_labels", ["idx", "nodes"])
        super().__init__(expr=expr, **params)

    @param.depends("expr", watch=True)
    def _extract_expressions_from_expr(self):
        with edit_readonly(self):
            self.expressions = sorted(
                set(postorder_traversal(self.expr)),
                # NOTE: sort by operation count, then by string representation
                # for tie-breaking. This key should ensure deterministic ordering
                key=lambda expr: (count_ops(expr), str(expr))
            )


class table_of_arguments(_TableCommon):
    """
    Nicely print the arguments of a symbolic expression as a table with two 
    columns: an index, and the argument itself. The index can later be used 
    to retrive the argument we are interested in, without having to resort
    to pattern matching operations.

    This class extracts the arguments of a symbolic expression.
    There are two disadvantages in using this class for large or 
    huge expressions:

    1. screen space: if ``auto_show=True`` the table will be automatically
       visualized on the screen. The greater the number of arguments, the more
       time to show it on the screen and the more space will be used.
       This can be mitigated by filtering the table with the ``select``
       keyword arguments (see examples below).
    2. memory usage.

    Examples
    --------

    Let's consider a simple expression. By default, this class is going
    to visualize all arguments of the expression:

    >>> from sympy import symbols
    >>> from sympy_equation import table_of_arguments
    >>> a, b, c, d = symbols("a, b, c, d")
    >>> expr = (a / 2 + b / 3) * (c - d)
    >>> toa = table_of_arguments(expr, use_latex=False)
    idx   | args     
    ------|----------
    0     | c - d    
    1     | a/2 + b/3

    Here, ``expr`` is a multiplication of two terms. Let's say we would like 
    to extract the argument ``a/2 + b/3``. Then, we index the table:

    >>> toa[1]
    a/2 + b/3

    Let's consider a different example, this time an addition. Here, we will 
    filter the table using the ``select`` keyword argument in order visualize
    only the arguments containing ``gamma``:

    >>> p1, p2, v1, v2, gamma = symbols("p1, p2, v1, v2, gamma")
    >>> expr = gamma/2 - gamma*v2/(2*v1) + gamma*p2/(2*p1) - gamma*p2*v2/(2*p1*v1) + Rational(1, 2) + v2/(2*v1) - p2/(2*p1) + p2*v2/(2*p1*v1)
    >>> toa = table_of_arguments(expr, use_latex=False, filter=[gamma])
    idx   | args                  
    ------|-----------------------
    1     | gamma/2               
    4     | gamma*p2/(2*p1)       
    5     | -gamma*v2/(2*v1)      
    7     | -gamma*p2*v2/(2*p1*v1)

    We can index the table like before, or we can also get all the selected
    expressions:

    >>> selected_expressions = toa.get_selected_expressions()
    [gamma/2, gamma*p2/(2*p1), -gamma*v2/(2*v1), -gamma*p2*v2/(2*p1*v1)]
    
    And then apply some operation, for example:

    >>> new_add = sum(selected_expressions)
    >>> new_add
    gamma/2 - gamma*v2/(2*v1) + gamma*p2/(2*p1) - gamma*p2*v2/(2*p1*v1)
    >>> new_add.factor()
    gamma*(p1 + p2)*(v1 - v2)/(2*p1*v1)

    See Also
    --------
    table_of_nodes
    """

    def __init__(self, expr, **params):
        params.setdefault("column_labels", ["idx", "args"])
        super().__init__(expr=expr, **params)

    @param.depends("expr", watch=True)
    def _extract_expressions_from_expr(self):
        with edit_readonly(self):
            self.expressions = list(self.expr.args)


def process_arguments_of_add(expr, indices_groups, func, check=True):
    """
    Given an addition composed of several terms, this function performs
    the following:
    
    1. select a group of terms.
    2. add them together to create.
    3. apply `func` to the result of step 2, which will compute a 
       new expression.
    4. replace the addition of step 2 with the new expression.

    the function ``table_of_arguments`` can be used to select the 
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

    >>> from sympy import symbols
    >>> from sympy_equation import process_arguments_of_add, table_of_arguments
    >>> gamma, v1, v2, p1, p2 = symbols("gamma, v1, v2, p1, p2")
    >>> expr = gamma - gamma*v2/v1 + gamma*p2/p1 + 1 + v2/v1 - p2/p1
    >>> table_of_arguments(expr, use_latex=False)
    index | expr        
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
    table_of_arguments
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
    a/(2*a - e) + b/(2*a - e) - c/(d*(2*a - e))

    """
    if not isinstance(expr, Expr):
        return expr

    if denominator is None:
        numerator, denominator = fraction(expr)
    else:
        numerator = expr

    if not denominator.is_Add:
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
