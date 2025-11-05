from sympy_equation import (
    Eqn,
    table_of_expressions,
    process_arguments_of_add,
    divide_term_by_term,
    collect_reciprocal,
    split_two_terms_add,
)
from sympy import symbols, Rational, factor, Eq, Mul
from contextlib import redirect_stdout
import io
import pytest
from unittest.mock import patch
from IPython.display import Markdown


a, b, c, d, e = symbols("a:e")
e1, e2, p1, p2, v1, v2 = symbols("e1, e2, p1, p2, v1, v2", real=True, positive=True)
Cv, R, T1, T2, gamma = symbols("C_v, R, T1, T2, gamma", real=True, positive=True)
hugoniot = Eqn(e2 - e1, (p1 + p2) / 2 * (v1 - v2))
eq1 = Eqn(
    p2*v2/(p1*v1),
    gamma/2 - gamma*v2/(2*v1) + gamma*p2/(2*p1) - gamma*p2*v2/(2*p1*v1) + Rational(1, 2) + v2/(2*v1) - p2/(2*p1) + p2*v2/(2*p1*v1)
)
expr1 = gamma - gamma*v2/v1 + gamma*p2/p1 + 1 + v2/v1 - p2/p1
expr2 = (gamma + 1 - v2*(gamma - 1)/v1 + p2*(gamma - 1)/p1)/(gamma - 1)
expr3 = a + b - c / d
expr4 = 1 / (gamma - 1) + gamma / (gamma - 1) + p2 / p1 - v2 / v1


@pytest.mark.parametrize("mode", ["args", "nodes"])
def test_table_of_expressions_wrong_expr_type(mode):
    pytest.raises(TypeError, lambda: table_of_expressions(0, mode=mode))
    pytest.raises(TypeError, lambda: table_of_expressions("test", mode=mode))
    pytest.raises(TypeError, lambda: table_of_expressions(pytest, mode=mode))


def test_table_of_expression_find_results():
    # verify that the results of find, which is a set, is sorted
    # according to the number of operations and the string representation
    found = expr1.find(Mul)
    assert len(found) == 4

    f = io.StringIO()
    with redirect_stdout(f):
        table_of_expressions(found, use_latex=False)
    output = f.getvalue()
    assert output == """idx   | exprs       
------|-------------
0     | v2/v1       
1     | -p2/p1      
2     | gamma*p2/p1 
3     | -gamma*v2/v1
"""


@pytest.mark.parametrize("expr, expected", [
    (
        hugoniot,
        """idx   | args                   
------|------------------------
0     | -e1 + e2               
1     | (p1/2 + p2/2)*(v1 - v2)
"""
    ),
    (
        hugoniot.lhs,
        """idx   | args
------|-----
0     | e2  
1     | -e1 
"""
    ),
    (
        hugoniot.rhs,
        """idx   | args       
------|------------
0     | v1 - v2    
1     | p1/2 + p2/2
"""
    ),
    (
        eq1.rhs,
        """idx   | args                  
------|-----------------------
0     | 1/2                   
1     | gamma/2               
2     | v2/(2*v1)             
3     | -p2/(2*p1)            
4     | gamma*p2/(2*p1)       
5     | -gamma*v2/(2*v1)      
6     | p2*v2/(2*p1*v1)       
7     | -gamma*p2*v2/(2*p1*v1)
"""
    ),
    (
        # NOTE: very important this test, because here I test that mode="args"
        # is not going to sort arguments
        expr4,
        """idx   | args             
------|------------------
0     | 1/(gamma - 1)    
1     | gamma/(gamma - 1)
2     | p2/p1            
3     | -v2/v1           
"""
    )
])
def test_table_of_expressions_mode_args_text(expr, expected):
    f = io.StringIO()
    with redirect_stdout(f):
        table_of_expressions(expr, mode="args", use_latex=False)
    output = f.getvalue()
    assert output == expected


@pytest.mark.parametrize("expr, expected", [
    (
        hugoniot,
        r"""| idx | args |
|:-----:|:------|
| 0 | $- e_{1} + e_{2}$ |
| 1 | $\left(\frac{p_{1}}{2} + \frac{p_{2}}{2}\right) \left(v_{1} - v_{2}\right)$ |
"""
    ),
    (
        eq1.rhs,
        r"""| idx | args |
|:-----:|:------|
| 0 | $\frac{1}{2}$ |
| 1 | $\frac{\gamma}{2}$ |
| 2 | $\frac{v_{2}}{2 v_{1}}$ |
| 3 | $- \frac{p_{2}}{2 p_{1}}$ |
| 4 | $\frac{\gamma p_{2}}{2 p_{1}}$ |
| 5 | $- \frac{\gamma v_{2}}{2 v_{1}}$ |
| 6 | $\frac{p_{2} v_{2}}{2 p_{1} v_{1}}$ |
| 7 | $- \frac{\gamma p_{2} v_{2}}{2 p_{1} v_{1}}$ |
"""
    )
])
def test_table_of_expressions_mode_args_markdown(expr, expected):
    # Patch display *at its import location*, i.e., IPython.display.display
    with patch("IPython.display.display") as mock_display:
        table_of_expressions(expr, mode="args", use_latex=True)

    # Ensure display was called exactly once
    mock_display.assert_called_once()

    # Retrieve the argument passed to display()
    arg = mock_display.call_args[0][0]

    # Verify it’s a Markdown instance with the expected content
    assert isinstance(arg, Markdown)
    assert arg.data == expected


@pytest.mark.parametrize("expr, auto_show, expected", [
    (
        hugoniot,
        True,
        """idx   | nodes                             
------|-----------------------------------
0     | e1                                
1     | e2                                
2     | p1                                
3     | p2                                
4     | v1                                
5     | v2                                
6     | -1                                
7     | -e1                               
8     | -e1 + e2                          
9     | -v2                               
10    | 1/2                               
11    | p1/2                              
12    | p2/2                              
13    | v1 - v2                           
14    | p1/2 + p2/2                       
15    | (p1/2 + p2/2)*(v1 - v2)           
16    | -e1 + e2 = (p1/2 + p2/2)*(v1 - v2)
"""
    ),
    (
        hugoniot,
        False,
        ""
    ),
    (
        eq1.rhs,
        True,
        """idx   | nodes                                                                                                              
------|--------------------------------------------------------------------------------------------------------------------
0     | gamma                                                                                                              
1     | p1                                                                                                                 
2     | p2                                                                                                                 
3     | v1                                                                                                                 
4     | v2                                                                                                                 
5     | -1                                                                                                                 
6     | 1/2                                                                                                                
7     | 1/p1                                                                                                               
8     | 1/v1                                                                                                               
9     | gamma/2                                                                                                            
10    | -1/2                                                                                                               
11    | v2/(2*v1)                                                                                                          
12    | -p2/(2*p1)                                                                                                         
13    | gamma*p2/(2*p1)                                                                                                    
14    | -gamma*v2/(2*v1)                                                                                                   
15    | p2*v2/(2*p1*v1)                                                                                                    
16    | -gamma*p2*v2/(2*p1*v1)                                                                                             
17    | gamma/2 - gamma*v2/(2*v1) + gamma*p2/(2*p1) - gamma*p2*v2/(2*p1*v1) + 1/2 + v2/(2*v1) - p2/(2*p1) + p2*v2/(2*p1*v1)
"""
    ),
    (
        eq1.rhs,
        False,
        ""
    )
])
def test_table_of_expressions_mode_nodes_text(expr, auto_show, expected):
    f = io.StringIO()
    with redirect_stdout(f):
        table_of_expressions(
            expr, mode="nodes", use_latex=False, auto_show=auto_show)
    output = f.getvalue()
    assert output == expected


@pytest.mark.parametrize("expr, auto_show, expected", [
    (
        hugoniot,
        True,
        r"""| idx | nodes |
|:-----:|:------|
| 0 | $e_{1}$ |
| 1 | $e_{2}$ |
| 2 | $p_{1}$ |
| 3 | $p_{2}$ |
| 4 | $v_{1}$ |
| 5 | $v_{2}$ |
| 6 | $-1$ |
| 7 | $- e_{1}$ |
| 8 | $- e_{1} + e_{2}$ |
| 9 | $- v_{2}$ |
| 10 | $\frac{1}{2}$ |
| 11 | $\frac{p_{1}}{2}$ |
| 12 | $\frac{p_{2}}{2}$ |
| 13 | $v_{1} - v_{2}$ |
| 14 | $\frac{p_{1}}{2} + \frac{p_{2}}{2}$ |
| 15 | $\left(\frac{p_{1}}{2} + \frac{p_{2}}{2}\right) \left(v_{1} - v_{2}\right)$ |
| 16 | $- e_{1} + e_{2} = \left(\frac{p_{1}}{2} + \frac{p_{2}}{2}\right) \left(v_{1} - v_{2}\right)$ |
"""
    ),
    (
        hugoniot,
        False,
        ""
    ),
    (
        eq1.rhs,
        True,
        r"""| idx | nodes |
|:-----:|:------|
| 0 | $\gamma$ |
| 1 | $p_{1}$ |
| 2 | $p_{2}$ |
| 3 | $v_{1}$ |
| 4 | $v_{2}$ |
| 5 | $-1$ |
| 6 | $\frac{1}{2}$ |
| 7 | $\frac{1}{p_{1}}$ |
| 8 | $\frac{1}{v_{1}}$ |
| 9 | $\frac{\gamma}{2}$ |
| 10 | $- \frac{1}{2}$ |
| 11 | $\frac{v_{2}}{2 v_{1}}$ |
| 12 | $- \frac{p_{2}}{2 p_{1}}$ |
| 13 | $\frac{\gamma p_{2}}{2 p_{1}}$ |
| 14 | $- \frac{\gamma v_{2}}{2 v_{1}}$ |
| 15 | $\frac{p_{2} v_{2}}{2 p_{1} v_{1}}$ |
| 16 | $- \frac{\gamma p_{2} v_{2}}{2 p_{1} v_{1}}$ |
| 17 | $\frac{\gamma}{2} - \frac{\gamma v_{2}}{2 v_{1}} + \frac{\gamma p_{2}}{2 p_{1}} - \frac{\gamma p_{2} v_{2}}{2 p_{1} v_{1}} + \frac{1}{2} + \frac{v_{2}}{2 v_{1}} - \frac{p_{2}}{2 p_{1}} + \frac{p_{2} v_{2}}{2 p_{1} v_{1}}$ |
"""
    ),
    (
        eq1.rhs,
        False,
        ""
    )
])
def test_table_of_expressions_mode_nodes_markdown(expr, auto_show, expected):
    # Patch display *at its import location*, i.e., IPython.display.display
    with patch("IPython.display.display") as mock_display:
        table_of_expressions(
            expr, mode="nodes", use_latex=True, auto_show=auto_show)

    # if auto_show=True, call_count=1, otherwise call_count=0
    assert mock_display.call_count == auto_show

    if auto_show:
        # Retrieve the argument passed to display()
        arg = mock_display.call_args[0][0]

        # Verify it’s a Markdown instance with the expected content
        assert isinstance(arg, Markdown)
        assert arg.data == expected


@pytest.mark.parametrize("expr, idx, expected", [
    (hugoniot, 0, e1),
    (hugoniot, 1, e2),
    (hugoniot, 8, e2 - e1),
    (hugoniot, 14, p1/2 + p2/2),
    (eq1.rhs, 0, gamma),
    (eq1.rhs, 8, 1/v1),
    (eq1.rhs, 16, -gamma*p2*v2/(2*p1*v1)),
])
def test_table_of_expressions_mode_nodes_getitem(expr, idx, expected):
    t = table_of_expressions(expr, mode="nodes", auto_show=False)
    assert t[idx] == expected
    assert len(t) == len(t.expressions)


@pytest.mark.parametrize("expr, select, expected", [
    (
        hugoniot,
        [],
        """idx   | nodes                             
------|-----------------------------------
0     | e1                                
1     | e2                                
2     | p1                                
3     | p2                                
4     | v1                                
5     | v2                                
6     | -1                                
7     | -e1                               
8     | -e1 + e2                          
9     | -v2                               
10    | 1/2                               
11    | p1/2                              
12    | p2/2                              
13    | v1 - v2                           
14    | p1/2 + p2/2                       
15    | (p1/2 + p2/2)*(v1 - v2)           
16    | -e1 + e2 = (p1/2 + p2/2)*(v1 - v2)
"""
    ),
    (
        hugoniot,
        [v2],
        """idx   | nodes                             
------|-----------------------------------
5     | v2                                
9     | -v2                               
13    | v1 - v2                           
15    | (p1/2 + p2/2)*(v1 - v2)           
16    | -e1 + e2 = (p1/2 + p2/2)*(v1 - v2)
"""
    ),
    (
        hugoniot,
        [v2, p1/2],
        """idx   | nodes                             
------|-----------------------------------
5     | v2                                
9     | -v2                               
11    | p1/2                              
13    | v1 - v2                           
14    | p1/2 + p2/2                       
15    | (p1/2 + p2/2)*(v1 - v2)           
16    | -e1 + e2 = (p1/2 + p2/2)*(v1 - v2)
"""
    )
])
def test_table_of_expressions_mode_nodes_filter(expr, select, expected):
    f = io.StringIO()
    with redirect_stdout(f):
        table_of_expressions(
            expr, mode="nodes", use_latex=False, auto_show=True, select=select)
    output = f.getvalue()
    assert output == expected


@pytest.mark.parametrize("expr, select, expected_idx", [
    ( hugoniot, [], [] ),
    ( hugoniot, [v2], [5, 9, 13, 15, 16] ),
    ( hugoniot, [v2, p1/2], [5, 9, 11, 13, 14, 15, 16] )
])
def test_table_of_expressions_filter_idx_selected_expressions_1(expr, select, expected_idx):
    t = table_of_expressions(
        expr, mode="nodes", use_latex=False, auto_show=False, select=select)
    assert t.idx_selected_expressions == expected_idx
    assert len(t.get_selected_expressions()) == len(expected_idx)


def test_table_of_expressions_filter_idx_selected_expressions_2():
    t = table_of_expressions(
        hugoniot, mode="nodes", use_latex=False, auto_show=False, select=[v2])
    assert t.get_selected_expressions() == [
        v2, -v2, v1 - v2, (p1/2 + p2/2)*(v1 - v2), hugoniot
    ]


@pytest.mark.parametrize("expr, indices_groups, func, expected", [
    (
        expr1 * expr2,
        [0, 1],
        factor,
        expr1 * expr2   # Same result because it is not an addition
    ),
    (
        expr1,
        [2, 3],
        factor,
        gamma - gamma*v2/v1 + 1 + v2/v1 + p2*(gamma - 1)/p1
    ),
    (
        expr1,
        [2, 5],
        factor,
        gamma + gamma*p2/p1 + 1 - v2*(gamma - 1)/v1 - p2/p1
    ),
    (
        expr1,
        [[2, 5], [3, 4]],
        factor,
        gamma + 1 - v2*(gamma - 1)/v1 + p2*(gamma - 1)/p1
    ),
    (
        expr1,
        [[3, 4], [2, 5]],
        factor,
        gamma + 1 - v2*(gamma - 1)/v1 + p2*(gamma - 1)/p1
    )
])
def test_process_arguments_of_add_1(expr, indices_groups, func, expected):
    res = process_arguments_of_add(expr, indices_groups, func)
    assert res.equals(expected)


def test_process_arguments_of_add_errors():
    # indices are not in the correct format
    pytest.raises(ValueError, lambda : process_arguments_of_add(expr1, {0, 1}, factor))


@pytest.mark.parametrize("expr, denominator, expected", [
    (expr2, None, gamma/(gamma - 1) + 1/(gamma - 1) - v2/v1 + p2/p1),
    (expr3, 2*a - e, a/(2*a - e) + b/(2*a - e) - c/(d*(2*a - e))),
    # the numerator is not an addition. Return expr unmodified
    ((expr3 + 1) * expr3 / (a - 1), None, (expr3 + 1) * expr3 / (a - 1))
])
def test_divide_term_by_term(expr, denominator, expected):
    res = divide_term_by_term(expr, denominator=denominator)
    assert res.equals(expected)


@pytest.mark.parametrize("expr, term_to_collect, expected", [
    (a + 1, a, a*(1 + 1/a)),
    (-1 + v2*(gamma + 1)/(v1*(gamma - 1)), v2/v1, v2*(-v1/v2 + (gamma + 1)/(gamma - 1))/v1),
    # expr is not an addition: return expr unmodified
    ((a + 1) * b, a, (a + 1) * b),
    # the term to collect is not part of any addend: return expr unmodified
    (a + 1, b, a + 1),
])
def test_collect_reciprocal(expr, term_to_collect, expected):
    res = collect_reciprocal(expr, term_to_collect)
    assert res.equals(expected)


@pytest.mark.parametrize("eq, expected", [
    # valid forms
    (a + b, Eqn(a, -b)),
    (a - b, Eqn(a, b)),
    (Eq(a + b, 0), Eq(a, -b)),   # LHS is addition
    (Eq(0, a + b), Eq(a, -b)),   # RHS is addition
    (Eq(0, a - b), Eq(a, b)),   # RHS is addition
    (Eqn(a + b, 0), Eqn(a, -b)),   # LHS is addition
    (Eqn(0, a + b), Eqn(a, -b)),   # RHS is addition
    (Eqn(0, a - b), Eqn(a, b)),   # RHS is addition

    # unchanged forms
    (a + b + c, a + b + c),
    (Eq(a, b), Eq(a, b)),        # no addition
    (Eq(a + b, c), Eq(a + b, c)),# neither side is zero
    (Eq(a + b + c, 0), Eq(a + b + c, 0)), # more than 2 terms
    (Eqn(a, b), Eqn(a, b)),        # no addition
    (Eqn(a + b, c), Eqn(a + b, c)),# neither side is zero
    (Eqn(a + b + c, 0), Eqn(a + b + c, 0)), # more than 2 terms

    # non-equation input
    ("not an equation", "not an equation"),
])
def test_split_two_terms_add(eq, expected):
    res = split_two_terms_add(eq)
    assert res == expected
    assert type(res) is type(expected)
