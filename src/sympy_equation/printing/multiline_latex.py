"""
The function ``multiline_latex`` comes directly from sympy, hence it is 
subject to sympy's license:


Copyright (c) 2006-2025 SymPy Development Team

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of SymPy nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

from sympy import Expr
from sympy_equation.printing.latex import extended_latex
from typing import Callable


def multiline_latex(
    lhs: Expr,
    rhs: Expr,
    terms_per_line: int=1,
    environment: str="eqnarray",
    use_dots: bool=False,
    latex_printer: Callable[[Expr], [str]]=None,
    **settings
):
    r"""
    This function generates a LaTeX equation with a multiline right-hand side
    in an ``align*``, ``eqnarray`` or ``IEEEeqnarray`` environment.

    It is an extension of :py:func:`sympy.printing.latex.multiline_latex`
    as it allows the user to customize the Latex printer.

    Parameters
    ==========

    lhs : Expr
        Left-hand side of equation
    rhs : Expr
        Right-hand side of equation
    terms_per_line : integer, optional
        Number of terms per line to print. Default is 1.
    environment : "string", optional
        Which LaTeX wnvironment to use for the output. Options are "align*"
        (default), "eqnarray", and "IEEEeqnarray".
    use_dots : boolean, optional
        If ``True``, ``\\dots`` is added to the end of each line. Default is ``False``.
    latex_printer : callable, optional
        A function generating latex code for the symbolic expressions.
        If not provided, :py:func:`extended_latex` will be used.
    settings : 
        Keyword arguments to be passed to ``latex_printer``.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, exp, log, I
    >>> from sympy_equation import multiline_latex
    >>> x, y, alpha = symbols('x y alpha')
    >>> f = Function("f")(x, y)
    >>> expr = sin(alpha*y) + exp(I*alpha) - cos(log(y)) + f.diff(y)
    >>> print(multiline_latex(x, expr))
    \begin{eqnarray}
    x & = & e^{i \alpha} \nonumber\\
    & & + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)} \nonumber\\
    & & + \frac{\partial}{\partial y} f{\left(x,y \right)} 
    \end{eqnarray}

    Using at most two terms per line:

    >>> print(multiline_latex(x, expr, 2))
    \begin{eqnarray}
    x & = & e^{i \alpha} + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)} + \frac{\partial}{\partial y} f{\left(x,y \right)} 
    \end{eqnarray}

    Customizing the printer in order to hide arguments from applied undefined
    functions and show their derivative as subscripts:

    >>> print(multiline_latex(x, expr, 2, applied_undef_args=None, derivative="subscript"))
    \begin{eqnarray}
    x & = & e^{i \alpha} + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)} + f_{y} 
    \end{eqnarray}

    Using ``align*`` and dots:

    >>> print(multiline_latex(x, expr, terms_per_line=2, environment="align*", use_dots=True))
    \begin{align*}
    x = & e^{i \alpha} + \sin{\left(\alpha y \right)} \dots\\
    & - \cos{\left(\log{\left(y \right)} \right)} + \frac{\partial}{\partial y} f{\left(x,y \right)} 
    \end{align*}

    Using ``IEEEeqnarray``:

    >>> print(multiline_latex(x, expr, environment="IEEEeqnarray"))
    \begin{IEEEeqnarray}{rCl}
    x & = & e^{i \alpha} \nonumber\\
    & & + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)} \nonumber\\
    & & + \frac{\partial}{\partial y} f{\left(x,y \right)} 
    \end{IEEEeqnarray}

    """
    if latex_printer is None:
        latex_printer = extended_latex

    # Based on code from https://github.com/sympy/sympy/issues/3001
    if environment == "eqnarray":
        result = r'\begin{eqnarray}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{eqnarray}'
        doubleet = True
    elif environment == "IEEEeqnarray":
        result = r'\begin{IEEEeqnarray}{rCl}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{IEEEeqnarray}'
        doubleet = True
    elif environment == "align*":
        result = r'\begin{align*}' + '\n'
        first_term = '= &'
        nonumber = ''
        end_term =  '\n\\end{align*}'
        doubleet = False
    else:
        raise ValueError("Unknown environment: {}".format(environment))

    dots = ''
    if use_dots:
        dots=r'\dots'

    terms = rhs.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = ''
        term_end = ''
        sign = '+'
        if term_count > terms_per_line:
            if doubleet:
                term_start = '& & '
            else:
                term_start = '& '
            term_count = 1
        if term_count == terms_per_line:
            # End of line
            if i < n_terms-1:
                # There are terms remaining
                term_end = dots + nonumber + r'\\' + '\n'
            else:
                term_end = ''

        if term.as_ordered_factors()[0] == -1:
            term = -1*term
            sign = r'-'
        if i == 0: # beginning
            if sign == '+':
                sign = ''
            result += r'{:s} {:s}{:s} {:s} {:s}'.format(
                latex_printer(lhs, **settings),
                first_term,
                sign,
                latex_printer(term, **settings),
                term_end
            )
        else:
            result += r'{:s}{:s} {:s} {:s}'.format(
                term_start,
                sign,
                latex_printer(term, **settings),
                term_end
            )
        term_count += 1

    result += end_term
    return result
