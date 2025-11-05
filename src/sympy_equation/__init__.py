import importlib.metadata
from sympy_equation.algebraic_equation import (
    equation_config,
    Equation,
    Eqn,
    solve,
)
from sympy_equation.utils import (
    table_of_expressions,
    process_arguments_of_add,
    divide_term_by_term,
    collect_reciprocal,
    split_two_terms_add,
)

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    pass


__all__ = [
    "equation_config",
    "Equation",
    "Eqn",
    "solve",
    "table_of_expressions",
    "process_arguments_of_add",
    "divide_term_by_term",
    "collect_reciprocal",
    "split_two_terms_add",
]
