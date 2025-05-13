import importlib.metadata
from sympy_equation.algebraic_equation import (
    equation_config,
    Equation,
    Eqn,
    solve,
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
]
