"""
.. include:: ../ReadMe.md
.. include:: ../Development Notes.md
"""
__docformat__ = "numpy"
from algebra_with_sympy.algebraic_equation import (
    algwsym_config,
    Equation,
    Eqn,
    solve,
    solveset,
)

from algebra_with_sympy.version import __version__
algwsym_version = __version__

__all__ = [
    "algwsym_config",
    "Equation",
    "Eqn",
    "solve",
    "solveset",
]
