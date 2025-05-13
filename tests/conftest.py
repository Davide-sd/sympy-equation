import pytest
from sympy_equation.preparser import init_ipython_session


@pytest.fixture
def ipython_shell():
    return init_ipython_session(create_shell=True)
