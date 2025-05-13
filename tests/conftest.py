import pytest
import IPython


@pytest.fixture
def ipython_shell():
    return IPython.get_ipython()
