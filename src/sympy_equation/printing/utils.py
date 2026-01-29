import inspect
import sys
from functools import update_wrapper
from sympy.printing.printer import Printer
from typing import Type


class _PrintFunction:
    """
    Function wrapper to replace ``**settings`` in the signature with printer defaults
    """
    def __init__(self, f, print_cls: Type[Printer]):
        # find all the non-setting arguments
        params = list(inspect.signature(f).parameters.values())
        assert params.pop(-1).kind == inspect.Parameter.VAR_KEYWORD
        self.__other_params = params

        self.__print_cls = print_cls
        update_wrapper(self, f)

    def __reduce__(self):
        # Since this is used as a decorator, it replaces the original function.
        # The default pickling will try to pickle self.__wrapped__ and fail
        # because the wrapped function can't be retrieved by name.
        return self.__wrapped__.__qualname__

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        settings = self.__print_cls.param.values()
        return inspect.Signature(
            parameters=self.__other_params + [
                inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
                for k, v in settings.items()
            ],
            return_annotation=self.__wrapped__.__annotations__.get('return', inspect.Signature.empty)  # type:ignore
        )


def print_function(print_cls):
    """ A decorator to replace kwargs with the printer settings in __signature__ """
    def decorator(f):
        if sys.version_info < (3, 9):
            # We have to create a subclass so that `help` actually shows the docstring in older Python versions.
            # IPython and Sphinx do not need this, only a raw Python console.
            cls = type(f'{f.__qualname__}_PrintFunction', (_PrintFunction,), {"__doc__": f.__doc__})
        else:
            cls = _PrintFunction
        return cls(f, print_cls)
    return decorator
