#!/usr/bin/env bash

echo 'Core tests:'
pdm run pytest --ignore='Developer Testing' --ignore-glob='*test_preparser.py' --ignore-glob='*test_numerics.py'

echo 'Preparser and numerics tests (require ipython environment):'
pdm run ipython -m pytest tests/test_preparser.py tests/test_numerics.py

echo 'Doc tests:'
pdm run pytest --ignore='tests' --ignore='Developer Testing' --ignore-glob='*old*'  --doctest-modules
