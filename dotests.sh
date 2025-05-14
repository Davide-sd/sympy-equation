#!/usr/bin/env bash

echo 'Core tests:'
pdm run --venv for-test coverage run -m pytest
pdm run --venv for-test coverage html

echo 'Doc tests:'
pytest --venv for-test --ignore='tests' --doctest-modules
