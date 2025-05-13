#!/usr/bin/env bash

echo 'Core tests:'
pdm run coverage run -m pytest
pdm run coverage html

echo 'Doc tests:'
pytest --ignore='tests' --ignore='Developer Testing' --ignore-glob='*old*'  --doctest-modules
