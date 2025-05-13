#!/usr/bin/env bash

pdm run coverage run -m pytest
pdm run coverage html