#!/bin/bash

echo "Running Black..."
black .

echo "Running Flake8..."
flake8 .

echo "Running Pylint..."
find . -name "*.py" | xargs pylint