#!/usr/bin/bash

# This script should be run in the dev virtual environment
# source dev/bin/activate
# This script runs formatting, linting and static type checking.
# Then tests are run.
# Default behaviour is just to run on the directory agate.
# The specific files can be passed using a wildcard in quotes e.g. '*.py'

set -uo pipefail

echo "RUNNING WITH";
python3 --version;
which python3;
printf "\n"

echo "RUNNING BLACK";
eval "python3 -m black ${1:-agate}";
printf "\n"

echo "RUNNING PYLINT";
eval "python3 -m pylint --max-line-length 88 ${1:-agate}";
printf "\n"

echo "RUNNING MYPY";
eval "python3 -m mypy ${1:-agate}";
printf "\n"

echo "RUNNING PYTEST";
python3 -m pytest;
