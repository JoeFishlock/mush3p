#!/usr/bin/env bash

# Script to create virtual python environment called dev
# then install the developement dependencies in the new environment.

set -euo pipefail

python3 -m venv dev;
source dev/bin/activate;
pip install --upgrade pip;
pip install -r requirements/dev.txt;
