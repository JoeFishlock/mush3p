#!/usr/bin/env bash

# Script to create virtual python environment called venv
# then install the user dependencies in the new environment.

set -euo pipefail

python -m venv venv;
source venv/bin/activate;
pip install --upgrade pip;
pip install -r requirements/requirements.txt;
