#!/bin/bash -ex

source environment

VIRTUALENV_DIR="${DEV_VIRTUALENV_DIR}"

source ${VIRTUALENV_DIR}/bin/activate

pip install pip --upgrade

pip install -r misc/requirements.txt

pip install -r setup-dev-env.txt

python setup.py develop
