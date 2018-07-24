#!/bin/bash -eu

export VERSION_ENV_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${VERSION_ENV_SCRIPT_DIR}/version-env.sh

echo -n ${PROJECT_VERSION}
