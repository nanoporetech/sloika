projectVersion:=$(shell ./scripts/show-version.sh)
ifndef projectVersion
$(error $${projectVersion} is empty (not set))
endif

pyDirs:=sloika test bin models misc
pyFiles:=$(shell find *.py ${pyDirs} -type f -regextype sed -regex ".*\.py")

include Makefile.common

.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python3-virtualenv python3-pip python3-setuptools  git \
	    libblas3 libblas-dev python3-dev lsb-release virtualenv

.PHONY: workflow
workflow:
	${inDevEnv} $${SCRIPTS_DIR}/workflow.sh
	${inEnv} if [[ ! -e $${BUILD_DIR}/workflow/training/model_final.pkl ]]; then exit 1; fi
