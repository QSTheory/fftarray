# Makefile for Sphinx documentation

PYTHON = python
SPHINXBUILD = sphinx-build
# directory for all sphinx build files
BUILD_DIR = build
# directory for the local and collective doc
HTML_DIR = $(BUILD_DIR)/html
# directory for git archive, storing the all files for each version
SRC_DIR = $(BUILD_DIR)/src

.PHONY: help all_versions local clean versions_list

# default
help:
	@echo "Please use \`make <target>' where <target> is one of:"
	@echo "  clean           - Remove generated doc files and start fresh"
	@echo "  versions_list   - Generate $(HTML_DIR)/versions.json"
	@echo "  all_versions    - Build the docs for all versions"
	@echo "  local           - Build the docs for the current local version"

# clean build directory and generated files
clean:
	rm -rf $(BUILD_DIR) source/api/generated source/examples
	rm -f source/api/array.rst source/api/dimension.rst

# generate build/html/versions.json
versions_list:
	$(PYTHON) helpers/generate_versions.py

# build all versions of the documentation
all_versions: clean versions_list
	@echo "Fetching versions..."
	@VERSIONS=$$(jq -r '.[].version' $(HTML_DIR)/versions.json); \
	for VERSION in $$VERSIONS; do \
		echo "Building docs for $$VERSION"; \
		if git rev-parse --verify $$VERSION >/dev/null 2>&1; then \
			mkdir -p $(SRC_DIR)/$$VERSION; \
			cd .. && \
			git archive --format=tar --prefix=docs/$(SRC_DIR)/$$VERSION/ $$VERSION | tar -x; \
			cd docs/$(SRC_DIR)/$$VERSION/docs && \
			$(MAKE) local; \
			cd ../../../.. && \
			mkdir -p $(HTML_DIR)/$$VERSION; \
			mv $(SRC_DIR)/$$VERSION/docs/build/html/local/* $(HTML_DIR)/$$VERSION; \
		else \
			echo "Warning: Version $$VERSION not found. Skipping."; \
		fi; \
	done

# build local documentation (current version)
local: clean versions_list
	# generates source/examples
	$(PYTHON) helpers/create_nblinks.py
	# generates source/api/array.rst and source/api/dimension.rst
	$(PYTHON) helpers/parse_classes.py
	# build docs using sphinx
	$(SPHINXBUILD) --color -b html source $(HTML_DIR)/local -v
