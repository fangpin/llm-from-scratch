# Documentation toolchain targets. Designed to be either included from a
# project's own Makefile (`include doc.mk`) or invoked directly:
#   make -f doc.mk docs DOC=<feishu-doc-url>

VENV := .venv
PY := $(VENV)/bin/python
SPHINX := $(VENV)/bin/sphinx-build

.PHONY: docs-install docs-sync docs-html docs docs-serve docs-clean

docs-install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements-docs.txt
	@command -v lark-cli >/dev/null 2>&1 && echo "lark-cli: $(shell lark-cli --version 2>/dev/null | head -1)" || { \
		echo "lark-cli not found, installing @larksuite/cli via npm..."; \
		command -v npm >/dev/null 2>&1 || { echo "npm is required to install lark-cli"; exit 1; }; \
		npm install -g @larksuite/cli; \
	}
	@echo "hint: 如果尚未登录飞书，请运行 lark-cli auth login"

ifndef FROM
docs-sync:
	$(PY) doc_scripts/sync_lark_doc.py $(if $(DOC),--doc "$(DOC)")
else
docs-sync:
	$(PY) doc_scripts/sync_lark_doc.py --from-file "$(FROM)"
endif

docs-html:
	$(SPHINX) -b html docs/source docs/_build/html

docs-export: docs-html
	$(PY) doc_scripts/export_platform_posts.py --image-base "$(or $(IMAGE_BASE),pages)"

docs: docs-sync docs-html docs-export
	@echo "open docs/_build/html/index.html"

docs-serve: docs-html
	$(PY) -m http.server 8000 --directory docs/_build/html

docs-clean:
	rm -rf docs/_build
