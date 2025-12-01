# -*- makefile -*-
.PHONY: build clean test docs livehtml

src_dir = atmosp
docs_dir = docs

MODULE = atmosp

# build package
build:
	python -m build

# house cleaning
clean:
	rm -rf dist/* build/* *.egg-info

# run tests
test:
	python -m pytest atmosp/tests/

# make docs and open index
docs:
	$(MAKE) -C $(docs_dir) html
	open $(docs_dir)/build/html/index.html

# live docs reload
livehtml:
	$(MAKE) -C $(docs_dir) livehtml
