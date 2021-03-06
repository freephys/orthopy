VERSION=$(shell python3 -c "import orthopy; print(orthopy.__version__)")

default:
	@echo "\"make publish\"?"

tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

# https://packaging.python.org/distributing/#id72
upload: setup.py
	# Make sure we're on the master branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	rm -f dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel --universal
	gpg --detach-sign -a dist/*
	# https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
	twine upload dist/*.tar.gz
	twine upload dist/*.whl

publish: tag upload
