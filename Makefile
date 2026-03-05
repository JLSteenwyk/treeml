install:
	python setup.py install

develop:
	python setup.py develop

test: test.unit test.integration

test.unit:
	python3 -m pytest -m "not integration"

test.integration:
	python3 -m pytest -m "integration"

test.fast:
	python3 -m pytest -m "not (integration or slow)"

test.coverage: coverage.unit coverage.integration

coverage.unit:
	python -m pytest --cov=./ -m "not integration" --cov-report=xml:unit.coverage.xml

coverage.integration:
	python -m pytest --cov=./ -m "integration" --cov-report=xml:integration.coverage.xml
