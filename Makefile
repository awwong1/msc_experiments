freeze:
	pip freeze | grep -v "pkg-resources" | tee requirements.txt
test:
	python3 -m unittest discover -s ./tests -p "test_*.py"
