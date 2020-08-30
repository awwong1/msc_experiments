freeze:
	pip freeze | grep -v "pkg-resources" | tee requirements.txt
