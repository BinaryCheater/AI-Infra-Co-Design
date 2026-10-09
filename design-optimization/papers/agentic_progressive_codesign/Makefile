.PHONY: all validate render clean

all: validate
	tectonic paper.tex --keep-logs

validate:
	python3 validate.py

render: all
	mkdir -p build/previews
	pdftoppm -r 110 -png paper.pdf build/previews/page

clean:
	rm -f paper.aux paper.log paper.out paper.xdv paper.synctex.gz
