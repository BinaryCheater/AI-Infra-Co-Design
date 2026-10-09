.PHONY: all render clean

all:
	tectonic paper.tex --keep-logs

render: all
	mkdir -p build/previews
	pdftoppm -r 110 -png paper.pdf build/previews/page

clean:
	rm -f paper.aux paper.log paper.out paper.xdv paper.synctex.gz
