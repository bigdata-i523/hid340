FILE=report

all:
	pdflatex ${FILE}
	bibtex ${FILE}
	pdflatex ${FILE}
	pdflatex ${FILE}

pack:
	rm -rf ${FILE}
	mkdir ${FILE}
	mkdir ${FILE}/tex
	mkdir ${FILE}/images
	cp ${FILE}.tex ${FILE}
	cp ${FILE}.bib ${FILE}
	cp ${FILE}.pdf ${FILE}
	cp tex/sig-alternate-2013.cls ${FILE}/tex
	cp -r images/*.pdf ${FILE}/images
	tar cvf ${FILE}.tar ${FILE}

clean:
	rm -rf *~ *.aux *.bbl *.dvi *.log *.out *.blg *.pdf *.toc *.fdb_latexmk *.fls *.fff *.lof *.lot *.ttt *.cut
	rm -rf _region_.*

view:
	open ${FILE}.pdf

# all dependce tracking taking care of by Latexmk
fast:
	latexmk -pdf ${FILE}

watch:
	latexmk -pvc -view=pdf ${FILE}

.PHONY: all clean view fast watch
