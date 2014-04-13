#!/bin/bash

rm *aux
rm *bbl
pdflatex main.tex > /dev/null
bibtex main.aux
pdflatex main.tex > /dev/null 
pdflatex main.tex 
