# Compile the Markdown files with pandoc

TARGETS=$(patsubst %.ipynb,%.html,$(patsubst %.md,%.html,$(wildcard *.md *.ipynb)))

all: $(TARGETS)

python-check:
	python -m compileall src

%.html: %.md
	pandoc -s --variable maxwidth=1000px --mathjax -o $@ $<

%.html: %.ipynb
	pandoc -s --variable maxwidth=1000px --mathjax -o $@ $<
