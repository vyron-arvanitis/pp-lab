# Compile the Markdown files with pandoc

TARGETS=$(patsubst %.ipynb,%.html,$(patsubst %.md,%.html,$(wildcard *.md *.ipynb)))

all: $(TARGETS)

%.html: %.md
	pandoc -s --variable maxwidth=1000px --mathjax -o $@ $<

%.html: %.ipynb
	pandoc -s --variable maxwidth=1000px --mathjax -o $@ $<
