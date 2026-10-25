# ---------------------------------------------------------------------------
#  latexmk configuration for the BracU thesis
# ---------------------------------------------------------------------------

# Build a PDF via pdflatex (not DVI/PS).
$pdf_mode = 1;

# biblatex + biber. latexmk normally auto-detects this from the .bcf file,
# but setting it explicitly avoids surprises if the .bcf is stale.
$bibtex_use = 2;

# ---------------------------------------------------------------------------
#  Nomenclature (nomencl package)
#
#  latexmk does not know about .nlo -> .nls out of the box. Without the rule
#  below, \printnomenclature renders an EMPTY page: the entries are written to
#  main.nlo but nothing ever converts them into main.nls.
# ---------------------------------------------------------------------------
add_cus_dep('nlo', 'nls', 0, 'makenomencl');

sub makenomencl {
    my ($base) = @_;
    return system("makeindex -s nomencl.ist -o \"$base.nls\" \"$base.nlo\"");
}

# NOTE: do NOT set $makeindex here. That variable controls the ordinary
# index (.idx -> .ind) and has nothing to do with the nomenclature. Pointing it
# at nomencl.ist would silently break a real index if one is ever added.

# ---------------------------------------------------------------------------
#  Files removed by `latexmk -c`
# ---------------------------------------------------------------------------
$clean_ext .= " nls nlo bbl run.xml synctex.gz";
