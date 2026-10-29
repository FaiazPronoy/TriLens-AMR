$pdf_mode = 1;
$bibtex_use = 2;

add_cus_dep('nlo', 'nls', 0, 'makenomencl');

sub makenomencl {
    my ($base) = @_;
    return system("makeindex -s nomencl.ist -o \"$base.nls\" \"$base.nlo\"");
}

$clean_ext .= " nls nlo bbl run.xml synctex.gz";
