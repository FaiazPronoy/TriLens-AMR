from trilens.determinants.amrfinder_install import DATABASE, SOFTWARE, db_files


def test_pins():
    assert SOFTWARE == "4.2.7" and DATABASE == "2026-03-24.1"


def test_db_index_keeps_files_only():
    html = (
        '<a href="/pathogen/x/">up</a><a href="changelogs/">c</a><a href="AMRProt.fa">a</a>'
        '<a href="version.txt">v</a><a href="?C=M;O=A">s</a><a href="AMRProt.fa">a</a>'
    )
    assert db_files(html) == ["AMRProt.fa", "version.txt"]
