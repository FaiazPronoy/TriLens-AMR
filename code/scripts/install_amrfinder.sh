#!/usr/bin/env bash
# Installs AMRFinderPlus 4.2.7 (bioconda, via micromamba) and its database into $1 (default /tmp/amr).
# Needs internet: github.com (micromamba), conda.anaconda.org (packages), ftp.ncbi.nlm.nih.gov (database).
# Prints the path of the amrfinder binary on its last line.
set -euo pipefail
PREFIX="${1:-/tmp/amr}"
MM="$PREFIX/micromamba"
mkdir -p "$MM"
if [ ! -x "$MM/bin/micromamba" ]; then
    curl -sSL "https://github.com/mamba-org/micromamba-releases/releases/download/2.3.2-0/micromamba-linux-64.tar.bz2" \
        | tar -xj -C "$MM" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$MM/root"
if [ ! -x "$PREFIX/env/bin/amrfinder" ]; then
    "$MM/bin/micromamba" create -y -q -p "$PREFIX/env" -c conda-forge -c bioconda ncbi-amrfinderplus=4.2.7
fi
"$PREFIX/env/bin/amrfinder" -u >&2
"$PREFIX/env/bin/amrfinder" --version >&2
echo "$PREFIX/env/bin"
