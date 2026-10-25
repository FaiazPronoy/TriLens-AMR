# =============================================================================
#  pack_for_kaggle.ps1
#
#  Makes the zip to upload as the Kaggle dataset "trilens-code":
#      upload\trilens-code.zip      the code folder, without data and results
#  and lists the data files to upload as the dataset "trilens-data".
#  The full re-run starts from an empty results folder, so no results are uploaded.
#
#  Run after setup_data.ps1:   right-click -> "Run with PowerShell"
#  Re-run it whenever the code changes, then upload a new dataset version.
# =============================================================================

$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$Code = Join-Path (Split-Path $Here -Parent) "code"
$Out  = Join-Path $Here "upload"
$Tmp  = Join-Path $env:TEMP "trilens_pack"

if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Out, $Tmp | Out-Null

function Make-Zip([string]$Folder, [string]$Zip) {
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    # tar.exe ships with Windows 10/11 and writes zip entries with forward slashes,
    # which Kaggle (Linux) unpacks correctly.
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($tar) {
        & tar.exe -a -c -f $Zip -C (Split-Path $Folder -Parent) (Split-Path $Folder -Leaf)
        if ($LASTEXITCODE -ne 0) { throw "tar failed for $Folder" }
    } else {
        Compress-Archive -Path $Folder -DestinationPath $Zip -Force
    }
    $mb = [math]::Round((Get-Item $Zip).Length / 1MB, 1)
    Write-Host "  [ok] $(Split-Path $Zip -Leaf)  ($mb MB)"
}

Write-Host "`n[1/2] code" -ForegroundColor Cyan
# Full paths for the top-level folders: a bare "data" or "experiments" would also
# drop src\data and src\experiments, which are code.
& robocopy $Code (Join-Path $Tmp "code") /E /NFL /NDL /NJH /NJS /NP `
    /XD (Join-Path $Code "data") (Join-Path $Code "results") (Join-Path $Code "experiments") `
        (Join-Path $Code "outputs") __pycache__ .ipynb_checkpoints | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed for code" }
Make-Zip (Join-Path $Tmp "code") (Join-Path $Out "trilens-code.zip")

Write-Host "`n[2/2] data: upload these 12 files directly (no zip) as ONE dataset 'trilens-data'" -ForegroundColor Cyan
$proc = Join-Path $Code "data\processed"
$gies = Join-Path $Code "data\Giessen_dataset"
$files = @()
foreach ($f in @("arcadia_v2_master_features.pkl", "arcadia_v2_gene_presence_absence.pkl",
                 "arcadia_v2_gene_filtered.pkl", "arcadia_v2_gene_filtered__mlst_holdout.pkl",
                 "arcadia_v2_gene_filtered__phylogeny_clean.pkl", "arcadia_v2_snp_matrix.pkl",
                 "mlst_lineage_holdout_split.pkl", "ported_cluster_labels_v2.pkl",
                 "cds_eggNog.tsv", "genes.gff")) { $files += (Join-Path $proc $f) }
foreach ($f in @("cip_ctx_ctz_gen_multi_data.csv", "cip_ctx_ctz_gen_pheno.csv")) { $files += (Join-Path $gies $f) }
$total = 0; $bad = 0
foreach ($p in $files) {
    $name = Split-Path $p -Leaf
    if (Test-Path $p) {
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 0); $total += $mb
        Write-Host ("  {0,-48} {1,6} MB" -f $name, $mb)
    } else {
        Write-Host "  MISSING: $name (run setup_data.ps1 first)" -ForegroundColor Red; $bad++
    }
}
Write-Host ("  total about {0:N1} GB" -f ($total / 1024))
Write-Host "  folders: $proc"
Write-Host "           $gies"
if ($bad -eq 0) { Write-Host "  all 12 files present" -ForegroundColor Green }

Remove-Item $Tmp -Recurse -Force
Write-Host "`ntrilens-code.zip is in: $Out" -ForegroundColor Green
Read-Host "Press Enter to close"
