$ErrorActionPreference = 'Stop'
Set-Location 'd:\Workspace\Github\Machine-Learning-Projects'

$inv = Import-Csv '.\audit_phase1\project_inventory.csv'
$dup = Import-Csv '.\audit_phase1\duplicate_projects.csv'
$clean = Import-Csv '.\audit_phase1\cleanup_candidates.csv'
$links = Import-Csv '.\audit_phase1\dataset_links.csv'
$out = '.\audit_phase1\phase1_report.md'

$total = $inv.Count
$withNotebook = ($inv | Where-Object {[int]$_.notebooks -gt 0}).Count
$withScript = ($inv | Where-Object {[int]$_.scripts_py -gt 0}).Count
$localData = ($inv | Where-Object {$_.dataset_source -eq 'local'}).Count
$linkOnly = ($inv | Where-Object {$_.dataset_source -eq 'link_only'}).Count
$noneData = ($inv | Where-Object {$_.dataset_source -eq 'none_detected'}).Count
$broken = ($inv | Where-Object {$_.structure_flags -ne 'none'}).Count
$missingDataset = ($inv | Where-Object {$_.structure_flags -like '*no_dataset_or_link_detected*'}).Count

$lines = @()
$lines += '# Phase 1 - Full Workspace Parsing Report'
$lines += ''
$lines += '## Workspace Architecture'
$lines += "- Total projects parsed: $total"
$lines += "- Projects with notebooks: $withNotebook"
$lines += "- Projects with Python scripts: $withScript"
$lines += "- Dataset source = local: $localData"
$lines += "- Dataset source = link_only: $linkOnly"
$lines += "- Dataset source = none_detected: $noneData"
$lines += "- Projects with structural flags: $broken"
$lines += "- Projects flagged no_dataset_or_link_detected: $missingDataset"
$lines += ''
$lines += '## Project-wise Breakdown (All Projects)'
$lines += '| Project | Purpose (declared) | Dataset source | ML type (declared_from_names) | Structure flags | Evidence notebooks/scripts | Dataset evidence |'
$lines += '|---|---|---|---|---|---|---|'
foreach($r in $inv){
  $e1 = ($r.ml_type_evidence -replace '\|', '; ')
  $e2 = ($r.dataset_examples -replace '\|', '; ')
  $lines += "| $($r.project_path) | $($r.purpose_declared) | $($r.dataset_source) | $($r.ml_type_declared_from_names) | $($r.structure_flags) | $e1 | $e2 |"
}

$lines += ''
$lines += '## Missing Datasets / Structure Issues'
$lines += '| Project | Flags | Dataset source |'
$lines += '|---|---|---|'
foreach($r in ($inv | Where-Object {$_.structure_flags -ne 'none'})){
  $lines += "| $($r.project_path) | $($r.structure_flags) | $($r.dataset_source) |"
}

$lines += ''
$lines += '## Duplicate Projects'
$lines += '| Normalized title | Count | Projects |'
$lines += '|---|---:|---|'
if($dup.Count -eq 0){
  $lines += '| (none detected) | 0 | - |'
} else {
  foreach($d in $dup){
    $lines += "| $($d.normalized_title) | $($d.count) | $($d.projects) |"
  }
}

$lines += ''
$lines += '## Cleanup Candidates'
$lines += '| Project | Candidate | Reason |'
$lines += '|---|---|---|'
foreach($c in $clean){
  $lines += "| $($c.project) | $($c.candidate) | $($c.reason) |"
}

$lines += ''
$lines += '## Dataset Links (Extracted)'
$lines += '| Project | File | Line | URL |'
$lines += '|---|---|---:|---|'
foreach($l in ($links | Select-Object -First 400)){
  $lines += "| $($l.project) | $($l.file) | $($l.line) | $($l.url) |"
}
$lines += ''
$lines += '> Note: Dataset links section is capped to first 400 rows in this report for readability; full extracted links are in audit_phase1/dataset_links.csv.'

$lines | Set-Content -Path $out -Encoding UTF8
Write-Output "WROTE $out"
(Get-Content $out | Measure-Object -Line).Lines