$ErrorActionPreference = 'Stop'
$root = 'd:\Workspace\Github\Machine-Learning-Projects'
$auditDir = Join-Path $root 'audit_phase1'
New-Item -ItemType Directory -Force -Path $auditDir | Out-Null

$datasetExt = @('.csv','.tsv','.xlsx','.xls','.json','.parquet','.feather','.pkl','.pickle','.npy','.npz','.jpg','.jpeg','.png','.bmp','.tif','.tiff','.mp4','.avi','.wav','.txt','.data')
$outputExt = @('.pkl','.pickle','.joblib','.h5','.hdf5','.pt','.pth','.onnx','.sav','.model')
$projectDirs = Get-ChildItem -Path $root -Directory | Where-Object { $_.Name -notlike '.git' -and $_.Name -notlike 'audit_phase1' } | Sort-Object Name

$rows = @()
$linkRows = @()
$cleanupRows = @()

foreach($proj in $projectDirs){
  $projPath = $proj.FullName
  $relativeProj = $proj.FullName.Substring($root.Length+1)
  $files = Get-ChildItem -Path $projPath -Recurse -File -ErrorAction SilentlyContinue

  $ipynb = $files | Where-Object { $_.Extension -ieq '.ipynb' }
  $py = $files | Where-Object { $_.Extension -ieq '.py' }
  $datasets = $files | Where-Object { $datasetExt -contains $_.Extension.ToLower() }
  $artifacts = $files | Where-Object { $outputExt -contains $_.Extension.ToLower() }

  $textCandidates = $files | Where-Object { $_.Extension -match '^(?i)\.(ipynb|py|md|txt|rst|json)$' }
  $projectLinks = @()

  foreach($tf in $textCandidates){
    try{
      $lineNum=0
      Get-Content -Path $tf.FullName -ErrorAction Stop | ForEach-Object {
        $lineNum++
        $matches = [regex]::Matches($_,'(https?://[^\s\)\]""\'']+)')
        foreach($m in $matches){
          $url=$m.Value.TrimEnd('.',',',';')
          if($url){
            $relFile = $tf.FullName.Substring($root.Length+1)
            $projectLinks += [pscustomobject]@{
              project = $relativeProj
              file = $relFile
              line = $lineNum
              url = $url
            }
          }
        }
      }
    } catch {}
  }

  $hasLocalDataset = $datasets.Count -gt 0
  $hasLink = $projectLinks.Count -gt 0
  $datasetSource = if($hasLocalDataset){'local'} elseif($hasLink){'link_only'} else {'none_detected'}

  $purpose = $proj.Name

  $signalText = ($proj.Name + ' ' + (($ipynb | Select-Object -ExpandProperty Name) -join ' ') + ' ' + (($py | Select-Object -ExpandProperty Name) -join ' ')).ToLower()
  $mlTypes = @()
  if($signalText -match 'classification|classifier'){ $mlTypes += 'classification' }
  if($signalText -match 'regression|predict|prediction|forecast|arima|time series|lstm'){ $mlTypes += 'regression_or_forecasting' }
  if($signalText -match 'sentiment|nlp|text|spam|summarization|translation|tfidf|bow'){ $mlTypes += 'nlp' }
  if($signalText -match 'image|vision|opencv|cnn|mnist|face|mask|ocr|captcha|traffic sign|dog|cat|plant|pneumonia|digit'){ $mlTypes += 'cv' }
  if($signalText -match 'clustering|kmeans|segmentation'){ $mlTypes += 'clustering' }
  if($signalText -match 'recommend|recommender'){ $mlTypes += 'recommendation' }
  if($mlTypes.Count -eq 0){ $mlTypes += 'unspecified_from_names' }
  $mlTypes = $mlTypes | Select-Object -Unique

  $structureIssue = @()
  if($ipynb.Count -eq 0 -and $py.Count -eq 0){ $structureIssue += 'no_notebook_or_script' }
  if(($ipynb.Count -gt 0 -or $py.Count -gt 0) -and -not $hasLocalDataset -and -not $hasLink){ $structureIssue += 'no_dataset_or_link_detected' }
  if($ipynb.Count -gt 0 -and ($ipynb | Where-Object { $_.Name -match 'Untitled' }).Count -gt 0){ $structureIssue += 'contains_untitled_notebook' }
  if($structureIssue.Count -eq 0){ $structureIssue += 'none' }

  foreach($f in $files){
    $rel = $f.FullName.Substring($root.Length+1)
    if($f.Name -match '\(1\)\.ipynb$' -or $f.Name -match '^Untitled.*\.ipynb$'){
      $cleanupRows += [pscustomobject]@{project=$relativeProj; candidate=$rel; reason='duplicate_or_temp_notebook_name'}
    }
    if($f.Name -match '^link_to_dataset\.txt$|^linkt_to_dataset\.txt$|^link_to_test\.txt$'){
      $cleanupRows += [pscustomobject]@{project=$relativeProj; candidate=$rel; reason='manual_dataset_link_pointer'}
    }
    if($f.Extension -in @('.pkl','.pickle','.joblib','.h5','.pt','.pth','.onnx','.sav','.model')){
      $cleanupRows += [pscustomobject]@{project=$relativeProj; candidate=$rel; reason='model_or_output_artifact'}
    }
  }

  $rows += [pscustomobject]@{
    project = $relativeProj
    project_path = $relativeProj
    purpose_declared = $purpose
    purpose_evidence = $relativeProj
    notebooks = $ipynb.Count
    scripts_py = $py.Count
    dataset_files = $datasets.Count
    dataset_source = $datasetSource
    dataset_examples = (($datasets | Select-Object -First 5 | ForEach-Object { $_.FullName.Substring($root.Length+1) }) -join ' | ')
    dataset_link_count = $projectLinks.Count
    dataset_link_examples = (($projectLinks | Select-Object -First 5 | ForEach-Object { "$($_.file):$($_.line) -> $($_.url)" }) -join ' | ')
    output_artifacts = $artifacts.Count
    ml_type_declared_from_names = ($mlTypes -join '|')
    ml_type_evidence = (($ipynb | Select-Object -First 3 | ForEach-Object { $_.FullName.Substring($root.Length+1) }) + ($py | Select-Object -First 2 | ForEach-Object { $_.FullName.Substring($root.Length+1) })) -join ' | '
    structure_flags = ($structureIssue -join '|')
  }

  $linkRows += $projectLinks
}

$normMap = @{}
foreach($p in $projectDirs){
  $title = $p.Name -replace '^Machine Learning Project[s]?\s*\d*\s*-\s*','' -replace '^Project\s*\d+\s*-\s*',''
  $norm = ($title.ToLower() -replace '[^a-z0-9]+',' ').Trim()
  if(-not $normMap.ContainsKey($norm)){ $normMap[$norm] = @() }
  $normMap[$norm] += $p.Name
}
$dupRows = @()
foreach($k in $normMap.Keys){
  if($normMap[$k].Count -gt 1){
    $dupRows += [pscustomobject]@{normalized_title=$k; projects=($normMap[$k] -join ' | '); count=$normMap[$k].Count}
  }
}

$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $auditDir 'project_inventory.csv')
$linkRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $auditDir 'dataset_links.csv')
$cleanupRows | Sort-Object project,candidate -Unique | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $auditDir 'cleanup_candidates.csv')
$dupRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $auditDir 'duplicate_projects.csv')

$arch = @()
$arch += "Workspace: $root"
$arch += "Generated: $(Get-Date -Format s)"
$arch += "Total projects: $($projectDirs.Count)"
$arch += ""
$idx=0
foreach($p in $projectDirs){
  $idx++
  $arch += "$idx. $($p.Name)"
}
$arch | Set-Content -Encoding UTF8 -Path (Join-Path $auditDir 'workspace_architecture.txt')

Write-Output "DONE"
Write-Output "Projects=$($projectDirs.Count)"
Write-Output "Inventory=$(Join-Path $auditDir 'project_inventory.csv')"
Write-Output "Links=$(Join-Path $auditDir 'dataset_links.csv')"
Write-Output "Cleanup=$(Join-Path $auditDir 'cleanup_candidates.csv')"
Write-Output "Duplicates=$(Join-Path $auditDir 'duplicate_projects.csv')"
Write-Output "Architecture=$(Join-Path $auditDir 'workspace_architecture.txt')"