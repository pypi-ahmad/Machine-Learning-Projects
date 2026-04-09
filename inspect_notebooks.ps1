$base = "E:\Github\Machine-Learning-Projects\Classification"

# 1. Groundhog Day
$nb = Get-ChildItem "$base\Groundhog Day Predictions" -Filter *.ipynb | Select-Object -First 1
$raw = [IO.File]::ReadAllText($nb.FullName)
$lines = $raw -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|columns" | Select-Object -First 10
Write-Host "=== Groundhog Day ==="
foreach ($l in $lines) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 2. Weather
$nb2 = Get-ChildItem "$base\Weather Classification - Decision Trees" -Filter *.ipynb | Select-Object -First 1
$raw2 = [IO.File]::ReadAllText($nb2.FullName)
$lines2 = $raw2 -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|clean_data|humidity" | Select-Object -First 10
Write-Host "`n=== Weather ==="
foreach ($l in $lines2) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 3. Traffic
$nb3 = Get-ChildItem "$base\Traffic Congestion Prediction" -Filter *.ipynb | Select-Object -First 1
$raw3 = [IO.File]::ReadAllText($nb3.FullName)
$lines3 = $raw3 -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|X\s*=" | Select-Object -First 10
Write-Host "`n=== Traffic ==="
foreach ($l in $lines3) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 4. Income
$nb4 = Get-ChildItem "$base\Income Classification" -Filter *.ipynb | Select-Object -First 1
$raw4 = [IO.File]::ReadAllText($nb4.FullName)
$lines4 = $raw4 -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|income|salary" | Select-Object -First 10
Write-Host "`n=== Income ==="
foreach ($l in $lines4) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 5. Social Network Ads
$nb5 = Get-ChildItem "$base\Social Network Ads Analysis" -Filter *.ipynb | Select-Object -First 1
$raw5 = [IO.File]::ReadAllText($nb5.FullName)
$lines5 = $raw5 -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|iloc|Purchased" | Select-Object -First 10
Write-Host "`n=== Social Network Ads ==="
foreach ($l in $lines5) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 6. Earthquake
$nb6 = Get-ChildItem "$base\Earthquake Prediction" -Filter *.ipynb | Select-Object -First 1
$raw6 = [IO.File]::ReadAllText($nb6.FullName)
$lines6 = $raw6 -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|Magnitude|Depth" | Select-Object -First 10
Write-Host "`n=== Earthquake ==="
foreach ($l in $lines6) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 7. Flower Species
$nb7 = Get-ChildItem "$base\Flower Species Classification" -Filter *.ipynb | Select-Object -First 1
$raw7 = [IO.File]::ReadAllText($nb7.FullName)
$lines7 = $raw7 -split "`n" | Select-String -Pattern "y\s*=|\.drop|target|predict|species|variety" | Select-Object -First 10
Write-Host "`n=== Flower Species ==="
foreach ($l in $lines7) { $t=$l.Line.Trim(); Write-Host $t.Substring(0,[Math]::Min(150,$t.Length)) }

# 8. Data Sources for image projects
Write-Host "`n=== DATA SOURCES ==="
$imgProjects = @("Autoencoder Fashion MNIST","CIFAR-10 Classification","Cotton Disease Prediction","Dog vs Cat Classification","Fashion MNIST Analysis","Digit Recognition - MNIST Sequence","Plant Disease Recognition","Pneumonia Classification")
foreach ($p in $imgProjects) {
    $proj = "$base\$p"
    $nb = Get-ChildItem $proj -Filter "*.ipynb" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nb) {
        $c = [IO.File]::ReadAllText($nb.FullName)
        $ds = [regex]::Matches($c, "(?:load_data|keras\.datasets|ImageDataGenerator|flow_from_directory|image_dataset|torchvision|datasets\.load)") | ForEach-Object { $_.Value } | Select-Object -Unique
        Write-Host "$p : $($ds -join ', ')"
    } else {
        Write-Host "$p : NO NOTEBOOK FOUND"
    }
}
