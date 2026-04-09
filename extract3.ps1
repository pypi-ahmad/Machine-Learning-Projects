# Deep scan - look for the actual y= or target column more carefully
$checks = @(
    @("Spam Email Classification", "spam_email_classification.ipynb"),
    @("Advanced Credit Card Fraud Detection", "Handling_Imbalanced_Data-Under_Sampling.ipynb"),
    @("SONAR Rock vs Mine Prediction", "SONAR Rock vs Mine Prediction Pyspark.ipynb"),
    @("Marketing Campaign Prediction", "predict-response-by-using-random forest(1).ipynb"),
    @("Titanic Survival Prediction", "Untitled.ipynb"),
    @("Groundhog Day Predictions", "ground-hog-day-predictions(1).ipynb"),
    @("Cotton Disease Prediction", "Cotton_Disease_inceptionv3.ipynb"),
    @("Bayesian Logistic Regression - Bank Marketing", "Bayesian Logistic Regression_bank marketing.ipynb"),
    @("H2O Higgs Boson", "H2O Higgs Boson.ipynb"),
    @("Traffic Congestion Prediction", "traffic_prediction.ipynb"),
    @("Income Classification", "Untitled.ipynb"),
    @("Autoencoder Fashion MNIST", "Autoencoder_Fashion_MNIST.ipynb"),
    @("CIFAR-10 Classification", "04_image_classification_with_CNN(Colab).ipynb"),
    @("Titanic - Handling Missing Values", "Missing _Value3.ipynb"),
    @("Hand Digit Recognition", "Untitled.ipynb"),
    @("Logistic Regression Balanced", "Logistic Regression balanced.ipynb")
)

foreach ($item in $checks) {
    $nb = "E:\Github\Machine-Learning-Projects\Classification\$($item[0])\$($item[1])"
    if (Test-Path $nb) {
        $c = Get-Content $nb -Raw
        Write-Host "=== $($item[0]) ==="
        # Find lines with column/feature/target references
        $lines = [regex]::Matches($c, '(?:columns|features|target|label|CLASS|class|output|prediction|predict|response|Response|survived|Survived)[\w]*[^\\n]{0,80}') | ForEach-Object { $_.Value } | Select-Object -First 10
        foreach ($l in $lines) { Write-Host "  $l" }
        # Also search for .pop or specific variable assignments
        $pop = [regex]::Matches($c, '\.pop\(\s*[''"](\w+)[''"]\s*\)') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        if ($pop) { Write-Host "  pop: $($pop -join ',')" }
        # StringIndexer or VectorAssembler
        $si = [regex]::Matches($c, 'StringIndexer\([^)]*outputCol\s*=\s*[''"](\w+)[''"]') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        if ($si) { Write-Host "  StringIndexer output: $($si -join ',')" }
        $cols = [regex]::Matches($c, "columns\(\)\s*\[\s*(-?\d+)\s*\]") | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        if ($cols) { Write-Host "  columns[idx]: $($cols -join ',')" }
        Write-Host ""
    }
}
