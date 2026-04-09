$projects = @{
    "Advanced Credit Card Fraud Detection" = "Handling_Imbalanced_Data-Under_Sampling.ipynb"
    "Autoencoder Fashion MNIST" = "Autoencoder_Fashion_MNIST.ipynb"
    "CIFAR-10 Classification" = "04_image_classification_with_CNN(Colab).ipynb"
    "Cotton Disease Prediction" = "Cotton_Disease_inceptionv3.ipynb"
    "Credit Card Fraud - Imbalanced Dataset" = "Hanling_Imbalanced_Data.ipynb"
    "Flower Species Classification" = "Untitled.ipynb"
    "Fraud Detection" = "Untitled.ipynb"
    "Groundhog Day Predictions" = "ground-hog-day-predictions(1).ipynb"
    "Hand Digit Recognition" = "Untitled.ipynb"
    "Healthcare Heart Disease Prediction" = "Untitled.ipynb"
    "Income Classification" = "Untitled.ipynb"
    "SONAR Rock vs Mine Prediction" = "SONAR Rock vs Mine Prediction Pyspark.ipynb"
    "Spam Email Classification" = "spam_email_classification.ipynb"
    "Titanic - Handling Missing Values" = "Missing _Value3.ipynb"
    "Titanic Survival Prediction" = "Untitled.ipynb"
    "Traffic Congestion Prediction" = "traffic_prediction.ipynb"
    "Weather Classification - Decision Trees" = "Weather Data Classification using Decision Trees.ipynb"
    "Marketing Campaign Prediction" = "predict-response-by-using-random forest(1).ipynb"
    "Social Network Ads Analysis" = "prediction-of-if-the-item-is-purchase(1).ipynb"
    "Bayesian Logistic Regression - Bank Marketing" = "Bayesian Logistic Regression_bank marketing.ipynb"
    "Employee Turnover Prediction" = "Untitled.ipynb"
    "Logistic Regression Balanced" = "Logistic Regression balanced.ipynb"
    "H2O Higgs Boson" = "H2O Higgs Boson.ipynb"
    "Earthquake Prediction" = "Untitled.ipynb"
}

foreach ($entry in $projects.GetEnumerator()) {
    $nb = "E:\Github\Machine-Learning-Projects\Classification\$($entry.Key)\$($entry.Value)"
    if (Test-Path $nb) {
        $c = Get-Content $nb -Raw
        $y1 = [regex]::Matches($c, '(?:y|Y|y_train|Y_train)\s*=\s*\w+\[\s*[''"](\w+)[''"]\s*\]') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        $d1 = [regex]::Matches($c, '\.drop\(\s*(?:columns\s*=\s*)?\[?\s*[''"](\w+)[''"]') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        $t1 = [regex]::Matches($c, 'target\s*=\s*[''"](\w+)[''"]') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        $iloc = [regex]::Matches($c, 'iloc\s*\[\s*:?\s*,\s*(-?\d+)\s*\]') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        $lbl = [regex]::Matches($c, '(?:label|Label|response|Response)\s*=\s*\w+\[\s*[''"](\w+)[''"]\s*\]') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        Write-Host "$($entry.Key)"
        Write-Host "  y=: $($y1 -join ',')"
        Write-Host "  drop=: $($d1 -join ',')"
        Write-Host "  target=: $($t1 -join ',')"
        Write-Host "  iloc=: $($iloc -join ',')"
        Write-Host "  label=: $($lbl -join ',')"
        Write-Host "---"
    } else {
        Write-Host "$($entry.Key) | FILE NOT FOUND"
        Write-Host "---"
    }
}
