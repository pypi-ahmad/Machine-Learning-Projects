"""
Master overhaul script: generates modern pipeline.py for every project.
Run: python _overhaul_all.py

Families & new models (April 2026):
  1. Tabular Classification  → CatBoost(GPU), LightGBM(GPU), XGBoost(CUDA), FLAML
  2. Tabular Regression       → CatBoostReg(GPU), LightGBMReg(GPU), XGBReg(CUDA), FLAML
  3. Fraud/Imbalanced         → CatBoost, LightGBM, XGBoost + threshold tuning
  4. Anomaly Detection        → PyOD 2 (ECOD, COPOD, IForest, SUOD)
  5. Clustering               → UMAP + HDBSCAN + GMM
  6. NLP Classification       → ModernBERT / XLM-R
  7. NLP Generation           → Qwen3-Instruct (Ollama)
  8. Image Classification     → DINOv2 / ConvNeXt V2
  9. CV Detection             → YOLO (Ultralytics)
 10. Face/Gesture             → MediaPipe + InsightFace
 11. OCR                      → PaddleOCR
 12. Recommendation           → implicit, LightFM, Sentence Transformers
 13. Time Series              → AutoGluon-TS, Chronos-Bolt, StatsForecast
 14. Reinforcement Learning   → PPO / SAC (Stable-Baselines3)
 15. Audio/Speech             → Whisper, Wav2Vec2, XTTS-v2
"""
import json, os, sys, textwrap
from pathlib import Path

BASE = Path(r"e:\Github\Machine-Learning-Projects")
sys.path.insert(0, str(BASE))

# ════════════════════════════════════════════════════════════════════════
# FAMILY 1: TABULAR CLASSIFICATION
# New models: CatBoost(GPU), LightGBM(GPU), XGBoost(CUDA), FLAML AutoML
# ════════════════════════════════════════════════════════════════════════
TABULAR_CLF_PROJECTS = {
    "Classification/Adult Salary Prediction": {"csv": "adult_data.csv", "target": "salary"},
    "Classification/Breast Cancer Detection": {"csv": "__sklearn__", "target": "target", "loader": "sklearn.datasets.load_breast_cancer"},
    "Classification/Breast Cancer Prediction": {"csv": "Wisconsin-bc-data.csv", "target": "diagnosis", "drop": ["id", "Unnamed: 32"]},
    "Classification/Credit Risk Modeling - German Credit": {"csv": "__registry__", "target": "Risk", "registry_key": "credit_risk_modeling"},
    "Classification/Customer Churn Prediction - Telecom": {"csv": "__registry__", "target": "customerstatus", "registry_key": "customer_churn_telecom"},
    "Classification/Customer Lifetime Value Prediction": {"csv": "__registry__", "target": "LTVCluster", "registry_key": "customer_lifetime_value"},
    "Classification/Diabetes Classification": {"csv": "__inline__", "target": "Outcome"},
    "Classification/Diabetes ML Analysis": {"csv": "diabetes2.csv", "target": "Outcome"},
    "Classification/Drinking Water Potability": {"csv": "water_potability.csv", "target": "Potability"},
    "Classification/Drug Classification": {"csv": "__registry__", "target": "Drug", "registry_key": "drug_classification"},
    "Classification/Employee Turnover Analysis": {"csv": "HR_comma_sep.csv", "target": "left"},
    "Classification/Employee Turnover Prediction": {"csv": "dataset.csv", "target": "left"},
    "Classification/Flower Species Classification": {"csv": "__sklearn__", "target": "target", "loader": "sklearn.datasets.load_iris"},
    "Classification/Glass Classification": {"csv": "__registry__", "target": "Type", "registry_key": "glass_classification"},
    "Classification/Groundhog Day Predictions": {"csv": "dataset.csv", "target": "PunxPhil"},
    "Classification/Hand Digit Recognition": {"csv": "__sklearn__", "target": "target", "loader": "sklearn.datasets.load_digits"},
    "Classification/Healthcare Heart Disease Prediction": {"csv": "heart.csv", "target": "target"},
    "Classification/Heart Disease Prediction": {"csv": "heart.csv", "target": "target"},
    "Classification/Income Classification": {"csv": "income_evaluation.csv", "target": "income"},
    "Classification/Iris Dataset Analysis": {"csv": "Iris.csv", "target": "Species", "drop": ["Id"]},
    "Classification/Loan Default Prediction": {"csv": "__registry__", "target": "Status", "registry_key": "loan_default_prediction"},
    "Classification/Loan Prediction Analysis": {"csv": "Loan Prediction Dataset.csv", "target": "Loan_Status"},
    "Classification/Logistic Regression Balanced": {"csv": "banking.csv", "target": "y"},
    "Classification/Marketing Campaign Prediction": {"csv": "marketing_campaign.csv", "target": "Response"},
    "Classification/Mobile Price Classification": {"csv": "__registry__", "target": "price_range", "registry_key": "mobile_price_classification"},
    "Classification/Simple Classification Problem": {"csv": "__inline__", "target": "fruit_label"},
    "Classification/Social Network Ads Analysis": {"csv": "Social_Network_Ads.csv", "target": "Purchased"},
    "Classification/Student Performance Prediction": {"csv": "__registry__", "target": "G3", "registry_key": "student_performance"},
    "Classification/Titanic - Handling Missing Values": {"csv": "titanic.csv", "target": "survived"},
    "Classification/Titanic Survival Prediction": {"csv": "__seaborn__", "target": "survived", "loader": "seaborn.titanic"},
    "Classification/Weather Classification - Decision Trees": {"csv": "daily_weather.csv", "target": "high_humidity_label"},
    "Classification/Wine Quality Analysis": {"csv": "winequality.csv", "target": "quality"},
    "Classification/Wine Quality Prediction": {"csv": "winequality-red.csv", "target": "quality"},
    "Classification/Autoencoder for Customer Churn": {"csv": "WA_Fn-UseC_-Telco-Customer-Churn.csv", "target": "Churn"},
    "Classification/Bayesian Logistic Regression - Bank Marketing": {"csv": "banking.csv", "target": "y"},
    "Classification/Boston House Classification": {"csv": "__registry__", "target": "MEDV", "registry_key": "boston_house_classification"},
    "Classification/H2O Higgs Boson": {"csv": "training.csv", "target": "Label"},
    "Classification/Earthquake Prediction": {"csv": "database.csv", "target": "Magnitude"},
}

def gen_tabular_clf_pipeline(project_path, config):
    """Generate a modern tabular classification pipeline."""
    target = config["target"]
    csv = config.get("csv", "")
    drop_cols = config.get("drop", [])
    loader = config.get("loader", "")
    registry_key = config.get("registry_key", "")

    # Build data loading code
    if csv == "__sklearn__":
        module, func = loader.rsplit(".", 1)
        data_load = f'''    from {module} import {func}
    data = {func}()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["{target}"] = data.target'''
    elif csv == "__seaborn__":
        data_load = f'''    import seaborn as sns
    df = sns.load_dataset("titanic")'''
    elif csv == "__registry__":
        data_load = f'''    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from core.data_loader import load_dataset
    df = load_dataset("{registry_key}")'''
    elif csv == "__inline__":
        data_load = f'''    # Load data from the notebook's inline dataset or local file
    csv_files = list(Path(os.path.dirname(__file__)).glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["{target}"] = data.target'''
    else:
        data_load = f'''    data_dir = Path(os.path.dirname(__file__))
    csv_path = data_dir / "{csv}"
    if not csv_path.exists():
        csv_files = list(data_dir.glob("*.csv"))
        csv_path = csv_files[0] if csv_files else csv_path
    df = pd.read_csv(csv_path)'''

    drop_code = ""
    if drop_cols:
        drop_code = f'''
    # Drop irrelevant columns
    df.drop(columns={drop_cols}, inplace=True, errors="ignore")'''

    template = textwrap.dedent(f'''\
        """
        Modern Tabular Classification Pipeline (April 2026)
        Models: CatBoost (GPU), LightGBM (GPU), XGBoost (CUDA), FLAML AutoML
        """
        import os, sys, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        from sklearn.metrics import (
            accuracy_score, classification_report, f1_score,
            roc_auc_score, confusion_matrix
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        warnings.filterwarnings("ignore")

        TARGET = "{target}"


        def load_data():
        __DATA_LOAD____DROP_CODE__
            print(f"Dataset shape: {{df.shape}}")
            print(f"Target distribution:\\n{{df[TARGET].value_counts()}}")
            return df


        def preprocess(df):
            """Auto-preprocess: encode categoricals, handle missing values, split."""
            df = df.copy()

            # Drop rows where target is missing
            df.dropna(subset=[TARGET], inplace=True)

            # Encode target if non-numeric
            le_target = None
            if df[TARGET].dtype == "object" or df[TARGET].dtype.name == "category":
                le_target = LabelEncoder()
                df[TARGET] = le_target.fit_transform(df[TARGET])

            y = df[TARGET]
            X = df.drop(columns=[TARGET])

            # Separate numeric and categorical
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()

            # Fill missing
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
            for c in cat_cols:
                X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "unknown")

            # Ordinal encode categoricals (tree models handle this natively)
            if cat_cols:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[cat_cols] = oe.fit_transform(X[cat_cols])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 50 else None
            )
            print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")
            return X_train, X_test, y_train, y_test, le_target


        def train_and_evaluate(X_train, X_test, y_train, y_test):
            """Train CatBoost, LightGBM, XGBoost (all GPU) + FLAML AutoML."""
            results = {{}}
            n_classes = y_train.nunique()
            is_binary = n_classes == 2

            # ── 1. CatBoost (GPU) ──
            try:
                from catboost import CatBoostClassifier
                cb = CatBoostClassifier(
                    iterations=1000, learning_rate=0.05, depth=8,
                    task_type="GPU", devices="0",
                    eval_metric="AUC" if is_binary else "MultiClass",
                    early_stopping_rounds=50, verbose=100,
                    auto_class_weights="Balanced",
                )
                cb.fit(X_train, y_train, eval_set=(X_test, y_test))
                y_pred = cb.predict(X_test).flatten()
                results["CatBoost"] = {{"model": cb, "preds": y_pred}}
                print(f"\\n✓ CatBoost Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
            except Exception as e:
                print(f"✗ CatBoost failed: {{e}}")

            # ── 2. LightGBM (GPU) ──
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=1000, learning_rate=0.05, max_depth=8,
                    device="gpu", gpu_platform_id=0, gpu_device_id=0,
                    objective="binary" if is_binary else "multiclass",
                    class_weight="balanced", verbose=-1,
                    n_jobs=-1,
                )
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
                )
                y_pred = lgb_model.predict(X_test)
                results["LightGBM"] = {{"model": lgb_model, "preds": y_pred}}
                print(f"\\n✓ LightGBM Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
            except Exception as e:
                print(f"✗ LightGBM failed: {{e}}")

            # ── 3. XGBoost (CUDA) ──
            try:
                from xgboost import XGBClassifier
                xgb_model = XGBClassifier(
                    n_estimators=1000, learning_rate=0.05, max_depth=8,
                    device="cuda", tree_method="hist",
                    objective="binary:logistic" if is_binary else "multi:softmax",
                    eval_metric="auc" if is_binary else "mlogloss",
                    early_stopping_rounds=50, verbosity=1,
                    n_jobs=-1,
                )
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)], verbose=100,
                )
                y_pred = xgb_model.predict(X_test)
                results["XGBoost"] = {{"model": xgb_model, "preds": y_pred}}
                print(f"\\n✓ XGBoost Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
            except Exception as e:
                print(f"✗ XGBoost failed: {{e}}")

            # ── 4. FLAML AutoML ──
            try:
                from flaml import AutoML
                automl = AutoML()
                automl.fit(
                    X_train, y_train,
                    task="classification",
                    time_budget=120,
                    metric="accuracy",
                    verbose=1,
                    gpu_per_trial=1,
                )
                y_pred = automl.predict(X_test)
                results["FLAML"] = {{"model": automl, "preds": y_pred}}
                print(f"\\n✓ FLAML Best: {{automl.best_estimator}} - Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
            except Exception as e:
                print(f"✗ FLAML failed: {{e}}")

            return results


        def report(results, y_test, save_dir="."):
            """Print comparison report and save confusion matrices."""
            print("\\n" + "=" * 60)
            print("MODEL COMPARISON")
            print("=" * 60)
            best_name, best_acc = None, 0
            for name, res in results.items():
                y_pred = res["preds"]
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                print(f"\\n── {{name}} ──")
                print(f"  Accuracy: {{acc:.4f}}  |  F1 (weighted): {{f1:.4f}}")
                print(classification_report(y_test, y_pred, zero_division=0))

                if acc > best_acc:
                    best_acc, best_name = acc, name

                # Save confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"{{name}} Confusion Matrix")
                ax.set_ylabel("Actual")
                ax.set_xlabel("Predicted")
                fig.savefig(os.path.join(save_dir, f"confusion_{{name.lower().replace(' ', '_')}}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close(fig)

            print(f"\\n🏆 Best Model: {{best_name}} ({{best_acc:.4f}})")
            return best_name


        def main():
            print("=" * 60)
            print("MODERN TABULAR CLASSIFICATION PIPELINE")
            print("Models: CatBoost(GPU) | LightGBM(GPU) | XGBoost(CUDA) | FLAML")
            print("=" * 60)

            df = load_data()
            X_train, X_test, y_train, y_test, le_target = preprocess(df)
            results = train_and_evaluate(X_train, X_test, y_train, y_test)

            if results:
                save_dir = os.path.dirname(os.path.abspath(__file__))
                report(results, y_test, save_dir)
            else:
                print("\\n⚠ No models trained successfully.")


        if __name__ == "__main__":
            main()
    ''')
    return template.replace("__DATA_LOAD__", data_load).replace("__DROP_CODE__", drop_code)


# ════════════════════════════════════════════════════════════════════════
# FAMILY 2: TABULAR REGRESSION
# New models: CatBoostRegressor(GPU), LightGBMRegressor(GPU), XGBoostRegressor(CUDA), FLAML
# ════════════════════════════════════════════════════════════════════════
TABULAR_REG_PROJECTS = {
    "Regression/Boston Housing Analysis": {"csv": "__sklearn__", "target": "target", "loader": "sklearn.datasets.load_boston"},
    "Regression/Boston Housing Prediction Analysis": {"csv": "__sklearn__", "target": "target", "loader": "sklearn.datasets.load_boston"},
    "Regression/House Price Prediction - Detailed": {"csv": "kc_house_data.csv", "target": "price"},
    "Regression/House Price prediction": {"csv": "__registry__", "target": "SalePrice", "registry_key": "house_price_prediction"},
    "Regression/Insurance premium prediction": {"csv": "__registry__", "target": "charges", "registry_key": "insurance_premium_prediction"},
    "Regression/Gold Price Prediction": {"csv": "gold_price.csv", "target": "Close"},
    "Regression/Flight Fare Prediction": {"csv": "__registry__", "target": "Price", "registry_key": "flight_fare_prediction"},
    "Regression/Car Price Prediction": {"csv": "__registry__", "target": "selling_price", "registry_key": "car_price_prediction"},
    "Regression/Data Scientist Salary Prediction": {"csv": "__registry__", "target": "Salary", "registry_key": "data_scientist_salary"},
    "Regression/Medical Cost Personal": {"csv": "__registry__", "target": "charges", "registry_key": "medical_cost_personal"},
    "Regression/Bengaluru House Price Prediction": {"csv": "__registry__", "target": "price", "registry_key": "bengaluru_house_price"},
    "Regression/BigMart Sales Prediction": {"csv": "__registry__", "target": "Item_Outlet_Sales", "registry_key": "bigmart_sales"},
    "Regression/Bike Sharing Demand Analysis": {"csv": "__registry__", "target": "cnt", "registry_key": "bike_sharing_demand"},
    "Regression/Black Friday Sales Prediction": {"csv": "__registry__", "target": "Purchase", "registry_key": "black_friday_sales"},
    "Regression/Black Friday Sales Analysis": {"csv": "__registry__", "target": "Purchase", "registry_key": "black_friday_sales_analysis"},
    "Regression/Bitcoin Price Prediction": {"csv": "__inline__", "target": "Close"},
    "Regression/Bitcoin Price Prediction - Advanced": {"csv": "__inline__", "target": "Close"},
    "Regression/California Housing Prediction": {"csv": "__sklearn__", "target": "target", "loader": "sklearn.datasets.fetch_california_housing"},
    "Regression/Car Price Prediction - Feature Based": {"csv": "__inline__", "target": "price"},
    "Regression/China GDP Estimation": {"csv": "__inline__", "target": "Value"},
    "Regression/Crop yield prediction": {"csv": "__registry__", "target": "yield", "registry_key": "crop_yield_prediction"},
    "Regression/Diabetes Prediction - Pima Indians": {"csv": "__inline__", "target": "Outcome"},
    "Regression/Employee Future Prediction": {"csv": "__registry__", "target": "LeaveOrNot", "registry_key": "employee_future_prediction"},
    "Regression/Energy Usage Prediction - Buildings": {"csv": "__registry__", "target": "energy_consumption", "registry_key": "energy_usage_buildings"},
    "Regression/Flight Delay Prediction": {"csv": "__registry__", "target": "dep_delayed_15min", "registry_key": "flight_delay_prediction"},
    "Regression/Future Sales Prediction": {"csv": "__inline__", "target": "Sales"},
    "Regression/Heart disease prediction": {"csv": "__registry__", "target": "target", "registry_key": "heart_disease_regression"},
    "Regression/Hotel Booking Cancellation Prediction": {"csv": "__registry__", "target": "is_canceled", "registry_key": "hotel_booking"},
    "Regression/House Price - Regularized Linear and XGBoost": {"csv": "__inline__", "target": "SalePrice"},
    "Regression/IPL First Innings Prediction - Advanced": {"csv": "__inline__", "target": "total"},
    "Regression/IPL First Innings Score Prediction": {"csv": "__inline__", "target": "total"},
    "Regression/Job Salary prediction": {"csv": "__registry__", "target": "salary_in_usd", "registry_key": "job_salary_prediction"},
    "Regression/Mercari Price Suggestion - LightGBM": {"csv": "__inline__", "target": "price"},
    "Regression/Rainfall Amount Prediction": {"csv": "__inline__", "target": "rainfall"},
    "Regression/Rainfall Prediction": {"csv": "__inline__", "target": "rainfall"},
    "Regression/Stock price prediction": {"csv": "__inline__", "target": "Close"},
    "Regression/TPOT Mercedes Prediction": {"csv": "__inline__", "target": "y"},
    "Regression/Tesla Car Price Prediction": {"csv": "__inline__", "target": "price"},
    "Regression/UCLA Admission Prediction": {"csv": "__inline__", "target": "Chance of Admit"},
    "Regression/50 Startups Success Prediction": {"csv": "__inline__", "target": "Profit"},
    "Regression/Bank Customer churn prediction": {"csv": "__inline__", "target": "Exited"},
}

def gen_tabular_reg_pipeline(project_path, config):
    """Generate a modern tabular regression pipeline."""
    target = config["target"]
    csv = config.get("csv", "")
    drop_cols = config.get("drop", [])
    loader = config.get("loader", "")
    registry_key = config.get("registry_key", "")

    if csv == "__sklearn__":
        module, func = loader.rsplit(".", 1)
        data_load = f'''    from {module} import {func}
    data = {func}()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["{target}"] = data.target'''
    elif csv == "__registry__":
        data_load = f'''    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from core.data_loader import load_dataset
    df = load_dataset("{registry_key}")'''
    elif csv == "__inline__":
        data_load = f'''    data_dir = Path(os.path.dirname(__file__))
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError("No CSV data found in project folder.")'''
    else:
        data_load = f'''    data_dir = Path(os.path.dirname(__file__))
    csv_path = data_dir / "{csv}"
    if not csv_path.exists():
        csv_files = list(data_dir.glob("*.csv"))
        csv_path = csv_files[0] if csv_files else csv_path
    df = pd.read_csv(csv_path)'''

    template = textwrap.dedent(f'''\
        """
        Modern Tabular Regression Pipeline (April 2026)
        Models: CatBoost (GPU), LightGBM (GPU), XGBoost (CUDA), FLAML AutoML
        """
        import os, sys, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore")

        TARGET = "{target}"


        def load_data():
        __DATA_LOAD__
            print(f"Dataset shape: {{df.shape}}")
            print(f"Target stats:\\n{{df[TARGET].describe()}}")
            return df


        def preprocess(df):
            """Auto-preprocess: encode categoricals, handle missing, split."""
            df = df.copy()
            df.dropna(subset=[TARGET], inplace=True)

            y = df[TARGET]
            X = df.drop(columns=[TARGET])

            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()

            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
            for c in cat_cols:
                X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "unknown")

            if cat_cols:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[cat_cols] = oe.fit_transform(X[cat_cols])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )
            print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")
            return X_train, X_test, y_train, y_test


        def train_and_evaluate(X_train, X_test, y_train, y_test):
            """Train CatBoostRegressor, LightGBMRegressor, XGBoostRegressor (GPU) + FLAML."""
            results = {{}}

            # ── 1. CatBoost Regressor (GPU) ──
            try:
                from catboost import CatBoostRegressor
                cb = CatBoostRegressor(
                    iterations=1000, learning_rate=0.05, depth=8,
                    task_type="GPU", devices="0",
                    eval_metric="RMSE",
                    early_stopping_rounds=50, verbose=100,
                )
                cb.fit(X_train, y_train, eval_set=(X_test, y_test))
                y_pred = cb.predict(X_test)
                results["CatBoost"] = {{"model": cb, "preds": y_pred}}
                print(f"\\n✓ CatBoost RMSE: {{mean_squared_error(y_test, y_pred, squared=False):.4f}}")
            except Exception as e:
                print(f"✗ CatBoost failed: {{e}}")

            # ── 2. LightGBM Regressor (GPU) ──
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=1000, learning_rate=0.05, max_depth=8,
                    device="gpu", gpu_platform_id=0, gpu_device_id=0,
                    verbose=-1, n_jobs=-1,
                )
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
                )
                y_pred = lgb_model.predict(X_test)
                results["LightGBM"] = {{"model": lgb_model, "preds": y_pred}}
                print(f"\\n✓ LightGBM RMSE: {{mean_squared_error(y_test, y_pred, squared=False):.4f}}")
            except Exception as e:
                print(f"✗ LightGBM failed: {{e}}")

            # ── 3. XGBoost Regressor (CUDA) ──
            try:
                from xgboost import XGBRegressor
                xgb_model = XGBRegressor(
                    n_estimators=1000, learning_rate=0.05, max_depth=8,
                    device="cuda", tree_method="hist",
                    eval_metric="rmse",
                    early_stopping_rounds=50, verbosity=1,
                    n_jobs=-1,
                )
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)], verbose=100,
                )
                y_pred = xgb_model.predict(X_test)
                results["XGBoost"] = {{"model": xgb_model, "preds": y_pred}}
                print(f"\\n✓ XGBoost RMSE: {{mean_squared_error(y_test, y_pred, squared=False):.4f}}")
            except Exception as e:
                print(f"✗ XGBoost failed: {{e}}")

            # ── 4. FLAML AutoML ──
            try:
                from flaml import AutoML
                automl = AutoML()
                automl.fit(
                    X_train, y_train,
                    task="regression",
                    time_budget=120,
                    metric="rmse",
                    verbose=1,
                    gpu_per_trial=1,
                )
                y_pred = automl.predict(X_test)
                results["FLAML"] = {{"model": automl, "preds": y_pred}}
                print(f"\\n✓ FLAML Best: {{automl.best_estimator}} - RMSE: {{mean_squared_error(y_test, y_pred, squared=False):.4f}}")
            except Exception as e:
                print(f"✗ FLAML failed: {{e}}")

            return results


        def report(results, y_test, save_dir="."):
            """Print comparison and save prediction plots."""
            print("\\n" + "=" * 60)
            print("MODEL COMPARISON")
            print("=" * 60)
            best_name, best_rmse = None, float("inf")
            for name, res in results.items():
                y_pred = res["preds"]
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"\\n── {{name}} ──")
                print(f"  RMSE: {{rmse:.4f}}  |  MAE: {{mae:.4f}}  |  R²: {{r2:.4f}}")

                if rmse < best_rmse:
                    best_rmse, best_name = rmse, name

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(y_test, y_pred, alpha=0.4, s=10)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{{name}} - Predicted vs Actual")
                fig.savefig(os.path.join(save_dir, f"scatter_{{name.lower().replace(' ', '_')}}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close(fig)

            print(f"\\n🏆 Best Model: {{best_name}} (RMSE: {{best_rmse:.4f}})")


        def main():
            print("=" * 60)
            print("MODERN TABULAR REGRESSION PIPELINE")
            print("Models: CatBoost(GPU) | LightGBM(GPU) | XGBoost(CUDA) | FLAML")
            print("=" * 60)

            df = load_data()
            X_train, X_test, y_train, y_test = preprocess(df)
            results = train_and_evaluate(X_train, X_test, y_train, y_test)

            if results:
                save_dir = os.path.dirname(os.path.abspath(__file__))
                report(results, y_test, save_dir)
            else:
                print("\\n⚠ No models trained successfully.")


        if __name__ == "__main__":
            main()
    ''')
    return template.replace("__DATA_LOAD__", data_load)


# ════════════════════════════════════════════════════════════════════════
# FAMILY 3: FRAUD / IMBALANCED CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════
FRAUD_PROJECTS = {
    "Classification/Advanced Credit Card Fraud Detection": {"csv": "creditcard.csv", "target": "Class"},
    "Classification/Credit Card Fraud - Imbalanced Dataset": {"csv": "creditcard.csv", "target": "Class"},
    "Classification/Fraud Detection": {"csv": "payment_fraud.csv", "target": "label"},
    "Anomaly detection and fraud detection/Fraud Detection in Financial Transactions": {"csv": "__inline__", "target": "isFraud"},
    "Anomaly detection and fraud detection/Insurance Fraud Detection": {"csv": "__inline__", "target": "fraud_reported"},
    "Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS": {"csv": "__inline__", "target": "isFraud"},
    "Anomaly detection and fraud detection/Fraudulent Credit Card Transaction Detection": {"csv": "__inline__", "target": "Class"},
}

def gen_fraud_pipeline(project_path, config):
    target = config["target"]
    csv = config.get("csv", "")

    if csv == "__inline__":
        data_load = f'''    data_dir = Path(os.path.dirname(__file__))
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError("No CSV data found.")'''
    else:
        data_load = f'''    data_dir = Path(os.path.dirname(__file__))
    csv_path = data_dir / "{csv}"
    if not csv_path.exists():
        csv_files = list(data_dir.glob("*.csv"))
        csv_path = csv_files[0] if csv_files else csv_path
    df = pd.read_csv(csv_path)'''

    template = textwrap.dedent(f'''\
        """
        Fraud / Imbalanced Classification Pipeline (April 2026)
        Models: CatBoost, LightGBM, XGBoost — all GPU, class-weight balanced + threshold tuning
        """
        import os, sys, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.metrics import (
            accuracy_score, classification_report, f1_score,
            precision_recall_curve, average_precision_score,
            roc_auc_score, confusion_matrix
        )
        from sklearn.calibration import CalibratedClassifierCV
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        warnings.filterwarnings("ignore")

        TARGET = "{target}"


        def load_data():
        __DATA_LOAD__
            print(f"Dataset shape: {{df.shape}}")
            print(f"Class distribution:\\n{{df[TARGET].value_counts(normalize=True)}}")
            fraud_rate = df[TARGET].mean()
            print(f"Fraud rate: {{fraud_rate:.4%}}")
            return df


        def preprocess(df):
            df = df.copy()
            df.dropna(subset=[TARGET], inplace=True)

            y = df[TARGET]
            X = df.drop(columns=[TARGET])

            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()

            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
            for c in cat_cols:
                X[c] = X[c].fillna("unknown")

            if cat_cols:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[cat_cols] = oe.fit_transform(X[cat_cols])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test


        def find_best_threshold(y_true, y_proba):
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


        def train_and_evaluate(X_train, X_test, y_train, y_test):
            results = {{}}
            scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

            # ── CatBoost (GPU) ──
            try:
                from catboost import CatBoostClassifier
                cb = CatBoostClassifier(
                    iterations=1000, learning_rate=0.03, depth=8,
                    task_type="GPU", devices="0",
                    scale_pos_weight=scale,
                    eval_metric="F1", early_stopping_rounds=50, verbose=100,
                )
                cb.fit(X_train, y_train, eval_set=(X_test, y_test))
                y_proba = cb.predict_proba(X_test)[:, 1]
                thresh = find_best_threshold(y_test, y_proba)
                y_pred = (y_proba >= thresh).astype(int)
                results["CatBoost"] = {{"preds": y_pred, "proba": y_proba, "thresh": thresh}}
                print(f"\\n✓ CatBoost F1: {{f1_score(y_test, y_pred):.4f}} (threshold={{thresh:.3f}})")
            except Exception as e:
                print(f"✗ CatBoost: {{e}}")

            # ── LightGBM (GPU) ──
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=1000, learning_rate=0.03, max_depth=8,
                    device="gpu", scale_pos_weight=scale,
                    verbose=-1, n_jobs=-1,
                )
                lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
                y_proba = lgb_model.predict_proba(X_test)[:, 1]
                thresh = find_best_threshold(y_test, y_proba)
                y_pred = (y_proba >= thresh).astype(int)
                results["LightGBM"] = {{"preds": y_pred, "proba": y_proba, "thresh": thresh}}
                print(f"\\n✓ LightGBM F1: {{f1_score(y_test, y_pred):.4f}} (threshold={{thresh:.3f}})")
            except Exception as e:
                print(f"✗ LightGBM: {{e}}")

            # ── XGBoost (CUDA) ──
            try:
                from xgboost import XGBClassifier
                xgb_model = XGBClassifier(
                    n_estimators=1000, learning_rate=0.03, max_depth=8,
                    device="cuda", tree_method="hist",
                    scale_pos_weight=scale,
                    eval_metric="aucpr", early_stopping_rounds=50, verbosity=1,
                    n_jobs=-1,
                )
                xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
                y_proba = xgb_model.predict_proba(X_test)[:, 1]
                thresh = find_best_threshold(y_test, y_proba)
                y_pred = (y_proba >= thresh).astype(int)
                results["XGBoost"] = {{"preds": y_pred, "proba": y_proba, "thresh": thresh}}
                print(f"\\n✓ XGBoost F1: {{f1_score(y_test, y_pred):.4f}} (threshold={{thresh:.3f}})")
            except Exception as e:
                print(f"✗ XGBoost: {{e}}")

            return results


        def report(results, y_test, save_dir="."):
            print("\\n" + "=" * 60)
            print("FRAUD DETECTION - MODEL COMPARISON")
            print("=" * 60)
            for name, res in results.items():
                y_pred, y_proba = res["preds"], res["proba"]
                print(f"\\n── {{name}} (threshold={{res['thresh']:.3f}}) ──")
                print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
                print(f"  AUPRC: {{average_precision_score(y_test, y_proba):.4f}}")
                print(f"  ROC-AUC: {{roc_auc_score(y_test, y_proba):.4f}}")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
                ax.set_title(f"{{name}} Confusion Matrix")
                fig.savefig(os.path.join(save_dir, f"cm_{{name.lower()}}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close(fig)


        def main():
            print("=" * 60)
            print("FRAUD / IMBALANCED CLASSIFICATION PIPELINE")
            print("CatBoost(GPU) | LightGBM(GPU) | XGBoost(CUDA) + threshold tuning")
            print("=" * 60)
            df = load_data()
            X_train, X_test, y_train, y_test = preprocess(df)
            results = train_and_evaluate(X_train, X_test, y_train, y_test)
            if results:
                report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


        if __name__ == "__main__":
            main()
    ''')
    return template.replace("__DATA_LOAD__", data_load)


# ════════════════════════════════════════════════════════════════════════
# GENERATOR: Write all pipelines
# ════════════════════════════════════════════════════════════════════════

# Import template generators from _templates/
from _templates import clustering as clustering_tmpl
from _templates import anomaly as anomaly_tmpl
from _templates import nlp_clf as nlp_clf_tmpl
from _templates import nlp_gen as nlp_gen_tmpl
from _templates import image_clf as image_clf_tmpl
from _templates import cv_detection as cv_detection_tmpl
from _templates import face_gesture as face_gesture_tmpl
from _templates import ocr as ocr_tmpl
from _templates import recommendation as recommendation_tmpl
from _templates import time_series as time_series_tmpl
from _templates import rl as rl_tmpl
from _templates import audio as audio_tmpl


# ════════════════════════════════════════════════════════════════════════
# FAMILY 4: ANOMALY DETECTION (PyOD 2)
# ════════════════════════════════════════════════════════════════════════
ANOMALY_PROJECTS = {
    "Anomaly detection and fraud detection/Anomaly Detection - Numenta Benchmark": {},
    "Anomaly detection and fraud detection/Anomaly Detection - Social Networks Twitter Bot": {},
    "Anomaly detection and fraud detection/Anomaly Detection in Images - CIFAR-10": {},
    "Anomaly detection and fraud detection/Banknote Authentication": {},
    "Anomaly detection and fraud detection/Breast Cancer Detection - Wisconsin Dataset": {},
    "Anomaly detection and fraud detection/Intrusion Detection": {},
    "Anomaly detection and fraud detection/Traffic Flow Prediction - METR-LA": {},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 5: CLUSTERING (UMAP + HDBSCAN + GMM)
# ════════════════════════════════════════════════════════════════════════
CLUSTERING_PROJECTS = {
    "Clustering/Credit Card Customer Segmentation": {},
    "Clustering/Customer Segmentation": {},
    "Clustering/Customer Segmentation - Bank": {},
    "Clustering/Financial Time Series Clustering": {},
    "Clustering/Housing Price Segmentation": {},
    "Clustering/KMeans Clustering - Imagery Analysis": {},
    "Clustering/Mall Customer Segmentation": {},
    "Clustering/Mall Customer Segmentation - Advanced": {},
    "Clustering/Mall Customer Segmentation - Detailed": {},
    "Clustering/Mall Customer Segmentation Data": {},
    "Clustering/Online Retail Customer Segmentation": {},
    "Clustering/Online Retail Segmentation Analysis": {},
    "Clustering/Spotify Song Cluster Analysis": {},
    "Clustering/Turkiye Student Evaluation - Advanced": {},
    "Clustering/Turkiye Student Evaluation Analysis": {},
    "Clustering/Vehicle Crash Data Clustering": {},
    "Clustering/Weather Data Clustering - KMeans": {},
    "Clustering/Wholesale Customer Segmentation": {},
    "Clustering/Wholesale Segmentation Analysis": {},
    "Clustering/Wine Segmentation": {},
    "Classification/Customer Segmentation - E-Commerce": {},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 6: NLP CLASSIFICATION (ModernBERT)
# ════════════════════════════════════════════════════════════════════════
NLP_CLF_PROJECTS = {
    "Classification/Cyberbullying Classification": {"target": "label", "text_col": "text"},
    "Classification/Movie Genre Classification": {"target": "genre", "text_col": "description"},
    "Classification/Spam Email Classification": {"target": "label", "text_col": "text"},
    "NLP/Amazon Alexa Review Sentiment": {"target": "feedback", "text_col": "verified_reviews"},
    "NLP/Amazon Sentiment Analysis": {"target": "sentiment", "text_col": "text"},
    "NLP/Clinton vs Trump Tweets Analysis": {"target": "handle", "text_col": "text"},
    "NLP/Consumer Complaints Analysis": {"target": "product", "text_col": "complaint"},
    "NLP/Disaster or Not Disaster": {"target": "target", "text_col": "text"},
    "NLP/DJIA Sentiment Analysis - News Headlines": {"target": "Label", "text_col": "text"},
    "NLP/DJIA Sentiment Analysis - Stock Prediction": {"target": "Label", "text_col": "text"},
    "NLP/Fake News Detection": {"target": "label", "text_col": "text"},
    "NLP/GitHub Bugs Prediction": {"target": "label", "text_col": "text"},
    "NLP/Hate Speech Detection": {"target": "label", "text_col": "tweet"},
    "NLP/IMDB Sentiment Analysis - Deep Learning": {"target": "sentiment", "text_col": "review"},
    "NLP/IMDB Sentiment Review Analysis": {"target": "sentiment", "text_col": "review"},
    "NLP/Message Spam Detection": {"target": "label", "text_col": "message"},
    "NLP/Movie Review Sentiments": {"target": "label", "text_col": "text"},
    "NLP/Restaurant Review Sentiment Analysis": {"target": "Liked", "text_col": "Review"},
    "NLP/Resume Screening": {"target": "Category", "text_col": "Resume"},
    "NLP/Sentiment Analysis": {"target": "sentiment", "text_col": "text"},
    "NLP/Sentiment Analysis - Flask Web App": {"target": "sentiment", "text_col": "text"},
    "NLP/Sentiment Analysis - Restaurant Reviews": {"target": "Liked", "text_col": "Review"},
    "NLP/SMS Spam Detection": {"target": "label", "text_col": "message"},
    "NLP/SMS Spam Detection - Detailed": {"target": "label", "text_col": "message"},
    "NLP/SMS Spam Detection Analysis": {"target": "label", "text_col": "message"},
    "NLP/Spam Classifier": {"target": "label", "text_col": "message"},
    "NLP/Spam SMS Classification": {"target": "label", "text_col": "message"},
    "NLP/Text Classification": {"target": "label", "text_col": "text"},
    "NLP/Text Classification - Keras Consumer Complaints": {"target": "product", "text_col": "complaint"},
    "NLP/Text Classification with NLP": {"target": "label", "text_col": "text"},
    "NLP/Three-Way Sentiment Analysis - Tweets": {"target": "sentiment", "text_col": "text"},
    "NLP/Twitter Sentiment Analysis": {"target": "sentiment", "text_col": "text"},
    "NLP/Twitter Sentiment Analysis - ML": {"target": "sentiment", "text_col": "text"},
    "NLP/Twitter US Airline Sentiment": {"target": "airline_sentiment", "text_col": "text"},
    "NLP/US Election Prediction": {"target": "label", "text_col": "text"},
    "Deep Learning/Amazon Alexa Sentiment Analysis": {"target": "feedback", "text_col": "verified_reviews"},
    "Deep Learning/IMDB Sentiment Analysis": {"target": "sentiment", "text_col": "review"},
    "Deep Learning/News Category Prediction": {"target": "category", "text_col": "headline"},
    "Deep Learning/Sentiment Analysis - Flask App": {"target": "sentiment", "text_col": "text"},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 7: NLP GENERATION / TRANSLATION / CHATBOT (Ollama)
# ════════════════════════════════════════════════════════════════════════
NLP_GEN_PROJECTS = {
    "NLP/Document Summary Creator": {"task": "summarization"},
    "NLP/Language Translation Model": {"task": "translation"},
    "NLP/Language Translator": {"task": "translation"},
    "NLP/Next Word Prediction": {"task": "generation"},
    "NLP/Text Generation": {"task": "generation"},
    "NLP/Text Summarization": {"task": "summarization"},
    "NLP/Text Summarization - Medium": {"task": "summarization"},
    "NLP/Text Summarization - Word Frequency": {"task": "summarization"},
    "NLP/Text Summarization - Word Frequency Method": {"task": "summarization"},
    "NLP/Spell Checker": {"task": "generation"},
    "NLP/Spelling Correction": {"task": "generation"},
    "NLP/Autocorrect": {"task": "generation"},
    "NLP/NLP for Other Languages": {"task": "translation"},
    "Deep Learning/Chatbot": {"task": "chatbot"},
    "Deep Learning/ChatBot - Neural Network": {"task": "chatbot"},
    "Deep Learning/Movie Title Prediction": {"task": "generation"},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 8: IMAGE CLASSIFICATION (DINOv2)
# ════════════════════════════════════════════════════════════════════════
IMAGE_CLF_PROJECTS = {
    "Classification/Autoencoder Fashion MNIST": {"n_classes": 10},
    "Classification/CIFAR-10 Classification": {"n_classes": 10},
    "Classification/Cotton Disease Prediction": {"n_classes": 4},
    "Classification/Digit Recognition - MNIST Sequence": {"n_classes": 10},
    "Classification/Dog vs Cat Classification": {"n_classes": 2},
    "Classification/Fashion MNIST Analysis": {"n_classes": 10},
    "Classification/Garbage Classification": {"n_classes": 6},
    "Classification/Plant Disease Recognition": {"n_classes": 38},
    "Classification/Pneumonia Classification": {"n_classes": 2},
    "Computer Vision/Indian Classical Dance Classification": {"n_classes": 8},
    "Computer Vision/Traffic Sign Recognition": {"n_classes": 43},
    "Computer Vision/Traffic Sign Recognizer": {"n_classes": 43},
    "Deep Learning/Advanced ResNet-50": {"n_classes": 10},
    "Deep Learning/Arabic Character Recognition": {"n_classes": 28},
    "Deep Learning/Bottle vs Can Classification": {"n_classes": 2},
    "Deep Learning/Brain Tumor Recognition": {"n_classes": 4},
    "Deep Learning/Cactus Aerial Image Recognition": {"n_classes": 2},
    "Deep Learning/Cat vs Dog Classification": {"n_classes": 2},
    "Deep Learning/Clothing Prediction - Flask App": {"n_classes": 10},
    "Deep Learning/Dance Form Identification": {"n_classes": 8},
    "Deep Learning/Diabetic Retinopathy": {"n_classes": 5},
    "Deep Learning/Fingerprint Recognition": {"n_classes": 10},
    "Deep Learning/Glass Detection": {"n_classes": 2},
    "Deep Learning/Happy House Predictor": {"n_classes": 2},
    "Deep Learning/Keep Babies Safe": {"n_classes": 2},
    "Deep Learning/Lego Brick Classification": {"n_classes": 16},
    "Deep Learning/Pneumonia Detection": {"n_classes": 2},
    "Deep Learning/Sheep Breed Classification - CNN": {"n_classes": 4},
    "Deep Learning/Skin Cancer Recognition": {"n_classes": 7},
    "Deep Learning/Walking or Running Classification": {"n_classes": 2},
    "Deep Learning/World Currency Coin Detection": {"n_classes": 10},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 9: CV DETECTION (YOLO)
# ════════════════════════════════════════════════════════════════════════
CV_DETECTION_PROJECTS = {
    "Computer Vision/Car and Pedestrian Tracker": {"task": "track"},
    "Computer Vision/Document Word Detection": {"task": "detect"},
    "Computer Vision/Lane Finder": {"task": "detect"},
    "Computer Vision/Captcha Recognition": {"task": "detect"},
    "Deep Learning/Landmark Detection": {"task": "detect"},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 10: FACE/GESTURE (MediaPipe + InsightFace)
# ════════════════════════════════════════════════════════════════════════
FACE_GESTURE_PROJECTS = {
    "Computer Vision/Face Detection - OpenCV": {"task": "face_detection"},
    "Computer Vision/Face Expression Identifier": {"task": "face_detection"},
    "Computer Vision/Face Mask Detection": {"task": "face_detection"},
    "Computer Vision/Gesture Control Media Player": {"task": "hand_gesture"},
    "Computer Vision/Home Security": {"task": "face_detection"},
    "Computer Vision/Live Smile Detector": {"task": "face_detection"},
    "Computer Vision/Room Security - Webcam": {"task": "face_detection"},
    "Computer Vision/Face Recognition Door Lock - AWS Rekognition": {"task": "face_recognition"},
    "Deep Learning/Caffe Face Detector - OpenCV": {"task": "face_detection"},
    "Deep Learning/Face Gender and Ethnicity Recognizer": {"task": "face_recognition"},
    "Deep Learning/Face Mask Detection": {"task": "face_detection"},
    "Deep Learning/Parkinson Pose Estimation": {"task": "pose"},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 11: OCR (PaddleOCR)
# ════════════════════════════════════════════════════════════════════════
OCR_PROJECTS = {
    "Computer Vision/Image Text Extraction - OCR": {},
    "Computer Vision/Image to Text Conversion - OCR": {},
    "Computer Vision/QR Code Readability": {},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 12: RECOMMENDATION SYSTEMS
# ════════════════════════════════════════════════════════════════════════
RECOMMENDATION_PROJECTS = {
    "Recommendation Systems/Article Recommendation System": {},
    "Recommendation Systems/Articles Recommender": {},
    "Recommendation Systems/Book Recommendation System": {},
    "Recommendation Systems/Building Recommender in an Hour": {},
    "Recommendation Systems/Collaborative Filtering - TensorFlow": {},
    "Recommendation Systems/E-Commerce Recommendation System": {},
    "Recommendation Systems/Event Recommendation System": {},
    "Recommendation Systems/Hotel Recommendation System": {},
    "Recommendation Systems/Million Songs Recommendation Engine": {},
    "Recommendation Systems/Movie Recommendation Engine": {},
    "Recommendation Systems/Movie Recommendation System": {},
    "Recommendation Systems/Movies Recommender": {},
    "Recommendation Systems/Music Recommendation System": {},
    "Recommendation Systems/Recipe Recommendation System": {},
    "Recommendation Systems/Recommender Systems Fundamentals": {},
    "Recommendation Systems/Recommender with Surprise Library": {},
    "Recommendation Systems/Restaurant Recommendation System": {},
    "Recommendation Systems/Seattle Hotels Recommender": {},
    "Recommendation Systems/TV Show Recommendation System": {},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 13: TIME SERIES (AutoGluon-TS + Chronos-Bolt)
# ════════════════════════════════════════════════════════════════════════
TIME_SERIES_PROJECTS = {
    "Time Series Analysis/Cryptocurrency Price Forecasting": {"target": "Close"},
    "Time Series Analysis/Electricity Demand Forecasting": {"target": "value"},
    "Time Series Analysis/Forecasting with ARIMA": {"target": "value"},
    "Time Series Analysis/Gold Price Forecasting": {"target": "Close"},
    "Time Series Analysis/Granger Causality Test": {"target": "value"},
    "Time Series Analysis/Mini Course Sales Forecasting": {"target": "sales"},
    "Time Series Analysis/Pollution Forecasting": {"target": "pollution"},
    "Time Series Analysis/Power Consumption - LSTM": {"target": "value"},
    "Time Series Analysis/Promotional Time Series": {"target": "sales"},
    "Time Series Analysis/Rossmann Store Sales Forecasting": {"target": "Sales"},
    "Time Series Analysis/Smart Home Temperature Forecasting": {"target": "temperature"},
    "Time Series Analysis/Solar Power Generation Forecasting": {"target": "power"},
    "Time Series Analysis/Stock Market Analysis - Tech Stocks": {"target": "Close"},
    "Time Series Analysis/Stock Price Forecasting": {"target": "Close"},
    "Time Series Analysis/Store Item Demand Forecasting": {"target": "sales"},
    "Time Series Analysis/Time Series Forecasting": {"target": "value"},
    "Time Series Analysis/Time Series Forecasting - Introduction": {"target": "value"},
    "Time Series Analysis/Time Series with LSTM": {"target": "value"},
    "Time Series Analysis/Traffic Forecast": {"target": "value"},
    "Time Series Analysis/US Gasoline and Diesel Prices 1995-2021": {"target": "value"},
    "Time Series Analysis/Weather Forecasting": {"target": "temperature"},
    "Deep Learning/Amazon Stock Price Analysis": {"target": "Close"},
    "Deep Learning/Hourly Energy Demand and Weather": {"target": "demand"},
    "Deep Learning/Stock Market Prediction": {"target": "Close"},
    "Deep Learning/Electric Car Temperature Prediction": {"target": "temperature"},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 14: REINFORCEMENT LEARNING (PPO / SAC)
# ════════════════════════════════════════════════════════════════════════
RL_PROJECTS = {
    "Reinforcement Learning/Cliff Walking": {"env": "CliffWalking-v0", "algo": "PPO"},
    "Reinforcement Learning/Frozen Lake": {"env": "FrozenLake-v1", "algo": "PPO"},
    "Reinforcement Learning/Gridworld Navigation": {"env": "CartPole-v1", "algo": "PPO"},
    "Reinforcement Learning/Lunar Landing": {"env": "LunarLander-v3", "algo": "PPO"},
    "Reinforcement Learning/Taxi Navigation": {"env": "Taxi-v3", "algo": "PPO"},
}

# ════════════════════════════════════════════════════════════════════════
# FAMILY 15: AUDIO / SPEECH (Whisper, Wav2Vec2, XTTS-v2)
# ════════════════════════════════════════════════════════════════════════
AUDIO_PROJECTS = {
    "Speech and Audio processing/Audio Denoising": {"task": "denoising"},
    "Speech and Audio processing/Music Genre Prediction - Million Songs": {"task": "classification"},
    "Speech and Audio processing/Voice Cloning": {"task": "cloning"},
    "Deep Learning/Cat and Dog Voice Recognition": {"task": "classification"},
}

# ════════════════════════════════════════════════════════════════════════
# ADDITIONAL TABULAR CLASSIFICATION projects (Deep Learning / other)
# ════════════════════════════════════════════════════════════════════════
EXTRA_TABULAR_CLF = {
    "Classification/SONAR Rock vs Mine Prediction": {"csv": "__inline__", "target": "label"},
    "Classification/Traffic Congestion Prediction": {"csv": "__inline__", "target": "target"},
    "Deep Learning/Advanced Churn Modeling": {"csv": "__inline__", "target": "Exited"},
    "Deep Learning/Bank Marketing Analysis": {"csv": "__inline__", "target": "y"},
    "Deep Learning/Campus Recruitment Analysis": {"csv": "__inline__", "target": "status"},
    "Deep Learning/COVID-19 Drug Recovery": {"csv": "__inline__", "target": "target"},
    "Deep Learning/Disease Prediction": {"csv": "__inline__", "target": "target"},
}

EXTRA_TABULAR_REG = {
    "Deep Learning/Concrete Strength Prediction": {"csv": "__inline__", "target": "strength"},
    "Deep Learning/Earthquake Prediction": {"csv": "__inline__", "target": "magnitude"},
    "Regression/Ad Demand Forecast - Avito": {"csv": "__inline__", "target": "deal_probability"},
}


def write_pipeline(project_rel_path, content):
    """Write pipeline.py to a project folder."""
    proj_dir = BASE / project_rel_path
    if not proj_dir.exists():
        print(f"  SKIP (not found): {project_rel_path}")
        return False
    out = proj_dir / "pipeline.py"
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def main():
    total = 0

    # Family 1: Tabular Classification
    print("\n═══ FAMILY 1: Tabular Classification ═══")
    all_tabular_clf = {**TABULAR_CLF_PROJECTS, **EXTRA_TABULAR_CLF}
    for path, cfg in all_tabular_clf.items():
        content = gen_tabular_clf_pipeline(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 2: Tabular Regression
    print("\n═══ FAMILY 2: Tabular Regression ═══")
    all_tabular_reg = {**TABULAR_REG_PROJECTS, **EXTRA_TABULAR_REG}
    for path, cfg in all_tabular_reg.items():
        content = gen_tabular_reg_pipeline(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 3: Fraud / Imbalanced
    print("\n═══ FAMILY 3: Fraud / Imbalanced ═══")
    for path, cfg in FRAUD_PROJECTS.items():
        content = gen_fraud_pipeline(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 4: Anomaly Detection
    print("\n═══ FAMILY 4: Anomaly Detection (PyOD 2) ═══")
    for path, cfg in ANOMALY_PROJECTS.items():
        content = anomaly_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 5: Clustering
    print("\n═══ FAMILY 5: Clustering (UMAP + HDBSCAN + GMM) ═══")
    for path, cfg in CLUSTERING_PROJECTS.items():
        content = clustering_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 6: NLP Classification
    print("\n═══ FAMILY 6: NLP Classification (ModernBERT) ═══")
    for path, cfg in NLP_CLF_PROJECTS.items():
        content = nlp_clf_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 7: NLP Generation
    print("\n═══ FAMILY 7: NLP Generation (Ollama Qwen3) ═══")
    for path, cfg in NLP_GEN_PROJECTS.items():
        content = nlp_gen_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 8: Image Classification
    print("\n═══ FAMILY 8: Image Classification (DINOv2) ═══")
    for path, cfg in IMAGE_CLF_PROJECTS.items():
        content = image_clf_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 9: CV Detection
    print("\n═══ FAMILY 9: CV Detection (YOLO) ═══")
    for path, cfg in CV_DETECTION_PROJECTS.items():
        content = cv_detection_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 10: Face/Gesture
    print("\n═══ FAMILY 10: Face/Gesture (MediaPipe + InsightFace) ═══")
    for path, cfg in FACE_GESTURE_PROJECTS.items():
        content = face_gesture_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 11: OCR
    print("\n═══ FAMILY 11: OCR (PaddleOCR) ═══")
    for path, cfg in OCR_PROJECTS.items():
        content = ocr_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 12: Recommendation
    print("\n═══ FAMILY 12: Recommendation (implicit + LightFM) ═══")
    for path, cfg in RECOMMENDATION_PROJECTS.items():
        content = recommendation_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 13: Time Series
    print("\n═══ FAMILY 13: Time Series (AutoGluon + Chronos-Bolt) ═══")
    for path, cfg in TIME_SERIES_PROJECTS.items():
        content = time_series_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 14: RL
    print("\n═══ FAMILY 14: Reinforcement Learning (PPO / SAC) ═══")
    for path, cfg in RL_PROJECTS.items():
        content = rl_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    # Family 15: Audio/Speech
    print("\n═══ FAMILY 15: Audio/Speech (Whisper + XTTS-v2) ═══")
    for path, cfg in AUDIO_PROJECTS.items():
        content = audio_tmpl.generate(path, cfg)
        if write_pipeline(path, content):
            total += 1
            print(f"  ✓ {path}")

    print(f"\n{'='*60}")
    print(f"TOTAL PIPELINES GENERATED: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
