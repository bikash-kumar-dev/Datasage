# agents/trainer.py
import os
import joblib
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Try loading XGBoost if installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


class TrainerAgent:
    def __init__(self):
        # Will hold the final pipeline once training completes
        self.pipeline = None
        self.best_model_name = None
        self.best_accuracy = None

        # Columns detected during training (used for feature importance)
        self.numeric_cols = []
        self.categorical_cols = []

    # ----------------------------------------------------------
    # TRAINING + MODEL COMPARISON
    # ----------------------------------------------------------
    def train(self, df, target):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        print(Fore.BLUE + "\n[INFO] Training models with comparison..." + Style.RESET_ALL)

        X = df.drop(columns=[target])
        y = df[target]

        # Detect numeric / categorical columns
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        print(Fore.CYAN + f"[INFO] Numeric columns: {self.numeric_cols}" + Style.RESET_ALL)
        print(Fore.CYAN + f"[INFO] Categorical columns: {self.categorical_cols}" + Style.RESET_ALL)

        # Drop ID-like categorical columns (unique per row)
        for col in list(self.categorical_cols):
            if X[col].nunique() == len(X):
                print(Fore.YELLOW + f"[WARNING] Dropping ID-like column: {col}" + Style.RESET_ALL)
                X = X.drop(columns=[col])
                self.categorical_cols.remove(col)

        # Preprocessing: scale numeric, one-hot encode categorical
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
            ],
            remainder="drop",
        )

        # Define candidate models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "SVM (RBF)": SVC(probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=200),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(eval_metric="logloss", use_label_encoder=False)

        print(Fore.BLUE + f"[INFO] Models available: {list(models.keys())}" + Style.RESET_ALL)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}

        # Train each model and record accuracy on test set
        for name, model in models.items():
            print(Fore.YELLOW + f"[INFO] Training {name}..." + Style.RESET_ALL)

            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            accuracy = pipeline.score(X_test, y_test)

            print(Fore.GREEN + f"[SUCCESS] {name} Accuracy: {accuracy:.4f}" + Style.RESET_ALL)
            results[name] = (accuracy, pipeline)

        # Pick best by accuracy
        self.best_model_name = max(results, key=lambda x: results[x][0])
        self.best_accuracy, self.pipeline = results[self.best_model_name]

        print(Fore.MAGENTA + f"\n[BEST MODEL] {self.best_model_name} ({self.best_accuracy:.4f})" + Style.RESET_ALL)

        # Save pipeline to artifacts
        os.makedirs("artifacts/models", exist_ok=True)
        joblib.dump(self.pipeline, "artifacts/models/model.pkl")
        print(Fore.GREEN + "[SUCCESS] Saved best model to artifacts/models/model.pkl" + Style.RESET_ALL)

        return self.pipeline, self.best_accuracy, self.best_model_name

    # ----------------------------------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------------------------------
    def get_feature_importance(self):
        """
        Return a dict {feature_name: importance_score} sorted descending,
        or return None if the model doesn't support feature importance.
        """

        if self.pipeline is None:
            raise ValueError("No trained model found. Train a model first.")

        model = self.pipeline.named_steps["model"]
        preprocessor = self.pipeline.named_steps["preprocessor"]

        # numeric feature names are known
        numeric_features = list(self.numeric_cols)  # copy

        importances = {}

        # Case A: tree-based models (RandomForest, XGBoost)
        if hasattr(model, "feature_importances_"):
            # feature_importances_ align to preprocessor output columns,
            # but we only reliably map numeric cols (others may be encoded)
            # If there are categorical cols, OneHotEncoder expands them; here we map numeric only.
            fi_vals = model.feature_importances_
            # If preprocessor has both numeric and cat, feature_importances_ length will be larger.
            # Best effort: map first len(numeric_features) entries to numeric_features (since we standardized numeric first).
            n_num = len(numeric_features)
            if n_num == 0:
                # no numeric features to show
                return None
            for col, score in zip(numeric_features, fi_vals[:n_num]):
                importances[col] = float(score)

        # Case B: linear models (LogisticRegression, linear SVM) with coefficients
        elif hasattr(model, "coef_"):
            coef_values = model.coef_
            # multiclass may have multiple rows; take absolute mean across classes
            if coef_values.ndim == 1 or coef_values.shape[0] == 1:
                vals = abs(coef_values.ravel())
            else:
                vals = abs(coef_values).mean(axis=0)

            if len(vals) < len(numeric_features):
                # fallback: only map what we can
                count = min(len(vals), len(numeric_features))
                for col, score in zip(numeric_features[:count], vals[:count]):
                    importances[col] = float(score)
            else:
                for col, score in zip(numeric_features, vals[: len(numeric_features)]):
                    importances[col] = float(score)

        else:
            # Model doesn't support importance/coefs (e.g., certain SVC kernels)
            return None

        # Sort descending and return
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        return importances
