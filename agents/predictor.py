# agents/predictor.py
import os
import joblib
import pandas as pd

class PredictorAgent:
    def __init__(self, model_dir="artifacts/models"):
        self.model_dir = model_dir

    def load_model(self, model_name="random_forest.pkl"):
        """Load the trained model."""
        model_path = os.path.join(self.model_dir, model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        model = joblib.load(model_path)
        return model

    def predict(self, df, model_name="random_forest.pkl"):
        """Run predictions on new data."""
        model = self.load_model(model_name)

        # Predict classes
        preds = model.predict(df)

        # Predict probabilities
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)

        # Save predictions
        output_path = os.path.join("artifacts", "models", "predictions.csv")
        output_df = df.copy()
        output_df["prediction"] = preds

        if proba is not None:
            # Add probability columns
            for idx, cls in enumerate(model.classes_):
                output_df[f"prob_{cls}"] = proba[:, idx]

        output_df.to_csv(output_path, index=False)

        return {
            "predictions_path": output_path,
            "predictions": preds.tolist(),
            "probabilities": proba.tolist() if proba is not None else None
        }
