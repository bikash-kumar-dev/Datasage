#eda.py
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EDAAgent:
    def __init__(self, artifact_dir="artifacts/eda"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def run(self, df):
        print("[INFO] Performing EDA...")

        # 1. Save data preview
        preview_path = os.path.join(self.artifact_dir, "preview.csv")
        df.head().to_csv(preview_path, index=False)
        print(f"[SUCCESS] Saved preview: {preview_path}")

        # 2. Save summary statistics (numeric + categorical)
        summary_path = os.path.join(self.artifact_dir, "summary.csv")
        df.describe(include="all").to_csv(summary_path)
        print(f"[SUCCESS] Saved summary: {summary_path}")

        # 3. Handle numeric-only correlation
        numeric_df = df.select_dtypes(include=["int64", "float64"])

        if numeric_df.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
            corr_path = os.path.join(self.artifact_dir, "correlation_matrix.png")
            plt.savefig(corr_path)
            plt.close()
            print(f"[SUCCESS] Saved correlation matrix: {corr_path}")
        else:
            print("[WARNING] Not enough numeric columns for correlation matrix.")

        # 4. Missing values heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False)
        missing_path = os.path.join(self.artifact_dir, "missing_values.png")
        plt.savefig(missing_path)
        plt.close()
        print(f"[SUCCESS] Saved missing values heatmap: {missing_path}")

        # 5. Class balance (auto-detect target column)
        possible_targets = ["target", "status", "label", "class", "Outcome"]
        target_col = None

        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

        if target_col:
            plt.figure(figsize=(6, 4))
            df[target_col].value_counts().plot(kind="bar")
            class_path = os.path.join(self.artifact_dir, f"class_balance_{target_col}.png")
            plt.savefig(class_path)
            plt.close()
            print(f"[SUCCESS] Saved class balance plot: {class_path}")
        else:
            print("[INFO] No target column detected. Skipping class balance plot.")

        print("[SUCCESS] EDA Completed.")
