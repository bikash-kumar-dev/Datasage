# agents/cleaning_agent.py

import pandas as pd
from agents.utils import info, success, warn

class CleaningAgent:
    
    def clean(self, df, target_column=None):
        info("Starting interactive data cleaning...")

        # ====================================================
        # 1️⃣ Missing Values
        # ====================================================
        missing = df.isnull().sum()

        if missing.sum() > 0:
            warn("Missing values found:")
            print(missing[missing > 0])

            choice = input("\nFix missing values? (yes/no): ").strip().lower()
            if choice == "yes":
                method = input("Choose method (mean/median/mode/drop): ").strip().lower()

                for col in df.columns:
                    if col == target_column:  # Skip target
                        continue

                    if df[col].isnull().sum() > 0:
                        if method == "mean" and df[col].dtype != "object":
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif method == "median" and df[col].dtype != "object":
                            df[col].fillna(df[col].median(), inplace=True)
                        elif method == "mode":
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        elif method == "drop":
                            df.dropna(inplace=True)
                            break

                success("Missing values fixed.")
            else:
                warn("Skipped missing value treatment.")
        else:
            success("No missing values found.")

        # ====================================================
        # 2️⃣ Remove Duplicates
        # ====================================================
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            warn(f"{duplicates} duplicate rows found.")
            choice = input("Remove duplicates? (yes/no): ").strip().lower()
            if choice == "yes":
                df.drop_duplicates(inplace=True)
                success("Duplicates removed.")
            else:
                warn("Skipped removing duplicates.")
        else:
            success("No duplicates found.")

        # ====================================================
        # 3️⃣ Constant Columns
        # ====================================================
        constant_cols = [c for c in df.columns if df[c].nunique() == 1]

        if len(constant_cols) > 0:
            warn(f"Constant columns found: {constant_cols}")

            # Make sure target is never removed
            if target_column in constant_cols:
                constant_cols.remove(target_column)

            choice = input("Remove constant columns? (yes/no): ").strip().lower()
            if choice == "yes":
                df = df.drop(columns=constant_cols)
                success("Constant columns removed.")
            else:
                warn("Skipped constant column removal.")
        else:
            success("No constant columns found.")

        # ====================================================
        # 4️⃣ Outliers (optional)
        # ====================================================
        choice = input("\nDetect & cap outliers (IQR)? (yes/no): ").strip().lower()
        if choice == "yes":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

            for col in numeric_cols:
                if col == target_column:
                    continue

                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                df[col] = df[col].clip(lower, upper)

            success("Outliers capped using IQR.")
        else:
            warn("Skipped outlier treatment.")

        success("🧹 Cleaning complete!")
        return df
