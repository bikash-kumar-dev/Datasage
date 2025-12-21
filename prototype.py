# prototype.py
import os
import sys
import json

# Agents
from agents.data_agent import DataAgent
from agents.cleaning import CleaningAgent
from agents.eda import EDAAgent
from agents.trainer import TrainerAgent
from agents.predictor import PredictorAgent
from agents.search_agent import SearchAgent   # <-- New feature of DataSage

# Utils
from agents.utils import (
    info, success, warn, error,
    save_memory, load_memory, validate_target
)

# Intent classifier
from agents.intent import IntentClassifier


class DataSage:
    def __init__(self):
        self.data_agent = DataAgent()
        self.cleaning_agent = CleaningAgent()
        self.eda_agent = EDAAgent()
        self.trainer_agent = TrainerAgent()
        self.predictor_agent = PredictorAgent()
        self.search_agent = SearchAgent()      # <-- Search Agent initialization

        self.intent_classifier = IntentClassifier()
        self.memory = load_memory()
        info("Loaded session memory.")

    # ============================================================
    def run(self):
        print("\nDataSage - AI-Enhanced Prototype (Classification)")
        print("Now understands natural language.")
        print("Say things like: 'clean this data', 'train a model', 'show memory', etc.")
        print("============================================================\n")

        while True:
            text = input("User > ").strip()
            if not text:
                continue

            if text.lower() in ["exit", "quit"]:
                success("Goodbye!")
                save_memory(self.memory)
                break

            # First, try to interpret with the intent classifier
            handled = self.handle_nlp_command(text)
            if handled:
                # already handled
                continue

            # If not handled by intent classifier AND a previous search exists,
            # try treating the input as a follow-up to the last web search.
            if hasattr(self.search_agent, "last_results") and self.search_agent.last_results:
                follow_handled = self.search_agent.follow_up(text)
                if follow_handled:
                    continue

            # fallback
            error("I didn't understand that. Try saying: 'help', 'train model', 'clean data', etc.")

    # ============================================================
    def handle_nlp_command(self, text: str):
        intent = self.intent_classifier.classify(text)

        # ---------------- INTERNET SEARCH (NEW) -------------------
        if intent == "search_web":
            # extract a query after keywords; if user typed only "search" we will warn
            q = text.lower().replace("search", "").replace("google", "").replace("web", "").replace("internet", "")
            query = q.strip()
            if not query:
                warn("Please provide a search query after 'search'. Example: search what is overfitting")
                return True
            res = self.search_agent.search_and_explain(query)
            # store last search summary to memory (optional)
            if res and "summary" in res:
                self.memory["_last_search_summary"] = res["summary"]
            return True

        # ---------------- UPLOAD -------------------
        if intent == "upload":
            if ".csv" in text:
                try:
                    path = text[text.index(".csv") - 200:text.index(".csv") + 4]
                    path = path.split()[-1].strip('"\'')
                    self.upload_dataset(path)
                except Exception:
                    warn("Please provide a path to a CSV file.")
            else:
                warn("Please provide a path to a CSV file.")
            return True

        # ---------------- CLEAN -------------------
        if intent == "clean":
            df = self.data_agent.get()
            if df is None:
                warn("Upload a dataset first.")
                return True
            if "target" not in self.memory:
                warn("Set target first using: set target <column>")
                return True

            cleaned_df = self.cleaning_agent.clean(df, self.memory["target"])

            try:
                # prefer a .set method if present
                setter = getattr(self.data_agent, "set", None)
                if callable(setter):
                    setter(cleaned_df)
                else:
                    self.data_agent.df = cleaned_df
            except Exception:
                warn("Could not update dataset in DataAgent.")
            success("Dataset cleaned.")
            return True

        # ---------------- SET TARGET -------------------
        if intent == "set_target":
            df = self.data_agent.get()
            if df is None:
                warn("Upload dataset first.")
                return True
            # try to extract column name from text
            words = [w.strip('",.') for w in text.split()]
            for w in words:
                if w in df.columns:
                    self.memory["target"] = w
                    save_memory(self.memory)
                    success(f"Target set to: {w}")
                    return True
            warn("No valid column found in your message. Use exact column name.")
            return True

        # ---------------- PREVIEW -------------------
        if intent == "preview":
            df = self.data_agent.get()
            if df is None:
                warn("Upload a dataset first.")
                return True
            print(df.head())
            return True

        # ---------------- EDA -------------------
        if intent == "eda":
            self.run_eda()
            return True

        # ---------------- TRAIN -------------------
        if intent == "train":
            self.train_model()
            return True

        # ---------------- SHOW MODEL -------------------
        if intent == "show_model":
            print(self.memory.get("model_metadata", "No model trained yet."))
            return True

        # ---------------- FEATURE IMPORTANCE -------------------
        if intent == "feature_importance":
            try:
                fi = self.trainer_agent.get_feature_importance()
            except Exception as e:
                warn(str(e))
                return True

            if fi is None:
                warn("Model does not support feature importance.")
                return True

            print("\n=== Feature Importance ===")
            for name, val in fi.items():
                print(f"{name:<30}: {val:.4f}")
            return True

        # ---------------- EXPORT MEMORY -------------------
        if intent == "export_memory":
            self.export_memory()
            return True

        # ---------------- HELP -------------------
        if intent == "help":
            print("\nAvailable commands:")
            print("- upload <path>")
            print("- clean data")
            print("- set target <column>")
            print("- preview")
            print("- eda")
            print("- train model")
            print("- show model")
            print("- feature importance")
            print("- export memory")
            print("- search <query>")
            print("- exit\n")
            return True

        # ---------------- EXIT -------------------
        if intent == "exit":
            success("Goodbye!")
            save_memory(self.memory)
            sys.exit()

        # Not handled by intent classifier
        return False

    # ============================================================
    # Implementations
    # ============================================================
    def upload_dataset(self, path):
        try:
            df = self.data_agent.load(path)
            self.memory["dataset_path"] = path
            save_memory(self.memory)
            success(f"Dataset loaded: {path}")
        except Exception as e:
            error(f"Error loading dataset: {e}")

    def run_eda(self):
        df = self.data_agent.get()
        if df is None:
            warn("Upload dataset first.")
            return
        success("Running EDA…")
        self.eda_agent.run(df)
        success("EDA files saved to artifacts/eda/")

    def train_model(self):
        df = self.data_agent.get()
        if df is None:
            warn("Upload dataset first.")
            return
        if "target" not in self.memory:
            warn("Set target first.")
            return

        target = self.memory["target"]
        try:
            validate_target(df, target)
        except Exception as e:
            error(e)
            return

        model, acc, model_name = self.trainer_agent.train(df, target)
        self.memory["model_metadata"] = {
            "target": target,
            "accuracy": acc,
            "model_file": "artifacts/models/model.pkl"
        }
        save_memory(self.memory)
        success(f"Best Model: {model_name} (Acc: {acc:.4f})")

    def export_memory(self):
        print("\n=== SESSION MEMORY ===")
        print(json.dumps(self.memory, indent=2))


if __name__ == "__main__":
    ds = DataSage()
    ds.run()
