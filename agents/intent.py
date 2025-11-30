# agents/intent.py

class IntentClassifier:
    """
    Very small rule-based intent classifier.
    Keep rules clear & non-overlapping.
    """

    def classify(self, text: str):
        text = text.lower().strip()

        # -------- FEATURE IMPORTANCE (match first) --------
        if "feature importance" in text or ("importance" in text and "feature" in text):
            return "feature_importance"

        # -------- INTERNET SEARCH (NEW) --------
        if "search" in text or "google" in text or "web" in text or "internet" in text:
            return "search_web"

        # -------- UPLOAD --------
        if "upload" in text or "load" in text or "import " in text:
            return "upload"

        # -------- CLEAN --------
        if "clean" in text or "fix data" in text or "remove noise" in text:
            return "clean"

        # -------- SET TARGET --------
        if "target" in text and ("set" in text or "choose" in text or "to " in text):
            return "set_target"

        # -------- PREVIEW --------
        if "preview" in text or "show data" in text or "first rows" in text or "head" in text:
            return "preview"

        # -------- EDA --------
        if "eda" in text or "analysis" in text or "explore data" in text:
            return "eda"

        # -------- TRAIN --------
        if "train" in text or "build model" in text or "run model" in text:
            return "train"

        # -------- SHOW MODEL --------
        if "show model" in text or "trained model" in text or "model info" in text:
            return "show_model"

        # -------- EXPORT MEMORY --------
        if "export memory" in text or "show memory" in text or "print memory" in text:
            return "export_memory"

        # -------- HELP --------
        if "help" in text or "what can you do" in text or "commands" in text:
            return "help"

        # -------- EXIT --------
        if "exit" in text or "quit" in text:
            return "exit"

        return "unknown"
