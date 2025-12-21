# 📊 DataSage  
### **The Smartest AI-Powered Helper for Data Scientists**  
An intelligent, natural-language-driven, multi-capability ML agent built as part of the **Kaggle Agents Intensive Capstone Project**.

DataSage transforms traditional data science tasks by enabling users to upload data, clean datasets, perform EDA, train ML models, inspect feature importance, and even run real-time internet searches — all using simple English commands.

---

# 🎯 Capstone Objective  
The Kaggle Agents Intensive Capstone requires building an AI agent that can:

- Understand natural language  
- Call tools autonomously  
- Maintain memory  
- Perform multi-step workflows  
- Integrate external APIs  
- Demonstrate real-world utility  

**DataSage** fulfills all these requirements through an end-to-end data science automation pipeline.

---

# 🌟 Features & Capabilities

### 🧠 **1. Natural Language Interface**
You can talk to DataSage like talking to an assistant:

- upload my dataset
- clean the dataset
- perform eda
- train a best model
- show feature importance
- search what is reinforcement learning

No coding needed — the agent interprets the intent and triggers the correct tools.

---

### 📂 **2. Smart Dataset Handling**
- Load datasets by specifying file path in natural language  
- Automatically remembers last opened dataset  
- Validates file and format  

---

### 🧼 **3. Automated Data Cleaning**
Command: `clean data`

Includes:
- Missing values check  
- Duplicate removal  
- Constant column detection  
- Optional IQR-based outlier capping  
- Interactive cleaning steps  

---

### 🔍 **4. Exploratory Data Analysis (EDA)**
Command: `eda`

Automatically generates:
- Data preview CSV  
- Statistical summary CSV  
- Correlation matrix heatmap  
- Missing value heatmap  
- Target distribution plot  

All saved to: eda

---

### 🤖 **5. Model Training & Comparison**
The `trainer_agent` trains 4 ML models:
- Logistic Regression  
- SVM (RBF)  
- Random Forest  
- XGBoost  

Automatically selects the **best-performing model** and saves:artifacts/models/model.pkl

---

### 📈 **6. Feature Importance**
Based on the selected model, DataSage shows:
- Ranked feature importances  
- Human-readable display  

Example output:
Glucose : 1.08
BMI : 0.77
Age : 0.43

---

### 🌍 **7. Real-Time Internet Search**
The `search_agent` performs:
- Live search  
- Multi-source summary  
- Stores last result in session memory  

---

### 🧠 **Session Memory**
Saved automatically inside:artifacts/session_memory.json

Includes:
- Last dataset  
- Target column  
- Best model & accuracy  
- Last search query  

Loaded automatically on startup.

---

# 🧱 Project Structure  

```text
DATASAGE-PROTOTYPE/
│
├── agents/
│   ├── __init__.py
│   ├── intent.py
│   ├── data_agent.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── trainer.py
│   ├── predictor.py
│   ├── search_agent.py
│   └── utils.py
│
├── artifacts/
│   ├── eda/
│   ├── models/
│   ├── class_balance_Outcome.png
│   ├── class_balance_target.png
│   ├── correlation_matrix.png
│   ├── preview.csv
│   └── session_memory.json
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
│
├── models/
│   └── random_forest.pkl
│
├── venv/   # virtual environment
│
├── prototype.py   # main entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Getting Started

### 1️⃣ Clone the repository
```
git clone https://github.com/bikash-kumar-dev/Datasage
cd DATASAGE-PROTOTYPE
```

### 2️⃣ Create & activate virtual environment  
Windows:
```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Run DataSage 
```
python prototype.py
```

---

# 🎮 Sample Usage (Real Output)

User > upload my dataset data/diabetes.csv
[SUCCESS] Dataset loaded

User > clean the dataset
[SUCCESS] Cleaning complete!

User > perform eda
[SUCCESS] EDA files saved to artifacts/eda/

User > train a best model
[BEST MODEL] Logistic Regression (0.7532)

User > show memory
{
"dataset_path": "data/diabetes.csv",
"target": "Outcome",
"model": "Logistic Regression",
"accuracy": 0.7532
}

User > search what is reinforcement learning
[Internet search results]

---

# 🏆 Why DataSage is a Strong Capstone Project

### ✔ Multi-Agent Architecture  
Agents:  
- Intent agent  
- Data agent  
- Cleaning agent  
- EDA agent  
- Training agent  
- Predictor agent  
- Search agent  

### ✔ Real-World ML Pipeline  
From raw CSV → model training → exportable artifacts.

### ✔ Natural Language + Tool Calling  
Handles multiple workflows through language instructions.

### ✔ Persistent Memory  
Saves & restores previous state.

### ✔ Modular, Scalable, Extensible  
Agents can be extended, replaced, or improved easily.

---

# 📌 Future Enhancements

- Add Streamlit UI
- adding regression type of problem  
- Add AutoML and Hyperparameter tuning  
- Integrate PDF-to-Table extraction  
- Add multi-modal (image + text) support  
- Add SHAP based explainability  

---

# 👨‍💻 Author  
**Bikash Kumar Naik**  
AI/ML Developer • Kaggle Agents Intensive Participant  
Project: **DataSage — AI Agent for Data Scientists**
