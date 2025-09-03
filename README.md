## Fraud Detection & Risk Analytics Platform for ZenithPay Solutions

[Link to Live Project](https://frauddetection-riskanalysis-czzjjkrx6dkp9tzsbj55jc.streamlit.app/)

[Link to DagsHub Repository](https://dagshub.com/AnupamKNN/Fraud_Detection_-_Risk_Analysis)

[Link to Presentation Video](https://youtu.be/60J0vSLglvk)

[Link to LinkedIn post](https://www.linkedin.com/posts/anupam-singh-1l_genai-ai-machinelearning-activity-7363413661572284416-C13B?utm_source=share&utm_medium=member_desktop&rcm=ACoAACTx8xsBG5OdxDsxposmyvR-JD_HZhoD33I)

[Link to sample dataset to test the application](https://drive.google.com/file/d/19hhN7x7NdK38JEpQSWm6o3kJp9s-FYuo/view)

[Link to sample data creation script](https://github.com/AnupamKNN/Fraud_Detection_-_Risk_Analysis/blob/main/sample_data_creation.py)

### 🏢 About the Company


ZenithPay Solutions Pvt. Ltd. is a rapidly growing fintech company founded in 2018, specializing in digital payments, SME banking, and cross-border remittance services across India, Southeast Asia, and the Middle East.
With over 1.5 million active merchants and 10 million monthly transactions, ZenithPay’s mission is to make financial services fast, secure, and accessible for small to medium-sized enterprises.

However, with transaction volumes skyrocketing, fraudulent activities such as credit card fraud, account takeovers, synthetic identity fraud, and money laundering have become increasing threats.

ZenithPay’s leadership decided to invest in an AI-powered, real-time Fraud Detection & Risk Analytics system to protect revenue, maintain compliance, and build customer trust.

---

### 👥 Project Stakeholders and Team

| Role | Name | Responsibility |
|------|------|----------------|
| **Chief Risk Officer (Stakeholder)** | Meera Krishnan | Defined compliance requirements, risk scoring policies |
| **Head of Fraud Analytics (Stakeholder)** | Rohit Malhotra | Provided fraud pattern datasets & domain knowledge |
| **Project Manager** | Arjun Deshpande | Managed timelines, coordinated between ML & IT |
| **Lead Data Scientist** | Anupam Singh | Designed ML pipeline, feature engineering, model selection, LLM Integration |
| **MLOps Engineer** | Sonal Mehta | Built CI/CD, Dockerization, and DVC/MLflow integration |
| **Frontend Developer** | Kartik Rane | Created Streamlit UI dashboards & analytics |
| **Data Engineer** | Nisha Verma | Managed data ingestion, transformation, storage |

---

### 📉 Business Problem
ZenithPay’s old fraud checks were rule-based and static — failing to catch sophisticated patterns and causing many false alerts.
The challenge was to detect, explain, and act on fraudulent transactions in near real-time while minimizing disruption for genuine customers.

---

### 🎯 Project Objectives
✅ Real-time transaction scoring using ML models trained on historical fraud patterns
✅ Interactive, cloud-accessible fraud analytics dashboard with individual transaction explanations using LLM
✅ CI/CD pipeline with data and model versioning via DVC and MLflow tracking
✅ Streamlit Cloud deployment for easy stakeholder review


### 📊 Expected Impact
Metric	Projected Outcome
🔻 Fraud Losses	35–45% Reduction
🔺 Analyst Efficiency	50% faster investigations
✔️ False Positive Rate	Reduce by 20–25%
⚡ Operational Efficiency	Real-time scoring & one-click export

---

### ⚙️ Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Language** | Python 3.10 |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **MLOps Performance Tracking** | MLflow, DVC |
| **Deployment** | Streamlit Cloud |
| **Database** | MongoDB |
| **CI/CD Pipelines** | GitHub Actions |
| **Containerization** | Docker, GitHub Container Registry (GHCR) |
| **LLM Implementation** | LangChain + ChatGroq (model: `llama-3.3-70b-versatile`) |

---

### 🖥 Streamlit Dashboard Features
📌 Real-time fraud risk scoring using KPI cards

📊 Fraud trend visualizations

🔍 Filter/search high-risk transactions

📢 Model explanations per transaction

📤 Export reports for audits & compliance

---

### 🛠 Setup & Run

You have two primary ways to run this application: the recommended **Docker method** for quick and easy setup or a **manual local installation** for development and contribution.

#### Option 1: Run with Docker (Recommended)

**Prerequisites:**

* Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system.

> **For contributors/maintainers:** If the repository is private, authenticate with GHCR using:
> ```bash
> docker login ghcr.io
> ```
> Use a GitHub Personal Access Token for authentication.

**Steps:**

1.  **Pull the Docker Image:**
 The Docker image is pre-built and pushed to GitHub Container Registry (GHCR). Pull it directly:
   ```bash
   docker pull ghcr.io/AnupamKNN/fraud_detection_-_risk_analysis:latest
   ```

**Or build locally from source:**

   ```bash
   git clone https://github.com/AnupamKNN/fraud_detection_-_risk_analysis:latest
   cd fraud_detection_-_risk_analysis
   
   docker build -t fraud_detection_-_risk_analysis:latest
   ```

2.  **Run the Docker Container:**
Once the image is pulled, you can run the application. The container will listen on map port `8501` on your local machine (adjust if your app uses a different port).
   ```bash
   docker run -p 8501:8501 ghcr.io/AnupamKNN/fraud_detection_-_risk_analysis:latest
   ```

3.  **Access the Application:**
Navigate to [http://localhost:8501](http://localhost:8501) in your web browser to access the Streamlit dashboard.


### 🛠 Option 2: Manual Local Setup (For Development)

If a manual setup is preferred, follow these steps:

**Prerequisites:**

* **Python 3.10** or higher installed.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AnupamKNN/Fraud_Detection_-_Risk_Analysis.git
    ```

2. *** Locate to the project folder:***

    ```bash
    cd Fraud_Detection_-_Risk_Analysis
    ```

3.  **Create a Virtual Environment and Install Dependencies:**

 **Using `venv` (recommended for Python projects):**

#### 1️⃣ Create a virtual environment
    ```bash
    python3.10 -m venv venv
    ```

#### 2️⃣ Activate the virtual environment

##### For Linux/macOS:
    ```bash
    source venv/bin/activate
    ```

##### For Windows (PowerShell):
    ```bash
    .\venv\Scripts\Activate.ps1
    ```

##### For Windows (Command Prompt):
    ```bash
    .\venv\Scripts\activate
    ```

#### 3️⃣ Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4) **Configure environment**
cp .env.example .env
### 🔑 Environment Variables  

After copying `.env.example` to `.env`, edit the file and set the required keys:  

- `GROQ_API_KEY=<your_key>`  
- `LANGCHAIN_API_KEY=<your_key>`  
- `MLFLOW_TRACKING_USERNAME=<your_username>`  
- `MLFLOW_TRACKING_PASSWORD=<your_password>`  
- `MLFLOW_TRACKING_URI=<your_tracking_uri>`  

> **Note:** MLflow credentials are only required for developers or contributors who need experiment tracking.

5) **Ensure artifacts & data exist**
- final_models/model.keras
- final_models/preprocessor.pkl
- templates/config.py has valid HIST_PATH pointing to historical CSV

6) **Run the app**
streamlit run app.py


### Explanation:
1. Creates a Conda virtual environment** named `venv` with Python 3.10.
2. Activates the environment.
3. Installs dependencies from the `requirements.txt` file.  

This makes it easy for anyone cloning your repo to set up their environment correctly! ✅

---

### 🎯 Model Training & Evaluation

The fraud detection models were trained using supervised learning algorithms from Scikit-learn and XGBoost. The training process included:

- Feature scaling and transformation

- Handling class imbalance with SMOTETomek

- Hyperparameter tuning with RandomizedSearchCV

- Model selection based on validation performance tracked in MLflow


#### 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of total predictions that are correct |
| **Precision** | Proportion of transactions predicted as fraud that are actually fraudulent *(TP / (TP + FP))* |
| **F1 Score** | Harmonic mean of precision and recall — balances false positives and false negatives |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve; measures separability between fraud and non-fraud classes |


## 📊 Model Performance Summary

### 🔍 Before Hyperparameter Tuning

#### ✅ Training Results

| Rank | Model                       | F1 Score | ROC AUC | Accuracy |
|------|-----------------------------|----------|---------|----------|
| 1️⃣  | RandomForestClassifier       | 100.00%  | 100.00% | 100.00%  |
| 1️⃣  | XGBClassifier                | 100.00%  | 100.00% | 100.00%  |
| 1️⃣  | DecisionTreeClassifier       | 100.00%  | 100.00% | 100.00%  |
| 4️⃣  | CatBoostClassifier           | 99.99%   | 100.00% | 99.99%   |
| 5️⃣  | KNeighborsClassifier         | 99.94%   | 100.00% | 99.94%   |
| 6️⃣  | GradientBoostingClassifier   | 98.93%   | 99.95%  | 98.93%   |
| 7️⃣  | Logistic Regression          | 95.54%   | 99.31%  | 95.62%   |
| 8️⃣  | AdaBoost                     | 95.24%   | 99.33%  | 95.34%   |

#### 🧪 Test Results

| Rank | Model                       | F1 Score | ROC AUC | Accuracy |
|------|-----------------------------|----------|---------|----------|
| 1️⃣  | XGBClassifier                | 88.21%   | 97.62%  | 99.96%   |
| 2️⃣  | RandomForestClassifier       | 83.60%   | 97.32%  | 99.95%   |
| 3️⃣  | CatBoostClassifier           | 82.59%   | 97.87%  | 99.94%   |
| 4️⃣  | KNeighborsClassifier         | 60.56%   | 94.85%  | 99.80%   |
| 5️⃣  | DecisionTreeClassifier       | 47.32%   | 88.14%  | 99.71%   |
| 6️⃣  | GradientBoostingClassifier   | 28.66%   | 98.27%  | 99.23%   |
| 7️⃣  | AdaBoost                     | 10.24%   | 96.75%  | 97.29%   |
| 8️⃣  | Logistic Regression          | 10.10%   | 97.40%  | 97.19%   |

---

### 🚀 After Hyperparameter Tuning

#### ✅ Training Results

| Rank | Model                          | F1 Score | ROC AUC | Accuracy |
|------|--------------------------------|----------|---------|----------|
| 1️⃣  | K-Neighbors Classifier         | 100.00%  | 100.00% | 100.00%  |
| 1️⃣  | Random Forest Classifier       | 100.00%  | 100.00% | 100.00%  |
| 3️⃣  | XGB-Classifier                 | 99.99%   | 100.00% | 99.99%   |
| 4️⃣  | Gradient Boosting Classifier   | 99.98%   | 100.00% | 99.98%   |
| 5️⃣  | CatBoostClassifier             | 99.98%   | 100.00% | 99.98%   |
| 6️⃣  | Decision Tree Classifier       | 99.68%   | 99.98%  | 99.68%   |
| 7️⃣  | Logistic Regression            | 95.55%   | 99.31%  | 95.63%   |
| 8️⃣  | AdaBoost Classifier            | 92.71%   | 98.55%  | 93.09%   |

#### 🧪 Test Results

| Rank | Model                          | F1 Score | ROC AUC | Accuracy |
|------|--------------------------------|----------|---------|----------|
| 1️⃣  | Random Forest Classifier       | 84.32%   | 96.34%  | 99.95%   |
| 2️⃣  | CatBoostClassifier             | 80.98%   | 97.57%  | 99.93%   |
| 2️⃣  | XGB-Classifier                 | 80.98%   | 98.27%  | 99.93%   |
| 4️⃣  | Gradient Boosting Classifier   | 78.30%   | 96.86%  | 99.92%   |
| 5️⃣  | K-Neighbors Classifier         | 70.00%   | 94.35%  | 99.87%   |
| 6️⃣  | Decision Tree Classifier       | 27.81%   | 90.69%  | 99.29%   |
| 7️⃣  | AdaBoost Classifier            | 15.44%   | 96.76%  | 98.31%   |
| 8️⃣  | Logistic Regression            | 10.07%   | 97.40%  | 97.18%   |


As per the above results, we can see that Random Forest Classifier is the best models with accuracy of 99.95% and ROC AUC score of 96.34%. So, let us choose **Random Forest Classifier** as our final model.

---

### 🏭 Production-Ready Pipeline

This project includes a fully automated, production-grade ML pipeline consisting of:

- `data_ingestion.py` – Fetches and stores raw data.
- `data_validation.py` – Ensures data quality, schema compliance, and integrity.
- `data_transformation.py` – Preprocesses and transforms data for model consumption and pushes preprocessor for production use.
- `model_trainer.py` – Trains and evaluates models and pushes the best model based on evaluation metrics.

**Note:**  
The pipeline is designed to automatically deliver the best-performing model from the latest run based on evaluation metrics, ensuring that production always uses the most optimal version.


### 📈 Pipeline Flow

 ```mermaid
flowchart LR
    A[📥 Data Ingestion: Fetch & store raw data] --> 
    B[✅ Data Validation: Check schema & data quality] --> 
    C[🔄 Data Transformation: Clean & preprocess data] --> 
    D[🤖 Model Training & Evaluation: Train & compare models] --> 
    E[🚀 Best Model to Production: Deploy latest optimal model]
```


---

### 🚀 Usage

1. **Upload Transaction Data** – Analyst uploads a CSV file containing recent transactions with attributes like amount, time, merchant, device info, etc.  
2. **Set Fraud Probability Threshold** – Adjust the slider to tune sensitivity for fraud classification.  
3. **ML Model Scoring** – The trained model preprocesses the data, engineers features, and predicts a fraud probability for each transaction.  
4. **View AI-Assisted Insights** – The dashboard highlights predicted frauds, shows KPIs, and (optionally) explains individual fraud cases in plain language using **LangChain + Groq LLM**.  
5. **Download Reports** – Export full prediction results or fraud-only records for compliance, audits, or further review.  
6. **Natural Language Querying** – Ask direct questions about the dataset (e.g., *"Show all frauds over $1000"*) and receive instant LLM-generated answers.

---

### 🔥 Results & Insights

📌 The fraud detection model accurately identifies suspicious transactions, enabling the fintech’s risk teams to:  

- ✅ **Prevent financial losses** by detecting high-probability fraud before settlement.  
- ✅ **Prioritize investigations** for transactions with the greatest detected risk.  
- ✅ **Improve analyst productivity** with instant KPIs and batch summaries.  
- ✅ **Enhance decision-making** through AI-generated explanations in non-technical language.  
- ✅ **Adapt quickly to evolving fraud patterns** with a retrainable, modular ML pipeline.  


---

### ✅ Final Deliverables

This project delivers a complete, production-ready fraud detection and analytics platform equipped with AI-assisted insights:

| Deliverable | Description |
|-------------|-------------|
| 📦 **Fraud Detection Training Pipeline** | End-to-end ML pipeline with data ingestion, feature engineering, model training, evaluation, and artifact storage (`model.pkl`, `preprocessor.pkl`). |
| 🧮 **Feature Engineering Module** | Adds temporal features (e.g., transaction hour sine/cosine encoding, elapsed days) and human-readable timestamps for richer fraud modeling. |
| 🔄 **Version-Controlled Workflow** | DVC for dataset & model versioning and MLflow for experiment tracking and reproducibility. |
| 🤖 **AI-Powered Explanations & Summaries** | Integrated LangChain + ChatGroq LLM to explain per-transaction fraud predictions in plain language and summarize batch fraud statistics for analysts. |
| 📊 **Interactive Fraud Analytics Dashboard** | Streamlit app to upload CSVs, set probability thresholds, visualize KPIs, explore fraud probability distributions, filter fraudulent records, and download reports. |
| 💬 **Natural Language Querying** | Users can directly ask questions about the uploaded dataset using LLM-powered Pandas agents (e.g., “Show all frauds over $1000”). |
| 🚀 **Automated CI/CD Deployment** | GitHub Actions pipeline for linting, testing, artifact syncing with DVC, building/pushing container images to GHCR, and auto-deploying to Streamlit Cloud. |

---

💡 **Enjoyed this project?**  

If you found this repository **helpful** or **inspiring**, please consider:

⭐ **Starring** the repo — it helps others discover the project  
🍴 **Forking** it — so you can explore, customize, and build upon it  

Your support keeps the innovation going! 🚀

---

📂 **Testing the App**  

To try out the application, you can:  
- 📎 Use the **[Sample Dataset](<LINK_TO_SAMPLE_DATASET>)** I’ve attached.  
- 🛠 Or generate your own dataset using the [`sample_data_creation.py`](sample_data_creation.py) file in this repository.  

By default, it creates **5,000 records**, but you can easily modify the script to produce **any number of records** you want.  


### Notes & Conventions
* **Security:** Keep `GROQ_API_KEY`, `LANGCHAIN_API_KEY=<your_key>`, `MLFLOW_TRACKING_USERNAME=<your_username>`, `MLFLOW_TRACKING_PASSWORD=<your_password>` &`MLFLOW_TRACKING_URI=<your_tracking_uri>`  in `.env` (LLM features are optional)
* **Reproducibility:** Lock versions in `requirements.txt`; track runs with MLflow

[![GitHub Repo Stars](https://img.shields.io/github/stars/anupamknn/fraud_detection_-_risk_analysis?style=social)](https://github.com/anupamknn/fraud_detection_-_risk_analysis)
