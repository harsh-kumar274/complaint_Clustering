Public Complaint Clustering & E‑Governance Dashboard An intelligent NLP‑driven complaint analysis system that automatically clusters large‑scale citizen complaints and presents actionable insights through an interactive dashboard, enabling data‑driven decision‑making for e‑governance.

📌 Project Overview Government grievance portals receive thousands of citizen complaints daily in unstructured textual form. Manual analysis of such data is slow, inefficient, and error‑prone.

This project addresses the problem by:

Converting complaint text into semantic embeddings

Grouping similar complaints using unsupervised machine learning

Storing results in a query‑efficient database

Visualizing insights via an interactive dashboard

🎯 Key Features 🔍 Automatic Complaint Clustering using MiniBatch K‑Means

🧠 Semantic Understanding with transformer‑based sentence embeddings

📊 Interactive Dashboard with filters, pagination, and charts

🗂️ Fast Data Retrieval using SQLite database

📉 UMAP Visualization for cluster validation

📥 CSV Export for reporting and analysis

🧩 System Architecture Synthetic / Real Complaint Data ↓ Text Preprocessing (NLP) ↓ Sentence Embeddings (MiniLM) ↓ Clustering (MiniBatch K‑Means) ↓ SQLite Database ↓ Streamlit Dashboard UI ⚙️ Technologies Used Programming Language: Python

NLP: SentenceTransformers (all‑MiniLM‑L6‑v2), NLTK

Machine Learning: Scikit‑Learn (MiniBatch K‑Means)

Dimensionality Reduction: UMAP

Database: SQLite

Visualization: Plotly

Dashboard: Streamlit

Data Handling: Pandas, NumPy

📁 Project Structure ├── synthetic_complaints.csv ├── large_pipeline_200k.py # ML pipeline (embeddings + clustering) ├── import_to_sqlite.py # Load clustered data into SQLite ├── streamlit_sql_app.py # Dashboard application ├── output/ │ ├── embeddings_memmap.npy │ ├── labels_kmeans_k12.npy │ ├── clustered_complaints_200k.csv │ ├── complaints.db │ ├── umap_sample.npy │ └── kmeans_topics_k12.json ├── requirements.txt └── README.md 🚀 How to Run the Project 1️⃣ Clone the Repository git clone https://github.com/harsh-kumar274/complaint_Clustering.git cd public-complaint-clustering-dashboard 2️⃣ Install Dependencies pip install -r requirements.txt 3️⃣ Run the ML Pipeline python large_pipeline_200k.py This will:

Generate sentence embeddings

Perform clustering

Compute UMAP visualization

Save clustered results

4️⃣ Import Data into SQLite python import_to_sqlite.py 5️⃣ Launch the Dashboard streamlit run streamlit_sql_app.py 📊 Dashboard Capabilities Filter complaints by cluster, state, and keywords

Paginated table view for large datasets

Cluster distribution bar chart (multi‑color)

UMAP scatter plot for cluster validation

Download filtered complaint data

🧪 Model Evaluation Since this is an unsupervised learning problem, traditional accuracy is not applicable.

Evaluation methods used:

Silhouette Score for cluster quality

Keyword interpretability per cluster

UMAP visualization for separation validation

Domain knowledge alignment with real complaint categories

🌍 Real‑World Applications Smart governance & grievance redressal systems

Complaint prioritization and issue tracking

Policy planning and resource allocation

Scalable text analytics for public administration

🔮 Future Enhancements Integration with real government complaint portals

Multilingual complaint support

Sentiment and urgency analysis

Real‑time streaming complaint ingestion

Cloud deployment (AWS / Azure)
