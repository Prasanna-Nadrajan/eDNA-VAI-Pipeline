# 🧬 AI-Driven Deep-Sea eDNA Biodiversity Assessment

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://edna-biodiversity-pipeline-production.up.railway.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An innovative, database-independent AI pipeline for assessing biodiversity in deep-sea ecosystems using environmental DNA (eDNA).

## � Overview

The Centre for Marine Living Resources and Ecology (CMLRE) collects eDNA samples from remote deep-sea ecosystems. Traditional bioinformatics tools (like BLAST) rely heavily on reference databases, which lack comprehensive genetic sequences for deep-sea organisms. This leads to misclassified reads and a significant underestimation of biodiversity.

**Our Solution**: An AI-first pipeline that leverages **unsupervised deep learning** to analyze genetic patterns directly, allowing for the identification of distinct taxa and the discovery of novel species without needing prior cataloging.

---

## ⚙️ Architecture & Workflow

The system follows a streamlined 4-stage pipeline:

### 1. 📥 Data Ingestion
*   **Input**: Raw `.fasta` or `.fastq` sequencing files.
*   **Process**: reads DNA sequences (A, T, C, G) and filters quality.

### 2. � Vectorization (K-mer Encoding)
*   **Method**: Converts biological sequences into numerical vectors using **k-mer frequency analysis** (default k=4).
*   **Output**: A high-dimensional numerical matrix representing the genetic signature of each sequence.

### 3. 🧠 AI Analysis (The Core)
*   **Variational Autoencoder (VAE)**: A deep neural network compresses the sparse k-mer vectors into a dense, continuous **latent space**. This captures the intrinsic structure of the genetic data.
*   **DBSCAN Clustering**: A density-based clustering algorithm groups the latent embeddings.
    *   **Dense Clusters** = Identified Taxa (Species/Genus)
    *   **Outliers/Noise** = **Potentially Novel Taxa** (Unknown to science)

### 4. 📊 Visualization & Metrics
*   **Latent Space Projection**: Visualizes the clusters using PCA (2D).
*   **K-mer Heatmaps**: Shows the frequency of dominant genetic patterns.
*   **Biodiversity Indices**: Automatically calculates Shannon, Simpson, and Pielou's evenness indices.

---

## ✨ Key Features

*   **Database-Independent**: Does not require reference databases like NCBI/GenBank.
*   **Novel Taxa Discovery**: Specifically designed to highlight genetic outliers that could be new species.
*   **Interactive Web Dashboard**: User-friendly interface to upload files and visualize analysis in real-time.
*   **Production-Ready**: Deployed and scalable using Flask and Gunicorn.

---

## 🏗️ Tech Stack

*   **Frontend**: HTML5, TailwindCSS, Chart.js
*   **Backend**: Python, Flask, Werkzeug
*   **Machine Learning**: TensorFlow (Keras), Scikit-learn, NumPy, Pandas
*   **Visualization**: Matplotlib, Seaborn
*   **Deployment**: Railway, Gunicorn

---

## � Installation & Usage

### Prerequisites
*   Python 3.8+
*   pip

### Local Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Prasanna-Nadrajan/eDNA-VAI-Pipeline.git
    cd eDNA-VAI-Pipeline
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```

5.  **Access the Dashboard**
    Open your browser and navigate to: `http://127.0.0.1:5000`

### Login Credentials (Demo)
*   **Admin**: `admin` / `password`
*   **User**: `alice` / `wonderland123`

---

## � Project Structure

```
├── app.py                 # Main Flask application entry point
├── main.py                # AI Pipeline logic (VAE, Clustering, Metrics)
├── sample.fasta           # Included sample dataset (50 sequences)
├── requirements.txt       # Python dependencies
├── Procfile               # Production startup command
├── railway.toml           # Cloud deployment config
├── templates/             # HTML Templates
│   ├── home.html          # Analysis dashboard
│   └── login.html         # Login page
└── plots/                 # Directory for generated visualizations
```

## 🤝 Acknowledgments

Special thanks to **CMLRE** for the inspiration and problem statement regarding deep-sea conservation. This project was developed to address real-world challenges in marine biodiversity assessment.

---
*License: MIT*
