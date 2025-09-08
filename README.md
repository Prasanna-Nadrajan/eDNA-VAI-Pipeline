# 🧬 eDNA-AI-Biodiversity  
### AI-Driven Deep-Sea Biodiversity Assessment from Environmental DNA (eDNA)

An innovative AI-first pipeline for analyzing deep-sea eDNA to accurately assess biodiversity and discover novel taxa. This project overcomes the limitations of traditional bioinformatics by leveraging unsupervised deep learning, minimizing reliance on incomplete reference databases.

---

## 🚨 Problem Statement

The Centre for Marine Living Resources and Ecology (CMLRE) collects eDNA samples from remote deep-sea ecosystems. Traditional tools like **BLAST** and **QIIME2** rely heavily on reference databases (e.g., NCBI), which lack comprehensive genetic sequences for deep-sea organisms. This leads to:

- ❌ Misclassified or unassigned genetic reads  
- 📉 Significant underestimation of biodiversity

> 🔍 **Key Finding:**  
> The provided `.tar.gz` and `.sqlite3` files were static BLAST databases—not raw sequencing data—highlighting the limitations of traditional reference-based methods.

---

## 💡 Our Solution: A Paradigm Shift

We introduce a novel, AI-driven pipeline that operates independently of reference databases. The core is built using **TensorFlow** and **Scikit-learn**, enabling unsupervised learning and discovery of novel taxa.

---

## ⚙️ Technical Workflow

### 1. 🔄 Data Pre-processing
- Convert raw eDNA reads (~200bp) into numerical **k-mer frequency vectors**
- Transforms genetic sequences into neural-network-compatible format

### 2. 🧠 Unsupervised Deep Learning
- Implement a **Variational Autoencoder (VAE)** using TensorFlow
- Learns intrinsic genetic patterns and embeds sequences into a lower-dimensional space

### 3. 🧬 Clustering & Discovery
- Use **DBSCAN** to:
  - Group similar sequences into distinct taxa
  - Detect outliers as **"noise" or novel taxa**

---

## 📊 Team Progress

| Module                        | Completion |
|------------------------------|------------|
| Research & Problem Validation| ✅ 100%     |
| Core AI Model Development    | 🔧 90%      |
| Backend Functional Prototype | 🔧 20%      |
| Frontend Design & UI/UX      | 🎨 20%      |

---

## 🧪 Under Development

### 🔙 Backend (`ai_pipeline.py`)
- Finalizing API endpoint for real-time data transfer
- Implements full AI pipeline: mock data → VAE → DBSCAN → JSON output

### 🔜 Frontend (`index.html`)
- Single-file dashboard with:
  - Interactive charts
  - Searchable table of identified and novel taxa
  - Dynamic API integration

---

## 🚀 Prototype & Demonstration

### ✅ `ai_pipeline.py`
- Simulates the full pipeline
- Outputs structured JSON with identified and novel taxa

### ✅ `index.html`
- Visualizes results with biodiversity metrics and abundance charts

---

## 🧰 Getting Started

### 🔧 Dependencies

Install required packages via pip:

```bash
pip install numpy pandas scikit-learn tensorflow
```

### ▶️ Run the Pipeline

```bash
python ai_pipeline.py
```

This will print the complete analysis results (identified + novel taxa) in JSON format to your console.

---

## 📁 Repository Structure

```
├── ai_pipeline.py       # Backend logic with VAE + DBSCAN
├── index.html           # Frontend dashboard
├── README.md            # Project documentation
└── requirements.txt     # (Optional) Dependency list
```

---

## 🌊 Impact

This project redefines biodiversity assessment in deep-sea ecosystems by:
- Eliminating dependency on incomplete reference databases
- Enabling discovery of previously unknown taxa
- Providing a scalable, full-stack AI solution for marine ecology

---

## 🤝 Acknowledgments

Special thanks to **CMLRE** for providing the dataset and research context. This project is part of the **Smart India Hackathon (SIH)** initiative to solve real-world scientific challenges using cutting-edge technology.
