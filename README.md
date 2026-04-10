

---

# 🔢 NLP Feature Engineering App

An interactive **Streamlit application** that transforms raw text into numerical representations using core **NLP feature engineering techniques** like **Bag of Words (BoW), TF-IDF, and N-grams**.

---

## 🚀 Features

* 📂 Upload CSV datasets
* 🔍 Select any text column dynamically
* ⚙️ Multiple feature extraction methods:

  * Bag of Words (BoW)
  * TF-IDF (Term Frequency – Inverse Document Frequency)
* 🔗 Flexible N-gram support:

  * Unigrams
  * Bigrams
  * Trigrams
  * Combined (Unigram + Bigram)
* 🎚️ Adjustable feature size (`max_features`)
* 📊 Feature matrix visualization
* 🔝 Top feature importance display
* 📥 Download generated feature matrix

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **Scikit-learn**

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nlp-feature-engineering-app.git
cd nlp-feature-engineering-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

---

## ⚙️ How It Works

1. Upload a CSV file
2. Select the text column
3. Choose:

   * Feature extraction method (BoW / TF-IDF)
   * N-gram range
   * Max number of features
4. Click **"Generate Features"**
5. View:

   * Feature matrix
   * Top features
6. Download results

---

## 🧪 Feature Engineering Pipeline

```
Raw Text
   ↓
Vectorization (BoW / TF-IDF)
   ↓
N-gram Extraction
   ↓
Feature Matrix (Numerical Representation)
```

---

## 📊 Example Use Cases

* Text classification (ML models)
* Sentiment analysis
* Spam detection
* Document similarity analysis
* NLP model preprocessing pipelines

---

## ⚠️ Notes

* Input column must contain textual data
* Output matrix size depends on `max_features`
* Larger n-grams increase dimensionality and sparsity
* TF-IDF normalizes importance, unlike raw counts in BoW

---

## 🤝 Contributing

Contributions are welcome!
Fork the repo and submit a pull request.

---
