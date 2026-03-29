# 🌍 Language Identification System

**Author:** Nelson Ruthari Kariuki (24S01ACS002)  
**Course:** CSC423: Special Topics (NLP Term Project)  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://language-identification-system-hpcjmizeyip7iezdvwvnxf.streamlit.app/)

## 📌 Project Overview
This project is an end-to-end Natural Language Processing (NLP) pipeline designed to identify the language of short text inputs (1-2 sentences). It specifically targets a mix of high-resource, low-resource, and zero-resource African languages: **English, Swahili, Kikuyu, and Sheng** (Kenyan street slang).

## ✨ Key Features
* **Hybrid Data Collection:** Combines Hugging Face open-source datasets (English/Swahili), custom web scraping via Wikipedia (Kikuyu), and combinatorial synthetic data generation (Sheng).
* **Custom Slang Handling:** Utilizes RegEx to preserve unique morphological structures in hyphenated Sheng tokens prior to standard punctuation removal.
* **Character N-Gram Extraction:** Employs Scikit-Learn's `TfidfVectorizer` (2 to 4 character n-grams) to capture the underlying linguistic "DNA" rather than relying on strict word-level vocabularies.
* **Code-Mixing Analysis:** Powered by a Logistic Regression classifier capable of outputting exact probability distributions to handle code-mixed sentences (e.g., English/Swahili blending).
* **Interactive UI:** Fully deployed as a responsive web application using Streamlit.

## 📂 Repository Structure
```text
├── data/
│   ├── cleaned_master_dataset.csv     # Final preprocessed and balanced dataset (2000 rows) ready for ML
│   ├── english_swahili_data.csv       # Raw subset extracted from the Hugging Face NLP corpus
│   ├── kikuyu_data.csv                # Raw dataset scraped from Kikuyu Wikipedia
│   ├── master_language_dataset.csv    # The merged but uncleaned compilation of all four languages
│   └── sheng_data.csv                 # Raw synthetically generated Sheng street slang dataset
├── models/
│   ├── language_model_lr.pkl          # Final deployed Logistic Regression model (handles code-mixing)
│   ├── language_model_svm.pkl         # Archived Support Vector Machine model (baseline comparison)
│   └── tfidf_vectorizer.pkl           # Trained Character N-gram vectorizer (2-4 chars)
├── app.py                             # Streamlit web application frontend for real-time inference
├── Data_Preprocessing.ipynb           # Notebook for text cleaning, tokenization, and Sheng RegEx handling
├── Feature_Extraction.ipynb           # Notebook for transforming cleaned text into TF-IDF mathematical vectors
├── Kikuyu_Scraper.ipynb               # Web scraping script utilizing BeautifulSoup and NLTK for Wikipedia
├── LanguageID_Project.ipynb           # Comprehensive master notebook overviewing the project pipeline
├── Merging_Datasets.ipynb             # Script to concatenate and shuffle the isolated language datasets
├── Model_Training.ipynb               # Notebook for training, comparing, and evaluating the ML algorithms
├── README.md                          # Main project documentation
├── requirements.txt                   # List of Python dependencies for deployment
└── Sheng_Generator.ipynb              # Combinatorial matrix script for generating synthetic Sheng data
