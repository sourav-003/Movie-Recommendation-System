# 🎬 Movie Recommendation System (Content-Based)

## 📌 Project Overview
This project implements a **Content-Based Movie Recommendation System** using metadata from movies and credits datasets.  
It suggests movies similar to a given movie based on textual features such as **genres, cast, crew, keywords, and overview**.  

The system leverages **Natural Language Processing (NLP)** and **Cosine Similarity** to find and recommend movies with the highest similarity scores.  

---

## 📂 Dataset
The project uses two datasets:  

1. **movies.csv** → Contains columns like `id`, `title`, `genres`, `overview`, `keywords`.  
2. **credits.csv** → Contains columns like `cast`, `crew`, `title`.  

Both datasets were merged on the **title** column to create a single, enriched dataset.  

---

## 🎯 Objectives
- Perform **data cleaning and preprocessing** to handle nulls, duplicates, and irrelevant data.  
- Engineer features from text-based attributes (genres, cast, crew, keywords, overview).  
- Build a **content-based filtering system** using **vectorization and similarity metrics**.  
- Create a function that recommends **Top 5 movies** similar to any given input movie.  

---

## 🔎 Exploratory Data Analysis (EDA)
- Inspected dataset shape and column structure.  
- Checked missing values and dropped null entries.  
- Merged datasets to unify movie and credit information.  
- Extracted unique movie counts before and after cleaning.  
- Verified dataset integrity.  

---

## 🛠️ Methodology

### 1️⃣ Data Preprocessing
- Selected features: `genres`, `keywords`, `overview`, `cast`, `crew`.  
- Converted JSON-like fields into plain text lists.  
- Combined features into a single column (`context`) representing each movie.  

### 2️⃣ Feature Engineering
Applied **CountVectorizer** with `max_features=5000` and `stop_words='english'` to convert text data into vectors.  
This produced a **5000-dimensional sparse vector representation** for each movie.  

### 3️⃣ Similarity Computation
Used **Cosine Similarity** from scikit-learn:  
`from sklearn.metrics.pairwise import cosine_similarity`  
`similarity = cosine_similarity(vector)`  

This generates a **similarity matrix** where each entry `(i, j)` represents how similar movie `i` is to movie `j`.  

### 4️⃣ Recommendation Function
Defined a function that:  
- Locates the **movie index**  
- Retrieves **similarity scores**  
- Sorts them in **descending order**  
- Returns the **Top 5 most similar movies**  

Example:  
`recommendation("Avatar")`  

**Output →** A list of 5 movies similar to *Avatar*.  

---

## 📊 Results
- The system successfully recommends movies based on content similarity.  
- Example: Inputting *Avatar* recommends movies with similar **sci-fi, fantasy, or action** themes.  
- The model handles **~4800+ movies** efficiently with vectorization.  

---

## 🚀 How to Run

### 1️⃣ Clone Repository  
`git clone https://github.com/your-username/movie-recommendation-system.git`  
`cd movie-recommendation-system`  

### 2️⃣ Install Dependencies  
`pip install -r requirements.txt`  

### 3️⃣ Run Notebook  
`jupyter notebook "Recommendation System - Movie dataset.ipynb"`  

---

## 🖥️ Tech Stack
- **Python**  
- **Pandas, NumPy** → Data manipulation  
- **Matplotlib, Seaborn** → Visualization  
- **Scikit-learn** → CountVectorizer & Cosine Similarity  
- **NLP** → Text feature engineering  

---

## 🎬 Live Demo
The project is deployed and live on Hugging Face Spaces!  

👉 [Click here to try it out](https://huggingface.co/spaces/Sourav-003/movie-recommendation-system)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Sourav-003/movie-recommendation-system)

---

## 🔮 Future Improvements
- Deploy as a **Streamlit / Flask Web App** for real-time recommendations.  
- Add **Collaborative Filtering** to combine user ratings with content-based results.  
- Use **Deep Learning / Embeddings (Word2Vec, BERT, Doc2Vec)** for richer semantic similarity.  
- Optimize with **Approximate Nearest Neighbors (ANN)** for large-scale datasets.  

---

## 🙏 Acknowledgements
- Dataset from [TMDB (The Movie Database)](https://www.themoviedb.org/)  
- Libraries: scikit-learn, pandas, seaborn, matplotlib  

---

✨ Developed with ❤️ by [Your Name]  
