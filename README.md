# 🧠 Fake Job Post Prediction

This project predicts whether a job posting is **real or fake** using both **Machine Learning (ML)** and **Deep Learning (DL)** models. It applies **NLP preprocessing**, trains multiple models, and deploys the final system with **FastAPI (backend)** and a **basic HTML frontend**.  

---


---

## 🚀 Features  

- Preprocessed text (lowercasing, punctuation removal, stopwords removal, tokenization).  
- ML models trained (Logistic Regression, Random Forest, SVM).  
- Deep Learning model (Keras Sequential with Embedding + LSTM/Dense layers).  
- Comparison of ML vs DL performance.  
- Word importance analysis (which words indicate fake vs real).  
- REST API with FastAPI for predictions.  
- Frontend (HTML + JS + Axios) for user interaction.  

---

## 📊 Results  

### 🔹 Machine Learning (Logistic Regression)  

- Accuracy: **97%**  
- High precision but lower recall on **fake job class** (misses some fraudulent jobs).


# Regression model report #
<img width="662" height="267" alt="Screenshot 2025-08-29 014525" src="https://github.com/user-attachments/assets/3c38a050-f67c-4aec-a759-ba156ac8adde" />


👉 This shows the **classification report** of Logistic Regression.  
- The model is **excellent at predicting real jobs**.  
- But it sometimes misses **fake jobs**, meaning it’s conservative.  

---

### 🔹 Deep Learning Model  

- Accuracy: **98%**  
- Better **recall** on fake jobs compared to ML.  
- Learns deeper patterns in text (e.g., scammy language, repeated phrases).  

## LSTIM model report ##
<img width="623" height="227" alt="Screenshot 2025-08-29 015100" src="https://github.com/user-attachments/assets/c39a07ad-981e-406e-ac87-0e2688290c8b" />
 

👉 This shows the **classification report** for the Deep Learning model.  
- Both **precision & recall are well-balanced**.  
- Performs better on **fraudulent class** than Logistic Regression.  

📌## Training Accuracy over every Epoch ## 
<img width="1077" height="457" alt="image" src="https://github.com/user-attachments/assets/d104ba44-4c0c-4c43-ba61-1260bcb78d10" />
 

👉 This plot shows the **training & validation accuracy over epochs**.  
- Accuracy improves steadily and converges without overfitting.  

---

### 🔹 Word Importance  

## Words that influences the most in both positive and negative way ##
<img width="741" height="635" alt="Screenshot 2025-08-29 014933" src="https://github.com/user-attachments/assets/c8036e88-f4e9-40b3-9b2d-7e5d6da20909" />


👉 This highlights the most **important words**:  
- Fake job indicators: *money, entry-level, clerk, earn, immediate*.  
- Real job indicators: *team, project, developer, client, experience*.  

This helps explain **why the model makes predictions**.  

---

## 🔀 ML vs DL Comparison  

| Aspect              | Machine Learning (LogReg) | Deep Learning (LSTM) |
|---------------------|---------------------------|-----------------------|
| Accuracy            | ~97%                     | ~98%                 |
| Precision (Fake)    | High                     | High                 |
| Recall (Fake)       | Lower (misses some)      | Better (catches more)|
| Training Time       | Fast (seconds)           | Slower (minutes)     |
| Interpretability    | Easy (coefficients)      | Harder (black box)   |

✅ **Takeaway**:  
- Logistic Regression is simpler & interpretable, good for a baseline.  
- Deep Learning generalizes better for fraud detection.  

---

## ⚙️ Tech Stack  
---

## 📚 Libraries Used  

- **NumPy** → For numerical computations and array handling.  
- **Pandas** → For dataset loading, cleaning, and manipulation.  
- **Matplotlib & Seaborn** → For data visualization and plotting graphs.  
- **Scikit-learn** → For preprocessing, feature extraction (TF-IDF), and ML models (Logistic Regression, Random Forest, SVM).  
- **TensorFlow / Keras** → For Deep Learning model (Embedding, LSTM, Dense layers).  
- **NLTK / re (regex)** → For text preprocessing (stopwords removal, tokenization, cleaning text).  
- **FastAPI** → To create backend REST API for prediction.  
- **Uvicorn** → ASGI server to run FastAPI.  


---

- **Python 3.10+**  
- **FastAPI** (backend)  
- **HTML + Axios** (frontend)  
- **Scikit-learn** (ML models)  
- **TensorFlow/Keras** (DL model)  
- **Matplotlib / Seaborn** (visualizations)  

---


