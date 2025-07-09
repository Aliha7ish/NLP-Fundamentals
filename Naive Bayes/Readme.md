# 💬 Binary Sentiment Analysis with Naive Bayes (From Scratch)

This project implements a **Naive Bayes classifier** from scratch using **log priors** and **log likelihoods** for binary sentiment analysis (positive vs. negative). No machine learning libraries like `scikit-learn` were used — just raw Python and math!

The goal is to classify tweets or sentences as either **positive** or **negative** based on word frequencies learned from labeled data.

---

## 📚 Key Concepts

### 🧠 What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on Bayes’ Theorem, with the “naive” assumption that all features (words) are **independent** given the label.

It works great for:
- Spam detection
- Sentiment analysis
- Document classification

### ⚙️ Implementation Details

- **Training** is done using:
  - `logprior = log(P(positive) / P(negative))`
  - `loglikelihood[word] = log(P(word|positive) / P(word|negative))`
- Prediction is made using:
  - `logprob = logprior + sum(loglikelihood[word])`

> This method avoids numerical underflow by computing everything in log-space.

---

## 🧾 Dataset

The dataset used in this project is a set of labeled tweets (can be substituted with any binary-labeled text set):

- Format: (text, label)  
  - `label = 1` → positive  
  - `label = 0` → negative

---


## ⚙️ How to Run

### 1. Clone the repository

```
git clone https://github.com/Aliha7ish/NLP-Fundamentals.git
cd NLP-Fundamentals/Naive Bayes
```

### 2- Install dependencies

```
pip install -r requirements.txt
```
