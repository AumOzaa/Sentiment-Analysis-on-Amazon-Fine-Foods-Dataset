
# 📊 Sentiment Analysis on Amazon Fine Food Reviews

This project performs sentiment analysis on Amazon Fine Food product reviews using both:

- **VADER (Valence Aware Dictionary and sEntiment Reasoner)** for lexicon-based scoring
- **`Roberta-base` from Hugging Face Transformers** for deep learning-based classification

The goal is to analyze customer sentiment (positive/negative/neutral) from written reviews.

---

## 📂 Project Structure

```
📁 Sentiment_Analysis/
├── Sentiment_Analysis.ipynb   # Main notebook with analysis
├── requirements.txt           # All necessary dependencies
├── README.md                  # This file
```

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `nltk` → VADER sentiment analyzer
  - `transformers` → Hugging Face Roberta model
  - `scikit-learn` → Evaluation metrics and model utilities

---

## 🧪 How to Run

1. **Install requirements**

```bash
pip install -r requirements.txt
```

2. **Run the Jupyter Notebook**

```bash
jupyter notebook Sentiment_Analysis.ipynb
```

---

## 📈 Output

- VADER-based polarity scores (`positive`, `negative`, `neutral`, `compound`)
- RoBERTa predictions (fine-tuned or zero-shot)
- Evaluation metrics like accuracy and confusion matrix

---

## 🧠 Learning Outcomes

- How to clean and preprocess text
- Lexicon vs Transformer-based sentiment analysis
- Using Hugging Face models for zero-shot classification
- Comparing and evaluating model performance

---

## 📌 Notes

- Dataset: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Roberta model used: `cardiffnlp/twitter-roberta-base-sentiment` from Hugging Face

---

## ✅ Requirements

Here’s a starter `requirements.txt` file too:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
transformers
torch
```

You can generate the full list using:

```bash
pip freeze > requirements.txt
```

---