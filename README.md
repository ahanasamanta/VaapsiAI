# VaapsiAI — E-Commerce Return Reason Prediction System

E-commerce return rates in clothing hover between 25–40% globally. 
Most sellers find out why their products are being returned weeks after the damage is done — through manual logs, gut feeling, or not at all.

VaapsiAI was built to change that. It reads a customer review the moment it's posted and predicts the return reason automatically, giving sellers 
the chance to act before a pattern becomes a crisis.

## The 5 Categories

Every return gets classified into one of:

**Defective · Wrong Item · Not as Described · Changed Mind · Sizing Issue**

The model takes a customer review, star rating, product category, and price as input — and returns a predicted label with a confidence score 
and full sentiment breakdown.

## Model Performance

Two classifiers were trained and evaluated on a 4,085-row balanced dataset built from 2,000 raw records through a six-step preprocessing pipeline 
(text cleaning → sentiment scoring → feature engineering → class balancing).

| Logistic Regression | Accuracy - 89.47% | Weighted F1 score - 0.8943 
| Random Forest | Accuracy - 95.83% | Weighted F1 score - 0.9584 

Random Forest was selected for deployment. The strongest predictors were star rating, sentiment_compound, price tier, and TF-IDF bigrams like 
"wrong size", "not fit", and "defect arrived".

## Tech Stack

`Python` · `Flask` · `scikit-learn` · `TF-IDF (bigrams, 500 dims)` · 
`pandas` · `numpy` · `HTML` · `CSS` · `JavaScript` · `joblib`

## Seller Dashboard

The frontend was designed with a non-technical seller in mind. Seven pages — overview analytics, real-time prediction interface, review 
logging, product catalogue, returns history, platform-specific pages (Amazon, Flipkart, Meesho, Myntra), and a seller profile module.

The prediction page surfaces not just the label, but the confidence percentage, positive/negative/compound sentiment scores, and probability 
bars across all five classes — so the seller understands the reasoning, not just the answer.

## Repository Structure

|  File                           | Description                                  |
|---------------------------------|-------------|
| `app.py`                        | Flask backend + `/predict` REST API          |
| `data_pipeline.py`              | Preprocessing + model training pipeline      |
| `index.html`                    | Full seller dashboard (single-file frontend) |
| `returns_clothing_clean.csv`    | Cleaned, balanced dataset (4,085 rows)       |
| `tfidf.pkl`                     | Fitted TF-IDF vectorizer                     |
| `requirements.txt`              | Project dependencies                         |
| `VaapsiAI_Practicum_Report.pdf` | Full project documentation                   |

> `rf_model.pkl` is excluded due to size (29.6MB).  
> To reproduce: run `data_pipeline.py` against `returns_clothing_clean.csv`.

## Getting Started

```bash
pip install -r requirements.txt
python app.py
```

Navigate to `http://localhost:5000` to open the dashboard.

*Practicum (Integrated Project) · B.Tech AI & DS · JEMTEC, 
affiliated to GGSIPU, New Delhi · 2026*