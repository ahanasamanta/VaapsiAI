# =============================================================================
# FLASK BACKEND — Return Reason Prediction System
# Person 3: Frontend Developer
# Project: VapsiAI | JEMTC, Greater Noida
# =============================================================================
# HOW TO RUN:
#   1. Install dependencies:
#        pip install flask scikit-learn pandas numpy scipy joblib
#   2. Make sure model files are in the same folder (see STEP 0 below)
#   3. Run:   python app.py
#   4. Open:  http://127.0.0.1:5000  in your browser
# =============================================================================

import os
import re
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# =============================================================================
# STEP 0 — Load Trained Model & TF-IDF (saved by Person 2)
# =============================================================================
# Person 2 must save their model using:
#   import joblib
#   joblib.dump(rf_model, 'rf_model.pkl')
#   joblib.dump(tfidf,    'tfidf.pkl')
#
# If model files are not found, we fall back to a demo mode (rule-based).

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
TFIDF_PATH = os.path.join(os.path.dirname(__file__), 'tfidf.pkl')

model = None
tfidf = None

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️  Model files not found. Running in demo mode. ({e})")

# =============================================================================
# Label mapping — matches LabelEncoder.fit(df['return_reason'])
# LabelEncoder sorts alphabetically, so the order is:
#   0 → Changed Mind
#   1 → Defective
#   2 → Not as Described
#   3 → Sizing Issue
#   4 → Wrong Item
# =============================================================================
LABEL_MAP = {
    0: 'Changed Mind',
    1: 'Defective',
    2: 'Not as Described',
    3: 'Sizing Issue',
    4: 'Wrong Item'
}

LABEL_ICONS = {
    'Defective':        '🔧',
    'Wrong Item':       '📦',
    'Not as Described': '🏷️',
    'Changed Mind':     '💭',
    'Sizing Issue':     '📏'
}

LABEL_ADVICE = {
    'Defective':        'Likely a manufacturing or shipping quality issue. Prioritise QC checks.',
    'Wrong Item':       'Order fulfilment error detected. Review warehouse picking process.',
    'Not as Described': 'Product listing may be misleading. Update images or description.',
    'Changed Mind':     'Customer preference change — consider offering exchange options.',
    'Sizing Issue':     'Size guide may need improvement. Add detailed measurements to listing.'
}

# =============================================================================
# STEP 1 — Text cleaning (same logic as Person 1's pipeline)
# =============================================================================
STOP_WORDS = set([
    'i','me','my','we','our','you','your','he','she','it','they','them',
    'what','which','who','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','do','does','did',
    'will','would','could','should','may','might','shall','can','need',
    'a','an','the','and','but','or','nor','so','for','yet','both','either',
    'not','no','in','on','at','by','with','about','against','between',
    'through','during','before','after','above','below','to','from','up',
    'down','of','off','over','under','again','then','once','here','there',
    'when','where','why','how','all','any','each','more','most','other',
    'into','out','than','too','very','just','now','also','as','if','its'
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# =============================================================================
# STEP 2 — Sentiment scoring (same rule-based logic as Person 1)
# =============================================================================
POSITIVE_WORDS = set([
    'good','great','love','excellent','perfect','happy','satisfied','nice',
    'best','wonderful','fantastic','amazing','quality','comfortable',
    'recommend','pleased','awesome','superb'
])
NEGATIVE_WORDS = set([
    'bad','terrible','awful','worst','hate','poor','wrong','broken',
    'defective','damaged','disappointed','useless','horrible','return',
    'refund','never','waste','fake','small','large','tight','loose',
    'defect','fault','misleading','incorrect'
])

def get_sentiment(text):
    tokens = text.split()
    total  = len(tokens) if tokens else 1
    pos    = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg    = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return round(pos/total, 4), round(neg/total, 4), round((pos-neg)/(pos+neg+1), 4)

# =============================================================================
# STEP 3 — Rule-based fallback predictor (used when model is not loaded)
# =============================================================================
def rule_based_predict(review_text, rating):
    text = str(review_text).lower()
    if any(w in text for w in ['defect','damage','broken','fault','not working','stopped','malfunction']):
        return 0
    elif any(w in text for w in ['wrong','incorrect','mismatch','different','not what','mistaken']):
        return 1
    elif any(w in text for w in ['size','small','large','tight','loose','fit','short','long','big']):
        return 4
    elif any(w in text for w in ['color','description','picture','looks','mislead','fake','advertised']):
        return 2
    elif float(rating) >= 4:
        return 3
    elif float(rating) <= 2:
        return 0
    else:
        return 3

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    review_text = data.get('review_text', '').strip()
    rating      = float(data.get('rating', 3))
    price_usd   = float(data.get('price_usd', 50))
    order_qty   = int(data.get('order_quantity', 1))
    category    = data.get('product_category', 'Clothing & Apparel')

    if not review_text:
        return jsonify({'error': 'Please enter a review.'}), 400

    # Clean text & compute sentiment
    review_clean = clean_text(review_text)
    sent_pos, sent_neg, sent_comp = get_sentiment(review_clean)
    review_length = len(review_text.split())

    # Price features
    price_bucket = 0 if price_usd < 25 else (1 if price_usd < 75 else (2 if price_usd < 120 else 3))
    is_high_value = 1 if price_usd > 75 else 0

    # One-hot category
    cat_clothing   = 1 if category == 'Clothing & Apparel' else 0
    cat_electronics= 1 if category == 'Electronics' else 0
    cat_home       = 1 if category == 'Home & Kitchen' else 0
    cat_skincare   = 1 if category == 'Skincare & Beauty' else 0

    if model and tfidf:
        # ── Real model prediction ──────────────────────────────────────────
        import scipy.sparse as sp
        X_text   = tfidf.transform([review_clean])
        X_num    = np.array([[
            rating, review_length, price_usd, order_qty,
            sent_pos, sent_neg, sent_comp,
            price_bucket, is_high_value,
            cat_clothing, cat_electronics, cat_home, cat_skincare
        ]])
        X_final  = sp.hstack([X_text, sp.csr_matrix(X_num)])
        pred_idx = int(model.predict(X_final)[0])
        proba    = model.predict_proba(X_final)[0].tolist()
    else:
        # ── Demo / fallback ────────────────────────────────────────────────
        pred_idx = rule_based_predict(review_text, rating)
        proba    = [0.05] * 5
        proba[pred_idx] = 0.80

    predicted_label = LABEL_MAP[pred_idx]
    confidence      = round(proba[pred_idx] * 100, 1)

    # Build class probability list for chart
    class_probas = [
        {'label': LABEL_MAP[i], 'prob': round(proba[i]*100, 1)}
        for i in range(5)
    ]

    return jsonify({
        'prediction':  predicted_label,
        'icon':        LABEL_ICONS[predicted_label],
        'confidence':  confidence,
        'advice':      LABEL_ADVICE[predicted_label],
        'probabilities': class_probas,
        'mode':        'model' if (model and tfidf) else 'demo'
    })


# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)
