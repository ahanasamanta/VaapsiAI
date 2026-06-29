# =============================================================================
# DATA PIPELINE — Return Reason Prediction System
# Person 1: Data Engineer
# Project: VapsiAI
# Output: returns_clothing_clean.csv (5270 rows x 24 cols)
# =============================================================================

import pandas as pd
import numpy as np
import re
import os
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# -----------------------------------------------------------------------------
# STEP 0 — Mount Google Drive
# -----------------------------------------------------------------------------
drive.mount('/content/drive')

FOLDER       = '/content/drive/MyDrive/VapsiAI'
REVIEWS_PATH = os.path.join(FOLDER, 'train.csv')
ONLINE_PATH  = os.path.join(FOLDER, 'Online Sales Data.csv')
OUTPUT_PATH  = os.path.join(FOLDER, 'returns_clothing_clean.csv')

# -----------------------------------------------------------------------------
# STEP 1 — Load Raw Datasets
# -----------------------------------------------------------------------------
print("Loading datasets...")
df_raw    = pd.read_csv(REVIEWS_PATH)
online_df = pd.read_csv(ONLINE_PATH)
print(f"  Amazon Reviews raw shape : {df_raw.shape}")
print(f"  Online Sales shape       : {online_df.shape}")

# -----------------------------------------------------------------------------
# STEP 2 — Filter to English & Map to 4 Product Categories
# -----------------------------------------------------------------------------
df_raw = df_raw[df_raw['language'] == 'en'].copy()

CATEGORY_MAP = {
    'electronics'             : 'Electronics',
    'pc'                      : 'Electronics',
    'wireless'                : 'Electronics',
    'camera'                  : 'Electronics',
    'apparel'                 : 'Clothing & Apparel',
    'shoes'                   : 'Clothing & Apparel',
    'beauty'                  : 'Skincare & Beauty',
    'drugstore'               : 'Skincare & Beauty',
    'personal_care_appliances': 'Skincare & Beauty',
    'home'                    : 'Home & Kitchen',
    'kitchen'                 : 'Home & Kitchen',
    'furniture'               : 'Home & Kitchen',
}

df_raw = df_raw[df_raw['product_category'].isin(CATEGORY_MAP.keys())].copy()
df_raw['product_category'] = df_raw['product_category'].map(CATEGORY_MAP)
print(f"\nAfter category filter: {df_raw.shape}")
print(df_raw['product_category'].value_counts())

# -----------------------------------------------------------------------------
# STEP 3 — Sample & Standardise Schema
# -----------------------------------------------------------------------------
df = df_raw.groupby('product_category').apply(
    lambda x: x.sample(min(800, len(x)), random_state=42)
).reset_index(drop=True)

df = df.rename(columns={
    'stars'       : 'rating',
    'review_body' : 'review_text',
    'review_title': 'review_summary'
})

df['price_usd']      = np.random.uniform(10, 150, size=len(df)).round(2)
df['order_quantity'] = np.random.randint(1, 5, size=len(df))
df['brand']          = 'Unknown'
print(f"\nSampled shape: {df.shape}")

# -----------------------------------------------------------------------------
# STEP 4 — Handle Missing Values
# -----------------------------------------------------------------------------
df['review_text'] = df['review_text'].fillna('no review provided')
df['rating']      = pd.to_numeric(df['rating'], errors='coerce')
df['rating']      = df['rating'].fillna(df['rating'].median())
df.drop_duplicates(subset=['review_id'], inplace=True)
df.dropna(subset=['product_category'], inplace=True)
print(f"\nAfter cleaning nulls: {df.shape}")
print(f"Remaining nulls: {df.isnull().sum().sum()}")

# -----------------------------------------------------------------------------
# STEP 5 — Assign Return Reason (rule-based from review text + rating)
# -----------------------------------------------------------------------------
def assign_return_reason(row):
    text   = str(row['review_text']).lower()
    rating = float(row['rating'])

    if any(w in text for w in ['defect','damage','broken','fault','not working','stopped','malfunction']):
        return 'Defective'
    elif any(w in text for w in ['wrong','incorrect','mismatch','different','not what','mistaken']):
        return 'Wrong Item'
    elif any(w in text for w in ['size','small','large','tight','loose','fit','short','long','big']):
        return 'Sizing Issue'
    elif any(w in text for w in ['color','description','picture','looks','mislead','fake','advertised']):
        return 'Not as Described'
    elif rating >= 4:
        return 'Changed Mind'
    elif rating <= 2:
        return 'Defective'
    else:
        return 'Changed Mind'

df['return_reason'] = df.apply(assign_return_reason, axis=1)
print(f"\nReturn reason distribution:\n{df['return_reason'].value_counts()}")

# -----------------------------------------------------------------------------
# STEP 6 — Text Cleaning
# -----------------------------------------------------------------------------
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

df['review_clean'] = df['review_text'].apply(clean_text)
print("\nText cleaning complete.")

# -----------------------------------------------------------------------------
# STEP 7 — Sentiment Analysis (rule-based, no external API)
# -----------------------------------------------------------------------------
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

df[['sentiment_positive','sentiment_negative','sentiment_compound']] = \
    df['review_clean'].apply(lambda x: pd.Series(get_sentiment(x)))
print("Sentiment analysis complete.")

# -----------------------------------------------------------------------------
# STEP 8 — Feature Engineering
# -----------------------------------------------------------------------------
df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))

df['price_bucket_encoded'] = pd.cut(
    df['price_usd'],
    bins=[0, 25, 75, 120, 99999],
    labels=[0, 1, 2, 3]
).astype(float).astype('Int64')

df['is_high_value'] = (df['price_usd'] > 75).astype(int)
df = pd.get_dummies(df, columns=['product_category'], prefix='cat')
print(f"\nShape after feature engineering: {df.shape}")

# -----------------------------------------------------------------------------
# STEP 9 — Class Balancing (oversample minority classes)
# -----------------------------------------------------------------------------
print(f"\nBefore balancing:\n{df['return_reason'].value_counts()}")

majority_size = df['return_reason'].value_counts().max()
balanced = [
    resample(df[df['return_reason'] == r],
             replace=True, n_samples=majority_size, random_state=42)
    for r in df['return_reason'].unique()
]
df_balanced = pd.concat(balanced).reset_index(drop=True)
print(f"\nAfter balancing:\n{df_balanced['return_reason'].value_counts()}")
print(f"Final shape: {df_balanced.shape}")

# -----------------------------------------------------------------------------
# STEP 10 — Encode Target Column
# -----------------------------------------------------------------------------
le = LabelEncoder()
df_balanced['return_reason_encoded'] = le.fit_transform(df_balanced['return_reason'])
print(f"\nEncoding map: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Fix any remaining nulls
df_balanced.fillna('unknown', inplace=True)

# -----------------------------------------------------------------------------
# STEP 11 — Export Final Dataset
# -----------------------------------------------------------------------------
os.makedirs(FOLDER, exist_ok=True)
df_balanced.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")

# -----------------------------------------------------------------------------
# STEP 12 — Final Verification
# -----------------------------------------------------------------------------
df_check = pd.read_csv(OUTPUT_PATH)
print("\n========== FINAL VERIFICATION ==========")
print(f"Rows    : {len(df_check)}")
print(f"Columns : {len(df_check.columns)}")
print(f"Nulls   : {df_check.isnull().sum().sum()}")
print(f"\nClass distribution:\n{df_check['return_reason'].value_counts()}")
print(f"\nCategory columns: {[c for c in df_check.columns if c.startswith('cat_')]}")
print("========================================")
print("\nPipeline complete! returns_clothing_clean.csv is ready for Person 2.")
