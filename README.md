# 🏨 HotelOps AI — Booking Behaviour Analytics

A Streamlit web app that turns hotel booking data and a trained Logistic Regression model into actionable operational intelligence for hotel managers.

## 📊 Features

| Page | What it does |
|------|-------------|
| **Overview Dashboard** | KPIs, cancellation rates, lead time distribution, monthly ADR trends |
| **Booking Risk Scorer** | Enter any booking → get AI cancellation probability + recommended actions |
| **Guest Segment Analysis** | Cancel rates by customer type, market segment, deposit type, loyalty |
| **Demand & Seasonality** | Monthly demand patterns, dynamic staffing planner |
| **Meal & Operations** | Meal preference breakdown, food waste cost estimator |
| **What-If Simulator** | Simulate revenue recovery from loyalty, deposit, and reconfirmation strategies |

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub

```bash
# Create a new repo on GitHub, then:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo, branch `main`, and file `app.py`
5. Click **"Deploy"** — done in ~2 minutes

Your app will be live at:
`https://YOUR_USERNAME-YOUR_REPO-app-XXXX.streamlit.app`

## 🗂️ File Structure

```
hotel_app/
├── app.py                      # Main Streamlit application
├── hotel_model.pkl             # Trained Logistic Regression pipeline
├── combined_hotel_dataset.csv  # Hotel booking dataset (119,390 rows)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚠️ Important: sklearn Version

The model was trained with **scikit-learn 1.6.1**. The `requirements.txt` pins this version exactly. Do not change it or the model will fail to load.

## 🏃 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Dependencies

- `streamlit` — web app framework
- `pandas` / `numpy` — data processing
- `scikit-learn==1.6.1` — model loading (version must match training)
- `joblib` — model serialisation
- `plotly` — interactive charts

## 📈 Model Info

- **Algorithm:** Logistic Regression (sklearn Pipeline)
- **Preprocessing:** ColumnTransformer with OneHotEncoder (categorical) + passthrough (numeric)
- **Accuracy:** ~80%
- **Target:** `is_canceled` (1 = cancelled, 0 = completed)
- **Features:** 26 features including lead time, deposit type, customer type, special requests, booking modifications
