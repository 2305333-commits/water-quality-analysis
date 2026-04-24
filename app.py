from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json, os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Train models on startup ──────────────────────────────────────────────────
# We generate synthetic training data since we don't have the CSV at runtime.
# In production: load your actual water_potability.csv here.

np.random.seed(42)
n = 2000

def make_synthetic_data(n):
    ph           = np.random.normal(7.08, 1.57, n)
    hardness     = np.random.normal(196, 32, n)
    solids       = np.random.normal(21917, 8642, n)
    chloramines  = np.random.normal(7.13, 1.58, n)
    sulfate      = np.random.normal(333, 41, n)
    conductivity = np.random.normal(426, 80, n)
    organic_c    = np.random.normal(14.36, 3.32, n)
    thm          = np.random.normal(66.4, 16, n)
    turbidity    = np.random.normal(3.97, 0.78, n)

    # Potability rule: roughly based on WHO thresholds
    score = (
        ((ph >= 6.5) & (ph <= 8.5)).astype(int) +
        (hardness < 250).astype(int) +
        (solids < 30000).astype(int) +
        (chloramines < 4).astype(int) +
        (sulfate < 400).astype(int) +
        (turbidity < 4).astype(int) +
        (thm < 80).astype(int) +
        (organic_c < 15).astype(int) +
        (conductivity < 500).astype(int)
    )
    potability = (score >= 6).astype(int)

    df = pd.DataFrame({
        'ph': ph, 'Hardness': hardness, 'Solids': solids,
        'Chloramines': chloramines, 'Sulfate': sulfate,
        'Conductivity': conductivity, 'Organic_carbon': organic_c,
        'Trihalomethanes': thm, 'Turbidity': turbidity,
        'Potability': potability
    })
    return df

df = pd.read_csv("water_potability.csv").dropna()
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr  = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
rf  = RandomForestClassifier(n_estimators=200, random_state=42)

lr.fit(X_train_sc, y_train)
knn.fit(X_train_sc, y_train)
rf.fit(X_train, y_train)

model_stats = {
    'Logistic Regression': round(accuracy_score(y_test, lr.predict(X_test_sc))*100, 2),
    'KNN':                 round(accuracy_score(y_test, knn.predict(X_test_sc))*100, 2),
    'Random Forest':       round(accuracy_score(y_test, rf.predict(X_test))*100, 2),
}

feat_names = list(X.columns)
feat_imp   = rf.feature_importances_.tolist()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        model_stats=model_stats,
        feat_names=feat_names,
        feat_imp=feat_imp)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        vals = [
            float(data['ph']),
            float(data['hardness']),
            float(data['solids']),
            float(data['chloramines']),
            float(data['sulfate']),
            float(data['conductivity']),
            float(data['organic_carbon']),
            float(data['trihalomethanes']),
            float(data['turbidity']),
        ]
        arr    = np.array(vals).reshape(1, -1)
        arr_sc = scaler.transform(arr)

        results = {}
        for name, model, use_scaled in [
            ('Logistic Regression', lr,  True),
            ('KNN',                 knn, True),
            ('Random Forest',       rf,  False),
        ]:
            inp   = arr_sc if use_scaled else arr
            pred  = int(model.predict(inp)[0])
            proba = model.predict_proba(inp)[0].tolist()
            results[name] = {
                'prediction': pred,
                'confidence': round(max(proba)*100, 1),
                'prob_safe':  round(proba[1]*100, 1),
                'prob_unsafe':round(proba[0]*100, 1),
            }

        # Majority vote
        votes    = sum(1 for r in results.values() if r['prediction'] == 1)
        majority = 1 if votes >= 2 else 0

        return jsonify({'success': True, 'results': results,
                        'majority': majority, 'votes': votes})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)