# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os
import numpy as np

def evaluate_model():
    # 1. Прочитайте файл с гиперпараметрами
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # 2. Загрузите данные и ранее обученную модель
    data = pd.read_csv('data/initial_data.csv')
    model = joblib.load('models/fitted_model.pkl')

    # 3. Разделите данные
    X = data.drop(columns='target')
    y = data['target']

    # 4. Настройте кросс-валидацию
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.get('random_state', 42))

    # 5. Проведите кросс-валидацию
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=['f1', 'roc_auc'],
        return_train_score=False,
        n_jobs=-1
    )

    # 6. Сохраните результаты
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as f:
        json.dump({k: round(float(np.mean(v)), 4) for k, v in cv_results.items()}, f)

if __name__ == '__main__':
    evaluate_model()
