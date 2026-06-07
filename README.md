# Taller 3 — MLOps: Predicción de desempeño estudiantil

Proyecto CRISP-DM con notebook de análisis, experimentación (Optuna + MLflow) y API FastAPI para inferencia.

## Ver experimentos MLflow (para revisión)

```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir en el navegador: **http://127.0.0.1:5000**

- **Experimento:** `student_performance_gradeclass`
- **Runs:** 6 modelos (`logistic_regression`, `knn`, `svc`, `decision_tree`, `random_forest`, `xgboost`)
- Métricas registradas: `cv_f1_macro_best`, `test_accuracy`, `test_precision_macro`, `test_recall_macro`, `test_f1_macro`

## Levantar la API

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Probar predicción (ejemplos por defecto):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{}'
```
