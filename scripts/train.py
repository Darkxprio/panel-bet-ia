import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import argparse # <-- Importamos argparse
from ml_models.advanced_predictor import AdvancedPredictor
from utils import get_logger, calculate_poisson_probabilities, safe_get_feature

logger = get_logger("trainer")

def train_models(dataset_path: str, model_output_path: str):
    logger.info(f"Cargando dataset desde {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df = df.dropna() # Eliminar filas con datos faltantes

    feature_columns = [col for col in df.columns if not col.startswith('result_')]
    X = df[feature_columns]
    
    logger.info("Inicializando Advanced Predictor para entrenamiento...")
    advanced_predictor = AdvancedPredictor()

    targets = {
        'goals_home': 'result_home_goals',
        'goals_away': 'result_away_goals',
        'btts': 'result_btts',
        'over_2_5': 'result_over_2_5',
        'over_0_5_fh': 'result_over_0_5_fh',
        'over_1_5_fh': 'result_over_1_5_fh'
    }

    for model_name, target_column in targets.items():
        logger.info(f"Entrenando modelo para '{model_name}'...")
        y = df[target_column]
        
        scaler = advanced_predictor.scalers[model_name]
        X_scaled = scaler.fit_transform(X)
        
        model = advanced_predictor.models[model_name]
        model.fit(X_scaled, y)

    advanced_predictor.is_trained = True
    logger.info("✅ Todos los modelos han sido entrenados.")

    joblib.dump(advanced_predictor, model_output_path)
    logger.info(f"🧠 Predictor avanzado entrenado y guardado en: {model_output_path}")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Entrenar los modelos de ML del Advanced Predictor.")
    parser.add_argument("--dataset", type=str, required=True, help="Ruta al archivo CSV del dataset de entrenamiento.")
    parser.add_argument("--output", type=str, required=True, help="Ruta donde se guardará el modelo entrenado (.pkl).")
    
    args = parser.parse_args()
    train_models(args.dataset, args.output)