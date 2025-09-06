import pandas as pd
import time
import os
import json
from services import api_service
from data_processing.feature_engineering import calculate_league_strengths, create_feature_vector
from utils import get_logger

logger = get_logger("data_collector")

SEASONS_TO_CHECK = [2024, 2023, 2022] 

def create_training_dataset(output_path: str):
    """
    Recolecta datos históricos de todas las ligas posibles, procesa las features
    y guarda el resultado en un dataset de entrenamiento reanudable.
    """
    processed_keys = set()
    if os.path.exists(output_path):
        logger.info(f"Archivo de dataset encontrado en {output_path}. Se intentará reanudar.")
        df_existing = pd.read_csv(output_path)
        for index, row in df_existing.iterrows():
            processed_keys.add(f"{row['league_id']}_{row['season']}_{row['fixture_id']}")
        logger.info(f"{len(df_existing)} partidos ya procesados. Se omitirán.")
        all_training_data = df_existing.to_dict('records')
    else:
        all_training_data = []

    all_leagues = api_service.get_all_leagues()
    if not all_leagues:
        logger.error("No se pudo obtener la lista de ligas. Abortando.")
        return

    for league_data in all_leagues:
        # === INICIO DE LA CORRECCIÓN ===
        league_info = league_data['league']
        league_id = league_info['id']
        league_name = league_info['name']
        
        for season in SEASONS_TO_CHECK:
            # Ahora buscamos en la ruta correcta: league_info['seasons']
            if any(s['year'] == season for s in league_info.get('seasons', [])):
        # === FIN DE LA CORRECCIÓN ===
                
                logger.info(f"--- Iniciando recolección para: {league_name} - Temporada {season} ---")
                
                season_fixtures = api_service.get_season_fixtures(league_id, season)
                if not season_fixtures or len(season_fixtures) < 50:
                    logger.warning(f"No se encontraron suficientes partidos ({len(season_fixtures)}) para {league_name} {season}. Omitiendo.")
                    continue

                logger.info("Calculando fuerzas de la liga (Dixon-Coles)...")
                league_strengths = calculate_league_strengths(season_fixtures)
                if not league_strengths:
                    logger.warning(f"No se pudieron calcular las fuerzas para {league_name} {season}. Omitiendo.")
                    continue

                total_matches = len(season_fixtures)
                logger.info(f"Generando features para {total_matches} partidos...")
                
                for i, match in enumerate(season_fixtures):
                    fixture_id = match['fixture']['id']
                    match_key = f"{league_id}_{season}_{fixture_id}"
                    if match_key in processed_keys:
                        continue

                    match_date = pd.to_datetime(match['fixture']['date'])
                    home_id = match['teams']['home']['id']
                    away_id = match['teams']['away']['id']

                    home_history = [m for m in season_fixtures if pd.to_datetime(m['fixture']['date']) < match_date and (home_id in (m['teams']['home']['id'], m['teams']['away']['id']))]
                    away_history = [m for m in season_fixtures if pd.to_datetime(m['fixture']['date']) < match_date and (away_id in (m['teams']['home']['id'], m['teams']['away']['id']))]

                    if len(home_history) < 5 or len(away_history) < 5:
                        continue

                    features = create_feature_vector(match, home_history, away_history, league_strengths, league_data)
                    
                    home_goals = match['goals']['home']
                    away_goals = match['goals']['away']
                    home_goals_ht = match['score']['halftime']['home']
                    away_goals_ht = match['score']['halftime']['away']

                    features['fixture_id'] = fixture_id
                    features['league_id'] = league_id
                    features['season'] = season
                    features['result_home_goals'] = home_goals
                    features['result_away_goals'] = away_goals
                    features['result_btts'] = 1 if home_goals > 0 and away_goals > 0 else 0
                    features['result_over_2_5'] = 1 if (home_goals + away_goals) > 2.5 else 0
                    features['result_over_0_5_fh'] = 1 if (home_goals_ht + away_goals_ht) > 0.5 else 0
                    features['result_over_1_5_fh'] = 1 if (home_goals_ht + away_goals_ht) > 1.5 else 0
                    
                    all_training_data.append(features)
                
                logger.info(f"Guardando {len(all_training_data)} filas en el CSV...")
                df = pd.DataFrame(all_training_data)
                df.to_csv(output_path, index=False)
                logger.info(f"Recolección para {league_name} {season} completada.")
                time.sleep(10)

    logger.info("✅ Proceso de recolección de datos finalizado.")

if __name__ == "__main__":
    create_training_dataset('data/training_dataset.csv')