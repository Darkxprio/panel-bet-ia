from datetime import date, datetime, timedelta
from typing import Dict, Any, List
import time

from services import api_service, db_service
from ml_models.predictor import Predictor
from data_processing.feature_engineering import create_feature_vector, calculate_league_strengths
from helpers import find_value_bet, create_prediction_entry
from constants import (
    MIN_MATCHES_REQUIRED,
    MIN_LEAGUE_MATURITY_DAYS,
    MSG_PIPELINE_START,
    MSG_NO_MATCHES,
    MSG_LEAGUE_IMMATURE,
    MSG_INSUFFICIENT_DATA,
    MSG_NO_FEATURES,
    MSG_NO_ODDS,
    MSG_PARSING_ERROR
)
from utils import (
    validate_match_data, 
    validate_odds_data, 
    format_team_names,
    setup_logging,
    get_logger,
    log_pipeline_summary,
    log_prediction_result
)

logger = setup_logging("panel_bet_ia", level="INFO")
api_logger = get_logger("api")
predictions_logger = get_logger("predictions")


def _validate_and_enrich_league(league_details: Dict, today: date) -> Dict:
    """
    Valida la madurez de la liga y la enriquece con las fuerzas de Dixon-Coles.
    """
    is_valid, season, strengths = False, None, {}
    
    if league_details:
        current_season = next((s for s in league_details.get('seasons', []) if s.get('current')), None)
        if current_season:
            season = current_season['year']
            start = datetime.strptime(current_season['start'], '%Y-%m-%d').date()
            
            if (today - start) > timedelta(days=MIN_LEAGUE_MATURITY_DAYS):
                season_fixtures = api_service.get_season_fixtures(league_details['league']['id'], season)
                if len(season_fixtures) > 30:
                    strengths = calculate_league_strengths(season_fixtures)
                    is_valid = True if strengths else False

    # Guardamos los detalles completos de la liga para usarlos en las features
    return {'is_valid': is_valid, 'season': season, 'strengths': strengths, 'details': league_details}


def _analyze_all_markets(match: Dict, features: Any, initial_bets: List, prob_winner: Dict, prob_btts: float, prob_over_2_5: float, prob_over_0_5_fh: float, prob_over_1_5_fh: float) -> List[Dict]:
    valuable_predictions = []
    
    try:
        # Mercado 1X2
        winner_bet = next((b for b in initial_bets if b['id'] == 1), None)
        if winner_bet:
            odds_map = {v['value']: float(v['odd']) for v in winner_bet['values']}
            if 'Home' in odds_map and find_value_bet(prob_winner['home'], odds_map['Home']):
                market_info = {'bet_id': 1, 'value_str': 'Home', 'current_best_odd': odds_map['Home']}
                best_odds = api_service.find_best_odds_for_market(match['fixture']['id'], market_info)
                valuable_predictions.append(create_prediction_entry(match, features, '1X2', 'Gana Local', best_odds['best_odd'], prob_winner['home']))
        
        # Mercado BTTS
        btts_bet = next((b for b in initial_bets if b['id'] == 8), None)
        if btts_bet:
            btts_yes_odds = next((float(v['odd']) for v in btts_bet['values'] if v['value'] == 'Yes'), None)
            if btts_yes_odds and find_value_bet(prob_btts, btts_yes_odds):
                market_info = {'bet_id': 8, 'value_str': 'Yes', 'current_best_odd': btts_yes_odds}
                best_odds = api_service.find_best_odds_for_market(match['fixture']['id'], market_info)
                valuable_predictions.append(create_prediction_entry(match, features, 'Ambos Anotan', 'Sí', best_odds['best_odd'], prob_btts))
        
        # Mercado Over/Under 2.5
        over_under_bet = next((b for b in initial_bets if b['id'] == 5), None)
        if over_under_bet:
            over_2_5_odds = next((float(v['odd']) for v in over_under_bet['values'] if v['value'] == 'Over 2.5'), None)
            if over_2_5_odds and find_value_bet(prob_over_2_5, over_2_5_odds):
                market_info = {'bet_id': 5, 'value_str': 'Over 2.5', 'current_best_odd': over_2_5_odds}
                best_odds = api_service.find_best_odds_for_market(match['fixture']['id'], market_info)
                valuable_predictions.append(create_prediction_entry(match, features, 'Goles Totales', 'Más de 2.5', best_odds['best_odd'], prob_over_2_5))
        
        # Mercado Goles Primera Mitad
        fh_goals_bet = next((b for b in initial_bets if b['id'] == 21), None)
        if fh_goals_bet:
            fh_odds_map = {v['value']: float(v['odd']) for v in fh_goals_bet['values']}
            
            # Over 0.5 FH
            if 'Over 0.5' in fh_odds_map and find_value_bet(prob_over_0_5_fh, fh_odds_map['Over 0.5']):
                market_info = {'bet_id': 21, 'value_str': 'Over 0.5', 'current_best_odd': fh_odds_map['Over 0.5']}
                best_odds = api_service.find_best_odds_for_market(match['fixture']['id'], market_info)
                valuable_predictions.append(create_prediction_entry(match, features, 'Goles 1ra Mitad', 'Más de 0.5', best_odds['best_odd'], prob_over_0_5_fh))
            
            # Over 1.5 FH
            if 'Over 1.5' in fh_odds_map and find_value_bet(prob_over_1_5_fh, fh_odds_map['Over 1.5']):
                market_info = {'bet_id': 21, 'value_str': 'Over 1.5', 'current_best_odd': fh_odds_map['Over 1.5']}
                best_odds = api_service.find_best_odds_for_market(match['fixture']['id'], market_info)
                valuable_predictions.append(create_prediction_entry(match, features, 'Goles 1ra Mitad', 'Más de 1.5', best_odds['best_odd'], prob_over_1_5_fh))
    
    except (StopIteration, KeyError, IndexError, ValueError) as e:
        logger.warning(f"⚠️ Error parseando cuotas: {e}")
    
    return valuable_predictions


def run_daily_pipeline():
    start_time = time.time()
    logger.info("🚀 Iniciando el pipeline de predicción diaria...")
    today = date.today()
    
    # Inicializar predictor
    try:
        predictor = Predictor()
        logger.info("✅ Predictor inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando predictor: {e}", exc_info=True)
        return
    
    # Obtener partidos del día
    fixtures = api_service.get_daily_fixtures(today.strftime("%Y-%m-%d"))
    if not fixtures:
        logger.warning("⚠️ No se encontraron partidos para hoy. Finalizando.")
        return

    logger.info(f"📅 Procesando {len(fixtures)} partidos para {today}")
    valuable_predictions = []
    league_info_cache = {}
    analyzed_matches = 0
    
    for match in fixtures:
        match_id = match.get('fixture', {}).get('id', 'unknown')
        
        # Validar datos del partido
        if not validate_match_data(match):
            logger.warning(f"⚠️ Partido {match_id}: Datos inválidos, omitiendo")
            continue
        
        teams = format_team_names(match)
        logger.info(f"🔍 Analizando: {teams} (ID: {match_id})")
        
        try:
            # Validar madurez de la liga
            league_id = match['league']['id']
            league_info = league_info_cache.get(league_id)
            
            if not league_info:
                logger.debug(f"Liga {league_id} no está en caché. Realizando análisis completo...")
                
                 # Obtenemos todos los detalles de la liga
                league_details = api_service.get_league_details(league_id)
                
                # Usamos una función de ayuda para la validación y el enriquecimiento
                league_info = _validate_and_enrich_league(league_details, today)
                
                # Guardar la información completa en el caché
                league_info_cache[league_id] = league_info
                
                # Si es válida, ENRIQUECERLA con las fuerzas de la liga
                if league_info['is_valid']:
                    season_year = league_info['season']
                    season_fixtures = api_service.get_season_fixtures(league_id, season_year)
                    
                    if len(season_fixtures) > 20:
                        strengths = calculate_league_strengths(season_fixtures)
                        # Añadir las fuerzas al diccionario de información de la liga
                        league_info['strengths'] = strengths
                        if not strengths:
                            league_info['is_valid'] = False # Marcar como inválida si no se pudieron calcular
                    else:
                        league_info['is_valid'] = False # No hay suficientes partidos en la temporada
                
                # Guardar la información completa (o la invalidación) en el caché
                league_info_cache[league_id] = league_info
            
            if not league_info['is_valid'] or not league_info.get('strengths'):
                logger.info(f"⏭️ Liga {league_id} no tiene suficientes datos para el modelo Dixon-Coles.")
                continue
            
            # Obtener historial de equipos
            season_year = league_info['season']
            league_strengths = league_info['strengths']
            home_history = api_service.get_team_last_matches(match['teams']['home']['id'], season_year, last_n=20)
            away_history = api_service.get_team_last_matches(match['teams']['away']['id'], season_year, last_n=20)
            
            if len(home_history) < MIN_MATCHES_REQUIRED or len(away_history) < MIN_MATCHES_REQUIRED:
                logger.info(f"⏭️ {teams}: Datos históricos insuficientes.")
                continue
            
            # Generar características
            features = create_feature_vector(match, home_history, away_history, league_info['strengths'],league_info['details'])
            if features.empty:
                logger.warning(f"⚠️ {teams}: No se pudieron generar características.")
                continue
            
            # Obtener cuotas
            initial_bets = api_service.get_initial_odds(match['fixture']['id'])
            if not initial_bets:
                logger.info(f"⏭️ {teams}: Sin cuotas disponibles del bookmaker primario.")
                continue

            # Realizar predicciones
             # 1. Realizar todas las predicciones pasando los argumentos correctos
            logger.debug(f"🤖 {teams}: Generando predicciones...")
            prob_winner = predictor.predict_winner_probabilities(match, features, league_strengths)
            prob_btts = predictor.predict_btts_probability(features)
            prob_over_2_5 = predictor.predict_over_2_5_probability(features)
            prob_over_0_5_fh = predictor.predict_over_0_5_fh_probability(features)
            prob_over_1_5_fh = predictor.predict_over_1_5_fh_probability(features)

            # 2. Analizar mercados pasando las probabilidades ya calculadas
            logger.debug(f"💰 {teams}: Buscando oportunidades de valor...")
            market_predictions = _analyze_all_markets(
                match, features, initial_bets, 
                prob_winner, prob_btts, prob_over_2_5, prob_over_0_5_fh, prob_over_1_5_fh
            )
            
            if market_predictions:
                valuable_predictions.extend(market_predictions)
                logger.info(f"✅ {teams}: {len(market_predictions)} predicciones con valor encontradas")
            else:
                logger.debug(f"❌ {teams}: Sin oportunidades de valor")
            
            analyzed_matches += 1
            
        except Exception as e:
            logger.error(f"❌ Error procesando {teams} (ID: {match_id}): {e}", exc_info=True)
            continue

    # Guardar resultados
    execution_time = time.time() - start_time
    
    if valuable_predictions:
        logger.info(f"💾 Guardando {len(valuable_predictions)} predicciones con valor...")
        try:
            rows_saved = db_service.save_predictions(valuable_predictions)
            logger.info(f"✅ Se guardaron exitosamente {rows_saved} predicciones en la base de datos")
        except Exception as e:
            logger.error(f"❌ Error guardando predicciones: {e}", exc_info=True)
            rows_saved = 0
    else:
        logger.info("ℹ️ No se encontraron apuestas de valor el día de hoy")
        rows_saved = 0
    
    # Log resumen final
    log_pipeline_summary(
        total_matches=len(fixtures),
        analyzed_matches=analyzed_matches,
        valuable_predictions=len(valuable_predictions),
        saved_predictions=rows_saved,
        execution_time=execution_time
    )

if __name__ == "__main__":
    run_daily_pipeline()