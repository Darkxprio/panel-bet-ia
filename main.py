"""
Pipeline principal para Panel Bet IA

Ejecuta el proceso diario de análisis y predicción de apuestas deportivas.
"""

from datetime import date, datetime, timedelta
from typing import Dict, Any, List

from services import api_service, db_service
from ml_models.predictor import Predictor
from data_processing.feature_engineering import create_feature_vector
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
from utils import validate_match_data, validate_odds_data, format_team_names


def _validate_league_maturity(league_id: int, today: date) -> Dict[str, Any]:
    """
    Valida si una liga es lo suficientemente madura para hacer predicciones.
    
    Args:
        league_id: ID de la liga
        today: Fecha actual
        
    Returns:
        Diccionario con is_valid y season
    """
    league_details = api_service.get_league_details(league_id)
    is_valid, season = False, None
    
    if league_details:
        current_season = next((s for s in league_details.get('seasons', []) if s.get('current')), None)
        if current_season:
            season = current_season['year']
            start = datetime.strptime(current_season['start'], '%Y-%m-%d').date()
            if (today - start) > timedelta(days=MIN_LEAGUE_MATURITY_DAYS):
                is_valid = True
    
    return {'is_valid': is_valid, 'season': season}


def _analyze_all_markets(match: Dict, features: Any, initial_bets: List, prob_winner: Dict, prob_btts: float, prob_over_2_5: float, prob_over_0_5_fh: float, prob_over_1_5_fh: float) -> List[Dict]:
    """
    Analiza todos los mercados disponibles en busca de valor.
    
    Args:
        match: Datos del partido
        features: Características calculadas
        initial_bets: Cuotas iniciales
        prob_winner: Probabilidades 1X2
        prob_btts: Probabilidad BTTS
        prob_over_2_5: Probabilidad Over 2.5
        prob_over_0_5_fh: Probabilidad Over 0.5 FH
        prob_over_1_5_fh: Probabilidad Over 1.5 FH
        
    Returns:
        Lista de predicciones con valor
    """
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
        print(MSG_PARSING_ERROR.format(e))
    
    return valuable_predictions


def run_daily_pipeline():
    """
    Ejecuta el pipeline diario de predicción de apuestas.
    
    Proceso:
    1. Obtiene partidos del día
    2. Filtra ligas maduras
    3. Obtiene historial de equipos
    4. Genera características
    5. Predice probabilidades
    6. Busca apuestas de valor
    7. Guarda predicciones
    """
    print(MSG_PIPELINE_START)
    today = date.today()
    
    # Inicializar predictor
    predictor = Predictor()
    
    # Obtener partidos del día
    fixtures = api_service.get_daily_fixtures(today.strftime("%Y-%m-%d"))
    if not fixtures:
        print(MSG_NO_MATCHES)
        return

    valuable_predictions = []
    league_info_cache = {}
    
    for match in fixtures:
        # Validar datos del partido
        if not validate_match_data(match):
            print(f"  -> Omitiendo: Datos del partido inválidos")
            continue
        
        teams = format_team_names(match)
        print(f"\nAnalizando: {teams} (ID: {match['fixture']['id']})")
        
        # Validar madurez de la liga
        league_id = match['league']['id']
        league_info = league_info_cache.get(league_id)
        
        if not league_info:
            league_info = _validate_league_maturity(league_id, today)
            league_info_cache[league_id] = league_info
        
        if not league_info['is_valid']:
            print(MSG_LEAGUE_IMMATURE.format(league_id))
            continue
        
        # Obtener historial de equipos
        season_year = league_info['season']
        home_history = api_service.get_team_last_matches(match['teams']['home']['id'], season_year, last_n=20)
        away_history = api_service.get_team_last_matches(match['teams']['away']['id'], season_year, last_n=20)
        
        if len(home_history) < MIN_MATCHES_REQUIRED or len(away_history) < MIN_MATCHES_REQUIRED:
            print(MSG_INSUFFICIENT_DATA)
            continue
        
        # Generar características
        features = create_feature_vector(match, home_history, away_history)
        if features.empty:
            print(MSG_NO_FEATURES)
            continue
        
        # Obtener cuotas
        initial_bets = api_service.get_initial_odds(match['fixture']['id'])
        if not initial_bets or not validate_odds_data(initial_bets):
            print(MSG_NO_ODDS)
            continue

        print("  -> Prediciendo probabilidades para todos los mercados...")
        prob_winner = predictor.predict_winner_probabilities(features)
        prob_btts = predictor.predict_btts_probability(features)
        prob_over_2_5 = predictor.predict_over_2_5_probability(features)
        prob_over_0_5_fh = predictor.predict_over_0_5_fh_probability(features)
        prob_over_1_5_fh = predictor.predict_over_1_5_fh_probability(features) # <-- NUEVA PREDICCIÓN

        # Analizar mercados en busca de valor
        print("  -> Analizando mercados en busca de valor...")
        market_predictions = _analyze_all_markets(match, features, initial_bets, prob_winner, prob_btts, prob_over_2_5, prob_over_0_5_fh, prob_over_1_5_fh)
        valuable_predictions.extend(market_predictions)

    if valuable_predictions:
        print(f"\n✅ Pipeline completo. Se encontraron {len(valuable_predictions)} apuestas de valor. Guardando...")
        rows_saved = db_service.save_predictions(valuable_predictions)
        print(f"✅ Se guardaron exitosamente {rows_saved} nuevas predicciones.")
    else:
        print("\n✅ Pipeline completo. No se encontraron apuestas de valor el día de hoy.")

if __name__ == "__main__":
    run_daily_pipeline()