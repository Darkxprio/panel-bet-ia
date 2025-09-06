import requests
from config import API_HOST, API_HEADERS
from typing import List, Dict, Any
import time
from constants import PRIMARY_BOOKMAKER_ID, SECONDARY_BOOKMAKER_IDS, TEAM_HISTORY_MATCHES
from utils import get_logger, log_api_call

# Logger para API calls
logger = get_logger("api")

def get_daily_fixtures(date_str: str) -> list:
    url = f"https://{API_HOST}/fixtures"
    params = {"date": date_str}
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Buscando partidos para la fecha: {date_str}")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        
        response_time = time.time() - start_time
        fixtures = response.json().get('response', [])
        
        # Log API call
        log_api_call(
            endpoint="fixtures",
            params=params,
            response_time=response_time,
            status_code=response.status_code,
            response_size=len(str(fixtures))
        )
        
        logger.info(f"✅ Encontrados {len(fixtures)} partidos para {date_str}")
        return fixtures
        
    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        logger.error(f"❌ Error obteniendo partidos del día {date_str}: {e}")
        
        # Log failed API call
        log_api_call(
            endpoint="fixtures", 
            params=params,
            response_time=response_time,
            status_code=getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0
        )
        return []

def get_league_details(league_id: int) -> dict:
    url = f"https://{API_HOST}/leagues"
    params = {"id": str(league_id)}
    start_time = time.time()
    
    try:
        logger.debug(f"📋 Obteniendo detalles para la liga ID: {league_id}")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        
        response_time = time.time() - start_time
        league_data = response.json().get('response', [])
        
        log_api_call(
            endpoint="leagues",
            params=params,
            response_time=response_time,
            status_code=response.status_code,
            response_size=len(str(league_data))
        )
        
        if league_data:
            logger.debug(f"✅ Detalles obtenidos para liga {league_id}")
            return league_data[0]
        else:
            logger.warning(f"⚠️ No se encontraron detalles para liga {league_id}")
            return {}
        
    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        logger.error(f"❌ Error obteniendo detalles de liga {league_id}: {e}")
        
        log_api_call(
            endpoint="leagues",
            params=params,
            response_time=response_time,
            status_code=getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0
        )
        return {}

def get_team_last_matches(team_id: int, season: int, last_n: int = TEAM_HISTORY_MATCHES) -> list:
    url = f"https://{API_HOST}/fixtures"
    params = {
        "team": str(team_id),
        "season": str(season),
        "last": str(last_n)
    }
    try:
        logger.debug(f"📊 Obteniendo últimos {last_n} partidos para equipo {team_id}")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.5) 
        matches = response.json().get('response', [])
        logger.debug(f"✅ Obtenidos {len(matches)} partidos para equipo {team_id}")
        return matches
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error obteniendo historial del equipo {team_id}: {e}")
        return []

def _get_odds_from_single_bookmaker(fixture_id: int, bookmaker_id: int) -> List[Dict[str, Any]]:
    url = f"https://{API_HOST}/odds"
    params = {"fixture": str(fixture_id), "bookmaker": str(bookmaker_id)}
    
    try:
        logger.debug(f"💰 Obteniendo cuotas del bookmaker {bookmaker_id} para partido {fixture_id}")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.3)
        odds_response = response.json().get('response', [])
        if odds_response:
            bets = odds_response[0]['bookmakers'][0]['bets']
            logger.debug(f"✅ Obtenidas {len(bets)} cuotas del bookmaker {bookmaker_id}")
            return bets
        else:
            logger.warning(f"⚠️ Sin cuotas disponibles del bookmaker {bookmaker_id}")
            return []
    except (requests.exceptions.RequestException, IndexError) as e:
        logger.error(f"❌ Error obteniendo cuotas del bookmaker {bookmaker_id}: {e}")
        return []

def get_initial_odds(fixture_id: int) -> List[Dict[str, Any]]:
    return _get_odds_from_single_bookmaker(fixture_id, PRIMARY_BOOKMAKER_ID)

def find_best_odds_for_market(fixture_id: int, market_info: Dict) -> Dict[str, Any]:
    best_odd = market_info['current_best_odd']
    
    for bookmaker_id in SECONDARY_BOOKMAKER_IDS:
        bets = _get_odds_from_single_bookmaker(fixture_id, bookmaker_id)
        if not bets:
            continue
        
        try:
            target_bet = next(b for b in bets if b['id'] == market_info['bet_id'])
            target_value = next(v for v in target_bet['values'] if v['value'] == market_info['value_str'])
            new_odd = float(target_value['odd'])
            
            if new_odd > best_odd:
                logger.info(f"🎯 Mejor cuota encontrada: {new_odd} del bookmaker {bookmaker_id} (anterior: {best_odd})")
                best_odd = new_odd
                
        except (StopIteration, KeyError):
            continue
            
    return {'best_odd': best_odd}

def get_season_fixtures(league_id: int, season: int) -> List[Dict[str, Any]]:
    """
    Obtiene todos los partidos de una temporada para una liga específica,
    necesarios para el modelo Dixon-Coles.
    """
    url = f"https://{API_HOST}/fixtures"
    params = {"league": str(league_id), "season": str(season)}
    start_time = time.time()

    try:
        logger.info(f"📚 Obteniendo historial completo de la temporada para liga {league_id}, año {season}")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        
        response_time = time.time() - start_time
        all_fixtures = response.json().get('response', [])
        
        # Filtramos solo los partidos que ya han terminado (FT) para el análisis
        finished_fixtures = [f for f in all_fixtures if f['fixture']['status']['short'] == 'FT']

        log_api_call(
            endpoint="fixtures (season)",
            params=params,
            response_time=response_time,
            status_code=response.status_code,
            response_size=len(str(all_fixtures))
        )
        
        logger.info(f"✅ Encontrados {len(finished_fixtures)} partidos finalizados para el análisis de la liga.")
        return finished_fixtures

    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        logger.error(f"❌ Error obteniendo los partidos de la temporada para liga {league_id}: {e}")
        
        log_api_call(
            endpoint="fixtures (season)",
            params=params,
            response_time=response_time,
            status_code=getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0
        )
        return []

def get_all_leagues() -> List[Dict[str, Any]]:
    url = f"https://{API_HOST}/leagues"
    params = {"type": "league"}
    
    logger.info("🌍 Obteniendo la lista de todas las ligas del mundo (excluyendo copas)...")
    try:
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        leagues = response.json().get('response', [])
        logger.info(f"✅ Se encontraron {len(leagues)} ligas en total.")
        return leagues
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error obteniendo la lista de todas las ligas: {e}")
        return []