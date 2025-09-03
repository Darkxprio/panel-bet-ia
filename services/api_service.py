import requests
from config import API_HOST, API_HEADERS
from typing import List, Dict, Any
import time
from constants import PRIMARY_BOOKMAKER_ID, SECONDARY_BOOKMAKER_IDS, TEAM_HISTORY_MATCHES

def get_daily_fixtures(date_str: str) -> list:
    url = f"https://{API_HOST}/fixtures"
    params = {"date": date_str}
    try:
        print(f"Buscando partidos para la fecha: {date_str}...")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        fixtures = response.json().get('response', [])
        print(f"Se encontraron {len(fixtures)} partidos.")
        return fixtures
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener los partidos del día: {e}")
        return []

def get_league_details(league_id: int) -> dict:
    url = f"https://{API_HOST}/leagues"
    params = {"id": str(league_id)}
    try:
        print(f"  -> Obteniendo detalles para la liga ID: {league_id}...")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        
        league_data = response.json().get('response', [])
        if league_data:
            return league_data[0]
        return {}
        
    except requests.exceptions.RequestException as e:
        print(f"  -> Error al obtener detalles de la liga {league_id}: {e}")
        return {}

def get_team_last_matches(team_id: int, season: int, last_n: int = TEAM_HISTORY_MATCHES) -> list:
    url = f"https://{API_HOST}/fixtures"
    params = {
        "team": str(team_id),
        "season": str(season),
        "last": str(last_n)
    }
    try:
        print(f"  -> Obteniendo los últimos {last_n} partidos para el equipo ID: {team_id}...")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.5) 
        return response.json().get('response', [])
    except requests.exceptions.RequestException as e:
        print(f"  -> Error al obtener el historial del equipo {team_id}: {e}")
        return []

def _get_odds_from_single_bookmaker(fixture_id: int, bookmaker_id: int) -> List[Dict[str, Any]]:
    url = f"https://{API_HOST}/odds"
    params = {"fixture": str(fixture_id), "bookmaker": str(bookmaker_id)}
    
    try:
        print(f"  - Getting odds from bookmaker ID: {bookmaker_id}...")
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.3)
        odds_response = response.json().get('response', [])
        if odds_response:
            return odds_response[0]['bookmakers'][0]['bets']
        return []
    except (requests.exceptions.RequestException, IndexError) as e:
        print(f"  - Could not get odds for bookmaker {bookmaker_id}: {e}")
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
                print(f"  -> Found better odd: {new_odd} from bookmaker {bookmaker_id} (previous was {best_odd})")
                best_odd = new_odd
                
        except (StopIteration, KeyError):
            continue
            
    return {'best_odd': best_odd}