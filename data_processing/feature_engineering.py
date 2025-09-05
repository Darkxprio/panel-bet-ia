import pandas as pd
import numpy as np
from typing import List, Dict, Any

def calculate_team_performance_metrics(team_history: List[Dict[str, Any]], team_id: int) -> pd.Series:
    if not team_history:
        return pd.Series(dtype='float64')

    total_matches = len(team_history)
    
    wins = draws = losses = 0
    goals_scored = goals_conceded = 0
    btts_count = over_2_5_count = 0
    over_0_5_fh_count = over_1_5_fh_count = 0
    clean_sheets = 0
    failed_to_score = 0
    
    home_wins = home_draws = home_losses = 0
    away_wins = away_draws = away_losses = 0
    home_goals_scored = home_goals_conceded = 0
    away_goals_scored = away_goals_conceded = 0
    
    recent_matches_count = min(5, total_matches)
    recent_wins = recent_draws = 0
    recent_goals_scored = recent_goals_conceded = 0
    
    goals_per_match_trend = []
    goals_conceded_trend = []
    
    for i, match in enumerate(team_history):
        is_home = match['teams']['home']['id'] == team_id
        
        home_goals = match['goals']['home'] if match['goals']['home'] is not None else 0
        away_goals = match['goals']['away'] if match['goals']['away'] is not None else 0
        
        team_goals = home_goals if is_home else away_goals
        opponent_goals = away_goals if is_home else home_goals
        
        goals_scored += team_goals
        goals_conceded += opponent_goals
        
        goals_per_match_trend.append(team_goals)
        goals_conceded_trend.append(opponent_goals)
        
        if team_goals > opponent_goals:
            wins += 1
            if is_home: home_wins += 1
            else: away_wins += 1
        elif team_goals == opponent_goals:
            draws += 1
            if is_home: home_draws += 1
            else: away_draws += 1
        else:
            losses += 1
            if is_home: home_losses += 1
            else: away_losses += 1
            
        if is_home:
            home_goals_scored += team_goals
            home_goals_conceded += opponent_goals
        else:
            away_goals_scored += team_goals
            away_goals_conceded += opponent_goals
            
        if opponent_goals == 0: clean_sheets += 1
        if team_goals == 0: failed_to_score += 1
        if home_goals > 0 and away_goals > 0: btts_count += 1
        if (home_goals + away_goals) > 2.5: over_2_5_count += 1
        
        home_goals_ht = match['score']['halftime']['home'] if match['score']['halftime']['home'] is not None else 0
        away_goals_ht = match['score']['halftime']['away'] if match['score']['halftime']['away'] is not None else 0
        first_half_total = home_goals_ht + away_goals_ht
        
        if first_half_total > 0.5: over_0_5_fh_count += 1
        if first_half_total > 1.5: over_1_5_fh_count += 1
        
        if i < recent_matches_count:
            recent_goals_scored += team_goals
            recent_goals_conceded += opponent_goals
            if team_goals > opponent_goals: recent_wins += 1
            elif team_goals == opponent_goals: recent_draws += 1

    home_matches = home_wins + home_draws + home_losses
    away_matches = away_wins + away_draws + away_losses
    
    goal_trend = _calculate_trend(goals_per_match_trend)
    defense_trend = _calculate_trend(goals_conceded_trend, inverse=True)
    
    metrics = {
        'avg_goals_scored': goals_scored / total_matches,
        'avg_goals_conceded': goals_conceded / total_matches,
        'goal_difference': goals_scored - goals_conceded,
        'win_percentage': (wins / total_matches) * 100,
        'draw_percentage': (draws / total_matches) * 100,
        'loss_percentage': (losses / total_matches) * 100,
        'btts_percentage': (btts_count / total_matches) * 100,
        'over_2_5_percentage': (over_2_5_count / total_matches) * 100,
        'over_0_5_fh_percentage': (over_0_5_fh_count / total_matches) * 100,
        'over_1_5_fh_percentage': (over_1_5_fh_count / total_matches) * 100,
        'clean_sheet_percentage': (clean_sheets / total_matches) * 100,
        'failed_to_score_percentage': (failed_to_score / total_matches) * 100,
        'recent_avg_goals_scored': recent_goals_scored / max(recent_matches_count, 1),
        'recent_avg_goals_conceded': recent_goals_conceded / max(recent_matches_count, 1),
        'recent_points_per_game': (recent_wins * 3 + recent_draws) / max(recent_matches_count, 1),
        'home_avg_goals_scored': home_goals_scored / home_matches if home_matches > 0 else 0,
        'away_avg_goals_scored': away_goals_scored / away_matches if away_matches > 0 else 0,
        'home_avg_goals_conceded': home_goals_conceded / home_matches if home_matches > 0 else 0,
        'away_avg_goals_conceded': away_goals_conceded / away_matches if away_matches > 0 else 0,
        'goal_scoring_trend': goal_trend,
        'defensive_trend': defense_trend,
        'goals_scored_variance': np.var(goals_per_match_trend) if len(goals_per_match_trend) > 1 else 0,
        'goals_conceded_variance': np.var(goals_conceded_trend) if len(goals_conceded_trend) > 1 else 0,
    }
    
    return pd.Series(metrics)

def _calculate_trend(values: List[float], inverse: bool = False) -> float:
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = np.arange(n)
    y = np.array(values)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return -slope if inverse else slope

def _calculate_head_to_head_stats(home_history: List[Dict], away_history: List[Dict], home_id: int, away_id: int) -> Dict[str, float]:
    h2h_matches = []
    
    for match in home_history + away_history:
        if (match['teams']['home']['id'] == home_id and match['teams']['away']['id'] == away_id) or \
           (match['teams']['home']['id'] == away_id and match['teams']['away']['id'] == home_id):
            h2h_matches.append(match)
    
    unique_h2h = list({m['fixture']['id']: m for m in h2h_matches if m.get('fixture', {}).get('id')}.values())
    
    if not unique_h2h:
        return {
            'h2h_matches_count': 0,
            'h2h_btts_percentage': 50.0,
            'h2h_over_2_5_percentage': 50.0
        }
    
    btts_count = over_2_5_count = 0
    
    for match in unique_h2h:
        home_goals = match['goals']['home'] or 0
        away_goals = match['goals']['away'] or 0
        if home_goals > 0 and away_goals > 0: btts_count += 1
        if (home_goals + away_goals) > 2.5: over_2_5_count += 1
            
    matches_count = len(unique_h2h)
    
    return {
        'h2h_matches_count': matches_count,
        'h2h_btts_percentage': (btts_count / matches_count) * 100,
        'h2h_over_2_5_percentage': (over_2_5_count / matches_count) * 100
    }

def _calculate_comparative_metrics(home_metrics: pd.Series, away_metrics: pd.Series) -> Dict[str, float]:
    home_attack = home_metrics.get('avg_goals_scored', 1.2)
    away_defense = away_metrics.get('avg_goals_conceded', 1.2)
    away_attack = away_metrics.get('avg_goals_scored', 1.2)
    home_defense = away_metrics.get('avg_goals_conceded', 1.2)

    return {
        'attack_vs_defense_home': home_attack / away_defense if away_defense > 0 else 1.0,
        'attack_vs_defense_away': away_attack / home_defense if home_defense > 0 else 1.0,
        'form_difference': home_metrics.get('recent_points_per_game', 1.5) - away_metrics.get('recent_points_per_game', 1.5)
    }

def _calculate_contextual_factors(
    match_data: Dict, 
    league_details: Dict,
    home_history: List,
    away_history: List
) -> Dict:
    contextual = {}
    
    # --- 1. Features de Tiempo (Día, Hora, Mes) ---
    try:
        match_date = datetime.fromisoformat(match_data['fixture']['date'].replace('Z', '+00:00'))
        contextual['day_of_week'] = match_date.weekday()
        contextual['is_weekend'] = 1.0 if match_date.weekday() >= 5 else 0.0
        contextual['month'] = match_date.month
    except (KeyError, ValueError):
        contextual['day_of_week'] = 3
        contextual['is_weekend'] = 0.0
        contextual['month'] = 6

    # --- 2. Features de Competición (Tipo y Progreso) ---
    try:
        # competition_type: 1 para Liga, 2 para Copa
        contextual['competition_type'] = 1.0 if league_details['league']['type'] == 'League' else 2.0
        
        # season_progress: 0.0 (inicio) a 1.0 (final)
        season_info = next((s for s in league_details.get('seasons', []) if s.get('current')), None)
        start_date = datetime.strptime(season_info['start'], '%Y-%m-%d').date()
        end_date = datetime.strptime(season_info['end'], '%Y-%m-%d').date()
        today = datetime.now().date()
        
        total_duration = (end_date - start_date).days
        elapsed_duration = (today - start_date).days
        contextual['season_progress'] = elapsed_duration / total_duration if total_duration > 0 else 0.5
    except (KeyError, TypeError, StopIteration):
        contextual['competition_type'] = 1.0
        contextual['season_progress'] = 0.5

    # --- 3. Features de Fatiga y Descanso ---
    def get_days_since_last_match(history: List, current_match_date_str: str) -> float:
        if not history:
            return 7.0 # Valor por defecto si no hay historial
        last_match_date = datetime.fromisoformat(history[0]['fixture']['date'].replace('Z', '+00:00'))
        current_match_date = datetime.fromisoformat(current_match_date_str.replace('Z', '+00:00'))
        return (current_match_date - last_match_date).days

    home_last_match_days = get_days_since_last_match(home_history, match_data['fixture']['date'])
    away_last_match_days = get_days_since_last_match(away_history, match_data['fixture']['date'])
    
    contextual['home_days_since_last_match'] = home_last_match_days
    contextual['away_days_since_last_match'] = away_last_match_days
    # rest_day_advantage: positivo = ventaja local, negativo = ventaja visitante
    contextual['rest_day_advantage'] = home_last_match_days - away_last_match_days

    return contextual

def calculate_league_strengths(season_fixtures: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not season_fixtures:
        return {}

    total_home_goals = sum(f['goals']['home'] for f in season_fixtures if f['goals']['home'] is not None)
    total_away_goals = sum(f['goals']['away'] for f in season_fixtures if f['goals']['away'] is not None)
    num_matches = len(season_fixtures)
    
    if num_matches == 0:
        return {}

    avg_home_goals_league = total_home_goals / num_matches
    avg_away_goals_league = total_away_goals / num_matches

    team_stats = {}
    for match in season_fixtures:
        home_id = match['teams']['home']['id']
        away_id = match['teams']['away']['id']
        
        team_stats.setdefault(home_id, {'scored_home': 0, 'conceded_home': 0, 'games_home': 0})
        team_stats.setdefault(away_id, {'scored_away': 0, 'conceded_away': 0, 'games_away': 0})

        team_stats[home_id]['scored_home'] += match['goals']['home']
        team_stats[home_id]['conceded_home'] += match['goals']['away']
        team_stats[home_id]['games_home'] += 1
        team_stats[away_id]['scored_away'] += match['goals']['away']
        team_stats[away_id]['conceded_away'] += match['goals']['home']
        team_stats[away_id]['games_away'] += 1

    team_strengths = {}
    for team_id, stats in team_stats.items():
        avg_scored_home = stats['scored_home'] / stats['games_home'] if stats['games_home'] > 0 else 0
        avg_conceded_home = stats['conceded_home'] / stats['games_home'] if stats['games_home'] > 0 else 0
        avg_scored_away = stats['scored_away'] / stats['games_away'] if stats['games_away'] > 0 else 0
        avg_conceded_away = stats['conceded_away'] / stats['games_away'] if stats['games_away'] > 0 else 0
        
        team_strengths[team_id] = {
            'attack_home': avg_scored_home / avg_home_goals_league if avg_home_goals_league > 0 else 1.0,
            'defense_home': avg_conceded_home / avg_away_goals_league if avg_away_goals_league > 0 else 1.0,
            'attack_away': avg_scored_away / avg_away_goals_league if avg_away_goals_league > 0 else 1.0,
            'defense_away': avg_conceded_away / avg_home_goals_league if avg_home_goals_league > 0 else 1.0
        }
        
    league_averages = {
        'avg_home_goals': avg_home_goals_league,
        'avg_away_goals': avg_away_goals_league
    }

    return {'teams': team_strengths, 'league_averages': league_averages}

def create_feature_vector(
    match_data: Dict[str, Any], 
    home_team_history: List[Dict[str, Any]], 
    away_team_history: List[Dict[str, Any]],
    league_strengths: Dict[str, Any],
    league_details: Dict[str, Any]
) -> pd.Series:
    home_id = match_data['teams']['home']['id']
    away_id = match_data['teams']['away']['id']
    
    home_metrics = calculate_team_performance_metrics(home_team_history, home_id)
    away_metrics = calculate_team_performance_metrics(away_team_history, away_id)
    h2h_stats = _calculate_head_to_head_stats(home_team_history, away_team_history, home_id, away_id)
    comparative_metrics = _calculate_comparative_metrics(home_metrics, away_metrics)
    contextual_factors = _calculate_contextual_factors(match_data, league_details, home_history, away_history)
    
    team_strengths = league_strengths.get('teams', {})
    league_averages = league_strengths.get('league_averages', {})
    home_str = team_strengths.get(home_id, {})
    away_str = team_strengths.get(away_id, {})
    
    home_exp_goals = (home_str.get('attack_home', 1.0) * away_str.get('defense_away', 1.0) * league_averages.get('avg_home_goals', 1.5))
    away_exp_goals = (away_str.get('attack_away', 1.0) * home_str.get('defense_home', 1.0) * league_averages.get('avg_away_goals', 1.2))

    dixon_coles_features = {
        'home_expected_goals': home_exp_goals,
        'away_expected_goals': away_exp_goals,
        'home_attack_strength': home_str.get('attack_home', 1.0),
        'home_defense_strength': home_str.get('defense_home', 1.0),
        'away_attack_strength': away_str.get('attack_away', 1.0),
        'away_defense_strength': away_str.get('defense_away', 1.0)
    }

    home_metrics = home_metrics.add_prefix('home_')
    away_metrics = away_metrics.add_prefix('away_')
    
    all_features = pd.concat([
        home_metrics,
        away_metrics,
        pd.Series(h2h_stats),
        pd.Series(comparative_metrics),
        pd.Series(contextual_factors),
        pd.Series(dixon_coles_features)
    ])
    
    return all_features.fillna(0)