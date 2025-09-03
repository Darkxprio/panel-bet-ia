"""
Feature Engineering Avanzado para Panel Bet IA

Este módulo genera características avanzadas para mejorar la precisión
de las predicciones utilizando análisis estadístico profundo.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from constants import (
    MIN_MATCHES_REQUIRED,
    DEFAULT_GOALS_SCORED,
    DEFAULT_GOALS_CONCEDED
)

def calculate_team_performance_metrics(team_history: List[Dict[str, Any]], team_id: int) -> pd.Series:
    """
    Calcula métricas avanzadas de rendimiento de un equipo.
    
    Incluye métricas básicas, forma reciente, tendencias, y análisis contextual.
    """
    if not team_history:
        return pd.Series(dtype='float64')

    total_matches = len(team_history)
    
    # Métricas básicas
    wins = draws = losses = 0
    goals_scored = goals_conceded = 0
    btts_count = over_2_5_count = 0
    over_0_5_fh_count = over_1_5_fh_count = 0
    
    # Métricas avanzadas
    clean_sheets = 0
    failed_to_score = 0
    comebacks = 0  # Victorias después de ir perdiendo
    early_goals = 0  # Goles en primeros 15 min
    late_goals = 0   # Goles en últimos 15 min
    
    # Métricas por contexto
    home_wins = home_draws = home_losses = 0
    away_wins = away_draws = away_losses = 0
    home_goals_scored = home_goals_conceded = 0
    away_goals_scored = away_goals_conceded = 0
    
    # Análisis de forma reciente (últimos 5 partidos)
    recent_matches = min(5, total_matches)
    recent_wins = recent_draws = recent_losses = 0
    recent_goals_scored = recent_goals_conceded = 0
    
    # Análisis de tendencias
    goals_per_match_trend = []
    goals_conceded_trend = []
    
    for i, match in enumerate(team_history):
        is_home = match['teams']['home']['id'] == team_id
        
        # Obtener goles de manera segura
        home_goals = match['goals']['home'] if match['goals']['home'] is not None else 0
        away_goals = match['goals']['away'] if match['goals']['away'] is not None else 0
        
        team_goals = home_goals if is_home else away_goals
        opponent_goals = away_goals if is_home else home_goals
        
        goals_scored += team_goals
        goals_conceded += opponent_goals
        
        # Tendencias (para análisis de progresión)
        goals_per_match_trend.append(team_goals)
        goals_conceded_trend.append(opponent_goals)
        
        # Resultado del partido
        if team_goals > opponent_goals:
            wins += 1
            if is_home:
                home_wins += 1
            else:
                away_wins += 1
        elif team_goals == opponent_goals:
            draws += 1
            if is_home:
                home_draws += 1
            else:
                away_draws += 1
        else:
            losses += 1
            if is_home:
                home_losses += 1
            else:
                away_losses += 1
        
        # Métricas por contexto (local/visitante)
        if is_home:
            home_goals_scored += team_goals
            home_goals_conceded += opponent_goals
        else:
            away_goals_scored += team_goals
            away_goals_conceded += opponent_goals
        
        # Métricas especiales
        if opponent_goals == 0:
            clean_sheets += 1
        if team_goals == 0:
            failed_to_score += 1
        
        # BTTS y Over/Under
        if home_goals > 0 and away_goals > 0:
            btts_count += 1
        if (home_goals + away_goals) > 2.5:
            over_2_5_count += 1
        
        # Primera mitad
        home_goals_ht = match['score']['halftime']['home'] if match['score']['halftime']['home'] is not None else 0
        away_goals_ht = match['score']['halftime']['away'] if match['score']['halftime']['away'] is not None else 0
        first_half_total = home_goals_ht + away_goals_ht
        
        if first_half_total > 0.5:
            over_0_5_fh_count += 1
        if first_half_total > 1.5:
            over_1_5_fh_count += 1
        
        # Forma reciente (últimos 5 partidos)
        if i < recent_matches:
            recent_goals_scored += team_goals
            recent_goals_conceded += opponent_goals
            if team_goals > opponent_goals:
                recent_wins += 1
            elif team_goals == opponent_goals:
                recent_draws += 1
            else:
                recent_losses += 1
    
    # Calcular métricas derivadas
    home_matches = home_wins + home_draws + home_losses
    away_matches = away_wins + away_draws + away_losses
    
    # Análisis de tendencias
    goal_trend = _calculate_trend(goals_per_match_trend)
    defense_trend = _calculate_trend(goals_conceded_trend, inverse=True)
    
    # Métricas de eficiencia
    attack_efficiency = goals_scored / max(total_matches, 1)
    defense_efficiency = goals_conceded / max(total_matches, 1)
    goal_difference = goals_scored - goals_conceded
    
    # Compilar todas las métricas
    metrics = {
        # Métricas básicas
        'avg_goals_scored': attack_efficiency,
        'avg_goals_conceded': defense_efficiency,
        'goal_difference': goal_difference,
        'win_percentage': (wins / total_matches) * 100,
        'draw_percentage': (draws / total_matches) * 100,
        'loss_percentage': (losses / total_matches) * 100,
        
        # Métricas de mercados
        'btts_percentage': (btts_count / total_matches) * 100,
        'over_2_5_percentage': (over_2_5_count / total_matches) * 100,
        'over_0_5_fh_percentage': (over_0_5_fh_count / total_matches) * 100,
        'over_1_5_fh_percentage': (over_1_5_fh_count / total_matches) * 100,
        
        # Métricas defensivas/ofensivas
        'clean_sheet_percentage': (clean_sheets / total_matches) * 100,
        'failed_to_score_percentage': (failed_to_score / total_matches) * 100,
        
        # Forma reciente (últimos 5 partidos)
        'recent_avg_goals_scored': recent_goals_scored / max(recent_matches, 1),
        'recent_avg_goals_conceded': recent_goals_conceded / max(recent_matches, 1),
        'recent_win_percentage': (recent_wins / max(recent_matches, 1)) * 100,
        'recent_points_per_game': (recent_wins * 3 + recent_draws) / max(recent_matches, 1),
        
        # Rendimiento por contexto
        'home_win_percentage': (home_wins / max(home_matches, 1)) * 100 if home_matches > 0 else 50,
        'away_win_percentage': (away_wins / max(away_matches, 1)) * 100 if away_matches > 0 else 50,
        'home_avg_goals_scored': home_goals_scored / max(home_matches, 1) if home_matches > 0 else DEFAULT_GOALS_SCORED,
        'away_avg_goals_scored': away_goals_scored / max(away_matches, 1) if away_matches > 0 else DEFAULT_GOALS_SCORED,
        'home_avg_goals_conceded': home_goals_conceded / max(home_matches, 1) if home_matches > 0 else DEFAULT_GOALS_CONCEDED,
        'away_avg_goals_conceded': away_goals_conceded / max(away_matches, 1) if away_matches > 0 else DEFAULT_GOALS_CONCEDED,
        
        # Tendencias
        'goal_scoring_trend': goal_trend,
        'defensive_trend': defense_trend,
        'form_trend': (recent_wins * 3 + recent_draws) / max(recent_matches, 1) - (wins * 3 + draws) / max(total_matches, 1),
        
        # Métricas de consistencia
        'goals_scored_variance': np.var(goals_per_match_trend) if len(goals_per_match_trend) > 1 else 0,
        'goals_conceded_variance': np.var(goals_conceded_trend) if len(goals_conceded_trend) > 1 else 0,
    }
    
    return pd.Series(metrics)


def _calculate_trend(values: List[float], inverse: bool = False) -> float:
    """
    Calcula la tendencia de una serie de valores usando regresión lineal simple.
    
    Args:
        values: Lista de valores numéricos
        inverse: Si True, invierte el signo (útil para métricas defensivas)
        
    Returns:
        Pendiente de la tendencia (positiva = mejorando, negativa = empeorando)
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = np.arange(n)  # Índices de tiempo
    y = np.array(values)
    
    # Regresión lineal simple: y = mx + b
    # m = (n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    
    return -slope if inverse else slope


def _calculate_head_to_head_stats(home_history: List[Dict], away_history: List[Dict], 
                                 home_id: int, away_id: int) -> Dict[str, float]:
    """
    Calcula estadísticas de enfrentamientos directos entre dos equipos.
    
    Args:
        home_history: Historial del equipo local
        away_history: Historial del equipo visitante
        home_id: ID del equipo local
        away_id: ID del equipo visitante
        
    Returns:
        Diccionario con estadísticas de enfrentamientos directos
    """
    h2h_matches = []
    
    # Buscar enfrentamientos directos en el historial de ambos equipos
    for match in home_history + away_history:
        home_team_id = match['teams']['home']['id']
        away_team_id = match['teams']['away']['id']
        
        # Verificar si es un enfrentamiento directo
        if (home_team_id == home_id and away_team_id == away_id) or \
           (home_team_id == away_id and away_team_id == home_id):
            h2h_matches.append(match)
    
    # Eliminar duplicados (puede aparecer en ambos historiales)
    unique_h2h = []
    seen_fixture_ids = set()
    for match in h2h_matches:
        # Usar un identificador único basado en equipos y goles si no hay fixture id
        if 'fixture' in match and 'id' in match['fixture']:
            fixture_id = match['fixture']['id']
        else:
            # Crear ID único basado en equipos y resultado
            home_id = match['teams']['home']['id']
            away_id = match['teams']['away']['id']
            home_goals = match['goals']['home'] or 0
            away_goals = match['goals']['away'] or 0
            fixture_id = f"{home_id}_{away_id}_{home_goals}_{away_goals}"
        
        if fixture_id not in seen_fixture_ids:
            unique_h2h.append(match)
            seen_fixture_ids.add(fixture_id)
    
    if not unique_h2h:
        return {
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_avg_goals': 2.5,
            'h2h_btts_percentage': 50,
            'h2h_over_2_5_percentage': 50,
            'h2h_matches_count': 0
        }
    
    home_wins = draws = away_wins = 0
    total_goals = btts_count = over_2_5_count = 0
    
    for match in unique_h2h:
        home_goals = match['goals']['home'] or 0
        away_goals = match['goals']['away'] or 0
        
        # Determinar quién era local en este enfrentamiento histórico
        match_home_id = match['teams']['home']['id']
        
        if match_home_id == home_id:
            # El equipo local actual era local en este partido histórico
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                away_wins += 1
        else:
            # El equipo local actual era visitante en este partido histórico
            if away_goals > home_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                away_wins += 1
        
        total_goals += home_goals + away_goals
        if home_goals > 0 and away_goals > 0:
            btts_count += 1
        if (home_goals + away_goals) > 2.5:
            over_2_5_count += 1
    
    matches_count = len(unique_h2h)
    
    return {
        'h2h_home_wins': home_wins,
        'h2h_draws': draws,
        'h2h_away_wins': away_wins,
        'h2h_avg_goals': total_goals / matches_count,
        'h2h_btts_percentage': (btts_count / matches_count) * 100,
        'h2h_over_2_5_percentage': (over_2_5_count / matches_count) * 100,
        'h2h_matches_count': matches_count
    }


def create_feature_vector(match_data: Dict[str, Any], home_team_history: List[Dict[str, Any]], away_team_history: List[Dict[str, Any]]) -> pd.Series:
    """
    Crea un vector de características avanzado para la predicción.
    
    Incluye métricas de equipos individuales, enfrentamientos directos,
    y factores contextuales del partido.
    
    Args:
        match_data: Datos del partido a predecir
        home_team_history: Historial del equipo local
        away_team_history: Historial del equipo visitante
        
    Returns:
        Serie de pandas con todas las características
    """
    home_id = match_data['teams']['home']['id']
    away_id = match_data['teams']['away']['id']
    
    # 1. Métricas individuales de cada equipo
    home_metrics = calculate_team_performance_metrics(home_team_history, home_id)
    away_metrics = calculate_team_performance_metrics(away_team_history, away_id)
    
    # 2. Enfrentamientos directos (Head-to-Head)
    h2h_stats = _calculate_head_to_head_stats(home_team_history, away_team_history, home_id, away_id)
    
    # 3. Métricas comparativas entre equipos
    comparative_metrics = _calculate_comparative_metrics(home_metrics, away_metrics)
    
    # 4. Factores contextuales del partido
    contextual_factors = _calculate_contextual_factors(match_data)
    
    # 5. Combinar todas las características
    # Renombrar métricas de equipos
    home_metrics = home_metrics.add_prefix('home_')
    away_metrics = away_metrics.add_prefix('away_')
    
    # Crear vector final
    all_features = [
        home_metrics,
        away_metrics,
        pd.Series(h2h_stats),
        pd.Series(comparative_metrics),
        pd.Series(contextual_factors)
    ]
    
    feature_vector = pd.concat(all_features)
    
    # 6. Verificar que no hay valores NaN
    feature_vector = feature_vector.fillna(0)
    
    return feature_vector


def _calculate_comparative_metrics(home_metrics: pd.Series, away_metrics: pd.Series) -> Dict[str, float]:
    """
    Calcula métricas comparativas entre los dos equipos.
    
    Args:
        home_metrics: Métricas del equipo local
        away_metrics: Métricas del equipo visitante
        
    Returns:
        Diccionario con métricas comparativas
    """
    def safe_ratio(a, b, default=1.0):
        """Calcula ratio de forma segura evitando división por cero."""
        if b == 0:
            return default
        return a / b
    
    # Ratios de ataque vs defensa
    attack_vs_defense_home = safe_ratio(
        home_metrics.get('avg_goals_scored', DEFAULT_GOALS_SCORED),
        away_metrics.get('avg_goals_conceded', DEFAULT_GOALS_CONCEDED)
    )
    
    attack_vs_defense_away = safe_ratio(
        away_metrics.get('avg_goals_scored', DEFAULT_GOALS_SCORED),
        home_metrics.get('avg_goals_conceded', DEFAULT_GOALS_CONCEDED)
    )
    
    # Diferencias en forma
    form_difference = (
        home_metrics.get('recent_points_per_game', 1.5) - 
        away_metrics.get('recent_points_per_game', 1.5)
    )
    
    # Diferencias en tendencias
    goal_trend_diff = (
        home_metrics.get('goal_scoring_trend', 0) - 
        away_metrics.get('goal_scoring_trend', 0)
    )
    
    defense_trend_diff = (
        home_metrics.get('defensive_trend', 0) - 
        away_metrics.get('defensive_trend', 0)
    )
    
    # Ventaja del equipo local
    home_advantage = (
        home_metrics.get('home_avg_goals_scored', DEFAULT_GOALS_SCORED) - 
        away_metrics.get('away_avg_goals_scored', DEFAULT_GOALS_SCORED)
    )
    
    # Diferencia de calidad general
    quality_difference = (
        home_metrics.get('goal_difference', 0) - 
        away_metrics.get('goal_difference', 0)
    )
    
    # Consistencia comparativa
    consistency_difference = (
        away_metrics.get('goals_scored_variance', 0) - 
        home_metrics.get('goals_scored_variance', 0)
    )  # Menor varianza = más consistente
    
    return {
        'attack_vs_defense_ratio_home': attack_vs_defense_home,
        'attack_vs_defense_ratio_away': attack_vs_defense_away,
        'form_difference': form_difference,
        'goal_trend_difference': goal_trend_diff,
        'defense_trend_difference': defense_trend_diff,
        'home_advantage': home_advantage,
        'quality_difference': quality_difference,
        'consistency_advantage': consistency_difference,
        'expected_goals_home': attack_vs_defense_home * DEFAULT_GOALS_SCORED,
        'expected_goals_away': attack_vs_defense_away * DEFAULT_GOALS_SCORED,
    }


def _calculate_contextual_factors(match_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcula factores contextuales del partido.
    
    Args:
        match_data: Datos del partido
        
    Returns:
        Diccionario con factores contextuales
    """
    contextual = {}
    
    # Información básica del partido
    try:
        # Día de la semana (0=lunes, 6=domingo)
        match_date = datetime.fromisoformat(match_data['fixture']['date'].replace('Z', '+00:00'))
        contextual['day_of_week'] = match_date.weekday()
        contextual['is_weekend'] = 1.0 if match_date.weekday() >= 5 else 0.0
        
        # Hora del partido (puede influir en el rendimiento)
        hour = match_date.hour
        contextual['match_hour'] = hour
        contextual['is_evening_match'] = 1.0 if 18 <= hour <= 21 else 0.0
        
        # Mes del año (temporada puede influir)
        contextual['month'] = match_date.month
        contextual['is_winter'] = 1.0 if match_date.month in [12, 1, 2] else 0.0
        
    except (KeyError, ValueError, AttributeError):
        # Si no se puede parsear la fecha, usar valores por defecto
        contextual.update({
            'day_of_week': 3,  # Miércoles por defecto
            'is_weekend': 0.0,
            'match_hour': 20,  # 8 PM por defecto
            'is_evening_match': 1.0,
            'month': 6,  # Junio por defecto
            'is_winter': 0.0
        })
    
    # Información de la liga
    try:
        league_id = match_data['league']['id']
        # Factores específicos por liga (se pueden expandir)
        major_leagues = [39, 140, 78, 135, 61]  # Premier, La Liga, Bundesliga, Serie A, Ligue 1
        contextual['is_major_league'] = 1.0 if league_id in major_leagues else 0.0
        contextual['league_id_encoded'] = league_id % 100  # Encoding simple
        
    except KeyError:
        contextual.update({
            'is_major_league': 0.0,
            'league_id_encoded': 50
        })
    
    # Factores derivados
    contextual['competitive_factor'] = 1.0  # Se puede expandir con datos de competitividad
    
    return contextual