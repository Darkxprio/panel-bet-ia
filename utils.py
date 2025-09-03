"""
Utilidades comunes para Panel Bet IA

Funciones reutilizables que se usan en múltiples partes del proyecto.
"""

import math
from typing import List, Dict, Any
from constants import MAX_GOALS_POISSON


def calculate_poisson_probabilities(expected_goals: float, max_goals: int = MAX_GOALS_POISSON) -> List[float]:
    """
    Calcula la probabilidad de que ocurran 0, 1, 2, ... max_goals
    usando la distribución de Poisson.
    
    Args:
        expected_goals: Valor lambda (goles esperados)
        max_goals: Máximo número de goles a considerar
    
    Returns:
        Lista de probabilidades para 0, 1, 2, ... max_goals
    """
    if expected_goals <= 0:
        raise ValueError("expected_goals debe ser mayor que 0")
    
    probabilities = []
    # Precalcular e^(-λ) para optimización
    exp_neg_lambda = math.exp(-expected_goals)
    
    for k in range(max_goals + 1):
        # Fórmula de Poisson: P(k; λ) = (λ^k * e^-λ) / k!
        prob = (math.pow(expected_goals, k) * exp_neg_lambda) / math.factorial(k)
        probabilities.append(prob)
    
    return probabilities


def validate_match_data(match: Dict[str, Any]) -> bool:
    """
    Valida que los datos del partido tengan la estructura esperada.
    
    Args:
        match: Diccionario con datos del partido
        
    Returns:
        True si los datos son válidos, False en caso contrario
    """
    required_fields = [
        'fixture', 'teams', 'league'
    ]
    
    if not all(field in match for field in required_fields):
        return False
    
    # Validar estructura de teams
    if not all(team in match['teams'] for team in ['home', 'away']):
        return False
    
    # Validar que tengan IDs y nombres
    for team_type in ['home', 'away']:
        team = match['teams'][team_type]
        if not all(field in team for field in ['id', 'name']):
            return False
    
    # Validar fixture
    if not all(field in match['fixture'] for field in ['id', 'date']):
        return False
    
    # Validar league
    if not all(field in match['league'] for field in ['id', 'name']):
        return False
    
    return True


def validate_odds_data(odds_data: List[Dict]) -> bool:
    """
    Valida que los datos de cuotas tengan la estructura esperada.
    
    Args:
        odds_data: Lista de mercados con cuotas
        
    Returns:
        True si los datos son válidos, False en caso contrario
    """
    if not odds_data or not isinstance(odds_data, list):
        return False
    
    for market in odds_data:
        if not isinstance(market, dict):
            return False
        
        if 'id' not in market or 'values' not in market:
            return False
        
        if not isinstance(market['values'], list):
            return False
        
        for value in market['values']:
            if not isinstance(value, dict):
                return False
            if not all(field in value for field in ['value', 'odd']):
                return False
    
    return True


def safe_get_feature(features, key: str, default_value: float) -> float:
    """
    Obtiene un valor de features de manera segura con valor por defecto.
    
    Args:
        features: Series de pandas con las características
        key: Clave a buscar
        default_value: Valor por defecto si no existe
        
    Returns:
        Valor de la característica o valor por defecto
    """
    try:
        return float(features.get(key, default_value))
    except (ValueError, TypeError):
        return default_value


def format_team_names(match: Dict[str, Any]) -> str:
    """
    Formatea los nombres de los equipos para mostrar.
    
    Args:
        match: Diccionario con datos del partido
        
    Returns:
        String formateado con los nombres de los equipos
    """
    try:
        home_name = match['teams']['home']['name']
        away_name = match['teams']['away']['name']
        return f"{home_name} vs {away_name}"
    except KeyError:
        return "Equipos desconocidos"


def calculate_implied_probability(odds: float) -> float:
    """
    Calcula la probabilidad implícita de una cuota.
    
    Args:
        odds: Cuota decimal
        
    Returns:
        Probabilidad implícita (0-1)
    """
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def format_percentage(value: float) -> str:
    """
    Formatea un valor decimal como porcentaje.
    
    Args:
        value: Valor decimal (0-1)
        
    Returns:
        String formateado como porcentaje
    """
    return f"{value:.2%}"
