"""
Funciones auxiliares para análisis de apuestas y cálculo de valor.
"""

from typing import Dict, Any
from constants import (
    VALUE_BET_MARGIN, 
    KELLY_HIGH_THRESHOLD, 
    KELLY_MEDIUM_THRESHOLD,
    MSG_VALUE_FOUND
)
from utils import calculate_implied_probability, format_percentage


def find_value_bet(calculated_prob: float, market_odds: float, margin: float = VALUE_BET_MARGIN) -> bool:
    """
    Determina si una apuesta tiene valor positivo.
    
    Args:
        calculated_prob: Probabilidad calculada por nuestro modelo (0-1)
        market_odds: Cuota del mercado
        margin: Margen mínimo requerido para considerar valor
        
    Returns:
        True si la apuesta tiene valor, False en caso contrario
    """
    if market_odds <= 1 or calculated_prob <= 0:
        return False
    
    implied_prob = calculate_implied_probability(market_odds)
    edge = (calculated_prob / implied_prob) - 1
    
    return edge > margin


def calculate_confidence(probability: float, odds: float) -> str:
    """
    Calcula el nivel de confianza basado en el criterio de Kelly.
    
    Args:
        probability: Probabilidad calculada (0-1)
        odds: Cuota decimal
        
    Returns:
        Nivel de confianza: 'Alta', 'Media' o 'Baja'
    """
    if odds <= 1 or probability <= 0:
        return 'Baja'

    edge = (probability * odds) - 1
    kelly_percentage = edge / (odds - 1)

    if kelly_percentage > KELLY_HIGH_THRESHOLD:
        return 'Alta'
    elif kelly_percentage > KELLY_MEDIUM_THRESHOLD:
        return 'Media'
    else:
        return 'Baja'


def calculate_edge(probability: float, odds: float) -> float:
    """
    Calcula el edge (ventaja) de una apuesta.
    
    Args:
        probability: Probabilidad calculada (0-1)
        odds: Cuota decimal
        
    Returns:
        Edge como decimal (ej: 0.15 = 15% de ventaja)
    """
    if odds <= 1 or probability <= 0:
        return 0.0
    
    return (probability * odds) - 1

def create_prediction_entry(match: Dict, features: Any, market: str, prediction: str, odds: float, probability: float) -> Dict[str, Any]:
    """
    Crea una entrada de predicción con todos los datos necesarios.
    
    Args:
        match: Datos del partido
        features: Características calculadas
        market: Tipo de mercado
        prediction: Predicción específica
        odds: Cuota encontrada
        probability: Probabilidad calculada
        
    Returns:
        Diccionario con todos los datos de la predicción
    """
    edge = calculate_edge(probability, odds)
    confidence_level = calculate_confidence(probability, odds)
    implied_prob = calculate_implied_probability(odds)

    reasoning = (
        f"Valor detectado. Prob: {format_percentage(probability)}, "
        f"Cuota: {odds}, Implícita: {format_percentage(implied_prob)}, "
        f"Edge: {format_percentage(edge)}."
    )
    
    print(MSG_VALUE_FOUND.format(market, prediction, confidence_level, reasoning))

    return {
        'fixture_id': match['fixture']['id'],
        'matchTimestampUTC': match['fixture']['date'],
        'league': match['league']['name'],
        'country': match['league']['country'],
        'teamA': match['teams']['home']['name'],
        'teamA_logo_url': match['teams']['home']['logo'],
        'teamB': match['teams']['away']['name'],
        'teamB_logo_url': match['teams']['away']['logo'],
        'market': market,
        'prediction': prediction,
        'odds': odds,
        'calculated_probability': probability,
        'value_edge': edge,
        'confidence': confidence_level,
        'reasoning': reasoning
    }