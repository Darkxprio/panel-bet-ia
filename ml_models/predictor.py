"""
Predictor principal para Panel Bet IA

Sistema de predicción ensemble que combina múltiples enfoques
para maximizar la precisión de las predicciones.
"""

import pandas as pd
from typing import Dict, Optional
from constants import (
    MAX_GOALS_POISSON,
    DEFAULT_GOALS_SCORED,
    DEFAULT_GOALS_CONCEDED,
    DEFAULT_BTTS_PERCENTAGE,
    DEFAULT_OVER_2_5_PERCENTAGE,
    DEFAULT_OVER_0_5_FH_PERCENTAGE,
    DEFAULT_OVER_1_5_FH_PERCENTAGE
)
from utils import calculate_poisson_probabilities, safe_get_feature, get_logger

# Logger para modelos ML
logger = get_logger("ml_models")

# Importar el predictor avanzado
try:
    from .advanced_predictor import AdvancedPredictor
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    AdvancedPredictor = None


class Predictor:
    """
    Predictor ensemble que combina métodos estadísticos tradicionales
    con algoritmos avanzados de machine learning.
    """
    
    def __init__(self, model_path: str = None, use_ensemble: bool = True):
        """
        Inicializa el predictor ensemble.
        
        Args:
            model_path: Ruta al modelo entrenado (para uso futuro)
            use_ensemble: Si usar ensemble o solo método estadístico
        """
        self.model = None
        self.model_path = model_path
        self.use_ensemble = use_ensemble and ADVANCED_AVAILABLE
        
        # Inicializar predictor avanzado si está disponible
        if self.use_ensemble:
            try:
                self.advanced_predictor = AdvancedPredictor()
                logger.info("🧠 Ensemble Predictor initialized with Advanced ML + Statistical models.")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize advanced predictor: {e}")
                self.use_ensemble = False
                self.advanced_predictor = None
                logger.info("🤖 Statistical Predictor initialized (fallback mode).")
        else:
            self.advanced_predictor = None
            logger.info("🤖 Statistical Predictor initialized.")

    def predict_winner_probabilities(self, features: pd.Series) -> Dict[str, float]:
        """
        Predice probabilidades 1X2 usando ensemble de modelos.
        
        Combina predicciones estadísticas tradicionales con ML avanzado
        para maximizar la precisión.
        
        Args:
            features: Serie con las características del partido
            
        Returns:
            Diccionario con probabilidades de 'home', 'draw', 'away'
        """
        if self.use_ensemble and self.advanced_predictor:
            # Obtener predicciones de ambos modelos
            statistical_pred = self._statistical_winner_prediction(features)
            advanced_pred = self.advanced_predictor.predict_winner_probabilities(features)
            
            # Combinar predicciones con pesos
            ensemble_pred = self._combine_predictions(
                statistical_pred, advanced_pred, 
                statistical_weight=0.3, advanced_weight=0.7
            )
            
            return ensemble_pred
        else:
            # Solo predicción estadística
            return self._statistical_winner_prediction(features)
    
def _statistical_winner_prediction(self, match: Dict, features: pd.Series, league_strengths: Dict) -> Dict[str, float]:
    """
    Predicción estadística HÍBRIDA.
    - Usa Dixon-Coles si hay suficientes datos de la liga.
    - Usa un modelo de forma/historial si la temporada es muy joven.
    """
    home_id = match['teams']['home']['id']
    away_id = match['teams']['away']['id']
    
    # --- MODO 1: TEMPORADA MADURA (USA DIXON-COLES) ---
    if league_strengths and league_strengths.get('teams'):
        logger.debug(f"Usando modelo Dixon-Coles para {match['fixture']['id']}")
        
        team_str = league_strengths['teams']
        league_avg = league_strengths['league_averages']
        home_team_str = team_str.get(home_id)
        away_team_str = team_str.get(away_id)

        if not all([home_team_str, away_team_str, league_avg]):
            logger.warning(f"Faltan datos de fuerza, usando fallback para {match['fixture']['id']}")
            # Si falla, pasa al Modo 2
        else:
            home_exp_goals = home_team_str['attack_home'] * away_team_str['defense_away'] * league_avg['avg_home_goals']
            away_exp_goals = away_team_str['attack_away'] * home_team_str['defense_home'] * league_avg['avg_away_goals']
            
            home_poisson_probs = calculate_poisson_probabilities(home_exp_goals, MAX_GOALS_POISSON)
            away_poisson_probs = calculate_poisson_probabilities(away_exp_goals, MAX_GOALS_POISSON)

            prob_home_win, prob_draw, prob_away_win = 0.0, 0.0, 0.0
            
            for hg in range(MAX_GOALS_POISSON + 1):
                for ag in range(MAX_GOALS_POISSON + 1):
                    score_prob = home_poisson_probs[hg] * away_poisson_probs[ag]
                    if hg > ag: prob_home_win += score_prob
                    elif hg == ag: prob_draw += score_prob
                    else: prob_away_win += score_prob
            
            total_prob = prob_home_win + prob_draw + prob_away_win
            if total_prob > 0:
                return {
                    'home': prob_home_win / total_prob,
                    'draw': prob_draw / total_prob,
                    'away': prob_away_win / total_prob
                }

    # --- MODO 2: INICIO DE TEMPORADA (FALLBACK A MODELO DE FORMA) ---
    logger.debug(f"Usando modelo de FORMA RECIENTE para {match['fixture']['id']} (inicio de temporada)")

    # Usamos las features de forma reciente que ya calculaste
    home_attack = safe_get_feature(features, 'home_recent_avg_goals_scored', 1.2)
    away_defense = safe_get_feature(features, 'away_recent_avg_goals_conceded', 1.2)
    away_attack = safe_get_feature(features, 'away_recent_avg_goals_scored', 1.0)
    home_defense = safe_get_feature(features, 'home_recent_avg_goals_conceded', 1.2)

    home_exp_goals = (home_attack + away_defense) / 2
    away_exp_goals = (away_attack + home_defense) / 2
    
    # Reutilizamos el cálculo de Poisson con estos goles esperados más simples
    home_poisson_probs = calculate_poisson_probabilities(home_exp_goals, MAX_GOALS_POISSON)
    away_poisson_probs = calculate_poisson_probabilities(away_exp_goals, MAX_GOALS_POISSON)
    
    prob_home_win, prob_draw, prob_away_win = 0.0, 0.0, 0.0
    
    for hg in range(MAX_GOALS_POISSON + 1):
        for ag in range(MAX_GOALS_POISSON + 1):
            score_prob = home_poisson_probs[hg] * away_poisson_probs[ag]
            if hg > ag: prob_home_win += score_prob
            elif hg == ag: prob_draw += score_prob
            else: prob_away_win += score_prob

    total_prob = prob_home_win + prob_draw + prob_away_win
    if total_prob == 0:
        return {'home': 0.33, 'draw': 0.34, 'away': 0.33}
        
    return {
        'home': prob_home_win / total_prob,
        'draw': prob_draw / total_prob,
        'away': prob_away_win / total_prob
    }

    def predict_btts_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de BTTS usando ensemble de modelos.
        """
        if self.use_ensemble and self.advanced_predictor:
            statistical_pred = self._statistical_btts_prediction(features)
            advanced_pred = self.advanced_predictor.predict_btts_probability(features)
            
            # Combinar predicciones
            return 0.4 * statistical_pred + 0.6 * advanced_pred
        else:
            return self._statistical_btts_prediction(features)

    def predict_over_2_5_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de Over 2.5 usando ensemble de modelos.
        """
        if self.use_ensemble and self.advanced_predictor:
            statistical_pred = self._statistical_over_2_5_prediction(features)
            advanced_pred = self.advanced_predictor.predict_over_2_5_probability(features)
            
            return 0.4 * statistical_pred + 0.6 * advanced_pred
        else:
            return self._statistical_over_2_5_prediction(features)

    def predict_over_0_5_fh_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de Over 0.5 FH usando ensemble de modelos.
        """
        if self.use_ensemble and self.advanced_predictor:
            statistical_pred = self._statistical_over_0_5_fh_prediction(features)
            advanced_pred = self.advanced_predictor.predict_over_0_5_fh_probability(features)
            
            return 0.4 * statistical_pred + 0.6 * advanced_pred
        else:
            return self._statistical_over_0_5_fh_prediction(features)

    def predict_over_1_5_fh_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de Over 1.5 FH usando ensemble de modelos.
        """
        if self.use_ensemble and self.advanced_predictor:
            statistical_pred = self._statistical_over_1_5_fh_prediction(features)
            advanced_pred = self.advanced_predictor.predict_over_1_5_fh_probability(features)
            
            return 0.4 * statistical_pred + 0.6 * advanced_pred
        else:
            return self._statistical_over_1_5_fh_prediction(features)
    
    # Métodos estadísticos mejorados
    def _statistical_btts_prediction(self, features: pd.Series) -> float:
        """Predicción estadística mejorada para BTTS."""
        home_btts_prob = safe_get_feature(features, 'home_btts_percentage', DEFAULT_BTTS_PERCENTAGE) / 100
        away_btts_prob = safe_get_feature(features, 'away_btts_percentage', DEFAULT_BTTS_PERCENTAGE) / 100
        
        # Ajuste por enfrentamientos directos si disponible
        h2h_btts = safe_get_feature(features, 'h2h_btts_percentage', DEFAULT_BTTS_PERCENTAGE) / 100
        h2h_matches = safe_get_feature(features, 'h2h_matches_count', 0)
        
        if h2h_matches >= 3:
            # Dar más peso al historial directo
            return 0.3 * home_btts_prob + 0.3 * away_btts_prob + 0.4 * h2h_btts
        else:
            return (home_btts_prob + away_btts_prob) / 2
    
    def _statistical_over_2_5_prediction(self, features: pd.Series) -> float:
        """Predicción estadística mejorada para Over 2.5."""
        home_over_prob = safe_get_feature(features, 'home_over_2_5_percentage', DEFAULT_OVER_2_5_PERCENTAGE) / 100
        away_over_prob = safe_get_feature(features, 'away_over_2_5_percentage', DEFAULT_OVER_2_5_PERCENTAGE) / 100
        
        # Ajuste por enfrentamientos directos
        h2h_over = safe_get_feature(features, 'h2h_over_2_5_percentage', DEFAULT_OVER_2_5_PERCENTAGE) / 100
        h2h_matches = safe_get_feature(features, 'h2h_matches_count', 0)
        
        if h2h_matches >= 3:
            return 0.3 * home_over_prob + 0.3 * away_over_prob + 0.4 * h2h_over
        else:
            return (home_over_prob + away_over_prob) / 2
    
    def _statistical_over_0_5_fh_prediction(self, features: pd.Series) -> float:
        """Predicción estadística para Over 0.5 FH."""
        home_over_fh_prob = safe_get_feature(features, 'home_over_0_5_fh_percentage', DEFAULT_OVER_0_5_FH_PERCENTAGE) / 100
        away_over_fh_prob = safe_get_feature(features, 'away_over_0_5_fh_percentage', DEFAULT_OVER_0_5_FH_PERCENTAGE) / 100
        
        return (home_over_fh_prob + away_over_fh_prob) / 2
    
    def _statistical_over_1_5_fh_prediction(self, features: pd.Series) -> float:
        """Predicción estadística para Over 1.5 FH."""
        home_over_fh_prob = safe_get_feature(features, 'home_over_1_5_fh_percentage', DEFAULT_OVER_1_5_FH_PERCENTAGE) / 100
        away_over_fh_prob = safe_get_feature(features, 'away_over_1_5_fh_percentage', DEFAULT_OVER_1_5_FH_PERCENTAGE) / 100
        
        return (home_over_fh_prob + away_over_fh_prob) / 2
    
    def _combine_predictions(self, pred1: Dict[str, float], pred2: Dict[str, float], 
                           statistical_weight: float = 0.3, advanced_weight: float = 0.7) -> Dict[str, float]:
        """
        Combina predicciones de múltiples modelos usando pesos.
        
        Args:
            pred1: Predicciones del modelo estadístico
            pred2: Predicciones del modelo avanzado
            statistical_weight: Peso del modelo estadístico
            advanced_weight: Peso del modelo avanzado
            
        Returns:
            Predicciones combinadas
        """
        combined = {}
        
        for key in pred1.keys():
            if key in pred2:
                combined[key] = (statistical_weight * pred1[key] + 
                               advanced_weight * pred2[key])
            else:
                combined[key] = pred1[key]
        
        # Normalizar probabilidades para 1X2
        if 'home' in combined and 'draw' in combined and 'away' in combined:
            total = combined['home'] + combined['draw'] + combined['away']
            if total > 0:
                combined['home'] /= total
                combined['draw'] /= total
                combined['away'] /= total
        
        return combined