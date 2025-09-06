"""
Predictor Avanzado para Panel Bet IA

Implementa algoritmos de machine learning más sofisticados para
mejorar significativamente la precisión de las predicciones.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from utils import get_logger, calculate_poisson_probabilities, safe_get_feature
logger = get_logger("advanced_predictor")

from constants import (
    MAX_GOALS_POISSON,
    DEFAULT_GOALS_SCORED,
    DEFAULT_GOALS_CONCEDED,
    DEFAULT_BTTS_PERCENTAGE,
    DEFAULT_OVER_2_5_PERCENTAGE,
    DEFAULT_OVER_0_5_FH_PERCENTAGE,
    DEFAULT_OVER_1_5_FH_PERCENTAGE
)


class AdvancedPredictor:
    """
    Predictor avanzado que utiliza múltiples algoritmos de ML
    y técnicas estadísticas sofisticadas.
    """
    
    def __init__(self):
        """Inicializa el predictor avanzado con múltiples modelos."""
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Inicializar modelos para diferentes mercados
        self._initialize_models()
        
    
    def _initialize_models(self):
        """Inicializa los modelos de ML para cada mercado."""
        # Modelos para predicción de goles (regresión)
        self.models['goals_home'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.models['goals_away'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Modelos para clasificación binaria (BTTS, Over/Under)
        self.models['btts'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )
        
        self.models['over_2_5'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )
        
        self.models['over_0_5_fh'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        self.models['over_1_5_fh'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # Scalers para normalización
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def predict_winner_probabilities(self, features: pd.Series) -> Dict[str, float]:
        """
        Predice probabilidades 1X2 usando un enfoque híbrido:
        1. Predicción de goles esperados con ML
        2. Distribución de Poisson mejorada
        3. Ajustes contextuales
        """
        # Preparar features
        feature_array = self._prepare_features(features)
        
        if not self.is_trained:
            # Si no hay modelos entrenados, usar método estadístico mejorado
            return self._statistical_winner_prediction(features)
        
        try:
            # Predecir goles esperados usando ML
            home_goals_pred = self.models['goals_home'].predict([feature_array])[0]
            away_goals_pred = self.models['goals_away'].predict([feature_array])[0]
            
            # Asegurar valores positivos
            home_goals_pred = max(0.1, home_goals_pred)
            away_goals_pred = max(0.1, away_goals_pred)
            
        except Exception:
            # Fallback al método estadístico
            return self._statistical_winner_prediction(features)
        
        # Aplicar ajustes contextuales
        home_goals_pred, away_goals_pred = self._apply_contextual_adjustments(
            features, home_goals_pred, away_goals_pred
        )
        
        # Calcular probabilidades usando Poisson mejorado
        return self._calculate_winner_probabilities_from_goals(
            home_goals_pred, away_goals_pred, features
        )
    
    def predict_btts_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de BTTS usando algoritmos avanzados.
        """
        feature_array = self._prepare_features(features)
        
        if not self.is_trained:
            return self._statistical_btts_prediction(features)
        
        try:
            # Predicción con ML
            btts_prob = self.models['btts'].predict([feature_array])[0]
            btts_prob = np.clip(btts_prob, 0.05, 0.95)  # Limitar entre 5% y 95%
            
            # Aplicar ajustes basados en contexto
            btts_prob = self._apply_btts_adjustments(features, btts_prob)
            
            return float(btts_prob)
            
        except Exception:
            return self._statistical_btts_prediction(features)
    
    def predict_over_2_5_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de Over 2.5 goles.
        """
        feature_array = self._prepare_features(features)
        
        if not self.is_trained:
            return self._statistical_over_prediction(features, 2.5)
        
        try:
            over_prob = self.models['over_2_5'].predict([feature_array])[0]
            over_prob = np.clip(over_prob, 0.05, 0.95)
            
            # Ajustes contextuales
            over_prob = self._apply_over_adjustments(features, over_prob, 2.5)
            
            return float(over_prob)
            
        except Exception:
            return self._statistical_over_prediction(features, 2.5)
    
    def predict_over_0_5_fh_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de Over 0.5 goles en primera mitad.
        """
        feature_array = self._prepare_features(features)
        
        if not self.is_trained:
            return self._statistical_fh_prediction(features, 0.5)
        
        try:
            over_prob = self.models['over_0_5_fh'].predict([feature_array])[0]
            over_prob = np.clip(over_prob, 0.1, 0.95)  # Primera mitad tiene mayor probabilidad
            
            return float(over_prob)
            
        except Exception:
            return self._statistical_fh_prediction(features, 0.5)
    
    def predict_over_1_5_fh_probability(self, features: pd.Series) -> float:
        """
        Predice probabilidad de Over 1.5 goles en primera mitad.
        """
        feature_array = self._prepare_features(features)
        
        if not self.is_trained:
            return self._statistical_fh_prediction(features, 1.5)
        
        try:
            over_prob = self.models['over_1_5_fh'].predict([feature_array])[0]
            over_prob = np.clip(over_prob, 0.05, 0.8)
            
            return float(over_prob)
            
        except Exception:
            return self._statistical_fh_prediction(features, 1.5)
    
    def _prepare_features(self, features: pd.Series) -> np.ndarray:
        """
        Prepara las características para los modelos de ML.
        """
        # Seleccionar las características más importantes
        important_features = [
            'home_avg_goals_scored', 'home_avg_goals_conceded',
            'away_avg_goals_scored', 'away_avg_goals_conceded',
            'home_recent_avg_goals_scored', 'home_recent_avg_goals_conceded',
            'away_recent_avg_goals_scored', 'away_recent_avg_goals_conceded',
            'home_goal_scoring_trend', 'home_defensive_trend',
            'away_goal_scoring_trend', 'away_defensive_trend',
            'home_win_percentage', 'away_win_percentage',
            'home_btts_percentage', 'away_btts_percentage',
            'home_over_2_5_percentage', 'away_over_2_5_percentage',
            'attack_vs_defense_ratio_home', 'attack_vs_defense_ratio_away',
            'form_difference', 'quality_difference',
            'h2h_avg_goals', 'h2h_btts_percentage',
            'is_major_league', 'is_weekend'
        ]
        
        feature_vector = []
        for feature_name in important_features:
            value = safe_get_feature(features, feature_name, 0.0)
            feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def _statistical_winner_prediction(self, features: pd.Series) -> Dict[str, float]:
        """
        Predicción estadística mejorada para 1X2.
        """
        # Obtener goles esperados usando múltiples métodos
        home_exp_1 = safe_get_feature(features, 'home_avg_goals_scored', DEFAULT_GOALS_SCORED)
        away_def_1 = safe_get_feature(features, 'away_avg_goals_conceded', DEFAULT_GOALS_CONCEDED)
        
        away_exp_1 = safe_get_feature(features, 'away_avg_goals_scored', DEFAULT_GOALS_SCORED)
        home_def_1 = safe_get_feature(features, 'home_avg_goals_conceded', DEFAULT_GOALS_CONCEDED)
        
        # Método 1: Ataque vs Defensa
        home_exp_goals_1 = (home_exp_1 + away_def_1) / 2
        away_exp_goals_1 = (away_exp_1 + home_def_1) / 2
        
        # Método 2: Forma reciente
        home_recent = safe_get_feature(features, 'home_recent_avg_goals_scored', home_exp_1)
        away_recent = safe_get_feature(features, 'away_recent_avg_goals_scored', away_exp_1)
        
        # Método 3: Ajuste por contexto (local/visitante)
        home_context = safe_get_feature(features, 'home_home_avg_goals_scored', home_exp_1)
        away_context = safe_get_feature(features, 'away_away_avg_goals_scored', away_exp_1)
        
        # Combinar métodos con pesos
        home_exp_goals = (
            0.4 * home_exp_goals_1 +
            0.35 * home_recent +
            0.25 * home_context
        )
        
        away_exp_goals = (
            0.4 * away_exp_goals_1 +
            0.35 * away_recent +
            0.25 * away_context
        )
        
        # Aplicar ajustes contextuales
        home_exp_goals, away_exp_goals = self._apply_contextual_adjustments(
            features, home_exp_goals, away_exp_goals
        )
        
        return self._calculate_winner_probabilities_from_goals(
            home_exp_goals, away_exp_goals, features
        )
    
    def _calculate_winner_probabilities_from_goals(self, home_goals: float, away_goals: float, features: pd.Series) -> Dict[str, float]:
        """
        Calcula probabilidades 1X2 a partir de goles esperados usando Poisson mejorado.
        """
        # Calcular probabilidades de Poisson
        home_probs = calculate_poisson_probabilities(home_goals, MAX_GOALS_POISSON)
        away_probs = calculate_poisson_probabilities(away_goals, MAX_GOALS_POISSON)
        
        # Calcular probabilidades de resultados
        prob_home_win = prob_draw = prob_away_win = 0.0
        
        for home_score in range(MAX_GOALS_POISSON + 1):
            for away_score in range(MAX_GOALS_POISSON + 1):
                prob = home_probs[home_score] * away_probs[away_score]
                
                if home_score > away_score:
                    prob_home_win += prob
                elif home_score == away_score:
                    prob_draw += prob
                else:
                    prob_away_win += prob
        
        # Normalizar
        total = prob_home_win + prob_draw + prob_away_win
        if total == 0:
            return {'home': 0.33, 'draw': 0.34, 'away': 0.33}
        
        # Aplicar ajustes finales basados en características avanzadas
        home_adj = 1.0 + safe_get_feature(features, 'home_advantage', 0) * 0.1
        form_adj = safe_get_feature(features, 'form_difference', 0) * 0.05
        
        prob_home_win *= home_adj * (1 + form_adj)
        prob_away_win *= (1 - form_adj)
        
        # Re-normalizar
        total = prob_home_win + prob_draw + prob_away_win
        
        return {
            'home': prob_home_win / total,
            'draw': prob_draw / total,
            'away': prob_away_win / total
        }
    
    def _apply_contextual_adjustments(self, features: pd.Series, home_goals: float, away_goals: float) -> Tuple[float, float]:
        """
        Aplica ajustes contextuales a los goles esperados.
        """
        # Ajuste por ventaja local
        home_advantage = safe_get_feature(features, 'home_advantage', 0)
        home_goals += home_advantage * 0.1
        
        # Ajuste por forma reciente
        form_diff = safe_get_feature(features, 'form_difference', 0)
        home_goals += form_diff * 0.05
        away_goals -= form_diff * 0.05
        
        # Ajuste por tendencias
        home_trend = safe_get_feature(features, 'home_goal_scoring_trend', 0)
        away_trend = safe_get_feature(features, 'away_goal_scoring_trend', 0)
        
        home_goals += home_trend * 0.1
        away_goals += away_trend * 0.1
        
        # Ajuste por enfrentamientos directos
        h2h_avg = safe_get_feature(features, 'h2h_avg_goals', 2.5)
        if h2h_avg > 0:
            total_expected = home_goals + away_goals
            if total_expected > 0:
                adjustment = (h2h_avg / total_expected) * 0.2
                home_goals *= (1 + adjustment)
                away_goals *= (1 + adjustment)
        
        # Asegurar valores mínimos
        home_goals = max(0.1, home_goals)
        away_goals = max(0.1, away_goals)
        
        return home_goals, away_goals
    
    def _statistical_btts_prediction(self, features: pd.Series) -> float:
        """
        Predicción estadística mejorada para BTTS.
        """
        # Múltiples enfoques para BTTS
        home_btts = safe_get_feature(features, 'home_btts_percentage', DEFAULT_BTTS_PERCENTAGE) / 100
        away_btts = safe_get_feature(features, 'away_btts_percentage', DEFAULT_BTTS_PERCENTAGE) / 100
        
        # Enfoque 1: Promedio simple
        btts_prob_1 = (home_btts + away_btts) / 2
        
        # Enfoque 2: Basado en capacidad ofensiva
        home_attack = safe_get_feature(features, 'home_avg_goals_scored', DEFAULT_GOALS_SCORED)
        away_attack = safe_get_feature(features, 'away_avg_goals_scored', DEFAULT_GOALS_SCORED)
        
        # Probabilidad de que ambos marquen basada en Poisson
        home_no_goals = np.exp(-home_attack)
        away_no_goals = np.exp(-away_attack)
        btts_prob_2 = 1 - (home_no_goals + away_no_goals - home_no_goals * away_no_goals)
        
        # Enfoque 3: Historial directo
        h2h_btts = safe_get_feature(features, 'h2h_btts_percentage', DEFAULT_BTTS_PERCENTAGE) / 100
        
        # Combinar enfoques
        if safe_get_feature(features, 'h2h_matches_count', 0) >= 3:
            # Si hay suficientes enfrentamientos directos, darles más peso
            btts_prob = 0.3 * btts_prob_1 + 0.3 * btts_prob_2 + 0.4 * h2h_btts
        else:
            btts_prob = 0.5 * btts_prob_1 + 0.5 * btts_prob_2
        
        return np.clip(btts_prob, 0.05, 0.95)
    
    def _statistical_over_prediction(self, features: pd.Series, threshold: float) -> float:
        """
        Predicción estadística para Over/Under.
        """
        home_goals = safe_get_feature(features, 'expected_goals_home', DEFAULT_GOALS_SCORED)
        away_goals = safe_get_feature(features, 'expected_goals_away', DEFAULT_GOALS_SCORED)
        
        total_expected = home_goals + away_goals
        
        # Usar distribución de Poisson para calcular P(X > threshold)
        probs = calculate_poisson_probabilities(total_expected, int(threshold + 10))
        
        over_prob = sum(probs[int(threshold) + 1:])
        
        # Ajuste por historial directo
        if threshold == 2.5:
            h2h_over = safe_get_feature(features, 'h2h_over_2_5_percentage', DEFAULT_OVER_2_5_PERCENTAGE) / 100
            if safe_get_feature(features, 'h2h_matches_count', 0) >= 3:
                over_prob = 0.7 * over_prob + 0.3 * h2h_over
        
        return np.clip(over_prob, 0.05, 0.95)
    
    def _statistical_fh_prediction(self, features: pd.Series, threshold: float) -> float:
        """
        Predicción estadística para goles en primera mitad.
        """
        if threshold == 0.5:
            home_fh = safe_get_feature(features, 'home_over_0_5_fh_percentage', DEFAULT_OVER_0_5_FH_PERCENTAGE) / 100
            away_fh = safe_get_feature(features, 'away_over_0_5_fh_percentage', DEFAULT_OVER_0_5_FH_PERCENTAGE) / 100
        else:  # 1.5
            home_fh = safe_get_feature(features, 'home_over_1_5_fh_percentage', DEFAULT_OVER_1_5_FH_PERCENTAGE) / 100
            away_fh = safe_get_feature(features, 'away_over_1_5_fh_percentage', DEFAULT_OVER_1_5_FH_PERCENTAGE) / 100
        
        return np.clip((home_fh + away_fh) / 2, 0.05, 0.95)
    
    def _apply_btts_adjustments(self, features: pd.Series, base_prob: float) -> float:
        """
        Aplica ajustes contextuales a la probabilidad de BTTS.
        """
        # Ajuste por liga (ligas más ofensivas)
        if safe_get_feature(features, 'is_major_league', 0) > 0:
            base_prob *= 1.05  # Pequeño boost para ligas importantes
        
        # Ajuste por fin de semana (equipos más relajados)
        if safe_get_feature(features, 'is_weekend', 0) > 0:
            base_prob *= 1.02
        
        return np.clip(base_prob, 0.05, 0.95)
    
    def _apply_over_adjustments(self, features: pd.Series, base_prob: float, threshold: float) -> float:
        """
        Aplica ajustes contextuales a las probabilidades Over/Under.
        """
        # Ajuste por calidad de equipos
        quality_diff = abs(safe_get_feature(features, 'quality_difference', 0))
        if quality_diff > 10:  # Gran diferencia de calidad
            base_prob *= 0.95  # Partidos más cerrados tienden a tener menos goles
        
        return np.clip(base_prob, 0.05, 0.95)
