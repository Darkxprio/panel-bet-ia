"""
Constantes centralizadas para Panel Bet IA

Este archivo contiene todas las constantes y configuraciones
que se usan a lo largo del proyecto para evitar números mágicos.
"""

# =============================================================================
# CONFIGURACIÓN DE PREDICCIÓN
# =============================================================================
MIN_MATCHES_REQUIRED = 5           # Mínimo de partidos necesarios para predicción
MIN_LEAGUE_MATURITY_DAYS = 30      # Días mínimos desde inicio de temporada
VALUE_BET_MARGIN = 0.05            # Margen mínimo para considerar apuesta de valor (5%)
MAX_GOALS_POISSON = 10             # Máximo de goles para cálculos de Poisson
TEAM_HISTORY_MATCHES = 20          # Número de partidos históricos a analizar

# =============================================================================
# BOOKMAKER IDs
# =============================================================================
PRIMARY_BOOKMAKER_ID = 8           # Bet365
SECONDARY_BOOKMAKER_IDS = [3, 4, 6, 11]  # Betfair, Pinnacle, Bwin, 1xBet

# =============================================================================
# MARKET IDs (Para identificar mercados en la API)
# =============================================================================
MARKET_1X2 = 1                     # Resultado final
MARKET_BTTS = 8                    # Ambos equipos anotan
MARKET_OVER_UNDER_2_5 = 5          # Más/Menos 2.5 goles
MARKET_FIRST_HALF_GOALS = 21       # Goles primera mitad

# =============================================================================
# VALORES POR DEFECTO PARA FEATURES
# =============================================================================
DEFAULT_GOALS_SCORED = 1.2         # Goles promedio por defecto
DEFAULT_GOALS_CONCEDED = 1.2       # Goles concedidos por defecto
DEFAULT_BTTS_PERCENTAGE = 50       # Porcentaje BTTS por defecto
DEFAULT_OVER_2_5_PERCENTAGE = 50   # Porcentaje Over 2.5 por defecto
DEFAULT_OVER_0_5_FH_PERCENTAGE = 70  # Porcentaje Over 0.5 FH por defecto
DEFAULT_OVER_1_5_FH_PERCENTAGE = 30  # Porcentaje Over 1.5 FH por defecto

# =============================================================================
# CONFIGURACIÓN DE CONFIANZA (KELLY CRITERION)
# =============================================================================
KELLY_HIGH_THRESHOLD = 0.1         # Kelly > 10% = Confianza Alta
KELLY_MEDIUM_THRESHOLD = 0.05      # Kelly > 5% = Confianza Media
# Kelly <= 5% = Confianza Baja

# =============================================================================
# MENSAJES DE ESTADO
# =============================================================================
MSG_PIPELINE_START = "🚀 Iniciando el pipeline de predicción diaria..."
MSG_NO_MATCHES = "No se encontraron partidos para hoy. Finalizando."
MSG_LEAGUE_IMMATURE = "Omitiendo: La liga {} no es suficientemente madura."
MSG_INSUFFICIENT_DATA = "Omitiendo: Datos históricos insuficientes."
MSG_NO_FEATURES = "Omitiendo: No se pudieron generar las características."
MSG_NO_ODDS = "Omitiendo: No hay cuotas disponibles del bookmaker primario."
MSG_VALUE_FOUND = "¡VALOR ENCONTRADO! Mercado: {}, Pred: {}, Confianza: {}, {}"
MSG_PARSING_ERROR = "Omitiendo: No se pudieron parsear las cuotas necesarias del bookmaker primario. Error: {}"
