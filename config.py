"""
Configuración principal para Panel Bet IA

Carga variables de entorno y valida configuración básica.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# =============================================================================
# CONFIGURACIÓN DE BASE DE DATOS
# =============================================================================
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE', 'betting_predictions')
}

# =============================================================================
# CONFIGURACIÓN DE API
# =============================================================================
API_KEY = os.getenv('API_KEY')
API_HOST = os.getenv('API_HOST', 'api-football-v1.p.rapidapi.com')

API_HEADERS = {
    'X-RapidAPI-Key': API_KEY,
    'X-RapidAPI-Host': API_HOST
}

# =============================================================================
# VALIDACIÓN DE CONFIGURACIÓN CRÍTICA
# =============================================================================
def validate_config():
    """Valida que la configuración crítica esté presente."""
    missing_vars = []
    
    if not API_KEY:
        missing_vars.append('API_KEY')
    if not DB_CONFIG['user']:
        missing_vars.append('DB_USER')
    if not DB_CONFIG['password']:
        missing_vars.append('DB_PASSWORD')
    
    if missing_vars:
        raise ValueError(
            f"Variables de entorno faltantes: {', '.join(missing_vars)}. "
            f"Revisa tu archivo .env"
        )

# Validar configuración al importar
validate_config()