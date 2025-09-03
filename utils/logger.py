"""
Sistema de Logging Profesional para Panel Bet IA

Configuración centralizada de logging con archivos rotativos,
diferentes niveles y formato estructurado.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Crear directorio de logs si no existe
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Formatter que añade colores a la consola para mejor legibilidad."""
    
    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Verde
        'WARNING': '\033[33m',   # Amarillo
        'ERROR': '\033[31m',     # Rojo
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Aplicar color solo en consola
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset_color = self.COLORS['RESET']
            
            # Colorear solo el nivel
            original_levelname = record.levelname
            record.levelname = f"{log_color}{record.levelname}{reset_color}"
            
            formatted = super().format(record)
            record.levelname = original_levelname  # Restaurar para otros handlers
            
            return formatted
        else:
            return super().format(record)


def setup_logging(
    name: str = "panel_bet_ia",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura el sistema de logging profesional.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Si escribir logs a archivo
        log_to_console: Si mostrar logs en consola
        max_file_size: Tamaño máximo del archivo de log en bytes
        backup_count: Número de archivos de backup a mantener
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers si ya está configurado
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Formato para archivos de log (más detallado)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formato para consola (más limpio)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Handler para archivo principal
    if log_to_file:
        main_log_file = LOGS_DIR / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Handler separado para errores
        error_log_file = LOGS_DIR / f"{name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Handler para consola
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Log inicial
    logger.info(f"Logger '{name}' inicializado - Nivel: {level}")
    if log_to_file:
        logger.info(f"Logs guardándose en: {LOGS_DIR.absolute()}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger configurado.
    
    Args:
        name: Nombre del logger. Si es None, usa el logger principal.
        
    Returns:
        Logger configurado
    """
    if name is None:
        name = "panel_bet_ia"
    
    logger = logging.getLogger(name)
    
    # Si no está configurado, configurarlo con valores por defecto
    if not logger.handlers:
        return setup_logging(name)
    
    return logger


def log_function_call(func):
    """
    Decorador para loggear llamadas a funciones automáticamente.
    
    Usage:
        @log_function_call
        def my_function(param1, param2):
            return result
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        
        # Log entrada
        logger.debug(f"Iniciando {func_name} con args={args}, kwargs={kwargs}")
        
        try:
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log éxito
            logger.debug(f"Completado {func_name} en {duration:.2f}s")
            return result
            
        except Exception as e:
            # Log error
            logger.error(f"Error en {func_name}: {e}", exc_info=True)
            raise
    
    return wrapper


def log_api_call(endpoint: str, params: dict, response_time: float, status_code: int, response_size: int = 0):
    """
    Log especializado para llamadas a API.
    
    Args:
        endpoint: URL del endpoint
        params: Parámetros de la llamada
        response_time: Tiempo de respuesta en segundos
        status_code: Código de estado HTTP
        response_size: Tamaño de la respuesta en bytes
    """
    logger = get_logger("api")
    
    status_emoji = "✅" if 200 <= status_code < 300 else "❌"
    
    logger.info(
        f"{status_emoji} API Call - {endpoint} | "
        f"Status: {status_code} | "
        f"Time: {response_time:.2f}s | "
        f"Size: {response_size} bytes | "
        f"Params: {params}"
    )


def log_prediction_result(
    match_id: int, 
    teams: str, 
    market: str, 
    prediction: str, 
    probability: float, 
    odds: float, 
    confidence: str,
    has_value: bool
):
    """
    Log especializado para resultados de predicción.
    
    Args:
        match_id: ID del partido
        teams: Nombres de los equipos
        market: Tipo de mercado
        prediction: Predicción específica
        probability: Probabilidad calculada
        odds: Cuota encontrada
        confidence: Nivel de confianza
        has_value: Si la apuesta tiene valor
    """
    logger = get_logger("predictions")
    
    value_emoji = "💰" if has_value else "❌"
    
    logger.info(
        f"{value_emoji} Predicción - {teams} (ID: {match_id}) | "
        f"Mercado: {market} | "
        f"Predicción: {prediction} | "
        f"Prob: {probability:.1%} | "
        f"Cuota: {odds} | "
        f"Confianza: {confidence} | "
        f"Valor: {'SÍ' if has_value else 'NO'}"
    )


def log_pipeline_summary(
    total_matches: int,
    analyzed_matches: int, 
    valuable_predictions: int,
    saved_predictions: int,
    execution_time: float
):
    """
    Log especializado para resumen del pipeline.
    
    Args:
        total_matches: Total de partidos encontrados
        analyzed_matches: Partidos analizados
        valuable_predictions: Predicciones con valor encontradas
        saved_predictions: Predicciones guardadas en BD
        execution_time: Tiempo total de ejecución
    """
    logger = get_logger("pipeline")
    
    success_rate = (valuable_predictions / max(analyzed_matches, 1)) * 100
    
    logger.info(
        f"📊 RESUMEN PIPELINE | "
        f"Partidos: {total_matches} encontrados, {analyzed_matches} analizados | "
        f"Valor: {valuable_predictions} predicciones ({success_rate:.1f}%) | "
        f"Guardadas: {saved_predictions} | "
        f"Tiempo: {execution_time:.1f}s"
    )


# Configurar logging por defecto al importar el módulo
_default_logger = None

def init_default_logging():
    """Inicializa el logging por defecto si no está configurado."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger

# Inicializar automáticamente
init_default_logging()
