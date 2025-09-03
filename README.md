# 🎯 Panel Bet IA - Sistema de Predicción de Apuestas Deportivas

**Por Jordano Cuadros - Lion IA**

Sistema de inteligencia artificial para predicción de apuestas deportivas, utilizando análisis estadístico y machine learning para identificar oportunidades de valor.

## 🚀 Características

- **📊 Análisis Estadístico**: Distribución de Poisson y métricas avanzadas
- **💰 Detección de Valor**: Identificación automática de apuestas con valor positivo
- **⚡ Pipeline Diario**: Procesamiento automatizado de partidos
- **🔍 Múltiples Mercados**: 1X2, BTTS, Over/Under, Goles por tiempo

## 📁 Estructura

```
panel-bet-ia/
├── config.py                      # Configuración
├── main.py                        # Pipeline principal
├── helpers.py                     # Funciones auxiliares
├── data_processing/
│   └── feature_engineering.py    # Procesamiento de datos
├── ml_models/
│   └── predictor.py              # Predictor principal
├── services/
│   ├── api_service.py            # APIs
│   └── db_service.py             # Base de datos
├── migrations/
│   └── 001_initial_schema.sql    # Schema de BD
└── scripts/
    └── setup_database.py         # Setup de BD
```

## ⚙️ Instalación Rápida

### Requisitos
- Python 3.8+
- MySQL/MariaDB
- API Key de RapidAPI (Football API)

### Setup
1. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar variables**
   ```bash
   # Copiar y editar configuración
   copy env.example .env
   # Editar .env con tus credenciales
   ```

3. **Setup base de datos**
   ```bash
   python scripts/setup_database.py
   ```

## 🚀 Uso

**Ejecución diaria:**
```bash
python main.py
```

## 🔧 Configuración Importante

Edita el archivo `.env` con tus credenciales:
```
API_KEY=tu_rapidapi_key_aqui
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=tu_password
DB_DATABASE=betting_predictions
```

## 📊 Mercados Soportados

- **1X2** - Resultado final
- **BTTS** - Ambos equipos anotan  
- **Over/Under 2.5** - Más/Menos de 2.5 goles
- **Over/Under 0.5 FH** - Goles 1ra mitad

## ⚠️ Disclaimer

Este software es para fines educativos. Las apuestas conllevan riesgo financiero. Úsalo bajo tu propia responsabilidad.

---
**Panel Bet IA - Jordano Cuadros | Lion IA**
