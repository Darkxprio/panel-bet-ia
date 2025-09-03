import mysql.connector
from config import DB_CONFIG

def save_predictions(predictions: list):
    if not predictions:
        return 0
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO daily_predictions 
            (
                fixture_id, matchTimestampUTC, league, country, 
                teamA, teamA_logo_url, teamB, teamB_logo_url, 
                market, prediction, odds, calculated_probability, value_edge,
                confidence, reasoning
            )
            VALUES 
            (
                %(fixture_id)s, %(matchTimestampUTC)s, %(league)s, %(country)s,
                %(teamA)s, %(teamA_logo_url)s, %(teamB)s, %(teamB_logo_url)s,
                %(market)s, %(prediction)s, %(odds)s, %(calculated_probability)s, %(value_edge)s,
                %(confidence)s, %(reasoning)s
            )
        """
        cursor.executemany(sql, predictions)
        conn.commit()
        return cursor.rowcount
    except mysql.connector.Error as e:
        print(f"Error saving to DB: {e}")
        return 0
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()