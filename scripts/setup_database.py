#!/usr/bin/env python3
"""
Database Setup Script for Panel Bet IA

This script sets up the database schema and initial data.
Usage: python scripts/setup_database.py
"""

import sys
import os
from pathlib import Path
import mysql.connector
from mysql.connector import Error

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import DB_CONFIG
except ImportError:
    print("Error: Could not import DB_CONFIG. Make sure config.py exists and .env is configured.")
    sys.exit(1)


def run_migration_file(cursor, migration_file: Path):
    """Run a single migration file"""
    try:
        print(f"Running migration: {migration_file.name}")
        
        with open(migration_file, 'r', encoding='utf-8') as file:
            migration_sql = file.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement:
                cursor.execute(statement)
        
        print(f"✅ Migration {migration_file.name} completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error running migration {migration_file.name}: {e}")
        return False


def setup_database():
    """Set up the database schema"""
    connection = None
    cursor = None
    
    try:
        # Connect to MySQL server (without specifying database initially)
        connection_config = DB_CONFIG.copy()
        database_name = connection_config.pop('database', 'betting_predictions')
        
        print("🔌 Connecting to MySQL server...")
        connection = mysql.connector.connect(**connection_config)
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        print(f"🗄️ Creating database '{database_name}' if it doesn't exist...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute(f"USE {database_name}")
        
        # Get migrations directory
        migrations_dir = Path(__file__).parent.parent / 'migrations'
        
        if not migrations_dir.exists():
            print("❌ Migrations directory not found!")
            return False
        
        # Get all .sql files and sort them
        migration_files = sorted([f for f in migrations_dir.glob('*.sql')])
        
        if not migration_files:
            print("⚠️ No migration files found in migrations directory")
            return True
        
        print(f"📁 Found {len(migration_files)} migration files")
        
        # Run each migration
        success_count = 0
        for migration_file in migration_files:
            if run_migration_file(cursor, migration_file):
                success_count += 1
            else:
                print(f"❌ Stopping at failed migration: {migration_file.name}")
                break
        
        # Commit all changes
        connection.commit()
        
        print(f"\n✅ Database setup completed successfully!")
        print(f"📊 Executed {success_count}/{len(migration_files)} migrations")
        
        # Verify tables were created
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"📋 Created tables: {[table[0] for table in tables]}")
        
        return True
        
    except Error as e:
        print(f"❌ Database error: {e}")
        if connection:
            connection.rollback()
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            print("🔌 Database connection closed")


def main():
    """Main function"""
    print("🚀 Starting Panel Bet IA Database Setup")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(__file__).parent.parent / '.env'
    if not env_file.exists():
        print("⚠️ Warning: .env file not found. Make sure database configuration is set.")
        print("You can copy env.example to .env and configure your database settings.")
    
    success = setup_database()
    
    if success:
        print("\n🎉 Database setup completed successfully!")
        print("You can now run the main application: python main.py")
    else:
        print("\n❌ Database setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
