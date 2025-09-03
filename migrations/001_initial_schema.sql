-- =============================================================================
-- Initial Database Schema for Panel Bet IA
-- =============================================================================
-- Migration: 001_initial_schema
-- Created: 2025-01-09
-- Description: Create initial tables for betting predictions system

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS betting_predictions 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE betting_predictions;

-- =============================================================================
-- Daily Predictions Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS daily_predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    fixture_id INT NOT NULL,
    matchTimestampUTC DATETIME NOT NULL,
    league VARCHAR(255) NOT NULL,
    country VARCHAR(100) NOT NULL,
    teamA VARCHAR(255) NOT NULL,
    teamA_logo_url TEXT,
    teamB VARCHAR(255) NOT NULL,
    teamB_logo_url TEXT,
    market VARCHAR(100) NOT NULL,
    prediction VARCHAR(255) NOT NULL,
    odds DECIMAL(10,2) NOT NULL,
    calculated_probability DECIMAL(5,4) NOT NULL,
    value_edge DECIMAL(5,4) NOT NULL,
    confidence ENUM('Alta', 'Media', 'Baja') NOT NULL,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for better performance
    INDEX idx_fixture_id (fixture_id),
    INDEX idx_match_date (matchTimestampUTC),
    INDEX idx_league (league),
    INDEX idx_market (market),
    INDEX idx_confidence (confidence),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- Prediction Results Table (for backtesting and tracking)
-- =============================================================================
CREATE TABLE IF NOT EXISTS prediction_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    prediction_id BIGINT NOT NULL,
    actual_result VARCHAR(100),
    is_correct BOOLEAN,
    profit_loss DECIMAL(10,2),
    match_finished_at DATETIME,
    result_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (prediction_id) REFERENCES daily_predictions(id) ON DELETE CASCADE,
    INDEX idx_prediction_id (prediction_id),
    INDEX idx_is_correct (is_correct),
    INDEX idx_match_finished (match_finished_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- Team Statistics Cache
-- =============================================================================
CREATE TABLE IF NOT EXISTS team_stats_cache (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    team_id INT NOT NULL,
    season_year INT NOT NULL,
    matches_played INT DEFAULT 0,
    wins INT DEFAULT 0,
    draws INT DEFAULT 0,
    losses INT DEFAULT 0,
    goals_scored INT DEFAULT 0,
    goals_conceded INT DEFAULT 0,
    btts_count INT DEFAULT 0,
    over_2_5_count INT DEFAULT 0,
    over_0_5_fh_count INT DEFAULT 0,
    over_1_5_fh_count INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_team_season (team_id, season_year),
    INDEX idx_team_id (team_id),
    INDEX idx_season (season_year),
    INDEX idx_last_updated (last_updated)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- Model Performance Tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    market_type VARCHAR(50) NOT NULL,
    date_range_start DATE NOT NULL,
    date_range_end DATE NOT NULL,
    total_predictions INT NOT NULL,
    correct_predictions INT NOT NULL,
    accuracy DECIMAL(5,4) NOT NULL,
    avg_odds DECIMAL(5,2),
    total_roi DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_model_name (model_name),
    INDEX idx_market_type (market_type),
    INDEX idx_date_range (date_range_start, date_range_end),
    INDEX idx_accuracy (accuracy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- System Logs
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    level ENUM('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') NOT NULL,
    module VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    extra_data JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_level (level),
    INDEX idx_module (module),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert initial model performance record
INSERT IGNORE INTO model_performance 
(model_name, market_type, date_range_start, date_range_end, total_predictions, correct_predictions, accuracy, avg_odds, total_roi, sharpe_ratio, max_drawdown)
VALUES 
('PoissonPredictor', '1X2', '2024-01-01', '2024-12-31', 0, 0, 0.0000, 0.00, 0.0000, 0.0000, 0.0000),
('PoissonPredictor', 'BTTS', '2024-01-01', '2024-12-31', 0, 0, 0.0000, 0.00, 0.0000, 0.0000, 0.0000),
('PoissonPredictor', 'Over/Under', '2024-01-01', '2024-12-31', 0, 0, 0.0000, 0.00, 0.0000, 0.0000, 0.0000);

-- Log the migration
INSERT INTO system_logs (level, module, message, extra_data)
VALUES ('INFO', 'Migration', 'Initial schema created successfully', JSON_OBJECT('migration', '001_initial_schema', 'version', '1.0.0'));
