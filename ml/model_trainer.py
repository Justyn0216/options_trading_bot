# options_trading_bot/ml/model_trainer.py

import pandas as pd
import numpy as np
import logging
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains and manages machine learning models for option trading
    """
    
    def __init__(self, data_manager):
        """
        Initialize model trainer
        
        Args:
            data_manager: DataManager instance
        """
        self.data_manager = data_manager
        
        # Local directories
        self.models_dir = "models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # Model file paths
        self.entry_model_path = os.path.join(self.models_dir, "entry_model.joblib")
        self.exit_model_path = os.path.join(self.models_dir, "exit_model.joblib")
        self.profit_model_path = os.path.join(self.models_dir, "profit_model.joblib")
        self.scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        
        # Load models if they exist
        self.entry_model = self._load_model(self.entry_model_path)
        self.exit_model = self._load_model(self.exit_model_path)
        self.profit_model = self._load_model(self.profit_model_path)
        self.scaler = self._load_model(self.scaler_path)
        
        # Feature columns
        self.feature_columns = [
            'strike', 'days_to_expiry', 'iv_at_entry', 'delta', 'gamma', 
            'theta', 'vega', 'rho', 'price_difference_percent'
        ]
    
    def _load_model(self, model_path):
        """
        Load model from file
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            object: Loaded model or None if file doesn't exist
        """
        try:
            if os.path.exists(model_path):
                return joblib.load(model_path)
            return None
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None
    
    def _save_model(self, model, model_path):
        """
        Save model to file
        
        Args:
            model: Model to save
            model_path (str): Path to save model
            
        Returns:
            bool: True if model was saved successfully
        """
        try:
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_path}: {e}")
            return False
    
    def _prepare_training_data(self):
        """
        Prepare training data from historical trades
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, scaler
        """
        # Load historical trade data
        trades_df = self.data_manager.load_trades()
        
        if trades_df.empty:
            logger.warning("No historical trade data available for training")
            return None, None, None, None, None
        
        # Clean and prepare data
        trades_df = trades_df.dropna(subset=['pnl_percent'])
        
        # Create target variables
        # 1. Entry signal (whether the trade was profitable)
        trades_df['profitable'] = (trades_df['pnl_percent'] > 0).astype(int)
        
        # 2. Exit signal (whether the exit reason was take_profit)
        trades_df['good_exit'] = (trades_df['exit_reason'] == 'take_profit').astype(int)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in trades_df.columns:
                logger.warning(f"Missing column in training data: {col}")
                return None, None, None, None, None
        
        # Extract features
        X = trades_df[self.feature_columns].copy()
        
        # Create target variables
        y_entry = trades_df['profitable']
        y_exit = trades_df['good_exit']
        y_profit = trades_df['pnl_percent']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_entry_train, y_entry_test = train_test_split(
            X_scaled, y_entry, test_size=0.2, random_state=42
        )
        _, _, y_exit_train, y_exit_test = train_test_split(
            X_scaled, y_exit, test_size=0.2, random_state=42
        )
        _, _, y_profit_train, y_profit_test = train_test_split(
            X_scaled, y_profit, test_size=0.2, random_state=42
        )
        
        logger.info(f"Prepared training data with {len(X_train)} samples")
        
        return (X_train, X_test, 
                y_entry_train, y_entry_test, 
                y_exit_train, y_exit_test, 
                y_profit_train, y_profit_test, 
                scaler)
    
    def train_models(self):
        """
        Train machine learning models
        
        Returns:
            bool: True if models were trained successfully
        """
        try:
            # Prepare training data
            result = self._prepare_training_data()
            
            if result is None or result[0] is None:
                logger.warning("Could not prepare training data")
                return False
                
            (X_train, X_test, 
             y_entry_train, y_entry_test, 
             y_exit_train, y_exit_test, 
             y_profit_train, y_profit_test, 
             scaler) = result
            
            # 1. Train entry model (predicts if a trade will be profitable)
            entry_model = RandomForestClassifier(n_estimators=100, random_state=42)
            entry_model.fit(X_train, y_entry_train)
            
            # Evaluate entry model
            y_entry_pred = entry_model.predict(X_test)
            entry_accuracy = accuracy_score(y_entry_test, y_entry_pred)
            logger.info(f"Entry model accuracy: {entry_accuracy:.4f}")
            logger.info(f"Entry model classification report:\n{classification_report(y_entry_test, y_entry_pred)}")
            
            # 2. Train exit model (predicts if take profit is likely)
            exit_model = RandomForestClassifier(n_estimators=100, random_state=42)
            exit_model.fit(X_train, y_exit_train)
            
            # Evaluate exit model
            y_exit_pred = exit_model.predict(X_test)
            exit_accuracy = accuracy_score(y_exit_test, y_exit_pred)
            logger.info(f"Exit model accuracy: {exit_accuracy:.4f}")
            logger.info(f"Exit model classification report:\n{classification_report(y_exit_test, y_exit_pred)}")
            
            # 3. Train profit model (predicts expected profit percentage)
            profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
            profit_model.fit(X_train, y_profit_train)
            
            # Evaluate profit model
            y_profit_pred = profit_model.predict(X_test)
            profit_rmse = np.sqrt(mean_squared_error(y_profit_test, y_profit_pred))
            logger.info(f"Profit model RMSE: {profit_rmse:.4f}")
            
            # Save models
            self._save_model(entry_model, self.entry_model_path)
            self._save_model(exit_model, self.exit_model_path)
            self._save_model(profit_model, self.profit_model_path)
            self._save_model(scaler, self.scaler_path)
            
            # Update instance variables
            self.entry_model = entry_model
            self.exit_model = exit_model
            self.profit_model = profit_model
            self.scaler = scaler
            
            logger.info("Successfully trained and saved all models")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}", exc_info=True)
            return False
    
    def predict_entry(self, option_data):
        """
        Predict whether to enter a trade
        
        Args:
            option_data (dict): Option data
            
        Returns:
            tuple: (probability of profit, recommendation)
        """
        if self.entry_model is None or self.scaler is None:
            logger.warning("Entry model not available")
            return None, None
            
        try:
            # Extract features
            features = []
            for col in self.feature_columns:
                if col in option_data:
                    features.append(option_data[col])
                else:
                    logger.warning(f"Missing feature for prediction: {col}")
                    return None, None
            
            # Scale features
            X = self.scaler.transform([features])
            
            # Predict probability of profit
            probabilities = self.entry_model.predict_proba(X)[0]
            profit_probability = probabilities[1]  # Probability of class 1 (profitable)
            
            # Make recommendation
            if profit_probability >= 0.7:
                recommendation = "strong_buy"
            elif profit_probability >= 0.6:
                recommendation = "buy"
            elif profit_probability <= 0.3:
                recommendation = "avoid"
            else:
                recommendation = "neutral"
                
            return profit_probability, recommendation
            
        except Exception as e:
            logger.error(f"Error predicting entry: {e}")
            return None, None
    
    def predict_profit(self, option_data):
        """
        Predict expected profit percentage
        
        Args:
            option_data (dict): Option data
            
        Returns:
            float: Expected profit percentage
        """
        if self.profit_model is None or self.scaler is None:
            logger.warning("Profit model not available")
            return None
            
        try:
            # Extract features
            features = []
            for col in self.feature_columns:
                if col in option_data:
                    features.append(option_data[col])
                else:
                    logger.warning(f"Missing feature for prediction: {col}")
                    return None
            
            # Scale features
            X = self.scaler.transform([features])
            
            # Predict profit percentage
            expected_profit = self.profit_model.predict(X)[0]
            
            return expected_profit
            
        except Exception as e:
            logger.error(f"Error predicting profit: {e}")
            return None
    
    def filter_opportunities(self, opportunities):
        """
        Filter and rank opportunities using ML models
        
        Args:
            opportunities (list): List of option opportunities
            
        Returns:
            list: Filtered and ranked opportunities
        """
        if not self.models_available() or not opportunities:
            return opportunities
            
        try:
            # Evaluate each opportunity
            for opportunity in opportunities:
                # Predict profit probability
                profit_prob, recommendation = self.predict_entry(opportunity)
                if profit_prob is not None:
                    opportunity['ml_profit_probability'] = profit_prob
                    opportunity['ml_recommendation'] = recommendation
                
                # Predict expected profit
                expected_profit = self.predict_profit(opportunity)
                if expected_profit is not None:
                    opportunity['ml_expected_profit'] = expected_profit
            
            # Filter out opportunities with negative recommendations
            filtered = [op for op in opportunities if op.get('ml_recommendation') not in ['avoid']]
            
            # Rank by combination of factors
            if filtered:
                # Create a score based on both ML predictions and price difference
                for op in filtered:
                    ml_score = op.get('ml_profit_probability', 0) * 0.5 + op.get('price_difference_percent', 0) / 100 * 0.5
                    op['ml_score'] = ml_score
                
                # Sort by ML score
                sorted_opportunities = sorted(filtered, key=lambda x: x.get('ml_score', 0), reverse=True)
                return sorted_opportunities
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering opportunities: {e}")
            return opportunities
    
    def models_available(self):
        """
        Check if ML models are available
        
        Returns:
            bool: True if models are available
        """
        return (self.entry_model is not None and 
                self.profit_model is not None and 
                self.scaler is not None)
    
    def get_feature_importance(self):
        """
        Get feature importance from models
        
        Returns:
            dict: Feature importance for each model
        """
        if not self.models_available():
            return None
            
        try:
            # Get feature importance from each model
            entry_importance = self.entry_model.feature_importances_
            profit_importance = self.profit_model.feature_importances_
            
            # Create dictionary mapping features to importance
            entry_features = {feature: importance for feature, importance in zip(self.feature_columns, entry_importance)}
            profit_features = {feature: importance for feature, importance in zip(self.feature_columns, profit_importance)}
            
            return {
                'entry_model': entry_features,
                'profit_model': profit_features
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
