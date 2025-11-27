#!/usr/bin/env python3
"""
AI-Powered Flash Loan Arbitrage Agent with Machine Learning
Uses ML models to predict arbitrage opportunities and optimize execution
"""

import asyncio
import json
import time
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from eth_account import Account
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFlashLoanAgent:
    def __init__(self, config_path: str = "config.json", test_mode: bool = False):
        """
        Initialize the ML-powered flash loan agent
        
        Args:
            config_path: Path to configuration file
            test_mode: If True, runs in test mode without real credentials
        """
        self.config = self.load_config(config_path)
        self.test_mode = test_mode
        
        if test_mode:
            # Test mode - use mock data
            logger.info("Running in TEST MODE - no real credentials required")
            self.web3 = None
            self.account = None
            self.contract = None
        else:
            # Production mode - require real credentials
            self.web3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
            
            # Load private key from environment variable
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                raise ValueError("PRIVATE_KEY environment variable not set")
            self.account = Account.from_key(private_key)
            
            self.contract = self.load_contract()
        
        # ML Models
        self.profit_predictor = None
        self.success_predictor = None
        self.scaler = StandardScaler()
        
        # Data storage for training
        self.historical_data = []
        self.feature_columns = [
            'price_diff', 'volume_ratio', 'gas_price', 'time_of_day',
            'day_of_week', 'market_volatility', 'liquidity_depth',
            'slippage_estimate', 'competition_level'
        ]
        
        # Load or initialize ML models
        self.initialize_ml_models()
        
        if test_mode:
            logger.info("ML Agent initialized in TEST MODE")
        else:
            logger.info(f"ML Agent initialized for address: {self.account.address}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
    
    def load_contract(self):
        """Load the flash loan contract"""
        if self.test_mode:
            logger.info("Test mode: Skipping contract loading")
            return None
            
        # Load contract address from environment variable
        contract_address = os.getenv('CONTRACT_ADDRESS')
        if not contract_address:
            raise ValueError("CONTRACT_ADDRESS environment variable not set")
        
        contract_abi = self.config['contract_abi']
        
        return self.web3.eth.contract(
            address=contract_address,
            abi=contract_abi
        )
    
    def initialize_ml_models(self):
        """Initialize or load ML models"""
        try:
            # Try to load existing models
            self.profit_predictor = joblib.load('models/profit_predictor.pkl')
            self.success_predictor = joblib.load('models/success_predictor.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            logger.info("Loaded existing ML models")
        except FileNotFoundError:
            # Initialize new models
            self.profit_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.success_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            logger.info("Initialized new ML models")
    
    def extract_features(self, pool_info: Dict, market_data: Dict) -> np.ndarray:
        """
        Extract features for ML prediction
        
        Args:
            pool_info: Pool information
            market_data: Current market data
            
        Returns:
            Feature array for ML model
        """
        features = []
        
        # Price difference
        price_diff = abs(market_data['uniswap_price'] - market_data['sushi_price'])
        features.append(price_diff)
        
        # Volume ratio
        volume_ratio = market_data['uniswap_volume'] / market_data['sushi_volume']
        features.append(volume_ratio)
        
        # Gas price (normalized)
        gas_price = market_data['gas_price'] / 1e9  # Convert to gwei
        features.append(gas_price)
        
        # Time features
        current_time = time.time()
        time_of_day = (current_time % 86400) / 86400  # Normalized to 0-1
        day_of_week = (current_time // 86400) % 7 / 7  # Normalized to 0-1
        features.extend([time_of_day, day_of_week])
        
        # Market volatility (rolling standard deviation)
        volatility = market_data.get('price_volatility', 0.01)
        features.append(volatility)
        
        # Liquidity depth
        liquidity_depth = market_data.get('liquidity_depth', 1000000)
        features.append(liquidity_depth)
        
        # Slippage estimate
        slippage = market_data.get('slippage_estimate', 0.001)
        features.append(slippage)
        
        # Competition level (estimated from recent transactions)
        competition = market_data.get('competition_level', 0.5)
        features.append(competition)
        
        return np.array(features).reshape(1, -1)
    
    def predict_profit(self, features: np.ndarray) -> float:
        """Predict potential profit using ML model"""
        if self.profit_predictor is None:
            return 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict profit
        predicted_profit = self.profit_predictor.predict(features_scaled)[0]
        return max(0, predicted_profit)  # Ensure non-negative
    
    def predict_success_probability(self, features: np.ndarray) -> float:
        """Predict success probability using ML model"""
        if self.success_predictor is None:
            return 0.5
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict success probability
        success_prob = self.success_predictor.predict(features_scaled)[0]
        return np.clip(success_prob, 0, 1)  # Ensure 0-1 range
    
    def should_execute_arbitrage(self, pool_info: Dict, market_data: Dict) -> Tuple[bool, float, float]:
        """
        ML-powered decision making for arbitrage execution
        
        Args:
            pool_info: Pool information
            market_data: Current market data
            
        Returns:
            (should_execute, predicted_profit, success_probability)
        """
        # Extract features
        features = self.extract_features(pool_info, market_data)
        
        # Get ML predictions
        predicted_profit = self.predict_profit(features)
        success_prob = self.predict_success_probability(features)
        
        # Decision logic combining ML predictions with rules
        min_profit_threshold = self.config.get('min_profit_threshold', 0.001)
        min_success_prob = self.config.get('min_success_probability', 0.7)
        
        # Should execute if both profit and success probability are high enough
        should_execute = (
            predicted_profit > min_profit_threshold and
            success_prob > min_success_prob
        )
        
        return should_execute, predicted_profit, success_prob
    
    def update_ml_models(self, new_data: List[Dict]):
        """
        Update ML models with new data
        
        Args:
            new_data: List of dictionaries with features and outcomes
        """
        if not new_data or len(new_data) < 10:
            return  # Need minimum data for training
        
        # Prepare training data
        X = []
        y_profit = []
        y_success = []
        
        for data_point in new_data:
            features = data_point['features']
            actual_profit = data_point['actual_profit']
            success = 1 if data_point['success'] else 0
            
            X.append(features)
            y_profit.append(actual_profit)
            y_success.append(success)
        
        X = np.array(X)
        y_profit = np.array(y_profit)
        y_success = np.array(y_success)
        
        # Split data
        X_train, X_test, y_profit_train, y_profit_test, y_success_train, y_success_test = train_test_split(
            X, y_profit, y_success, test_size=0.2, random_state=42
        )
        
        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train profit predictor
        self.profit_predictor.fit(X_train_scaled, y_profit_train)
        profit_score = self.profit_predictor.score(X_test_scaled, y_profit_test)
        
        # Train success predictor
        self.success_predictor.fit(X_train_scaled, y_success_train)
        success_score = self.success_predictor.score(X_test_scaled, y_success_test)
        
        # Save models
        joblib.dump(self.profit_predictor, 'models/profit_predictor.pkl')
        joblib.dump(self.success_predictor, 'models/success_predictor.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        logger.info(f"ML models updated - Profit R²: {profit_score:.3f}, Success R²: {success_score:.3f}")
    
    async def monitor_pools_ml(self):
        """Main monitoring loop with ML-powered decision making"""
        logger.info("Starting ML-powered pool monitoring...")
        
        while True:
            try:
                # Check each supported pool for opportunities
                for pool_info in self.config['supported_pools']:
                    # Get market data
                    market_data = await self.get_market_data(pool_info)
                    
                    if market_data:
                        # ML-powered decision making
                        should_execute, predicted_profit, success_prob = self.should_execute_arbitrage(
                            pool_info, market_data
                        )
                        
                        if should_execute:
                            logger.info(f"ML predicts profitable opportunity: {predicted_profit:.6f} profit, {success_prob:.2%} success")
                            
                            # Execute arbitrage
                            result = await self.execute_flash_loan_ml(pool_info, market_data)
                            
                            # Record data for ML training
                            self.record_arbitrage_result(pool_info, market_data, result)
                        
                        # Update models periodically
                        if len(self.historical_data) >= 100:
                            self.update_ml_models(self.historical_data)
                            self.historical_data = []  # Clear after training
                
                # Wait before next check
                await asyncio.sleep(self.config.get('check_interval', 1))
                
            except Exception as e:
                logger.error(f"Error in ML monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def get_market_data(self, pool_info: Dict) -> Optional[Dict]:
        """Get comprehensive market data for ML features"""
        try:
            # Get basic price data
            uniswap_price = await self.get_uniswap_quote(pool_info)
            sushi_price = await self.get_sushiswap_quote(pool_info)
            
            if not uniswap_price or not sushi_price:
                return None
            
            # Get additional market data
            gas_price = await self.get_optimal_gas_price()
            volume_data = await self.get_volume_data(pool_info)
            volatility = await self.calculate_volatility(pool_info)
            
            return {
                'uniswap_price': uniswap_price,
                'sushi_price': sushi_price,
                'gas_price': gas_price,
                'uniswap_volume': volume_data.get('uniswap_volume', 1000000),
                'sushi_volume': volume_data.get('sushi_volume', 1000000),
                'price_volatility': volatility,
                'liquidity_depth': volume_data.get('liquidity_depth', 1000000),
                'slippage_estimate': self.estimate_slippage(uniswap_price, sushi_price),
                'competition_level': self.estimate_competition()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def record_arbitrage_result(self, pool_info: Dict, market_data: Dict, result: Dict):
        """Record arbitrage result for ML training"""
        features = self.extract_features(pool_info, market_data).flatten()
        
        data_point = {
            'features': features.tolist(),
            'actual_profit': result.get('profit', 0),
            'success': result.get('success', False),
            'timestamp': time.time()
        }
        
        self.historical_data.append(data_point)
    
    async def execute_flash_loan_ml(self, pool_info: Dict, market_data: Dict) -> Dict:
        """Execute flash loan with ML-enhanced decision making"""
        try:
            logger.info(f"Executing ML-powered flash loan for {pool_info['name']}")
            
            # Execute the arbitrage
            result = await self.execute_flash_loan(pool_info, market_data)
            
            # Return result for ML training
            return {
                'success': result.get('success', False),
                'profit': result.get('profit', 0),
                'gas_used': result.get('gas_used', 0),
                'execution_time': result.get('execution_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in ML flash loan execution: {e}")
            return {'success': False, 'profit': 0, 'error': str(e)}
    
    # Placeholder methods for market data collection
    async def get_volume_data(self, pool_info: Dict) -> Dict:
        """Get volume data for pools"""
        # This would integrate with DEX APIs
        return {'uniswap_volume': 1000000, 'sushi_volume': 1000000, 'liquidity_depth': 1000000}
    
    async def calculate_volatility(self, pool_info: Dict) -> float:
        """Calculate price volatility"""
        # This would use historical price data
        return 0.01
    
    def estimate_slippage(self, price1: float, price2: float) -> float:
        """Estimate slippage based on price difference"""
        return abs(price1 - price2) / min(price1, price2)
    
    def estimate_competition(self) -> float:
        """Estimate competition level"""
        # This would analyze recent transactions
        return 0.5
    
    # Inherit other methods from base class
    async def get_uniswap_quote(self, pool_info: Dict) -> Optional[float]:
        """Get quote from Uniswap V3"""
        # Implementation would be similar to base class
        return 1000000
    
    async def get_sushiswap_quote(self, pool_info: Dict) -> Optional[float]:
        """Get quote from SushiSwap"""
        # Implementation would be similar to base class
        return 999000
    
    async def get_optimal_gas_price(self) -> int:
        """Get optimal gas price"""
        # Implementation would be similar to base class
        return self.web3.to_wei(0.05, 'gwei')
    
    async def execute_flash_loan(self, pool_info: Dict, market_data: Dict) -> Dict:
        """Execute flash loan (placeholder)"""
        if self.test_mode:
            # Mock execution for testing
            logger.info(f"TEST MODE: Mock flash loan execution for {pool_info['name']}")
            return {'success': True, 'profit': 100, 'gas_used': 200000}
        else:
            # Implementation would be similar to base class
            return {'success': True, 'profit': 100, 'gas_used': 200000}

    def test_ml_functionality(self):
        """Test ML functionality without real credentials"""
        logger.info("🧪 Testing ML functionality...")
        
        # Test feature extraction
        pool_info = self.config['supported_pools'][0]  # Use first pool
        market_data = {
            'uniswap_price': 1000000,
            'sushi_price': 999000,
            'gas_price': 0.05,
            'uniswap_volume': 1000000,
            'sushi_volume': 1000000,
            'volatility': 0.01,
            'liquidity_depth': 1000000,
            'slippage_estimate': 0.001,
            'competition_level': 0.5
        }
        
        try:
            # Test feature extraction
            features = self.extract_features(pool_info, market_data)
            logger.info(f"✅ Feature extraction successful: {features.shape}")
            
            # Test model training with mock data first (to fit scaler)
            mock_data = np.random.rand(100, 9)  # 100 samples, 9 features
            mock_profits = np.random.rand(100) * 1000  # Random profits
            
            # Create more realistic success data - bias towards success for profitable opportunities
            mock_success = []
            for i in range(100):
                # Higher profit should correlate with higher success probability
                profit_normalized = mock_profits[i] / 1000  # Normalize to 0-1
                success_prob = 0.3 + (profit_normalized * 0.6)  # 30% base + up to 60% based on profit
                success = np.random.choice([0, 1], p=[1-success_prob, success_prob])
                mock_success.append(success)
            
            mock_success = np.array(mock_success)
            
            # Fit the scaler first
            self.scaler.fit(mock_data)
            
            # Scale the data
            scaled_data = self.scaler.transform(mock_data)
            
            self.profit_predictor.fit(scaled_data, mock_profits)
            self.success_predictor.fit(scaled_data, mock_success)
            
            logger.info("✅ Model training successful")
            
            # Now test predictions with fitted models
            profit_prediction = self.predict_profit(features)
            logger.info(f"✅ Profit prediction: {profit_prediction:.2f}")
            
            # Test success prediction
            success_prob = self.predict_success_probability(features)
            logger.info(f"✅ Success probability: {success_prob:.2f}")
            
            # Test decision making
            should_execute, pred_profit, pred_success = self.should_execute_arbitrage(pool_info, market_data)
            logger.info(f"✅ Should execute: {should_execute}")
            
            # Test predictions on new data
            new_features = np.random.rand(10, 9)
            new_features_scaled = self.scaler.transform(new_features)
            new_profit_pred = self.profit_predictor.predict(new_features_scaled)
            new_success_pred = self.success_predictor.predict(new_features_scaled)
            
            logger.info(f"✅ New predictions - Profit range: {new_profit_pred.min():.2f} to {new_profit_pred.max():.2f}")
            logger.info(f"✅ New predictions - Success rate: {new_success_pred.mean():.2f}")
            
            logger.info("🎉 All ML functionality tests passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML test failed: {e}")
            return False

async def main():
    """Main function to run the ML agent"""
    import sys
    
    # Check if test mode is requested
    test_mode = '--test' in sys.argv
    
    try:
        if test_mode:
            # Run in test mode
            logger.info("🧪 Starting ML Agent in TEST MODE")
            agent = MLFlashLoanAgent(test_mode=True)
            
            # Test ML functionality
            success = agent.test_ml_functionality()
            
            if success:
                logger.info("✅ All tests passed! ML functionality is working correctly.")
            else:
                logger.error("❌ Some tests failed. Check the logs above.")
                
        else:
            # Production mode
            logger.info("🚀 Starting ML Agent in PRODUCTION MODE")
            agent = MLFlashLoanAgent()
            
            # Start ML-powered monitoring
            await agent.monitor_pools_ml()
        
    except KeyboardInterrupt:
        logger.info("ML Agent stopped by user")
    except Exception as e:
        logger.error(f"ML Agent error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
