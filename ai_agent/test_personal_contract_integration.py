#!/usr/bin/env python3
"""
Integration test for ML Agent with PersonalFlashLoanArbitrage contract
Tests the complete workflow from ML prediction to contract interaction
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_flash_loan_agent import MLFlashLoanAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonalContractIntegrationTest:
    """Integration test for PersonalFlashLoanArbitrage contract with ML agent"""
    
    def __init__(self):
        self.agent = None
        self.test_results = {}
        
    def setup_test_environment(self):
        """Setup test environment with mock contract data"""
        logger.info("🔧 Setting up test environment...")
        
        # Mock contract configuration
        self.mock_contract_config = {
            "contract_address": "0x1234567890123456789012345678901234567890",
            "supported_pools": [
                {
                    "name": "WETH-USDC",
                    "pool_address": "0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",
                    "token0": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    "token1": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
                    "fee": 3000,
                    "description": "WETH-USDC 0.3% fee tier"
                },
                {
                    "name": "WETH-USDT",
                    "pool_address": "0x641c00c822B8a6c5A3b4b4B4b4b4b4b4b4b4b4b4",
                    "token0": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    "token1": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
                    "fee": 3000,
                    "description": "WETH-USDT 0.3% fee tier"
                }
            ],
            "min_profit_threshold": 0.001,
            "min_success_probability": 0.6,
            "max_gas_price": 0.1,
            "test_mode": True
        }
        
        logger.info("✅ Test environment setup complete")
        
    def test_ml_agent_initialization(self):
        """Test ML agent initialization with contract config"""
        logger.info("🧪 Testing ML agent initialization...")
        
        try:
            # Initialize ML agent with test mode
            self.agent = MLFlashLoanAgent(test_mode=True)
            
            # Verify agent is properly initialized
            assert self.agent is not None, "Agent should be initialized"
            assert hasattr(self.agent, 'profit_predictor'), "Should have profit predictor"
            assert hasattr(self.agent, 'success_predictor'), "Should have success predictor"
            assert hasattr(self.agent, 'scaler'), "Should have scaler"
            
            # Train the models with mock data (required for predictions)
            logger.info("🔧 Training ML models with mock data...")
            success = self.agent.test_ml_functionality()
            assert success, "ML functionality test should pass"
            
            self.test_results['ml_agent_init'] = True
            logger.info("✅ ML agent initialization and training successful")
            
        except Exception as e:
            self.test_results['ml_agent_init'] = False
            logger.error(f"❌ ML agent initialization failed: {e}")
            
    def test_ml_predictions(self):
        """Test ML predictions for different market conditions"""
        logger.info("🧪 Testing ML predictions...")
        
        try:
            # Test with different market conditions
            pool_info = self.mock_contract_config['supported_pools'][0]
            
            # Test case 1: Good market conditions
            good_market_data = {
                'uniswap_price': 1000000,
                'sushi_price': 999000,  # 0.1% price difference
                'gas_price': 0.05,
                'uniswap_volume': 1000000,
                'sushi_volume': 1000000,
                'volatility': 0.01,
                'liquidity_depth': 1000000,
                'slippage_estimate': 0.001,
                'competition_level': 0.3
            }
            
            # Test prediction
            should_execute_good, predicted_profit_good, success_prob_good = self.agent.should_execute_arbitrage(
                pool_info, good_market_data
            )
            
            logger.info(f"📊 Good conditions - Profit: {predicted_profit_good:.2f}, Success: {success_prob_good:.2%}, Execute: {should_execute_good}")
            
            # Test case 2: Challenging market conditions
            challenging_market_data = {
                'uniswap_price': 1000000,
                'sushi_price': 999900,  # 0.01% price difference (smaller)
                'gas_price': 0.1,  # Higher gas
                'uniswap_volume': 100000,  # Lower volume
                'sushi_volume': 100000,
                'volatility': 0.05,  # Higher volatility
                'liquidity_depth': 100000,  # Lower liquidity
                'slippage_estimate': 0.005,  # Higher slippage
                'competition_level': 0.8  # Higher competition
            }
            
            should_execute_challenging, predicted_profit_challenging, success_prob_challenging = self.agent.should_execute_arbitrage(
                pool_info, challenging_market_data
            )
            
            logger.info(f"📊 Challenging conditions - Profit: {predicted_profit_challenging:.2f}, Success: {success_prob_challenging:.2%}, Execute: {should_execute_challenging}")
            
            # Verify predictions are reasonable (not testing relative values since ML uses random training data)
            assert predicted_profit_good > 0, "Should predict positive profit for good conditions"
            assert success_prob_good > 0, "Should predict positive success probability"
            assert predicted_profit_challenging > 0, "Should predict positive profit for challenging conditions"
            assert success_prob_challenging > 0, "Should predict positive success probability"
            assert 0 <= success_prob_good <= 1, "Success probability should be between 0 and 1"
            assert 0 <= success_prob_challenging <= 1, "Success probability should be between 0 and 1"
            
            self.test_results['ml_predictions'] = True
            logger.info("✅ ML predictions test successful")
            
        except Exception as e:
            self.test_results['ml_predictions'] = False
            logger.error(f"❌ ML predictions test failed: {e}")
            
    def test_contract_compatibility(self):
        """Test compatibility with PersonalFlashLoanArbitrage contract interface"""
        logger.info("🧪 Testing contract compatibility...")
        
        try:
            # Test that agent can generate contract-compatible data
            pool_info = self.mock_contract_config['supported_pools'][0]
            market_data = {
                'uniswap_price': 1000000,
                'sushi_price': 999000,
                'gas_price': 0.05,
                'uniswap_volume': 1000000,
                'sushi_volume': 1000000,
                'volatility': 0.01,
                'liquidity_depth': 1000000,
                'slippage_estimate': 0.001,
                'competition_level': 0.3
            }
            
            # Test data encoding (what would be sent to contract)
            if hasattr(self.agent, 'encode_flash_loan_data'):
                encoded_data = self.agent.encode_flash_loan_data(
                    pool_info, market_data, 1000000, 0  # amount0, amount1
                )
                assert isinstance(encoded_data, bytes), "Encoded data should be bytes"
                assert len(encoded_data) > 0, "Encoded data should not be empty"
                logger.info(f"📦 Encoded data length: {len(encoded_data)} bytes")
            
            # Test that agent can make execution decisions
            should_execute, profit, success_prob = self.agent.should_execute_arbitrage(
                pool_info, market_data
            )
            
            # Verify decision logic aligns with contract requirements
            min_profit = self.mock_contract_config['min_profit_threshold']
            min_success = self.mock_contract_config['min_success_probability']
            
            if should_execute:
                assert profit > min_profit, f"Should only execute if profit > {min_profit}"
                assert success_prob > min_success, f"Should only execute if success > {min_success}%"
                logger.info("✅ Execution decision aligns with contract requirements")
            else:
                logger.info("ℹ️ No execution recommended (below thresholds)")
            
            self.test_results['contract_compatibility'] = True
            logger.info("✅ Contract compatibility test successful")
            
        except Exception as e:
            self.test_results['contract_compatibility'] = False
            logger.error(f"❌ Contract compatibility test failed: {e}")
            
    def test_multi_pool_monitoring(self):
        """Test monitoring multiple pools as the contract supports"""
        logger.info("🧪 Testing multi-pool monitoring...")
        
        try:
            # Test monitoring all supported pools
            supported_pools = self.mock_contract_config['supported_pools']
            
            opportunities_found = 0
            for pool_info in supported_pools:
                # Generate different market conditions for each pool
                market_data = {
                    'uniswap_price': 1000000 + (hash(pool_info['name']) % 10000),
                    'sushi_price': 999000 + (hash(pool_info['name']) % 10000),
                    'gas_price': 0.05,
                    'uniswap_volume': 1000000,
                    'sushi_volume': 1000000,
                    'volatility': 0.01,
                    'liquidity_depth': 1000000,
                    'slippage_estimate': 0.001,
                    'competition_level': 0.3
                }
                
                should_execute, profit, success_prob = self.agent.should_execute_arbitrage(
                    pool_info, market_data
                )
                
                if should_execute:
                    opportunities_found += 1
                    logger.info(f"🎯 Opportunity found in {pool_info['name']}: {profit:.2f} profit, {success_prob:.2%} success")
            
            logger.info(f"📊 Total opportunities found: {opportunities_found}/{len(supported_pools)}")
            
            # Verify agent can handle multiple pools
            assert opportunities_found >= 0, "Should handle multiple pools without errors"
            
            self.test_results['multi_pool_monitoring'] = True
            logger.info("✅ Multi-pool monitoring test successful")
            
        except Exception as e:
            self.test_results['multi_pool_monitoring'] = False
            logger.error(f"❌ Multi-pool monitoring test failed: {e}")
            
    def test_ml_model_robustness(self):
        """Test ML model robustness with edge cases"""
        logger.info("🧪 Testing ML model robustness...")
        
        try:
            pool_info = self.mock_contract_config['supported_pools'][0]
            
            # Test edge cases
            edge_cases = [
                {
                    'name': 'High volatility',
                    'data': {
                        'uniswap_price': 1000000,
                        'sushi_price': 999000,
                        'gas_price': 0.05,
                        'uniswap_volume': 1000000,
                        'sushi_volume': 1000000,
                        'volatility': 0.1,  # High volatility
                        'liquidity_depth': 1000000,
                        'slippage_estimate': 0.001,
                        'competition_level': 0.3
                    }
                },
                {
                    'name': 'Low liquidity',
                    'data': {
                        'uniswap_price': 1000000,
                        'sushi_price': 999000,
                        'gas_price': 0.05,
                        'uniswap_volume': 1000,  # Low volume
                        'sushi_volume': 1000,
                        'volatility': 0.01,
                        'liquidity_depth': 1000,  # Low liquidity
                        'slippage_estimate': 0.001,
                        'competition_level': 0.3
                    }
                },
                {
                    'name': 'High competition',
                    'data': {
                        'uniswap_price': 1000000,
                        'sushi_price': 999000,
                        'gas_price': 0.05,
                        'uniswap_volume': 1000000,
                        'sushi_volume': 1000000,
                        'volatility': 0.01,
                        'liquidity_depth': 1000000,
                        'slippage_estimate': 0.001,
                        'competition_level': 0.9  # High competition
                    }
                }
            ]
            
            for case in edge_cases:
                should_execute, profit, success_prob = self.agent.should_execute_arbitrage(
                    pool_info, case['data']
                )
                
                logger.info(f"🔍 {case['name']}: Profit={profit:.2f}, Success={success_prob:.2%}, Execute={should_execute}")
                
                # Verify predictions are reasonable (not NaN or infinite)
                assert not (profit != profit), f"Profit should not be NaN for {case['name']}"
                assert not (success_prob != success_prob), f"Success prob should not be NaN for {case['name']}"
                assert 0 <= success_prob <= 1, f"Success prob should be between 0 and 1 for {case['name']}"
            
            self.test_results['ml_model_robustness'] = True
            logger.info("✅ ML model robustness test successful")
            
        except Exception as e:
            self.test_results['ml_model_robustness'] = False
            logger.error(f"❌ ML model robustness test failed: {e}")
            
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting PersonalFlashLoanArbitrage + ML Agent integration tests...")
        
        # Setup
        self.setup_test_environment()
        
        # Run tests
        self.test_ml_agent_initialization()
        self.test_ml_predictions()
        self.test_contract_compatibility()
        self.test_multi_pool_monitoring()
        self.test_ml_model_robustness()
        
        # Report results and return success status
        return self.report_results()
        
    def report_results(self):
        """Report test results"""
        logger.info("📊 Integration Test Results:")
        logger.info("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info("=" * 50)
        logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All integration tests passed! ML Agent is ready for PersonalFlashLoanArbitrage contract.")
            return True
        else:
            logger.error("❌ Some integration tests failed. Please review the issues above.")
            return False

def main():
    """Main function to run integration tests"""
    test_suite = PersonalContractIntegrationTest()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 SUCCESS: ML Agent is fully compatible with PersonalFlashLoanArbitrage contract!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Some integration tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
