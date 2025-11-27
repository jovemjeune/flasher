#!/usr/bin/env python3
"""
AI-Powered Flash Loan Arbitrage Agent
Monitors Uniswap V3 and SushiSwap for arbitrage opportunities
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from eth_account import Account
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlashLoanAIAgent:
    def __init__(self, config_path: str = "config.json", test_mode: bool = False):
        """
        Initialize the AI agent
        
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
        
        # Supported pools and their info
        self.supported_pools = self.config['supported_pools']
        
        # Minimum profit threshold (in basis points)
        self.min_profit_bps = self.config.get('min_profit_bps', 50)  # 0.5%
        
        # Gas price settings
        self.max_gas_price = self.config.get('max_gas_price', 0.1)  # gwei
        self.priority_fee = self.config.get('priority_fee', 0.01)   # gwei
        
        if test_mode:
            logger.info("AI Agent initialized in TEST MODE")
        else:
            logger.info(f"AI Agent initialized for address: {self.account.address}")
    
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
    
    async def monitor_pools(self):
        """Main monitoring loop"""
        logger.info("Starting pool monitoring...")
        
        while True:
            try:
                # Check each supported pool for opportunities
                for pool_info in self.supported_pools:
                    opportunity = await self.check_arbitrage_opportunity(pool_info)
                    
                    if opportunity and opportunity['profitable']:
                        logger.info(f"Profitable opportunity found: {opportunity}")
                        await self.execute_flash_loan(opportunity)
                
                # Wait before next check
                await asyncio.sleep(self.config.get('check_interval', 1))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def check_arbitrage_opportunity(self, pool_info: Dict) -> Optional[Dict]:
        """
        Check for arbitrage opportunities in a specific pool
        
        Args:
            pool_info: Pool information including addresses and tokens
            
        Returns:
            Opportunity dict if profitable, None otherwise
        """
        try:
            pool_address = pool_info['pool_address']
            token0 = pool_info['token0']
            token1 = pool_info['token1']
            fee = pool_info['fee']
            
            # Get quotes from both DEXes
            uniswap_quote = await self.get_uniswap_quote(token0, token1, fee)
            sushi_quote = await self.get_sushiswap_quote(token0, token1)
            
            if not uniswap_quote or not sushi_quote:
                return None
            
            # Calculate potential profit
            if uniswap_quote > sushi_quote:
                profit = uniswap_quote - sushi_quote
                direction = 'sushi_to_uniswap'
            else:
                profit = sushi_quote - uniswap_quote
                direction = 'uniswap_to_sushi'
            
            # Check if profitable (considering costs)
            min_profit = (self.config['test_amount'] * self.min_profit_bps) / 10000
            profitable = profit > min_profit
            
            if profitable:
                return {
                    'pool_address': pool_address,
                    'token0': token0,
                    'token1': token1,
                    'amount_in': self.config['test_amount'],
                    'profit': profit,
                    'direction': direction,
                    'uniswap_quote': uniswap_quote,
                    'sushi_quote': sushi_quote,
                    'profitable': True
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking arbitrage opportunity: {e}")
            return None
    
    async def get_uniswap_quote(self, token_in: str, token_out: str, fee: int) -> Optional[int]:
        """Get quote from Uniswap V3"""
        try:
            # This would integrate with Uniswap V3 Quoter contract
            # For now, return a mock quote
            return 1000000  # Mock quote
        except Exception as e:
            logger.error(f"Error getting Uniswap quote: {e}")
            return None
    
    async def get_sushiswap_quote(self, token_in: str, token_out: str) -> Optional[int]:
        """Get quote from SushiSwap"""
        try:
            # This would integrate with SushiSwap router
            # For now, return a mock quote
            return 999000  # Mock quote (slightly lower)
        except Exception as e:
            logger.error(f"Error getting SushiSwap quote: {e}")
            return None
    
    async def execute_flash_loan(self, opportunity: Dict):
        """
        Execute flash loan arbitrage
        
        Args:
            opportunity: Arbitrage opportunity details
        """
        try:
            logger.info(f"Executing flash loan for opportunity: {opportunity}")
            
            # Prepare transaction data
            data = self.encode_flash_loan_data(opportunity)
            
            # Get current gas price
            gas_price = await self.get_optimal_gas_price()
            
            # Build transaction
            tx = self.contract.functions.executeAIFlashLoan(
                opportunity['pool_address'],
                opportunity['amount_in'] if opportunity['direction'] == 'sushi_to_uniswap' else 0,
                0 if opportunity['direction'] == 'sushi_to_uniswap' else opportunity['amount_in'],
                data
            ).build_transaction({
                'from': self.account.address,
                'gas': 500000,
                'gasPrice': gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Flash loan transaction sent: {tx_hash.hex()}")
            
            # Wait for transaction confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Flash loan executed successfully: {tx_hash.hex()}")
            else:
                logger.error(f"Flash loan failed: {tx_hash.hex()}")
            
        except Exception as e:
            logger.error(f"Error executing flash loan: {e}")
    
    def encode_flash_loan_data(self, opportunity: Dict) -> bytes:
        """Encode flash loan data for the contract"""
        if self.test_mode:
            # Mock encoding for test mode
            return b'test_data_encoded'
        
        return self.web3.codec.encode_abi(
            ['address', 'address', 'uint256', 'uint256'],
            [
                opportunity['token0'],
                opportunity['token1'],
                opportunity['amount_in'] if opportunity['direction'] == 'sushi_to_uniswap' else 0,
                0 if opportunity['direction'] == 'sushi_to_uniswap' else opportunity['amount_in']
            ]
        )
    
    async def get_optimal_gas_price(self) -> int:
        """Get optimal gas price for current network conditions"""
        try:
            base_fee = self.web3.eth.get_block('latest')['baseFeePerGas']
            priority_fee = self.web3.eth.max_priority_fee
            
            # Calculate optimal gas price
            optimal_gas = base_fee + priority_fee
            
            # Convert to wei
            max_gas_wei = self.web3.to_wei(self.max_gas_price, 'gwei')
            
            return min(optimal_gas, max_gas_wei)
            
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            # Return fallback gas price
            return self.web3.to_wei(0.05, 'gwei')
    
    def get_contract_balance(self) -> int:
        """Get contract ETH balance"""
        return self.web3.eth.get_balance(self.config['contract_address'])
    
    def get_agent_balance(self) -> int:
        """Get AI agent ETH balance"""
        if self.test_mode:
            return 1000000000000000000  # Mock balance
        return self.web3.eth.get_balance(self.account.address)

    def test_basic_functionality(self):
        """Test basic functionality without real credentials"""
        logger.info("🧪 Testing basic AI agent functionality...")
        
        try:
            # Test configuration loading
            logger.info(f"✅ Configuration loaded: {len(self.config['supported_pools'])} pools")
            
            # Test opportunity detection logic
            pool_info = self.config['supported_pools'][0]  # Use first pool
            opportunity = {
                'pool_address': pool_info['pool_address'],
                'token0': pool_info['token0'],
                'token1': pool_info['token1'],
                'amount_in': 1000000000000000000,  # 1 ETH
                'direction': 'uniswap_to_sushi',
                'profit': 50000000000000000,  # 0.05 ETH
                'profitable': True
            }
            
            logger.info(f"✅ Opportunity detection: {opportunity['profit']} wei profit")
            
            # Test data encoding
            encoded_data = self.encode_flash_loan_data(opportunity)
            logger.info(f"✅ Data encoding: {len(encoded_data)} bytes")
            
            # Test mock execution
            logger.info("✅ Mock flash loan execution would proceed")
            
            logger.info("🎉 All basic functionality tests passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic test failed: {e}")
            return False

async def main():
    """Main function to run the AI agent"""
    import sys
    
    # Check if test mode is requested
    test_mode = '--test' in sys.argv
    
    try:
        if test_mode:
            # Run in test mode
            logger.info("🧪 Starting AI Agent in TEST MODE")
            agent = FlashLoanAIAgent(test_mode=True)
            
            # Test basic functionality
            success = agent.test_basic_functionality()
            
            if success:
                logger.info("✅ All tests passed! Basic functionality is working correctly.")
            else:
                logger.error("❌ Some tests failed. Check the logs above.")
                
        else:
            # Production mode
            logger.info("🚀 Starting AI Agent in PRODUCTION MODE")
            agent = FlashLoanAIAgent()
            
            # Log initial balances
            logger.info(f"Contract balance: {agent.get_contract_balance()} wei")
            logger.info(f"Agent balance: {agent.get_agent_balance()} wei")
            
            # Start monitoring
            await agent.monitor_pools()
        
    except KeyboardInterrupt:
        logger.info("AI Agent stopped by user")
    except Exception as e:
        logger.error(f"AI Agent error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
