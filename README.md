# Personal Flash Loan Arbitrage Protocol

A simplified, open-source flash loan arbitrage system designed for personal use only. This protocol combines a smart contract with a machine learning agent to execute automated arbitrage operations across multiple DEXes.

## 🚀 Features

### Core Functionality
- **Owner-Only Access**: Only the contract owner can execute flash loan operations
- **ML-Powered AI Agent**: Machine learning agent for intelligent arbitrage decisions
- **Multi-Pool Support**: Support for multiple Uniswap V3 pools
- **Cross-DEX Arbitrage**: Execute arbitrage between Uniswap V3 and SushiSwap
- **Gas Optimization**: Built-in gas price controls and optimization

### Smart Contract Features
- **Pool Management**: Add/remove/update supported pools
- **Configuration**: Update AI agent, profit thresholds, and gas limits
- **Security**: Reentrancy protection, access control, emergency functions
- **Event Logging**: Comprehensive event emission for monitoring

### ML Agent Features
- **Profit Prediction**: Gradient Boosting Regressor for profit estimation
- **Success Probability**: Random Forest Regressor for success prediction
- **Feature Engineering**: 9-dimensional market data analysis
- **Online Learning**: Real-time model updates with new data
- **Decision Making**: Combines ML predictions with business rules

## 📋 Requirements

- **Solidity**: ^0.8.19
- **Foundry**: Latest version
- **OpenZeppelin**: v5.4.0
- **Python**: 3.8+ (for ML agent)
- **Node.js**: 16+ (for deployment scripts)

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd flasher

# Install Foundry dependencies
forge install OpenZeppelin/openzeppelin-contracts --no-commit

# Install Python dependencies
cd ai_agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
# Copy example files
cp ai_agent/.env.example ai_agent/.env
cp ai_agent/config.json.example ai_agent/config.json

# Edit with your configuration
nano ai_agent/.env
nano ai_agent/config.json
```

## 🚀 Deployment

### 1. Deploy Contract
```bash
# Set environment variables
export PRIVATE_KEY="your-private-key"
export AI_AGENT_ADDRESS="your-ai-agent-address"

# Deploy to Arbitrum One
forge script script/DeployPersonalFlashLoan.s.sol --rpc-url arbitrum_one --broadcast --verify
```

### 2. Add Supported Pools
```bash
# Add pools after deployment
forge script script/AddPools.s.sol --rpc-url arbitrum_one --broadcast
```

## 🤖 ML Agent Usage

### Test Mode (Safe)
```bash
cd ai_agent
source venv/bin/activate

# Test ML agent
python ml_flash_loan_agent.py --test

# Test integration with contract
python test_personal_contract_integration.py
```

### Production Mode
```bash
# Run ML agent in production
python ml_flash_loan_agent.py
```

## 🧪 Testing

### Smart Contract Tests
```bash
# Run all tests
forge test --match-contract PersonalFlashLoanArbitrageTest

# Run with gas report
forge test --match-contract PersonalFlashLoanArbitrageTest --gas-report
```

### ML Agent Tests
```bash
cd ai_agent
source venv/bin/activate

# Test ML functionality
python ml_flash_loan_agent.py --test

# Test integration
python test_personal_contract_integration.py
```

## 📊 Test Results

### Smart Contract
- **34 tests passed** ✅
- **1 test skipped** (gas price test)
- **0 tests failed** ✅

### ML Agent Integration
- **5/5 integration tests passed** ✅
- **ML predictions working** ✅
- **Contract compatibility verified** ✅
- **Multi-pool monitoring functional** ✅

## 🔒 Security

### Access Control
- Only owner can manage pools and configuration
- Only owner and AI agent can execute flash loans
- No public access to critical functions

### ML Agent Security
- Test mode prevents accidental mainnet transactions
- Environment variable protection for sensitive data
- Input validation and error handling

## 📚 Documentation

- [AI Agent Guide](docs/AI_AGENT_GUIDE.md)
- [ML Agent Technical Documentation](docs/ML_AGENT_TECHNICAL_DOCUMENTATION.md)
- [ML Agent Quick Reference](docs/ML_AGENT_QUICK_REFERENCE.md)
- [Integration Test Summary](INTEGRATION_TEST_SUMMARY.md)
- [Personal Use Guide](README_PERSONAL.md)

## 🌐 Supported Networks

### Primary: Arbitrum One
- **RPC**: `https://arb1.arbitrum.io/rpc`
- **Chain ID**: 42161
- **Benefits**: Low gas fees, fast finality

### Supported Pools
- WETH-USDC, WETH-USDT, UNI-USDC, UNI-USDT, WETH-DAI, UNI-DAI

## ⚠️ Important Disclaimers

### Personal Use Only
- This software is licensed for personal use only
- Commercial use requires separate license
- No redistribution allowed

### Risk Warning
- Flash loan arbitrage involves significant financial risks
- No guarantee of profitability
- Market conditions can change rapidly
- Use at your own risk

## 📄 License

This project is licensed under the Personal Use License. See [LICENSE](LICENSE) for details.

---

**⚠️ DISCLAIMER: This software is for personal use only. Use at your own risk.**

