// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/PersonalFlashLoanArbitrage.sol";

/**
 * @title DeployPersonalFlashLoan
 * @dev Deployment script for PersonalFlashLoanArbitrage contract
 */
contract DeployPersonalFlashLoan is Script {
    // Uniswap V3 Factory address (same across all networks)
    address public constant UNISWAP_V3_FACTORY = 0x1F98431c8aD98523631AE4a59f267346ea31F984;
    
    function run() external {
        // Get AI agent address from environment (optional)
        address aiAgent = vm.envOr("AI_AGENT_ADDRESS", address(0));
        
        // Configuration parameters
        uint256 minProfitThreshold = 0.001 ether; // 0.001 ETH minimum profit
        uint256 maxGasPrice = 0.1 ether; // 0.1 ETH maximum gas price
        
        // SECURE METHOD (Recommended): Use cast wallet import
        // 1. Import wallet: cast wallet import deployer --interactive
        // 2. Get address: cast wallet address deployer
        // 3. Run script with: --froms $(cast wallet address deployer)
        // 
        // LEGACY METHOD: Use PRIVATE_KEY env var (less secure)
        // export PRIVATE_KEY="your-key"
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0));
        if (deployerPrivateKey != 0) {
            vm.startBroadcast(deployerPrivateKey);
        } else {
            // No PRIVATE_KEY set - use account from --froms flag (cast wallet)
            vm.startBroadcast();
        }
        
        // Deploy the contract with factory address
        PersonalFlashLoanArbitrage personalFlashLoan = new PersonalFlashLoanArbitrage(
            UNISWAP_V3_FACTORY,
            aiAgent,
            minProfitThreshold,
            maxGasPrice
        );
        
        vm.stopBroadcast();
        
        // Log deployment information
        console.log("PersonalFlashLoanArbitrage deployed to:", address(personalFlashLoan));
        console.log("Factory:", address(personalFlashLoan.factory()));
        console.log("Owner:", personalFlashLoan.owner());
        console.log("AI Agent:", personalFlashLoan.aiAgent());
        console.log("Min Profit Threshold:", personalFlashLoan.minProfitThreshold());
        console.log("Max Gas Price:", personalFlashLoan.maxGasPrice());
        
        // Save deployment info
        string memory deploymentInfo = string(abi.encodePacked(
            "Deployment Information:\n",
            "Contract Address: ", vm.toString(address(personalFlashLoan)), "\n",
            "Factory: ", vm.toString(address(personalFlashLoan.factory())), "\n",
            "Owner: ", vm.toString(personalFlashLoan.owner()), "\n",
            "AI Agent: ", vm.toString(personalFlashLoan.aiAgent()), "\n",
            "Min Profit Threshold: ", vm.toString(personalFlashLoan.minProfitThreshold()), "\n",
            "Max Gas Price: ", vm.toString(personalFlashLoan.maxGasPrice()), "\n",
            "Network: ", vm.toString(block.chainid), "\n",
            "Block Number: ", vm.toString(block.number), "\n",
            "Timestamp: ", vm.toString(block.timestamp)
        ));
        
        vm.writeFile("deployment-info.txt", deploymentInfo);
        console.log("Deployment information saved to deployment-info.txt");
    }
}

