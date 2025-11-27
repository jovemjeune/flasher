// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/PersonalFlashLoanArbitrage.sol";

/**
 * @title AddPools
 * @dev Script to add supported pools to the AI flash loan contract
 */
contract AddPools is Script {
    // Arbitrum One addresses
    address public constant WETH = 0x82aF49447D8a07e3bd95BD0d56f35241523fBab1;
    address public constant USDC = 0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8;
    address public constant USDT = 0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9;
    address public constant UNI = 0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0;
    address public constant DAI = 0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1;
    
    // Pool addresses - VERIFIED on Arbitrum One
    // Only WETH-USDC pool is confirmed to exist
    address public constant WETH_USDC_POOL = 0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443; // ✅ VERIFIED - 0.3% fee
    
    // TODO: Find correct addresses for these pools
    // These addresses were found to be INVALID on Arbitrum One
    address public constant WETH_USDT_POOL = 0x0000000000000000000000000000000000000000; // ❌ INVALID - needs correct address
    address public constant UNI_USDC_POOL = 0x0000000000000000000000000000000000000000; // ❌ INVALID - needs correct address
    address public constant UNI_USDT_POOL = 0x0000000000000000000000000000000000000000; // ❌ INVALID - needs correct address
    address public constant WETH_DAI_POOL = 0x0000000000000000000000000000000000000000; // ❌ INVALID - needs correct address
    address public constant UNI_DAI_POOL = 0x0000000000000000000000000000000000000000; // ❌ INVALID - needs correct address
    
    function run() external {
        address payable contractAddress = payable(vm.envAddress("CONTRACT_ADDRESS"));
        
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
        
        PersonalFlashLoanArbitrage flashLoan = PersonalFlashLoanArbitrage(contractAddress);
        
        // Add only verified pools
        console.log("Adding WETH-USDC pool (VERIFIED)...");
        flashLoan.addPool(WETH_USDC_POOL, WETH, USDC, 3000);
        
        // TODO: Uncomment these when correct addresses are found
        // console.log("Adding WETH-USDT pool...");
        // flashLoan.addPool(WETH_USDT_POOL, WETH, USDT, 3000);
        
        // console.log("Adding UNI-USDC pool...");
        // flashLoan.addPool(UNI_USDC_POOL, UNI, USDC, 3000);
        
        // console.log("Adding UNI-USDT pool...");
        // flashLoan.addPool(UNI_USDT_POOL, UNI, USDT, 3000);
        
        // console.log("Adding WETH-DAI pool...");
        // flashLoan.addPool(WETH_DAI_POOL, WETH, DAI, 3000);
        
        // console.log("Adding UNI-DAI pool...");
        // flashLoan.addPool(UNI_DAI_POOL, UNI, DAI, 3000);
        
        console.log("All pools added successfully!");
        
        // Verify pools were added
        address[] memory pools = flashLoan.getSupportedPools();
        console.log("Total supported pools:", pools.length);
        
        vm.stopBroadcast();
    }
}
