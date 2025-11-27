// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "@uniswap/v3-core/contracts/interfaces/pool/IUniswapV3PoolActions.sol";
import "@uniswap/v3-core/contracts/interfaces/callback/IUniswapV3FlashCallback.sol";
import "./CallbackValidation.sol";

/**
 * @title PersonalFlashLoanArbitrage
 * @dev Simplified flash loan arbitrage contract for personal use only
 * @notice This contract allows only the owner to execute flash loan arbitrage operations
 * @author Your Name
 */
contract PersonalFlashLoanArbitrage is Ownable, ReentrancyGuard, Pausable, IUniswapV3FlashCallback {
    
    // ============ STRUCTS ============
    
    struct PoolInfo {
        address poolAddress;
        address token0;
        address token1;
        uint24 fee;
        bool isActive;
    }
    
    // ============ STATE VARIABLES ============
    
    /// @notice Uniswap V3 Factory address (immutable, same across all networks)
    /// @dev Factory address: 0x1F98431c8aD98523631AE4a59f267346ea31F984
    IUniswapV3Factory public immutable factory;
    
    /// @notice Mapping of pool addresses to pool information
    mapping(address => PoolInfo) public supportedPools;
    
    /// @notice Mapping of pool addresses to pool keys for callback validation
    mapping(address => CallbackValidation.PoolKey) private poolKeys;
    
    /// @notice Array of all supported pool addresses
    address[] public poolAddresses;
    
    /// @notice AI agent address (can be updated by owner)
    address public aiAgent;
    
    /// @notice Minimum profit threshold (in wei)
    uint256 public minProfitThreshold;
    
    /// @notice Maximum gas price for operations (in wei)
    uint256 public maxGasPrice;
    
    // ============ EVENTS ============
    
    event PoolAdded(address indexed pool, address indexed token0, address indexed token1, uint24 fee);
    event PoolRemoved(address indexed pool);
    event PoolUpdated(address indexed pool, bool isActive);
    event AIAgentUpdated(address indexed oldAgent, address indexed newAgent);
    event MinProfitThresholdUpdated(uint256 oldThreshold, uint256 newThreshold);
    event MaxGasPriceUpdated(uint256 oldGasPrice, uint256 newGasPrice);
    event FlashLoanExecuted(
        address indexed pool,
        address indexed token0,
        address indexed token1,
        uint256 amount0,
        uint256 amount1,
        uint256 profit
    );
    event EmergencyWithdraw(address indexed token, uint256 amount);
    
    // ============ MODIFIERS ============
    
    modifier onlyOwnerOrAI() {
        require(
            msg.sender == owner() || msg.sender == aiAgent,
            "PersonalFlashLoan: Only owner or AI agent"
        );
        _;
    } 
    
    modifier poolExists(address pool) {
        require(supportedPools[pool].poolAddress != address(0), "PersonalFlashLoan: Pool not supported");
        _;
    }
    
    modifier notPaused() {
        require(!paused(), "PersonalFlashLoan: Contract is paused");
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    constructor(
        address _factory,
        address _aiAgent,
        uint256 _minProfitThreshold,
        uint256 _maxGasPrice
    ) Ownable(msg.sender) {
        require(_factory != address(0), "PersonalFlashLoan: Invalid factory address");
        factory = IUniswapV3Factory(_factory);
        aiAgent = _aiAgent;
        minProfitThreshold = _minProfitThreshold;
        maxGasPrice = _maxGasPrice;
    }
    
    // ============ POOL MANAGEMENT ============
    
    /**
     * @notice Add a new supported pool (only owner)
     * @param pool The pool address
     * @param token0 First token address
     * @param token1 Second token address
     * @param fee Pool fee (e.g., 3000 for 0.3%)
     * @dev Validates that the pool exists in the Uniswap V3 factory
     */
    function addPool(
        address pool,
        address token0,
        address token1,
        uint24 fee
    ) external onlyOwner {
        require(pool != address(0), "PersonalFlashLoan: Invalid pool address");
        require(token0 != address(0) && token1 != address(0), "PersonalFlashLoan: Invalid token addresses");
        require(token0 != token1, "PersonalFlashLoan: Token addresses must be different");
        require(supportedPools[pool].poolAddress == address(0), "PersonalFlashLoan: Pool already exists");
        
        // Verify pool exists in Uniswap V3 factory
        address verifiedPool = factory.getPool(token0, token1, fee);
        require(verifiedPool == pool && verifiedPool != address(0), "PersonalFlashLoan: Pool not found in factory");
        
        // Sort tokens to match Uniswap's token0 < token1 convention
        (address sortedToken0, address sortedToken1) = token0 < token1 ? (token0, token1) : (token1, token0);
        
        // Create pool key for callback validation
        CallbackValidation.PoolKey memory poolKey = CallbackValidation.getPoolKey(token0, token1, fee);
        poolKeys[pool] = poolKey;
        
        supportedPools[pool] = PoolInfo({
            poolAddress: pool,
            token0: sortedToken0,
            token1: sortedToken1,
            fee: fee,
            isActive: true
        });
        
        poolAddresses.push(pool);
        
        emit PoolAdded(pool, sortedToken0, sortedToken1, fee);
    }
    
    /**
     * @notice Remove a supported pool (only owner)
     * @param pool The pool address to remove
     */
    function removePool(address pool) external onlyOwner poolExists(pool) {
        // Mark as inactive instead of deleting to preserve history
        supportedPools[pool].isActive = false;
        
        emit PoolRemoved(pool);
    }
    
    /**
     * @notice Update pool status (only owner)
     * @param pool The pool address
     * @param isActive New active status
     */
    function updatePoolStatus(address pool, bool isActive) external onlyOwner poolExists(pool) {
        supportedPools[pool].isActive = isActive;
        
        emit PoolUpdated(pool, isActive);
    }
    
    // ============ CONFIGURATION ============
    
    /**
     * @notice Update AI agent address (only owner)
     * @param newAgent New AI agent address
     */
    function updateAIAgent(address newAgent) external onlyOwner {
        require(newAgent != address(0), "PersonalFlashLoan: Invalid AI agent address");
        
        address oldAgent = aiAgent;
        aiAgent = newAgent;
        
        emit AIAgentUpdated(oldAgent, newAgent);
    }
    
    /**
     * @notice Update minimum profit threshold (only owner)
     * @param newThreshold New minimum profit threshold
     */
    function updateMinProfitThreshold(uint256 newThreshold) external onlyOwner {
        uint256 oldThreshold = minProfitThreshold;
        minProfitThreshold = newThreshold;
        
        emit MinProfitThresholdUpdated(oldThreshold, newThreshold);
    }
    
    /**
     * @notice Update maximum gas price (only owner)
     * @param newGasPrice New maximum gas price
     */
    function updateMaxGasPrice(uint256 newGasPrice) external onlyOwner {
        uint256 oldGasPrice = maxGasPrice;
        maxGasPrice = newGasPrice;
        
        emit MaxGasPriceUpdated(oldGasPrice, newGasPrice);
    }
    
    // ============ FLASH LOAN EXECUTION ============
    
    /**
     * @notice Execute flash loan arbitrage (only owner or AI agent)
     * @param pool The pool address to execute flash loan on
     * @param amount0 Amount of token0 to borrow (0 if not borrowing)
     * @param amount1 Amount of token1 to borrow (0 if not borrowing)
     * @param data Encoded data for the flash loan callback
     */
    function executeFlashLoan(
        address pool,
        uint256 amount0,
        uint256 amount1,
        bytes calldata data
    ) external onlyOwnerOrAI nonReentrant notPaused poolExists(pool) {
        require(supportedPools[pool].isActive, "PersonalFlashLoan: Pool is inactive");
        require(tx.gasprice <= maxGasPrice, "PersonalFlashLoan: Gas price too high");
        require(amount0 > 0 || amount1 > 0, "PersonalFlashLoan: Must borrow at least one token");
        
        // Execute the flash loan
        _executeFlashLoan(pool, amount0, amount1, data);
    }
    
    /**
     * @notice Internal function to execute flash loan
     * @param pool The pool address
     * @param amount0 Amount of token0 to borrow
     * @param amount1 Amount of token1 to borrow
     * @param data Encoded data for callback
     */
    function _executeFlashLoan(
        address pool,
        uint256 amount0,
        uint256 amount1,
        bytes calldata data
    ) internal {
        // Use proper interface for type safety
        IUniswapV3PoolActions(pool).flash(
            address(this),
            amount0,
            amount1,
            data
        );
    }
    
    /**
     * @notice Uniswap V3 flash loan callback
     * @param fee0 The fee for token0
     * @param fee1 The fee for token1
     * @param data Encoded data from the flash loan call
     * @dev CRITICAL: Verifies caller is a legitimate Uniswap V3 pool from the factory
     */
    function uniswapV3FlashCallback(
        uint256 fee0,
        uint256 fee1,
        bytes calldata data
    ) external override {
        // Get pool key from storage
        CallbackValidation.PoolKey memory poolKey = poolKeys[msg.sender];
        require(poolKey.token0 != address(0), "PersonalFlashLoan: Pool key not found");
        
        // CRITICAL: Verify the caller is a legitimate Uniswap V3 pool from the factory
        CallbackValidation.verifyCallback(address(factory), poolKey);
        
        // Verify pool is still supported and active
        PoolInfo memory poolInfo = supportedPools[msg.sender];
        require(poolInfo.poolAddress != address(0), "PersonalFlashLoan: Pool not supported");
        require(poolInfo.isActive, "PersonalFlashLoan: Pool is inactive");
        
        // Decode the data to get the amounts borrowed
        (uint256 amount0, uint256 amount1) = abi.decode(data, (uint256, uint256));
        
        // Verify actual token balances match expected amounts (prevent manipulation)
        if (amount0 > 0) {
            uint256 balance0 = IERC20(poolInfo.token0).balanceOf(address(this));
            require(balance0 >= amount0, "PersonalFlashLoan: Insufficient token0 received");
        }
        
        if (amount1 > 0) {
            uint256 balance1 = IERC20(poolInfo.token1).balanceOf(address(this));
            require(balance1 >= amount1, "PersonalFlashLoan: Insufficient token1 received");
        }
        
        // Execute arbitrage logic here
        // For now, we'll just repay the loan with fees
        // In a real implementation, you would:
        // 1. Swap tokens on different DEXes
        // 2. Calculate profit
        // 3. Repay the loan + fees
        // 4. Keep the profit
        
        // Repay token0 if borrowed
        if (amount0 > 0) {
            _safeTransfer(poolInfo.token0, msg.sender, amount0 + fee0);
        }
        
        // Repay token1 if borrowed
        if (amount1 > 0) {
            _safeTransfer(poolInfo.token1, msg.sender, amount1 + fee1);
        }
        
        // Emit event (profit would be calculated in real implementation)
        emit FlashLoanExecuted(msg.sender, poolInfo.token0, poolInfo.token1, amount0, amount1, 0);
    }
    
    // ============ UTILITY FUNCTIONS ============
    
    /**
     * @notice Get pool information
     * @param pool The pool address
     * @return PoolInfo struct
     */
    function getPoolInfo(address pool) external view returns (PoolInfo memory) {
        return supportedPools[pool];
    }
    
    /**
     * @notice Get all supported pool addresses
     * @return Array of pool addresses
     */
    function getSupportedPools() external view returns (address[] memory) {
        return poolAddresses;
    }
    
    /**
     * @notice Get number of supported pools
     * @return Number of pools
     */
    function getPoolCount() external view returns (uint256) {
        return poolAddresses.length;
    }
    
    /**
     * @notice Check if a pool is supported and active
     * @param pool The pool address
     * @return True if pool is supported and active
     */
    function isPoolActive(address pool) external view returns (bool) {
        return supportedPools[pool].poolAddress != address(0) && supportedPools[pool].isActive;
    }
    
    // ============ EMERGENCY FUNCTIONS ============
    
    /**
     * @notice Pause the contract (only owner)
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @notice Unpause the contract (only owner)
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @notice Emergency withdraw tokens (only owner)
     * @param token Token address (address(0) for ETH)
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner nonReentrant {
        if (token == address(0)) {
            // Withdraw ETH using call() instead of transfer() for safety
            require(address(this).balance >= amount, "PersonalFlashLoan: Insufficient ETH balance");
            (bool success, ) = payable(owner()).call{value: amount}("");
            require(success, "PersonalFlashLoan: ETH transfer failed");
        } else {
            // Withdraw ERC20 token
            _safeTransfer(token, owner(), amount);
        }
        
        emit EmergencyWithdraw(token, amount);
    }
    
    // ============ INTERNAL FUNCTIONS ============
    
    /**
     * @notice Safe token transfer
     * @param token Token address
     * @param to Recipient address
     * @param amount Amount to transfer
     */
    function _safeTransfer(address token, address to, uint256 amount) internal {
        (bool success, bytes memory data) = token.call(
            abi.encodeWithSignature("transfer(address,uint256)", to, amount)
        );
        
        require(success && (data.length == 0 || abi.decode(data, (bool))), "PersonalFlashLoan: Transfer failed");
    }
    
    /**
     * @notice Safe token approval
     * @param token Token address
     * @param spender Spender address
     * @param amount Amount to approve
     */
    function _safeApprove(address token, address spender, uint256 amount) internal {
        (bool success, bytes memory data) = token.call(
            abi.encodeWithSignature("approve(address,uint256)", spender, amount)
        );
        
        require(success && (data.length == 0 || abi.decode(data, (bool))), "PersonalFlashLoan: Approval failed");
    }
    
    // ============ RECEIVE FUNCTION ============
    
    /**
     * @notice Allow contract to receive ETH
     */
    receive() external payable {}
}
