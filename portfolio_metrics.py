

def MVO(weights, cov_matrix):
    """
    Mean-Variance Optimization: => Optimization-based weighting
    Calculate the portfolio variance given weights and covariance matrix.

    Args:
        weights (np.ndarray): Array of asset weights in the portfolio.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Portfolio variance.
    """
    return weights.T @ cov_matrix @ weights

def value_weighting(MCAP):
    """
    Value Weighting: => Market Cap-based weighting
    Calculate weights based on market capitalization.

    Args:
        MCAP (pd.Series): Series of market capitalizations for each asset.

    Returns:
        np.ndarray: Array of weights based on market capitalization.
    """
    total_market_cap = MCAP.sum()
    weights = MCAP / total_market_cap
    return weights.values