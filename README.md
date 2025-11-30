This will be the main hub for all the group data projects

https://challengedata.ens.fr/participants/challenges/146/

Let this be the current challenge

Quant Resources:
- https://github.com/wilsonfreitas/awesome-quant
- https://github.com/cybergeekgyan/Quant-Developers-Resources

This seems really good!
https://github.com/microsoft/qlib


Workflow:

1. Get the data
2. Train the model (we can decide on the time step)
3. Portfolio Optimization (Return vs. Risk -> Risk is the confidence level)
4. Construct our portfolio based on what our model predicts
5. Backtest model and evaluate

The models are 'return prediction' models

Convex Optimization Library: https://www.cvxpy.org/index.html

# Xuanthe's Notes from the last QRT lecture

## Signals
- Some examples are Value Signals (i.e., Cash Flow / Price of Stock) to gauge the 'value' of a stock
- Also momentum signals (i.e., if the returns trend upwards, buy into it)
- Mean Reversion is another one

Having more signals => minimizes the risk from one signal failing

## Portfolio Construction
- Our objective function is going to be a combination of the alphas AND MUST also have the risk appetite (otherwise, the model will assign all the weight into the stock with the highest return)
- We can do bottom up (combining our signals into an overall score to dictate positions in portfolio) or 'top-down' (implement each portfolio strategy, and then combine them together for an average)
- Important to consider constraints on how much we can hold of a stock at a time, or what percentage of ADV (average daily volume) we want to consume
- The objective function can become very complicated (essentially a utility optimization function our weights multiplied by expected return - our risk appetite)
- QRT's ADV is 2.5%

### Limitations of Portfolio constuction
- The current objective function inherently biases stocks that are measured with the highest error
- 'Risk-Adjusted' -> we want our stocks to be uncorrelated to the market (so we need to estimate the betas -> the correlation of that stock to the market; if the market moves, how much does the stock move with the market?)
- Increasing idiosyncratci risk => increase decoupling from the market

### Dealing with the model uncertainty
- Two approaches; regularize the weight (let the model be unstable; regularization) or handle the covariance matrix appropiately 

### Covariance issues
- We want a stable portfolio
- How we going to form our covariance matrix? (Cholesky? PCA?)
- Market Impact -> market moves against you in trades

### Transaction limitaitons
- Liquidity is important: we want the market to have enough liquidity to absord our trades
- Transaction costs => non-linear profitability!

There is existing literature in the photos Nut sent. Look at images for more detailed steps
