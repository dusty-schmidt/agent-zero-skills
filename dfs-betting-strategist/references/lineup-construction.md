# Lineup Construction

## The Knapsack Problem

Lineup construction is framed as a **Knapsack Problem**: maximizing value (points) subject to a weight constraint (salary cap). This mathematical optimization challenge appears across multiple domains from resource allocation to portfolio management.

## Convexity and Tail Risk

### Profile Types

#### Concave Profile
- **Definition**: Limited upside, unlimited downside
- **Examples**: Selling options, playing "safe" players in a tournament
- **Risk**: High probability of mediocre outcomes

#### Convex Profile  
- **Definition**: Limited downside (entry fee), unlimited upside (1st place)
- **Examples**: Buying options, playing high-variance lineups
- **Strategy**: Professionals "trim the tails" during construction

### Tail Risk Management
Professionals ignore:
- **Bottom 1%**: Injury risk, impossible outcomes
- **Top 1%**: Unrealistic outcomes, "unicorn" performances

**Focus**: The "realizable range" where outcomes are statistically possible.

### Black Swan Hunting
- **Definition**: Tail events that the market has mispriced
- **Strategy**: Explicitly hunt for improbable but massive payoffs
- **Risk**: Accept high variance in exchange for asymmetric upside

## The Barbell Portfolio

### Nassim Taleb's Optimal Structure
The Barbell Strategy is optimal for navigating uncertainty:

#### DFS Application
- **80%** of bankroll: Low-risk Head-to-Heads (The Floor)
- **20%** of bankroll: High-risk, high-leverage Tournaments (The Ceiling)
- **Avoid**: Medium risk contests (small-field GPPs) with tournament risk but capped upside

#### Life Application
- **Keep your day job** (Safety)
- **Build a side business** (High Risk/Reward)
- **Don't become a "freelancer"** (Medium Risk, Capped Reward)

#### Strategic Benefits
- **Ensures survival** while maximizing exposure to positive variance
- **Prevents ruin** through conservative base allocation
- **Captures upside** through high-variance opportunities

## Mathematical Optimization

### Constraints and Variables

#### Objective Function
```
Maximize: Expected Points
Subject to:
- Salary Cap ≤ Available Budget
- Position Requirements Met
- Player Availability Constraints
```

#### Decision Variables
- **Binary**: Include/exclude each player
- **Continuous**: Position allocation percentages
- **Integer**: Exact player counts per position

### Portfolio Optimization Techniques

#### Modern Portfolio Theory
- **Mean-Variance Optimization**: Balance expected return and risk
- **Efficient Frontier**: Optimal risk-return combinations
- **Correlation Adjustment**: Account for player interactions

#### Machine Learning Approaches
- **Reinforcement Learning**: Learn optimal strategies through simulation
- **Genetic Algorithms**: Evolve lineup solutions over generations
- **Neural Networks**: Predict optimal player combinations

## Variance Engineering

### Stacking Strategies

#### Positive Correlation Stacking
- **QB-WR combinations**: Linked performance outcomes
- **Team stacks**: Entire offensive units
- **Game stacks**: Players from high-scoring games

**Benefits**:
- Amplify upside when predictions correct
- Create "diamond hands" portfolios
- Increase probability of tournament wins

#### Negative Correlation Hedging
- **Cross-game correlation**: Opposing team players
- **Defense vs. Offense**: Contrarian positioning
- **Contrarian stacks**: Against popular correlations

**Benefits**:
- Reduce overall variance
- Smooth return profiles
- Protect against correlation breakdown

### Diversification Strategies

#### Positional Diversification
- **Spread risk across positions**: Don't over-concentrate
- **Balance high and low variance**: Mix safe and volatile players
- **Avoid correlation clusters**: Don't stack too many correlated players

#### Game-Based Diversification
- **Multiple games**: Reduce single-game dependency
- **Time diversification**: Spread across different game times
- **Weather considerations**: Account for environmental factors

## Advanced Construction Techniques

### Salary Efficiency Optimization

#### Value-Based Selection
```
Value Score = Expected Points / Salary Cost
```

#### Marginal Value Analysis
- **Compare alternatives**: Each player vs. replacement option
- **Budget allocation**: Spend more on high-efficiency positions
- **Opportunity cost**: Value of alternative player combinations

### Ceiling-Floor Optimization

#### Cash Game Strategy
- **Maximize floor**: High probability of cashing
- **Minimize variance**: Consistent, predictable outcomes
- **Target median outcomes**: Avoid extreme performances

#### Tournament Strategy
- **Maximize ceiling**: Access to first place payouts
- **Accept variance**: High probability of missing but high upside
- **Target outlier outcomes**: 99th percentile performances

## Meta-Game Considerations

### Ownership Projection
- **Predict field behavior**: What will others do?
- **Find overlooked players**: Low ownership, high upside
- **Avoid crowded plays**: High ownership, limited upside

### Game Theory Applications
- **Nash Equilibrium**: Optimal strategies assuming opponents play optimally
- **Evolutionary Stable Strategy**: Strategies that resist invasion
- **Zero-Sum Dynamics**: Your gain equals others' loss

### Tournament Structure
- **Payout distribution**: Skewed toward top performers
- **Field size**: Larger fields reward outlier outcomes
- **Entry fees**: Higher fees require higher expected value

## Quality Assurance and Validation

### Backtesting Frameworks
- **Historical performance**: Test strategies on past data
- **Monte Carlo simulation**: Generate thousands of possible outcomes
- **Cross-validation**: Ensure strategies work across different time periods

### Risk Metrics
- **Value at Risk (VaR)**: Maximum expected loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline

### Performance Attribution
- **Skill vs. Luck**: Separate decision quality from variance
- **Process evaluation**: Measure decision-making, not outcomes
- **Continuous improvement**: Learn from both wins and losses

---

*Lineup construction is the art of turning probability distributions into competitive advantage. The professional engineer their portfolios to capture the specific outcomes that create wealth.*