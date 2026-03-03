# Variance and Correlation

## Systems Engineering and Variance

**Correlation** is the degree to which events are linked. In DFS, correlation is used to engineer specific volatility profiles for lineups, controlling the risk-reward characteristics of investment decisions.

## Stacking and the Reduction of Variables

### The Lottery Analogy
To win a lottery with 6 numbers, you must predict 6 independent events correctly. The probability is:
```
P = P1 × P2 × P3 × P4 × P5 × P6
```

If the lottery were rigged so that if Number 1 is drawn, Number 2 is automatically drawn, you only need to predict 5 events. The odds improve exponentially.

### Stacking in Practice (Positive Correlation)
In DFS, pairing a **Quarterback (QB)** with his **Wide Receiver (WR)** links their outcomes:
- If the QB throws a touchdown, it is highly probable the WR caught it
- You capture double points for a single event
- **Impact**: Both players succeed or fail together

### Variance Engineering
**Stacking increases variance**:
- **Floor Effect**: If the QB fails, the WR likely fails (the floor drops)
- **Ceiling Effect**: If the QB hits his ceiling, the WR likely hits his (the ceiling raises)

## Systems Engineering Applications

### Series System (The Chain)
**Definition**: Components are independent but all must work for success
**Reliability Formula**: R = R1 × R2 × R3
**Characteristics**: 
- Fragile system
- DFS lineup of un-correlated players is a Series System for ceiling outcomes
- Everything must go right independently

### Parallel System (Redundancy)
**Definition**: Components support each other
**Characteristics**:
- Stacking creates a pseudo-parallel system
- Success of one component drives success of another
- More robust system design

## Geometric Mean vs. Arithmetic Mean

### Fundamental Distinction
The provided concepts introduce the crucial distinction between **Geometric Mean** and **Arithmetic Mean** in the context of growth and correlation.

### Arithmetic Mean
**Definition**: Simple average
**Relevance**: Independent, additive events
**Formula**: (x1 + x2 + ... + xn) / n

### Geometric Mean
**Definition**: The nth root of the product
**Relevance**: Correlated, multiplicative, or compounded series
**Formula**: (x1 × x2 × ... × xn)^(1/n)

### The Volatility Drag
In investment portfolios, high volatility drags down the geometric mean (CAGR):

**Example**: Portfolio that goes +50% then -50%
- **Arithmetic Mean**: 0% (50% + (-50%) / 2)
- **Geometric Mean**: -13.4% ($100 → $150 → $75)
- **Actual Return**: -25%

## Strategic Applications

### Long-Term Wealth Preservation (Cash Games/Retirement)
**Strategy**: Minimize variance
**Method**: Negative Correlation/Diversification
**Goal**: Protect the geometric mean
**Application**: Index funds, balanced portfolios, low-risk investments

### Short-Term Wealth Creation (GPPs/Startups)
**Strategy**: Maximize variance
**Method**: Positive Correlation/Concentration
**Goal**: Access fat tails
**Risk**: Accept high "risk of ruin" (geometric mean sacrifice)
**Application**: Venture capital, aggressive growth stocks, lottery tickets

## Correlation Types and Applications

### Positive Correlation
**Definition**: Events move in the same direction
**Examples**:
- QB and WR performance
- Tech stock rallies (high growth stocks move together)
- Economic booms (consumer spending and employment)

**Applications**:
- **Stacking**: Link complementary positions
- **Momentum**: Ride winning trends
- **Concentration**: Bet big on high-conviction ideas

### Negative Correlation
**Definition**: Events move in opposite directions
**Examples**:
- Stocks and bonds
- Gold and dollar strength
- Defensive vs. growth stocks

**Applications**:
- **Hedging**: Reduce overall portfolio risk
- **Market Neutral**: Profit regardless of direction
- **Risk Management**: Limit downside exposure

### Zero Correlation
**Definition**: Events are independent
**Examples**:
- Unrelated industries
- Random events
- Diversified portfolios

**Applications**:
- **Risk Reduction**: Lower overall variance
- **Smooth Returns**: Consistent performance
- **Stability**: Predictable outcomes

## Portfolio Construction Using Correlation

### Diversification Benefits
The mathematical benefit of diversification:
```
Portfolio Variance = Σ(wi² × σi²) + Σ Σ(wi × wj × σi × σj × ρij)
```

Where:
- wi = weight of asset i
- σi = standard deviation of asset i
- ρij = correlation between assets i and j

### Optimal Correlation Matrix
- **High correlation**: Reduces diversification benefits
- **Low/negative correlation**: Maximizes risk reduction
- **Dynamic adjustment**: Correlation changes over time

## Real-World Applications

### Corporate Strategy
- **Vertical Integration**: Increase correlation between business units
- **Geographic Diversification**: Reduce correlation across markets
- **Product Portfolio**: Balance correlated and uncorrelated offerings

### Career Development
- **Skill Stacking**: Correlated skills amplify each other
- **Network Effects**: Relationships create positive correlations
- **Industry Exposure**: Diversify career risk across sectors

### Investment Management
- **Asset Allocation**: Balance correlated and uncorrelated assets
- **Rebalancing**: Take advantage of correlation changes
- **Risk Parity**: Equal risk contribution from each position

## Advanced Correlation Concepts

### Conditional Correlation
Correlation changes based on market conditions:
- **Bull markets**: Growth stocks more correlated
- **Bear markets**: Flight to quality increases correlation
- **Crisis periods**: "All correlations go to 1"

### Time-Varying Correlation
- **Short-term**: Higher correlation (momentum effects)
- **Long-term**: Lower correlation (mean reversion)
- **Business cycles**: Correlation varies with economic conditions

### Correlation vs. Causation
- **Correlation**: Statistical relationship
- **Causation**: One event directly affects another
- **Danger**: Mistaking correlation for causation in strategy

---

*Understanding variance and correlation allows for precise engineering of risk profiles. The professional controls not just what outcomes are possible, but how those outcomes relate to each other.*