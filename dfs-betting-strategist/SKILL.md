---
name: dfs-betting-strategist
description: Comprehensive betting and DFS strategy framework combining game theory, probabilistic decision-making, lineup optimization, and bankroll management. Use for Daily Fantasy Sports (NBA, NASCAR), sports betting, and any competitive decision-making under uncertainty. Covers EV calculations, Kelly criterion, leverage analysis, contest selection, correlation-adjusted sizing, and portfolio construction.
license: Proprietary
tags: [dfs, betting, game-theory, kelly-criterion, bankroll, nba, nascar, gpp, ownership, leverage, correlation, portfolio, risk-management]
author: GOB
version: 3.0.0
---

# DFS & Betting Strategist

A unified framework for rational decision-making under uncertainty, derived from Daily Fantasy Sports and applicable across all competitive betting and investment domains.

## Core Philosophy

> *The winner is not the one who predicts the future, but the one who constructs a portfolio robust enough to thrive in any future that arrives. The edge is not in the prediction; the edge is in the structure.*

Process over results. Always.

---

## Part I: Universal Decision Framework (Game Theory)

### Step 0: Identify the Ecosystem

Before any analysis, identify the game structure. This determines every subsequent decision.

**Questions:**
- Zero-sum, negative-sum, or positive-sum?
- What is the **rake**? (fees, friction, transaction costs — must be beaten to profit)
- Who are the participants? Professionals or recreational players?
- Skill gap: large (deterministic outcomes possible) or small (variance dominates)?

**The Four Corners:**
| Friction | Skill Gap | Profile | Analog |
|----------|-----------|---------|--------|
| High | High | Small % big winners, massive attrition | VC, distressed debt |
| High | Low | Near-pure negative sum | Slots, MLM |
| Low | High | Meritocracy, sharps dominate | HFT, pro poker |
| Low | Low | Variance dominates, slow churn | Index funds |

**Key insight:** A participant must have an edge exceeding the rake just to break even.

**Paradox of Skill:** In mature markets (narrow skill gaps), luck increasingly determines single-event outcomes.

---

### The Foundational Formulas

#### Expected Value
```
EV = Σ(P × V)
```
Calculate before every material decision. Compare to alternatives. Never judge by outcome.

#### Kelly Criterion — Position Sizing
```
f* = (bp - q) / b

b = net odds (payout per $1 risked)
p = probability of winning
q = probability of losing (1 - p)
f* = fraction of bankroll to risk
```

**Implementation:**
- **Full Kelly**: maximum long-run growth
- **Half Kelly**: reduced volatility, still strong growth (recommended baseline)
- **Quarter Kelly**: conservative, psychological comfort
- **2% Rule**: Never risk >2% of total bankroll on a single position

Scale up during growth periods. Scale down during drawdowns.

#### Leverage Score — Contrarian Value
```
Leverage Score = True Probability / Ownership%
```

- Score > 1.0: Positive leverage (underowned relative to merit)
- Score < 1.0: Negative leverage ("bad chalk" — overowned)

---

### The Professional's Decision Framework

**Pre-Decision:**
1. Ecosystem ID — game type, rake, participants, constraints
2. Information Assessment — what do I have? What's my processing edge?
3. Value Calculation — full EV analysis across all outcomes
4. Strategic Positioning — where is the crowd? What's my leverage?

**Execution:**
5. Risk Management — Kelly sizing, downside protection, exit criteria
6. Implementation — execute systematically, monitor for new information

**Post-Decision:**
7. Outcome Evaluation — what happened vs. expectation?
8. Process Assessment — was the decision sound regardless of outcome?
9. System Update — what assumptions were wrong? Update models.

---

## Part II: DFS-Specific Strategy

### GPP vs Cash Game Strategy

| Mode | Goal | Ownership Target | Construction |
|---|---|---|---|
| GPP (tournaments) | Win / top 1-5% | Low — below field median | Contrarian, high variance |
| Cash (50/50, H2H) | Top 50% | High — follow consensus | Chalk, high floor |

---

### The Ownership Target Rule

Derive it from your generated lineup pool each slate:

```python
win_target_cutoff = lineup_pool["Ownership"].quantile(0.25)
```

**Why it works:** The 25th percentile of your generated pool is where contrarian builds live.

**Empirical validation:**
- NASCAR Feb 2026: Builder median 184.2% | Win cutoff (p25) 166.3%
- NBA historical winners: 96% contain ≥1 player at <15% ownership
- Winner total ownership: 100-150% vs field average 175-210%

---

### The Dangerous Triplet: Combination Breaking

**Definition:** Any 3-player combination appearing in >[30% | slate-relative] of vanilla builds.

**Slate-relative threshold:**
| Slate Size | Triplet Frequency Trigger |
|---|---|
| Large (10+ games) | >40% |
| Medium (6-9 games) | >35% |
| Small (2-5 games) | >25% |
| Showdown (1 game) | >20% |

**Salary-Weighted Risk Adjustment:**

| Triplet Cost | Risk Level | Exposure Cap |
|---|---|---|
| $13-15k (NBA) | Moderate | 35% |
| $17-19k (NBA) | High | 25% |
| $20k+ (NBA) | Critical | 15% |

**Builder constraint:**
```
GROUP RULE: "Max 2 of {Player A, Player B, Player C} at 40% portfolio exposure"
```

---

### Exposure Caps & Tier Construction

```
Players > 50% proj ownership:  max 35-40% of your lineups
Players 30-50% proj ownership: max 30-35% of your lineups
Players 15-30% proj ownership: max 25-30% of your lineups
```

**Ownership-Tiered Construction:**
```
Tier A (>40% proj own):  max 2 players per lineup
Tier B (15-40% proj own): min 2 players per lineup
Tier C (<15% proj own):  min 1 player per lineup
```

---

### Multi-Run Pooling Strategy

```
1. Run builder N times (500 lineups per run)
2. Track new lineup rate = new_unique / total
3. Continue until new lineup rate < 40%
4. Typical: 4-6 runs
5. Combine all unique lineups
6. Extract best N using quality metrics
```

**Extraction Formula:**
```python
tier1 = pool[pool["Ownership"] <= p25].head(int(N*0.25))  # Win shots
tier2 = pool[(pool["Ownership"] > p25) & (pool["Ownership"] <= p50)].head(int(N*0.45))  # Core
tier3 = pool[pool["Ownership"] > p50].head(int(N*0.30))  # Floor
```

---

### Diversity Enforcement

| Target | Description |
|---|---|
| Min 2 unique | No two lineups share 5+ players (hard floor) |
| Min 3 unique | No two lineups share 4+ players (preferred) |

**Greedy Swap Algorithm:**
1. Calculate all pairwise overlaps
2. Find lineup in most violations
3. Replace with best lineup from reserve pool that creates 0 new violations
4. Repeat until violations = 0

---

### Leverage Analysis

```
Leverage = My Exposure % - Projected Field Ownership %
```

**Target profile:**
- Chalk (>50% proj own): -20 to -35% leverage
- Mid-tier (15-40%): +5 to +15% leverage
- Low-own (<15%): +10 to +20% leverage

---

### Contest Selection Theory

| Contest Type | Typical ROI | Optimal Allocation |
|---|---|---|
| Single Entry | **+15-20%** | 25% of volume |
| 20-max | +8-12% | 20% of volume |
| 150-max MEGA | 0-5% | 5% of volume |

**Required Edge:**
- MEGA GPP (50k+): 17%+ edge
- Large GPP (10k-50k): 15% edge
- Small GPP (<1k): 10-11% edge

---

### Crisis Correlation & Kelly Sizing

**Crisis Correlation:** When all lineup correlations → 1.0 simultaneously (NBA blowouts, NASCAR crashes).

**Mitigation:**
```
If ρ̄ > 0.6: Reduce GPP allocation 30%
If ρ̄ > 0.8: Reduce entries 50%
```

**Correlation-Adjusted Kelly:**
```
Effective entries = Calculated entries / √ρ̄
```

At ρ̄ = 0.5: 150 entries = 212 effective entries

---

### Barbell Bankroll Allocation (Taleb)

- **80% of bankroll:** Low-risk/high-frequency (H2H, Double-Ups)
- **20% of bankroll:** High-risk GPPs (MEGAs, qualifiers)

Absorbs variance while maintaining geometric growth potential.

---

## Part III: Sport-Specific Mechanics

### NBA (DraftKings Main Slate)

**Format:**
- 8 slots: PG, SG, SF, PF, C, G, F, UTIL
- $50,000 salary cap

**Key Dynamics:**
- Ownership spreads on large slates (10+ games)
- Value concentrates: cheap chalk hits 40-50% ownership
- Correlation matters: same-game stacking, opposing fades

**Late Swap Tactics:**
| News Type | Response | Timing |
|---|---|---|
| Star out | Mass pivot to value | Within 60 min of lock |
| Role change | Moderate exposure increase | 30-45 min pre-lock |

---

### NASCAR Cup Series

**Format:**
- 6 drivers (all same position: D)
- Scoring: finish + laps led + fastest laps + dominator bonus

**Key Dynamics:**
- Highly concentrated ownership (top 2-3 at 50-70%+)
- Chalk busts harder (mechanical failures, crashes)
- Win target cutoff: 15-35% below builder median

**Anti-Correlation Exploitation:**
- Fade previous-race winners at >40% ownership (12-15% underperformance)
- Crisis correlation week (Daytona 500): Reduce stakes 25%

---

## Part IV: Scripts

All scripts in `scripts/` directory.

| Script | Purpose |
|---|---|
| `monte_carlo_nba.py` | NBA simulation engine |
| `monte_carlo_nascar.py` | NASCAR simulation engine |
| `pool_analyzer.py` | Analyze pool: ownership, win zone, exposure |
| `multi_run_combiner.py` | Combine runs, deduplicate, extract best N |
| `diversity_enforcer.py` | Enforce min-unique constraints |
| `dk_entry_builder.py` | Generate DK upload CSV |
| `exposure_chart.py` | Generate exposure vs ownership chart |
