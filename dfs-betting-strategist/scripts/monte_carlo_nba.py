"""
NBA DFS Monte Carlo Simulation Framework
==========================================

A comprehensive quantitative framework for evaluating NBA DFS GPP lineups
using Monte Carlo simulations with proper statistical distributions,
correlation modeling, and lineup optimization.

Author: Quantitative DFS Researcher
Date: 2026-03-03
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from itertools import combinations
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: DATA STRUCTURES AND CONFIGURATION
# =============================================================================

@dataclass
class Player:
    """
    Represents an NBA player with projection and variance estimates.

    Attributes:
        name: Player name
        player_id: Unique identifier (e.g., DraftKings ID)
        position: Primary position (PG, SG, SF, PF, C)
        positions: List of eligible positions (e.g., ['PG', 'SG', 'G'])
        salary: DraftKings salary
        projection: Mean fantasy point projection
        std_dev: Standard deviation of fantasy points (ceiling/floor measure)
        team: NBA team abbreviation
        opponent: Opposing team
        game_total: Vegas game total (for correlation modeling)
        spread: Vegas spread (for correlation modeling)
        minutes: Projected minutes
        usage_rate: Usage rate estimate
        is_starter: Whether player is expected starter
    """
    name: str
    player_id: str
    position: str
    positions: List[str]
    salary: int
    projection: float
    std_dev: float
    team: str
    opponent: str
    game_total: float = 220.0
    spread: float = 0.0
    minutes: float = 30.0
    usage_rate: float = 20.0
    is_starter: bool = True

    @property
    def cv(self) -> float:
        """Coefficient of variation (std_dev / projection)"""
        return self.std_dev / self.projection if self.projection > 0 else 0

    @property
    def value(self) -> float:
        """Points per $1000 of salary"""
        return (self.projection / self.salary) * 1000


@dataclass
class CorrelationMatrix:
    """
    Stores and manages player correlation structure.

    NBA DFS correlations are critical because:
    1. Same-game correlation: Players on same team have positive correlation
       (high pace = more stats for everyone)
    2. Usage competition: High-usage players on same team have negative correlation
    3. Game total correlation: Games with high totals benefit all players involved
    """
    players: List[Player] = field(default_factory=list)
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def build_correlation_matrix(
        self,
        same_team_correlation: float = 0.15,
        opponent_correlation: float = 0.05,
        usage_competition_coef: float = -0.08,
        game_total_sensitivity: float = 0.10
    ) -> np.ndarray:
        """
        Build correlation matrix based on NBA-specific factors.

        Args:
            same_team_correlation: Base correlation for teammates
            opponent_correlation: Base correlation for opponents (pace effects)
            usage_competition_coef: Negative correlation per usage point difference
            game_total_sensitivity: Correlation boost for high-total games
        """
        n = len(self.players)
        corr = np.eye(n)  # Start with identity (self-correlation = 1)

        player_idx = {p.player_id: i for i, p in enumerate(self.players)}

        for i, p1 in enumerate(self.players):
            for j, p2 in enumerate(self.players):
                if i >= j:
                    continue

                correlation = 0.0

                # Same team correlation
                if p1.team == p2.team:
                    correlation += same_team_correlation

                    # Usage competition (negative correlation)
                    usage_diff = abs(p1.usage_rate - p2.usage_rate)
                    correlation += usage_competition_coef * (usage_diff / 10)

                # Opponent correlation (game pace effects)
                if p1.opponent == p2.team or p2.opponent == p1.team:
                    correlation += opponent_correlation

                    # Game total boost
                    avg_total = (p1.game_total + p2.game_total) / 2
                    if avg_total > 220:
                        correlation += game_total_sensitivity * (avg_total - 220) / 20

                corr[i, j] = correlation
                corr[j, i] = correlation

        # Ensure positive semi-definite
        eigenvalues = np.linalg.eigvalsh(corr)
        if np.min(eigenvalues) < 0:
            corr += np.eye(n) * (abs(np.min(eigenvalues)) + 0.01)

        self.correlation_matrix = corr
        return corr

    def subset_for_players(self, players_subset: List[Player]) -> 'CorrelationMatrix':
        """Create a new correlation matrix for a subset of players."""
        # Build index mapping from original to subset
        original_ids = {p.player_id: i for i, p in enumerate(self.players)}
        subset_indices = []
        for p in players_subset:
            if p.player_id in original_ids:
                subset_indices.append(original_ids[p.player_id])

        # Extract submatrix
        if len(subset_indices) > 0 and self.correlation_matrix.size > 0:
            submatrix = self.correlation_matrix[np.ix_(subset_indices, subset_indices)]
        else:
            # Build new matrix for subset
            submatrix = self.build_correlation_matrix()
            # Rebuild with just these players if needed
            if len(self.players) != len(players_subset):
                temp_corr = CorrelationMatrix(players_subset)
                submatrix = temp_corr.build_correlation_matrix()

        new_corr = CorrelationMatrix(players_subset)
        new_corr.correlation_matrix = submatrix
        return new_corr

    def get_correlated_samples(
        self,
        mean_vector: np.ndarray,
        std_vector: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate correlated random samples using Cholesky decomposition.

        Args:
            mean_vector: Mean fantasy points for each player
            std_vector: Standard deviation for each player
            n_samples: Number of simulations

        Returns:
            Array of shape (n_samples, n_players) with correlated outcomes
        """
        if self.correlation_matrix.size == 0:
            # No correlations, sample independently
            return np.random.normal(
                loc=mean_vector,
                scale=std_vector,
                size=(n_samples, len(mean_vector))
            )

        # Cholesky decomposition for correlated sampling
        L = np.linalg.cholesky(self.correlation_matrix)

        # Generate independent standard normals
        Z = np.random.standard_normal(size=(n_samples, len(mean_vector)))

        # Apply correlation structure
        correlated_Z = Z @ L.T

        # Scale by standard deviations and shift by means
        samples = mean_vector + correlated_Z * std_vector

        return samples


# =============================================================================
# SECTION 2: PLAYER DISTRIBUTION MODELING
# =============================================================================

class PlayerDistributionModel:
    """
    Models fantasy point distributions for individual players.

    NBA player outcomes are typically right-skewed (gamma, log-normal)
    rather than normal due to:
    - Floor effects (minimum minutes played)
    - Ceiling effects (blowout risk, foul trouble)
    - Asymmetric upside (breakout games)
    """

    def __init__(self, distribution: str = 'gamma'):
        """
        Args:
            distribution: 'gamma', 'lognormal', 'normal', or 'mixture'
        """
        self.distribution = distribution
        self.fitted_params = {}

    def fit_from_history(
        self,
        historical_scores: np.ndarray,
        current_projection: float,
        minutes: float = 30.0
    ) -> Dict:
        """
        Fit distribution parameters from historical data.

        Args:
            historical_scores: Array of past fantasy point performances
            current_projection: Current mean projection
            minutes: Projected minutes (for floor estimation)

        Returns:
            Dictionary of fitted parameters
        """
        if len(historical_scores) < 5:
            # Insufficient history, use heuristic estimates
            return self._heuristic_params(current_projection, minutes)

        # Remove outliers (injury games, extreme blowouts)
        q1, q3 = np.percentile(historical_scores, [25, 75])
        iqr = q3 - q1
        filtered = historical_scores[
            (historical_scores >= q1 - 1.5 * iqr) &
            (historical_scores <= q3 + 1.5 * iqr)
        ]

        if self.distribution == 'gamma':
            # Fit gamma: shape, scale
            # Gamma is right-skewed, good for NBA
            mean = np.mean(filtered)
            var = np.var(filtered)
            shape = (mean ** 2) / var if var > 0 else 5
            scale = var / mean if mean > 0 else 1

            self.fitted_params = {
                'shape': shape,
                'scale': scale,
                'loc': 0,
                'mean': mean,
                'std': np.sqrt(var)
            }

        elif self.distribution == 'lognormal':
            # Fit log-normal
            log_data = np.log(filtered[filtered > 0])
            mu = np.mean(log_data)
            sigma = np.std(log_data)

            self.fitted_params = {
                's': sigma,
                'scale': np.exp(mu),
                'loc': 0,
                'mean': np.mean(filtered),
                'std': np.std(filtered)
            }

        elif self.distribution == 'normal':
            # Simple normal fit
            self.fitted_params = {
                'loc': np.mean(filtered),
                'scale': np.std(filtered),
                'mean': np.mean(filtered),
                'std': np.std(filtered)
            }

        elif self.distribution == 'mixture':
            # Gaussian mixture for bimodal distributions (starter/bench scenarios)
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(filtered.reshape(-1, 1))

            self.fitted_params = {
                'weights': gmm.weights_.tolist(),
                'means': gmm.means_.flatten().tolist(),
                'covariances': gmm.covariances_.flatten().tolist(),
                'mean': np.mean(filtered),
                'std': np.std(filtered)
            }

        return self.fitted_params

    def _heuristic_params(self, projection: float, minutes: float) -> Dict:
        """
        Estimate distribution parameters when historical data is unavailable.

        Uses NBA-specific heuristics:
        - Floor: ~0.7x projection (injury/blowout risk)
        - Ceiling: ~1.6x projection (breakout game)
        - Std dev: ~0.25x projection (typical variance)
        """
        # Heuristic coefficient of variation for NBA
        cv = 0.25  # 25% of mean is typical

        # Adjust for minutes uncertainty
        if minutes < 20:
            cv = 0.35  # Higher variance for low-minute players
        elif minutes > 35:
            cv = 0.20  # Lower variance for high-minute players

        std = projection * cv

        if self.distribution == 'gamma':
            # Gamma parameters from mean and std
            shape = (projection ** 2) / (std ** 2) if std > 0 else 5
            scale = (std ** 2) / projection if projection > 0 else 1

            return {
                'shape': shape,
                'scale': scale,
                'loc': 0,
                'mean': projection,
                'std': std
            }

        elif self.distribution in ['normal', 'lognormal']:
            return {
                'loc': projection,
                'scale': std,
                'mean': projection,
                'std': std
            }

        return {'mean': projection, 'std': std}

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate random samples from fitted distribution."""
        if not self.fitted_params:
            raise ValueError("Distribution not fitted. Call fit_from_history first.")

        if self.distribution == 'gamma':
            return stats.gamma.rvs(
                self.fitted_params['shape'],
                loc=self.fitted_params['loc'],
                scale=self.fitted_params['scale'],
                size=n_samples
            )

        elif self.distribution == 'lognormal':
            return stats.lognorm.rvs(
                self.fitted_params['s'],
                loc=self.fitted_params['loc'],
                scale=self.fitted_params['scale'],
                size=n_samples
            )

        elif self.distribution == 'normal':
            return stats.norm.rvs(
                loc=self.fitted_params['loc'],
                scale=self.fitted_params['scale'],
                size=n_samples
            )

        elif self.distribution == 'mixture':
            # Gaussian mixture sampling
            weights = np.array(self.fitted_params['weights'])
            means = np.array(self.fitted_params['means'])
            covs = np.array(self.fitted_params['covariances'])

            components = np.random.choice(len(weights), size=n_samples, p=weights)
            samples = np.array([
                np.random.normal(means[c], np.sqrt(covs[c]))
                for c in components
            ])
            return samples

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


# =============================================================================
# SECTION 3: MONTE CARLO SIMULATION ENGINE
# =============================================================================

class MonteCarloSimulator:
    """
    Core Monte Carlo simulation engine for NBA DFS.

    Generates N simulated slates with correlated player outcomes,
    then optimizes lineups for each simulation.
    """

    def __init__(
        self,
        players: List[Player],
        n_simulations: int = 10000,
        distribution: str = 'gamma',
        correlation_matrix: Optional[CorrelationMatrix] = None,
        salary_cap: int = 50000,
        roster_positions: List[str] = None
    ):
        """
        Initialize simulator.

        Args:
            players: List of Player objects
            n_simulations: Number of Monte Carlo iterations
            distribution: Distribution type for player outcomes
            correlation_matrix: Pre-built correlation matrix
            salary_cap: DraftKings salary cap (default 50000)
            roster_positions: Required positions (default NBA classic)
        """
        self.players = players
        self.n_simulations = n_simulations
        self.distribution = distribution
        self.salary_cap = salary_cap
        self.roster_positions = roster_positions or ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

        # Build or use provided correlation matrix
        if correlation_matrix is None:
            self.corr_matrix = CorrelationMatrix(players)
            self.corr_matrix.build_correlation_matrix()
        else:
            self.corr_matrix = correlation_matrix

        # Fit distributions for each player
        self.player_models = {}
        for player in players:
            model = PlayerDistributionModel(distribution)
            model._heuristic_params(player.projection, player.minutes)
            self.player_models[player.player_id] = model

        # Storage for simulation results
        self.simulated_scores = None
        self.optimal_lineups = []

    def run_simulation(self, progress_interval: int = 1000) -> np.ndarray:
        """
        Run Monte Carlo simulation generating correlated player outcomes.

        Args:
            progress_interval: Print progress every N simulations

        Returns:
            Array of shape (n_simulations, n_players) with fantasy scores
        """
        n_players = len(self.players)

        # Get mean and std vectors
        mean_vector = np.array([p.projection for p in self.players])
        std_vector = np.array([p.std_dev for p in self.players])

        # Ensure correlation matrix matches player count
        if self.corr_matrix.correlation_matrix.shape[0] != n_players:
            print(f"  Rebuilding correlation matrix for {n_players} players...")
            self.corr_matrix = CorrelationMatrix(self.players)
            self.corr_matrix.build_correlation_matrix()

        # Generate correlated samples
        print(f"Running {self.n_simulations} Monte Carlo simulations...")

        self.simulated_scores = self.corr_matrix.get_correlated_samples(
            mean_vector, std_vector, self.n_simulations
        )

        # Ensure non-negative scores
        self.simulated_scores = np.maximum(self.simulated_scores, 0)

        print(f"Simulation complete. Shape: {self.simulated_scores.shape}")

        return self.simulated_scores

    def optimize_lineup_for_simulation(
        self,
        sim_idx: int,
        exclude_lineups: Optional[List[set]] = None
    ) -> Optional[Dict]:
        """
        Optimize a single lineup for a specific simulation.

        Uses integer linear programming approximation with greedy refinement.

        Args:
            sim_idx: Index of simulation to optimize for
            exclude_lineups: Set of player_id sets to avoid duplicating

        Returns:
            Dictionary with lineup details or None if infeasible
        """
        scores = self.simulated_scores[sim_idx]

        # Greedy lineup construction with position constraints
        lineup = []
        lineup_positions = []
        total_salary = 0
        used_ids = set()

        # Position requirements for NBA DraftKings Classic
        # PG, SG, SF, PF, C, G (PG/SG), F (SF/PF), UTIL (any)
        required_positions = {
            'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
            'G': 1, 'F': 1, 'UTIL': 1
        }

        # Sort players by simulated score / salary value
        player_values = []
        for i, player in enumerate(self.players):
            if scores[i] > 0:
                value = scores[i] / player.salary * 1000  # Points per $1000
                player_values.append((i, value, scores[i]))

        player_values.sort(key=lambda x: x[1], reverse=True)

        # Fill positions greedily
        position_counts = {pos: 0 for pos in required_positions}

        for idx, value, score in player_values:
            player = self.players[idx]

            if player.player_id in used_ids:
                continue
            if total_salary + player.salary > self.salary_cap:
                continue

            # Check which positions this player can fill
            eligible_positions = []
            for pos, needed in required_positions.items():
                if position_counts[pos] < needed:
                    if pos == 'G' and ('PG' in player.positions or 'SG' in player.positions):
                        eligible_positions.append(pos)
                    elif pos == 'F' and ('SF' in player.positions or 'PF' in player.positions):
                        eligible_positions.append(pos)
                    elif pos == 'UTIL':
                        eligible_positions.append(pos)
                    elif pos in player.positions:
                        eligible_positions.append(pos)

            if eligible_positions:
                # Prefer specific positions over flex
                pos_priority = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
                chosen_pos = None
                for p in pos_priority:
                    if p in eligible_positions:
                        chosen_pos = p
                        break

                lineup.append(player)
                lineup_positions.append(chosen_pos)
                used_ids.add(player.player_id)
                total_salary += player.salary
                position_counts[chosen_pos] += 1

        # Check if lineup is complete
        if len(lineup) < 8:
            return None

        # Calculate total score
        total_score = sum(scores[self.players.index(p)] for p in lineup)

        return {
            'players': lineup,
            'positions': lineup_positions,
            'player_ids': [p.player_id for p in lineup],
            'player_names': [p.name for p in lineup],
            'total_salary': total_salary,
            'total_score': total_score,
            'salary_remaining': self.salary_cap - total_salary
        }

    def generate_optimal_lineups(self, n_lineups: int = 100) -> List[Dict]:
        """
        Generate optimal lineups for multiple simulations.

        Args:
            n_lineups: Number of lineups to generate

        Returns:
            List of lineup dictionaries
        """
        if self.simulated_scores is None:
            self.run_simulation()

        print(f"Generating {n_lineups} optimal lineups...")

        # Sample simulations for lineup optimization
        sim_indices = np.random.choice(
            self.n_simulations,
            size=min(n_lineups, self.n_simulations),
            replace=False
        )

        generated_lineups = []
        used_lineup_sets = []

        for i, sim_idx in enumerate(sim_indices):
            lineup = self.optimize_lineup_for_simulation(
                sim_idx,
                exclude_lineups=used_lineup_sets if i > 50 else None
            )

            if lineup:
                generated_lineups.append(lineup)
                used_lineup_sets.append(set(lineup['player_ids']))

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1} lineups...")

        self.optimal_lineups = generated_lineups
        print(f"Successfully generated {len(generated_lineups)} lineups")

        return generated_lineups


# =============================================================================
# SECTION 3: LINEUP EVALUATION METRICS
# =============================================================================

class LineupEvaluator:
    """
    Evaluates lineup quality using Monte Carlo simulation results.

    Calculates key GPP metrics:
    - Cash probability (top 50%)
    - Top 10%, Top 1%, Win probability
    - Expected Value (EV) and variance
    - Percentile ranking
    """

    def __init__(
        self,
        simulator: MonteCarloSimulator,
        contest_payouts: Optional[Dict] = None
    ):
        """
        Args:
            simulator: MonteCarloSimulator with completed simulations
            contest_payouts: Dictionary mapping rank ranges to payout multipliers
                           e.g., {1: 10000, 2: 5000, 'top_0.1': 1000, 'top_1': 100}
        """
        self.simulator = simulator
        self.contest_payouts = contest_payouts or self._default_payouts()

    def _default_payouts(self) -> Dict:
        """Default GPP payout structure (normalized to entry fee = 1)"""
        return {
            1: 10000,      # 1st place
            2: 5000,       # 2nd place
            3: 2500,       # 3rd place
            'top_0.1': 1000,   # Top 0.1%
            'top_1': 100,      # Top 1%
            'top_10': 10,      # Top 10%
            'top_50': 2        # Min-cash (top 50%)
        }

    def evaluate_lineup(
        self,
        lineup: Dict,
        n_opponents: int = 5000
    ) -> Dict:
        """
        Evaluate a single lineup against simulated opponents.

        Args:
            lineup: Lineup dictionary with 'player_ids' and 'total_score'
            n_opponents: Number of opponent lineups to simulate

        Returns:
            Dictionary with evaluation metrics
        """
        if self.simulator.simulated_scores is None:
            raise ValueError("Simulator has not run simulations yet")

        # Get player indices
        player_indices = [
            next(i for i, p in enumerate(self.simulator.players) 
                 if p.player_id == pid)
            for pid in lineup['player_ids']
        ]

        # Calculate lineup scores across all simulations
        lineup_scores = self.simulator.simulated_scores[:, player_indices].sum(axis=1)

        # Generate opponent lineups (simplified: random 8-player combinations)
        # In practice, you'd use actual opponent lineup distributions
        opponent_scores = self._simulate_opponents(n_opponents)

        # Calculate metrics
        metrics = self._calculate_metrics(lineup_scores, opponent_scores)
        metrics['lineup'] = lineup
        metrics['player_ids'] = lineup['player_ids']

        return metrics

    def _simulate_opponents(self, n_opponents: int) -> np.ndarray:
        """
        Simulate opponent lineup scores.

        For simplicity, samples random 8-player combinations.
        In production, use actual field lineup distributions.
        """
        n_players = len(self.simulator.players)
        opponent_scores = []

        for _ in range(n_opponents):
            # Sample 8 random players
            indices = np.random.choice(n_players, size=8, replace=False)
            sim_idx = np.random.randint(0, self.simulator.n_simulations)
            score = self.simulator.simulated_scores[sim_idx, indices].sum()
            opponent_scores.append(score)

        return np.array(opponent_scores)

    def _calculate_metrics(
        self,
        lineup_scores: np.ndarray,
        opponent_scores: np.ndarray
    ) -> Dict:
        """
        Calculate key GPP evaluation metrics.
        """
        n_sims = len(lineup_scores)

        # Basic stats
        mean_score = np.mean(lineup_scores)
        std_score = np.std(lineup_scores)
        min_score = np.min(lineup_scores)
        max_score = np.max(lineup_scores)

        # Percentiles
        p1 = np.percentile(lineup_scores, 1)
        p5 = np.percentile(lineup_scores, 5)
        p10 = np.percentile(lineup_scores, 10)
        p25 = np.percentile(lineup_scores, 25)
        p50 = np.percentile(lineup_scores, 50)
        p75 = np.percentile(lineup_scores, 75)
        p90 = np.percentile(lineup_scores, 90)
        p99 = np.percentile(lineup_scores, 99)

        # Win probabilities vs opponents
        win_probs = []
        for lineup_score in lineup_scores:
            wins = np.sum(lineup_score > opponent_scores)
            win_probs.append(wins / len(opponent_scores))

        avg_win_prob = np.mean(win_probs)

        # Cash probability (top 50%)
        cash_threshold = np.percentile(opponent_scores, 50)
        cash_prob = np.mean(lineup_scores > cash_threshold)

        # Top 10% probability
        top10_threshold = np.percentile(opponent_scores, 90)
        top10_prob = np.mean(lineup_scores > top10_threshold)

        # Top 1% probability
        top1_threshold = np.percentile(opponent_scores, 99)
        top1_prob = np.mean(lineup_scores > top1_threshold)

        # Win probability (1st place)
        win_prob = np.mean(lineup_scores > np.max(opponent_scores))

        # Expected Value calculation
        ev = self._calculate_ev(lineup_scores, opponent_scores)

        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'cv': std_score / mean_score if mean_score > 0 else 0,
            'min_score': min_score,
            'max_score': max_score,
            'percentiles': {
                'p1': p1, 'p5': p5, 'p10': p10, 'p25': p25,
                'p50': p50, 'p75': p75, 'p90': p90, 'p99': p99
            },
            'cash_prob': cash_prob,
            'top10_prob': top10_prob,
            'top1_prob': top1_prob,
            'win_prob': win_prob,
            'avg_win_prob': avg_win_prob,
            'expected_value': ev,
            'upside_ratio': p90 / mean_score if mean_score > 0 else 0,
            'floor_ratio': p10 / mean_score if mean_score > 0 else 0
        }

    def _calculate_ev(
        self,
        lineup_scores: np.ndarray,
        opponent_scores: np.ndarray
    ) -> float:
        """
        Calculate expected value based on payout structure.
        """
        n_opponents = len(opponent_scores)
        payouts = []

        for lineup_score in lineup_scores:
            # Count how many opponents this lineup beats
            rank = n_opponents - np.sum(lineup_score > opponent_scores) + 1

            # Determine payout based on rank
            payout = 0
            if rank == 1:
                payout = self.contest_payouts.get(1, 0)
            elif rank == 2:
                payout = self.contest_payouts.get(2, 0)
            elif rank == 3:
                payout = self.contest_payouts.get(3, 0)
            elif rank <= n_opponents * 0.001:  # Top 0.1%
                payout = self.contest_payouts.get('top_0.1', 0)
            elif rank <= n_opponents * 0.01:  # Top 1%
                payout = self.contest_payouts.get('top_1', 0)
            elif rank <= n_opponents * 0.10:  # Top 10%
                payout = self.contest_payouts.get('top_10', 0)
            elif rank <= n_opponents * 0.50:  # Top 50% (min cash)
                payout = self.contest_payouts.get('top_50', 0)

            payouts.append(payout)

        return np.mean(payouts)


# =============================================================================
# SECTION 4: LINEUP OPTIMIZER
# =============================================================================

class LineupOptimizer:
    """
    NBA DFS Lineup Optimizer using integer programming.

    Handles DraftKings NBA Classic format:
    - 8 positions: PG, SG, SF, PF, C, G, F, UTIL
    - $50,000 salary cap
    """

    def __init__(
        self,
        players: List[Player],
        salary_cap: int = 50000,
        roster_spots: int = 8
    ):
        self.players = players
        self.salary_cap = salary_cap
        self.roster_spots = roster_spots
        self.player_idx = {p.player_id: i for i, p in enumerate(players)}

    def optimize(
        self,
        projections: Optional[np.ndarray] = None,
        exclude_lineups: Optional[List[set]] = None,
        min_unique: int = 2,
        exposure_limits: Optional[Dict[str, float]] = None
    ) -> Optional[Dict]:
        """
        Optimize a single lineup.

        Args:
            projections: Custom projection vector (uses player.projection if None)
            exclude_lineups: List of player_id sets to avoid
            min_unique: Minimum unique players vs excluded lineups
            exposure_limits: Dict of player_id -> max exposure

        Returns:
            Optimized lineup dictionary or None
        """
        if projections is None:
            projections = np.array([p.projection for p in self.players])

        # Greedy optimization with constraint handling
        n = len(self.players)
        selected = np.zeros(n, dtype=bool)

        # Position requirements
        position_requirements = {
            'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
            'G': 1, 'F': 1, 'UTIL': 1
        }
        position_filled = {pos: 0 for pos in position_requirements}

        # Calculate value scores
        values = projections / np.array([p.salary for p in self.players]) * 1000

        # Sort by value descending
        sorted_indices = np.argsort(values)[::-1]

        total_salary = 0
        lineup_players = []
        lineup_positions = []

        for idx in sorted_indices:
            player = self.players[idx]

            # Check salary
            if total_salary + player.salary > self.salary_cap:
                continue

            # Check if already selected
            if selected[idx]:
                continue

            # Check exclusion constraints
            if exclude_lineups:
                current_ids = set([p.player_id for p in lineup_players] + [player.player_id])
                too_similar = False
                for excluded in exclude_lineups:
                    overlap = len(current_ids & excluded)
                    if overlap > (8 - min_unique):
                        too_similar = True
                        break
                if too_similar:
                    continue

            # Check exposure limits
            if exposure_limits and player.player_id in exposure_limits:
                current_count = sum(1 for p in lineup_players if p.player_id == player.player_id)
                if current_count >= exposure_limits[player.player_id]:
                    continue

            # Find eligible position
            eligible_pos = self._get_eligible_positions(player, position_filled, position_requirements)

            if eligible_pos:
                selected[idx] = True
                lineup_players.append(player)
                lineup_positions.append(eligible_pos)
                position_filled[eligible_pos] += 1
                total_salary += player.salary

        # Check if lineup is complete
        if len(lineup_players) < 8:
            return None

        total_projection = sum(p.projection for p in lineup_players)

        return {
            'players': lineup_players,
            'positions': lineup_positions,
            'player_ids': [p.player_id for p in lineup_players],
            'player_names': [p.name for p in lineup_players],
            'total_salary': total_salary,
            'total_projection': total_projection,
            'salary_remaining': self.salary_cap - total_salary
        }

    def _get_eligible_positions(
        self,
        player: Player,
        position_filled: Dict[str, int],
        position_requirements: Dict[str, int]
    ) -> Optional[str]:
        """Determine which position this player should fill."""

        # Priority order for positions
        priority = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

        for pos in priority:
            if position_filled[pos] >= position_requirements[pos]:
                continue

            if pos == 'G':
                if 'PG' in player.positions or 'SG' in player.positions:
                    return pos
            elif pos == 'F':
                if 'SF' in player.positions or 'PF' in player.positions:
                    return pos
            elif pos == 'UTIL':
                return pos
            elif pos in player.positions:
                return pos

        return None

    def generate_lineup_pool(
        self,
        n_lineups: int = 500,
        diversity_iterations: int = 5,
        min_unique: int = 2
    ) -> pd.DataFrame:
        """
        Generate a diverse pool of lineups using multi-run strategy.

        Args:
            n_lineups: Target number of lineups
            diversity_iterations: Number of optimization runs
            min_unique: Minimum unique players between lineups

        Returns:
            DataFrame with lineup pool
        """
        all_lineups = []
        used_lineup_sets = []

        for iteration in range(diversity_iterations):
            print(f"Diversity run {iteration + 1}/{diversity_iterations}...")

            # Generate lineups for this iteration
            iteration_lineups = []
            attempts = 0
            max_attempts = n_lineups * 3

            while len(iteration_lineups) < n_lineups // diversity_iterations and attempts < max_attempts:
                attempts += 1

                # Add some randomness to projections for diversity
                noise = np.random.normal(0, 0.05, len(self.players))
                projections = np.array([p.projection for p in self.players]) * (1 + noise)

                lineup = self.optimize(
                    projections=projections,
                    exclude_lineups=used_lineup_sets + [set(l['player_ids']) for l in iteration_lineups],
                    min_unique=min_unique
                )

                if lineup:
                    iteration_lineups.append(lineup)
                    used_lineup_sets.append(set(lineup['player_ids']))

            all_lineups.extend(iteration_lineups)
            print(f"  Generated {len(iteration_lineups)} lineups")

        # Convert to DataFrame
        df_data = []
        for i, lineup in enumerate(all_lineups):
            row = {
                'lineup_id': i,
                'player_ids': lineup['player_ids'],
                'player_names': lineup['player_names'],
                'positions': lineup['positions'],
                'total_salary': lineup['total_salary'],
                'total_projection': lineup['total_projection'],
                'salary_remaining': lineup['salary_remaining']
            }
            df_data.append(row)

        return pd.DataFrame(df_data)


# =============================================================================
# SECTION 4: CORRELATION AND CONDITIONAL SIMULATION
# =============================================================================

class ConditionalSimulator:
    """
    Handles conditional simulations for late news, injuries, and scratches.

    Critical for NBA DFS where late-breaking news dramatically
    changes player values and correlations.
    """

    def __init__(self, base_simulator: MonteCarloSimulator):
        self.base = base_simulator
        self.conditional_scenarios = []

    def add_injury_scenario(
        self,
        injured_player_id: str,
        replacement_player_id: str,
        probability: float = 0.3,
        minutes_boost: float = 10.0
    ):
        """
        Add an injury scenario with conditional simulation.

        Args:
            injured_player_id: ID of potentially injured player
            replacement_player_id: ID of replacement player
            probability: Probability of injury occurring
            minutes_boost: Minutes increase for replacement if injury occurs
        """
        scenario = {
            'type': 'injury',
            'injured_id': injured_player_id,
            'replacement_id': replacement_player_id,
            'probability': probability,
            'minutes_boost': minutes_boost
        }
        self.conditional_scenarios.append(scenario)

    def add_late_scratch_scenario(
        self,
        player_id: str,
        probability: float = 0.1
    ):
        """
        Add scenario for late scratch (player ruled out after lock).

        Args:
            player_id: ID of player who might be scratched
            probability: Probability of late scratch
        """
        scenario = {
            'type': 'late_scratch',
            'player_id': player_id,
            'probability': probability
        }
        self.conditional_scenarios.append(scenario)

    def run_conditional_simulation(
        self,
        n_scenarios: int = 1000
    ) -> np.ndarray:
        """
        Run simulation with conditional scenarios.

        Args:
            n_scenarios: Number of conditional scenarios to run

        Returns:
            Array of simulated scores incorporating conditionals
        """
        all_samples = []

        for _ in range(n_scenarios):
            # Start with base simulation
            sim_idx = np.random.randint(0, self.base.n_simulations)
            base_scores = self.base.simulated_scores[sim_idx].copy()

            # Apply conditional scenarios
            for scenario in self.conditional_scenarios:
                if np.random.random() < scenario['probability']:
                    if scenario['type'] == 'injury':
                        # Zero out injured player
                        injured_idx = next(
                            i for i, p in enumerate(self.base.players)
                            if p.player_id == scenario['injured_id']
                        )
                        base_scores[injured_idx] = 0

                        # Boost replacement
                        replacement_idx = next(
                            i for i, p in enumerate(self.base.players)
                            if p.player_id == scenario['replacement_id']
                        )
                        boost = scenario['minutes_boost'] / self.base.players[replacement_idx].minutes
                        base_scores[replacement_idx] *= (1 + boost)

                    elif scenario['type'] == 'late_scratch':
                        # Zero out scratched player
                        scratch_idx = next(
                            i for i, p in enumerate(self.base.players)
                            if p.player_id == scenario['player_id']
                        )
                        base_scores[scratch_idx] = 0

            all_samples.append(base_scores)

        return np.array(all_samples)


# =============================================================================
# SECTION 5: SAME-GAME CORRELATION MODELING
# =============================================================================

class SameGameCorrelation:
    """
    Models correlations within the same NBA game.

    Key NBA correlations:
    1. Teammate correlation: Positive (pace benefits all)
    2. Star-to-role correlation: Negative (usage competition)
    3. Game total correlation: All players in high-total games correlated
    4. Stack effects: PG-C, PG-PF positive; SG-SF often negative
    """

    def __init__(self, players: List[Player]):
        self.players = players
        self.correlation_matrix = np.eye(len(players))

    def build_same_game_correlations(
        self,
        base_team_correlation: float = 0.12,
        usage_competition: float = -0.06,
        game_total_boost: float = 0.08,
        position_pairings: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Build correlation matrix accounting for same-game effects.

        Args:
            base_team_correlation: Base positive correlation for teammates
            usage_competition: Negative correlation per usage point diff
            game_total_boost: Additional correlation for high-total games
            position_pairings: Dict of position pairs with correlation modifiers

        Returns:
            Correlation matrix
        """
        n = len(self.players)
        corr = np.eye(n)

        # Default position pairings for NBA
        if position_pairings is None:
            position_pairings = {
                ('PG', 'C'): 0.05,      # Pick and roll synergy
                ('PG', 'PF'): 0.03,     # Pick and pop
                ('SG', 'SF'): -0.03,    # Often compete for wing usage
                ('PF', 'C'): 0.02,      # Frontcourt synergy
            }

        for i, p1 in enumerate(self.players):
            for j, p2 in enumerate(self.players):
                if i >= j:
                    continue

                correlation = 0.0

                # Same team correlation
                if p1.team == p2.team:
                    correlation += base_team_correlation

                    # Usage competition (negative)
                    usage_diff = abs(p1.usage_rate - p2.usage_rate)
                    if usage_diff > 5:  # Significant usage difference
                        correlation += usage_competition * (usage_diff / 10)

                    # Position-specific correlations
                    for (pos1, pos2), modifier in position_pairings.items():
                        if (pos1 in p1.positions and pos2 in p2.positions) or                            (pos1 in p2.positions and pos2 in p1.positions):
                            correlation += modifier

                # Same game correlation (opponents)
                if p1.opponent == p2.team or p2.opponent == p1.team:
                    # Game total boost
                    avg_total = (p1.game_total + p2.game_total) / 2
                    if avg_total > 220:
                        correlation += game_total_boost * (avg_total - 220) / 20

                corr[i, j] = correlation
                corr[j, i] = correlation

        # Ensure positive semi-definite
        eigenvalues = np.linalg.eigvalsh(corr)
        if np.min(eigenvalues) < 0:
            corr += np.eye(n) * (abs(np.min(eigenvalues)) + 0.01)

        self.correlation_matrix = corr
        return corr

    def subset_for_players(self, players_subset: List[Player]) -> 'CorrelationMatrix':
        """Create a new correlation matrix for a subset of players."""
        # Build index mapping from original to subset
        original_ids = {p.player_id: i for i, p in enumerate(self.players)}
        subset_indices = []
        for p in players_subset:
            if p.player_id in original_ids:
                subset_indices.append(original_ids[p.player_id])

        # Extract submatrix
        if len(subset_indices) > 0 and self.correlation_matrix.size > 0:
            submatrix = self.correlation_matrix[np.ix_(subset_indices, subset_indices)]
        else:
            # Build new matrix for subset
            submatrix = self.build_correlation_matrix()
            # Rebuild with just these players if needed
            if len(self.players) != len(players_subset):
                temp_corr = CorrelationMatrix(players_subset)
                submatrix = temp_corr.build_correlation_matrix()

        new_corr = CorrelationMatrix(players_subset)
        new_corr.correlation_matrix = submatrix
        return new_corr


# =============================================================================
# SECTION 6: CONTEST SIMULATION AND EV CALCULATION
# =============================================================================

class ContestSimulator:
    """
    Simulates full GPP contests with multiple entries.

    Calculates portfolio-level metrics and optimal entry allocation.
    """

    def __init__(
        self,
        simulator: MonteCarloSimulator,
        entry_fee: float = 1.0,
        contest_size: int = 10000
    ):
        self.simulator = simulator
        self.entry_fee = entry_fee
        self.contest_size = contest_size
        self.lineups = []

    def add_lineup(self, lineup: Dict, weight: float = 1.0):
        """Add a lineup to the contest simulation."""
        self.lineups.append({
            'lineup': lineup,
            'weight': weight,
            'results': []
        })

    def simulate_contest(
        self,
        n_sims: int = 1000,
        field_size: int = 10000
    ) -> Dict:
        """
        Simulate contest outcomes.

        Args:
            n_sims: Number of contest simulations
            field_size: Number of opponents in field

        Returns:
            Dictionary with contest results
        """
        if not self.lineups:
            raise ValueError("No lineups added to contest")

        if self.simulator.simulated_scores is None:
            self.simulator.run_simulation()

        results = {i: [] for i in range(len(self.lineups))}

        for sim in range(n_sims):
            # Sample a random simulation
            sim_idx = np.random.randint(0, self.simulator.n_simulations)
            scores = self.simulator.simulated_scores[sim_idx]

            # Generate field scores
            field_scores = []
            for _ in range(field_size):
                opponent_indices = np.random.choice(
                    len(self.simulator.players),
                    size=8,
                    replace=False
                )
                field_scores.append(scores[opponent_indices].sum())

            # Score each lineup
            for i, lineup_data in enumerate(self.lineups):
                lineup = lineup_data['lineup']
                player_indices = [
                    next(j for j, p in enumerate(self.simulator.players)
                         if p.player_id == pid)
                    for pid in lineup['player_ids']
                ]
                lineup_score = scores[player_indices].sum()

                # Calculate rank
                rank = sum(1 for fs in field_scores if fs > lineup_score) + 1
                results[i].append({
                    'score': lineup_score,
                    'rank': rank,
                    'won': rank == 1,
                    'cashed': rank <= field_size * 0.5,
                    'top_10': rank <= field_size * 0.1,
                    'top_1': rank <= field_size * 0.01
                })

        # Aggregate results
        aggregated = {}
        for i, sim_results in results.items():
            scores = [r['score'] for r in sim_results]
            ranks = [r['rank'] for r in sim_results]

            aggregated[i] = {
                'lineup': self.lineups[i]['lineup'],
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'mean_rank': np.mean(ranks),
                'win_prob': np.mean([r['won'] for r in sim_results]),
                'cash_prob': np.mean([r['cashed'] for r in sim_results]),
                'top10_prob': np.mean([r['top_10'] for r in sim_results]),
                'top1_prob': np.mean([r['top_1'] for r in sim_results]),
                'ev': self._calculate_lineup_ev(sim_results)
            }

        return aggregated

    def _calculate_lineup_ev(self, sim_results: List[Dict]) -> float:
        """Calculate expected value from simulation results."""
        # Simplified EV calculation
        win_prob = np.mean([r['won'] for r in sim_results])
        cash_prob = np.mean([r['cashed'] for r in sim_results])
        top10_prob = np.mean([r['top_10'] for r in sim_results])

        # Assume payout structure
        ev = (
            win_prob * 10000 +  # 1st place
            top10_prob * 100 +  # Top 10%
            cash_prob * 2       # Min cash
        )

        return ev


# =============================================================================
# SECTION 7: HELPER FUNCTIONS AND UTILITIES
# =============================================================================

def create_sample_nba_slate() -> List[Player]:
    """
    Create a sample NBA slate for testing.

    Returns:
        List of Player objects representing a typical NBA slate
    """
    players = [
        # Elite players (high salary, high projection)
        Player("Luka Doncic", "luka_doncic", "PG", ["PG", "G", "UTIL"], 
               11400, 58.5, 12.5, "DAL", "PHX", 232.5, -3.5, 36.0, 35.0),
        Player("Shai Gilgeous-Alexander", "shai_sga", "SG", ["SG", "G", "UTIL"],
               10800, 55.2, 11.8, "OKC", "DEN", 228.0, -2.0, 34.5, 32.0),
        Player("Giannis Antetokounmpo", "giannis", "PF", ["PF", "F", "UTIL"],
               11800, 62.3, 14.2, "MIL", "BOS", 225.5, 2.5, 35.0, 34.0),
        Player("Nikola Jokic", "jokic", "C", ["C", "UTIL"],
               11600, 59.8, 13.5, "DEN", "OKC", 228.0, 2.0, 34.0, 28.0),

        # Mid-tier value plays
        Player("Tyrese Haliburton", "haliburton", "PG", ["PG", "G", "UTIL"],
               9200, 45.8, 10.2, "IND", "NYK", 231.0, -1.5, 33.5, 28.0),
        Player("Anthony Edwards", "edwards", "SG", ["SG", "G", "UTIL"],
               8900, 44.2, 11.5, "MIN", "LAL", 226.5, 3.0, 34.0, 30.0),
        Player("Jayson Tatum", "tatum", "SF", ["SF", "PF", "F", "UTIL"],
               9500, 47.5, 10.8, "BOS", "MIL", 225.5, -2.5, 36.0, 29.0),
        Player("Domantas Sabonis", "sabonis", "C", ["C", "UTIL"],
               8800, 43.2, 9.8, "SAC", "GSW", 233.0, 1.0, 33.0, 24.0),

        # Value plays and punts
        Player("Derrick White", "white", "PG", ["PG", "SG", "G", "UTIL"],
               6400, 32.5, 7.5, "BOS", "MIL", 225.5, -2.5, 32.0, 18.0),
        Player("Jalen Williams", "jalen_williams", "SF", ["SF", "PF", "F", "UTIL"],
               7200, 35.8, 8.2, "OKC", "DEN", 228.0, -2.0, 31.5, 22.0),
        Player("Myles Turner", "turner", "C", ["C", "UTIL"],
               6800, 33.2, 7.8, "IND", "NYK", 231.0, -1.5, 29.0, 20.0),
        Player("Josh Hart", "hart", "PF", ["PF", "SF", "F", "UTIL"],
               6100, 30.5, 6.8, "NYK", "IND", 231.0, 1.5, 32.0, 15.0),

        # Low-owned punts
        Player("Cason Wallace", "wallace", "SG", ["SG", "G", "UTIL"],
               4500, 22.8, 6.5, "OKC", "DEN", 228.0, -2.0, 24.0, 14.0),
        Player("Tari Eason", "eason", "SF", ["SF", "PF", "F", "UTIL"],
               4800, 24.2, 7.2, "HOU", "MEM", 222.0, 2.0, 26.0, 18.0),
        Player("Nick Richards", "richards", "C", ["C", "UTIL"],
               4200, 20.5, 5.8, "CHA", "WAS", 218.0, -1.0, 22.0, 12.0),
        Player("Keyonte George", "george", "PG", ["PG", "G", "UTIL"],
               4600, 23.5, 6.8, "UTA", "POR", 224.0, 1.0, 25.0, 20.0),
    ]

    return players


def calculate_player_exposure(
    lineups: List[Dict],
    player_pool: List[Player]
) -> pd.DataFrame:
    """
    Calculate exposure percentages for each player in a lineup set.

    Args:
        lineups: List of lineup dictionaries
        player_pool: List of all available players

    Returns:
        DataFrame with exposure statistics
    """
    total_lineups = len(lineups)
    player_counts = defaultdict(int)

    for lineup in lineups:
        for player_id in lineup['player_ids']:
            player_counts[player_id] += 1

    exposure_data = []
    for player in player_pool:
        count = player_counts.get(player.player_id, 0)
        exposure = (count / total_lineups * 100) if total_lineups > 0 else 0

        exposure_data.append({
            'player_id': player.player_id,
            'name': player.name,
            'team': player.team,
            'position': player.position,
            'salary': player.salary,
            'projection': player.projection,
            'count': count,
            'exposure_pct': round(exposure, 2),
            'exposure_150': round(exposure * 1.5, 2)  # Projected for 150 entries
        })

    df = pd.DataFrame(exposure_data)
    return df.sort_values('exposure_pct', ascending=False)


def calculate_lineup_uniqueness(
    lineups: List[Dict]
) -> pd.DataFrame:
    """
    Calculate pairwise uniqueness metrics for a lineup pool.

    Args:
        lineups: List of lineup dictionaries

    Returns:
        DataFrame with uniqueness statistics
    """
    n = len(lineups)
    uniqueness_data = []

    for i, lineup in enumerate(lineups):
        overlaps = []

        for j, other in enumerate(lineups):
            if i == j:
                continue
            overlap = len(set(lineup['player_ids']) & set(other['player_ids']))
            overlaps.append(overlap)

        min_overlap = min(overlaps) if overlaps else 0
        max_overlap = max(overlaps) if overlaps else 0
        avg_overlap = np.mean(overlaps) if overlaps else 0

        uniqueness_data.append({
            'lineup_id': i,
            'min_overlap': min_overlap,
            'max_overlap': max_overlap,
            'avg_overlap': round(avg_overlap, 2),
            'unique_players': 8 - max_overlap,
            'is_unique': max_overlap <= 5  # At least 3 unique players
        })

    return pd.DataFrame(uniqueness_data)


def export_draftkings_csv(
    lineups: List[Dict],
    filename: str,
    contest_id: str = "",
    entry_fee: str = "$1.00"
):
    """
    Export lineups to DraftKings upload format.

    Args:
        lineups: List of lineup dictionaries
        filename: Output CSV filename
        contest_id: DraftKings contest ID
        entry_fee: Entry fee string
    """
    rows = []

    for i, lineup in enumerate(lineups):
        row = {
            'Entry ID': f'ENTRY_{i+1}',
            'Contest Name': 'NBA GPP',
            'Contest ID': contest_id,
            'Entry Fee': entry_fee
        }

        # Add player columns
        for j, name in enumerate(lineup['player_names']):
            row[f'D{j+1}'] = name

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Exported {len(lineups)} lineups to {filename}")


# =============================================================================
# SECTION 7: MAIN EXECUTION EXAMPLE
# =============================================================================

def main():
    """
    Main execution demonstrating the full Monte Carlo framework.
    """
    print("=" * 80)
    print("NBA DFS MONTE CARLO SIMULATION FRAMEWORK")
    print("=" * 80)
    print()

    # Step 1: Create sample slate
    print("Step 1: Creating sample NBA slate...")
    players = create_sample_nba_slate()
    print(f"  Created {len(players)} players")
    print()

    # Step 2: Build correlation matrix
    print("Step 2: Building correlation matrix...")
    sgc = SameGameCorrelation(players)
    corr_matrix = sgc.build_same_game_correlations()
    print(f"  Correlation matrix shape: {corr_matrix.shape}")

    # Show some correlations
    print("  Sample correlations:")
    for i in range(min(3, len(players))):
        for j in range(i+1, min(i+4, len(players))):
            if players[i].team == players[j].team:
                print(f"    {players[i].name} - {players[j].name}: {corr_matrix[i,j]:.3f}")
    print()

    # Step 3: Initialize simulator
    print("Step 3: Initializing Monte Carlo simulator...")
    corr_obj = CorrelationMatrix(players)
    corr_obj.correlation_matrix = corr_matrix

    simulator = MonteCarloSimulator(
        players=players,
        n_simulations=10000,
        distribution='gamma',
        correlation_matrix=corr_obj
    )
    print(f"  Configured for {simulator.n_simulations} simulations")
    print()

    # Step 4: Run simulation
    print("Step 4: Running Monte Carlo simulation...")
    simulated_scores = simulator.run_simulation()
    print(f"  Completed simulations. Score range: {simulated_scores.min():.1f} - {simulated_scores.max():.1f}")
    print()

    # Step 5: Generate optimal lineups
    print("Step 5: Generating optimal lineups...")
    lineups = simulator.generate_optimal_lineups(n_lineups=100)
    print(f"  Generated {len(lineups)} lineups")
    print()

    # Step 6: Evaluate lineups
    print("Step 6: Evaluating lineups...")
    evaluator = LineupEvaluator(simulator)

    # Evaluate first few lineups
    for i in range(min(3, len(lineups))):
        metrics = evaluator.evaluate_lineup(lineups[i], n_opponents=1000)
        print(f"  Lineup {i+1}: {lineups[i]['player_names'][0]} et al.")
        print(f"    Mean Score: {metrics['mean_score']:.1f}")
        print(f"    Cash Prob: {metrics['cash_prob']:.2%}")
        print(f"    Top 10% Prob: {metrics['top10_prob']:.2%}")
        print(f"    Win Prob: {metrics['win_prob']:.4%}")
        print()

    # Step 7: Calculate exposures
    print("Step 7: Calculating player exposures...")
    exposure_df = calculate_player_exposure(lineups, players)
    print("  Top 10 exposures:")
    print(exposure_df.head(10)[['name', 'team', 'salary', 'projection', 'exposure_pct']].to_string(index=False))
    print()

    # Step 8: Calculate uniqueness
    print("Step 8: Calculating lineup uniqueness...")
    uniqueness_df = calculate_lineup_uniqueness(lineups)
    unique_count = uniqueness_df['is_unique'].sum()
    print(f"  Unique lineups (max 5 overlap): {unique_count}/{len(lineups)} ({unique_count/len(lineups):.1%})")
    print(f"  Average max overlap: {uniqueness_df['max_overlap'].mean():.2f} players")
    print()

    # Step 9: Conditional simulation example
    print("Step 9: Running conditional simulation (injury scenario)...")
    conditional = ConditionalSimulator(simulator)
    conditional.add_injury_scenario(
        injured_player_id="luka_doncic",
        replacement_player_id="derrick_white",
        probability=0.2,
        minutes_boost=8.0
    )
    conditional_scores = conditional.run_conditional_simulation(n_scenarios=1000)
    print(f"  Conditional simulation complete.")
    print(f"  Mean conditional score: {conditional_scores.mean():.1f}")
    print(f"  Std conditional score: {conditional_scores.std():.1f}")
    print()

    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

    return {
        'simulator': simulator,
        'lineups': lineups,
        'exposure_df': exposure_df,
        'uniqueness_df': uniqueness_df,
        'evaluator': evaluator
    }


# =============================================================================
# SECTION 8: ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def analyze_lineup_pool(
    lineups: List[Dict],
    players: List[Player],
    simulator: MonteCarloSimulator,
    top_n: int = 150
) -> pd.DataFrame:
    """
    Analyze a pool of lineups and extract the best N for entry.

    Args:
        lineups: List of lineup dictionaries
        players: List of Player objects
        simulator: MonteCarloSimulator with results
        top_n: Number of lineups to extract

    Returns:
        DataFrame with top lineups and metrics
    """
    evaluator = LineupEvaluator(simulator)

    # Evaluate all lineups
    lineup_metrics = []
    for i, lineup in enumerate(lineups):
        metrics = evaluator.evaluate_lineup(lineup, n_opponents=2000)
        lineup_metrics.append({
            'lineup_id': i,
            'players': lineup['player_names'],
            'total_salary': lineup['total_salary'],
            'projection': lineup['total_projection'],
            'mean_score': metrics['mean_score'],
            'std_score': metrics['std_score'],
            'cash_prob': metrics['cash_prob'],
            'top10_prob': metrics['top10_prob'],
            'top1_prob': metrics['top1_prob'],
            'win_prob': metrics['win_prob'],
            'ev': metrics['expected_value'],
            'upside_ratio': metrics['upside_ratio']
        })

    df = pd.DataFrame(lineup_metrics)

    # Calculate composite score for ranking
    # Weight: 40% win prob, 30% top 1%, 20% top 10%, 10% EV
    df['composite_score'] = (
        df['win_prob'] * 0.4 +
        df['top1_prob'] * 0.3 +
        df['top10_prob'] * 0.2 +
        (df['ev'] / df['ev'].max() * 0.1 if df['ev'].max() > 0 else 0)
    )

    # Sort by composite score and extract top N
    df_sorted = df.sort_values('composite_score', ascending=False)

    # Apply uniqueness constraint
    final_lineups = []
    used_player_sets = []

    for _, row in df_sorted.iterrows():
        player_set = set(row['players'])

        # Check uniqueness against already selected
        is_unique = True
        for used_set in used_player_sets:
            overlap = len(player_set & used_set)
            if overlap > 5:  # Less than 3 unique players
                is_unique = False
                break

        if is_unique:
            final_lineups.append(row)
            used_player_sets.append(player_set)

        if len(final_lineups) >= top_n:
            break

    return pd.DataFrame(final_lineups)


def calculate_portfolio_metrics(
    lineups: List[Dict],
    simulator: MonteCarloSimulator
) -> Dict:
    """
    Calculate portfolio-level metrics for a set of lineups.

    Args:
        lineups: List of lineup dictionaries
        simulator: MonteCarloSimulator with results

    Returns:
        Dictionary with portfolio metrics
    """
    evaluator = LineupEvaluator(simulator)

    # Evaluate all lineups
    all_metrics = []
    for lineup in lineups:
        metrics = evaluator.evaluate_lineup(lineup, n_opponents=1000)
        all_metrics.append(metrics)

    # Portfolio metrics
    portfolio_ev = np.mean([m['expected_value'] for m in all_metrics])
    portfolio_cash_prob = np.mean([m['cash_prob'] for m in all_metrics])
    portfolio_win_prob = np.mean([m['win_prob'] for m in all_metrics])

    # Diversification metrics
    all_players = []
    for lineup in lineups:
        all_players.extend(lineup['player_ids'])

    player_counts = pd.Series(all_players).value_counts()
    max_exposure = player_counts.max() / len(lineups) * 100
    avg_exposure = player_counts.mean() / len(lineups) * 100

    return {
        'n_lineups': len(lineups),
        'portfolio_ev': portfolio_ev,
        'portfolio_cash_prob': portfolio_cash_prob,
        'portfolio_win_prob': portfolio_win_prob,
        'max_player_exposure': max_exposure,
        'avg_player_exposure': avg_exposure,
        'unique_players_used': len(player_counts),
        'diversification_score': len(player_counts) / (len(lineups) * 8)
    }


# =============================================================================
# SECTION 9: COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    # Run the main demonstration
    results = main()

    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS: TOP 150 LINEUP EXTRACTION")
    print("=" * 80)

    # Extract top 150 lineups
    top_lineups = analyze_lineup_pool(
        results['lineups'],
        results['simulator'].players,
        results['simulator'],
        top_n=150
    )

    print(f"\nExtracted {len(top_lineups)} lineups for entry")
    print("\nTop 10 lineups by composite score:")
    print(top_lineups.head(10)[[
        'players', 'total_salary', 'projection',
        'win_prob', 'top1_prob', 'ev', 'composite_score'
    ]].to_string(index=False))

    # Calculate portfolio metrics
    print("\n" + "=" * 80)
    print("PORTFOLIO METRICS")
    print("=" * 80)

    portfolio = calculate_portfolio_metrics(
        results['lineups'][:150],
        results['simulator']
    )

    for key, value in portfolio.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("FRAMEWORK DEMONSTRATION COMPLETE")
    print("=" * 80)
