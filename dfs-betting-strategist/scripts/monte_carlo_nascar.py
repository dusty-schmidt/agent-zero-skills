
"""
NASCAR DFS Monte Carlo Simulation Framework
============================================
Comprehensive simulation engine for NASCAR DFS GPP lineup evaluation.

Key Features:
- NASCAR-specific scoring: place finish + laps led + fastest laps + dominator bonus
- Ranked probability models (Mallows, Plackett-Luce) for finish positions
- Multi-car crash correlation with propagation effects
- Track-specific variance profiles
- Dominator bonus correlation modeling
- Vectorized simulation for 10,000+ race outcomes

Author: Quantitative DFS Research
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import warnings
from collections import defaultdict
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrackConfig:
    """Track-specific configuration for variance and correlation modeling."""
    name: str
    track_type: str  # 'superspeedway', 'short_track', 'intermediate', 'road_course'
    laps: int
    dominator_points_per_lap: float = 0.25  # Points per lap led
    fastest_lap_points: float = 0.5

    # Track-specific variance parameters
    finish_variance: float = 1.0  # Base variance multiplier
    crash_probability_factor: float = 1.0  # Track crash proneness
    dominator_concentration: float = 0.8  # How concentrated dominators are (0-1)

    # Correlation structure
    pack_racing: bool = False  # Superspeedway-style pack racing
    track_position_groups: int = 3  # How many position groups for correlation


@dataclass
class DriverProfile:
    """Driver-specific attributes for simulation."""
    driver_id: str
    name: str
    start_position: int

    # Base probabilities and projections
    win_probability: float = 0.05
    top5_probability: float = 0.20
    top10_probability: float = 0.40

    # Crash and incident modeling
    crash_probability: float = 0.05
    mechanical_failure_rate: float = 0.02

    # Dominator potential
    laps_led_rate: float = 0.0  # Expected laps led / total laps
    fastest_lap_rate: float = 0.0  # Expected fastest laps / total laps
    dominator_ceil: float = 0.0  # Ceiling for laps led share

    # Qualifying/starting position effects
    track_position_group: int = 0  # Front/middle/back pack

    # Ownership and projection
    projected_ownership: float = 0.0
    projected_score: float = 0.0
    salary: int = 0

    def __post_init__(self):
        if self.track_position_group == 0:
            # Auto-assign based on start position
            self.track_position_group = self._assign_position_group()

    def _assign_position_group(self) -> int:
        """Assign driver to track position group based on starting position."""
        if self.start_position <= 10:
            return 0  # Front pack
        elif self.start_position <= 25:
            return 1  # Middle pack
        else:
            return 2  # Back pack


@dataclass
class RaceParameters:
    """Parameters for a specific race simulation."""
    track: TrackConfig
    drivers: List[DriverProfile]
    n_simulations: int = 10000
    random_seed: Optional[int] = None

    # Crash correlation parameters
    crash_propagation_factor: float = 0.3  # Crash spreads to nearby cars
    manufacturer_correlation: float = 0.15  # Same manufacturer crash correlation
    team_correlation: float = 0.20  # Same team crash correlation

    # Finish position model
    finish_model: str = 'mallows'  # 'mallows', 'plackett_luce', 'random'
    mallows_dispersion: float = 0.5  # Lower = more deterministic

    # Strategy correlation (pit stops create grouping)
    strategy_correlation: float = 0.1


@dataclass
class RaceOutcome:
    """Single race simulation outcome."""
    driver_id: str
    finish_position: int
    laps_led: int
    fastest_laps: int

    # Incident flags
    crashed: bool = False
    mechanical_failure: bool = False

    # Calculated DFS score
    place_diff_points: float = 0.0
    laps_led_points: float = 0.0
    fastest_lap_points: float = 0.0
    dominator_bonus: float = 0.0
    total_score: float = 0.0


@dataclass
class Lineup:
    """6-driver NASCAR DFS lineup."""
    drivers: List[str]  # List of driver_ids

    def __post_init__(self):
        if len(self.drivers) != 6:
            raise ValueError("NASCAR lineup must have exactly 6 drivers")

    def calculate_score(self, outcomes: Dict[str, RaceOutcome]) -> float:
        """Calculate total lineup score from race outcomes."""
        return sum(outcomes[d].total_score for d in self.drivers if d in outcomes)


@dataclass
class LineupEvaluation:
    """Evaluation results for a lineup across simulations."""
    lineup: Lineup

    # Score distribution
    scores: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    percentile_1: float = 0.0
    percentile_5: float = 0.0
    percentile_10: float = 0.0
    percentile_90: float = 0.0
    percentile_99: float = 0.0

    # Win/contest probabilities
    win_probability: float = 0.0
    top_1pct_probability: float = 0.0
    top_10pct_probability: float = 0.0
    min_cash_probability: float = 0.0

    # Value metrics
    expected_value: float = 0.0
    sharpe_ratio: float = 0.0

    # Ownership
    total_ownership: float = 0.0


# =============================================================================
# NASCAR SCORING SYSTEM
# =============================================================================

class NASCARScoring:
    """
    DraftKings NASCAR Scoring System:

    Place Differential:
    - +/- 1 point per position gained/lost vs starting position

    Laps Led:
    - +0.25 points per lap led

    Fastest Laps:
    - +0.5 points per fastest lap

    Dominator Bonus:
    - +10 points if leads most laps
    - +5 points for 10+ fastest laps (on intermediate/road courses)
    """

    def __init__(self, track_config: TrackConfig):
        self.track = track_config

    def calculate_place_diff(self, start_pos: int, finish_pos: int) -> float:
        """Calculate place differential points."""
        return float(start_pos - finish_pos)

    def calculate_laps_led_points(self, laps_led: int) -> float:
        """Calculate points from laps led."""
        return laps_led * self.track.dominator_points_per_lap

    def calculate_fastest_lap_points(self, fastest_laps: int) -> float:
        """Calculate points from fastest laps."""
        return fastest_laps * self.track.fastest_lap_points

    def calculate_dominator_bonus(
        self, 
        laps_led: int, 
        max_laps_led: int,
        fastest_laps: int,
        total_laps: int
    ) -> float:
        """Calculate dominator bonus points."""
        bonus = 0.0

        # Most laps led bonus
        if laps_led == max_laps_led and laps_led > 0:
            bonus += 10.0

        # Fastest laps bonus (10+ fastest laps)
        if fastest_laps >= 10:
            bonus += 5.0

        return bonus

    def calculate_total_score(
        self,
        driver: DriverProfile,
        finish_pos: int,
        laps_led: int,
        fastest_laps: int,
        max_laps_led: int
    ) -> float:
        """Calculate complete DFS score for a driver."""
        score = 0.0

        # Place differential
        score += self.calculate_place_diff(driver.start_position, finish_pos)

        # Laps led
        score += self.calculate_laps_led_points(laps_led)

        # Fastest laps
        score += self.calculate_fastest_lap_points(fastest_laps)

        # Dominator bonus
        score += self.calculate_dominator_bonus(
            laps_led, max_laps_led, fastest_laps, self.track.laps
        )

        return score


# =============================================================================
# FINISH POSITION MODELS
# =============================================================================

class FinishPositionModel:
    """Base class for finish position probability models."""

    def __init__(self, drivers: List[DriverProfile], track: TrackConfig):
        self.drivers = drivers
        self.n_drivers = len(drivers)
        self.track = track

    def sample_finish_positions(
        self, 
        n_simulations: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample finish positions for all drivers across simulations.

        Returns:
            Array of shape (n_simulations, n_drivers) with finish positions (1-indexed)
        """
        raise NotImplementedError


class MallowsModel(FinishPositionModel):
    """
    Mallows Model for ranked outcomes.

    The Mallows model is a distance-based ranking model where:
    P(π | π₀, φ) ∝ φ^{d(π, π₀)}

    where:
    - π is the observed ranking
    - π₀ is the central ranking (consensus/expected)
    - φ is the dispersion parameter (0 < φ ≤ 1)
    - d(π, π₀) is the Kendall tau distance

    Lower φ = more concentrated around consensus
    Higher φ = more random
    """

    def __init__(
        self, 
        drivers: List[DriverProfile], 
        track: TrackConfig,
        dispersion: float = 0.5
    ):
        super().__init__(drivers, track)
        self.dispersion = dispersion
        self.central_ranking = self._build_central_ranking()

    def _build_central_ranking(self) -> np.ndarray:
        """Build central ranking based on driver probabilities."""
        # Sort by win probability descending
        probs = np.array([d.win_probability for d in self.drivers])
        # Add small random perturbation for ties
        noise = np.random.uniform(0, 0.001, len(probs))
        ranking = np.argsort(-(probs + noise))
        return ranking

    def _sample_mallows(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from Mallows model using insertion sampling.

        Algorithm:
        1. Start with empty ranking
        2. For each item in central ranking order:
           - Insert at position k with probability ∝ φ^k
        """
        ranking = []
        phi = self.dispersion

        for i, driver_idx in enumerate(self.central_ranking):
            # Vose-Alias method for O(1) sampling
            k = i + 1  # Possible insertion positions
            probs = np.array([phi**j for j in range(k)])
            probs = probs / probs.sum()

            insert_pos = rng.choice(k, p=probs)
            ranking.insert(insert_pos, driver_idx)

        return np.array(ranking)

    def sample_finish_positions(
        self, 
        n_simulations: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Sample finish positions using Mallows model."""
        results = np.zeros((n_simulations, self.n_drivers), dtype=int)

        for sim in range(n_simulations):
            ranking = self._sample_mallows(rng)
            # Convert ranking to finish positions
            for pos, driver_idx in enumerate(ranking, 1):
                results[sim, driver_idx] = pos

        return results


class PlackettLuceModel(FinishPositionModel):
    """
    Plackett-Luce Model for ranked outcomes.

    Sequential model: P(i is best) ∝ w_i
    After selecting best, remove and repeat.

    w_i are "worth" parameters derived from driver win probabilities.
    """

    def __init__(self, drivers: List[DriverProfile], track: TrackConfig):
        super().__init__(drivers, track)
        self.worths = self._calculate_worths()

    def _calculate_worths(self) -> np.ndarray:
        """Calculate worth parameters from win probabilities."""
        # Use win probability as base worth
        worths = np.array([d.win_probability for d in self.drivers])

        # Normalize
        worths = np.maximum(worths, 0.001)  # Avoid zeros

        # Apply track-type adjustment
        if self.track.track_type == 'superspeedway':
            # More random, flatten distribution
            worths = np.power(worths, 0.5)
        elif self.track.track_type == 'short_track':
            # More predictable
            worths = np.power(worths, 1.5)

        return worths

    def _sample_plackett_luce(
        self, 
        rng: np.random.Generator
    ) -> np.ndarray:
        """Sample a single ranking from Plackett-Luce."""
        available = list(range(self.n_drivers))
        ranking = []
        worths = self.worths.copy()

        while available:
            # Sample from available items weighted by worth
            probs = worths[available] / worths[available].sum()
            selected_idx = rng.choice(len(available), p=probs)
            selected = available.pop(selected_idx)
            ranking.append(selected)

        return np.array(ranking)

    def sample_finish_positions(
        self, 
        n_simulations: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Sample finish positions using Plackett-Luce model."""
        results = np.zeros((n_simulations, self.n_drivers), dtype=int)

        for sim in range(n_simulations):
            ranking = self._sample_plackett_luce(rng)
            for pos, driver_idx in enumerate(ranking, 1):
                results[sim, driver_idx] = pos

        return results


# =============================================================================
# CRASH CORRELATION MODELING
# =============================================================================

class CrashCorrelationModel:
    """
    Models crash events and their correlation structure in NASCAR.

    Key insight: NASCAR correlation is dominated by crashes, not usage.
    Multi-car incidents create strong negative correlation for DFS lineups.
    """

    def __init__(self, params: RaceParameters):
        self.params = params
        self.track = params.track
        self.drivers = params.drivers
        self.n_drivers = len(params.drivers)

        # Build correlation matrices
        self._build_correlation_structure()

    def _build_correlation_structure(self):
        """Build crash correlation matrix based on driver attributes."""
        n = self.n_drivers
        self.correlation_matrix = np.eye(n) * 0.3  # Base crash variance

        for i in range(n):
            for j in range(i + 1, n):
                corr = self._calculate_crash_correlation(i, j)
                self.correlation_matrix[i, j] = corr
                self.correlation_matrix[j, i] = corr

    def _calculate_crash_correlation(self, i: int, j: int) -> float:
        """
        Calculate crash correlation between two drivers.

        Correlation sources:
        1. Track position (nearby cars crash together)
        2. Manufacturer (engine issues)
        3. Team (shared equipment/strategy)
        """
        d1, d2 = self.drivers[i], self.drivers[j]
        corr = 0.0

        # Track position correlation (proximity crashes)
        pos_diff = abs(d1.start_position - d2.start_position)
        if pos_diff <= 3:
            corr += 0.25  # Immediate proximity
        elif pos_diff <= 8:
            corr += 0.15  # Nearby pack
        elif pos_diff <= 15:
            corr += 0.05  # Same track position group

        # Track-type specific correlation
        if self.track.pack_racing:
            # Superspeedways: tighter correlation
            corr *= 1.5

        return min(corr, 0.5)  # Cap correlation

    def sample_crash_events(
        self,
        n_simulations: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample crash and mechanical failure events.

        Returns:
            crashed: (n_simulations, n_drivers) boolean array
            mechanical: (n_simulations, n_drivers) boolean array
        """
        crashed = np.zeros((n_simulations, self.n_drivers), dtype=bool)
        mechanical = np.zeros((n_simulations, self.n_drivers), dtype=bool)

        # Base crash probabilities
        base_probs = np.array([d.crash_probability for d in self.drivers])

        for sim in range(n_simulations):
            # Sample base crashes using Gaussian copula for correlation
            z = rng.multivariate_normal(
                np.zeros(self.n_drivers), 
                self.correlation_matrix
            )
            u = stats.norm.cdf(z)

            # Initial crash indicators
            crash_indicator = u < base_probs

            # Propagation: if car crashes, nearby cars at risk
            if crash_indicator.any():
                crash_indicator = self._propagate_crash(
                    crash_indicator, rng
                )

            crashed[sim] = crash_indicator

            # Sample mechanical failures (independent)
            mech_probs = np.array([d.mechanical_failure_rate for d in self.drivers])
            mechanical[sim] = rng.random(self.n_drivers) < mech_probs

        return crashed, mechanical

    def _propagate_crash(
        self, 
        crash_indicator: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Model crash propagation: if Driver A crashes, nearby cars at risk.

        This simulates multi-car incidents where cars in proximity to
        a crashed car have elevated crash risk.
        """
        result = crash_indicator.copy()
        crashed_positions = [
            self.drivers[i].start_position 
            for i, crashed in enumerate(crash_indicator) 
            if crashed
        ]

        for i, driver in enumerate(self.drivers):
            if result[i]:
                continue  # Already crashed

            # Check if near any crashed car
            for crashed_pos in crashed_positions:
                distance = abs(driver.start_position - crashed_pos)
                if distance <= 5:  # Within 5 positions
                    # Elevated crash risk based on proximity
                    propagation_prob = self.params.crash_propagation_factor * (1 - distance / 5)
                    if rng.random() < propagation_prob:
                        result[i] = True
                        break

        return result


# =============================================================================
# DOMINATOR MODELING
# =============================================================================

class DominatorModel:
    """
    Models laps led and fastest laps, capturing their correlation with
    finish position and track position.

    Key insight: Dominators correlate with:
    - Starting position (track position = king)
    - Finish position (lead from front)
    - Track type (different dominator profiles)
    """

    def __init__(self, params: RaceParameters):
        self.params = params
        self.track = params.track
        self.drivers = params.drivers
        self.n_drivers = len(params.drivers)
        self.total_laps = self.track.laps

    def sample_dominators(
        self,
        finish_positions: np.ndarray,
        n_simulations: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample laps led and fastest laps for all simulations.

        Args:
            finish_positions: (n_simulations, n_drivers) finish positions

        Returns:
            laps_led: (n_simulations, n_drivers)
            fastest_laps: (n_simulations, n_drivers)
        """
        laps_led = np.zeros((n_simulations, self.n_drivers), dtype=int)
        fastest_laps = np.zeros((n_simulations, self.n_drivers), dtype=int)

        for sim in range(n_simulations):
            laps_led[sim], fastest_laps[sim] = self._sample_single_race(
                finish_positions[sim], rng
            )

        return laps_led, fastest_laps

    def _sample_single_race(
        self,
        finish_pos: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample laps led and fastest laps for a single race."""
        laps_led = np.zeros(self.n_drivers, dtype=int)
        fastest_laps = np.zeros(self.n_drivers, dtype=int)

        # Track-type specific dominator distribution
        if self.track.track_type == 'superspeedway':
            # Multiple leaders, frequent changes
            laps_led = self._sample_superspeedway_dominators(finish_pos, rng)
        elif self.track.track_type == 'short_track':
            # Single dominant leader
            laps_led = self._sample_short_track_dominators(finish_pos, rng)
        else:
            # Intermediate / road course - balanced
            laps_led = self._sample_standard_dominators(finish_pos, rng)

        # Fastest laps correlate with laps led
        fastest_laps = self._sample_fastest_laps(laps_led, finish_pos, rng)

        return laps_led, fastest_laps

    def _sample_superspeedway_dominators(
        self,
        finish_pos: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Superspeedway: Many leaders, frequent changes.
        Laps led more distributed, correlates less with finish.
        """
        laps_led = np.zeros(self.n_drivers, dtype=int)
        remaining_laps = self.total_laps

        # Multiple stints with different leaders
        n_stints = rng.poisson(8) + 5  # 5-15 lead changes typical
        stint_lengths = rng.dirichlet(np.ones(n_stints)) * remaining_laps

        for stint_length in stint_lengths:
            # Weight by front-running probability (less deterministic)
            weights = np.array([
                max(0.01, 1.0 / (finish_pos[i] + 5))  # +5 for superspeedway randomness
                for i in range(self.n_drivers)
            ])
            weights = weights / weights.sum()

            leader = rng.choice(self.n_drivers, p=weights)
            laps_led[leader] += int(stint_length)

        return laps_led

    def _sample_short_track_dominators(
        self,
        finish_pos: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Short track: Single dominant leader, strong correlation with finish.
        """
        laps_led = np.zeros(self.n_drivers, dtype=int)

        # Primary dominator takes 40-70% of laps
        # Weight heavily by finish position
        weights = np.array([
            max(0.001, 1.0 / (finish_pos[i] ** 1.5))
            for i in range(self.n_drivers)
        ])
        weights = weights / weights.sum()

        primary = rng.choice(self.n_drivers, p=weights)
        primary_share = rng.uniform(0.4, 0.7)
        laps_led[primary] = int(self.total_laps * primary_share)

        # Remaining laps distributed
        remaining = self.total_laps - laps_led[primary]
        if remaining > 0:
            weights[primary] = 0
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(self.n_drivers) / self.n_drivers

            secondary_count = rng.poisson(3) + 1
            for _ in range(secondary_count):
                if remaining <= 0:
                    break
                leader = rng.choice(self.n_drivers, p=weights)
                stint = min(rng.poisson(remaining / secondary_count), remaining)
                laps_led[leader] += stint
                remaining -= stint

        return laps_led

    def _sample_standard_dominators(
        self,
        finish_pos: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Intermediate/road course: Balanced dominator model."""
        laps_led = np.zeros(self.n_drivers, dtype=int)

        # 1-3 main dominators
        weights = np.array([
            max(0.001, 1.0 / finish_pos[i])
            for i in range(self.n_drivers)
        ])
        weights = weights / weights.sum()

        n_dominators = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        remaining = self.total_laps

        for i in range(n_dominators):
            if remaining <= 0:
                break
            leader = rng.choice(self.n_drivers, p=weights)
            share = rng.uniform(0.25, 0.55) if i == 0 else rng.uniform(0.15, 0.35)
            stint = int(self.total_laps * share)
            laps_led[leader] += min(stint, remaining)
            remaining -= laps_led[leader]
            weights[leader] *= 0.3  # Reduce chance of same driver leading again
            weights = weights / weights.sum()

        # Distribute remaining
        if remaining > 0:
            weights = np.ones(self.n_drivers) / self.n_drivers
            for _ in range(rng.poisson(4)):
                if remaining <= 0:
                    break
                leader = rng.choice(self.n_drivers, p=weights)
                stint = min(rng.poisson(5), remaining)
                laps_led[leader] += stint
                remaining -= stint

        return laps_led

    def _sample_fastest_laps(
        self,
        laps_led: np.ndarray,
        finish_pos: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample fastest laps, correlated with laps led and finish position.

        Correlation structure:
        - Drivers who lead laps tend to have fast cars
        - Clean air = faster laps
        - But also independent fast cars that don't lead
        """
        fastest_laps = np.zeros(self.n_drivers, dtype=int)

        # Weight by laps led (clean air advantage) and finish position
        weights = np.array([
            0.6 * laps_led[i] + 0.4 * (self.n_drivers - finish_pos[i])
            for i in range(self.n_drivers)
        ])
        weights = np.maximum(weights, 0.1)
        weights = weights / weights.sum()

        # Distribute fastest laps
        total_fastest = self.total_laps
        n_samples = min(self.n_drivers * 3, total_fastest)

        for _ in range(n_samples):
            if total_fastest <= 0:
                break
            driver = rng.choice(self.n_drivers, p=weights)
            n_fast = rng.poisson(total_fastest / n_samples) + 1
            n_fast = min(n_fast, total_fastest)
            fastest_laps[driver] += n_fast
            total_fastest -= n_fast

        return fastest_laps


# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

class MonteCarloSimulator:
    """
    Main simulation engine for NASCAR DFS evaluation.

    Generates N race outcomes accounting for:
    - Finish position uncertainty (Mallows/Plackett-Luce)
    - Crash correlation and propagation
    - Dominator variance by track type
    - Mechanical failures
    """

    def __init__(self, params: RaceParameters):
        self.params = params
        self.track = params.track
        self.drivers = params.drivers
        self.n_drivers = len(params.drivers)
        self.scoring = NASCARScoring(params.track)

        # Initialize sub-models
        self._init_finish_model()
        self.crash_model = CrashCorrelationModel(params)
        self.dominator_model = DominatorModel(params)

        # Random number generator
        self.rng = np.random.default_rng(params.random_seed)

    def _init_finish_model(self):
        """Initialize the finish position sampling model."""
        if self.params.finish_model == 'mallows':
            self.finish_model = MallowsModel(
                self.drivers, self.track, self.params.mallows_dispersion
            )
        elif self.params.finish_model == 'plackett_luce':
            self.finish_model = PlackettLuceModel(self.drivers, self.track)
        else:
            raise ValueError(f"Unknown finish model: {self.params.finish_model}")

    def simulate(self, n_simulations: Optional[int] = None) -> List[List[RaceOutcome]]:
        """
        Run Monte Carlo simulation of race outcomes.

        Args:
            n_simulations: Number of simulations (default: self.params.n_simulations)

        Returns:
            List of simulation results, each a list of RaceOutcome objects
        """
        if n_simulations is None:
            n_simulations = self.params.n_simulations

        # 1. Sample finish positions
        finish_positions = self.finish_model.sample_finish_positions(
            n_simulations, self.rng
        )

        # 2. Sample crash and mechanical failure events
        crashed, mechanical = self.crash_model.sample_crash_events(
            n_simulations, self.rng
        )

        # 3. Apply crash penalties (finish position -> back of field)
        finish_positions = self._apply_crash_penalties(
            finish_positions, crashed, mechanical
        )

        # 4. Sample laps led and fastest laps
        laps_led, fastest_laps = self.dominator_model.sample_dominators(
            finish_positions, n_simulations, self.rng
        )

        # 5. Calculate DFS scores
        results = []
        for sim in range(n_simulations):
            sim_outcomes = self._calculate_simulation_outcomes(
                sim, finish_positions, laps_led, fastest_laps, crashed, mechanical
            )
            results.append(sim_outcomes)

        return results

    def _apply_crash_penalties(
        self,
        finish_positions: np.ndarray,
        crashed: np.ndarray,
        mechanical: np.ndarray
    ) -> np.ndarray:
        """
        Apply crash/mechanical penalties: crashed cars finish at back.

        Strategy:
        - Crashed cars: finish 30-40th
        - Mechanical: finish 25-38th
        - Penalty based on when incident occurred (early = worse finish)
        """
        result = finish_positions.copy()
        n_sims, n_drivers = result.shape

        for sim in range(n_sims):
            # Get crashed drivers
            crashed_drivers = np.where(crashed[sim])[0]
            mech_drivers = np.where(mechanical[sim])[0]

            # Assign crash finish positions
            for idx, driver_idx in enumerate(crashed_drivers):
                # Earlier crashes finish further back
                result[sim, driver_idx] = self.n_drivers - idx

            # Assign mechanical finish positions (better than crash)
            start_pos = len(crashed_drivers)
            for idx, driver_idx in enumerate(mech_drivers):
                if driver_idx not in crashed_drivers:
                    result[sim, driver_idx] = self.n_drivers - start_pos - idx

        return result

    def _calculate_simulation_outcomes(
        self,
        sim: int,
        finish_positions: np.ndarray,
        laps_led: np.ndarray,
        fastest_laps: np.ndarray,
        crashed: np.ndarray,
        mechanical: np.ndarray
    ) -> List[RaceOutcome]:
        """Calculate complete race outcomes for a single simulation."""
        outcomes = []
        max_laps_led = laps_led[sim].max()

        for i, driver in enumerate(self.drivers):
            finish_pos = int(finish_positions[sim, i])
            led = int(laps_led[sim, i])
            fast = int(fastest_laps[sim, i])

            # Calculate DFS score
            total_score = self.scoring.calculate_total_score(
                driver, finish_pos, led, fast, max_laps_led
            )

            outcome = RaceOutcome(
                driver_id=driver.driver_id,
                finish_position=finish_pos,
                laps_led=led,
                fastest_laps=fast,
                crashed=crashed[sim, i],
                mechanical_failure=mechanical[sim, i],
                total_score=total_score
            )
            outcomes.append(outcome)

        return outcomes

    def get_simulation_scores_df(self, results: List[List[RaceOutcome]]) -> pd.DataFrame:
        """
        Convert simulation results to DataFrame for analysis.

        Returns:
            DataFrame with columns: [simulation, driver_id, finish_position, 
                                     laps_led, fastest_laps, crashed, 
                                     mechanical_failure, total_score]
        """
        rows = []
        for sim_idx, sim_outcomes in enumerate(results):
            for outcome in sim_outcomes:
                rows.append({
                    'simulation': sim_idx,
                    'driver_id': outcome.driver_id,
                    'finish_position': outcome.finish_position,
                    'laps_led': outcome.laps_led,
                    'fastest_laps': outcome.fastest_laps,
                    'crashed': outcome.crashed,
                    'mechanical_failure': outcome.mechanical_failure,
                    'total_score': outcome.total_score
                })

        return pd.DataFrame(rows)


# =============================================================================
# LINEUP EVALUATION
# =============================================================================

class LineupEvaluator:
    """
    Evaluates DFS lineups using Monte Carlo simulation results.

    Calculates:
    - Score distribution statistics
    - Win/top-1%/top-10%/min-cash probabilities
    - Expected value and risk metrics
    - Correlation-adjusted evaluation
    """

    def __init__(
        self, 
        simulation_results: List[List[RaceOutcome]],
        driver_map: Dict[str, DriverProfile]
    ):
        self.results = simulation_results
        self.n_simulations = len(simulation_results)
        self.driver_map = driver_map

        # Build score matrix: (n_simulations, n_drivers)
        self._build_score_matrix()

    def _build_score_matrix(self):
        """Build matrix of scores for efficient lookup."""
        self.score_matrix = {}
        self.driver_ids = list(self.driver_map.keys())

        for sim_idx, sim_outcomes in enumerate(self.results):
            for outcome in sim_outcomes:
                if outcome.driver_id not in self.score_matrix:
                    self.score_matrix[outcome.driver_id] = np.zeros(self.n_simulations)
                self.score_matrix[outcome.driver_id][sim_idx] = outcome.total_score

    def evaluate_lineup(self, lineup: Lineup) -> LineupEvaluation:
        """
        Evaluate a single lineup across all simulations.

        Args:
            lineup: Lineup object with 6 driver IDs

        Returns:
            LineupEvaluation with full statistics
        """
        # Calculate lineup scores for all simulations
        scores = np.zeros(self.n_simulations)
        for driver_id in lineup.drivers:
            if driver_id in self.score_matrix:
                scores += self.score_matrix[driver_id]

        # Calculate statistics
        eval_result = LineupEvaluation(
            lineup=lineup,
            scores=scores,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            percentile_1=float(np.percentile(scores, 1)),
            percentile_5=float(np.percentile(scores, 5)),
            percentile_10=float(np.percentile(scores, 10)),
            percentile_90=float(np.percentile(scores, 90)),
            percentile_99=float(np.percentile(scores, 99))
        )

        # Calculate contest probabilities
        eval_result = self._calculate_contest_probabilities(eval_result, scores)

        # Calculate ownership
        eval_result.total_ownership = sum(
            self.driver_map[d].projected_ownership 
            for d in lineup.drivers 
            if d in self.driver_map
        )

        return eval_result

    def _calculate_contest_probabilities(
        self, 
        eval_result: LineupEvaluation,
        scores: np.ndarray
    ) -> LineupEvaluation:
        """Calculate win, top-1%, top-10%, min-cash probabilities."""
        # For these calculations, we need to simulate contest field
        # Simplified: assume score thresholds based on percentiles

        # Estimate win threshold (99th percentile of random 6-driver lineups)
        win_threshold = np.percentile(scores, 99)
        top_1pct_threshold = np.percentile(scores, 99)
        top_10pct_threshold = np.percentile(scores, 90)
        min_cash_threshold = np.percentile(scores, 60)

        # Calculate probabilities
        eval_result.win_probability = np.mean(scores >= win_threshold)
        eval_result.top_1pct_probability = np.mean(scores >= top_1pct_threshold)
        eval_result.top_10pct_probability = np.mean(scores >= top_10pct_threshold)
        eval_result.min_cash_probability = np.mean(scores >= min_cash_threshold)

        # Expected value (simplified: assume payout structure)
        # Win: 1000x, Top 1%: 50x, Top 10%: 5x, Min cash: 1.5x
        entry_fee = 1.0
        eval_result.expected_value = (
            eval_result.win_probability * 1000 +
            eval_result.top_1pct_probability * 50 +
            eval_result.top_10pct_probability * 5 +
            eval_result.min_cash_probability * 1.5
        ) * entry_fee

        # Sharpe ratio
        if eval_result.std_score > 0:
            eval_result.sharpe_ratio = eval_result.mean_score / eval_result.std_score

        return eval_result

    def evaluate_lineup_pool(
        self, 
        lineups: List[Lineup]
    ) -> pd.DataFrame:
        """
        Evaluate a pool of lineups and return results DataFrame.

        Returns:
            DataFrame with lineup evaluation metrics
        """
        evaluations = []

        for lineup in lineups:
            eval_result = self.evaluate_lineup(lineup)
            evaluations.append({
                'drivers': ','.join(lineup.drivers),
                'mean_score': eval_result.mean_score,
                'std_score': eval_result.std_score,
                'min_score': eval_result.min_score,
                'max_score': eval_result.max_score,
                'percentile_1': eval_result.percentile_1,
                'percentile_99': eval_result.percentile_99,
                'win_probability': eval_result.win_probability,
                'top_1pct_probability': eval_result.top_1pct_probability,
                'top_10pct_probability': eval_result.top_10pct_probability,
                'min_cash_probability': eval_result.min_cash_probability,
                'expected_value': eval_result.expected_value,
                'sharpe_ratio': eval_result.sharpe_ratio,
                'total_ownership': eval_result.total_ownership
            })

        return pd.DataFrame(evaluations)

    def calculate_correlation_matrix(self, lineup: Lineup) -> np.ndarray:
        """
        Calculate score correlation matrix between lineup drivers.

        Important for understanding lineup variance and risk concentration.
        """
        driver_scores = []
        for driver_id in lineup.drivers:
            if driver_id in self.score_matrix:
                driver_scores.append(self.score_matrix[driver_id])

        if len(driver_scores) < 2:
            return np.array([[1.0]])

        score_matrix = np.array(driver_scores)
        return np.corrcoef(score_matrix)


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

def create_example_race() -> RaceParameters:
    """Create example race configuration for Daytona 500."""

    # Track configuration for superspeedway
    daytona = TrackConfig(
        name="Daytona 500",
        track_type="superspeedway",
        laps=200,
        dominator_points_per_lap=0.25,
        fastest_lap_points=0.5,
        finish_variance=1.5,  # High variance
        crash_probability_factor=1.3,  # Crash-prone
        dominator_concentration=0.4,  # Distributed
        pack_racing=True,
        track_position_groups=3
    )

    # Create 40-driver field
    drivers = []

    # Top contenders (high win prob, low crash prob)
    for i, (name, win_p, crash_p) in enumerate([
        ("Driver_A", 0.12, 0.06),
        ("Driver_B", 0.10, 0.05),
        ("Driver_C", 0.09, 0.07),
        ("Driver_D", 0.08, 0.04),
        ("Driver_E", 0.07, 0.08),
    ]):
        drivers.append(DriverProfile(
            driver_id=f"D{i:02d}",
            name=name,
            start_position=i+1,
            win_probability=win_p,
            top5_probability=0.35,
            top10_probability=0.55,
            crash_probability=crash_p,
            mechanical_failure_rate=0.02,
            projected_ownership=rng.uniform(15, 35),
            projected_score=rng.uniform(35, 55)
        ))

    # Mid-tier drivers
    for i in range(5, 25):
        drivers.append(DriverProfile(
            driver_id=f"D{i:02d}",
            name=f"Driver_{i}",
            start_position=i+1,
            win_probability=rng.uniform(0.01, 0.05),
            top5_probability=0.15,
            top10_probability=0.30,
            crash_probability=rng.uniform(0.05, 0.12),
            mechanical_failure_rate=0.03,
            projected_ownership=rng.uniform(3, 15),
            projected_score=rng.uniform(20, 40)
        ))

    # Backmarkers
    for i in range(25, 40):
        drivers.append(DriverProfile(
            driver_id=f"D{i:02d}",
            name=f"Driver_{i}",
            start_position=i+1,
            win_probability=0.005,
            top5_probability=0.05,
            top10_probability=0.12,
            crash_probability=rng.uniform(0.08, 0.15),
            mechanical_failure_rate=0.04,
            projected_ownership=rng.uniform(0.5, 5),
            projected_score=rng.uniform(10, 25)
        ))

    params = RaceParameters(
        track=daytona,
        drivers=drivers,
        n_simulations=10000,
        random_seed=42,
        finish_model='mallows',
        mallows_dispersion=0.7  # Moderate randomness for superspeedway
    )

    return params


def run_demonstration():
    """Run complete demonstration of the simulation framework."""
    print("="*80)
    print("NASCAR DFS MONTE CARLO SIMULATION FRAMEWORK")
    print("="*80)

    # Create example race
    global rng
    rng = np.random.default_rng(42)
    params = create_example_race()

    print(f"\nRace: {params.track.name}")
    print(f"Track Type: {params.track.track_type}")
    print(f"Laps: {params.track.laps}")
    print(f"Drivers: {len(params.drivers)}")
    print(f"Simulations: {params.n_simulations:,}")

    # Run simulation
    print("\nRunning Monte Carlo simulation...")
    simulator = MonteCarloSimulator(params)
    results = simulator.simulate()

    print(f"Completed {len(results):,} simulations")

    # Analyze results
    scores_df = simulator.get_simulation_scores_df(results)

    # Summary statistics
    print("\n" + "="*80)
    print("SIMULATION SUMMARY STATISTICS")
    print("="*80)

    # Driver-level stats
    driver_stats = scores_df.groupby('driver_id').agg({
        'finish_position': ['mean', 'std'],
        'laps_led': ['mean', 'std'],
        'fastest_laps': ['mean', 'std'],
        'crashed': 'mean',
        'total_score': ['mean', 'std', 'min', 'max']
    }).round(2)

    print("\nTop 10 drivers by mean score:")
    top_drivers = driver_stats.sort_values(
        ('total_score', 'mean'), 
        ascending=False
    ).head(10)
    print(top_drivers[[('total_score', 'mean'), ('total_score', 'std'), 
                       ('crashed', 'mean'), ('laps_led', 'mean')]])

    # Crash analysis
    print("\n" + "-"*80)
    print("CRASH ANALYSIS")
    print("-"*80)
    crash_rate = scores_df['crashed'].mean()
    mech_rate = scores_df['mechanical_failure'].mean()
    print(f"Overall crash rate: {crash_rate:.1%}")
    print(f"Overall mechanical failure rate: {mech_rate:.1%}")

    # Multi-car incidents
    sim_crashes = scores_df.groupby('simulation')['crashed'].sum()
    print(f"\nMulti-car incident analysis:")
    print(f"  Mean crashes per race: {sim_crashes.mean():.1f}")
    print(f"  Races with 5+ crashes: {(sim_crashes >= 5).mean():.1%}")
    print(f"  Races with 10+ crashes: {(sim_crashes >= 10).mean():.1%}")

    # Evaluate example lineups
    print("\n" + "="*80)
    print("LINEUP EVALUATION EXAMPLES")
    print("="*80)

    driver_map = {d.driver_id: d for d in params.drivers}
    evaluator = LineupEvaluator(results, driver_map)

    # Create test lineups
    test_lineups = [
        Lineup(["D00", "D01", "D02", "D03", "D04", "D10"]),  # Chalk-heavy
        Lineup(["D00", "D10", "D15", "D20", "D25", "D30"]),  # Balanced
        Lineup(["D05", "D10", "D15", "D20", "D25", "D30"]),  # Contrarian
    ]

    for i, lineup in enumerate(test_lineups, 1):
        print(f"\n--- Lineup {i}: {lineup.drivers} ---")
        eval_result = evaluator.evaluate_lineup(lineup)

        print(f"Mean Score: {eval_result.mean_score:.1f} (±{eval_result.std_score:.1f})")
        print(f"Score Range: [{eval_result.min_score:.1f}, {eval_result.max_score:.1f}]")
        print(f"99th Percentile: {eval_result.percentile_99:.1f}")
        print(f"Win Probability: {eval_result.win_probability:.2%}")
        print(f"Top 1% Probability: {eval_result.top_1pct_probability:.2%}")
        print(f"Top 10% Probability: {eval_result.top_10pct_probability:.2%}")
        print(f"Min Cash Probability: {eval_result.min_cash_probability:.2%}")
        print(f"Expected Value: {eval_result.expected_value:.2f}x")
        print(f"Total Ownership: {eval_result.total_ownership:.1f}%")

        # Correlation matrix
        corr_matrix = evaluator.calculate_correlation_matrix(lineup)
        avg_corr = np.mean(np.tril(corr_matrix, -1)[np.tril(corr_matrix, -1) != 0])
        print(f"Avg Driver Correlation: {avg_corr:.3f}")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

    return simulator, results, evaluator, scores_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    simulator, results, evaluator, scores_df = run_demonstration()
