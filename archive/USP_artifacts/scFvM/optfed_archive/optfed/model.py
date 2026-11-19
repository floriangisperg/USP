"""
OptFed Mechanistic Model Implementation

Based on: Schlögel et al. (2024) - OptFed: A general framework for data-driven
optimization of fed-batch bioprocesses.

Paper: https://doi.org/10.1016/j.csbj.2024.07.034
bioRxiv: https://www.biorxiv.org/content/10.1101/2024.07.31.605953v2

This module implements Stage I of the framework: the general process model
for E. coli fed-batch fermentation producing recombinant proteins.

Key features:
- Full maintenance model (Eq. 4c from paper)
- Temperature dependence (Eq. 4e from paper) - Q10 and Arrhenius models
- Two-phase feed strategy (exponential pre-induction, linear post-induction)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

# Physical constants for Eyring equation (from paper)
R_GAS = 8.314  # J/(mol·K) - Gas constant
K_B = 1.38e-23  # J/K - Boltzmann constant
H_PLANCK = 2.39e-30  # J*h - Planck constant


@dataclass
class TemperatureDependence:
    """
    Temperature dependence parameters for kinetic rates.

    Reference: Schlögel et al. (2024) - Eq. 4e

    Three models supported:
    1. Q10 model (simplified, commonly used in biology):
       k(T) = k_ref * Q10^((T - T_ref)/10)

    2. Arrhenius model (more rigorous):
       k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))

    3. Eyring model with enzyme inactivation (from paper Eq. 4e):
       k(T) = (E0 * k_B * T / h) * exp(-DG_cat / (R*T)) / (1 + exp(DH_eq / R * (1/T_eq - 1/T)))
    """
    # Temperature dependence model type
    model: str = "none"  # "none", "Q10", "arrhenius", "erying"

    # Reference temperature (C) - used for Q10 and Arrhenius models
    T_ref: float = 37.0

    # Q10 model parameters (simplified Arrhenius for biological systems)
    # k(T) = k_ref * Q10^((T - T_ref)/10)
    # Q10 = 2.0 means rate doubles for every 10C increase
    Q10_gamma_max: float = 2.0     # Temperature sensitivity for total uptake
    Q10_gamma_pi_max: float = 2.0  # Temperature sensitivity for production uptake
    Q10_gamma_alpha_min: float = 2.0  # Temperature sensitivity for maintenance

    # Arrhenius model parameters (more rigorous)
    # k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
    # Ea in J/mol, R = 8.314 J/(mol·K), T in Kelvin
    # Typical Ea for enzyme-catalyzed reactions: 30-100 kJ/mol
    Ea_gamma_max: float = 50000.0      # Activation energy for total uptake (J/mol)
    Ea_gamma_pi_max: float = 50000.0   # Activation energy for production uptake (J/mol)
    Ea_gamma_alpha_min: float = 50000.0  # Activation energy for maintenance (J/mol)

    # Eyring model parameters (from paper Eq. 4e)
    # k(T) = (E0 * k_B * T / h) * exp(-DG_cat / (R*T)) / (1 + exp(DH_eq / R * (1/T_eq - 1/T)))
    # E0: Hypothetical enzyme concentration (dimensionless scaling factor)
    # DG_cat: Catalytic activation energy (J/mol)
    # DH_eq: Enthalpy for enzyme inactivation (J/mol)
    # T_eq: Temperature where half the enzyme is active (K)

    # For gamma_max (total uptake)
    E0_gamma_max: float = 1.0         # Scaling factor
    DG_cat_gamma_max: float = 50000.0  # Catalytic activation energy (J/mol)
    DH_eq_gamma_max: float = 1e6       # Enthalpy for inactivation (J/mol) - not used in reduced model
    T_eq_gamma_max: float = 999.0      # Half-inactivation temp (K) - not used in reduced model

    # For gamma_pi_max (production uptake)
    E0_gamma_pi_max: float = 8.8e-9   # From paper Table 1
    DG_cat_gamma_pi_max: float = 5.2e4  # From paper Table 1 (J/mol)
    DH_eq_gamma_pi_max: float = 4.8e6   # From paper Table 1 (J/mol)
    T_eq_gamma_pi_max: float = 310.0    # From paper Table 1 (K) = 36.85 C

    # For gamma_alpha_min (maintenance)
    E0_gamma_alpha_min: float = 1.0
    DG_cat_gamma_alpha_min: float = 50000.0
    DH_eq_gamma_alpha_min: float = 1e6
    T_eq_gamma_alpha_min: float = 999.0


@dataclass
class ModelParameters:
    """
    Parameters for the OptFed mechanistic model.

    The model has 15-27 free parameters depending on temperature dependence.
    Yields can be derived from genome-scale metabolic models.
    """

    # Yield coefficients
    Y_XG: float = 0.5      # Biomass-to-substrate yield (gX/gG)
    Y_PG: float = 0.2      # Product-to-substrate yield (gP/gG)

    # Substrate uptake parameters (y)
    gamma_max: float = 0.5     # Maximum uptake rate (1/h) at T_ref
    K_m: float = 0.1           # Michaelis-Menten constant (g/L)
    K_G_inhib: float = 100.0   # Glucose inhibition constant (g/L)
    K_P_inhib: float = 10.0    # Product inhibition constant (g/L)
    K_X_inhib: float = 100.0   # Biomass inhibition constant (g/L)

    # Production uptake parameters (yp)
    gamma_pi_max: float = 0.2  # Max production uptake (1/h) at T_ref
    K_pi_m: float = 0.05       # Michaelis constant for production (1/h)

    # Maintenance parameters (ya) - activated by multiple variables
    # y_a = y_a_min * Pi (1 + i/K_a_i) for i in {y, G, n, P/X, X}
    gamma_alpha_min: float = 0.01  # Minimum maintenance uptake (1/h) at T_ref
    K_alpha_gamma: float = 1.0     # Activation by total uptake (1/h)
    K_alpha_G: float = 0.5         # Activation by glucose (g/L)
    K_alpha_n: float = 5.0         # Activation by generations (dimensionless)
    K_alpha_PX: float = 1.0        # Activation by P/X ratio (g/g)
    K_alpha_X: float = 50.0        # Activation by biomass (g/L)

    # Feed concentration
    Gf: float = 400.0         # Feed glucose concentration (g/L)

    # Temperature dependence (optional)
    temp_dependence: Optional[TemperatureDependence] = None


@dataclass
class InitialConditions:
    """
    Initial conditions for the bioreactor states.

    These should be calibrated from actual batch phase measurements
    for each cultivation run.
    """
    X0: float = 10.0     # Initial biomass (g) - ~10 g/L * 1 L
    P0: float = 0.0      # Initial product (g)
    G0: float = 5.0      # Initial substrate in reactor (g) - from batch medium
    V0: float = 1.1      # Initial volume (L)

    @classmethod
    def from_first_measurement(
        cls,
        X_first: float,
        P_first: float,
        V_first: float,
        G_first: float = 0.0
    ) -> 'InitialConditions':
        """
        Create initial conditions from first measurement.

        Args:
            X_first: First biomass measurement (g/L)
            P_first: First product measurement (g/L)
            V_first: First volume measurement (L)
            G_first: First substrate measurement (g/L), default 0

        Returns:
            InitialConditions with calculated initial values
        """
        return cls(
            X0=X_first * V_first,
            P0=P_first * V_first,
            G0=G_first * V_first,
            V0=V_first,
        )


@dataclass
class ControlProfile:
    """
    Control variables profile (time-dependent).

    The feed rate and temperature can be constant or time-dependent functions.
    Now supports two-phase operation with induction point.
    """

    feed_rate: float | Callable[[float], float] = 0.0  # L/h
    temperature: float | Callable[[float], float] = 37.0  # C

    # For two-phase control
    induction_time: Optional[float] = None  # Time of induction (h)
    pre_induction_feed: Optional[Callable[[float], float]] = None  # Pre-induction feed
    post_induction_feed: Optional[Callable[[float], float]] = None  # Post-induction feed
    pre_induction_temp: Optional[float] = None  # Pre-induction temperature (C)
    post_induction_temp: Optional[float] = None  # Post-induction temperature (C)

    def get_feed_rate(self, t: float) -> float:
        """Get feed rate at time t, handling two-phase control."""
        if self.induction_time is not None:
            if t < self.induction_time and self.pre_induction_feed is not None:
                return self.pre_induction_feed(t)
            elif t >= self.induction_time and self.post_induction_feed is not None:
                return self.post_induction_feed(t)

        if callable(self.feed_rate):
            return self.feed_rate(t)
        return self.feed_rate

    def get_temperature(self, t: float) -> float:
        """Get temperature at time t, handling two-phase control."""
        if self.induction_time is not None:
            if t < self.induction_time and self.pre_induction_temp is not None:
                return self.pre_induction_temp
            elif t >= self.induction_time and self.post_induction_temp is not None:
                return self.post_induction_temp

        if callable(self.temperature):
            return self.temperature(t)
        return self.temperature


class OptFedModel:
    """
    Core mechanistic model for E. coli fed-batch fermentation.

    Implements the system of differential equations from the paper:
    - dX/dt = y*X - (f/V)*X
    - dP/dt = pi*X - (f/V)*P
    - dG/dt = -y*X + (f/V)*(Gf - G)
    - dV/dt = f

    With the kinetic rate expressions:
    - y = y_max(T) * G / (K_m + G) * inhibition_terms
    - pi = y_p * Y_PG
    - y = y_y * Y_XG
    - y = y_y + y_p + y_a(T)

    Temperature dependence follows Eq. 4e from Schlögel et al. (2024).
    """

    def __init__(
        self,
        params: ModelParameters,
        init_cond: Optional[InitialConditions] = None,
    ):
        """
        Initialize the OptFed model.

        Args:
            params: Model parameters
            init_cond: Initial conditions (default: standard values)
        """
        self.params = params
        self.init_cond = init_cond or InitialConditions()

        # Initialize temperature dependence if not provided
        if self.params.temp_dependence is None:
            self.params.temp_dependence = TemperatureDependence(model="none")

    def _apply_temperature_correction(
        self,
        k_ref: float,
        T: float,
        param_type: str
    ) -> float:
        """
        Apply temperature correction to a rate constant.

        Args:
            k_ref: Reference rate constant at T_ref
            T: Current temperature (C)
            param_type: Type of parameter ("gamma_max", "gamma_pi_max", "gamma_alpha_min")

        Returns:
            Temperature-corrected rate constant
        """
        # Ensure temp_dependence is initialized
        if self.params.temp_dependence is None:
            self.params.temp_dependence = TemperatureDependence(model="none")

        td = self.params.temp_dependence

        if td.model == "none" or T == td.T_ref:
            return k_ref

        if td.model == "Q10":
            # Q10 model: k(T) = k_ref * Q10^((T - T_ref)/10)
            if param_type == "gamma_max":
                Q10 = td.Q10_gamma_max
            elif param_type == "gamma_pi_max":
                Q10 = td.Q10_gamma_pi_max
            elif param_type == "gamma_alpha_min":
                Q10 = td.Q10_gamma_alpha_min
            else:
                Q10 = 2.0  # Default

            return k_ref * (Q10 ** ((T - td.T_ref) / 10.0))

        elif td.model == "arrhenius":
            # Arrhenius model: k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
            T_K = T + 273.15  # Convert to Kelvin
            T_ref_K = td.T_ref + 273.15

            if param_type == "gamma_max":
                Ea = td.Ea_gamma_max
            elif param_type == "gamma_pi_max":
                Ea = td.Ea_gamma_pi_max
            elif param_type == "gamma_alpha_min":
                Ea = td.Ea_gamma_alpha_min
            else:
                Ea = 50000.0  # Default (J/mol)

            return k_ref * np.exp(-Ea / R_GAS * (1/T_K - 1/T_ref_K))

        elif td.model == "erying":
            # Eyring model with enzyme inactivation (from paper Eq. 4e)
            # k(T) = (E0 * k_B * T / h) * exp(-DG_cat / (R*T)) / (1 + exp(DH_eq / R * (1/T_eq - 1/T)))
            T_K = T + 273.15  # Convert to Kelvin

            # Get Eyring parameters for the specific rate type
            if param_type == "gamma_max":
                E0 = td.E0_gamma_max
                DG_cat = td.DG_cat_gamma_max
                DH_eq = td.DH_eq_gamma_max
                T_eq = td.T_eq_gamma_max
            elif param_type == "gamma_pi_max":
                E0 = td.E0_gamma_pi_max
                DG_cat = td.DG_cat_gamma_pi_max
                DH_eq = td.DH_eq_gamma_pi_max
                T_eq = td.T_eq_gamma_pi_max
            elif param_type == "gamma_alpha_min":
                E0 = td.E0_gamma_alpha_min
                DG_cat = td.DG_cat_gamma_alpha_min
                DH_eq = td.DH_eq_gamma_alpha_min
                T_eq = td.T_eq_gamma_alpha_min
            else:
                # Default if param_type not recognized
                E0 = 1.0
                DG_cat = 50000.0
                DH_eq = 1e6
                T_eq = 999.0

            # Eyring equation numerator: E0 * k_B * T / h * exp(-DG_cat / (R*T))
            k_T = E0 * (K_B * T_K / H_PLANCK) * np.exp(-DG_cat / (R_GAS * T_K))

            # Enzyme inactivation denominator: 1 + exp(DH_eq / R * (1/T_eq - 1/T))
            # If T_eq is very large (999 K), this term is essentially 1 (no inactivation)
            if T_eq > 500:  # No significant inactivation
                inactivation_factor = 1.0
            else:
                inactivation_factor = 1.0 + np.exp(DH_eq / R_GAS * (1/T_eq - 1/T_K))

            return k_T / inactivation_factor

        return k_ref

    def _substrate_uptake_total(
        self,
        G: float,
        P: float,
        X: float,
        V: float,
        T: float = 37.0
    ) -> float:
        """
        Calculate total substrate uptake rate y (1/h) with temperature dependence.

        Uses Michaelis-Menten kinetics with non-competitive inhibition:
        y = y_max(T) * G / (K_m + G) * Pi (1 / (1 + i/K_i))

        Args:
            G: Substrate concentration in reactor (g/L)
            P: Product concentration (g/L)
            X: Total biomass (g/L)
            V: Volume (L)
            T: Temperature (C)

        Returns:
            y: Total substrate uptake rate (1/h)
        """
        # Apply temperature correction to gamma_max
        gamma_max_T = self._apply_temperature_correction(
            self.params.gamma_max, T, "gamma_max"
        )

        # Michaelis-Menten term with numerical safeguard
        K_m_safe = max(self.params.K_m, 1e-12)
        mm_term = G / (K_m_safe + G)

        # Inhibition terms (non-competitive)
        inhib_G = 1.0 / (1.0 + G / self.params.K_G_inhib)
        inhib_P = 1.0 / (1.0 + P / self.params.K_P_inhib)
        inhib_X = 1.0 / (1.0 + X / self.params.K_X_inhib)

        gamma_total = gamma_max_T * mm_term * inhib_G * inhib_P * inhib_X

        return gamma_total

    def _calculate_maintenance(
        self,
        gamma_total: float,
        G: float,
        P: float,
        X: float,
        V: float,
        T: float = 37.0
    ) -> float:
        """
        Calculate maintenance uptake y_a with full activation terms and temperature dependence.

        y_a = y_a_min(T) * Pi (1 + i/K_a_i), for i in {y, G, n, P/X, X}

        Args:
            gamma_total: Total substrate uptake rate (1/h)
            G: Substrate concentration (g/L)
            P: Total product (g)
            X: Total biomass (g)
            V: Volume (L)
            T: Temperature (C)

        Returns:
            y_a: Maintenance uptake rate (1/h)
        """
        # Apply temperature correction to gamma_alpha_min
        gamma_alpha_min_T = self._apply_temperature_correction(
            self.params.gamma_alpha_min, T, "gamma_alpha_min"
        )

        # Calculate concentrations
        X_conc = X / V if V > 0 else 0.0  # g/L
        P_conc = P / V if V > 0 else 0.0  # g/L
        PX_ratio = P_conc / X_conc if X_conc > 0 else 0.0  # g/g

        # Calculate number of generations: n = log2[XV / (X0*V0)]
        # Using current total biomass relative to initial
        X_total = X  # g
        X0_total = self.init_cond.X0  # g
        V0 = self.init_cond.V0  # L
        XV_ratio = X_total / (X0_total * V0) if X0_total > 0 and V0 > 0 else 1.0
        n = np.log2(max(1.0, XV_ratio))

        # Maintenance with all activation terms
        # y_a = y_a_min * (1 + y/K_a_y) * (1 + G/K_a_G) * (1 + n/K_a_n) * (1 + P/X/K_a_PX) * (1 + X/K_a_X)
        gamma_alpha = (
            gamma_alpha_min_T
            * (1.0 + gamma_total / self.params.K_alpha_gamma)
            * (1.0 + G / self.params.K_alpha_G)
            * (1.0 + n / self.params.K_alpha_n)
            * (1.0 + PX_ratio / self.params.K_alpha_PX)
            * (1.0 + X_conc / self.params.K_alpha_X)
        )

        return gamma_alpha

    def _derivatives(self, t: float, y: np.ndarray, control: ControlProfile) -> np.ndarray:
        """
        Compute derivatives for the ODE system.

        Args:
            t: Time (h)
            y: State vector [X, P, G, V] (g, g, g, L)
            control: Control profile

        Returns:
            dy/dt: Derivatives [dX/dt, dP/dt, dG/dt, dV/dt]
        """
        X, P, G, V = y

        # Get control values at time t
        f = control.get_feed_rate(t)
        T = control.get_temperature(t)

        # Calculate concentrations
        X_conc = X / V if V > 0 else 0.0  # g/L
        P_conc = P / V if V > 0 else 0.0  # g/L
        G_conc = G / V if V > 0 else 0.0  # g/L

        # Calculate uptake rates with temperature dependence
        gamma_total = self._substrate_uptake_total(G_conc, P_conc, X_conc, V, T)

        # Calculate maintenance with full activation terms and temperature
        gamma_alpha = self._calculate_maintenance(gamma_total, G_conc, P, X, V, T)

        # Apply temperature correction to gamma_pi_max
        gamma_pi_max_T = self._apply_temperature_correction(
            self.params.gamma_pi_max, T, "gamma_pi_max"
        )

        # Production uptake (saturated by gamma_total - gamma_alpha)
        available_uptake = gamma_total - gamma_alpha
        # Add small epsilon to prevent division by zero
        K_pi_m_safe = max(self.params.K_pi_m, 1e-12)
        gamma_pi = gamma_pi_max_T * available_uptake / (
            K_pi_m_safe + available_uptake
        )
        gamma_pi = max(0.0, gamma_pi)

        # Specific product formation rate
        pi = gamma_pi * self.params.Y_PG

        # Growth rate: y_y = y - y_p - y_a
        gamma_mu = gamma_total - gamma_pi - gamma_alpha
        gamma_mu = max(0.0, gamma_mu)
        mu = gamma_mu * self.params.Y_XG

        # Differential equations
        dXdt = mu * X - (f / V) * X if V > 0 else 0.0
        dPdt = pi * X - (f / V) * P if V > 0 else 0.0
        dGdt = -gamma_total * X + (f / V) * (self.params.Gf - G) if V > 0 else 0.0
        dVdt = f

        return np.array([dXdt, dPdt, dGdt, dVdt])

    def simulate(
        self,
        t_span: tuple[float, float],
        t_eval: np.ndarray,
        control: ControlProfile,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> dict:
        """
        Simulate the bioreactor over a time period.

        Args:
            t_span: (t_start, t_end) time interval (h)
            t_eval: Time points for output (h)
            control: Control profile (feed rate, temperature)
            method: Integration method (default: RK45)
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Dictionary with time points and state trajectories
        """
        y0 = np.array([
            self.init_cond.X0,
            self.init_cond.P0,
            self.init_cond.G0,
            self.init_cond.V0,
        ])

        sol = solve_ivp(
            fun=lambda t, y: self._derivatives(t, y, control),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        # Calculate concentrations
        V = sol.y[3, :]
        X_conc = sol.y[0, :] / V
        P_conc = sol.y[1, :] / V
        G_conc = sol.y[2, :] / V

        # Track temperature at each time point
        temperatures = np.array([control.get_temperature(t) for t in sol.t])

        return {
            "t": sol.t,
            "X": sol.y[0, :],      # Total biomass (g)
            "P": sol.y[1, :],      # Total product (g)
            "G": sol.y[2, :],      # Total substrate (g)
            "V": sol.y[3, :],      # Volume (L)
            "X_conc": X_conc,      # Biomass concentration (g/L)
            "P_conc": P_conc,      # Product concentration (g/L)
            "G_conc": G_conc,      # Substrate concentration (g/L)
            "T": temperatures,     # Temperature profile (C)
            "success": sol.success,
        }


class FeedStrategy:
    """
    Helper class for defining common feed strategies.

    Now includes two-phase feed strategies for pre/post induction.
    """

    @staticmethod
    def exponential(f0: float, mu_f: float, t_shift: float = 0.0) -> Callable[[float], float]:
        """
        Exponential feed: f(t) = f0 * exp(y_f * (t - t_shift))

        Args:
            f0: Initial feed rate (L/h)
            mu_f: Specific growth rate for feed (1/h)
            t_shift: Time offset (h)
        """
        return lambda t: f0 * np.exp(mu_f * max(0, t - t_shift))

    @staticmethod
    def linear(phi1: float, phi2: float, t_shift: float = 0.0) -> Callable[[float], float]:
        """
        Linear feed: f(t) = phi1 + phi2 * (t - t_shift)

        Args:
            phi1: Intercept (L/h)
            phi2: Slope (L/h)
            t_shift: Time offset (h)
        """
        return lambda t: max(0.0, phi1 + phi2 * max(0, t - t_shift))

    @staticmethod
    def constant(rate: float) -> Callable[[float], float]:
        """Constant feed rate."""
        return lambda t: rate

    @staticmethod
    def from_function(func: Callable[[float], float]) -> Callable[[float], float]:
        """
        Create a feed strategy from a custom function.

        Args:
            func: Function that takes time (h) and returns feed rate (L/h)

        Returns:
            The same function (for consistency with other FeedStrategy methods)
        """
        return func

    @staticmethod
    def two_phase(
        f0_exp: float,
        mu_f_exp: float,
        phi1_lin: float,
        phi2_lin: float,
        t_induction: float
    ) -> Callable[[float], float]:
        """
        Two-phase feed: exponential pre-induction, linear post-induction.

        This is the standard feed strategy used in E. coli fed-batch:
        - Pre-induction: Exponential feed f(t) = f0 * exp(y_f * t)
        - Post-induction: Linear feed f(t) = phi1 + phi2 * (t - t_induction)

        Args:
            f0_exp: Initial feed rate for exponential phase (L/h)
            mu_f_exp: Specific growth rate for exponential feed (1/h)
            phi1_lin: Intercept for linear phase (L/h)
            phi2_lin: Slope for linear phase (L/h)
            t_induction: Induction time (h)
        """
        def feed_func(t: float) -> float:
            if t < t_induction:
                # Exponential pre-induction
                return f0_exp * np.exp(mu_f_exp * t)
            else:
                # Linear post-induction
                return max(0.0, phi1_lin + phi2_lin * (t - t_induction))

        return feed_func

    @staticmethod
    def create_two_phase_control(
        f0_exp: float,
        mu_f_exp: float,
        phi1_lin: float,
        phi2_lin: float,
        t_induction: float,
        T_pre: float = 37.0,
        T_post: float = 30.0,
    ) -> ControlProfile:
        """
        Create a complete two-phase control profile.

        Args:
            f0_exp: Initial feed rate for exponential phase (L/h)
            mu_f_exp: Specific growth rate for exponential feed (1/h)
            phi1_lin: Intercept for linear phase (L/h)
            phi2_lin: Slope for linear phase (L/h)
            t_induction: Induction time (h)
            T_pre: Pre-induction temperature (C)
            T_post: Post-induction temperature (C)

        Returns:
            ControlProfile with two-phase feed and temperature
        """
        return ControlProfile(
            feed_rate=0.0,  # Ignored, using two-phase methods
            temperature=37.0,  # Ignored, using two-phase methods
            induction_time=t_induction,
            pre_induction_feed=lambda t: f0_exp * np.exp(mu_f_exp * t),
            post_induction_feed=lambda t: max(0.0, phi1_lin + phi2_lin * (t - t_induction)),
            pre_induction_temp=T_pre,
            post_induction_temp=T_post,
        )


def create_default_model() -> OptFedModel:
    """Create a model with default parameter values."""
    params = ModelParameters()
    init_cond = InitialConditions()
    return OptFedModel(params, init_cond)


def create_model_with_temperature_dependence(
    temp_model: str = "Q10",
    T_ref: float = 37.0,
) -> OptFedModel:
    """
    Create a model with temperature dependence enabled.

    Args:
        temp_model: Temperature model ("none", "Q10", "arrhenius", "erying")
        T_ref: Reference temperature (C)

    Returns:
        OptFedModel with temperature dependence
    """
    if temp_model == "erying":
        # Eyring model with enzyme inactivation (from paper Eq. 4e)
        temp_dependence = TemperatureDependence(
            model="erying",
            T_ref=T_ref,
            # Production uptake parameters from paper Table 1
            E0_gamma_pi_max=8.8e-9,
            DG_cat_gamma_pi_max=5.2e4,
            DH_eq_gamma_pi_max=4.8e6,
            T_eq_gamma_pi_max=310.0,
        )
    else:
        # Q10 or Arrhenius model
        temp_dependence = TemperatureDependence(
            model=temp_model,
            T_ref=T_ref,
            Q10_gamma_max=2.0,
            Q10_gamma_pi_max=2.0,
            Q10_gamma_alpha_min=2.0,
        )

    params = ModelParameters(temp_dependence=temp_dependence)
    init_cond = InitialConditions()
    return OptFedModel(params, init_cond)


def create_model_erying() -> OptFedModel:
    """
    Create a model with Eyring temperature dependence (from paper Eq. 4e).

    Uses the parameters from Table 1 of the paper for gamma_pi_max.

    Returns:
        OptFedModel with Eyring temperature dependence
    """
    return create_model_with_temperature_dependence(temp_model="erying")


if __name__ == "__main__":
    # Example: Simple simulation with exponential feed
    model = create_default_model()

    # Exponential feed: 0.01 L/h initial, y_f = 0.2 1/h
    control = ControlProfile(feed_rate=FeedStrategy.exponential(f0=0.01, mu_f=0.2))

    # Simulate for 20 hours
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 201)

    result = model.simulate(t_span, t_eval, control)

    print(f"Simulation completed: {result['success']}")
    print(f"Final biomass: {result['X_conc'][-1]:.2f} g/L")
    print(f"Final product: {result['P_conc'][-1]:.2f} g/L")
    print(f"Final volume: {result['V'][-1]:.2f} L")

    # Example: Two-phase feed with temperature shift
    print("\n--- Two-phase feed example ---")
    model_temp = create_model_with_temperature_dependence(temp_model="Q10")

    control_two_phase = FeedStrategy.create_two_phase_control(
        f0_exp=0.01,
        mu_f_exp=0.15,
        phi1_lin=0.05,
        phi2_lin=0.001,
        t_induction=23.0,
        T_pre=37.0,
        T_post=30.0,
    )

    result_two = model_temp.simulate(t_span, t_eval, control_two_phase)

    print(f"Two-phase simulation completed: {result_two['success']}")
    print(f"Final biomass: {result_two['X_conc'][-1]:.2f} g/L")
    print(f"Final product: {result_two['P_conc'][-1]:.2f} g/L")
    print(f"Temperature range: {result_two['T'].min():.1f} - {result_two['T'].max():.1f} C")
