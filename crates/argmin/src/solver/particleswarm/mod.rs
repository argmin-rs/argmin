// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Particle Swarm Optimization (PSO)
//!
//! Particle Swarm Optimization is a population-based metaheuristic that simulates the social
//! behavior of bird flocking or fish schooling. Each particle in the swarm represents a potential
//! solution and moves through the search space influenced by its own best position and the best
//! position found by the swarm.
//!
//! This implementation provides the canonical PSO algorithm as described in \[0\], along with
//! several optional enhancements including time-varying coefficients, chaotic inertia, multiple
//! initialization strategies, mutation operators, and velocity control mechanisms.
//!
//! For details see [`ParticleSwarm`].
//!
//! ## Reference
//!
//! \[0\] Zambrano-Bigiarini, M. et.al. (2013): Standard Particle Swarm Optimisation 2011 at
//! CEC-2013: A baseline for future PSO improvements. 2013 IEEE Congress on Evolutionary
//! Computation. <https://doi.org/10.1109/CEC.2013.6557848>

use crate::core::{
    ArgminFloat, CostFunction, Error, PopulationState, Problem, Solver, State, SyncAlias, KV,
};
use argmin_math::{
    ArgminAdd, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminRandom, ArgminSub, ArgminZeroLike,
};
#[cfg(feature = "rand")]
use rand::{Rng, SeedableRng};
#[cfg(feature = "rand")]
use rand_distr::{Cauchy, Distribution, Normal};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Strategy for initializing particle positions
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum InitializationStrategy {
    /// Standard uniform random initialization (default)
    #[default]
    UniformRandom,
    /// Latin Hypercube Sampling - ensures uniform distribution across search space
    LatinHypercube,
    /// Opposition-Based Learning - generates both random and opposite positions, selects best
    OppositionBased,
}

/// Mutation strategy for avoiding local optima
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum MutationStrategy {
    /// No mutation
    #[default]
    None,
    /// Gaussian mutation with specified standard deviation
    Gaussian(f64),
    /// Cauchy mutation with specified scale parameter
    Cauchy(f64),
}

/// Determines which particles to apply mutation to
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum MutationApplication {
    /// No mutation application (default)
    #[default]
    None,
    /// Apply mutation only to the global best particle (most efficient)
    GlobalBestOnly,
    /// Apply mutation to all particles with given probability (maximum diversity)
    AllParticles,
    /// Apply mutation only to particles with below-average fitness (balanced approach)
    BelowAverage,
}

/// Strategy for controlling acceleration coefficients (cognitive and social factors)
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum AccelerationCoefficientStrategy<F> {
    /// Constant coefficients throughout optimization (canonical PSO)
    Constant {
        /// Cognitive coefficient (c1)
        cognitive: F,
        /// Social coefficient (c2)
        social: F,
    },
    /// Time-Varying Acceleration Coefficients (TVAC)
    TimeVarying {
        /// Initial cognitive factor
        c1_initial: F,
        /// Final cognitive factor
        c1_final: F,
        /// Initial social factor
        c2_initial: F,
        /// Final social factor
        c2_final: F,
    },
}

impl<F: ArgminFloat> Default for AccelerationCoefficientStrategy<F> {
    fn default() -> Self {
        // Canonical PSO default values
        AccelerationCoefficientStrategy::Constant {
            cognitive: float!(0.5 + 2.0f64.ln()),
            social: float!(0.5 + 2.0f64.ln()),
        }
    }
}

/// Strategy for controlling inertia weight
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum InertiaWeightStrategy<F> {
    /// Constant inertia weight (canonical PSO)
    Constant(F),
    /// Chaotic inertia weight using logistic map
    Chaotic {
        /// Minimum inertia weight
        w_min: F,
        /// Maximum inertia weight
        w_max: F,
        /// Current chaotic state (z in (0,1) for logistic map)
        state: F,
    },
}

impl<F: ArgminFloat> Default for InertiaWeightStrategy<F> {
    fn default() -> Self {
        // Canonical PSO default value
        InertiaWeightStrategy::Constant(float!(1.0f64 / (2.0 * 2.0f64.ln())))
    }
}

/// # Particle Swarm Optimization (PSO)
///
/// Canonical implementation of the particle swarm optimization method as outlined in \[0\], along
/// with several optional enhancements including time-varying coefficients, chaotic inertia,
/// multiple initialization strategies, mutation operators, and velocity control mechanisms.
///
/// The `rayon` feature enables parallel computation of the cost function, which can be beneficial
/// for expensive cost functions but may reduce performance for cheap ones. Benchmark both modes
/// to determine the best configuration for your problem.
///
/// ## Configuration
///
/// The solver is configured using builder methods. The canonical PSO uses constant inertia weight
/// and constant acceleration coefficients, with uniform random initialization. Several optional
/// enhancements are available:
///
/// ### Inertia Weight
///
/// The inertia weight controls particle momentum (defaults to `1/(2 * ln(2))`). It can be set
/// to a constant value via [`with_inertia_factor`](`ParticleSwarm::with_inertia_factor`), or
/// configured to vary chaotically via [`with_chaotic_inertia`](`ParticleSwarm::with_chaotic_inertia`)
/// using a logistic map \[6\].
///
/// ### Acceleration Coefficients
///
/// The cognitive (c1) and social (c2) coefficients control learning from personal and swarm
/// experience (both default to `0.5 + ln(2)`). They can be set to constant values via
/// [`with_cognitive_factor`](`ParticleSwarm::with_cognitive_factor`) and
/// [`with_social_factor`](`ParticleSwarm::with_social_factor`), or configured to vary linearly
/// over iterations via [`with_tvac`](`ParticleSwarm::with_tvac`) (time-varying acceleration
/// coefficients) \[3\].
///
/// ### Initialization Strategy
///
/// The initial particle positions can be configured via
/// [`with_initialization_strategy`](`ParticleSwarm::with_initialization_strategy`) with three
/// options: `UniformRandom` (default), `LatinHypercube` for stratified sampling \[7, 8\], or
/// `OppositionBased` to evaluate both random and opposite positions \[4, 5\].
///
/// ### Mutation
///
/// Mutation helps escape local optima and can be configured via
/// [`with_mutation`](`ParticleSwarm::with_mutation`) \[9, 10\]. Strategies include `None`
/// (default), `Gaussian`, or `Cauchy` distributions. The mutation can be applied to
/// `GlobalBestOnly`, `AllParticles`, or `BelowAverage` particles.
///
/// ### Velocity Control
///
/// Two velocity control mechanisms are available: [`with_velocity_clamping`](`ParticleSwarm::with_velocity_clamping`)
/// limits velocity magnitude to prevent divergence \[2\], and
/// [`with_velocity_mutation`](`ParticleSwarm::with_velocity_mutation`) reinitializes near-zero
/// velocity components to prevent stagnation \[3\].
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ## References
///
/// \[0\] Zambrano-Bigiarini, M. et.al. (2013): Standard Particle Swarm Optimisation 2011 at
/// CEC-2013: A baseline for future PSO improvements. 2013 IEEE Congress on Evolutionary
/// Computation. <https://doi.org/10.1109/CEC.2013.6557848>
///
/// \[1\] <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
///
/// \[2\] Shi, Y., & Eberhart, R. C. (1998). A modified particle swarm optimizer. In Proceedings
/// of the 1998 IEEE International Conference on Evolutionary Computation, IEEE World Congress on
/// Computational Intelligence (pp. 69-73). Anchorage, AK, USA.
/// <https://doi.org/10.1109/ICEC.1998.699146>
///
/// \[3\] Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004). Self-organizing hierarchical
/// particle swarm optimizer with time-varying acceleration coefficients. IEEE Transactions on
/// Evolutionary Computation, 8(3), 240-255. <https://doi.org/10.1109/TEVC.2004.826071>
///
/// \[4\] Tizhoosh, H. R. (2005). Opposition-Based Learning: A New Scheme for Machine
/// Intelligence. In Proceedings of International Conference on Computational Intelligence for
/// Modelling Control and Automation (CIMCA 2005) (Vol. I, pp. 695-701). Vienna, Austria.
/// <https://doi.org/10.1109/CIMCA.2005.1631345>
///
/// \[5\] Wang, H., Wu, Z., Rahnamayan, S., Liu, Y., & Ventresca, M. (2011). Enhancing particle
/// swarm optimization using generalized opposition-based learning. Information Sciences, 181(20),
/// 4699-4714. <https://doi.org/10.1016/j.ins.2011.03.016>
///
/// \[6\] Liu, B., Wang, L., Jin, Y. H., Tang, F., & Huang, D. X. (2005). Improved particle swarm
/// optimization combined with chaos. Chaos, Solitons & Fractals, 25(5), 1261-1271.
/// <https://doi.org/10.1016/j.chaos.2004.11.095>
///
/// \[7\] Jakubcová, M., Máca, P., & Pech, P. (2014). A Comparison of Selected Modifications
/// of the Particle Swarm Optimization Algorithm. Journal of Applied Mathematics, 2014, 293087.
/// <https://doi.org/10.1155/2014/293087>
///
/// \[8\] Kazimipour, B., Li, X., & Qin, A. K. (2014). A review of population initialization
/// techniques for evolutionary algorithms. In 2014 IEEE Congress on Evolutionary Computation
/// (CEC) (pp. 2585-2592). <https://doi.org/10.1109/CEC.2014.6900618>
///
/// \[9\] Stacey, A., Jancic, M., & Grundy, I. (2003). Particle swarm optimization with mutation.
/// In Proceedings of the 2003 Congress on Evolutionary Computation (CEC 2003) (Vol. 2, pp.
/// 1425-1430). Canberra, Australia. <https://doi.org/10.1109/CEC.2003.1299838>
///
/// \[10\] Yao, X., Liu, Y., & Lin, G. (1999). Evolutionary programming made faster. IEEE
/// Transactions on Evolutionary Computation, 3(2), 82-102. <https://doi.org/10.1109/4235.771163>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ParticleSwarm<P, F, R> {
    // Core PSO parameters
    /// Bounds on parameter space
    bounds: (P, P),
    /// Number of particles
    num_particles: usize,

    // PSO strategy parameters
    /// Inertia weight strategy
    inertia_strategy: InertiaWeightStrategy<F>,
    /// Acceleration coefficient strategy
    acceleration_strategy: AccelerationCoefficientStrategy<F>,

    // Population initialization
    /// Particle initialization strategy
    initialization_strategy: InitializationStrategy,

    // Velocity control
    /// Velocity clamping factor (as fraction of search space range, None = disabled)
    velocity_clamp_factor: Option<F>,
    /// Threshold below which velocity is considered near-zero and should be reinitialized (None = disabled)
    velocity_mutation_threshold: Option<F>,

    // Mutation parameters
    /// Mutation strategy
    mutation_strategy: MutationStrategy,
    /// Mutation probability (applied per iteration)
    mutation_probability: F,
    /// Which particles to apply mutation to
    mutation_application: MutationApplication,

    // Configuration for time-varying strategies
    /// Maximum iterations (for time-varying strategies like TVAC)
    max_iterations: usize,

    // Utilities
    /// Random number generator
    rng_generator: R,
}

impl<P, F> ParticleSwarm<P, F, rand::rngs::StdRng>
where
    P: Clone + SyncAlias + ArgminSub<P, P> + ArgminMul<F, P> + ArgminRandom + ArgminZeroLike,
    F: ArgminFloat,
{
    /// Construct a new instance of `ParticleSwarm`
    ///
    /// Takes the number of particles and bounds on the search space as inputs. `bounds` is a tuple
    /// `(lower_bound, upper_bound)`, where `lower_bound` and `upper_bound` are of the same type as
    /// the position of a particle (`P`) and of the same length as the problem as dimensions.
    ///
    /// The inertia weight on velocity and the social and cognitive acceleration factors can be
    /// adapted with [`with_inertia_factor`](`ParticleSwarm::with_inertia_factor`),
    /// [`with_cognitive_factor`](`ParticleSwarm::with_cognitive_factor`) and
    /// [`with_social_factor`](`ParticleSwarm::with_social_factor`), respectively.
    ///
    /// The weights and acceleration factors default to:
    ///
    /// * inertia: `1/(2 * ln(2))`
    /// * cognitive: `0.5 + ln(2)`
    /// * social: `0.5 + ln(2)`
    ///
    /// # Configuration Options
    ///
    /// **Inertia weight:**
    /// * [`with_inertia_factor`](`ParticleSwarm::with_inertia_factor`) - Constant inertia weight (default: `1/(2*ln(2))`)
    /// * [`with_chaotic_inertia`](`ParticleSwarm::with_chaotic_inertia`) - Chaotic inertia weight using logistic map
    ///
    /// **Acceleration coefficients:**
    /// * [`with_cognitive_factor`](`ParticleSwarm::with_cognitive_factor`) and [`with_social_factor`](`ParticleSwarm::with_social_factor`) - Constant coefficients (default: `0.5+ln(2)` each)
    /// * [`with_tvac`](`ParticleSwarm::with_tvac`) - Time-varying coefficients
    ///
    /// **Population initialization:**
    /// * [`with_initialization_strategy`](`ParticleSwarm::with_initialization_strategy`) - Choose strategy (default: UniformRandom)
    ///
    /// **Velocity control:**
    /// * [`with_velocity_clamping`](`ParticleSwarm::with_velocity_clamping`) - Limit velocity magnitude (default: None)
    /// * [`with_velocity_mutation`](`ParticleSwarm::with_velocity_mutation`) - Reinitialize near-zero velocities (default: None)
    ///
    /// **Mutation:**
    /// * [`with_mutation`](`ParticleSwarm::with_mutation`) - Apply Gaussian or Cauchy mutation to particles (default: None)
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> = ParticleSwarm::new((lower_bound, upper_bound), 40);
    /// ```
    pub fn new(bounds: (P, P), num_particles: usize) -> Self {
        ParticleSwarm {
            // Core PSO parameters
            bounds,
            num_particles,
            // PSO strategy parameters
            inertia_strategy: InertiaWeightStrategy::default(),
            acceleration_strategy: AccelerationCoefficientStrategy::default(),
            // Population initialization
            initialization_strategy: InitializationStrategy::UniformRandom,
            // Velocity control
            velocity_clamp_factor: None,
            velocity_mutation_threshold: None,
            // Mutation parameters
            mutation_strategy: MutationStrategy::None,
            mutation_probability: float!(0.0),
            mutation_application: MutationApplication::None,
            // Configuration for time-varying strategies
            max_iterations: 1000,
            // Utilities
            rng_generator: rand::rngs::StdRng::from_os_rng(),
        }
    }
}
impl<P, F, R0> ParticleSwarm<P, F, R0>
where
    P: Clone + SyncAlias + ArgminSub<P, P> + ArgminMul<F, P> + ArgminRandom + ArgminZeroLike,
    F: ArgminFloat,
    R0: Rng,
{
    /// Set the random number generator
    ///
    /// Defaults to `rand::rngs::StdRng::from_os_rng()`
    ///
    /// # Example
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # use rand::SeedableRng;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///     .with_rng_generator(rand_xoshiro::Xoroshiro128Plus::seed_from_u64(1729));
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_rng_generator<R1: Rng>(self, generator: R1) -> ParticleSwarm<P, F, R1> {
        ParticleSwarm {
            // Core PSO parameters
            bounds: self.bounds,
            num_particles: self.num_particles,
            // PSO strategy parameters
            inertia_strategy: self.inertia_strategy,
            acceleration_strategy: self.acceleration_strategy,
            // Population initialization
            initialization_strategy: self.initialization_strategy,
            // Velocity control
            velocity_clamp_factor: self.velocity_clamp_factor,
            velocity_mutation_threshold: self.velocity_mutation_threshold,
            // Mutation parameters
            mutation_strategy: self.mutation_strategy,
            mutation_probability: self.mutation_probability,
            mutation_application: self.mutation_application,
            // Configuration for time-varying strategies
            max_iterations: self.max_iterations,
            // Utilities
            rng_generator: generator,
        }
    }
}

impl<P, F, R> ParticleSwarm<P, F, R>
where
    P: Clone
        + SyncAlias
        + ArgminSub<P, P>
        + ArgminMul<F, P>
        + ArgminRandom
        + ArgminZeroLike
        + ArgminAdd<P, P>
        + ArgminMinMax,
    F: ArgminFloat,
    R: Rng,
{
    /// Set constant inertia factor on particle velocity
    ///
    /// Defaults to `1/(2 * ln(2))`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40).with_inertia_factor(0.5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_inertia_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: inertia factor must be >=0."
            ));
        }
        self.inertia_strategy = InertiaWeightStrategy::Constant(factor);
        Ok(self)
    }

    /// Enable chaotic inertia weight using logistic map
    ///
    /// Uses a logistic map (z_next = 4 * z * (1 - z)) to generate chaotic variation in the
    /// inertia weight. This helps particles escape local optima through non-linear dynamics.
    ///
    /// # Arguments
    ///
    /// * `w_min` - Minimum inertia weight
    /// * `w_max` - Maximum inertia weight
    ///
    /// The chaotic state z in (0,1) is mapped to w in [w_min, w_max].
    ///
    /// # Reference
    ///
    /// Liu, B., Wang, L., Jin, Y. H., Tang, F., & Huang, D. X. (2005). Improved particle swarm
    /// optimization combined with chaos. Chaos, Solitons & Fractals, 25(5), 1261-1271.
    /// <https://doi.org/10.1016/j.chaos.2004.11.095>
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///         .with_chaotic_inertia(0.4, 0.9)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_chaotic_inertia(mut self, w_min: F, w_max: F) -> Result<Self, Error> {
        if w_min < float!(0.0) || w_max < float!(0.0) || w_min >= w_max {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: chaotic inertia range must satisfy 0 <= w_min < w_max."
            ));
        }
        self.inertia_strategy = InertiaWeightStrategy::Chaotic {
            w_min,
            w_max,
            state: float!(0.7), // Initialize to avoid fixed points (0, 0.5, 1.0)
        };
        Ok(self)
    }

    /// Set cognitive acceleration factor
    ///
    /// Defaults to `0.5 + ln(2)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40).with_cognitive_factor(1.1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_cognitive_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: cognitive factor must be >=0."
            ));
        }
        // Update or create constant acceleration strategy
        match self.acceleration_strategy {
            AccelerationCoefficientStrategy::Constant { social, .. } => {
                self.acceleration_strategy = AccelerationCoefficientStrategy::Constant {
                    cognitive: factor,
                    social,
                };
            }
            _ => {
                self.acceleration_strategy = AccelerationCoefficientStrategy::Constant {
                    cognitive: factor,
                    social: float!(0.5 + 2.0f64.ln()),
                };
            }
        }
        Ok(self)
    }

    /// Set social acceleration factor
    ///
    /// Defaults to `0.5 + ln(2)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40).with_social_factor(1.1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_social_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: social factor must be >=0."
            ));
        }
        // Update or create constant acceleration strategy
        match self.acceleration_strategy {
            AccelerationCoefficientStrategy::Constant { cognitive, .. } => {
                self.acceleration_strategy = AccelerationCoefficientStrategy::Constant {
                    cognitive,
                    social: factor,
                };
            }
            _ => {
                self.acceleration_strategy = AccelerationCoefficientStrategy::Constant {
                    cognitive: float!(0.5 + 2.0f64.ln()),
                    social: factor,
                };
            }
        }
        Ok(self)
    }

    /// Enable Time-Varying Acceleration Coefficients (TVAC)
    ///
    /// TVAC adjusts cognitive (c1) and social (c2) acceleration coefficients over iterations
    /// to balance exploration and exploitation. Typically, c1 decreases and c2 increases over
    /// time to shift from exploration to exploitation.
    ///
    /// # Reference
    ///
    /// Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004). Self-organizing hierarchical
    /// particle swarm optimizer with time-varying acceleration coefficients. IEEE Transactions on
    /// Evolutionary Computation, 8(3), 240-255. <https://doi.org/10.1109/TEVC.2004.826071>
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// // Cognitive factor decreases from 2.5 to 0.5
    /// // Social factor increases from 0.5 to 2.5
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///         .with_tvac(2.5, 0.5, 0.5, 2.5, 1000)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tvac(
        mut self,
        c1_initial: F,
        c1_final: F,
        c2_initial: F,
        c2_final: F,
        max_iterations: usize,
    ) -> Result<Self, Error> {
        if c1_initial < float!(0.0)
            || c1_final < float!(0.0)
            || c2_initial < float!(0.0)
            || c2_final < float!(0.0)
        {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: TVAC parameters must be >=0."
            ));
        }
        self.acceleration_strategy = AccelerationCoefficientStrategy::TimeVarying {
            c1_initial,
            c1_final,
            c2_initial,
            c2_final,
        };
        self.max_iterations = max_iterations;
        Ok(self)
    }

    /// Set particle initialization strategy
    ///
    /// Choose between different strategies for initializing particle positions:
    ///
    /// - **UniformRandom** (default): Standard uniform random initialization
    /// - **LatinHypercube**: Latin Hypercube Sampling ensures uniform distribution across search
    ///   space by dividing each dimension into equal intervals and sampling once from each
    ///   interval. Better initial coverage than random, especially in high dimensions.
    /// - **OppositionBased**: Opposition-Based Learning generates both random and opposite
    ///   positions, then selects the best particles. Improves initial population quality.
    ///
    /// # References
    ///
    /// - Jakubcová, M., Máca, P., & Pech, P. (2014). A Comparison of Selected Modifications
    ///   of the Particle Swarm Optimization Algorithm. Journal of Applied Mathematics, 2014,
    ///   293087. <https://doi.org/10.1155/2014/293087>
    /// - Kazimipour, B., Li, X., & Qin, A. K. (2014). A review of population initialization
    ///   techniques for evolutionary algorithms. In 2014 IEEE Congress on Evolutionary Computation
    ///   (CEC) (pp. 2585-2592). <https://doi.org/10.1109/CEC.2014.6900618>
    /// - Tizhoosh, H. R. (2005). Opposition-Based Learning: A New Scheme for Machine
    ///   Intelligence. In Proceedings of International Conference on Computational Intelligence for
    ///   Modelling Control and Automation (CIMCA 2005) (Vol. I, pp. 695-701). Vienna, Austria.
    ///   <https://doi.org/10.1109/CIMCA.2005.1631345>
    /// - Wang, H., Wu, Z., Rahnamayan, S., Liu, Y., & Ventresca, M. (2011). Enhancing particle
    ///   swarm optimization using generalized opposition-based learning. Information Sciences,
    ///   181(20), 4699-4714. <https://doi.org/10.1016/j.ins.2011.03.016>
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::{ParticleSwarm, InitializationStrategy};
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// // Use Latin Hypercube Sampling
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
    ///         .with_initialization_strategy(InitializationStrategy::LatinHypercube);
    ///
    /// // Use Opposition-Based Learning
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///         .with_initialization_strategy(InitializationStrategy::OppositionBased);
    /// ```
    pub fn with_initialization_strategy(mut self, strategy: InitializationStrategy) -> Self {
        self.initialization_strategy = strategy;
        self
    }

    /// Enable velocity clamping
    ///
    /// Limits particle velocities to prevent explosive behavior. The velocity is clamped to
    /// `+-clamp_factor * (upper_bound - lower_bound)` component-wise.
    ///
    /// # Arguments
    ///
    /// * `clamp_factor` - Fraction of search space range (typically 0.1 to 0.2)
    ///
    /// # Reference
    ///
    /// Shi, Y., & Eberhart, R. C. (1998). A modified particle swarm optimizer. In Proceedings
    /// of the 1998 IEEE International Conference on Evolutionary Computation, IEEE World Congress
    /// on Computational Intelligence (pp. 69-73). Anchorage, AK, USA.
    /// <https://doi.org/10.1109/ICEC.1998.699146>
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///         .with_velocity_clamping(0.2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_velocity_clamping(mut self, clamp_factor: F) -> Result<Self, Error> {
        if clamp_factor <= float!(0.0) || clamp_factor > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: velocity clamp factor must be in (0, 1]."
            ));
        }
        self.velocity_clamp_factor = Some(clamp_factor);
        Ok(self)
    }

    /// Enable velocity mutation when velocity approaches zero
    ///
    /// Reinitializes velocity components when they fall below a threshold, preventing
    /// particle stagnation. This is part of the HPSO (Hierarchical PSO) approach.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Velocity threshold below which reinitialization occurs
    ///   (typically 0.001 to 0.01)
    ///
    /// # Reference
    ///
    /// Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004). Self-organizing hierarchical
    /// particle swarm optimizer with time-varying acceleration coefficients. IEEE Transactions on
    /// Evolutionary Computation, 8(3), 240-255. <https://doi.org/10.1109/TEVC.2004.826071>
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///         .with_velocity_mutation(0.001)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_velocity_mutation(mut self, threshold: F) -> Result<Self, Error> {
        if threshold < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: velocity mutation threshold must be >=0."
            ));
        }
        self.velocity_mutation_threshold = Some(threshold);
        Ok(self)
    }

    /// Enable mutation with specified strategy and application method
    ///
    /// Applies Gaussian or Cauchy mutation to particles with specified probability
    /// to help escape local optima. Cauchy mutation has heavier tails and provides larger jumps,
    /// making it more effective for escaping local optima.
    ///
    /// The mutation can be applied to different sets of particles:
    /// - `MutationApplication::None`: No mutation (default)
    /// - `MutationApplication::GlobalBestOnly`: Only mutate the global best (most efficient)
    /// - `MutationApplication::AllParticles`: Mutate all particles (maximum diversity)
    /// - `MutationApplication::BelowAverage`: Mutate below-average particles (balanced)
    ///
    /// # Arguments
    ///
    /// * `strategy` - Mutation strategy (Gaussian or Cauchy with distribution parameter)
    /// * `probability` - Probability of mutation per iteration (typically 0.01 to 0.1)
    /// * `application` - Which particles to apply mutation to
    ///
    /// # References
    ///
    /// - Stacey, A., Jancic, M., & Grundy, I. (2003). Particle swarm optimization with mutation.
    ///   In Proceedings of the 2003 Congress on Evolutionary Computation (CEC 2003) (Vol. 2, pp.
    ///   1425-1430). Canberra, Australia. <https://doi.org/10.1109/CEC.2003.1299838>
    /// - Yao, X., Liu, Y., & Lin, G. (1999). Evolutionary programming made faster. IEEE
    ///   Transactions on Evolutionary Computation, 3(2), 82-102.
    ///   <https://doi.org/10.1109/4235.771163>
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::{ParticleSwarm, MutationStrategy, MutationApplication};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// // Use Cauchy mutation on below-average particles with probability=0.05
    /// let pso: ParticleSwarm<_, f64, _> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40)
    ///         .with_mutation(MutationStrategy::Cauchy(0.1), 0.05, MutationApplication::BelowAverage)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_mutation(
        mut self,
        strategy: MutationStrategy,
        probability: F,
        application: MutationApplication,
    ) -> Result<Self, Error> {
        if probability < float!(0.0) || probability > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: mutation probability must be in [0, 1]."
            ));
        }
        self.mutation_strategy = strategy;
        self.mutation_probability = probability;
        self.mutation_application = application;
        Ok(self)
    }

    /// Get current cognitive and social coefficients based on strategy
    fn get_acceleration_coefficients(&mut self, current_iter: u64) -> (F, F) {
        match &mut self.acceleration_strategy {
            AccelerationCoefficientStrategy::Constant { cognitive, social } => {
                (*cognitive, *social)
            }
            AccelerationCoefficientStrategy::TimeVarying {
                c1_initial,
                c1_final,
                c2_initial,
                c2_final,
            } => {
                let progress = F::from_u64(current_iter).unwrap()
                    / F::from_u64(self.max_iterations as u64).unwrap();

                let cognitive = *c1_initial + (*c1_final - *c1_initial) * progress;
                let social = *c2_initial + (*c2_final - *c2_initial) * progress;
                (cognitive, social)
            }
        }
    }

    /// Get current inertia weight based on strategy
    fn get_inertia_weight(&mut self) -> F {
        match &mut self.inertia_strategy {
            InertiaWeightStrategy::Constant(weight) => *weight,
            InertiaWeightStrategy::Chaotic {
                w_min,
                w_max,
                state,
            } => {
                // Logistic map: z_next = 4 * z * (1 - z)
                *state = float!(4.0) * *state * (float!(1.0) - *state);

                // Map chaotic state to inertia weight range
                *w_min + (*w_max - *w_min) * *state
            }
        }
    }

    /// Apply mutation to a particle position
    ///
    /// This method is called based on the mutation application strategy to mutate
    /// particle positions. The mutation respects the mutation probability parameter.
    fn apply_mutation(&mut self, position: &P) -> P
    where
        P: ArgminAdd<P, P>,
    {
        if self.rng_generator.random::<f64>() >= self.mutation_probability.to_f64().unwrap() {
            return position.clone();
        }

        let (min, max) = &self.bounds;
        let range = max.sub(min);

        match self.mutation_strategy {
            MutationStrategy::None => position.clone(),
            MutationStrategy::Gaussian(sigma) => {
                let normal = Normal::new(0.0, sigma).unwrap();
                let mut mutated = position.clone();

                // Apply Gaussian perturbation
                let perturbation =
                    range.mul(&F::from_f64(normal.sample(&mut self.rng_generator)).unwrap());
                mutated = mutated.add(&perturbation);

                // Clamp to bounds
                P::min(&P::max(&mutated, min), max)
            }
            MutationStrategy::Cauchy(scale) => {
                let cauchy = Cauchy::new(0.0, scale).unwrap();
                let mut mutated = position.clone();

                // Apply Cauchy perturbation
                let perturbation =
                    range.mul(&F::from_f64(cauchy.sample(&mut self.rng_generator)).unwrap());
                mutated = mutated.add(&perturbation);

                // Clamp to bounds
                P::min(&P::max(&mutated, min), max)
            }
        }
    }

    /// Apply velocity mutation for near-zero velocity
    ///
    /// Checks if the velocity magnitude (L2 norm) falls below a threshold and
    /// reinitializes the entire velocity vector if so. This prevents particle stagnation
    /// by ensuring particles maintain some minimum velocity.
    fn mutate_velocity(&mut self, velocity: &P) -> P
    where
        P: ArgminL2Norm<F>,
    {
        let threshold = match self.velocity_mutation_threshold {
            Some(t) => t,
            None => return velocity.clone(),
        };

        let (min, max) = &self.bounds;
        let delta = max.sub(min);
        let delta_neg = delta.mul(&float!(-1.0));

        // Calculate the L2 norm (magnitude) of the velocity
        let velocity_magnitude = velocity.l2_norm();

        // Calculate the L2 norm of the search space range as a reference
        let range_magnitude = delta.l2_norm();

        // If velocity magnitude is below threshold fraction of the range, reinitialize
        if velocity_magnitude < threshold * range_magnitude {
            // Generate new random velocity
            P::rand_from_range(&delta_neg, &delta, &mut self.rng_generator)
        } else {
            velocity.clone()
        }
    }

    /// Initializes all particles randomly and sorts them by their cost function values
    fn initialize_particles<O: CostFunction<Param = P, Output = F> + SyncAlias>(
        &mut self,
        problem: &mut Problem<O>,
    ) -> Result<Vec<Particle<P, F>>, Error>
    where
        P: ArgminAdd<P, P>,
    {
        let (mut positions, velocities) = self.initialize_positions_and_velocities();

        // Apply Opposition-Based Learning if enabled
        if matches!(
            self.initialization_strategy,
            InitializationStrategy::OppositionBased
        ) {
            let (min, max) = &self.bounds;
            let opposite_positions: Vec<P> = positions
                .iter()
                .map(|pos| {
                    // Opposite position: opp = lower_bound + upper_bound - position
                    min.add(max).sub(pos)
                })
                .collect();

            // Evaluate both populations
            let costs = problem.bulk_cost(&positions)?;
            let opposite_costs = problem.bulk_cost(&opposite_positions)?;

            // Combine and select best num_particles
            let mut combined: Vec<(P, F)> = positions
                .into_iter()
                .zip(costs)
                .chain(opposite_positions.into_iter().zip(opposite_costs))
                .collect();

            combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            positions = combined
                .into_iter()
                .take(self.num_particles)
                .map(|(p, _)| p)
                .collect();
        }

        let costs = problem.bulk_cost(&positions)?;

        let velocities = if matches!(
            self.initialization_strategy,
            InitializationStrategy::OppositionBased
        ) {
            // Need to regenerate velocities if we selected from combined population
            let (min, max) = &self.bounds;
            let delta = max.sub(min);
            let delta_neg = delta.mul(&float!(-1.0));
            (0..self.num_particles)
                .map(|_| P::rand_from_range(&delta_neg, &delta, &mut self.rng_generator))
                .collect()
        } else {
            velocities
        };

        let mut particles = positions
            .into_iter()
            .zip(velocities)
            .zip(costs)
            .map(|((p, v), c)| Particle::new(p, c, v))
            .collect::<Vec<_>>();

        // sort them, such that the first one is the best one
        particles.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(particles)
    }

    /// Initializes positions and velocities for all particles
    fn initialize_positions_and_velocities(&mut self) -> (Vec<P>, Vec<P>) {
        let (min, max) = &self.bounds;
        let delta = max.sub(min);
        let delta_neg = delta.mul(&float!(-1.0));

        let positions = match self.initialization_strategy {
            InitializationStrategy::UniformRandom | InitializationStrategy::OppositionBased => {
                // Standard uniform random initialization
                // OppositionBased uses this, then adds opposite positions in initialize_particles
                (0..self.num_particles)
                    .map(|_| P::rand_from_range(min, max, &mut self.rng_generator))
                    .collect()
            }
            InitializationStrategy::LatinHypercube => {
                // Latin Hypercube Sampling initialization
                // Divide each dimension into num_particles intervals and sample once from each
                let n = self.num_particles;
                let interval_size = float!(1.0) / F::from_usize(n).unwrap();

                (0..n)
                    .map(|i| {
                        // For each particle, generate position using LHS principle
                        // Sample from interval [i/n, (i+1)/n] and map to [min, max]
                        let interval_start = F::from_usize(i).unwrap() * interval_size;
                        let random_offset = F::from_f64(self.rng_generator.random::<f64>())
                            .unwrap()
                            * interval_size;
                        let t = interval_start + random_offset; // t in [i/n, (i+1)/n]

                        // Linear interpolation: position = min + t * (max - min)
                        let scaled_delta = delta.mul(&t);
                        min.add(&scaled_delta)
                    })
                    .collect()
            }
        };

        let velocities = (0..self.num_particles)
            .map(|_| P::rand_from_range(&delta_neg, &delta, &mut self.rng_generator))
            .collect();

        (positions, velocities)
    }
}

impl<O, P, F, R> Solver<O, PopulationState<Particle<P, F>, F>> for ParticleSwarm<P, F, R>
where
    O: CostFunction<Param = P, Output = F> + SyncAlias,
    P: Clone
        + SyncAlias
        + ArgminAdd<P, P>
        + ArgminSub<P, P>
        + ArgminMul<F, P>
        + ArgminZeroLike
        + ArgminRandom
        + ArgminMinMax
        + ArgminL2Norm<F>,
    F: ArgminFloat,
    R: Rng,
{
    fn name(&self) -> &str {
        "Particle Swarm Optimization"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<P, F>, F>,
    ) -> Result<(PopulationState<Particle<P, F>, F>, Option<KV>), Error> {
        // Users can provide a population or it will be randomly created.
        let particles = match state.take_population() {
            Some(mut particles) if particles.len() == self.num_particles => {
                // sort them first
                particles.sort_by(|a, b| {
                    a.cost
                        .partial_cmp(&b.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                particles
            }
            Some(particles) => {
                return Err(argmin_error!(
                    InvalidParameter,
                    format!(
                        "`ParticleSwarm`: Provided list of particles is of length {}, expected {}",
                        particles.len(),
                        self.num_particles
                    )
                ))
            }
            None => self.initialize_particles(problem)?,
        };

        Ok((
            state
                .individual(particles[0].clone())
                .cost(particles[0].cost)
                .population(particles),
            None,
        ))
    }

    /// Perform one iteration of algorithm
    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<P, F>, F>,
    ) -> Result<(PopulationState<Particle<P, F>, F>, Option<KV>), Error> {
        // Get current coefficients and weights based on strategies
        let current_iter = state.get_iter();
        let (weight_cognitive, weight_social) = self.get_acceleration_coefficients(current_iter);
        let weight_inertia = self.get_inertia_weight();

        let mut best_particle = state.take_individual().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No current best individual in state."
        ))?;
        let mut best_cost = state.get_cost();
        let mut particles = state.take_population().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No population in state."
        ))?;

        let zero = P::zero_like(&best_particle.position);

        // Calculate velocity clamp bounds if enabled
        let velocity_bounds = if let Some(clamp_factor) = self.velocity_clamp_factor {
            let (min, max) = &self.bounds;
            let range = max.sub(min);
            let v_max = range.mul(&clamp_factor);
            let v_min = v_max.mul(&float!(-1.0));
            Some((v_min, v_max))
        } else {
            None
        };

        let positions: Vec<_> = particles
            .iter_mut()
            .map(|p| {
                // New velocity is composed of
                // 1) previous velocity (momentum),
                // 2) motion toward particle optimum and
                // 3) motion toward global optimum.

                // ad 1)
                let momentum = p.velocity.mul(&weight_inertia);

                // ad 2)
                let to_optimum = p.best_position.sub(&p.position);
                let pull_to_optimum =
                    P::rand_from_range(&zero, &to_optimum, &mut self.rng_generator);
                let pull_to_optimum = pull_to_optimum.mul(&weight_cognitive);

                // ad 3)
                let to_global_optimum = best_particle.position.sub(&p.position);
                let pull_to_global_optimum =
                    P::rand_from_range(&zero, &to_global_optimum, &mut self.rng_generator)
                        .mul(&weight_social);

                p.velocity = momentum.add(&pull_to_optimum).add(&pull_to_global_optimum);

                // Apply velocity clamping if enabled
                if let Some((ref v_min, ref v_max)) = velocity_bounds {
                    p.velocity = P::min(&P::max(&p.velocity, v_min), v_max);
                }

                // Apply velocity mutation if enabled (simplified)
                p.velocity = self.mutate_velocity(&p.velocity);

                let new_position = p.position.add(&p.velocity);

                // Limit to search window
                p.position = P::min(&P::max(&new_position, &self.bounds.0), &self.bounds.1);
                &p.position
            })
            .collect();

        let costs = problem.bulk_cost(&positions)?;

        for (p, c) in particles.iter_mut().zip(costs.into_iter()) {
            p.cost = c;

            if p.cost < p.best_cost {
                p.best_position = p.position.clone();
                p.best_cost = p.cost;

                if p.cost < best_cost {
                    best_particle.position = p.position.clone();
                    best_particle.best_position = p.position.clone();
                    best_particle.cost = p.cost;
                    best_particle.best_cost = p.cost;
                    best_cost = p.cost;
                }
            }
        }

        // Apply mutation based on the selected application strategy
        if !matches!(self.mutation_strategy, MutationStrategy::None)
            && !matches!(self.mutation_application, MutationApplication::None)
        {
            match self.mutation_application {
                MutationApplication::None => {
                    // No mutation, do nothing
                }
                MutationApplication::GlobalBestOnly => {
                    // Strategy 1: Mutate only the global best particle
                    let mutated_position = self.apply_mutation(&best_particle.position);
                    let mutated_cost = problem.cost(&mutated_position)?;

                    if mutated_cost < best_cost {
                        best_particle.position = mutated_position.clone();
                        best_particle.best_position = mutated_position;
                        best_particle.cost = mutated_cost;
                        best_particle.best_cost = mutated_cost;
                        best_cost = mutated_cost;
                    }
                }
                MutationApplication::AllParticles => {
                    // Strategy 2: Mutate all particles with given probability
                    let mutated_positions: Vec<P> = particles
                        .iter()
                        .map(|p| self.apply_mutation(&p.position))
                        .collect();
                    let mutated_costs = problem.bulk_cost(&mutated_positions)?;

                    for ((particle, mutated_pos), mutated_cost) in particles
                        .iter_mut()
                        .zip(mutated_positions.into_iter())
                        .zip(mutated_costs.into_iter())
                    {
                        // Only update if mutation improved the cost
                        if mutated_cost < particle.cost {
                            particle.position = mutated_pos.clone();
                            particle.cost = mutated_cost;

                            // Update personal best if improved
                            if mutated_cost < particle.best_cost {
                                particle.best_position = mutated_pos.clone();
                                particle.best_cost = mutated_cost;
                            }

                            // Update global best if improved
                            if mutated_cost < best_cost {
                                best_particle.position = mutated_pos.clone();
                                best_particle.best_position = mutated_pos;
                                best_particle.cost = mutated_cost;
                                best_particle.best_cost = mutated_cost;
                                best_cost = mutated_cost;
                            }
                        }
                    }
                }
                MutationApplication::BelowAverage => {
                    // Strategy 3: Mutate only particles with below-average fitness
                    let average_cost: F = particles
                        .iter()
                        .map(|p| p.cost)
                        .fold(float!(0.0), |acc, c| acc + c)
                        / F::from_usize(particles.len()).unwrap();

                    let mut mutated_positions = Vec::new();
                    let mut indices_to_mutate = Vec::new();

                    for (idx, particle) in particles.iter().enumerate() {
                        if particle.cost > average_cost {
                            mutated_positions.push(self.apply_mutation(&particle.position));
                            indices_to_mutate.push(idx);
                        }
                    }

                    if !mutated_positions.is_empty() {
                        let mutated_costs = problem.bulk_cost(&mutated_positions)?;

                        for (mutated_pos, mutated_cost) in
                            mutated_positions.into_iter().zip(mutated_costs.into_iter())
                        {
                            let idx = indices_to_mutate.remove(0);
                            let particle = &mut particles[idx];

                            // Only update if mutation improved the cost
                            if mutated_cost < particle.cost {
                                particle.position = mutated_pos.clone();
                                particle.cost = mutated_cost;

                                // Update personal best if improved
                                if mutated_cost < particle.best_cost {
                                    particle.best_position = mutated_pos.clone();
                                    particle.best_cost = mutated_cost;
                                }

                                // Update global best if improved
                                if mutated_cost < best_cost {
                                    best_particle.position = mutated_pos.clone();
                                    best_particle.best_position = mutated_pos;
                                    best_particle.cost = mutated_cost;
                                    best_particle.best_cost = mutated_cost;
                                    best_cost = mutated_cost;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((
            state
                .individual(best_particle)
                .cost(best_cost)
                .population(particles),
            None,
        ))
    }
}

/// A single particle
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Particle<T, F> {
    /// Position of particle
    pub position: T,
    /// Velocity of particle
    velocity: T,
    /// Cost of particle
    pub cost: F,
    /// Best position of particle so far
    best_position: T,
    /// Best cost of particle so far
    best_cost: F,
}

impl<T, F> Particle<T, F>
where
    T: Clone,
    F: ArgminFloat,
{
    /// Create a new particle with a given position, cost and velocity.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::Particle;
    /// let particle: Particle<Vec<f64>, f64> = Particle::new(vec![0.0, 1.4], 12.0, vec![0.1, 0.5]);
    /// ```
    pub fn new(position: T, cost: F, velocity: T) -> Particle<T, F> {
        Particle {
            position: position.clone(),
            velocity,
            cost,
            best_position: position,
            best_cost: cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, State};
    use approx::assert_relative_eq;

    test_trait_impl!(
        particleswarm,
        ParticleSwarm<Vec<f64>, f64, rand::rngs::StdRng>
    );

    #[test]
    fn test_new() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let pso: ParticleSwarm<_, f64, rand::rngs::StdRng> =
            ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40);
        let ParticleSwarm {
            inertia_strategy,
            acceleration_strategy,
            bounds,
            num_particles,
            ..
        } = pso;

        // Check default inertia strategy
        match inertia_strategy {
            InertiaWeightStrategy::Constant(weight) => {
                assert_relative_eq!(
                    weight,
                    (1.0f64 / (2.0 * 2.0f64.ln())),
                    epsilon = f64::EPSILON
                );
            }
            _ => panic!("Expected Constant inertia strategy"),
        }

        // Check default acceleration strategy
        match acceleration_strategy {
            AccelerationCoefficientStrategy::Constant { cognitive, social } => {
                assert_relative_eq!(cognitive, (0.5f64 + 2.0f64.ln()), epsilon = f64::EPSILON);
                assert_relative_eq!(social, (0.5f64 + 2.0f64.ln()), epsilon = f64::EPSILON);
            }
            _ => panic!("Expected Constant acceleration strategy"),
        }

        assert_eq!(lower_bound[0].to_ne_bytes(), bounds.0[0].to_ne_bytes());
        assert_eq!(lower_bound[1].to_ne_bytes(), bounds.0[1].to_ne_bytes());
        assert_eq!(upper_bound[0].to_ne_bytes(), bounds.1[0].to_ne_bytes());
        assert_eq!(upper_bound[1].to_ne_bytes(), bounds.1[1].to_ne_bytes());
        assert_eq!(num_particles, 40);
    }

    #[test]
    fn test_with_inertia_factor() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        for inertia in [0.0, f64::EPSILON, 0.5, 1.0, 1.2, 3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_inertia_factor(inertia);
            assert!(res.is_ok());
            match res.unwrap().inertia_strategy {
                InertiaWeightStrategy::Constant(weight) => {
                    assert_eq!(weight.to_ne_bytes(), inertia.to_ne_bytes());
                }
                _ => panic!("Expected Constant inertia strategy"),
            }
        }

        for inertia in [-f64::EPSILON, -0.5, -1.0, -1.2, -3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_inertia_factor(inertia);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`ParticleSwarm`: ",
                    "inertia factor must be >=0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_cognitive_factor() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        for cognitive in [0.0, f64::EPSILON, 0.5, 1.0, 1.2, 3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_cognitive_factor(cognitive);
            assert!(res.is_ok());
            match res.unwrap().acceleration_strategy {
                AccelerationCoefficientStrategy::Constant { cognitive: c, .. } => {
                    assert_eq!(c.to_ne_bytes(), cognitive.to_ne_bytes());
                }
                _ => panic!("Expected Constant acceleration strategy"),
            }
        }

        for cognitive in [-f64::EPSILON, -0.5, -1.0, -1.2, -3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_cognitive_factor(cognitive);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`ParticleSwarm`: ",
                    "cognitive factor must be >=0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_social_factor() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        for social in [0.0, f64::EPSILON, 0.5, 1.0, 1.2, 3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_social_factor(social);
            assert!(res.is_ok());
            match res.unwrap().acceleration_strategy {
                AccelerationCoefficientStrategy::Constant { social: s, .. } => {
                    assert_eq!(s.to_ne_bytes(), social.to_ne_bytes());
                }
                _ => panic!("Expected Constant acceleration strategy"),
            }
        }

        for social in [-f64::EPSILON, -0.5, -1.0, -1.2, -3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_social_factor(social);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`ParticleSwarm`: ",
                    "social factor must be >=0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_tvac() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        // Test valid parameters
        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_tvac(2.5, 0.5, 0.5, 2.5, 1000);
        assert!(res.is_ok());
        let pso = res.unwrap();

        // Verify strategy type and values
        match pso.acceleration_strategy {
            AccelerationCoefficientStrategy::TimeVarying {
                c1_initial,
                c1_final,
                c2_initial,
                c2_final,
            } => {
                assert_eq!(c1_initial.to_ne_bytes(), 2.5f64.to_ne_bytes());
                assert_eq!(c1_final.to_ne_bytes(), 0.5f64.to_ne_bytes());
                assert_eq!(c2_initial.to_ne_bytes(), 0.5f64.to_ne_bytes());
                assert_eq!(c2_final.to_ne_bytes(), 2.5f64.to_ne_bytes());
            }
            _ => panic!("Expected TimeVarying acceleration strategy"),
        }
        assert_eq!(pso.max_iterations, 1000);

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_tvac(0.0, 0.0, 0.0, 0.0, 500);
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_tvac(-1.0, 0.5, 0.5, 2.5, 1000);
        assert!(res.is_err());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_tvac(2.5, -0.5, 0.5, 2.5, 1000);
        assert!(res.is_err());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_tvac(2.5, 0.5, -0.5, 2.5, 1000);
        assert!(res.is_err());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_tvac(2.5, 0.5, 0.5, -2.5, 1000);
        assert!(res.is_err());
    }

    #[test]
    fn test_with_velocity_clamping() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_velocity_clamping(0.2);
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_velocity_clamping(1.0);
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_velocity_clamping(0.0);
        assert!(res.is_err());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_velocity_clamping(1.5);
        assert!(res.is_err());
    }

    #[test]
    fn test_with_initialization_strategy() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        let pso: ParticleSwarm<_, f64, _> =
            ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40);
        assert_eq!(
            pso.initialization_strategy,
            InitializationStrategy::UniformRandom
        );

        let pso: ParticleSwarm<_, f64, _> =
            ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_initialization_strategy(InitializationStrategy::LatinHypercube);
        assert_eq!(
            pso.initialization_strategy,
            InitializationStrategy::LatinHypercube
        );

        let pso: ParticleSwarm<_, f64, _> = ParticleSwarm::new((lower_bound, upper_bound), 40)
            .with_initialization_strategy(InitializationStrategy::OppositionBased);
        assert_eq!(
            pso.initialization_strategy,
            InitializationStrategy::OppositionBased
        );
    }

    #[test]
    fn test_with_mutation() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40).with_mutation(
            MutationStrategy::Gaussian(0.1),
            0.05,
            MutationApplication::GlobalBestOnly,
        );
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40).with_mutation(
            MutationStrategy::Cauchy(0.1),
            0.05,
            MutationApplication::AllParticles,
        );
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40).with_mutation(
            MutationStrategy::Gaussian(0.1),
            0.05,
            MutationApplication::BelowAverage,
        );
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40).with_mutation(
            MutationStrategy::Cauchy(0.1),
            0.05,
            MutationApplication::None,
        );
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40).with_mutation(
            MutationStrategy::Cauchy(0.1),
            1.5,
            MutationApplication::GlobalBestOnly,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_with_chaotic_inertia() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_chaotic_inertia(0.4, 0.9);
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_chaotic_inertia(0.9, 0.4);
        assert!(res.is_err());
    }

    #[test]
    fn test_with_velocity_mutation() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_velocity_mutation(0.001);
        assert!(res.is_ok());

        let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
            .with_velocity_mutation(-0.1);
        assert!(res.is_err());
    }

    #[test]
    fn test_mutation_application_strategies() {
        struct PsoProblem {
            counter: std::sync::Arc<std::sync::Mutex<usize>>,
        }

        impl CostFunction for PsoProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
                *self.counter.lock().unwrap() += 1;
                Ok(param.iter().map(|x| x * x).sum())
            }
        }

        let lower_bound: Vec<f64> = vec![-5.0, -5.0];
        let upper_bound: Vec<f64> = vec![5.0, 5.0];

        // Test GlobalBestOnly strategy
        let mut pso_global_best =
            ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 20)
                .with_mutation(
                    MutationStrategy::Gaussian(0.1),
                    1.0, // Always mutate for testing
                    MutationApplication::GlobalBestOnly,
                )
                .unwrap();

        let mut problem_global_best = Problem::new(PsoProblem {
            counter: std::sync::Arc::new(std::sync::Mutex::new(0)),
        });

        let state_global_best: PopulationState<Particle<Vec<f64>, f64>, f64> =
            PopulationState::new();
        let (mut state_global_best, _) = pso_global_best
            .init(&mut problem_global_best, state_global_best)
            .unwrap();

        for _ in 0..5 {
            (state_global_best, _) = pso_global_best
                .next_iter(&mut problem_global_best, state_global_best)
                .unwrap();
        }

        // Test AllParticles strategy
        let mut pso_all_particles =
            ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 20)
                .with_mutation(
                    MutationStrategy::Gaussian(0.1),
                    1.0, // Always mutate for testing
                    MutationApplication::AllParticles,
                )
                .unwrap();

        let mut problem_all_particles = Problem::new(PsoProblem {
            counter: std::sync::Arc::new(std::sync::Mutex::new(0)),
        });

        let state_all_particles: PopulationState<Particle<Vec<f64>, f64>, f64> =
            PopulationState::new();
        let (mut state_all_particles, _) = pso_all_particles
            .init(&mut problem_all_particles, state_all_particles)
            .unwrap();

        for _ in 0..5 {
            (state_all_particles, _) = pso_all_particles
                .next_iter(&mut problem_all_particles, state_all_particles)
                .unwrap();
        }

        // Test BelowAverage strategy
        let mut pso_below_avg = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 20)
            .with_mutation(
                MutationStrategy::Gaussian(0.1),
                1.0, // Always mutate for testing
                MutationApplication::BelowAverage,
            )
            .unwrap();

        let mut problem_below_avg = Problem::new(PsoProblem {
            counter: std::sync::Arc::new(std::sync::Mutex::new(0)),
        });

        let state_below_avg: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new();
        let (mut state_below_avg, _) = pso_below_avg
            .init(&mut problem_below_avg, state_below_avg)
            .unwrap();

        for _ in 0..5 {
            (state_below_avg, _) = pso_below_avg
                .next_iter(&mut problem_below_avg, state_below_avg)
                .unwrap();
        }

        // All strategies should have valid populations and costs
        assert!(state_global_best.get_population().is_some());
        assert!(state_all_particles.get_population().is_some());
        assert!(state_below_avg.get_population().is_some());

        assert!(state_global_best.get_cost() >= 0.0);
        assert!(state_all_particles.get_cost() >= 0.0);
        assert!(state_below_avg.get_cost() >= 0.0);
    }

    #[test]
    fn test_initialize_positions_and_velocities() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let num_particles = 100;
        let mut pso: ParticleSwarm<_, f64, _> =
            ParticleSwarm::new((lower_bound, upper_bound), num_particles);

        let (positions, velocities) = pso.initialize_positions_and_velocities();
        assert_eq!(positions.len(), num_particles);
        assert_eq!(velocities.len(), num_particles);

        for pos in positions {
            for elem in pos {
                assert!(elem <= 1.0f64);
                assert!(elem >= -1.0f64);
            }
        }

        for velo in velocities {
            for elem in velo {
                assert!(elem <= 2.0f64);
                assert!(elem >= -2.0f64);
            }
        }
    }

    #[test]
    fn test_initialize_particles() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let num_particles = 10;
        let mut pso: ParticleSwarm<_, f64, _> =
            ParticleSwarm::new((lower_bound, upper_bound), num_particles);

        struct PsoProblem {
            counter: std::sync::Arc<std::sync::Mutex<usize>>,
            values: [f64; 10],
        }

        impl CostFunction for PsoProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
                let mut counter = self.counter.lock().unwrap();
                let cost = self.values[*counter];
                *counter += 1;
                Ok(cost)
            }
        }

        let mut values = [1.0, 4.0, 10.0, 2.0, -3.0, 8.0, 4.4, 8.1, 6.4, 4.5];

        let mut problem = Problem::new(PsoProblem {
            counter: std::sync::Arc::new(std::sync::Mutex::new(0)),
            values,
        });

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let particles = pso.initialize_particles(&mut problem).unwrap();
        assert_eq!(particles.len(), num_particles);

        // at least assure that they are ordered correctly and have the correct cost.
        for (particle, cost) in particles.iter().zip(values.iter()) {
            assert_eq!(particle.cost.to_ne_bytes(), cost.to_ne_bytes());
        }
    }

    #[test]
    fn test_particle_new() {
        let init_position = vec![0.2, 3.0];
        let init_cost = 12.0;
        let init_velocity = vec![1.2, -1.3];

        let particle: Particle<Vec<f64>, f64> =
            Particle::new(init_position.clone(), init_cost, init_velocity.clone());
        let Particle {
            position,
            velocity,
            cost,
            best_position,
            best_cost,
        } = particle;

        assert_eq!(init_position, position);
        assert_eq!(init_position, best_position);
        assert_eq!(init_cost.to_ne_bytes(), cost.to_ne_bytes());
        assert_eq!(init_cost.to_ne_bytes(), best_cost.to_ne_bytes());
        assert_eq!(init_velocity, velocity);
    }

    #[test]
    fn test_init_provided_population_wrong_size() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let mut pso: ParticleSwarm<_, f64, _> = ParticleSwarm::new((lower_bound, upper_bound), 40);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new()
            .population(vec![Particle::new(vec![1.0, 2.0], 12.0, vec![0.1, 0.3])]);
        let res = pso.init(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`ParticleSwarm`: ",
                "Provided list of particles is of length 1, expected 40\"",
            )
        );
    }

    #[test]
    fn test_init_provided_population_correct_size() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let particle_a = Particle::new(vec![1.0, 2.0], 12.0, vec![0.1, 0.3]);
        let particle_b = Particle::new(vec![2.0, 3.0], 10.0, vec![0.2, 0.4]);
        let mut pso: ParticleSwarm<_, f64, _> = ParticleSwarm::new((lower_bound, upper_bound), 2);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> =
            PopulationState::new().population(vec![particle_a.clone(), particle_b.clone()]);
        let res = pso.init(&mut Problem::new(TestProblem::new()), state);
        assert!(res.is_ok());
        let (mut state, kv) = res.unwrap();
        assert!(kv.is_none());
        assert_eq!(*state.get_param().unwrap(), particle_b);
        let population = state.take_population().unwrap();
        // assert that it was sorted!
        assert_eq!(population[0], particle_b);
        assert_eq!(population[1], particle_a);
    }

    #[test]
    fn test_init_random_population() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let mut pso: ParticleSwarm<_, f64, _> = ParticleSwarm::new((lower_bound, upper_bound), 40);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new();
        let res = pso.init(&mut Problem::new(TestProblem::new()), state);
        assert!(res.is_ok());
        let (mut state, kv) = res.unwrap();
        assert!(kv.is_none());
        assert!(state.get_param().is_some());
        let population = state.take_population().unwrap();
        assert_eq!(population.len(), 40);
    }

    #[test]
    fn test_next_iter() {
        struct PsoProblem {
            counter: std::sync::Mutex<usize>,
            values: [f64; 10],
        }

        impl CostFunction for PsoProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
                let cost = self.values[*self.counter.lock().unwrap() % 10];
                *self.counter.lock().unwrap() += 1;
                Ok(cost)
            }
        }

        let values = [1.0, 4.0, 10.0, 2.0, -3.0, 8.0, 4.4, 8.1, 6.4, 4.4];

        let mut problem = Problem::new(PsoProblem {
            counter: std::sync::Mutex::new(0),
            values,
        });

        // setup
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let mut pso: ParticleSwarm<_, f64, _> = ParticleSwarm::new((lower_bound, upper_bound), 100);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new();

        // init
        let (mut state, _) = pso.init(&mut problem, state).unwrap();

        // next_iter
        for _ in 0..200 {
            (state, _) = pso.next_iter(&mut problem, state).unwrap();
            let population = state.get_population().unwrap();
            assert_eq!(population.len(), 100);
            for particle in population {
                for x in particle.position.iter() {
                    assert!(*x <= 1.0);
                    assert!(*x >= -1.0);
                }
            }
            assert_eq!(state.get_cost().to_ne_bytes(), (-3.0f64).to_ne_bytes());
        }
    }
}
