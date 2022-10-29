// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Observers
//!
//! Argmin offers an interface to observe the state of the solver at initialization as well as
//! after every iteration. This includes the parameter vector, gradient, Jacobian, Hessian,
//! iteration number, cost values and many more as well as solver-specific metrics. This interface
//! can be used to implement loggers, send the information to a storage or to plot metrics.
//!
//! The observer [`WriteToFile`](`crate::core::observers::WriteToFile`) saves the parameter vector
//! to disk and as such requires the parameter vector to be serializable. Hence this feature is
//! only available with the `serde1` feature.
//!
//! The observer [`SlogLogger`](`crate::core::observers::SlogLogger`) logs the progress of the
//! optimization to screen or to disk. This requires the `slog-logger` feature. Writing to disk
//! in addition requires the `serde1` feature.
//!
//! For each observer it can be defined how often it will observe the progress of the solver. This
//! is indicated via the enum `ObserverMode` which can be either `Always`, `Never`, `NewBest`
//! (whenever a new best solution is found) or `Every(i)` which means every `i`th iteration.
//!
//! Custom observers can be used as well by implementing the [`crate::core::observers::Observe`]
//! trait.
//!
//! ## Example
//!
//! ```rust
//! # #![allow(unused_imports)]
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # use argmin::core::{Error, Executor, CostFunction, Gradient, observers::ObserverMode};
//! # #[cfg(feature = "slog-logger")]
//! # use argmin::core::observers::SlogLogger;
//! # #[cfg(feature = "serde1")]
//! # use argmin::core::observers::{WriteToFile, WriteToFileSerializer};
//! # use argmin::solver::gradientdescent::SteepestDescent;
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # /// Implement `CostFunction` for `Rosenbrock`
//! # impl CostFunction for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Output = f64;
//! #
//! #     /// Apply the cost function to a parameter `p`
//! #     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock_2d(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # /// Implement `Gradient` for `Rosenbrock`
//! # impl Gradient for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Gradient = Vec<f64>;
//! #
//! #     /// Compute the gradient at parameter `p`.
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//! #         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #
//! # // Define cost function (must implement `CostFunction` and `Gradient`)
//! # let problem = Rosenbrock { a: 1.0, b: 100.0 };
//! #
//! # // Define initial parameter vector
//! # let init_param: Vec<f64> = vec![-1.2, 1.0];
//! #
//! # // Set up line search
//! # let linesearch = MoreThuenteLineSearch::new();
//! #
//! # // Set up solver
//! # let solver = SteepestDescent::new(linesearch);
//! #
//! // [...]
//!
//! let res = Executor::new(problem, solver)
//!     .configure(|config| config.param(init_param).max_iters(2))
//! # ;
//! # #[cfg(feature = "slog-logger")]
//! # let res = res
//!     // Add an observer which will log all iterations to the terminal (without blocking)
//!     .add_observer(SlogLogger::term_noblock(), ObserverMode::Always)
//! # ;
//! # #[cfg(feature = "serde1")]
//! # #[cfg(feature = "slog-logger")]
//! # let res = res
//!     // Log to file whenever a new best solution is found
//!     .add_observer(SlogLogger::file("solver.log", false)?, ObserverMode::NewBest)
//!     // Write parameter vector to `params/param.arg` every 20th iteration
//!     .add_observer(
//!         WriteToFile::new("params", "param", WriteToFileSerializer::JSON),
//!         ObserverMode::Every(20)
//!     )
//! # ;
//! # let res = res
//!     // run the solver on the defined problem
//!     .run()?;
//!
//! // [...]
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{}", e);
//! #         std::process::exit(1);
//! #     }
//! # }
//! ```

#[cfg(feature = "serde1")]
pub mod file;
#[cfg(feature = "slog-logger")]
pub mod slog_logger;

#[cfg(feature = "serde1")]
pub use file::*;
#[cfg(feature = "slog-logger")]
pub use slog_logger::*;

use crate::core::{Error, State, KV};
use std::default::Default;
use std::sync::{Arc, Mutex};

/// An interface which every observer is required to implement
///
/// # Example
///
/// ```
/// use argmin::core::{Error, KV, State};
/// use argmin::core::observers::Observe;
///
/// struct MyObserver {}
///
/// impl<I> Observe<I> for MyObserver
/// where
///     // Optional constraint on `I`. The `State` trait, which every state used in argmin needs to
///     // implement, offers a range of methods which can be useful.
///     I: State,
/// {
///     fn observe_init(&mut self, name: &str, kv: &KV) -> Result<(), Error> {
///         // Do something with `name` and/or `kv`
///         // Is executed after initialization of a solver
///         Ok(())
///     }
///
///     fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
///         // Do something with `state` and/or `kv`
///         // Is executed after each iteration of a solver
///         Ok(())
///     }
/// }
/// ```
pub trait Observe<I> {
    /// Called once after initialization of the solver.
    ///
    /// Has access to the name of the solver via `name` and to a key-value store `kv` with entries
    /// specific for each solver.
    fn observe_init(&mut self, _name: &str, _kv: &KV) -> Result<(), Error> {
        Ok(())
    }

    /// Called at every iteration of the solver
    ///
    /// Has access to the current `state` of the solver (which always implements
    /// [`State`](`crate::core::State`)) and to a key-value store `kv` with entries specific for
    /// each solver.
    fn observe_iter(&mut self, _state: &I, _kv: &KV) -> Result<(), Error> {
        Ok(())
    }
}

type ObserversVec<I> = Vec<(Arc<Mutex<dyn Observe<I>>>, ObserverMode)>;

/// Container for observers.
///
/// This type also implements [`Observe`] and therefore can be used like a single observer.
/// Each observer has an [`ObserverMode`] attached which indicates when the observer will be
/// called.
#[derive(Clone, Default)]
pub struct Observers<I> {
    /// Vector of `Observe`rs with the corresponding `ObserverMode`
    observers: ObserversVec<I>,
}

impl<I> Observers<I> {
    /// Construct a new empty `Observers` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::Observers;
    /// use argmin::core::IterState;
    ///
    /// let observers: Observers<IterState<Vec<f64>, (), (), (), f64>> = Observers::new();
    /// # assert!(observers.is_empty());
    /// ```
    pub fn new() -> Self {
        Observers { observers: vec![] }
    }

    /// Add another observer with a corresponding [`ObserverMode`].
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::{Observers, ObserverMode};
    /// # #[cfg(feature = "slog-logger")]
    /// use argmin::core::observers::SlogLogger;
    /// use argmin::core::IterState;
    ///
    /// let mut observers: Observers<IterState<Vec<f64>, (), (), (), f64>> = Observers::new();
    ///
    /// # #[cfg(feature = "slog-logger")]
    /// let logger = SlogLogger::term();
    /// # #[cfg(feature = "slog-logger")]
    /// observers.push(logger, ObserverMode::Always);
    /// # #[cfg(feature = "slog-logger")]
    /// # assert!(!observers.is_empty());
    /// ```
    pub fn push<OBS: Observe<I> + 'static>(
        &mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> &mut Self {
        self.observers.push((Arc::new(Mutex::new(observer)), mode));
        self
    }

    /// Returns true if there are no observers stored.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::Observers;
    /// use argmin::core::IterState;
    ///
    /// let observers: Observers<IterState<Vec<f64>, (), (), (), f64>> = Observers::new();
    /// assert!(observers.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }
}

/// Implementing [`Observe`] for [`Observers`] allows to use it like a single observer. In its
/// implementation it will loop over all stored observers, checks if the conditions for observing
/// are met and calls the actual observers if required.
impl<I: State> Observe<I> for Observers<I> {
    /// After initialization of the solver, this loops over all stored observers and calls them.
    fn observe_init(&mut self, name: &str, kv: &KV) -> Result<(), Error> {
        for l in self.observers.iter() {
            l.0.lock().unwrap().observe_init(name, kv)?
        }
        Ok(())
    }

    /// Called after each iteration.
    ///
    /// Loops over all observers, and based on whether the condition for calling the observers are
    /// met, calls them.
    fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
        for l in self.observers.iter_mut() {
            let iter = state.get_iter();
            let observer = &mut l.0.lock().unwrap();
            match l.1 {
                ObserverMode::Always => observer.observe_iter(state, kv),
                ObserverMode::Every(i) if iter % i == 0 => observer.observe_iter(state, kv),
                ObserverMode::NewBest if state.is_best() => observer.observe_iter(state, kv),
                ObserverMode::Never | ObserverMode::Every(_) | ObserverMode::NewBest => Ok(()),
            }?
        }
        Ok(())
    }
}

/// Indicates when to call an observer.
///
/// `Always` calls the observer in every iteration, `Every(X)` calls the observer every X
/// iterations, `NewBest` calls the observer only when a new best parameter vector is found and
/// `Never` deactivates the observer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ObserverMode {
    /// Never call the observer
    Never,
    /// Call observer in every iteration
    Always,
    /// Call observer every N iterations
    Every(u64),
    /// Call observer when new best is found
    NewBest,
}

impl Default for ObserverMode {
    /// The default for `ObserverMode` is `Always`
    fn default() -> ObserverMode {
        ObserverMode::Always
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(observermode, ObserverMode);

    #[test]
    fn test_observers() {
        use crate::core::observers::Observe;
        use crate::core::{Error, IterState, KV};

        struct TestStor {
            pub solver_name: String,
            pub init_called: usize,
            pub iter_called: usize,
        }

        impl TestStor {
            fn new() -> Arc<Mutex<TestStor>> {
                Arc::new(Mutex::new(TestStor {
                    solver_name: String::new(),
                    init_called: 0,
                    iter_called: 0,
                }))
            }
        }

        struct TestObs {
            data: Arc<Mutex<TestStor>>,
        }

        impl<I> Observe<I> for TestObs {
            fn observe_init(&mut self, name: &str, _kv: &KV) -> Result<(), Error> {
                self.data.lock().unwrap().solver_name = name.into();
                self.data.lock().unwrap().init_called += 1;
                Ok(())
            }

            fn observe_iter(&mut self, _state: &I, _kv: &KV) -> Result<(), Error> {
                self.data.lock().unwrap().iter_called += 1;
                Ok(())
            }
        }

        let test_stor_1 = TestStor::new();
        let test_obs_1 = TestObs {
            data: test_stor_1.clone(),
        };

        let test_stor_2 = TestStor::new();
        let test_obs_2 = TestObs {
            data: test_stor_2.clone(),
        };

        let test_stor_3 = TestStor::new();
        let test_obs_3 = TestObs {
            data: test_stor_3.clone(),
        };

        let test_stor_4 = TestStor::new();
        let test_obs_4 = TestObs {
            data: test_stor_4.clone(),
        };

        let storages = [test_stor_1, test_stor_2, test_stor_3, test_stor_4];

        type TState = IterState<Vec<f64>, (), (), (), f64>;

        let mut obs: Observers<TState> = Observers::new();
        obs.push(test_obs_1, ObserverMode::Never)
            .push(test_obs_2, ObserverMode::Always)
            .push(test_obs_3, ObserverMode::Every(3))
            .push(test_obs_4, ObserverMode::NewBest);

        obs.observe_init("test_solver", &kv!()).unwrap();

        // all `init_called` should be 1, all `iter_called` 0
        for s in storages.iter() {
            let observer = s.lock().unwrap();
            assert_eq!(observer.solver_name, "test_solver");
            assert_eq!(observer.init_called, 1);
            assert_eq!(observer.iter_called, 0);
        }

        let mut state: TState = IterState::new();
        obs.observe_iter(&state, &kv!()).unwrap();

        assert_eq!(storages[0].lock().unwrap().init_called, 1);
        assert_eq!(storages[0].lock().unwrap().iter_called, 0);
        assert_eq!(storages[1].lock().unwrap().init_called, 1);
        assert_eq!(storages[1].lock().unwrap().iter_called, 1);
        assert_eq!(storages[2].lock().unwrap().init_called, 1);
        assert_eq!(storages[2].lock().unwrap().iter_called, 1);
        assert_eq!(storages[3].lock().unwrap().init_called, 1);
        assert_eq!(storages[3].lock().unwrap().iter_called, 1);

        state.increment_iter();
        obs.observe_iter(&state, &kv!()).unwrap();

        assert_eq!(storages[0].lock().unwrap().init_called, 1);
        assert_eq!(storages[0].lock().unwrap().iter_called, 0);
        assert_eq!(storages[1].lock().unwrap().init_called, 1);
        assert_eq!(storages[1].lock().unwrap().iter_called, 2);
        assert_eq!(storages[2].lock().unwrap().init_called, 1);
        assert_eq!(storages[2].lock().unwrap().iter_called, 1);
        assert_eq!(storages[3].lock().unwrap().init_called, 1);
        assert_eq!(storages[3].lock().unwrap().iter_called, 1);

        state.increment_iter();
        state.increment_iter();
        obs.observe_iter(&state, &kv!()).unwrap();

        assert_eq!(storages[0].lock().unwrap().init_called, 1);
        assert_eq!(storages[0].lock().unwrap().iter_called, 0);
        assert_eq!(storages[1].lock().unwrap().init_called, 1);
        assert_eq!(storages[1].lock().unwrap().iter_called, 3);
        assert_eq!(storages[2].lock().unwrap().init_called, 1);
        assert_eq!(storages[2].lock().unwrap().iter_called, 2);
        assert_eq!(storages[3].lock().unwrap().init_called, 1);
        assert_eq!(storages[3].lock().unwrap().iter_called, 1);

        state.increment_iter();
        // "new best found"
        state.last_best_iter = state.iter;
        obs.observe_iter(&state, &kv!()).unwrap();

        assert_eq!(storages[0].lock().unwrap().init_called, 1);
        assert_eq!(storages[0].lock().unwrap().iter_called, 0);
        assert_eq!(storages[1].lock().unwrap().init_called, 1);
        assert_eq!(storages[1].lock().unwrap().iter_called, 4);
        assert_eq!(storages[2].lock().unwrap().init_called, 1);
        assert_eq!(storages[2].lock().unwrap().iter_called, 2);
        assert_eq!(storages[3].lock().unwrap().init_called, 1);
        assert_eq!(storages[3].lock().unwrap().iter_called, 2);
    }
}
