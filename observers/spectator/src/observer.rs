// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::collections::HashSet;

use anyhow::Error;
use argmin::core::{observers::Observe, ArgminFloat, State, KV};
use spectator::{Message, DEFAULT_PORT};
use time::Duration;
use uuid::Uuid;

use crate::sender::sender;

const DEFAULT_HOST: &str = "127.0.0.1";

/// Builder for the Spectator observer
///
/// # Example
///
/// ```
/// use argmin_observer_spectator::SpectatorBuilder;
///
/// let spectator = SpectatorBuilder::new()
///     // Optional: Name the optimization run
///     // Default: random uuid.
///     .with_name("optimization_run_1")
///     // Optional, defaults to 127.0.0.1
///     .with_host("127.0.0.1")
///     // Optional, defaults to 5498
///     .with_port(5498)
///     // Choose which metrics should automatically be selected.
///     // If omitted, all metrics will be selected.
///     .select(&["cost", "best_cost"])
///     // Build Spectator observer
///     .build();
/// ```
pub struct SpectatorBuilder {
    name: String,
    selected: HashSet<String>,
    capacity: usize,
    host: String,
    port: u16,
}

impl Default for SpectatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectatorBuilder {
    /// Creates a new `SpectatorBuilder`
    pub fn new() -> Self {
        SpectatorBuilder {
            name: Uuid::new_v4().to_string(),
            selected: HashSet::new(),
            capacity: 10_000,
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
        }
    }

    /// Set a name the optimization run will be identified with
    ///
    /// Defaults to a random UUID.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// let builder = SpectatorBuilder::new().with_name("optimization_run_1");
    /// # assert_eq!(builder.name().clone(), "optimization_run_1".to_string());
    /// ```
    pub fn with_name<T: AsRef<str>>(mut self, name: T) -> Self {
        self.name = name.as_ref().to_string();
        self
    }

    /// Set the host argmin spectator is running on.
    ///
    /// Defaults to 127.0.0.1.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// let builder = SpectatorBuilder::new().with_host("192.168.0.1");
    /// # assert_eq!(builder.host().clone(), "192.168.0.1".to_string());
    /// ```
    pub fn with_host<T: AsRef<str>>(mut self, host: T) -> Self {
        self.host = host.as_ref().to_string();
        self
    }

    /// Set the port Spectator is running on.
    ///
    /// Defaults to 5498.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// let builder = SpectatorBuilder::new().with_port(1234);
    /// # assert_eq!(builder.port(), 1234);
    /// ```
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the channel capacity
    ///
    /// A channel is used to queue messages for sending to Spectator. If the channel
    /// capacity is reached backpressure will be applied, effectively blocking the optimization.
    /// Defaults to 10000. Decrease this value in case memory consumption is too high and increase
    /// the value in case blocking causes negative effects.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// let builder = SpectatorBuilder::new().with_channel_capacity(1000);
    /// # assert_eq!(builder.channel_capacity(), 1000);
    /// ```
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Define which metrics will be selected in Spectator by default
    ///
    /// If none are set, all metrics will be selected and shown. Providing zero or more metrics
    /// via `select` disables all apart from the provided ones. Note that all data will be sent, and
    /// metrics can be selected and deselected using the GUI.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # use std::collections::HashSet;
    /// let builder = SpectatorBuilder::new().select(&["cost", "best_cost"]);
    /// # assert_eq!(builder.selected(), &HashSet::from(["cost".to_string(), "best_cost".to_string()]));
    /// ```
    pub fn select<T: AsRef<str>>(mut self, metrics: &[T]) -> Self {
        self.selected = metrics.iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Returns the name of the optimization run
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # let builder = SpectatorBuilder::new().with_name("test");
    /// let name = builder.name();
    /// # assert_eq!(name, &"test".to_string());
    /// ```
    pub fn name(&self) -> &String {
        &self.name
    }

    /// Returns the host this observer will connect to
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # let builder = SpectatorBuilder::new();
    /// let host = builder.host();
    /// # assert_eq!(host, &"127.0.0.1".to_string());
    /// ```
    pub fn host(&self) -> &String {
        &self.host
    }

    /// Returns the port this observer will connect to
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # let builder = SpectatorBuilder::new();
    /// let port = builder.port();
    /// # assert_eq!(port, 5498);
    /// ```
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Returns the channel capacity
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # let builder = SpectatorBuilder::new();
    /// let capacity = builder.channel_capacity();
    /// # assert_eq!(capacity, 10000);
    /// ```
    pub fn channel_capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the selected metrics
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # use std::collections::HashSet;
    /// # let builder = SpectatorBuilder::new().select(&["cost", "best_cost"]);
    /// let selected = builder.selected();
    /// # assert_eq!(selected, &HashSet::from(["cost".to_string(), "best_cost".to_string()]));
    /// ```
    pub fn selected(&self) -> &HashSet<String> {
        &self.selected
    }

    /// Build a Spectator instance from the builder
    ///
    /// This initiates the connection to the Spectator instance.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// let spectator = SpectatorBuilder::new().build();
    /// ```
    pub fn build(self) -> Spectator {
        let (tx, rx) = tokio::sync::mpsc::channel(self.capacity);
        std::thread::spawn(move || sender(rx, self.host, self.port));

        Spectator {
            tx,
            name: self.name,
            sending: true,
            selected: self.selected,
        }
    }
}

/// Observer which sends data to Spectator
// No #[derive(Clone)] on purpose: A clone will only overwrite information already present in the
// Spectator since the name cannot be changed.
pub struct Spectator {
    tx: tokio::sync::mpsc::Sender<Message>,
    name: String,
    sending: bool,
    selected: HashSet<String>,
}

impl Spectator {
    /// Places a `Message` on the sending queue
    fn send_msg(&mut self, message: Message) {
        if self.sending {
            if let Err(e) = self.tx.blocking_send(message) {
                eprintln!("Can't send to Spectator: {e}. Will stop trying.");
                self.sending = false;
            }
        }
    }

    /// Returns the name of the Spectator instance
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin_observer_spectator::SpectatorBuilder;
    /// # let spectator = SpectatorBuilder::new().with_name("flup").build();
    /// let name = spectator.name();
    /// # assert_eq!(name, &"flup".to_string());
    /// ```
    pub fn name(&self) -> &String {
        &self.name
    }
}

impl<I> Observe<I> for Spectator
where
    I: State,
    I::Param: IntoIterator<Item = I::Float> + Clone,
    I::Float: ArgminFloat,
    f64: From<I::Float>,
{
    /// Log basic information about the optimization after initialization.
    fn observe_init(&mut self, name: &str, state: &I, kv: &KV) -> Result<(), Error> {
        let init_param = state.get_param().map(|init_param| {
            init_param
                .clone()
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        });

        let message = Message::NewRun {
            name: self.name.clone(),
            solver: name.to_string(),
            max_iter: state.get_max_iters(),
            target_cost: f64::from(state.get_target_cost()),
            init_param,
            settings: kv.clone(),
            selected: self.selected.clone(),
        };

        self.send_msg(message);

        Ok(())
    }

    /// Logs information about the progress of the optimization after every iteration.
    fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
        let mut kv = kv.clone();
        let iter = state.get_iter();
        kv.insert("best_cost", state.get_best_cost().into());
        kv.insert("cost", state.get_cost().into());
        kv.insert("iter", iter.into());

        let message_samples = Message::Samples {
            name: self.name.clone(),
            iter,
            time: Duration::try_from(
                state
                    .get_time()
                    .unwrap_or(std::time::Duration::from_secs(0)),
            )?,
            termination_status: state.get_termination_status().clone(),
            kv,
        };

        self.send_msg(message_samples);

        let message_func_counts = Message::FuncCounts {
            name: self.name.clone(),
            iter,
            kv: state.get_func_counts().clone(),
        };

        self.send_msg(message_func_counts);

        if let Some(param) = state.get_param() {
            let param = param.clone().into_iter().map(f64::from).collect::<Vec<_>>();

            let message_param = Message::Param {
                name: self.name.clone(),
                iter,
                param,
            };

            self.send_msg(message_param);
        }

        if state.is_best() {
            if let Some(best_param) = state.get_best_param() {
                let best_param = best_param
                    .clone()
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();

                let message_best_param = Message::BestParam {
                    name: self.name.clone(),
                    iter,
                    param: best_param,
                };

                self.send_msg(message_best_param);
            }
        }

        Ok(())
    }

    fn observe_final(&mut self, state: &I) -> Result<(), Error> {
        let message = Message::Termination {
            name: self.name.clone(),
            termination_status: state.get_termination_status().clone(),
        };
        self.send_msg(message);
        Ok(())
    }
}
