// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Loggers based on the `slog` crate
//!
//! These loggers write general information about the optimization and information about the
//! progress of the optimization for each iteration of the algorithm to screen or into a file as
//! JSON.
//! See [`SlogLogger`] for details regarding usage.

use crate::core::observers::Observe;
use crate::core::{Error, State, KV};
use slog;
use slog::{info, o, Drain, Key, Record, Serializer};
use slog_async;
use slog_async::OverflowStrategy;
#[cfg(feature = "serde1")]
use slog_json;
use slog_term;
#[cfg(feature = "serde1")]
use std::fs::OpenOptions;
#[cfg(feature = "serde1")]
use std::sync::Mutex;

/// A logger using the [`slog`](https://crates.io/crates/slog) crate as backend.
#[derive(Clone)]
pub struct SlogLogger {
    /// the logger
    logger: slog::Logger,
}

impl SlogLogger {
    /// Log to the terminal.
    ///
    /// Will block execution when buffer is full.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::SlogLogger;
    ///
    /// let terminal_logger = SlogLogger::term();
    /// ```
    pub fn term() -> Self {
        SlogLogger::term_internal(OverflowStrategy::Block)
    }

    /// Log to the terminal without blocking execution.
    ///
    /// Messages may be lost in case of buffer overflow.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::SlogLogger;
    ///
    /// let terminal_logger = SlogLogger::term_noblock();
    /// ```
    pub fn term_noblock() -> Self {
        SlogLogger::term_internal(OverflowStrategy::Drop)
    }

    /// Create terminal logger with a given `OverflowStragegy`.
    fn term_internal(overflow_strategy: OverflowStrategy) -> Self {
        let decorator = slog_term::TermDecorator::new().build();
        let drain = slog_term::FullFormat::new(decorator)
            .use_original_order()
            .build()
            .fuse();
        let drain = slog_async::Async::new(drain)
            .overflow_strategy(overflow_strategy)
            .build()
            .fuse();
        SlogLogger {
            logger: slog::Logger::root(drain, o!()),
        }
    }

    /// Log JSON to a file while blocking execution in case of full buffers.
    ///
    /// If `truncate` is set to `true`, the content of existing log files will be cleared.
    ///
    /// Only available if the `serde1` feature is set.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::SlogLogger;
    ///
    /// let file_logger = SlogLogger::file("logfile.log", true);
    /// ```
    #[cfg(feature = "serde1")]
    pub fn file<N: AsRef<str>>(file: N, truncate: bool) -> Result<Self, Error> {
        SlogLogger::file_internal(file, OverflowStrategy::Block, truncate)
    }

    /// Log JSON to a file without blocking execution.
    ///
    /// Messages may be lost in case of buffer overflow.
    ///
    /// If `truncate` is set to `true`, the content of existing log files will be cleared.
    ///
    /// Only available if the `serde1` feature is set.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::SlogLogger;
    ///
    /// let file_logger = SlogLogger::file_noblock("logfile.log", true);
    /// ```
    #[cfg(feature = "serde1")]
    pub fn file_noblock<N: AsRef<str>>(file: N, truncate: bool) -> Result<Self, Error> {
        SlogLogger::file_internal(file, OverflowStrategy::Drop, truncate)
    }

    /// Create file logger with a given `OverflowStrategy`.
    ///
    /// Only available if the `serde1` feature is set.
    #[cfg(feature = "serde1")]
    fn file_internal<N: AsRef<str>>(
        file: N,
        overflow_strategy: OverflowStrategy,
        truncate: bool,
    ) -> Result<Self, Error> {
        // Logging to file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(truncate)
            .open(file.as_ref())?;
        let drain = Mutex::new(slog_json::Json::new(file).build()).map(slog::Fuse);
        let drain = slog_async::Async::new(drain)
            .overflow_strategy(overflow_strategy)
            .build()
            .fuse();
        Ok(SlogLogger {
            logger: slog::Logger::root(drain, o!()),
        })
    }
}

impl slog::KV for KV {
    fn serialize(&self, _record: &Record, serializer: &mut dyn Serializer) -> slog::Result {
        for idx in self.kv.iter().rev() {
            serializer.emit_str(Key::from(idx.0), &idx.1.to_string())?;
        }
        Ok(())
    }
}

struct LogState<I>(I);

impl<I> slog::KV for LogState<&'_ I>
where
    I: State,
{
    fn serialize(&self, _record: &Record, serializer: &mut dyn Serializer) -> slog::Result {
        for (k, &v) in self.0.get_func_counts().iter() {
            serializer.emit_u64(Key::from(k.clone()), v)?;
        }
        serializer.emit_str(Key::from("best_cost"), &self.0.get_best_cost().to_string())?;
        serializer.emit_str(Key::from("cost"), &self.0.get_cost().to_string())?;
        serializer.emit_u64(Key::from("iter"), self.0.get_iter())?;
        Ok(())
    }
}

impl<I> Observe<I> for SlogLogger
where
    I: State,
{
    /// Log basic information about the optimization after initialization.
    fn observe_init(&mut self, msg: &str, kv: &KV) -> Result<(), Error> {
        info!(self.logger, "{}", msg; kv);
        Ok(())
    }

    /// Logs information about the progress of the optimization after every iteration.
    fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
        info!(self.logger, ""; LogState(state), kv);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_slog_loggerv, SlogLogger);
}
