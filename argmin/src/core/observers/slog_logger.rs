// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Loggers based on the `slog` crate

use crate::core::{ArgminFloat, ArgminKV, Error, IterState, Observe, State};
use slog;
use slog::{info, o, Drain, Record, Serializer, KV};
use slog_async;
use slog_async::OverflowStrategy;
#[cfg(feature = "serde1")]
use slog_json;
use slog_term;
#[cfg(feature = "serde1")]
use std::fs::OpenOptions;
#[cfg(feature = "serde1")]
use std::sync::Mutex;

/// A logger based on `slog`
#[derive(Clone)]
pub struct ArgminSlogLogger {
    /// the logger
    logger: slog::Logger,
}

impl ArgminSlogLogger {
    /// Log to the terminal in a blocking way
    pub fn term() -> Self {
        ArgminSlogLogger::term_internal(OverflowStrategy::Block)
    }

    /// Log to the terminal in a non-blocking way (in case of overflow, messages are dropped)
    pub fn term_noblock() -> Self {
        ArgminSlogLogger::term_internal(OverflowStrategy::Drop)
    }

    /// Actual implementation of the logging to the terminal
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
        ArgminSlogLogger {
            logger: slog::Logger::root(drain, o!()),
        }
    }

    /// Log JSON to a file in a blocking way
    ///
    /// If `truncate` is set to `true`, the content of existing log files at `file` will be
    /// cleared.
    ///
    /// Only available when the `serde1` feature is set.
    #[cfg(feature = "serde1")]
    pub fn file(file: &str, truncate: bool) -> Result<Self, Error> {
        ArgminSlogLogger::file_internal(file, OverflowStrategy::Block, truncate)
    }

    /// Log JSON to a file in a non-blocking way (in case of overflow, messages are dropped)
    ///
    /// If `truncate` is set to `true`, the content of existing log files at `file` will be
    /// cleared.
    ///
    /// Only available when the `serde1` feature is set.
    #[cfg(feature = "serde1")]
    pub fn file_noblock(file: &str, truncate: bool) -> Result<Self, Error> {
        ArgminSlogLogger::file_internal(file, OverflowStrategy::Drop, truncate)
    }

    #[cfg(feature = "serde1")]
    /// Actual implementaiton of logging JSON to file
    ///
    /// Only available when the `serde1` feature is set.
    fn file_internal(
        file: &str,
        overflow_strategy: OverflowStrategy,
        truncate: bool,
    ) -> Result<Self, Error> {
        // Logging to file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(truncate)
            .open(file)?;
        let drain = Mutex::new(slog_json::Json::new(file).build()).map(slog::Fuse);
        let drain = slog_async::Async::new(drain)
            .overflow_strategy(overflow_strategy)
            .build()
            .fuse();
        Ok(ArgminSlogLogger {
            logger: slog::Logger::root(drain, o!()),
        })
    }
}

/// This type is necessary in order to be able to implement `slog::KV` on `ArgminKV`
pub struct ArgminSlogKV {
    /// Key value store
    pub kv: Vec<(&'static str, String)>,
}

impl KV for ArgminSlogKV {
    fn serialize(&self, _record: &Record, serializer: &mut dyn Serializer) -> slog::Result {
        for idx in self.kv.clone().iter().rev() {
            serializer.emit_str(idx.0, &idx.1.to_string())?;
        }
        Ok(())
    }
}

impl<P, G, H, J, F> KV for IterState<P, G, H, J, F>
where
    P: Clone,
    F: ArgminFloat,
{
    fn serialize(&self, _record: &Record, serializer: &mut dyn Serializer) -> slog::Result {
        // REENABLE THIS $%£"^! TODO TODO TODO TODO TODO
        // for (k, v) in self.get_func_counts().into_iter() {
        //     let key = k.clone();
        //     serializer.emit_u64(&k, v)?;
        // }
        serializer.emit_str("best_cost", &self.best_cost.to_string())?;
        serializer.emit_str("cost", &self.cost.to_string())?;
        serializer.emit_u64("iter", self.get_iter())?;
        Ok(())
    }
}

impl<'a> From<&'a ArgminKV> for ArgminSlogKV {
    fn from(i: &'a ArgminKV) -> ArgminSlogKV {
        ArgminSlogKV { kv: i.kv.clone() }
    }
}

impl<I: KV> Observe<I> for ArgminSlogLogger {
    /// Log general info
    fn observe_init(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
        info!(self.logger, "{}", msg; ArgminSlogKV::from(kv));
        Ok(())
    }

    /// This should be used to log iteration data only (because this is what may be saved in a CSV
    /// file or a database)
    fn observe_iter(&mut self, state: &I, kv: &ArgminKV) -> Result<(), Error> {
        info!(self.logger, ""; state, ArgminSlogKV::from(kv));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_slog_loggerv, ArgminSlogLogger);
}
