// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::collections::HashSet;

use anyhow::Error;
use argmin::core::{observers::Observe, ArgminFloat, State, KV};
use argmin_plotter::{Message, DEFAULT_PORT};
use futures::SinkExt;
use time::Duration;
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use uuid::Uuid;

const DEFAULT_HOST: &str = "127.0.0.1";

#[derive(Clone)]
pub struct PlotterBuilder {
    name: String,
    selected: HashSet<String>,
    capacity: usize,
    host: String,
    port: u16,
}

impl Default for PlotterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PlotterBuilder {
    pub fn new() -> Self {
        PlotterBuilder {
            name: Uuid::new_v4().to_string(),
            selected: HashSet::new(),
            capacity: 10_000,
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
        }
    }

    pub fn with_name<T: AsRef<str>>(mut self, name: T) -> Self {
        self.name = name.as_ref().to_string();
        self
    }

    pub fn select<T: AsRef<str>>(mut self, metrics: &[T]) -> Self {
        self.selected = metrics.iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    pub fn build(self) -> Plotter {
        let (tx, rx) = tokio::sync::mpsc::channel(self.capacity);
        std::thread::spawn(move || sender(rx, self.host, self.port));

        Plotter {
            tx,
            name: Uuid::new_v4().to_string(),
            sending: true,
            selected: self.selected,
        }
    }
}

/// todo
pub struct Plotter {
    tx: tokio::sync::mpsc::Sender<Message>,
    name: String,
    sending: bool,
    selected: HashSet<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn sender(
    mut rx: tokio::sync::mpsc::Receiver<Message>,
    host: String,
    port: u16,
) -> Result<(), anyhow::Error> {
    let codec = LengthDelimitedCodec::new();
    if let Ok(stream) = TcpStream::connect(format!("{host}:{port}")).await {
        let mut stream = Framed::new(stream, codec);
        while let Some(msg) = rx.recv().await {
            let msg = msg.pack()?;
            stream.send(msg).await?;
        }
    } else {
        eprintln!("Can't connect to argmin-plotter on {host}:{port}");
    }
    Ok(())
}

impl Plotter {
    fn send_msg(&mut self, message: Message) {
        if self.sending {
            if let Err(e) = self.tx.blocking_send(message) {
                eprintln!("Can't send to argmin-plotter: {e}. Will stop trying.");
                self.sending = false;
            }
        }
    }
}

impl<I> Observe<I> for Plotter
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
