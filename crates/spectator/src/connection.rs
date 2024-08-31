// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::{collections::HashMap, sync::Arc};

use argmin::core::TerminationStatus;
use eframe::egui;
use time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio_stream::StreamExt;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use crate::data::{FuncCount, Metric};
use crate::{data::Run, message::Message};

use super::data::Storage;

#[tokio::main]
pub async fn server(
    storage: Arc<Storage>,
    ctx: egui::Context,
    host: String,
    port: u16,
) -> Result<(), anyhow::Error> {
    let listener = TcpListener::bind(format!("{host}:{port}")).await?;
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let storage = Arc::clone(&storage);
                tokio::spawn(handle_connection(stream, storage, ctx.clone()));
            }
            Err(e) => {
                tracing::error!("error: {e:?}");
            }
        }
    }
}

async fn handle_connection(
    stream: TcpStream,
    storage: Arc<Storage>,
    ctx: egui::Context,
) -> Result<(), anyhow::Error> {
    let codec = LengthDelimitedCodec::new();
    let mut lines = Framed::new(stream, codec);

    while let Some(result) = lines.next().await {
        ctx.request_repaint();
        match result {
            Ok(line) => {
                match Message::unpack(&line) {
                    Ok(msg) => {
                        match msg {
                            Message::NewRun {
                                name,
                                solver,
                                max_iter,
                                target_cost,
                                init_param,
                                settings,
                                selected,
                            } => {
                                let mut tree = storage.tree.lock().unwrap();
                                tree.push_to_first_leaf(name.clone());
                                drop(tree);

                                let settings = settings
                                    .kv
                                    .into_iter()
                                    .map(|(k, v)| (k, v.as_string()))
                                    .collect();

                                storage.runs.insert(
                                    name.clone(),
                                    Run {
                                        name: name.clone(),
                                        solver,
                                        settings,
                                        selected,
                                        init_param: init_param.clone(),
                                        max_iter,
                                        target_cost,
                                        curr_iter: 0,
                                        best_iter: 0,
                                        curr_cost: f64::INFINITY,
                                        curr_best_cost: f64::INFINITY,
                                        time: Duration::new(0, 0),
                                        termination_status: TerminationStatus::NotTerminated,
                                        metrics: HashMap::new(),
                                        func_counts: HashMap::new(),
                                        func_cumulative: true,
                                        param: init_param.clone().map(|ip| (0, ip)),
                                        best_param: init_param.map(|ip| (0, ip)),
                                    },
                                );
                            }
                            Message::Samples {
                                name,
                                iter,
                                time,
                                termination_status,
                                kv,
                            } => {
                                if let Some(mut run) = storage.runs.get_mut(&name) {
                                    run.curr_iter = iter;
                                    run.time = time;
                                    run.termination_status = termination_status;
                                    for (k, _) in kv.keys() {
                                        let kv_val = kv.get(&k).unwrap().get_float().unwrap();
                                        // for easier access in overview window
                                        if k == "cost" {
                                            run.curr_cost = kv_val;
                                        }
                                        if k == "best_cost" {
                                            run.curr_best_cost = kv_val;
                                        }
                                        if let Some(val) = run.metrics.get_mut(&k) {
                                            val.push([f64::from(iter as u32), kv_val]);
                                        } else {
                                            let mut metric = Metric::new();

                                            metric.selected(
                                                run.selected.is_empty()
                                                    || run.selected.contains(&k),
                                            );

                                            metric.push([f64::from(iter as u32), kv_val]);
                                            run.add_metric(&k, metric);
                                        }
                                    }
                                }
                            }
                            Message::FuncCounts { name, iter, kv } => {
                                if let Some(mut run) = storage.runs.get_mut(&name) {
                                    for k in kv.keys() {
                                        let counts = kv.get(k).unwrap();
                                        if let Some(val) = run.func_counts.get_mut(k) {
                                            val.push([
                                                f64::from(iter as u32),
                                                f64::from(*counts as u32),
                                            ]);
                                        } else {
                                            let mut count = FuncCount::new();

                                            count.push([
                                                f64::from(iter as u32),
                                                f64::from(*counts as u32),
                                            ]);
                                            run.add_func_counts(k, count);
                                        }
                                    }
                                }
                            }
                            Message::Param { name, iter, param } => {
                                if let Some(mut run) = storage.runs.get_mut(&name) {
                                    run.param = Some((iter, param));
                                }
                            }
                            Message::BestParam { name, iter, param } => {
                                if let Some(mut run) = storage.runs.get_mut(&name) {
                                    run.best_iter = iter;
                                    run.best_param = Some((iter, param));
                                }
                            }
                            Message::Termination {
                                name,
                                termination_status,
                            } => {
                                if let Some(mut run) = storage.runs.get_mut(&name) {
                                    run.termination_status = termination_status;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Error: {e:?}");
                    }
                }
            }
            Err(e) => {
                tracing::error!("Error on decoding from socket: {:?}", e);
            }
        }
    }
    Ok(())
}
