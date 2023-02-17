// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::{collections::HashMap, sync::Arc};

use argmin::core::TerminationStatus;
use argmin_plotter::DEFAULT_PORT;
use eframe::egui;
use time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio_stream::StreamExt;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use crate::message::Message;
use crate::DEFAULT_HOST;

use super::data::{General, Storage};

#[tokio::main]
pub async fn server(storage: Arc<Storage>, ctx: egui::Context) -> Result<(), anyhow::Error> {
    let listener = TcpListener::bind(format!("{DEFAULT_HOST}:{DEFAULT_PORT}")).await?;
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
                                storage.general.insert(
                                    name.clone(),
                                    General {
                                        solver,
                                        settings,
                                        selected,
                                        init_param,
                                        max_iter,
                                        target_cost,
                                        curr_iter: 0,
                                        best_iter: 0,
                                        curr_cost: std::f64::INFINITY,
                                        curr_best_cost: std::f64::INFINITY,
                                        time: Duration::new(0, 0),
                                        termination_status: TerminationStatus::NotTerminated,
                                    },
                                );
                                storage.data.insert(name.clone(), HashMap::new());
                                storage.selected.insert(name, HashMap::new());
                            }
                            Message::Samples {
                                name,
                                iter,
                                time,
                                termination_status,
                                kv,
                            } => {
                                if let (Some(mut data), Some(mut selected), Some(mut general)) = (
                                    storage.data.get_mut(&name),
                                    storage.selected.get_mut(&name),
                                    storage.general.get_mut(&name),
                                ) {
                                    general.curr_iter = iter;
                                    general.time = time;
                                    general.termination_status = termination_status;
                                    for (k, _) in kv.keys() {
                                        let kv_val = kv.get(&k).unwrap().get_float().unwrap();
                                        // for easier access in overview window
                                        if k == "cost" {
                                            general.curr_cost = kv_val;
                                        }
                                        if k == "best_cost" {
                                            general.curr_best_cost = kv_val;
                                        }
                                        if let Some(val) = data.get_mut(&k) {
                                            val.push([f64::from(iter as u32), kv_val]);
                                        } else {
                                            // maybe allocate depending on max_iter (but with an
                                            // upper limit of say 1M)
                                            let mut arr = Vec::with_capacity(1_000_000);
                                            arr.push([f64::from(iter as u32), kv_val]);
                                            data.insert(k.clone(), arr);
                                            if !selected.contains_key(&k) {
                                                if general.selected.is_empty() {
                                                    selected.insert(k.clone(), true);
                                                } else {
                                                    selected.insert(
                                                        k.clone(),
                                                        general.selected.contains(&k),
                                                    );
                                                }
                                            }
                                        };
                                    }
                                };
                            }
                            Message::Param { name, iter, param } => {
                                if let Some(mut storage_param) = storage.param.get_mut(&name) {
                                    *storage_param = (iter, param);
                                } else {
                                    storage.param.insert(name, (iter, param));
                                }
                            }
                            Message::BestParam { name, iter, param } => {
                                if let Some(mut general) = storage.general.get_mut(&name) {
                                    general.best_iter = iter;
                                }
                                if let Some(mut storage_best_param) =
                                    storage.best_param.get_mut(&name)
                                {
                                    *storage_best_param = (iter, param);
                                } else {
                                    storage.best_param.insert(name, (iter, param));
                                }
                            }
                            Message::Termination {
                                name,
                                termination_status,
                            } => {
                                if let Some(mut general) = storage.general.get_mut(&name) {
                                    general.termination_status = termination_status;
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
