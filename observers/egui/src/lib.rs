// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::{collections::HashMap, marker::PhantomData};

use anyhow::Error;
use argmin::core::{observers::Observe, ArgminFloat, KvValue, State, KV};
use eframe::egui;
use egui::plot::{Line, Plot, PlotPoints};
use egui_extras::{Column, TableBuilder};
use ipc_channel::ipc;

/// todo
#[derive(Clone)]
pub struct EguiObserver<F> {
    sender: ipc::IpcSender<KV>,
    _float: PhantomData<F>,
}

impl<F> EguiObserver<F> {
    /// todo
    pub fn new() -> Result<Self, Error> {
        procspawn::init();
        let (sender, recv) = ipc::channel()?;
        // Maybe start GUI only when the first data is coming in?
        let _ = procspawn::spawn(recv, |recv| {
            let options = eframe::NativeOptions::default();
            eframe::run_native(
                "egui observer",
                options,
                Box::new(|cc| Box::new(EguiApp::new(cc, recv))),
            );
        });
        Ok(Self {
            sender,
            _float: PhantomData,
        })
    }
}

/// TODO
struct EguiApp {
    data: HashMap<String, Vec<KvValue>>,
    selected: HashMap<String, bool>,
    recv: ipc::IpcReceiver<KV>,
}

impl EguiApp {
    pub fn new(cc: &eframe::CreationContext<'_>, recv: ipc::IpcReceiver<KV>) -> Self {
        let selected = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            HashMap::new()
        };
        Self {
            data: HashMap::new(),
            selected,
            recv,
        }
    }
}

impl eframe::App for EguiApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.selected);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Not great!!
        ctx.request_repaint();
        if let Ok(res) = self.recv.try_recv() {
            for k in res.keys() {
                let k = k.0;
                if let Some(val) = self.data.get_mut(&k) {
                    // TODO: unwrap
                    val.push(res.get(&k).unwrap().clone());
                } else {
                    let mut arr = Vec::with_capacity(1000);
                    // TODO: unwrap
                    arr.push(res.get(&k).unwrap().clone());
                    self.data.insert(k.clone(), arr);
                    if !self.selected.contains_key(&k) {
                        self.selected.insert(k.clone(), true);
                    }
                };
            }
        }

        egui::SidePanel::left("metric_selection").show(ctx, |ui| {
            TableBuilder::new(ui)
                .column(Column::auto())
                .header(20.0, |mut header| {
                    header.col(|ui| {
                        ui.heading("");
                    });
                })
                .body(|mut body| {
                    body.row(30.0, |mut row| {
                        row.col(|ui| {
                            for key in self.data.keys() {
                                if let Some(selected) = self.selected.get_mut(key) {
                                    ui.checkbox(selected, key);
                                }
                            }
                        });
                    });
                });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("egui observer");

            egui::ScrollArea::vertical().show(ui, |ui| {
                for key in self.selected.iter().filter(|(_, &v)| v).map(|(k, _)| k) {
                    ui.label(key);
                    let curve: PlotPoints = self
                        .data
                        .get(key)
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, t)| [f64::from(i as u32), t.get_float().unwrap()])
                        .collect();
                    let line = Line::new(curve);
                    Plot::new(key)
                        .view_aspect(4.0)
                        .allow_scroll(false)
                        .show(ui, |plot_ui| plot_ui.line(line));
                }
            });
        });
    }
}

impl<I, F: ArgminFloat> Observe<I> for EguiObserver<F>
where
    I: State<Float = F>,
{
    /// Log basic information about the optimization after initialization.
    fn observe_init(&mut self, _msg: &str, _kv: &KV) -> Result<(), Error> {
        Ok(())
    }

    /// Logs information about the progress of the optimization after every iteration.
    fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
        let mut kv = kv.clone();
        kv.insert("best_cost", state.get_best_cost().into());
        kv.insert("cost", state.get_cost().into());
        kv.insert("iter", state.get_iter().into());
        self.sender.send(kv.clone()).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    // send_sync_test!(argmin_egui_observer, EguiObserver);
}
