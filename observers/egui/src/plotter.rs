// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use argmin::core::TerminationStatus;
use dashmap::DashMap;
use eframe::{
    egui::{
        self,
        plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints},
        CentralPanel, Id, LayerId, Ui, WidgetText,
    },
    epaint::Color32,
};
use egui_dock::{DockArea, Node, Style, TabViewer, Tree};
use egui_extras::{Column, TableBuilder};
use itertools::Itertools;
use time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio_stream::StreamExt;
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use uuid::Uuid;

use crate::{
    message::Message,
    telemetry::{get_subscriber, init_subscriber},
};

const NAME: &'static str = "argmin-plotter";

#[derive(Clone, Debug)]
enum View {
    Metrics,
    Params,
    Overview,
}

struct MyContext {
    pub style: Option<Style>,
    open_tabs: HashSet<RunName>,
    storage: Arc<Storage>,
    views: HashMap<RunName, View>,
}

type RunName = String;
type MetricName = String;
type Samples = HashMap<MetricName, Vec<(u64, f64)>>;
type Selected = HashMap<MetricName, bool>;

pub struct General {
    max_iter: u64,
    target_cost: f64,
    curr_iter: u64,
    best_iter: u64,
    curr_cost: f64,
    curr_best_cost: f64,
    time: Duration,
    termination_status: TerminationStatus,
}

pub struct Storage {
    general: DashMap<RunName, General>,
    data: DashMap<RunName, Samples>,
    param: DashMap<RunName, (u64, Vec<f64>)>,
    best_param: DashMap<RunName, (u64, Vec<f64>)>,
    selected: DashMap<RunName, Selected>,
    tree: Arc<Mutex<Tree<RunName>>>,
}

impl Storage {
    fn new(tree: Arc<Mutex<Tree<String>>>) -> Self {
        Storage {
            general: DashMap::new(),
            data: DashMap::new(),
            param: DashMap::new(),
            best_param: DashMap::new(),
            selected: DashMap::new(),
            tree,
        }
    }
}

pub struct PlotterApp {
    context: MyContext,
    tree: Arc<Mutex<Tree<String>>>,
}

#[tokio::main]
async fn server(storage: Arc<Storage>, ctx: egui::Context) -> Result<(), anyhow::Error> {
    let listener = TcpListener::bind("127.0.0.1:5498").await?;
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
                                max_iter,
                                target_cost,
                            } => {
                                let mut tree = storage.tree.lock().unwrap();
                                tree.push_to_first_leaf(name.clone());
                                drop(tree);
                                storage.general.insert(
                                    name.clone(),
                                    General {
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
                                        if k == "cost" {
                                            general.curr_cost = kv_val;
                                        }
                                        if k == "best_cost" {
                                            general.curr_best_cost = kv_val;
                                        }
                                        if let Some(val) = data.get_mut(&k) {
                                            // TODO: unwrap
                                            val.push((iter, kv_val));
                                        } else {
                                            let mut arr = Vec::with_capacity(1000);
                                            // TODO: unwrap
                                            arr.push((iter, kv_val));
                                            data.insert(k.clone(), arr);
                                            if !selected.contains_key(&k) {
                                                selected.insert(k.clone(), true);
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

impl PlotterApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, anyhow::Error> {
        // Set up logging
        let subscriber = get_subscriber(NAME.into(), "info".into(), std::io::stdout);
        init_subscriber(subscriber);

        let run_id = Uuid::new_v4();
        let span = tracing::info_span!(NAME, %run_id);
        let _span_guard = span.enter();
        let tree: Tree<String> = Tree::new(vec![]);
        // tree.push_to_first_leaf("Blah".to_owned());
        // let [a, b] = tree.split_left(NodeIndex::root(), 0.3, vec!["Inspector".to_owned()]);
        // let [_, _] = tree.split_below(
        //     a,
        //     0.7,
        //     vec!["File Browser".to_owned(), "Asset Manager".to_owned()],
        // );
        // let [_, _] = tree.split_below(b, 0.5, vec!["Hierarchy".to_owned()]);

        let mut open_tabs = HashSet::new();

        for node in tree.iter() {
            if let Node::Leaf { tabs, .. } = node {
                for tab in tabs {
                    open_tabs.insert(tab.clone());
                }
            }
        }

        let tree = Arc::new(Mutex::new(tree));

        // NEEDS TO BE PART OF STORAGE
        // let selected = if let Some(storage) = cc.storage {
        //     eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        // } else {
        //     HashMap::new()
        // };

        let storage = Arc::new(Storage::new(Arc::clone(&tree)));
        let db2 = Arc::clone(&storage);
        let egui_ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || server(db2, egui_ctx));

        let context = MyContext {
            style: None,
            open_tabs,
            storage,
            views: HashMap::new(),
        };
        Ok(Self { context, tree })
    }
}

impl TabViewer for MyContext {
    type Tab = String;

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        self.show_plots(tab, ui);
        // match tab.as_str() {
        //     "Simple Demo" => self.show_plots(ui),
        //     // "Style Editor" => self.style_editor(ui),
        //     _ => {
        //         ui.label(tab.as_str());
        //     }
        // }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        tab.as_str().into()
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> bool {
        self.open_tabs.remove(tab);
        true
    }
}

impl MyContext {
    fn show_metrics(&mut self, name: &String, ui: &mut Ui) {
        let data = self.storage.data.get(name).unwrap();
        let mut selected = self.storage.selected.get_mut(name).unwrap();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                TableBuilder::new(ui)
                    .column(Column::auto().at_least(120.0))
                    // .resizable(true)
                    .header(20.0, |mut header| {
                        header.col(|ui| {
                            ui.heading("Metrics");
                        });
                    })
                    .body(|mut body| {
                        body.row(30.0, |mut row| {
                            row.col(|ui| {
                                for key in data.keys().sorted() {
                                    if let Some(selected) = selected.get_mut(key) {
                                        ui.checkbox(selected, key);
                                    }
                                }
                            });
                        });
                    });
            });
            egui::ScrollArea::vertical()
                .id_source("fufu")
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        let height = ui.available_height();

                        let keys = selected
                            .iter()
                            .filter(|(_, &v)| v)
                            .map(|(k, _)| k)
                            .sorted()
                            .collect::<Vec<_>>();

                        let num_keys = keys.len() as f32;

                        for key in keys {
                            if data.contains_key(key.as_str()) {
                                ui.group(|ui| {
                                    // dodgy
                                    ui.set_max_height(height / num_keys - 20.0);
                                    ui.label(key);
                                    let curve: PlotPoints = data
                                        .get(key)
                                        .unwrap()
                                        .iter()
                                        .map(|(i, t)| [f64::from(*i as u32), *t])
                                        .collect();
                                    let line = Line::new(curve);
                                    Plot::new(key)
                                        // .view_aspect(4.0)
                                        // .height(height / (num_keys + 1.0))
                                        .allow_scroll(false)
                                        .show(ui, |plot_ui| plot_ui.line(line));
                                });
                            }
                        }
                    });
                });
        });
    }

    fn show_params(&mut self, name: &String, ui: &mut Ui) {
        // egui::ScrollArea::vertical()
        //     .id_source("ufuf")
        //     .show(ui, |ui| {
        ui.vertical(|ui| {
            let height = ui.available_height() * 0.95;

            if let Some(param) = self.storage.param.get(name) {
                ui.group(|ui| {
                    ui.set_max_height(height / 2.0);
                    // ui.label("Current parameter vector");
                    let chart = BarChart::new(
                        param
                            .1
                            .iter()
                            .enumerate()
                            .map(|(x, f)| Bar::new(x as f64, *f).width(0.95))
                            .collect(),
                    )
                    .color(Color32::LIGHT_BLUE)
                    .name(format!("Current (iter: {})", param.0));

                    Plot::new("Current Parameter Vector")
                        .legend(Legend::default())
                        // .clamp_grid(true)
                        .allow_scroll(false)
                        .allow_zoom(false)
                        .allow_boxed_zoom(false)
                        .allow_drag(false)
                        .auto_bounds_x()
                        .auto_bounds_y()
                        .set_margin_fraction([0.1, 0.3].into())
                        .reset()
                        .show(ui, |plot_ui| plot_ui.bar_chart(chart))
                        .response;
                });
            }

            if let Some(best_param) = self.storage.best_param.get(name) {
                // ui.label("Current best parameter vector");
                ui.group(|ui| {
                    ui.set_max_height(height / 2.0);
                    let chart = BarChart::new(
                        best_param
                            .1
                            .iter()
                            .enumerate()
                            .map(|(x, f)| Bar::new(x as f64, *f).width(0.95))
                            .collect(),
                    )
                    .color(Color32::LIGHT_BLUE)
                    .name(format!("Best (iter: {})", best_param.0));

                    Plot::new("Best Parameter Vector")
                        .legend(Legend::default())
                        // .clamp_grid(true)
                        .allow_scroll(false)
                        .allow_zoom(false)
                        .allow_boxed_zoom(false)
                        .allow_drag(false)
                        .auto_bounds_x()
                        .auto_bounds_y()
                        .set_margin_fraction([0.1, 0.3].into())
                        .reset()
                        .show(ui, |plot_ui| plot_ui.bar_chart(chart))
                        .response;
                });
            }
        });
        // });
    }

    fn show_overview(&mut self, name: &String, ui: &mut Ui) {
        if let Some(general) = self.storage.general.get(name) {
            ui.vertical(|ui| {
                ui.label(format!(
                    "Maximum number of iterations: {}",
                    general.max_iter
                ));
                ui.label(format!(
                    "Current iteration: {} ({:.2}%)",
                    general.curr_iter,
                    100.0 / (general.max_iter as f64) * (general.curr_iter as f64)
                ));
                ui.label(format!("Target cost: {}", general.target_cost));
                ui.label(format!("Current cost: {}", general.curr_cost));
                ui.label(format!(
                    "Current best cost: {} (in iteration {})",
                    general.curr_best_cost, general.best_iter
                ));
                ui.label(format!("Elapsed time: {}", general.time));
                if let TerminationStatus::Terminated(reason) = &general.termination_status {
                    ui.label(format!("Termination reason: {}", reason));
                }
            });
        }
    }

    fn show_plots(&mut self, name: &String, ui: &mut Ui) {
        ui.horizontal_top(|ui| {
            if ui.button("Metrics").clicked() {
                self.views.insert(name.clone(), View::Metrics);
            }
            if ui.button("Parameters").clicked() {
                self.views.insert(name.clone(), View::Params);
            }
            if ui.button("Overview").clicked() {
                self.views.insert(name.clone(), View::Overview);
            }
        });
        match self.views.get(name) {
            Some(View::Metrics) => self.show_metrics(name, ui),
            Some(View::Params) => self.show_params(name, ui),
            Some(View::Overview) => self.show_overview(name, ui),
            None => self.show_metrics(name, ui),
        }
    }
}

impl eframe::App for PlotterApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.context.storage.selected);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.0);

        // TopBottomPanel::top("egui_dock::MenuBar").show(ctx, |ui| {
        //     egui::menu::bar(ui, |ui| {
        //         ui.menu_button("View", |ui| {
        //             // allow certain tabs to be toggled
        //             for tab in &["File Browser", "Asset Manager"] {
        //                 if ui
        //                     .selectable_label(self.context.open_tabs.contains(*tab), *tab)
        //                     .clicked()
        //                 {
        //                     if let Some(index) = self.tree.find_tab(&tab.to_string()) {
        //                         self.tree.remove_tab(index);
        //                         self.context.open_tabs.remove(*tab);
        //                     } else {
        //                         self.tree.push_to_focused_leaf(tab.to_string());
        //                     }

        //                     ui.close_menu();
        //                 }
        //             }
        //         });
        //     })
        // });

        CentralPanel::default().show(ctx, |_ui| {
            let layer_id = LayerId::background();
            let max_rect = ctx.available_rect();
            let clip_rect = ctx.available_rect();
            let id = Id::new("egui_dock::DockArea");
            let mut ui = Ui::new(ctx.clone(), layer_id, id, max_rect, clip_rect);

            let mut style = self
                .context
                .style
                .get_or_insert(Style::from_egui(&ui.ctx().style()))
                .clone();
            style.show_close_buttons = false;
            let mut tree = self.tree.lock().unwrap();
            DockArea::new(&mut tree)
                .style(style)
                .show_inside(&mut ui, &mut self.context);
        });
    }
}
