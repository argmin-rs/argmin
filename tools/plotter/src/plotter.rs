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

use crate::{
    connection::server,
    data::{RunName, Storage},
};

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

pub struct PlotterApp {
    context: MyContext,
    tree: Arc<Mutex<Tree<String>>>,
}

impl PlotterApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, anyhow::Error> {
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
                            if let Some(data) = data.get(key) {
                                ui.group(|ui| {
                                    // dodgy
                                    ui.set_max_height(height / num_keys - 20.0);
                                    ui.label(key);
                                    let curve: PlotPoints = data.clone().into();
                                    let line = Line::new(curve).name(key);
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
        ui.vertical(|ui| {
            let height = ui.available_height() * 0.95;

            if let Some(best_param) = self.storage.best_param.get(name) {
                // ui.label("Current best parameter vector");
                ui.group(|ui| {
                    ui.set_max_height(height / 3.0);
                    let chart = BarChart::new(
                        best_param
                            .1
                            .iter()
                            .enumerate()
                            .map(|(x, f)| Bar::new(x as f64, *f).width(0.95))
                            .collect(),
                    )
                    .color(Color32::LIGHT_GREEN)
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
                        .show(ui, |plot_ui| plot_ui.bar_chart(chart));
                });
            }

            if let Some(param) = self.storage.param.get(name) {
                ui.group(|ui| {
                    ui.set_max_height(height / 3.0);
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
                        .show(ui, |plot_ui| plot_ui.bar_chart(chart));
                });
            }

            if let Some(general) = self.storage.general.get(name) {
                if let Some(ref init_param) = general.init_param {
                    ui.group(|ui| {
                        ui.set_max_height(height / 3.0);
                        let chart = BarChart::new(
                            init_param
                                .iter()
                                .enumerate()
                                .map(|(x, f)| Bar::new(x as f64, *f).width(0.95))
                                .collect(),
                        )
                        .color(Color32::LIGHT_RED)
                        .name("Initial");

                        Plot::new("Initial Parameter Vector")
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
                            .show(ui, |plot_ui| plot_ui.bar_chart(chart));
                    });
                }
            }
        });
    }

    fn show_overview(&mut self, name: &String, ui: &mut Ui) {
        if let Some(general) = self.storage.general.get(name) {
            ui.vertical(|ui| {
                ui.label(format!("Solver: {}", general.solver));
                general
                    .settings
                    .iter()
                    .map(|(k, v)| ui.label(format!("{}: {}", k, v)))
                    .count();
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
            style.show_close_buttons = true;
            let mut tree = self.tree.lock().unwrap();
            DockArea::new(&mut tree)
                .style(style)
                .show_inside(&mut ui, &mut self.context);
        });
    }
}
