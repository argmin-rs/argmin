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
    egui::{self, CentralPanel, Id, LayerId, Ui, WidgetText},
    epaint::Color32,
};
use egui_dock::{DockArea, DockState, Node, Style, TabViewer};
use egui_extras::{Column, TableBuilder};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};

use crate::{
    connection::server,
    data::{RunName, Storage},
};

#[derive(Clone, Debug)]
enum View {
    Metrics,
    Params,
    Overview,
    FuncCounts,
}

struct MyContext {
    pub style: Option<Style>,
    open_tabs: HashSet<RunName>,
    storage: Arc<Storage>,
    views: HashMap<RunName, View>,
}

pub struct PlotterApp {
    context: MyContext,
    dock_state: Arc<Mutex<DockState<String>>>,
}

impl PlotterApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        host: String,
        port: u16,
    ) -> Result<Self, anyhow::Error> {
        let dock_state: DockState<String> = DockState::new(vec![]);
        // tree.push_to_first_leaf("Blah".to_owned());
        // let [a, b] = tree.split_left(NodeIndex::root(), 0.3, vec!["Inspector".to_owned()]);
        // let [_, _] = tree.split_below(
        //     a,
        //     0.7,
        //     vec!["File Browser".to_owned(), "Asset Manager".to_owned()],
        // );
        // let [_, _] = tree.split_below(b, 0.5, vec!["Hierarchy".to_owned()]);

        let mut open_tabs = HashSet::new();

        for node in dock_state.main_surface().iter() {
            if let Node::Leaf { tabs, .. } = node {
                for tab in tabs {
                    open_tabs.insert(tab.clone());
                }
            }
        }

        let dock_state = Arc::new(Mutex::new(dock_state));

        // NEEDS TO BE PART OF STORAGE
        // let selected = if let Some(storage) = cc.storage {
        //     eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        // } else {
        //     HashMap::new()
        // };

        let storage = Arc::new(Storage::new(Arc::clone(&dock_state)));
        let db2 = Arc::clone(&storage);
        let egui_ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || server(db2, egui_ctx, host, port));

        let context = MyContext {
            style: None,
            open_tabs,
            storage,
            views: HashMap::new(),
        };
        Ok(Self {
            context,
            dock_state,
        })
    }
}

impl TabViewer for MyContext {
    type Tab = String;

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        self.show_plots(tab, ui);
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
        if let Some(mut run) = self.storage.runs.get_mut(name) {
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
                                    for (metric_name, selected) in run.get_metrics() {
                                        ui.checkbox(selected, metric_name);
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

                            let metric_names = run.get_selected_metrics();
                            let num_metrics = metric_names.len() as f32;

                            for name in metric_names {
                                if let Some(metric) = run.metrics.get(&name) {
                                    ui.group(|ui| {
                                        // dodgy
                                        ui.set_max_height(height / num_metrics - 20.0);
                                        let curve: PlotPoints = metric.get_data().clone().into();
                                        let line = Line::new(curve).name(&name);
                                        Plot::new(&name)
                                            .allow_scroll(false)
                                            .legend(Legend::default())
                                            .show(ui, |plot_ui| plot_ui.line(line));
                                    });
                                }
                            }
                        });
                    });
            });
        }
    }

    fn show_params(&mut self, name: &String, ui: &mut Ui) {
        if let Some(run) = self.storage.runs.get(name) {
            ui.vertical(|ui| {
                let height = ui.available_height() * 0.95;

                if let Some((iter, ref best_param)) = run.best_param {
                    ui.group(|ui| {
                        ui.set_max_height(height / 3.0);
                        let chart = BarChart::new(
                            best_param
                                .iter()
                                .enumerate()
                                .map(|(x, f)| Bar::new(x as f64, *f).width(0.95))
                                .collect(),
                        )
                        .color(Color32::LIGHT_GREEN)
                        .name(format!("Best (iter: {})", iter));

                        Plot::new("Best Parameter Vector")
                            .legend(Legend::default())
                            // .clamp_grid(true)
                            .allow_scroll(false)
                            .allow_zoom(false)
                            .allow_boxed_zoom(false)
                            .allow_drag(false)
                            .auto_bounds([true, true].into())
                            .set_margin_fraction([0.1, 0.3].into())
                            .reset()
                            .show(ui, |plot_ui| plot_ui.bar_chart(chart));
                    });
                }

                if let Some((iter, ref param)) = run.param {
                    ui.group(|ui| {
                        ui.set_max_height(height / 3.0);
                        let chart = BarChart::new(
                            param
                                .iter()
                                .enumerate()
                                .map(|(x, f)| Bar::new(x as f64, *f).width(0.95))
                                .collect(),
                        )
                        .color(Color32::LIGHT_BLUE)
                        .name(format!("Current (iter: {})", iter));

                        Plot::new("Current Parameter Vector")
                            .legend(Legend::default())
                            // .clamp_grid(true)
                            .allow_scroll(false)
                            .allow_zoom(false)
                            .allow_boxed_zoom(false)
                            .allow_drag(false)
                            .auto_bounds([true, true].into())
                            .set_margin_fraction([0.1, 0.3].into())
                            .reset()
                            .show(ui, |plot_ui| plot_ui.bar_chart(chart));
                    });
                }

                if let Some(ref init_param) = run.init_param {
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
                            .auto_bounds([true, true].into())
                            .set_margin_fraction([0.1, 0.3].into())
                            .reset()
                            .show(ui, |plot_ui| plot_ui.bar_chart(chart));
                    });
                }
            });
        }
    }

    fn show_func_counts(&mut self, name: &String, ui: &mut Ui) {
        if let Some(mut run) = self.storage.runs.get_mut(name) {
            ui.horizontal_top(|ui| {
                ui.checkbox(&mut run.func_cumulative, "Cumulative");
                egui::ScrollArea::vertical()
                    .id_source("func_counts")
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.set_max_height(ui.available_height());
                            Plot::new(name)
                                .allow_scroll(false)
                                .include_x(0.0)
                                .include_y(0.0)
                                .legend(Legend::default())
                                .show(ui, |plot_ui| {
                                    for name in run.func_counts.keys() {
                                        if let Some(counts) = run.func_counts.get(name) {
                                            let curve: PlotPoints =
                                                counts.get_data(run.func_cumulative).into();
                                            let line = Line::new(curve).name(name);
                                            plot_ui.line(line)
                                        }
                                    }
                                });
                        });
                    });
            });
        }
    }

    fn show_overview(&mut self, name: &String, ui: &mut Ui) {
        if let Some(run) = self.storage.runs.get(name) {
            ui.vertical(|ui| {
                ui.label(format!("Solver: {}", run.solver));
                run.settings
                    .iter()
                    .map(|(k, v)| ui.label(format!("{}: {}", k, v)))
                    .count();
                ui.label(format!("Maximum number of iterations: {}", run.max_iter));
                ui.label(format!(
                    "Current iteration: {} ({:.2}%)",
                    run.curr_iter,
                    100.0 / (run.max_iter as f64) * (run.curr_iter as f64)
                ));
                ui.label(format!("Target cost: {}", run.target_cost));
                ui.label(format!("Current cost: {}", run.curr_cost));
                ui.label(format!(
                    "Current best cost: {} (in iteration {})",
                    run.curr_best_cost, run.best_iter
                ));
                ui.label(format!("Elapsed time: {}", run.time));
                if let TerminationStatus::Terminated(reason) = &run.termination_status {
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
            if ui.button("Function evaluations").clicked() {
                self.views.insert(name.clone(), View::FuncCounts);
            }
            if ui.button("Overview").clicked() {
                self.views.insert(name.clone(), View::Overview);
            }
        });

        match self.views.get(name) {
            Some(View::Metrics) => self.show_metrics(name, ui),
            Some(View::Params) => self.show_params(name, ui),
            Some(View::FuncCounts) => self.show_func_counts(name, ui),
            Some(View::Overview) => self.show_overview(name, ui),
            None => self.show_metrics(name, ui),
        }
    }
}

impl eframe::App for PlotterApp {
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        // eframe::set_value(storage, eframe::APP_KEY, &self.context.storage.selected);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.0);

        // TopBottomPanel::top("egui_dock::MenuBar").show(ctx, |ui| {
        //     egui::menu::bar(ui, |ui| {
        //         ui.menu_button("View", |ui| {
        //             // allow certain tabs to be toggled
        //             for tab in self.context.open_tabs.iter() {
        //                 if ui
        //                     .selectable_label(self.context.open_tabs.contains(tab), tab)
        //                     .clicked()
        //                 {
        //                     let mut tree = self.tree.lock().unwrap();
        //                     if let Some(index) = tree.find_tab(&tab) {
        //                         tree.remove_tab(index);
        //                         // self.context.open_tabs.remove(&tab);
        //                     } else {
        //                         tree.push_to_focused_leaf(tab.to_string());
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

            let style = self
                .context
                .style
                .get_or_insert(Style::from_egui(&ui.ctx().style()))
                .clone();
            let mut dock_state = self.dock_state.lock().unwrap();
            DockArea::new(&mut dock_state)
                .style(style)
                .show_close_buttons(true)
                .show_inside(&mut ui, &mut self.context);
        });
    }
}
