// Copyright 2018-2024 argmin developers
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
use egui_dock::DockState;
use itertools::Itertools;
use time::Duration;

pub type RunName = String;
type MetricName = String;
type CountName = String;
type SettingName = String;

pub struct Metric {
    data: Vec<[f64; 2]>,
    selected: bool,
}

impl Metric {
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(1_000_000),
            selected: true,
        }
    }

    pub fn push(&mut self, val: [f64; 2]) -> &mut Self {
        self.data.push(val);
        self
    }

    pub fn selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    pub fn get_data(&self) -> &Vec<[f64; 2]> {
        &self.data
    }
}

pub struct FuncCount {
    data: Vec<[f64; 2]>,
}

impl FuncCount {
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(1_000_000),
        }
    }

    pub fn push(&mut self, val: [f64; 2]) -> &mut Self {
        self.data.push(val);
        self
    }

    pub fn get_data(&self, cumulative: bool) -> Vec<[f64; 2]> {
        if cumulative {
            self.data.clone()
        } else {
            self.data
                .iter()
                .zip(std::iter::once(&[0.0, 0.0]).chain(self.data.iter()))
                .map(|(a, b)| [a[0], a[1] - b[1]])
                .collect()
        }
    }
}

pub struct Run {
    pub solver: String,
    pub settings: HashMap<SettingName, String>,
    pub selected: HashSet<String>,
    pub init_param: Option<Vec<f64>>,
    pub max_iter: u64,
    pub target_cost: f64,
    pub curr_iter: u64,
    pub best_iter: u64,
    pub curr_cost: f64,
    pub curr_best_cost: f64,
    pub time: Duration,
    pub termination_status: TerminationStatus,
    pub metrics: HashMap<MetricName, Metric>,
    pub func_counts: HashMap<CountName, FuncCount>,
    pub func_cumulative: bool,
    pub param: Option<(u64, Vec<f64>)>,
    pub best_param: Option<(u64, Vec<f64>)>,
}

impl Run {
    pub fn add_metric<T: AsRef<str>>(&mut self, name: T, metric: Metric) -> &mut Self {
        self.metrics.insert(name.as_ref().to_string(), metric);
        self
    }

    pub fn get_metrics(&mut self) -> Vec<(String, &mut bool)> {
        self.metrics
            .iter_mut()
            .map(|(k, m)| (k.clone(), &mut m.selected))
            .sorted()
            .collect()
    }

    pub fn get_selected_metrics(&self) -> Vec<String> {
        self.metrics
            .iter()
            .filter(|(_, m)| m.selected)
            .map(|(k, _)| k.clone())
            .sorted()
            .collect()
    }

    pub fn add_func_counts<T: AsRef<str>>(&mut self, name: T, count: FuncCount) -> &mut Self {
        self.func_counts.insert(name.as_ref().to_string(), count);
        self
    }
}

pub struct Storage {
    pub runs: DashMap<RunName, Run>,
    pub tree: Arc<Mutex<DockState<RunName>>>,
}

impl Storage {
    pub fn new(tree: Arc<Mutex<DockState<String>>>) -> Self {
        Storage {
            runs: DashMap::new(),
            tree,
        }
    }
}
