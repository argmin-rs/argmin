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
use egui_dock::Tree;
use time::Duration;

pub type RunName = String;
type MetricName = String;
type SettingName = String;
type Samples = HashMap<MetricName, Vec<[f64; 2]>>;
type Selected = HashMap<MetricName, bool>;

pub struct General {
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
}

pub struct Storage {
    pub general: DashMap<RunName, General>,
    pub data: DashMap<RunName, Samples>,
    pub param: DashMap<RunName, (u64, Vec<f64>)>,
    pub best_param: DashMap<RunName, (u64, Vec<f64>)>,
    pub selected: DashMap<RunName, Selected>,
    pub tree: Arc<Mutex<Tree<RunName>>>,
}

impl Storage {
    pub fn new(tree: Arc<Mutex<Tree<String>>>) -> Self {
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
