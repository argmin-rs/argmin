// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod connection;
mod data;
mod message;
mod plotter;
mod telemetry;

use anyhow::Error;
use uuid::Uuid;

use plotter::PlotterApp;
use telemetry::{get_subscriber, init_subscriber};

const NAME: &str = "argmin-plotter";
const DEFAULT_HOST: &str = "0.0.0.0";

fn run() -> Result<(), Error> {
    // Set up logging
    let subscriber = get_subscriber(NAME.into(), "info".into(), std::io::stdout);
    init_subscriber(subscriber);
    let run_id = Uuid::new_v4();
    let span = tracing::info_span!(NAME, %run_id);
    let _span_guard = span.enter();

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        NAME,
        options,
        Box::new(|cc| Box::new(PlotterApp::new(cc).expect("Failed to start GUI"))),
    )
    .expect("Failed to start GUI.");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
        std::process::exit(1);
    }
}
