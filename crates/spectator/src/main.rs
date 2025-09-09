// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod app;
mod connection;
mod data;
mod message;
mod telemetry;

use anyhow::Error;
use uuid::Uuid;

use app::PlotterApp;
use clap::Parser;
use telemetry::{get_subscriber, init_subscriber};

use spectator::DEFAULT_PORT;

const NAME: &str = "spectator";
const DEFAULT_HOST: &str = "0.0.0.0";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Host address to bind to
    #[arg(short, long, default_value_t = DEFAULT_HOST.to_string())]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    port: u16,
}

fn run() -> Result<(), Error> {
    let Args { host, port } = Args::parse();

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
        Box::new(move |cc| Ok(Box::new(PlotterApp::new(cc, host, port)?))),
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
