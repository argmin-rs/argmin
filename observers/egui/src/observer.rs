// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::marker::PhantomData;

use anyhow::Error;
use argmin::core::{observers::Observe, ArgminFloat, State, KV};
use futures::SinkExt;
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use uuid::Uuid;

use crate::{message::Message, plotter::PlotterApp};

const HOST: &'static str = "127.0.0.1";
const PORT: u16 = 5498;

/// todo
pub struct EguiObserver<F> {
    tx: tokio::sync::mpsc::UnboundedSender<Message>,
    name: String,
    _float: PhantomData<F>,
}

#[tokio::main]
async fn sender(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Message>,
) -> Result<(), anyhow::Error> {
    let codec = LengthDelimitedCodec::new();
    let mut stream = Framed::new(TcpStream::connect(format!("{HOST}:{PORT}")).await?, codec);
    while let Some(msg) = rx.recv().await {
        let msg = msg.pack()?;
        stream.send(msg).await?;
    }
    Ok(())
}

impl<F> EguiObserver<F> {
    /// todo
    pub fn new() -> Result<Self, Error> {
        procspawn::init();
        // will be a dedicated application at some point
        let _ = procspawn::spawn((), |_| {
            let options = eframe::NativeOptions::default();
            eframe::run_native(
                "Plotter",
                options,
                // unwrap
                Box::new(|cc| Box::new(PlotterApp::new(cc).unwrap())),
            );
        });

        // todo
        std::thread::sleep(std::time::Duration::from_secs(1));

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        std::thread::spawn(|| sender(rx));

        Ok(Self {
            tx,
            name: Uuid::new_v4().to_string(),
            _float: PhantomData,
        })
    }
}

impl<I, F: ArgminFloat> Observe<I> for EguiObserver<F>
where
    I: State<Float = F>,
    I::Param: IntoIterator<Item = F> + Clone,
    f64: From<F>,
{
    /// Log basic information about the optimization after initialization.
    fn observe_init(&mut self, _msg: &str, _kv: &KV) -> Result<(), Error> {
        let message = Message::NewRun {
            name: self.name.clone(),
        };

        self.tx.send(message)?;

        Ok(())
    }

    /// Logs information about the progress of the optimization after every iteration.
    fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
        let mut kv = kv.clone();
        kv.insert("best_cost", state.get_best_cost().into());
        kv.insert("cost", state.get_cost().into());
        kv.insert("iter", state.get_iter().into());

        let message = Message::Samples {
            name: self.name.clone(),
            kv: kv.into(),
        };

        self.tx.send(message)?;

        if let Some(param) = state.get_param() {
            let param = param.clone().into_iter().map(f64::from).collect::<Vec<_>>();

            let message = Message::Param {
                name: self.name.clone(),
                param,
            };

            self.tx.send(message)?;
        }

        Ok(())
    }
}
