// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin_plotter::Message;
use futures::SinkExt;
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

#[tokio::main(flavor = "current_thread")]
pub(crate) async fn sender(
    mut rx: tokio::sync::mpsc::Receiver<Message>,
    host: String,
    port: u16,
) -> Result<(), anyhow::Error> {
    let codec = LengthDelimitedCodec::new();
    if let Ok(stream) = TcpStream::connect(format!("{host}:{port}")).await {
        let mut stream = Framed::new(stream, codec);
        while let Some(msg) = rx.recv().await {
            let msg = msg.pack()?;
            stream.send(msg).await?;
        }
    } else {
        eprintln!("Can't connect to argmin-plotter on {host}:{port}");
    }
    Ok(())
}
