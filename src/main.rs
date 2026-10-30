//! Darma v0: text -> understanding -> answer, a small measured trial of the core.
//! Stage 1: the phase vectors (src/vector.rs). Stage 2: the data, read from downloaded human-annotated sets (src/data.rs, `darma prepare-data`).

mod data;
mod vector;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "darma", about = "Darma v0: understanding by phase vectors")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// read the downloaded sets in data/raw into data/*.json (stage 2)
    PrepareData,
    /// strike the training sentences into memory (stage 3)
    Train,
    /// all measures and the baseline (stage 4)
    Eval,
    /// the whole path for one sentence
    Ask { sentence: String },
}

fn main() {
    match Cli::parse().cmd {
        Cmd::PrepareData => data::prepare(std::path::Path::new("data")).expect("reading data/raw, writing data/"),
        Cmd::Train | Cmd::Eval | Cmd::Ask { .. } => {
            eprintln!("not built yet: stages 1-2 are done");
            std::process::exit(2);
        }
    }
}
