//! Darma v0: text -> understanding -> answer, a small measured trial of the core.
//! Stage 1: the phase vectors (src/vector.rs) and their tests (`cargo test --release -- --nocapture`).

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
    /// make data/ (stage 2)
    GenerateData,
    /// strike the training sentences into memory (stage 3)
    Train,
    /// all measures and the baseline (stage 4)
    Eval,
    /// the whole path for one sentence
    Ask { sentence: String },
}

fn main() {
    match Cli::parse().cmd {
        Cmd::GenerateData | Cmd::Train | Cmd::Eval | Cmd::Ask { .. } => {
            eprintln!("not built yet: only stage 1 (src/vector.rs) is done; run `cargo test --release -- --nocapture`");
            std::process::exit(2);
        }
    }
}
