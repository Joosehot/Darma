//! Darma v0: text -> understanding -> answer, a small measured trial of the core.
//! vector.rs phase vectors; data.rs the downloaded sets; lexicon.rs atoms; memory.rs lightning training; egg.rs the
//! swim and the egg's three layers; eval.rs the measures and the baseline.

mod data;
mod egg;
mod eval;
mod lexicon;
mod memory;
mod vector;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "darma", about = "Darma v0: understanding by phase vectors")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// the dimension of every vector
    #[arg(long, default_value_t = 1024, global = true)]
    dim: usize,
    #[arg(long, default_value_t = 2026, global = true)]
    seed: u64,
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
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::PrepareData => data::prepare(std::path::Path::new("data")).expect("reading data/raw, writing data/"),
        Cmd::Train => {
            let d = eval::data();
            let at = lexicon::Atoms::new(cli.dim, cli.seed);
            let (m, s) = eval::train(&at, &d);
            println!("one pass: {} clauses, {} sense sentences in {s:.2} s; {} senses with context; membrane {:.3}, fertilisation {:.3}", d.train.len(), d.senses_train.len(), m.contexts.len(), m.membrane, m.fertile);
        }
        Cmd::Eval => eval::run(cli.dim, cli.seed),
        Cmd::Ask { sentence } => eval::ask(cli.dim, cli.seed, &sentence),
    }
}
