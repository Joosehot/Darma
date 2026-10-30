//! Darma v0: text -> understanding -> answer, a small measured trial of the core.
//! vector.rs phase vectors; data.rs the downloaded sets; lexicon.rs atoms; memory.rs lightning training; egg.rs the
//! swim and the egg's three layers; eval.rs the measures and the baseline.

mod data;
mod egg;
mod eval;
mod lexicon;
mod memory;
mod qa;
mod quantum;
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
    /// why the egg does not work: each layer measured on its own
    Diagnose,
    /// the quantum-minded core: told the training clauses, asked who/what (state, operators, measurement)
    Qa,
    /// the scale test: told every UD training sentence whole, asked about the marked clauses in their own words
    QaSentences,
    /// talk: tell Darma things and ask it with who/what
    Chat {
        /// how much a word's learned meaning turns its operator (0: none)
        #[arg(long, default_value_t = 0.0)]
        beta: f32,
        /// where what was told is kept between conversations
        #[arg(long, default_value = "darma_memory.txt")]
        memory: String,
        /// start with an empty memory
        #[arg(long)]
        fresh: bool,
    },
    /// the words whose meanings are nearest to these words' meanings
    Near { words: Vec<String> },
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
        Cmd::Diagnose => eval::diagnose(cli.dim, cli.seed),
        Cmd::Qa => qa::run(cli.dim, cli.seed),
        Cmd::QaSentences => qa::sentences(cli.dim, cli.seed),
        Cmd::Chat { beta, memory, fresh } => qa::chat(cli.dim, cli.seed, beta, &memory, fresh),
        Cmd::Ask { sentence } => eval::ask(cli.dim, cli.seed, &sentence),
        Cmd::Near { words } => {
            let d = eval::data();
            let at = lexicon::Atoms::new(cli.dim, cli.seed);
            let (m, _) = eval::train(&at, &d);
            for w in words {
                let v = m.meaning_of(&at, &w);
                let mut s: Vec<(f32, &String)> = m.meaning.iter().filter(|(k, _)| **k != w).map(|(k, x)| (v.sim(x), k)).collect();
                s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                println!("{w}: {}", s.iter().take(8).map(|(x, k)| format!("{k} {x:.2}")).collect::<Vec<_>>().join(", "));
            }
        }
    }
}
