//! Darma v0: text -> understanding -> answer, a small measured trial of the core.
//! vector.rs phase vectors; data.rs the downloaded sets; lexicon.rs atoms; memory.rs lightning training; egg.rs the
//! swim and the egg's three layers; eval.rs the measures and the baseline.

mod compose;
mod data;
mod dialog;
mod egg;
mod eval;
mod lexicon;
mod me;
mod memory;
mod qa;
mod quantum;
mod understand;
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
    /// the chat page in a browser
    Serve {
        #[arg(long, default_value_t = 5900)]
        port: u16,
        #[arg(long, default_value = "darma_memory.txt")]
        memory: String,
        /// begin with every UD training sentence in memory
        #[arg(long)]
        seed_ud: bool,
    },
    /// conversation from people's conversations: DailyDialog's test dialogues against Darma's answers (a short run)
    DialogEval {
        #[arg(default_value_t = 150)]
        dialogues: usize,
    },
    /// understanding: acts and emotions read from a line's state (DailyDialog's marks)
    UnderstandEval,
    /// Darma's own answers (understanding, decision, a form filled) against people's real answers
    ComposeEval,
    /// how Darma reads lines: their act and emotion
    Read { lines: Vec<String> },
    /// senses as measurements: which sense does memory hold in this place?
    QaSenses,
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
    /// the words whose meanings in people's conversations are nearest
    NearTalk { words: Vec<String> },
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
        Cmd::QaSenses => qa::senses(cli.dim, cli.seed),
        Cmd::UnderstandEval => understand::eval(cli.dim, cli.seed),
        Cmd::ComposeEval => compose::eval(cli.dim, cli.seed),
        Cmd::Read { lines } => {
            let at = lexicon::Atoms::new(cli.dim, cli.seed);
            let (u, _) = understand::strike_or_load(&at).expect("download DailyDialog first");
            for l in lines {
                let r = u.read(&lexicon::tokens(&l));
                let top = |v: &[(&str, f32)]| v.iter().take(3).map(|(n, s)| format!("{n} {s:.3}")).collect::<Vec<_>>().join(", ");
                println!("{l:<40} act: {:<11} ({})   emotion: {:<10} ({})", r.act, top(&r.act_scores), r.emotion, top(&r.emotion_scores));
            }
        }
        Cmd::DialogEval { dialogues } => dialog::eval(dialogues),
        Cmd::NearTalk { words } => {
            let d = dialog::Dialog::load("data/raw/dailydialog/train_marked.json").expect("download DailyDialog first");
            for w in words {
                println!("{w}: {}", d.near(&w, 8).iter().map(|(x, s)| format!("{x} {s:.2}")).collect::<Vec<_>>().join(", "));
            }
        }
        Cmd::Chat { beta: _, memory, fresh } => qa::chat(cli.dim, cli.seed, &memory, fresh),
        Cmd::Serve { port, memory, seed_ud } => qa::serve(cli.dim, cli.seed, &memory, port, seed_ud),
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
