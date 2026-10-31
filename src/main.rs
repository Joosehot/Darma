//! Darma v0: text -> understanding -> answer, a small measured trial of the core.
//! vector.rs phase vectors; data.rs the downloaded sets; lexicon.rs atoms; memory.rs lightning training; egg.rs the
//! swim and the egg's three layers; eval.rs the measures and the baseline.

mod compose;
mod data;
mod dialog;
mod egg;
mod eval;
mod grammar;
mod lexicon;
mod me;
mod memory;
mod qa;
mod quantum;
mod understand;
mod vector;
mod words;
mod know;

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
    /// the marks given on the chat page: how many answers were sensible, by how Darma made them
    Ratings,
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
    /// the word network (Primo's WordNet): alike words and feelings against a shuffled network
    WordsTest,
    /// emotions read from the word network on DailyDialog (threshold on validation, measured on test)
    WordsEval,
    /// what Darma knows from reading the dictionary: what is X, is X a Y (fixed questions, shuffled control)
    KnowTest,
    /// how Darma read each word of the dictionary
    KnowShow { words: Vec<String> },
    /// what Darma read of each word
    WordsDefs { words: Vec<String> },
    /// Darma's own answers (understanding, decision, a form filled) against people's real answers
    ComposeEval,
    /// the questions Darma's grammar makes of a verb and a noun
    Ask2 { verb: String, noun: String },
    /// Darma's own answer to lines (understanding, decision, a form filled), with how it chose
    Compose { lines: Vec<String> },
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
        Cmd::WordsTest => words::test(cli.seed),
        Cmd::WordsEval => words::eval(cli.dim, cli.seed),
        Cmd::KnowTest => know::test(cli.seed),
        Cmd::KnowShow { words } => know::show(cli.seed, &words),
        Cmd::WordsDefs { words } => words::show(cli.seed, &words),
        Cmd::ComposeEval => compose::eval(cli.dim, cli.seed),
        Cmd::Ask2 { verb, noun } => {
            let g = grammar::Grammar::learn("data/ud_short.json").expect("run prepare-data first");
            println!("{} patterns (each seen in {} sentences or more)", g.patterns.len(), grammar::MIN_SEEN);
            for (s, p, n) in g.ask_about(&verb, &noun).iter().take(12) {
                println!("  {s:<40} {p} ({n})");
            }
            let verbs: std::collections::HashSet<String> = serde_json::from_str::<std::collections::HashMap<String, u32>>(&std::fs::read_to_string("data/verbs.json").unwrap()).unwrap().into_iter().filter(|x| x.1 >= 2).map(|x| x.0).collect();
            let o = g.openings("data/ud_short.json", "data/raw/dailydialog/train_marked.json", &verbs).unwrap();
            println!("{} question beginnings (each said {} times or more)", o.begins.len(), grammar::MIN_SEEN);
            for (s, b, n, with) in g.ask_with(&o, &verb, &noun, "you", 0).iter().take(10) {
                println!("  {s:<40} \"{b} ...\" weighs {n}{}", if *with { "" } else { " (never with this verb)" });
            }
        }
        Cmd::Compose { lines } => {
            let at = lexicon::Atoms::new(cli.dim, cli.seed);
            let (u, _) = understand::strike_or_load(&at).expect("download DailyDialog first");
            let c = compose::Composer::learn("data/raw/dailydialog/train_marked.json").unwrap();
            let d = dialog::Dialog::load("data/raw/dailydialog/train_marked.json").unwrap();
            for l in lines {
                match c.answer(&u, &d, &lexicon::tokens(&l), &|_| None) {
                    Some((a, why)) => println!("{l}
  -> {a}
     ({why})"),
                    None => println!("{l}
  -> (nothing)"),
                }
            }
        }
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
        Cmd::Ratings => qa::ratings(),
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
