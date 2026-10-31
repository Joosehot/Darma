//! Knowledge from Wikipedia (Simple English, the lead of every article): each sentence heard into a structure by
//! Darma's own hearing and struck into its knowledge once, kept with a register per word (the sentences holding it),
//! so a question finds at once the sentences that hold every word it asks with. Answers come from the same thought as
//! for what the speaker told (cpt::fit), the article whose subject is asked first.
//!
//! Read as written, with two readings set by hand: what stands in brackets is left out ("April (Apr.) is ..."), and a
//! sentence after the first that begins with "It", "They" or "He"/"She" speaks of the article's subject.
use crate::cpt::{Cpt, Node};
use crate::quantum::words_of;
use crate::syntax::par_map;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct Fact {
    pub title: u32,
    pub text: String,
    pub nodes: Vec<Node>,
}

#[derive(Serialize, Deserialize)]
pub struct Wiki {
    pub titles: Vec<String>,
    pub facts: Vec<Fact>,
    /// a register per base form: the sentences holding it
    pub index: HashMap<String, Vec<u32>>,
}

fn unbracket(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ").replace(" ,", ",").replace(" .", ".")
}

impl Wiki {
    /// read the first `limit` articles' leads (0: all), heard on every core
    pub fn strike(cpt: &Cpt, limit: usize) -> Option<Wiki> {
        let text = std::fs::read_to_string("data/wiki/simple_lead.tsv").ok()?;
        let lines: Vec<&str> = text.lines().take(if limit == 0 { usize::MAX } else { limit }).collect();
        let titles: Vec<String> = lines.iter().map(|l| l.split('\t').next().unwrap_or("").to_string()).collect();
        let sents: Vec<(u32, String)> = lines
            .iter()
            .enumerate()
            .flat_map(|(t, l)| {
                let mut f = l.split('\t');
                let title = f.next().unwrap_or("").to_string();
                f.enumerate()
                    .map(|(k, s)| {
                        let mut s = unbracket(s);
                        if k > 0 {
                            for p in ["It ", "They ", "He ", "She "] {
                                if let Some(rest) = s.strip_prefix(p) {
                                    s = format!("{title} {rest}");
                                    break;
                                }
                            }
                        }
                        (t as u32, s)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // heard with the GPU (the same reading, faster) in batches, or wave by wave on every core without one
        let facts: Vec<Fact> = match crate::fast::Reader::new(cpt) {
            Some(mut r) => {
                let mut out = Vec::with_capacity(sents.len());
                let t0 = std::time::Instant::now();
                for (b, ch) in sents.chunks(20000).enumerate() {
                    let ws: Vec<Vec<String>> = ch.iter().map(|(_, s)| words_of(s)).collect();
                    let nodes = r.read(cpt, &ws, ".");
                    out.extend(ch.iter().zip(nodes).map(|((t, s), n)| Fact { title: *t, text: s.clone(), nodes: n }));
                    eprintln!("  heard {} of {} sentences on the GPU, {:.0} s", out.len(), sents.len(), t0.elapsed().as_secs_f64());
                    let _ = b;
                }
                out
            }
            None => par_map(&sents, |(t, s)| Fact { title: *t, text: s.clone(), nodes: cpt.hear_fresh(&words_of(s), ".") }),
        };
        let mut index: HashMap<String, Vec<u32>> = HashMap::new();
        for (i, f) in facts.iter().enumerate() {
            let mut ls: Vec<&str> = f.nodes.iter().filter(|n| n.pos != "PUNCT" && n.pos != "DET").map(|n| n.lemma.as_str()).collect();
            ls.sort();
            ls.dedup();
            for l in ls {
                index.entry(l.to_string()).or_default().push(i as u32);
            }
        }
        Some(Wiki { titles, facts, index })
    }

    /// the most that was struck before (all articles if they were), without striking
    pub fn load() -> Option<Wiki> {
        let mut best: Option<(usize, String)> = None;
        for e in std::fs::read_dir("data/strike").ok()?.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if let Some(k) = n.strip_prefix("wiki_simple_").and_then(|x| x.strip_suffix(".json")).and_then(|x| x.parse::<usize>().ok()) {
                let k = if k == 0 { usize::MAX } else { k };
                if best.as_ref().map_or(true, |b| k > b.0) {
                    best = Some((k, e.path().to_string_lossy().to_string()));
                }
            }
        }
        serde_json::from_str(&std::fs::read_to_string(best?.1).ok()?).ok()
    }

    pub fn strike_or_load(cpt: &Cpt, limit: usize) -> Option<(Wiki, bool)> {
        let path = format!("data/strike/wiki_simple_{limit}.json");
        if let Some(w) = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Wiki>(&s).ok()) {
            return Some((w, true));
        }
        let w = Wiki::strike(cpt, limit)?;
        let _ = std::fs::create_dir_all("data/strike");
        let _ = std::fs::write(&path, serde_json::to_string(&w).unwrap());
        Some((w, false))
    }

    /// the sentences that answer a heard question: those holding every word it asks with (found through the rarest
    /// word's register), then cpt::fit; the article whose subject is a word of the question first, then the best
    /// fitting, the shortest
    pub fn answer(&self, q: &[Node], small: &std::collections::HashSet<String>) -> Vec<(f32, &Fact)> {
        let asked = crate::cpt::asked_with(q);
        let Some(rare) = asked.iter().filter(|w| !small.contains(**w) || asked.len() == 1).filter_map(|w| self.index.get(*w)).min_by_key(|v| v.len()) else { return Vec::new() };
        let words: Vec<String> = q.iter().map(|x| x.lemma.clone()).chain(q.iter().map(|x| x.word.clone())).collect();
        let about = |f: &Fact| words.contains(&self.titles[f.title as usize].to_lowercase());
        let mut out: Vec<(f32, u32, &Fact)> = rare.iter().map(|&i| (i, &self.facts[i as usize])).filter_map(|(i, f)| crate::cpt::fit(q, &f.nodes).map(|s| (s + if about(f) { 10.0 } else { 0.0 }, i, f))).collect();
        // the article's own sentences in their order: its first says what it is
        out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        out.into_iter().map(|(s, _, f)| (s, f)).collect()
    }
}

/// `darma wiki-test [articles]`: strike the leads (or load them) and ask a few questions written before the first run
pub fn test(limit: usize) {
    let t0 = std::time::Instant::now();
    let Some(cpt) = Cpt::load(1) else { return println!("UD not found") };
    println!("grammar, hearing and speech ready in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let Some((w, loaded)) = Wiki::strike_or_load(&cpt, limit) else { return println!("data/wiki/simple_lead.tsv missing: run python tools/wiki_export.py") };
    println!("{} {} sentences of {} articles in {:.1} s; {} words with a register", if loaded { "loaded" } else { "heard and struck" }, w.facts.len(), w.titles.len(), t1.elapsed().as_secs_f64(), w.index.len());
    for q in ["what is art", "what is april", "where is bangladesh", "is april a month", "is art a country", "what is a country", "what was yugoslavia", "how many days does april have"] {
        let qn = cpt.hear_fresh(&words_of(q), "?");
        let a = w.answer(&qn, &cpt.small);
        match a.first() {
            Some((s, f)) => println!("  {q}? {} {}   ({}, {s:.2})", if crate::cpt::hole(&qn).is_none() { "Yes." } else { "" }, f.text, w.titles[f.title as usize]),
            None => println!("  {q}? I don't know."),
        }
    }
    println!("all in {:.1} s", t0.elapsed().as_secs_f64());
}
