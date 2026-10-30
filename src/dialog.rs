//! Conversation from people's conversations (DailyDialog, CC BY-NC-SA 4.0, 13,118 everyday dialogues).
//!
//! Every turn is kept together with the turn that answered it. What is said to Darma is measured against every kept
//! turn: their shared runs of one to three words, each weighed by how rare it is among the turns (log N/df), the
//! overlap normalised by both lengths (a cosine). This is the overlap of the two sentences' states in the core, with
//! rarer runs counting more; it is computed with an index of runs rather than with the vectors, which for 87,000
//! turns would not fit. Darma answers with what people answered to the turn that matched best. Nothing is written by
//! hand: every answer is a person's.

use crate::lexicon::tokens;
use std::collections::HashMap;

pub const RUN: usize = 3;

pub struct Dialog {
    /// every kept turn: its words, its text, and the turn that answered it (if any)
    turns: Vec<(Vec<String>, String, Option<usize>)>,
    index: HashMap<String, Vec<u32>>,
    df: HashMap<String, u32>,
    norm: Vec<f32>,
    recent: Vec<usize>,
}

fn runs(w: &[String]) -> Vec<String> {
    let mut r: Vec<String> = Vec::new();
    for n in 1..=RUN {
        for i in 0..w.len().saturating_sub(n - 1) {
            r.push(w[i..i + n].join(" "));
        }
    }
    r.sort();
    r.dedup();
    r
}

/// DailyDialog's text has a space before every mark ("Say , Jim , how ...?"): put them back where they belong
pub fn tidy(s: &str) -> String {
    let mut t = s.to_string();
    for (a, b) in [(" ’ ", "'"), (" ,", ","), (" .", "."), (" ?", "?"), (" !", "!"), (" ' ", "'"), (" n't", "n't"), (" 's", "'s"), (" 're", "'re"), (" 'm", "'m"), (" 'll", "'ll"), (" 've", "'ve"), (" 'd", "'d"), ("’ ", "'"), (" ’", "'")] {
        t = t.replace(a, b);
    }
    t
}

impl Dialog {
    pub fn load(path: &str) -> Option<Dialog> {
        let dialogs: Vec<Vec<String>> = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let mut turns = Vec::new();
        for d in &dialogs {
            let start = turns.len();
            for (k, u) in d.iter().enumerate() {
                let next = (k + 1 < d.len()).then(|| start + k + 1);
                turns.push((tokens(u), u.clone(), next));
            }
        }
        let mut index: HashMap<String, Vec<u32>> = HashMap::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        for (i, (w, _, _)) in turns.iter().enumerate() {
            for r in runs(w) {
                *df.entry(r.clone()).or_default() += 1;
                index.entry(r).or_default().push(i as u32);
            }
        }
        let n = turns.len() as f32;
        let idf = |r: &str| (n / *df.get(r).unwrap_or(&1) as f32).ln();
        let norm = turns.iter().map(|(w, _, _)| runs(w).iter().map(|r| idf(r).powi(2)).sum::<f32>().sqrt().max(1e-6)).collect();
        Some(Dialog { turns, index, df, norm, recent: Vec::new() })
    }

    fn idf(&self, r: &str) -> f32 {
        (self.turns.len() as f32 / *self.df.get(r).unwrap_or(&1) as f32).ln()
    }

    /// the kept turn most like what was said, and how alike (cosine): (turn, likeness)
    pub fn nearest(&self, words: &[String], skip: &[usize]) -> Option<(usize, f32)> {
        let q = runs(words);
        let qn = q.iter().map(|r| self.idf(r).powi(2)).sum::<f32>().sqrt().max(1e-6);
        let mut score: HashMap<u32, f32> = HashMap::new();
        for r in &q {
            let Some(ids) = self.index.get(r) else { continue };
            let w = self.idf(r).powi(2);
            // a run every tenth turn has says little and costs much: left out of the sum
            if ids.len() > self.turns.len() / 10 {
                continue;
            }
            for &i in ids {
                *score.entry(i).or_default() += w;
            }
        }
        score
            .into_iter()
            .map(|(i, s)| (i as usize, s / (qn * self.norm[i as usize])))
            .filter(|(i, _)| self.turns[*i].2.is_some() && !skip.contains(i))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(b.0.cmp(&a.0)))
    }

    /// Darma's answer: what people answered to the turn most like this one; (answer, the turn it matched, likeness).
    /// The same answer is not given twice in a row.
    pub fn reply(&mut self, words: &[String]) -> Option<(String, String, f32)> {
        let recent = self.recent.clone();
        let (i, s) = self.nearest(words, &recent)?;
        let next = self.turns[i].2?;
        self.recent.push(i);
        if self.recent.len() > 8 {
            self.recent.remove(0);
        }
        Some((tidy(&self.turns[next].1), tidy(&self.turns[i].1), s))
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }
}

/// `darma dialog-eval`: DailyDialog's own test dialogues. For each test turn that was answered, Darma's answer is
/// compared with the person's real answer by the share of their words in common (F1), against the same for an answer
/// drawn at random from the kept turns (the line to beat).
pub fn eval(path_train: &str, path_test: &str) {
    let Some(mut d) = Dialog::load(path_train) else { return println!("{path_train}: download DailyDialog first") };
    let test: Vec<Vec<String>> = serde_json::from_str(&std::fs::read_to_string(path_test).unwrap()).unwrap();
    let f1 = |a: &[String], b: &[String]| -> f32 {
        let common = a.iter().filter(|x| b.contains(x)).count() as f32;
        if common == 0.0 {
            0.0
        } else {
            let p = common / a.len() as f32;
            let r = common / b.len() as f32;
            2.0 * p * r / (p + r)
        }
    };
    let (mut n, mut f, mut fr, mut liked) = (0usize, 0f32, 0f32, 0f32);
    // DailyDialog repeats some dialogues between its train and test splits: turns matched at 0.9 or more are counted
    // apart (they are remembered, not answered)
    let (mut dup, mut nf, mut nfr) = (0usize, 0f32, 0f32);
    let mut shown = 0;
    let mut k = 7usize;
    for dlg in &test {
        for pair in dlg.windows(2) {
            let (said, real) = (tokens(&pair[0]), tokens(&pair[1]));
            let Some((ans, matched, s)) = d.reply(&said) else { continue };
            d.recent.clear();
            n += 1;
            f += f1(&tokens(&ans), &real);
            liked += s;
            k = (k * 1103515245 + 12345) % 2147483648;
            let rnd = &d.turns[k % d.len()].0;
            fr += f1(rnd, &real);
            if s >= 0.9 {
                dup += 1;
            } else {
                nf += f1(&tokens(&ans), &real);
                nfr += f1(rnd, &real);
            }
            if shown < 8 && n % 300 == 1 {
                shown += 1;
                println!("  said: {}\n  matched: {matched}  ({s:.2})\n  Darma: {ans}\n  person: {}\n", tidy(&pair[0]), tidy(&pair[1]));
            }
        }
    }
    println!("{n} test turns answered; word overlap with the real answer (F1): Darma {:.3}, a random kept answer {:.3}; mean likeness of the match {:.2}", f / n as f32, fr / n as f32, liked / n as f32);
    let m = (n - dup).max(1) as f32;
    println!("of them {dup} matched a training turn at 0.9 or more (repeated dialogues); on the other {}: Darma {:.3}, random {:.3}", n - dup, nf / m, nfr / m);
}
