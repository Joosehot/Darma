//! Conversation from people's conversations (DailyDialog, CC BY-NC-SA 4.0, 13,118 everyday dialogues).
//!
//! Every turn is kept together with the turn that answered it, and is matched sentence by sentence: what is said to
//! Darma is matched to one sentence of a person's turn. If that sentence ended the turn, Darma answers with what the
//! next person answered; if the person went on, Darma goes on as they did ("Hello, darling!" -> "Did you have a good
//! day today?", not the answer to that question). What is said to Darma is measured against every kept
//! turn: their shared runs of one to three words, each weighed by how rare it is among the turns (log N/df), the
//! overlap normalised by both lengths (a cosine). This is the overlap of the two sentences' states in the core, with
//! rarer runs counting more; it is computed with an index of runs rather than with the vectors, which for 87,000
//! turns would not fit. Darma answers with what people answered to the turn that matched best. Nothing is written by
//! hand: every answer is a person's.

use crate::lexicon::tokens;
use std::collections::HashMap;

pub const RUN: usize = 3;
/// how alike a kept turn must be to what was said for Darma to answer as people did (a cosine; set by hand)
pub const TALK_AT: f32 = 0.3;
/// below this, a match is too far to answer by; Darma asks back instead (set by hand)
pub const LOOSE_AT: f32 = 0.2;
/// how much the conversation so far weighs: likeness of the turn before a kept turn to what Darma said last (set by
/// hand)
pub const CONTEXT: f32 = 0.5;

pub struct Dialog {
    /// every kept turn: its words, its text, and the turn that answered it (if any)
    turns: Vec<(Vec<String>, String, Option<usize>)>,
    /// the turn before each kept turn in its dialogue
    prev: Vec<Option<usize>>,
    /// the first words of people's turns, with how many of those turns were questions (ended in "?") and how many
    /// there were: what opens a question, learned
    openers: HashMap<String, (u32, u32)>,
    /// of people's questions opening with each word: how many were answered with yes or no, of how many
    yes_no: HashMap<String, (u32, u32)>,
    /// what people most often answered with a question of their own (what they say when they need more to go on)
    pub ask_back: Vec<String>,
    /// every sentence of every turn: its words, its text, the turn it is in, and whether it ends that turn
    units: Vec<(Vec<String>, String, usize, bool)>,
    /// the sentence before each sentence in its dialogue
    unit_prev: Vec<Option<usize>>,
    index: HashMap<String, Vec<u32>>,
    df: HashMap<String, u32>,
    norm: Vec<f32>,
    recent: Vec<usize>,
}

/// a turn's sentences (DailyDialog spaces its marks: "Hello , darling ! Did you ...?")
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for tok in text.split_whitespace() {
        cur.push(tok);
        if matches!(tok, "." | "!" | "?") || tok.ends_with(['!', '?']) {
            out.push(cur.join(" "));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.join(" "));
    }
    out.into_iter().filter(|s| !tokens(s).is_empty()).collect()
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
        let mut prev = Vec::new();
        for d in &dialogs {
            let start = turns.len();
            for (k, u) in d.iter().enumerate() {
                let next = (k + 1 < d.len()).then(|| start + k + 1);
                turns.push((tokens(u), u.clone(), next));
                prev.push((k > 0).then(|| start + k - 1));
            }
        }
        let mut openers: HashMap<String, (u32, u32)> = HashMap::new();
        for (w, text, _) in &turns {
            if let Some(f) = w.first() {
                let e = openers.entry(f.clone()).or_default();
                e.1 += 1;
                if text.trim_end().ends_with('?') {
                    e.0 += 1;
                }
            }
        }
        // which questions people answer with yes or no ("Do you ...?" yes; "How's it going?" no); the words of a
        // yes or a no are written by hand
        let mut yes_no: HashMap<String, (u32, u32)> = HashMap::new();
        for (w, text, next) in &turns {
            if let (Some(f), Some(n), true) = (w.first(), next, text.trim_end().ends_with('?')) {
                let e = yes_no.entry(f.clone()).or_default();
                e.1 += 1;
                if matches!(turns[*n].0.first().map(|s| s.as_str()), Some("yes" | "no" | "yeah" | "yep" | "nope" | "sure")) {
                    e.0 += 1;
                }
            }
        }
        // the answers that were themselves questions, most common first ("Why?", "What do you mean?")
        let mut back: HashMap<String, u32> = HashMap::new();
        for (_, _, next) in &turns {
            if let Some(n) = next {
                let a = tidy(&turns[*n].1);
                if a.ends_with('?') && a.split_whitespace().count() <= 4 {
                    *back.entry(a).or_default() += 1;
                }
            }
        }
        let mut ask_back: Vec<(String, u32)> = back.into_iter().collect();
        ask_back.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let ask_back: Vec<String> = ask_back.into_iter().take(8).map(|x| x.0).collect();
        // the sentences, each with the turn it is in and whether it ends it
        let mut units: Vec<(Vec<String>, String, usize, bool)> = Vec::new();
        let mut unit_prev: Vec<Option<usize>> = Vec::new();
        for (ti, (_, text, _)) in turns.iter().enumerate() {
            let ss = sentences(text);
            let first_of_dialogue = prev[ti].is_none();
            for (k, s) in ss.iter().enumerate() {
                let up = if k == 0 && first_of_dialogue { None } else { units.len().checked_sub(1) };
                unit_prev.push(up);
                units.push((tokens(s), s.clone(), ti, k + 1 == ss.len()));
            }
        }
        let mut index: HashMap<String, Vec<u32>> = HashMap::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        for (i, (w, _, _, _)) in units.iter().enumerate() {
            for r in runs(w) {
                *df.entry(r.clone()).or_default() += 1;
                index.entry(r).or_default().push(i as u32);
            }
        }
        let n = units.len() as f32;
        let idf = |r: &str| (n / *df.get(r).unwrap_or(&1) as f32).ln();
        let norm = units.iter().map(|(w, _, _, _)| runs(w).iter().map(|r| idf(r).powi(2)).sum::<f32>().sqrt().max(1e-6)).collect();
        Some(Dialog { turns, prev, openers, yes_no, ask_back, units, unit_prev, index, df, norm, recent: Vec::new() })
    }

    fn idf(&self, r: &str) -> f32 {
        (self.units.len() as f32 / *self.df.get(r).unwrap_or(&1) as f32).ln()
    }

    /// what follows a sentence: the rest of its turn if the person went on, else the next person's turn
    fn follows(&self, u: usize) -> Option<String> {
        let (_, _, ti, last) = &self.units[u];
        if *last {
            self.turns[*ti].2.map(|n| tidy(&self.turns[n].1))
        } else {
            let rest: Vec<&str> = self.units[u + 1..].iter().take_while(|x| x.2 == *ti).map(|x| x.1.as_str()).collect();
            Some(tidy(&rest.join(" ")))
        }
    }

    fn has_follow(&self, u: usize) -> bool {
        let (_, _, ti, last) = &self.units[u];
        !*last || self.turns[*ti].2.is_some()
    }

    /// how alike a sentence's runs (known ones, with their norm) are to kept turn i (cosine)
    fn cos_to(&self, q: &[String], qn: f32, i: usize) -> f32 {
        let tr = runs(&self.units[i].0);
        q.iter().filter(|r| tr.binary_search(r).is_ok()).map(|r| self.idf(r).powi(2)).sum::<f32>() / (qn * self.norm[i])
    }

    fn known_runs(&self, words: &[String]) -> (Vec<String>, f32) {
        // only the runs people's conversations hold: a word never met there (a name, "Darma") says nothing of which
        // turn this is like, so it does not weigh in
        let q: Vec<String> = runs(words).into_iter().filter(|r| self.df.contains_key(r)).collect();
        let qn = q.iter().map(|r| self.idf(r).powi(2)).sum::<f32>().sqrt().max(1e-6);
        (q, qn)
    }

    /// the kept turn most like what was said: (turn, likeness to what was said). Two things beyond likeness:
    ///   the topic: the rarest known word of what was said; a turn holding it is looked for first, and taken if it is
    ///              alike enough (TALK_AT), so "what do you think about dogs" stays with dogs;
    ///   the conversation: a turn whose own turn before is like what Darma said last gains CONTEXT times that likeness.
    pub fn nearest(&self, words: &[String], context: Option<&[String]>, skip: &[usize]) -> Option<(usize, f32)> {
        let (q, qn) = self.known_runs(words);
        let mut score: HashMap<u32, f32> = HashMap::new();
        for r in &q {
            let Some(ids) = self.index.get(r) else { continue };
            let w = self.idf(r).powi(2);
            // a run every tenth turn has says little and costs much: left out of the sum
            if ids.len() > self.units.len() / 10 {
                continue;
            }
            for &i in ids {
                *score.entry(i).or_default() += w;
            }
        }
        let mut cands: Vec<(usize, f32)> = score
            .into_iter()
            .map(|(i, s)| (i as usize, s / (qn * self.norm[i as usize])))
            .filter(|(i, _)| self.has_follow(*i) && !skip.contains(i))
            .collect();
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // the topic: the rarest known word, if it is rare at all (in fewer than one turn in a hundred: "dogs",
        // "football"; not "know")
        let topic: Option<&String> = q
            .iter()
            .filter(|r| !r.contains(' '))
            .max_by(|a, b| self.idf(a).partial_cmp(&self.idf(b)).unwrap())
            .filter(|r| self.idf(r) >= 100f32.ln());
        // the turns holding the topic are taken from all candidates (a long frame like "what do you think about"
        // would push them out of the best few hundred)
        let topical: Vec<(usize, f32)> = match topic {
            Some(tw) => cands.iter().filter(|(i, _)| self.units[*i].0.contains(tw)).take(300).cloned().collect(),
            None => Vec::new(),
        };
        cands.truncate(300);
        let ctx = context.map(|c| self.known_runs(c));
        let total = |&(i, s): &(usize, f32)| -> f32 {
            let c = match (&ctx, self.unit_prev[i]) {
                (Some((cq, cn)), Some(p)) => CONTEXT * self.cos_to(cq, *cn, p),
                _ => 0.0,
            };
            s + c
        };
        let pick = |set: Vec<&(usize, f32)>| -> Option<(usize, f32)> {
            set.into_iter()
                .map(|x| (x.0, x.1, total(x)))
                // of equally alike: a sentence someone else answered before one its speaker went on from; then the
                // shortest answer; then the shortest sentence
                .max_by(|a, b| {
                    a.2.partial_cmp(&b.2)
                        .unwrap()
                        .then(self.units[a.0].3.cmp(&self.units[b.0].3))
                        .then(self.follows(b.0).map_or(0, |s| s.len()).cmp(&self.follows(a.0).map_or(0, |s| s.len())))
                        .then(self.units[b.0].1.len().cmp(&self.units[a.0].1.len()))
                        .then(b.0.cmp(&a.0))
                })
                .map(|x| (x.0, x.1))
        };
        if let Some(b) = pick(topical.iter().collect()).filter(|b| b.1 >= TALK_AT) {
            return Some(b);
        }
        pick(cands.iter().collect())
    }

    /// Darma's answer: what people answered to the turn most like this one; (answer, the turn it matched, likeness).
    /// The same answer is not given twice in a row.
    pub fn reply(&mut self, words: &[String], context: Option<&[String]>) -> Option<(String, String, f32)> {
        let recent = self.recent.clone();
        let (i, s) = self.nearest(words, context, &recent)?;
        let ans = self.follows(i)?;
        self.recent.push(i);
        if self.recent.len() > 8 {
            self.recent.remove(0);
        }
        Some((ans, tidy(&self.units[i].1), s))
    }

    /// the best answer that passes `test` (None: passes; Some(reason): let go), trying up to ten turns in order of
    /// likeness; returns (answer, the turn it matched, likeness, what was let go and why)
    pub fn reply_where(&mut self, words: &[String], context: Option<&[String]>, test: &dyn Fn(&str) -> Option<String>) -> Option<(String, String, f32, Vec<String>)> {
        let mut skip = self.recent.clone();
        let mut let_go = Vec::new();
        for _ in 0..10 {
            let (i, s) = self.nearest(words, context, &skip)?;
            let ans = self.follows(i)?;
            match test(&ans) {
                None => {
                    self.recent.push(i);
                    if self.recent.len() > 8 {
                        self.recent.remove(0);
                    }
                    return Some((ans, tidy(&self.units[i].1), s, let_go));
                }
                Some(why) => {
                    let_go.push(format!("\"{ans}\" ({why})"));
                    skip.push(i);
                }
            }
        }
        None
    }

    /// does a turn opening with this word ask, as people's turns do (at least 20 of them, three in four ending in
    /// "?": "Hello?" on the phone is a question often, but not that often)?
    pub fn opens_question(&self, first: &str) -> bool {
        self.openers.get(first).is_some_and(|(q, n)| *n >= 20 && *q * 4 >= *n * 3)
    }

    /// is a question opening with this word one people answer with yes or no (at least one in four times)? A word
    /// that opens no questions at all ("the dog chased the cat?") is a statement asked, and is answered so too.
    pub fn asks_yes_no(&self, first: &str) -> bool {
        match self.yes_no.get(first) {
            Some((y, n)) if *n >= 20 => *y * 4 >= *n,
            _ => true,
        }
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }
}

/// `darma dialog-eval`: DailyDialog's own test dialogues. For each test turn that was answered, Darma's answer is
/// compared with the person's real answer by the share of their words in common (F1), against the same for an answer
/// drawn at random from the kept turns (the line to beat).
pub fn eval(path_train: &str, path_test: &str) {
    for use_context in [false, true] {
        println!("== {}", if use_context { "with the turn before as context" } else { "each turn alone" });
        eval_once(path_train, path_test, use_context);
    }
}

fn eval_once(path_train: &str, path_test: &str, use_context: bool) {
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
        for (j, pair) in dlg.windows(2).enumerate() {
            let (said, real) = (tokens(&pair[0]), tokens(&pair[1]));
            let before = (j > 0).then(|| tokens(&dlg[j - 1]));
            let Some((ans, matched, s)) = d.reply(&said, if use_context { before.as_deref() } else { None }) else { continue };
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
            if shown < 3 && n % 1000 == 1 {
                shown += 1;
                println!("  said: {}\n  matched: {matched}  ({s:.2})\n  Darma: {ans}\n  person: {}\n", tidy(&pair[0]), tidy(&pair[1]));
            }
        }
    }
    println!("{n} test turns answered; word overlap with the real answer (F1): Darma {:.3}, a random kept answer {:.3}; mean likeness of the match {:.2}", f / n as f32, fr / n as f32, liked / n as f32);
    let m = (n - dup).max(1) as f32;
    println!("of them {dup} matched a training turn at 0.9 or more (repeated dialogues); on the other {}: Darma {:.3}, random {:.3}", n - dup, nf / m, nfr / m);
}
