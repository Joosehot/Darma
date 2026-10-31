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
/// the dimension of the words' meaning waves, and how far a word's neighbours reach (each side)
pub const MEANING_D: usize = 256;
pub const MEANING_WINDOW: usize = 2;
/// how much two lines' resonance in meaning weighs beside their shared runs (chosen on the validation split)
pub const RESONANCE: f32 = 0.3;

/// what understanding gives the choice: the act and emotion read in the line, the act people answer that act with,
/// and how much this weighs beside likeness
#[derive(Clone, Copy)]
pub struct Guide {
    pub act: usize,
    pub emotion: usize,
    pub answer_act: usize,
    pub weight: f32,
}

#[derive(serde::Deserialize)]
struct MarkedDialog {
    dialog: Vec<String>,
    act: Vec<i64>,
    emotion: Vec<i64>,
}

pub struct Dialog {
    /// each turn's act and emotion as people marked them (from the marked file)
    marks: Vec<Option<(usize, usize)>>,
    /// how people answer each act: counts of the answer's act
    pub act_after: Vec<Vec<u32>>,
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
    /// each sentence's runs (sorted), and what follows it (its text, its words joined): computed once, at loading
    unit_runs: Vec<Vec<String>>,
    follow: Vec<Option<String>>,
    follow_words: Vec<String>,
    index: HashMap<String, Vec<u32>>,
    df: HashMap<String, u32>,
    norm: Vec<f32>,
    recent: Vec<usize>,
    /// a buffer for summing scores, reused by every query
    scratch: std::cell::RefCell<Vec<f32>>,
    /// every word's meaning: a wave made of its neighbours in people's conversations (one pass), what all words share
    /// taken away; words used alike have waves alike
    meaning: HashMap<String, crate::vector::HVec>,
    pub resonance: f32,
    /// each word's nearest meanings, found when first asked (a cache)
    nearest_words: std::cell::RefCell<HashMap<String, Vec<(String, f32)>>>,
    /// whether a line's words also bring in the turns of the words nearest in meaning
    pub widen: bool,
}

/// how near in meaning a word must be to bring in its turns, and how many it may bring (set by hand)
pub const WIDEN_AT: f32 = 0.45;
pub const WIDEN_N: usize = 3;

/// a turn's sentences (DailyDialog spaces its marks: "Hello , darling ! Did you ...?")
pub fn sentences(text: &str) -> Vec<String> {
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
        let text = std::fs::read_to_string(path).ok()?;
        // the marked file (with acts and emotions) or the plain one
        let marked: Vec<MarkedDialog> = match serde_json::from_str::<Vec<MarkedDialog>>(&text) {
            Ok(m) => m,
            Err(_) => serde_json::from_str::<Vec<Vec<String>>>(&text).ok()?.into_iter().map(|d| MarkedDialog { act: vec![0; d.len()], emotion: vec![-1; d.len()], dialog: d }).collect(),
        };
        let mut turns = Vec::new();
        let mut prev = Vec::new();
        let mut marks = Vec::new();
        let mut act_after = vec![vec![0u32; 4]; 4];
        for d in &marked {
            let start = turns.len();
            for (k, u) in d.dialog.iter().enumerate() {
                let next = (k + 1 < d.dialog.len()).then(|| start + k + 1);
                turns.push((tokens(u), u.clone(), next));
                prev.push((k > 0).then(|| start + k - 1));
                let m = (d.act[k] > 0 && d.emotion[k] >= 0).then(|| ((d.act[k] as usize - 1).min(3), (d.emotion[k] as usize).min(6)));
                marks.push(m);
                if k + 1 < d.dialog.len() && d.act[k] > 0 && d.act[k + 1] > 0 {
                    act_after[(d.act[k] as usize - 1).min(3)][(d.act[k + 1] as usize - 1).min(3)] += 1;
                }
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
        let unit_runs: Vec<Vec<String>> = units.iter().map(|u| runs(&u.0)).collect();
        let mut index: HashMap<String, Vec<u32>> = HashMap::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        for (i, rs) in unit_runs.iter().enumerate() {
            for r in rs {
                *df.entry(r.clone()).or_default() += 1;
                index.entry(r.clone()).or_default().push(i as u32);
            }
        }
        let n = units.len() as f32;
        let idf = |r: &str| (n / *df.get(r).unwrap_or(&1) as f32).ln();
        let norm = unit_runs.iter().map(|rs| rs.iter().map(|r| idf(r).powi(2)).sum::<f32>().sqrt().max(1e-6)).collect();
        // what follows each sentence: the rest of its turn if the person went on, else the next person's turn
        let follow: Vec<Option<String>> = (0..units.len())
            .map(|u| {
                let (_, _, ti, last) = &units[u];
                if *last {
                    turns[*ti].2.map(|nx| tidy(&turns[nx].1))
                } else {
                    let rest: Vec<&str> = units[u + 1..].iter().take_while(|x| x.2 == *ti).map(|x| x.1.as_str()).collect();
                    Some(tidy(&rest.join(" ")))
                }
            })
            .collect();
        let follow_words: Vec<String> = follow.iter().map(|f| f.as_deref().map(|s| tokens(s).join(" ")).unwrap_or_default()).collect();
        // meanings: one pass over every sentence; each word (met at least three times) gathers its neighbours' atoms,
        // each weighed by how rare the neighbour is
        let atoms = crate::lexicon::Atoms::new(MEANING_D, 7);
        let mut meaning: HashMap<String, crate::vector::HVec> = HashMap::new();
        for (w, _, _, _) in &units {
            for (i, x) in w.iter().enumerate() {
                if df.get(x).map_or(true, |c| *c < 3) {
                    continue;
                }
                let m = meaning.entry(x.clone()).or_insert_with(|| crate::vector::HVec::zero(MEANING_D));
                for j in i.saturating_sub(MEANING_WINDOW)..(i + MEANING_WINDOW + 1).min(w.len()) {
                    if j != i {
                        m.add_scaled(&atoms.word(&w[j]), idf(&w[j]));
                    }
                }
            }
        }
        let mut mean = crate::vector::HVec::zero(MEANING_D);
        for v in meaning.values_mut() {
            *v = crate::memory::normalised(v.clone());
            mean.add(v);
        }
        let k = 1.0 / meaning.len().max(1) as f32;
        for v in meaning.values_mut() {
            v.add_scaled(&mean, -k);
            *v = crate::memory::normalised(v.clone());
        }
        Some(Dialog { marks, act_after, turns, prev, openers, yes_no, ask_back, units, unit_prev, unit_runs, follow, follow_words, index, df, norm, recent: Vec::new(), scratch: Default::default(), meaning, resonance: RESONANCE, nearest_words: Default::default(), widen: true })
    }

    fn idf(&self, r: &str) -> f32 {
        (self.units.len() as f32 / *self.df.get(r).unwrap_or(&1) as f32).ln()
    }

    /// what follows a sentence: the rest of its turn if the person went on, else the next person's turn
    fn follows(&self, u: usize) -> Option<String> {
        self.follow[u].clone()
    }

    fn has_follow(&self, u: usize) -> bool {
        self.follow[u].is_some()
    }

    /// would answering by this sentence only say back what was just said?
    fn echoes(&self, u: usize, said: &str) -> bool {
        let f = &self.follow_words[u];
        !f.is_empty() && said.contains(&format!(" {f} "))
    }

    /// how alike a sentence's runs (known ones, with their norm) are to kept turn i (cosine)
    fn cos_to(&self, q: &[String], qn: f32, i: usize) -> f32 {
        let tr = &self.unit_runs[i];
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
    /// a line's state in meaning: its words' meaning waves superposed, each weighed by how rare the word is
    pub fn meaning_state(&self, words: &[String]) -> crate::vector::HVec {
        let mut s = crate::vector::HVec::zero(MEANING_D);
        for w in words {
            if let Some(m) = self.meaning.get(w) {
                s.add_scaled(m, self.idf(w));
            }
        }
        s
    }

    /// the words whose meanings are nearest to a word's (for looking at what was learned)
    pub fn near(&self, w: &str, k: usize) -> Vec<(String, f32)> {
        let Some(v) = self.meaning.get(w) else { return Vec::new() };
        let mut s: Vec<(String, f32)> = self.meaning.iter().filter(|(x, _)| x.as_str() != w).map(|(x, m)| (x.clone(), v.sim(m))).collect();
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        s.truncate(k);
        s
    }

    /// has a word been met in people's conversations?
    pub fn knows(&self, w: &str) -> bool {
        self.df.contains_key(w)
    }

    /// how alike two lines are: half their shared runs weighed by rarity (a cosine), half their resonance in meaning
    pub fn alike(&self, a: &[String], b: &[String]) -> f32 {
        let (qa, na) = self.known_runs(a);
        let (qb, nb) = self.known_runs(b);
        let shared: f32 = qa.iter().filter(|r| qb.contains(r)).map(|r| self.idf(r).powi(2)).sum();
        let cos = if na * nb > 0.0 { shared / (na * nb) } else { 0.0 };
        let m = self.meaning_state(a).sim(&self.meaning_state(b)).max(0.0);
        0.5 * cos + 0.5 * m
    }

    /// the act people most often answer this act with
    pub fn answer_act(&self, act: usize) -> usize {
        (0..4).max_by_key(|&i| self.act_after[act][i]).unwrap_or(0)
    }

    /// how well a kept sentence fits what understanding decided: its line was of the same act, and of the same
    /// emotion, and what followed it is of the act Darma decided to answer with (each 1)
    fn fits(&self, u: usize, g: &Guide) -> f32 {
        let (_, _, ti, last) = &self.units[u];
        let Some((a, e)) = self.marks[*ti] else { return 0.0 };
        let next_act = if *last { self.turns[*ti].2.and_then(|n| self.marks[n]).map(|m| m.0) } else { Some(a) };
        (a == g.act) as u8 as f32 + (e == g.emotion) as u8 as f32 + (next_act == Some(g.answer_act)) as u8 as f32
    }

    pub fn nearest(&self, words: &[String], context: Option<&[String]>, skip: &[usize]) -> Option<(usize, f32)> {
        self.nearest_guided(words, context, skip, None)
    }

    pub fn nearest_guided(&self, words: &[String], context: Option<&[String]>, skip: &[usize], guide: Option<&Guide>) -> Option<(usize, f32)> {
        let (q, qn) = self.known_runs(words);
        let said = format!(" {} ", words.join(" "));
        // the scores, summed in one buffer the size of memory (touched places remembered, cleared after)
        let mut buf = self.scratch.borrow_mut();
        if buf.len() != self.units.len() {
            *buf = vec![0.0; self.units.len()];
        }
        let mut touched: Vec<u32> = Vec::new();
        for r in &q {
            let Some(ids) = self.index.get(r) else { continue };
            let w = self.idf(r).powi(2);
            // a run every tenth turn has says little and costs much: left out of the sum
            if ids.len() > self.units.len() / 10 {
                continue;
            }
            for &i in ids {
                if buf[i as usize] == 0.0 {
                    touched.push(i);
                }
                buf[i as usize] += w;
            }
        }
        // the words nearest in meaning bring in their turns too (weighed by how near): "exhausted" finds "tired"
        if self.widen {
            for w in words {
                let near = {
                    let mut cache = self.nearest_words.borrow_mut();
                    cache.entry(w.clone()).or_insert_with(|| self.near(w, WIDEN_N).into_iter().filter(|x| x.1 >= WIDEN_AT).collect()).clone()
                };
                for (x, s) in near {
                    let Some(ids) = self.index.get(&x) else { continue };
                    if ids.len() > self.units.len() / 10 {
                        continue;
                    }
                    let wgt = s * self.idf(&x).powi(2);
                    for &i in ids {
                        if buf[i as usize] == 0.0 {
                            touched.push(i);
                        }
                        buf[i as usize] += wgt;
                    }
                }
            }
        }
        let score: Vec<(u32, f32)> = touched.iter().map(|&i| (i, std::mem::take(&mut buf[i as usize]))).collect();
        drop(buf);
        let mut cands: Vec<(usize, f32)> = score
            .into_iter()
            .map(|(i, s)| (i as usize, s / (qn * self.norm[i as usize])))
            // not an answer that only says back what was just said ("...Don't you like watching it?" answered with
            // "Don't you like watching it?")
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
            Some(tw) => cands.iter().filter(|(i, _)| self.units[*i].0.contains(tw) && !self.echoes(*i, &said)).take(300).cloned().collect(),
            None => Vec::new(),
        };
        // not an answer that only says back what was just said (checked on the best few, not on all)
        cands.truncate(400);
        cands.retain(|(i, _)| !self.echoes(*i, &said));
        cands.truncate(300);
        let ctx = context.map(|c| self.known_runs(c));
        // the line's state in meaning, to resonate with each candidate's
        let qm = self.meaning_state(words);
        let res = self.resonance;
        let total = |&(i, s): &(usize, f32)| -> f32 {
            let c = match (&ctx, self.unit_prev[i]) {
                (Some((cq, cn)), Some(p)) => CONTEXT * self.cos_to(cq, *cn, p),
                _ => 0.0,
            };
            let g = guide.map_or(0.0, |g| g.weight * self.fits(i, g));
            let m = if res > 0.0 { res * qm.sim(&self.meaning_state(&self.units[i].0)).max(0.0) } else { 0.0 };
            s + c + g + m
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
                        .then(self.follow[b.0].as_ref().map_or(0, |s| s.len()).cmp(&self.follow[a.0].as_ref().map_or(0, |s| s.len())))
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
    pub fn reply(&mut self, words: &[String], context: Option<&[String]>, guide: Option<&Guide>) -> Option<(String, String, f32)> {
        let recent = self.recent.clone();
        let (i, s) = self.nearest_guided(words, context, &recent, guide)?;
        let ans = self.follows(i)?;
        self.recent.push(i);
        if self.recent.len() > 8 {
            self.recent.remove(0);
        }
        Some((ans, tidy(&self.units[i].1), s))
    }

    /// the best answer that passes `test` (None: passes; Some(reason): let go), trying up to ten turns in order of
    /// likeness; returns (answer, the turn it matched, likeness, what was let go and why)
    pub fn reply_where(&mut self, words: &[String], context: Option<&[String]>, guide: Option<&Guide>, test: &dyn Fn(&str) -> Option<String>) -> Option<(String, String, f32, Vec<String>)> {
        let mut skip = self.recent.clone();
        let mut let_go = Vec::new();
        for _ in 0..10 {
            let (i, s) = self.nearest_guided(words, context, &skip, guide)?;
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

/// how much understanding weighs beside likeness (chosen once on DailyDialog's validation split, 2026-10-31: 0.03;
/// 0 to 0.2 all gave about the same)
pub const GUIDE: f32 = 0.03;

/// `darma dialog-eval [dialogues]`: the first dialogues of DailyDialog's test split (150 unless told), one pass. For
/// each turn, Darma's answer against the person's real answer (shared words, F1), against a random kept answer; turns
/// repeated from training (matched at 0.9 or more) are left out.
pub fn eval(limit: usize) {
    let t0 = std::time::Instant::now();
    let at = crate::lexicon::Atoms::new(1024, 2026);
    let Some((u, loaded)) = crate::understand::strike_or_load(&at) else { return println!("download DailyDialog first") };
    let Some(mut d) = Dialog::load("data/raw/dailydialog/train_marked.json") else { return println!("download DailyDialog first") };
    println!("ready in {:.1} s (understanding {}, the dialogue memory built)", t0.elapsed().as_secs_f64(), if loaded { "loaded" } else { "struck and saved" });
    if std::env::var("DARMA_CHOOSE").is_ok() {
        for (w, widen) in [(0.0f32, false), (0.6, false), (0.6, true), (1.0, true)] {
            d.resonance = w;
            d.widen = widen;
            print!("  widened by meaning {widen}: ");
            let f = eval_once(&mut d, &u, "data/raw/dailydialog/validation_marked.json", GUIDE, false, limit);
            println!("  validation, resonance in meaning weighs {w}: F1 {f:.3}");
        }
        return;
    }
    let t1 = std::time::Instant::now();
    eval_once(&mut d, &u, "data/raw/dailydialog/test_marked.json", GUIDE, true, limit);
    println!("asked in {:.1} s", t1.elapsed().as_secs_f64());
}

fn eval_once(d: &mut Dialog, u: &crate::understand::Understanding, path: &str, weight: f32, show: bool, limit: usize) -> f32 {
    let mut test: Vec<MarkedDialog> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    test.truncate(limit);
    let f1 = |a: &[String], b: &[String]| -> f32 {
        let common = a.iter().filter(|x| b.contains(x)).count() as f32;
        if common == 0.0 { 0.0 } else { let p = common / a.len() as f32; let r = common / b.len() as f32; 2.0 * p * r / (p + r) }
    };
    let (mut n, mut dup, mut nf, mut nfr) = (0usize, 0usize, 0f32, 0f32);
    let mut k = 7usize;
    let mut shown = 0;
    for dlg in &test {
        for j in 0..dlg.dialog.len().saturating_sub(1) {
            let (said, real) = (tokens(&dlg.dialog[j]), tokens(&dlg.dialog[j + 1]));
            if said.is_empty() || real.is_empty() {
                continue;
            }
            let before = (j > 0).then(|| tokens(&dlg.dialog[j - 1]));
            let r = u.read(&said);
            let a = crate::understand::ACTS.iter().position(|x| *x == r.act).unwrap();
            let e = crate::understand::EMOTIONS.iter().position(|x| *x == r.emotion).unwrap();
            let g = Guide { act: a, emotion: e, answer_act: d.answer_act(a), weight };
            let Some((ans, matched, s)) = d.reply(&said, before.as_deref(), Some(&g)) else { continue };
            d.recent.clear();
            n += 1;
            k = (k * 1103515245 + 12345) % 2147483648;
            if s >= 0.9 {
                dup += 1;
                continue;
            }
            nf += f1(&tokens(&ans), &real);
            nfr += f1(&d.turns[k % d.len()].0, &real);
            if show && shown < 3 && n % 300 == 1 {
                shown += 1;
                println!("  said: {} [{} / {}]\n  matched: {matched} ({s:.2})\n  Darma: {ans}\n  person: {}\n", tidy(&dlg.dialog[j]), r.act, r.emotion, tidy(&dlg.dialog[j + 1]));
            }
        }
    }
    let m = (n - dup).max(1) as f32;
    if show {
        println!("{n} turns answered; {dup} matched a training turn at 0.9 or more (repeated, left out); on the other {}: Darma F1 {:.3}, random {:.3}", n - dup, nf / m, nfr / m);
    }
    nf / m
}
