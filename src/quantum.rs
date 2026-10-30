//! Darma's quantum-minded core (classical simulation), after the owner's four lines:
//!
//!   tila = kompleksivektori    the state is one complex vector: everything said lies in it at once
//!   sana = operaattori         a word turns the state: each component's phase is rotated by the word's phases,
//!                              after one step of the state's order (so "dog bit man" is not "man bit dog")
//!   kysymys = mittaus          a question forces a choice: each candidate for its hole gets an amplitude, and
//!                              its probability is the amplitude's square (Born's rule)
//!   vastaus = interferenssi    what memory holds that fits the question adds up in phase; what does not fit
//!                              has random phases and cancels. If nothing stands out, Darma does not understand.
//!
//! A sentence's state is the sum of the states its runs of 1, 2 and 3 words make (each run: the words' operators
//! applied in turn to the vacuum). Memory is the sum of the states of everything told (one addition each: the
//! lightning). A word's operator is its atom's phases, optionally mixed with its learned meaning (`beta`), so that
//! words of like meaning turn the state alike.

use crate::lexicon::{tokens, Atoms};
use crate::memory::Memory;
use crate::vector::HVec;
use num_complex::Complex32;
use std::collections::HashMap;

pub const RUN: usize = 3;
pub const HOLES: [&str; 2] = ["who", "what"];

pub struct Core<'a> {
    pub at: &'a Atoms,
    pub meaning: Option<&'a Memory>,
    pub beta: f32,
    vacuum: HVec,
    ops: std::cell::RefCell<HashMap<String, HVec>>,
    /// everything told, superposed
    pub psi: HVec,
    /// the key register: everything told, entangled with each of its words (sum over w of |w> (x) psi_w); measuring
    /// the register on a word of the question collapses memory to what was told with that word
    pub reg: HashMap<String, (HVec, usize, usize)>,
    /// the basis each register can be measured in: what was told together with the word (single words and runs of
    /// two), with how many times
    pub with: HashMap<String, HashMap<String, i64>>,
    pub told: usize,
    /// how far a measurement's winner must stand out of the runner-up to be an answer (learned from untold questions)
    pub threshold: f32,
    /// how many noise deviations the winner's own fidelity must stand above zero in its register (learned the same
    /// way; a register holding m runs has noise of about sqrt(m / 2D) on one run's fidelity)
    pub fid_threshold: f32,
}

impl<'a> Core<'a> {
    pub fn new(at: &'a Atoms, meaning: Option<&'a Memory>, beta: f32) -> Self {
        Core { at, meaning, beta, vacuum: at.get("vacuum"), ops: Default::default(), psi: HVec::zero(at.d), reg: HashMap::new(), with: HashMap::new(), told: 0, threshold: 0.0, fid_threshold: 0.0 }
    }

    /// a word as an operator: unit phases (its atom's, turned towards its meaning by beta)
    pub fn op(&self, w: &str) -> HVec {
        if let Some(o) = self.ops.borrow().get(w) {
            return o.clone();
        }
        let mut v = self.at.word(w);
        if let (Some(m), true) = (self.meaning, self.beta > 0.0) {
            if let Some(mv) = m.meaning.get(w) {
                v.add_scaled(mv, self.beta * (self.at.d as f32).sqrt());
            }
        }
        let o = v.phase();
        self.ops.borrow_mut().insert(w.to_string(), o.clone());
        o
    }

    /// the state a run of words makes: each word turns the state after one step of order
    fn run(&self, words: &[String]) -> HVec {
        let mut s = self.vacuum.clone();
        for w in words {
            s = s.shift().bind(&self.op(w));
        }
        s
    }

    /// the state of a sentence: every run of 1..RUN words; with `only`, just the runs of two or more words that
    /// contain that position (a word alone says nothing of how it stands to the others, only how often it was said)
    pub fn state(&self, words: &[String], only: Option<usize>) -> HVec {
        let mut s = HVec::zero(self.at.d);
        for n in (if only.is_some() { 2 } else { 1 })..=RUN {
            for i in 0..words.len().saturating_sub(n - 1) {
                if only.map_or(true, |k| k >= i && k < i + n) {
                    s.add(&self.run(&words[i..i + n]));
                }
            }
        }
        s
    }

    /// telling: the sentence's state is added to memory
    pub fn tell(&mut self, words: &[String]) {
        self.told_as(words, 1.0);
        self.told += 1;
    }

    pub fn forget(&mut self, words: &[String]) {
        self.told_as(words, -1.0);
        self.told -= 1;
    }

    fn told_as(&mut self, words: &[String], sign: f32) {
        let s = self.state(words, None);
        let nruns: usize = (1..=RUN).map(|n| words.len().saturating_sub(n - 1)).sum();
        self.psi.add_scaled(&s, sign);
        let mut u: Vec<&String> = words.iter().collect();
        u.sort();
        u.dedup();
        let d = self.at.d;
        let mut fills: Vec<String> = words.to_vec();
        fills.extend(words.windows(2).map(|p| p.join(" ")));
        for w in u {
            let e = self.reg.entry(w.clone()).or_insert_with(|| (HVec::zero(d), 0, 0));
            e.0.add_scaled(&s, sign);
            e.1 = (e.1 as i64 + sign as i64) as usize;
            e.2 = (e.2 as i64 + sign as i64 * nruns as i64) as usize;
            let b = self.with.entry(w.clone()).or_default();
            for f in &fills {
                let c = b.entry(f.clone()).or_default();
                *c += sign as i64;
                if *c <= 0 {
                    b.remove(f);
                }
            }
        }
    }

    /// the candidates a question can be measured on: what was told together with every word of the question
    pub fn candidates_for(&self, words: &[String]) -> Vec<String> {
        let keys: Vec<&String> = words.iter().filter(|w| !HOLES.contains(&w.as_str())).collect();
        let Some(first) = keys.iter().filter_map(|w| self.with.get(*w)).min_by_key(|b| b.len()) else { return Vec::new() };
        let mut c: Vec<String> = first.keys().filter(|f| keys.iter().all(|w| self.with.get(*w).is_some_and(|b| b.contains_key(*f)))).cloned().collect();
        c.sort();
        c
    }

    /// the registers to measure in: one per word of the question. An answer was told together with every one of
    /// them, so it must show in each; a word never told means memory cannot answer (None)
    fn collapse(&self, words: &[String]) -> Option<Vec<(&HVec, usize)>> {
        words.iter().filter(|w| !HOLES.contains(&w.as_str())).map(|w| self.reg.get(w).filter(|r| r.1 > 0).map(|r| (&r.0, r.2))).collect()
    }

    /// asking: the hole (who / what) is filled by each candidate in turn, and memory is measured against the runs of
    /// the question that hold it. Two readings of one measurement:
    ///   fidelity: for each run of the question that holds the hole, the in-phase part of memory's overlap with it
    ///             over a full match, at most 1; their mean. About 1 when memory holds the whole question so filled,
    ///             1/2 when it holds half of its runs, 0 when nothing fits (random phases cancel). This decides.
    ///   standing: the fidelity in noise deviations of its register (the least over the registers): how far the
    ///             candidate stands above what random phases give there.
    /// Returns (candidate, fidelity, standing), the highest fidelity first.
    pub fn measure(&self, words: &[String], candidates: &[String]) -> Vec<(String, f32, f32)> {
        let Some(h) = words.iter().position(|w| HOLES.contains(&w.as_str())) else { return Vec::new() };
        // memory knows nothing told with any word of the question: nothing to measure
        let Some(regs) = self.collapse(words) else { return Vec::new() };
        let mut out: Vec<(String, f32, f32)> = Vec::with_capacity(candidates.len());
        for c in candidates {
            // a candidate may be one word or a run of words told together ("the cat"): it fills the hole whole
            let fill: Vec<String> = c.split(' ').map(String::from).collect();
            if fill.iter().all(|x| words.contains(x)) {
                continue;
            }
            let mut q: Vec<String> = words[..h].to_vec();
            q.extend(fill.iter().cloned());
            q.extend(words[h + 1..].iter().cloned());
            let (lo, hi) = (h, h + fill.len());
            // each run that holds the hole is asked on its own whether memory holds it (in phase, at most once:
            // being told a thing five times is not more than knowing it); fidelity is their mean
            let mut runs: Vec<HVec> = Vec::new();
            for n in 2..=RUN {
                for i in 0..q.len().saturating_sub(n - 1) {
                    // a run that holds some of the fill and some of the question (not the fill alone)
                    let holds = i < hi && i + n > lo;
                    let beyond = i < lo || i + n > hi;
                    if holds && beyond {
                        runs.push(self.run(&q[i..i + n]));
                    }
                }
            }
            // in each register: the mean over runs of the in-phase overlap (at most 1); the answer's fidelity is the
            // least over the registers
            let (mut fid, mut z, mut amp) = (f32::MAX, f32::MAX, Complex32::new(0.0, 0.0));
            for (psi, m) in &regs {
                let mut f = 0.0f32;
                for r in &runs {
                    let a = psi.inner(r);
                    f += (a.re / self.at.d as f32).clamp(0.0, 1.0);
                    amp += a;
                }
                let f = f / runs.len().max(1) as f32;
                // the noise this register gives the mean fidelity over the runs
                let sigma = ((*m as f32) / (2.0 * self.at.d as f32) / runs.len().max(1) as f32).sqrt().max(1e-6);
                fid = fid.min(f);
                z = z.min(f / sigma);
            }
            out.push((c.clone(), fid, z));
        }
        let _ = amp_unused();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }

    /// how far the measurement's winner stands out: its fidelity over the runner-up's (the other candidates show
    /// what noise looks like in this very measurement)
    pub fn peak(m: &[(String, f32, f32)]) -> f32 {
        let Some(a) = m.first() else { return 0.0 };
        // the runner-up is the best candidate that shares no word with the winner ("dog" and "the dog" are one answer)
        let aw: Vec<&str> = a.0.split(' ').collect();
        let b = m.iter().skip(1).find(|x| !x.0.split(' ').any(|w| aw.contains(&w))).map(|x| x.1).unwrap_or(0.0);
        a.1 - b
    }

    /// an answer, or None: "en ymmärrä" (the winner does not stand out of the rest, or is itself no more than noise)
    pub fn answer(&self, words: &[String], candidates: &[String]) -> (Option<String>, Vec<(String, f32, f32)>) {
        let m = self.measure(words, candidates);
        let a = (Self::peak(&m) >= self.threshold && m[0].2 >= self.fid_threshold).then(|| m[0].0.clone());
        (a, m)
    }
}

fn amp_unused() {}

/// the questions a clause (agent, action, patient) can answer: (question words, the answer)
pub fn questions(g: &[String; 3]) -> [(Vec<String>, String); 2] {
    [
        (vec!["who".into(), g[1].clone(), g[2].clone()], g[0].clone()),
        (vec![g[0].clone(), g[1].clone(), "what".into()], g[2].clone()),
    ]
}

pub fn words_of(s: &str) -> Vec<String> {
    tokens(s)
}
