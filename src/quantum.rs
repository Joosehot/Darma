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
        let all: Vec<(&HVec, usize)> = words.iter().filter(|w| !HOLES.contains(&w.as_str())).map(|w| self.reg.get(w).filter(|r| r.1 > 0).map(|r| (&r.0, r.2))).collect::<Option<_>>()?;
        // a register can tell a full match (fidelity 1) from noise only if its noise on one run, sqrt(m / 2D), is at
        // most a third of it (three deviations); a register holding too much is not measured in. If none can, the
        // one holding least is.
        let limit = 2.0 * self.at.d as f32 / 9.0;
        let sharp: Vec<(&HVec, usize)> = all.iter().filter(|r| r.1 as f32 <= limit).cloned().collect();
        if sharp.is_empty() {
            all.into_iter().min_by_key(|r| r.1).map(|r| vec![r])
        } else {
            Some(sharp)
        }
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

    /// a yes/no question ("the dog chased the cat?"): the whole statement is measured. Every run of two or more of
    /// its words is asked in every register of its words; returns None when some word was never told (memory cannot
    /// say), else (fidelity, standing in noise deviations): the least over the registers and over the runs
    pub fn verify(&self, words: &[String]) -> Option<(f32, f32)> {
        let regs = self.collapse(words)?;
        let mut runs: Vec<HVec> = Vec::new();
        for n in 2..=RUN {
            for i in 0..words.len().saturating_sub(n - 1) {
                runs.push(self.run(&words[i..i + n]));
            }
        }
        if runs.is_empty() {
            return None;
        }
        let (mut fid, mut z) = (f32::MAX, f32::MAX);
        for (psi, m) in &regs {
            // a yes needs the whole statement: every run of it held (the least over the runs, not their mean: "the
            // cat" and "the dog" told apart do not make "the cat chased the dog")
            let f = runs.iter().map(|r| (psi.inner(r).re / self.at.d as f32).clamp(0.0, 1.0)).fold(f32::MAX, f32::min);
            let sigma = ((*m as f32) / (2.0 * self.at.d as f32)).sqrt().max(1e-6);
            fid = fid.min(f);
            z = z.min(f / sigma);
        }
        Some((fid, z))
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
        let a = (Self::peak(&m) >= self.threshold && m[0].2 >= self.fid_threshold).then(|| {
            // of overlapping answers held about as well ("the" and "the fish"), the longer says more: it wins if it
            // holds at least 90% of the winner's fidelity (a share set by hand)
            let w = &m[0];
            let ww: Vec<&str> = w.0.split(' ').collect();
            m.iter()
                .filter(|x| x.1 >= 0.9 * w.1 && x.0.split(' ').any(|p| ww.contains(&p)))
                .max_by_key(|x| x.0.split(' ').count())
                .map(|x| x.0.clone())
                .unwrap_or_else(|| w.0.clone())
        });
        (a, m)
    }
}

fn amp_unused() {}

impl<'a> Core<'a> {
    /// asking with one step of reasoning. If the question has no answer as it stands, each of its words is asked
    /// what it is ("rex is what", through every copula people marked in UD) and, for each word of that answer, the
    /// question is measured again with it in the word's place ("dog chased what"). Returns the answer and the facts
    /// it stood on.
    pub fn answer_reasoned(&self, words: &[String], copulas: &[String]) -> (Option<String>, Vec<String>) {
        if let (Some(a), _) = self.answer(words, &self.candidates_for(words)) {
            return (Some(a), Vec::new());
        }
        for (i, w) in words.iter().enumerate() {
            if HOLES.contains(&w.as_str()) || copulas.contains(w) {
                continue;
            }
            for c in copulas {
                let q = vec![w.clone(), c.clone(), "what".to_string()];
                let (Some(is), _) = self.answer(&q, &self.candidates_for(&q)) else { continue };
                for x in is.split(' ') {
                    if x == w {
                        continue;
                    }
                    let mut q2 = words.to_vec();
                    q2[i] = x.to_string();
                    if let (Some(a), _) = self.answer(&q2, &self.candidates_for(&q2)) {
                        return (Some(a), vec![format!("{w} {c} {is}")]);
                    }
                }
            }
        }
        (None, Vec::new())
    }

    /// a yes/no question with the same one step of reasoning: (said yes, the fact it stood on if any)
    pub fn verify_reasoned(&self, words: &[String], copulas: &[String], z_at: f32) -> Option<(bool, Vec<String>)> {
        let direct = self.verify(words);
        if direct.is_some_and(|x| x.1 >= z_at) {
            return Some((true, Vec::new()));
        }
        for (i, w) in words.iter().enumerate() {
            if copulas.contains(w) {
                continue;
            }
            for c in copulas {
                let q = vec![w.clone(), c.clone(), "what".to_string()];
                let (Some(is), _) = self.answer(&q, &self.candidates_for(&q)) else { continue };
                for x in is.split(' ').filter(|x| *x != w) {
                    let mut q2 = words.to_vec();
                    q2[i] = x.to_string();
                    if self.verify(&q2).is_some_and(|y| y.1 >= z_at) {
                        return Some((true, vec![format!("{w} {c} {is}")]));
                    }
                }
            }
        }
        direct.map(|_| (false, Vec::new()))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    fn core(at: &Atoms) -> Core<'_> {
        let mut c = Core::new(at, None, 0.0);
        for s in ["the dog chased the cat", "the cat ate the fish", "rex bit the man", "rex is a dog"] {
            c.tell(&w(s));
        }
        c.threshold = 0.5;
        c.fid_threshold = 3.0;
        c
    }

    #[test]
    fn a_told_thing_is_answered() {
        let at = Atoms::new(1024, 1);
        let c = core(&at);
        let q = w("who chased the cat");
        assert_eq!(c.answer(&q, &c.candidates_for(&q)).0.as_deref(), Some("the dog"));
        let q = w("the cat ate what");
        assert_eq!(c.answer(&q, &c.candidates_for(&q)).0.as_deref(), Some("the fish"));
    }

    #[test]
    fn the_wrong_way_round_is_not() {
        let at = Atoms::new(1024, 1);
        let c = core(&at);
        for q in ["who chased the dog", "who ate the cat", "the fish ate what"] {
            let q = w(q);
            assert_eq!(c.answer(&q, &c.candidates_for(&q)).0, None, "{q:?}");
        }
    }

    #[test]
    fn order_makes_a_different_state() {
        let at = Atoms::new(1024, 1);
        let c = Core::new(&at, None, 0.0);
        let a = c.state(&w("dog bit man"), None);
        let b = c.state(&w("man bit dog"), None);
        let s = a.sim(&b);
        // the whole states share their three single words (3 of 6 runs: about 0.5); the runs that say how the words
        // stand to each other share nothing
        let r = c.run(&w("dog bit man")).sim(&c.run(&w("man bit dog")));
        let r2 = c.run(&w("dog bit")).sim(&c.run(&w("man bit")));
        println!("'dog bit man' vs 'man bit dog': whole states {s:.3}, the three-word runs {r:.3}, 'dog bit' vs 'man bit' {r2:.3}");
        assert!(s < 0.6 && r.abs() < 0.1 && r2.abs() < 0.1);
    }

    #[test]
    fn yes_no_needs_the_whole_statement() {
        let at = Atoms::new(1024, 1);
        let c = core(&at);
        assert!(c.verify(&w("the dog chased the cat")).unwrap().1 > 3.0);
        assert!(c.verify(&w("the cat chased the dog")).unwrap().1 < 3.0);
        assert!(c.verify(&w("the bird ate the worm")).is_none());
    }

    #[test]
    fn one_step_of_reasoning() {
        let at = Atoms::new(1024, 1);
        let c = core(&at);
        let cop = vec!["is".to_string()];
        let (a, why) = c.answer_reasoned(&w("rex chased what"), &cop);
        assert_eq!(a.as_deref(), Some("the cat"));
        assert_eq!(why, vec!["rex is a dog".to_string()]);
    }

    #[test]
    fn forgetting_takes_it_back() {
        let at = Atoms::new(1024, 1);
        let mut c = core(&at);
        c.forget(&w("the cat ate the fish"));
        let q = w("the cat ate what");
        assert_eq!(c.answer(&q, &c.candidates_for(&q)).0, None);
    }
}
