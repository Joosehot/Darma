//! The core's memory and the lightning training: one pass, nothing optimised. Adding a sentence is adding vectors.
//!
//! For every training clause:
//!   - its structure bind(AGENT, a) + bind(ACTION, v) + bind(PATIENT, p) is stored (normalised);
//!   - each role's bundle gets the filler's atom (so a word's weight in a role is how often it filled it);
//!   - the order bundle gets the atom of the clause's order (which role came first, second, third in the sentence).
//! For every sense-marked training sentence (CoarseWSD-20), the sense's context vector gets its neighbours' atoms
//! (within `window` words), each weighted by how rare the neighbour is (log N/df, counted from the same sentences).
//!
//! Then the two thresholds of the egg are learned from the training clauses themselves, leaving each one out:
//! the membrane's (how like the rest of memory a known clause is) and the fertilisation's (how clearly a known
//! clause's roles read back after it is merged with what memory recalls). Each is a low quantile.

use crate::lexicon::{tokens, Atoms};
use crate::vector::HVec;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Clone)]
pub struct Clause {
    pub sentence: String,
    pub text: String,
}

impl Clause {
    /// the gold fillers in their surface forms (agent, action, patient): the three words of `text`
    pub fn gold(&self) -> [String; 3] {
        let w: Vec<String> = self.text.split(' ').map(String::from).collect();
        [w[0].clone(), w[1].clone(), w[2].clone()]
    }
}

#[derive(Deserialize, Clone)]
pub struct SenseCase {
    pub word: String,
    pub sentence: String,
    pub index: usize,
    pub sense: String,
}

pub const ORDERS: [&str; 6] = ["AVP", "APV", "VAP", "VPA", "PAV", "PVA"];

pub struct Memory {
    pub structures: Vec<(HVec, [String; 3])>,
    pub roles: [HVec; 3],
    pub order: HVec,
    pub contexts: HashMap<String, HVec>,
    pub idf: HashMap<String, f32>,
    pub window: usize,
    pub membrane: f32,
    pub fertile: f32,
    /// for the baseline only: counts, no vectors
    pub role_count: [HashMap<String, u32>; 3],
    pub sense_count: HashMap<String, HashMap<String, u32>>,
}

pub fn structure(at: &Atoms, f: &[String; 3]) -> HVec {
    let mut s = HVec::zero(at.d);
    for r in 0..3 {
        s.add(&at.role(r).bind(&at.word(&f[r])));
    }
    s
}

pub fn normalised(mut v: HVec) -> HVec {
    let n = v.norm().max(1e-9);
    for c in v.0.iter_mut() {
        *c /= n;
    }
    v
}

pub fn order_of(pos: [usize; 3]) -> &'static str {
    let mut idx = [0usize, 1, 2];
    idx.sort_by_key(|&r| pos[r]);
    let name: String = idx.iter().map(|&r| ['A', 'V', 'P'][r]).collect();
    ORDERS.iter().find(|o| **o == name).unwrap()
}

fn quantile(mut v: Vec<f32>, q: f32) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f32 - 1.0) * q) as usize]
}

impl Memory {
    pub fn train(at: &Atoms, clauses: &[Clause], senses: &[SenseCase], window: usize, q: f32) -> Memory {
        let d = at.d;
        let mut m = Memory {
            structures: Vec::new(),
            roles: [HVec::zero(d), HVec::zero(d), HVec::zero(d)],
            order: HVec::zero(d),
            contexts: HashMap::new(),
            idf: HashMap::new(),
            window,
            membrane: 0.0,
            fertile: 0.0,
            role_count: Default::default(),
            sense_count: HashMap::new(),
        };
        for c in clauses {
            let g = c.gold();
            m.structures.push((normalised(structure(at, &g)), g.clone()));
            for r in 0..3 {
                m.roles[r].add(&at.word(&g[r]));
                *m.role_count[r].entry(g[r].clone()).or_default() += 1;
            }
            let t = tokens(&c.sentence);
            if let (Some(a), Some(v), Some(p)) = (t.iter().position(|x| *x == g[0]), t.iter().position(|x| *x == g[1]), t.iter().position(|x| *x == g[2])) {
                m.order.add(&at.get(order_of([a, v, p])));
            }
        }
        // how rare each word is among the sense sentences
        let mut df: HashMap<String, u32> = HashMap::new();
        for s in senses {
            let mut t = tokens(&s.sentence);
            t.sort();
            t.dedup();
            for w in t {
                *df.entry(w).or_default() += 1;
            }
        }
        let n = senses.len().max(1) as f32;
        m.idf = df.into_iter().map(|(w, c)| (w, (n / c as f32).ln())).collect();
        for s in senses {
            let ctx = m.context(at, &s.sentence.split(' ').map(String::from).collect::<Vec<_>>(), s.index);
            m.contexts.entry(s.sense.clone()).or_insert_with(|| HVec::zero(d)).add(&ctx);
            *m.sense_count.entry(s.word.clone()).or_default().entry(s.sense.clone()).or_default() += 1;
        }
        // the thresholds, each a low quantile over the known clauses, each clause left out of its own memory
        let mut comp = Vec::new();
        let mut marg = Vec::new();
        for (i, (s, g)) in m.structures.iter().enumerate() {
            let others = m.structures.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| &x.0);
            comp.push(others.map(|o| s.sim(o)).fold(f32::MIN, f32::max));
            let words: Vec<String> = tokens(&clauses[i].sentence);
            marg.push(m.read_back(at, s, g, &words, Some(i)).1);
        }
        m.membrane = quantile(comp, q);
        m.fertile = quantile(marg, q);
        m
    }

    /// the context of position `i` in a sentence: its neighbours' atoms, weighted by rarity
    pub fn context(&self, at: &Atoms, words: &[String], i: usize) -> HVec {
        let mut c = HVec::zero(at.d);
        let lo = i.saturating_sub(self.window);
        let hi = (i + self.window + 1).min(words.len());
        for (j, w) in words.iter().enumerate().take(hi).skip(lo) {
            if j != i {
                let w = w.to_lowercase();
                c.add_scaled(&at.word(&w), *self.idf.get(&w).unwrap_or(&0.0));
            }
        }
        c
    }

    /// how often a word filled role r (read from the role bundle by interference, not from a count)
    pub fn role_weight(&self, at: &Atoms, w: &str, r: usize) -> f32 {
        let v = at.word(w);
        let dot: f32 = v.0.iter().zip(&self.roles[r].0).map(|(a, b)| (a * b.conj()).re).sum();
        (dot / at.d as f32).max(0.0)
    }

    pub fn order_weight(&self, at: &Atoms, o: &str) -> f32 {
        let v = at.get(o);
        let dot: f32 = v.0.iter().zip(&self.order.0).map(|(a, b)| (a * b.conj()).re).sum();
        (dot / at.d as f32).max(0.0)
    }

    /// what memory recalls for a structure: its five most alike stored structures, weighted by likeness
    pub fn recall(&self, s: &HVec, skip: Option<usize>) -> HVec {
        let mut sims: Vec<(f32, usize)> = self.structures.iter().enumerate().filter(|(j, _)| Some(*j) != skip).map(|(j, x)| (s.sim(&x.0), j)).collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut r = HVec::zero(s.dim());
        for &(w, j) in sims.iter().take(5) {
            r.add_scaled(&self.structures[j].0, w.max(0.0));
        }
        r
    }

    /// fertilisation's self-test: merge the structure with what memory recalls, unbind each role, measure it
    /// against the sentence's words. Returns whether every role reads back its own filler, and the smallest margin
    /// (best minus runner-up) over the three roles.
    pub fn read_back(&self, at: &Atoms, s: &HVec, f: &[String; 3], words: &[String], skip: Option<usize>) -> (bool, f32) {
        let mut state = s.clone();
        state.add(&normalised(self.recall(s, skip)));
        let mut uniq: Vec<&String> = words.iter().collect();
        uniq.sort();
        uniq.dedup();
        let atoms: Vec<(String, HVec)> = uniq.iter().map(|w| (w.to_string(), at.word(w))).collect();
        let (mut all, mut least) = (true, f32::MAX);
        for r in 0..3 {
            let q = state.unbind(&at.role(r));
            let (k, s1, s2) = crate::vector::cleanup(&q, atoms.iter().map(|(w, v)| (w.clone(), v))).unwrap();
            all &= k == f[r];
            least = least.min(if atoms.len() > 1 { s1 - s2 } else { s1 });
        }
        (all, least)
    }
}
