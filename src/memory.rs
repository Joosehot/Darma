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

#[derive(Deserialize, Clone)]
pub struct Tagged {
    pub words: Vec<String>,
    pub roles: Vec<String>,
}

/// the four things a word can be in a clause: agent, action, patient, or none of them
pub const ROLE_TAGS: [&str; 4] = ["A", "V", "P", "-"];
pub const SYNTAX_WINDOW: usize = 2;
pub const DIST_MAX: usize = 8;

pub const ORDERS: [&str; 6] = ["AVP", "APV", "VAP", "VPA", "PAV", "PVA"];

pub struct Memory {
    pub structures: Vec<(HVec, [String; 3])>,
    pub roles: [HVec; 3],
    pub order: HVec,
    pub contexts: HashMap<String, HVec>,
    /// every word's own memory of the roles it played: one bundle of role atoms per word
    pub word_roles: HashMap<String, HVec>,
    /// the neighbourhood each role has (the words around it), for words never seen
    pub role_ctx: [HVec; 4],
    /// how far apart agent and action, action and patient were
    pub dist: [HVec; 2],
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
    pub fn train(at: &Atoms, clauses: &[Clause], tagged: &[Tagged], senses: &[SenseCase], window: usize, q: f32) -> Memory {
        let d = at.d;
        let mut m = Memory {
            structures: Vec::new(),
            roles: [HVec::zero(d), HVec::zero(d), HVec::zero(d)],
            order: HVec::zero(d),
            contexts: HashMap::new(),
            word_roles: HashMap::new(),
            role_ctx: [HVec::zero(d), HVec::zero(d), HVec::zero(d), HVec::zero(d)],
            dist: [HVec::zero(d), HVec::zero(d)],
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
            }
            let t = tokens(&c.sentence);
            if let (Some(a), Some(v), Some(p)) = (t.iter().position(|x| *x == g[0]), t.iter().position(|x| *x == g[1]), t.iter().position(|x| *x == g[2])) {
                m.order.add(&at.get(order_of([a, v, p])));
                m.dist[0].add(&at.get(&format!("dAV:{}", a.abs_diff(v).min(DIST_MAX))));
                m.dist[1].add(&at.get(&format!("dVP:{}", v.abs_diff(p).min(DIST_MAX))));
            }
        }
        // every marked word of every training sentence strikes its role into the word's own memory, and its
        // neighbourhood into the role's
        for s in tagged {
            for (i, (w, r)) in s.words.iter().zip(&s.roles).enumerate() {
                let r = ROLE_TAGS.iter().position(|x| x == r).unwrap();
                m.word_roles.entry(w.clone()).or_insert_with(|| HVec::zero(d)).add(&at.get(&format!("r:{}", ROLE_TAGS[r])));
                let c = Self::near(at, &s.words, i);
                m.role_ctx[r].add(&normalised(c));
                if r < 3 {
                    *m.role_count[r].entry(w.clone()).or_default() += 1;
                }
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
        let _ = comp;
        // the membrane's threshold against chance: random readings of training sentences (three different words in
        // the three roles); a reading must fit its roles better than 1 - q of them
        {
            use rand::{Rng, SeedableRng};
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
            let mut chance = Vec::new();
            for s in tagged.iter().filter(|s| s.words.len() >= 3).take(3000) {
                let fit: Vec<[f32; 4]> = (0..s.words.len()).map(|i| m.fitness(at, &s.words, i)).collect();
                for _ in 0..5 {
                    let n = s.words.len();
                    let (a, v, p) = (rng.gen_range(0..n), rng.gen_range(0..n), rng.gen_range(0..n));
                    if a != v && v != p && a != p {
                        chance.push(fit[a][0].min(fit[v][1]).min(fit[p][2]));
                    }
                }
            }
            m.membrane = quantile(chance, 1.0 - q);
        }
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

    /// the plain neighbourhood of position i (SYNTAX_WINDOW words each side, each at its side and distance)
    pub fn near(at: &Atoms, words: &[String], i: usize) -> HVec {
        let mut c = HVec::zero(at.d);
        for k in 1..=SYNTAX_WINDOW {
            if i >= k {
                c.add(&at.get(&format!("pos:-{k}")).bind(&at.word(&words[i - k])));
            }
            if i + k < words.len() {
                c.add(&at.get(&format!("pos:+{k}")).bind(&at.word(&words[i + k])));
            }
        }
        c
    }

    /// how fit the word at position i is for each role (A, V, P, none): its own memory read by interference with the
    /// role atoms (how often it played each), with its neighbourhood's likeness to each role's as one more vote
    pub fn fitness(&self, at: &Atoms, words: &[String], i: usize) -> [f32; 4] {
        let c = Self::near(at, words, i);
        let mut pc = [0f32; 4];
        for r in 0..4 {
            pc[r] = c.sim(&self.role_ctx[r]).max(0.0);
        }
        let s: f32 = pc.iter().sum::<f32>().max(1e-9);
        let mut counts = [0f32; 4];
        if let Some(v) = self.word_roles.get(&words[i]) {
            for r in 0..4 {
                let a = at.get(&format!("r:{}", ROLE_TAGS[r]));
                counts[r] = (v.0.iter().zip(&a.0).map(|(x, y)| (x * y.conj()).re).sum::<f32>() / at.d as f32).max(0.0);
            }
        }
        let n: f32 = counts.iter().sum();
        [0, 1, 2, 3].map(|r| (counts[r] + pc[r] / s) / (n + 1.0))
    }

    /// the share of training clauses with this distance (read from the bundle)
    pub fn dist_share(&self, at: &Atoms, which: usize, k: usize) -> f32 {
        let name = format!("{}:{}", ["dAV", "dVP"][which], k.min(DIST_MAX));
        let v = at.get(&name);
        let dot: f32 = v.0.iter().zip(&self.dist[which].0).map(|(a, b)| (a * b.conj()).re).sum::<f32>() / at.d as f32;
        (dot / self.structures.len().max(1) as f32).max(1e-3)
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
