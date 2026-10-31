//! Words understood from a word network: Primo's frozen network (WordNet 3.1: every word a node, lines that say
//! what one word is to another: same as, is a, similar, derived, ...) imprinted into Darma as meaning waves. A word's
//! meaning is its own atom with the atoms of the words it reaches in the network superposed (a spreading of
//! activation: weaker with every step, and weaker through a word that many words are a kind of). So "sad" and
//! "unhappy" resonate though no conversation put them together, and "sad" resonates with "sadness".
use crate::vector::HVec;
use primogenitus::network::{Kind, Network, UNKNOWN};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;
use std::collections::HashMap;

pub const WORDS_D: usize = 512;
/// how far the activation spreads (steps)
pub const DEPTH: usize = 3;
/// a sense of a word counts by how common it is: the first sense 1, the second 1/2, ...; senses past this are left out
pub const SENSES: u8 = 3;
/// a line's emotion is read from the network when one emotion's name resonates with one of its words this much:
/// chosen on DailyDialog's validation split by `darma words-eval` (the test split only measured)
pub const FEEL_AT: f32 = 0.30;

pub fn network_path() -> Option<String> {
    ["data/network.bin", "../Primogenitus/motor/data/network.bin"].iter().find(|p| std::path::Path::new(p).exists()).map(|s| s.to_string())
}

pub struct Words {
    net: Network,
    seed: u64,
    lemma_cache: RefCell<HashMap<u32, HVec>>,
    word_cache: RefCell<HashMap<String, Option<HVec>>>,
}

/// how much a step along a line carries
fn carry(k: Kind) -> f32 {
    match k {
        Kind::Synonym => 0.8,
        Kind::Similar => 0.7,
        Kind::Derived | Kind::Attribute | Kind::Pertains => 0.6,
        Kind::Hypernym => 0.6,
        _ => 0.0,
    }
}

impl Words {
    pub fn load(seed: u64) -> Option<Words> {
        let net = Network::load(std::path::Path::new(&network_path()?)).ok()?;
        Some(Words { net, seed, lemma_cache: Default::default(), word_cache: Default::default() })
    }

    /// the control: the same network with every line sent to a random word (same size, same kinds, no meaning)
    pub fn shuffled(mut self) -> Words {
        let n = self.net.nodes.len() as u32;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ 0x5eed);
        for ls in self.net.lines.iter_mut() {
            for l in ls.iter_mut() {
                l.to = rng.gen_range(0..n);
                l.to_rank = UNKNOWN;
            }
        }
        self
    }

    pub fn nodes(&self) -> usize {
        self.net.nodes.len()
    }

    pub fn lines(&self) -> usize {
        self.net.line_count()
    }

    fn atom(&self, node: u32) -> HVec {
        let mut r = ChaCha8Rng::seed_from_u64(self.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ node as u64);
        HVec::random(WORDS_D, &mut r)
    }

    /// a word many words are a kind of ("entity", "feeling") says less of any one of them
    fn hub(&self, n: u32) -> f32 {
        let k = self.net.lines[n as usize].iter().filter(|l| l.kind == Kind::Hyponym).count();
        1.0 / (1.0 + (1.0 + k as f32).ln())
    }

    /// a base word's meaning: the activation spread from it through the network, each reached word's atom weighed by
    /// how much reached it
    fn lemma_meaning(&self, l: u32) -> HVec {
        if let Some(m) = self.lemma_cache.borrow().get(&l) {
            return m.clone();
        }
        let mut reached: HashMap<u32, f32> = HashMap::from([(l, 1.0)]);
        // (word, the sense it was reached in (UNKNOWN: any), weight)
        let mut front = vec![(l, UNKNOWN, 1.0f32)];
        for _ in 0..DEPTH {
            let mut next = Vec::new();
            for &(n, sense, w) in &front {
                for line in &self.net.lines[n as usize] {
                    let c = carry(line.kind);
                    if c == 0.0 || line.rank >= SENSES || (sense != UNKNOWN && line.rank != sense) {
                        continue;
                    }
                    let to = self.net.node(line.to).lemma;
                    let w2 = w * c / (1.0 + line.rank as f32) * self.hub(to);
                    if w2 < 0.02 {
                        continue;
                    }
                    let e = reached.entry(to).or_insert(0.0);
                    if w2 > *e {
                        *e = w2;
                        next.push((to, line.to_rank, w2));
                    }
                }
            }
            front = next;
        }
        let mut m = HVec::zero(WORDS_D);
        for (n, w) in reached {
            m.add_scaled(&self.atom(n), w);
        }
        self.lemma_cache.borrow_mut().insert(l, m.clone());
        m
    }

    /// a word's meaning (any form: "dogs", "went"): its base words' meanings, each by how often the word is used so
    pub fn meaning(&self, text: &str) -> Option<HVec> {
        if let Some(m) = self.word_cache.borrow().get(text) {
            return m.clone();
        }
        let mut lemmas: Vec<u32> = self.net.lookup(text).iter().map(|&i| self.net.node(i).lemma).collect();
        lemmas.sort();
        lemmas.dedup();
        let m = (!lemmas.is_empty()).then(|| {
            let mut m = HVec::zero(WORDS_D);
            for l in lemmas {
                let lm = self.lemma_meaning(l);
                let n = lm.norm().max(1e-6);
                m.add_scaled(&lm, (1.0 + (1.0 + self.net.node(l).fam as f32).ln()) / n);
            }
            m
        });
        self.word_cache.borrow_mut().insert(text.to_string(), m.clone());
        m
    }

    /// how alike two words are in meaning (1 for the same word, 0 for a word the network does not hold)
    pub fn sim(&self, a: &str, b: &str) -> f32 {
        if a == b {
            return 1.0;
        }
        match (self.meaning(a), self.meaning(b)) {
            (Some(x), Some(y)) => x.sim(&y).max(0.0),
            _ => 0.0,
        }
    }

    /// how alike two lines are: each word's best match in the other line (in meaning), weighed by the word's weight,
    /// both ways
    pub fn alike(&self, a: &[String], b: &[String], weight: &dyn Fn(&str) -> f32) -> f32 {
        let one = |x: &[String], y: &[String]| {
            let (mut s, mut n) = (0.0f32, 0.0f32);
            for w in x {
                let k = weight(w);
                s += k * y.iter().map(|v| self.sim(w, v)).fold(0.0, f32::max);
                n += k;
            }
            if n > 0.0 { s / n } else { 0.0 }
        };
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        0.5 * (one(a, b) + one(b, a))
    }

    /// the emotions a line's words resonate with: each emotion (by its name) and its strongest resonance in the line
    pub fn feelings(&self, line: &[String]) -> Vec<(&'static str, f32)> {
        crate::understand::EMOTIONS[1..].iter().map(|e| (*e, line.iter().map(|w| self.sim(w, e)).fold(0.0, f32::max))).collect()
    }

    /// the emotion read from the line's words, if one resonates at least `at` (the strongest)
    pub fn feeling(&self, line: &[String], at: f32) -> Option<&'static str> {
        self.feelings(line).into_iter().filter(|x| x.1 >= at).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|x| x.0)
    }
}

/// `darma words-test`: a fixed small test of what the network gives, against the same network shuffled (control).
/// The pairs were written before the first run and are not changed after it.
pub fn test(seed: u64) {
    let t0 = std::time::Instant::now();
    let Some(w) = Words::load(seed) else { return println!("Primo's network not found (data/network.bin)") };
    println!("network: {} words, {} lines, loaded in {:.1} s", w.nodes(), w.lines(), t0.elapsed().as_secs_f64());
    let alike = [("hi", "hello"), ("sad", "unhappy"), ("dog", "puppy"), ("car", "automobile"), ("angry", "mad"), ("sad", "sadness"), ("afraid", "fear"), ("happy", "happiness"), ("bye", "goodbye"), ("thanks", "thank")];
    let unlike = [("dog", "sky"), ("hello", "car"), ("sad", "carrot"), ("happy", "table"), ("angry", "music"), ("bye", "potato"), ("car", "fear"), ("puppy", "sadness"), ("hi", "river"), ("thanks", "dog")];
    let feel = [("i am so sad today", "sadness"), ("i am angry at my boss", "anger"), ("i am scared of the dark", "fear"), ("i am so happy for you", "happiness"), ("that is disgusting", "disgust"), ("wow what a surprise", "surprise"), ("the bus leaves at nine", "none"), ("what is your name", "none")];
    let shuffled = Words::load(seed).unwrap().shuffled();
    for (name, w) in [("the network", &w), ("shuffled (control)", &shuffled)] {
        let a: Vec<f32> = alike.iter().map(|(x, y)| w.sim(x, y)).collect();
        let u: Vec<f32> = unlike.iter().map(|(x, y)| w.sim(x, y)).collect();
        let umax = u.iter().cloned().fold(0.0, f32::max);
        let sep = a.iter().filter(|s| **s > umax).count();
        println!("\n{name}: alike pairs above every unlike pair: {sep} of {}", a.len());
        println!("  alike:  {}", alike.iter().zip(&a).map(|((x, y), s)| format!("{x}~{y} {s:.2}")).collect::<Vec<_>>().join(", "));
        println!("  unlike: {}", unlike.iter().zip(&u).map(|((x, y), s)| format!("{x}~{y} {s:.2}")).collect::<Vec<_>>().join(", "));
        let mut right = 0;
        for (line, want) in &feel {
            let got = w.feeling(&crate::quantum::words_of(line), FEEL_AT).unwrap_or("none");
            right += (got == *want) as usize;
            println!("  feeling of \"{line}\": {got} (want {want})");
        }
        println!("  feelings right: {right} of {}", feel.len());
    }
}

/// `darma words-eval`: the emotion read from the network on DailyDialog: the threshold chosen on the validation split,
/// then measured once on the test split, alone and with the learned understanding (the network speaks only where the
/// understanding read no emotion)
pub fn eval(dim: usize, seed: u64) {
    let Some(w) = Words::load(seed) else { return println!("Primo's network not found") };
    #[derive(serde::Deserialize)]
    struct Marked {
        dialog: Vec<String>,
        emotion: Vec<i64>,
    }
    let load = |p: &str| -> Vec<(Vec<String>, usize)> {
        let d: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        d.iter().flat_map(|m| m.dialog.iter().zip(&m.emotion).map(|(l, e)| (crate::quantum::words_of(l), (*e as usize).min(6))).collect::<Vec<_>>()).filter(|x| !x.0.is_empty()).collect()
    };
    let names = &crate::understand::EMOTIONS;
    let idx = |e: &str| names.iter().position(|x| *x == e).unwrap();
    let f1 = |pairs: &[(usize, usize)]| -> (f32, Vec<f32>) {
        let f: Vec<f32> = (0..7)
            .map(|i| {
                let tp = pairs.iter().filter(|p| p.0 == i && p.1 == i).count() as f32;
                let pred = pairs.iter().filter(|p| p.1 == i).count() as f32;
                let gold = pairs.iter().filter(|p| p.0 == i).count() as f32;
                let (p, r) = (if pred > 0.0 { tp / pred } else { 0.0 }, if gold > 0.0 { tp / gold } else { 0.0 });
                if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
            })
            .collect();
        (f.iter().sum::<f32>() / 7.0, f)
    };
    let t0 = std::time::Instant::now();
    let val = load("data/raw/dailydialog/validation_marked.json");
    let feels: Vec<Vec<(&str, f32)>> = val.iter().map(|(l, _)| w.feelings(l)).collect();
    let mut best = (0.0f32, -1.0f32);
    for k in 0..=12 {
        let at = 0.15 + 0.025 * k as f32;
        let pairs: Vec<(usize, usize)> = val.iter().zip(&feels).map(|((_, g), f)| (*g, f.iter().filter(|x| x.1 >= at).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map_or(0, |x| idx(x.0)))).collect();
        let m = f1(&pairs).0;
        println!("  validation: threshold {at:.3}: mean F1 {m:.3}");
        if m > best.1 {
            best = (at, m);
        }
    }
    println!("chosen on validation ({} turns, {:.1} s): threshold {:.3} (mean F1 {:.3}); FEEL_AT in the code is {FEEL_AT}", val.len(), t0.elapsed().as_secs_f64(), best.0, best.1);
    let test = load("data/raw/dailydialog/test_marked.json");
    let at = crate::lexicon::Atoms::new(dim, seed);
    let u = crate::understand::strike_or_load(&at).map(|x| x.0);
    let mut alone = Vec::new();
    let mut both = Vec::new();
    let mut learned = Vec::new();
    for (l, g) in &test {
        let n = w.feeling(l, best.0).map_or(0, idx);
        alone.push((*g, n));
        if let Some(u) = &u {
            let r = idx(u.read(l).emotion);
            learned.push((*g, r));
            both.push((*g, if r == 0 { n } else { r }));
        }
    }
    for (name, p) in [("the network alone", &alone), ("the learned understanding alone", &learned), ("understanding, then the network", &both)] {
        if p.is_empty() {
            continue;
        }
        let (m, f) = f1(p);
        println!("test ({} turns): {name}: mean F1 {m:.3};{}", p.len(), names.iter().zip(&f).map(|(n, x)| format!(" {n} {x:.2}")).collect::<String>());
    }
}
