//! Grammar in waves, taught by UD (people's marks on 12 544 real sentences): every place in a sentence is a wave of
//! what is there, each part bound to its own role (the word, the words before and after, the word's last letters),
//! and every word class and every role in the sentence (doer, done-to, the sentence's core, ...) is a prototype wave:
//! the places people marked so, superposed. Reading a new sentence is interference: each place takes the class (and
//! the role) its wave resonates with most.
//!
//! Strikes, each one pass over the sentences: the first adds every place to its class; each one after strikes only
//! where it reads wrong (the place added to its class and taken from the class it was read as), so what is already
//! known is not struck again. A pass reads a batch of sentences at once (on every core) and strikes after the batch.
use crate::vector::HVec;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const SYNTAX_D: usize = 2048;
/// sentences read at once before their strikes are made
pub const BATCH: usize = 1000;

/// one word of a marked sentence: as written (lowercase), its class, its role (the relation without its subtype),
/// whether it was written with a capital
#[derive(Clone)]
pub struct Tok {
    pub word: String,
    pub cap: bool,
    pub pos: String,
    pub rel: String,
}

pub fn load(path: &str) -> Vec<Vec<Tok>> {
    let mut out = Vec::new();
    let mut s: Vec<Tok> = Vec::new();
    for line in std::fs::read_to_string(path).unwrap_or_default().lines() {
        if line.trim().is_empty() {
            if !s.is_empty() {
                out.push(std::mem::take(&mut s));
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 8 || f[0].contains('-') || f[0].contains('.') {
            continue;
        }
        s.push(Tok { word: f[1].to_lowercase(), cap: f[1].chars().next().is_some_and(|c| c.is_uppercase()), pos: f[3].to_string(), rel: f[7].split(':').next().unwrap_or("").to_string() });
    }
    if !s.is_empty() {
        out.push(s);
    }
    out
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// the same work on every core: items in chunks, results in order
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    let size = items.len().div_ceil(n).max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = items.chunks(size).map(|c| sc.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

#[derive(Serialize, Deserialize)]
struct Struck {
    pos_names: Vec<String>,
    rel_names: Vec<String>,
    pos: Vec<HVec>,
    rel: Vec<HVec>,
}

pub struct Syntax {
    seed: u64,
    atoms: HashMap<String, HVec>,
    pub pos_names: Vec<String>,
    pub rel_names: Vec<String>,
    pos: Vec<HVec>,
    rel: Vec<HVec>,
}

fn last(w: &str, n: usize) -> Option<String> {
    let c: Vec<char> = w.chars().collect();
    (c.len() > n).then(|| c[c.len() - n..].iter().collect())
}

/// the names a place's waves are made of (role, value): for its class, and the extra ones for its role
fn class_parts(s: &[Tok], i: usize) -> Vec<(&'static str, String)> {
    let w = |k: isize| -> String {
        let j = i as isize + k;
        if j < 0 { "<s>".into() } else if j as usize >= s.len() { "</s>".into() } else { s[j as usize].word.clone() }
    };
    let mut p = vec![("w", w(0)), ("l1", w(-1)), ("l2", w(-2)), ("r1", w(1)), ("r2", w(2))];
    for (n, r) in [(1, "end1"), (2, "end2"), (3, "end3")] {
        if let Some(x) = last(&s[i].word, n) {
            p.push((r, x));
        }
    }
    if s[i].cap && i > 0 {
        p.push(("cap", "yes".into()));
    }
    p
}

impl Syntax {
    fn new(seed: u64, pos_names: Vec<String>, rel_names: Vec<String>) -> Syntax {
        let (np, nr) = (pos_names.len(), rel_names.len());
        Syntax { seed, atoms: HashMap::new(), pos_names, rel_names, pos: vec![HVec::zero(SYNTAX_D); np], rel: vec![HVec::zero(SYNTAX_D); nr] }
    }

    fn make(&self, name: &str) -> HVec {
        HVec::random(SYNTAX_D, &mut ChaCha8Rng::seed_from_u64(fnv(name) ^ self.seed))
    }

    /// every atom these sentences need, made once (so reading can run on every core)
    fn prepare(&mut self, sets: &[&[Vec<Tok>]]) {
        let mut names: Vec<String> = Vec::new();
        for set in sets {
            for s in set.iter() {
                for i in 0..s.len() {
                    for (r, v) in class_parts(s, i) {
                        names.push(format!("r:{r}"));
                        names.push(format!("v:{v}"));
                    }
                }
            }
        }
        for r in ["p", "pl1", "pl2", "pr1", "pr2", "p_pair", "p_pair_r", "verb_before", "verb_after"] {
            names.push(format!("r:{r}"));
        }
        for a in self.pos_names.clone() {
            names.push(format!("v:{a}"));
            for b in self.pos_names.clone().into_iter().chain(["<s>".to_string(), "</s>".to_string()]) {
                names.push(format!("v:{a}>{b}"));
                names.push(format!("v:{b}>{a}"));
            }
        }
        names.extend(["v:yes", "v:no", "v:<s>", "v:</s>"].map(String::from));
        names.sort();
        names.dedup();
        names.retain(|n| !self.atoms.contains_key(n));
        let made = par_map(&names, |n| self.make(n));
        self.atoms.extend(names.into_iter().zip(made));
    }

    fn bound(&self, role: &str, value: &str) -> HVec {
        let r = format!("r:{role}");
        let v = format!("v:{value}");
        let (ra, va) = (self.atoms.get(&r).cloned().unwrap_or_else(|| self.make(&r)), self.atoms.get(&v).cloned().unwrap_or_else(|| self.make(&v)));
        ra.bind(&va)
    }

    /// the wave of a place for its class: the word, its neighbours (two each way), its last 1-3 letters, a capital
    fn class_wave(&self, s: &[Tok], i: usize) -> HVec {
        let mut v = HVec::zero(SYNTAX_D);
        for (r, x) in class_parts(s, i) {
            v.add(&self.bound(r, &x));
        }
        v
    }

    /// the wave of a place for its role: its class wave, its own class and its neighbours' (as read), and whether a
    /// verb comes before it or after it in the sentence (as read)
    fn role_wave(&self, s: &[Tok], i: usize, tags: &[usize]) -> HVec {
        let mut v = self.class_wave(s, i);
        let t = |k: isize| -> String {
            let j = i as isize + k;
            if j < 0 { "<s>".into() } else if j as usize >= s.len() { "</s>".into() } else { self.pos_names[tags[j as usize]].clone() }
        };
        v.add(&self.bound("p", &t(0)));
        for (r, k) in [("pl1", -1), ("pl2", -2), ("pr1", 1), ("pr2", 2)] {
            v.add(&self.bound(r, &t(k)));
        }
        v.add(&self.bound("p_pair", &format!("{}>{}", t(-1), t(0))));
        v.add(&self.bound("p_pair_r", &format!("{}>{}", t(0), t(1))));
        let is_verb = |j: usize| matches!(self.pos_names[tags[j]].as_str(), "VERB" | "AUX");
        v.add(&self.bound("verb_before", if (0..i).any(is_verb) { "yes" } else { "no" }));
        v.add(&self.bound("verb_after", if (i + 1..s.len()).any(is_verb) { "yes" } else { "no" }));
        v
    }

    fn best(protos: &[HVec], v: &HVec) -> usize {
        (0..protos.len()).max_by(|&a, &b| v.sim(&protos[a]).partial_cmp(&v.sim(&protos[b])).unwrap()).unwrap_or(0)
    }

    pub fn read_classes(&self, s: &[Tok]) -> Vec<usize> {
        (0..s.len()).map(|i| Self::best(&self.pos, &self.class_wave(s, i))).collect()
    }

    pub fn read_roles(&self, s: &[Tok], tags: &[usize]) -> Vec<usize> {
        (0..s.len()).map(|i| Self::best(&self.rel, &self.role_wave(s, i, tags))).collect()
    }

    fn index(names: &[String]) -> HashMap<String, usize> {
        names.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect()
    }

    /// the first strike: every place added to its class and its role (roles read on people's classes)
    fn strike(&mut self, train: &[Vec<Tok>]) {
        let (pi, ri) = (Self::index(&self.pos_names), Self::index(&self.rel_names));
        let parts = par_map(&train.chunks(BATCH).collect::<Vec<_>>(), |c| {
            let mut pos = vec![HVec::zero(SYNTAX_D); self.pos_names.len()];
            let mut rel = vec![HVec::zero(SYNTAX_D); self.rel_names.len()];
            for s in c.iter() {
                let gold: Vec<usize> = s.iter().map(|t| pi[&t.pos]).collect();
                for i in 0..s.len() {
                    pos[gold[i]].add(&self.class_wave(s, i));
                    rel[ri[&s[i].rel]].add(&self.role_wave(s, i, &gold));
                }
            }
            (pos, rel)
        });
        for (pos, rel) in parts {
            for (a, b) in self.pos.iter_mut().zip(&pos) {
                a.add(b);
            }
            for (a, b) in self.rel.iter_mut().zip(&rel) {
                a.add(b);
            }
        }
    }

    /// a strike after the first: one more pass, striking only where it reads wrong (roles on its own classes, as it
    /// will read them). Returns how many places were struck (classes, roles)
    fn correct(&mut self, train: &[Vec<Tok>]) -> (usize, usize) {
        let (pi, ri) = (Self::index(&self.pos_names), Self::index(&self.rel_names));
        let (mut wc, mut wr) = (0, 0);
        for batch in train.chunks(BATCH) {
            // (a class? , wanted, read, the wave)
            let wrong: Vec<Vec<(bool, usize, usize, HVec)>> = par_map(batch, |s| {
                let mut out = Vec::new();
                let tags = self.read_classes(s);
                for i in 0..s.len() {
                    let want = pi[&s[i].pos];
                    if tags[i] != want {
                        out.push((true, want, tags[i], self.class_wave(s, i)));
                    }
                }
                for (i, got) in self.read_roles(s, &tags).into_iter().enumerate() {
                    let want = ri[&s[i].rel];
                    if got != want {
                        out.push((false, want, got, self.role_wave(s, i, &tags)));
                    }
                }
                out
            });
            for (class, want, got, v) in wrong.into_iter().flatten() {
                let p = if class { &mut self.pos } else { &mut self.rel };
                p[want].add(&v);
                p[got].add_scaled(&v, -1.0);
                if class { wc += 1 } else { wr += 1 }
            }
        }
        (wc, wr)
    }
}

/// grammar struck from UD's train split (`strikes` passes), saved, or loaded if struck before
pub fn strike_or_load(seed: u64, strikes: usize, train: &[Vec<Tok>], dev: Option<&dyn Fn(&Syntax, usize)>) -> (Syntax, bool) {
    let path = format!("data/strike/syntax_D{SYNTAX_D}_k{strikes}_s{seed}.json");
    if let Some(s) = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Struck>(&s).ok()) {
        let mut sx = Syntax::new(seed, s.pos_names, s.rel_names);
        sx.pos = s.pos;
        sx.rel = s.rel;
        return (sx, true);
    }
    let mut names = |f: fn(&Tok) -> String| {
        let mut v: Vec<String> = train.iter().flatten().map(f).collect();
        v.sort();
        v.dedup();
        v
    };
    let (pn, rn) = (names(|t| t.pos.clone()), names(|t| t.rel.clone()));
    let mut sx = Syntax::new(seed, pn, rn);
    sx.prepare(&[train]);
    let t = std::time::Instant::now();
    sx.strike(train);
    println!("strike 1: every place added, {:.1} s", t.elapsed().as_secs_f64());
    if let Some(f) = dev {
        f(&sx, 1);
    }
    for k in 1..strikes {
        let t = std::time::Instant::now();
        let (wc, wr) = sx.correct(train);
        println!("strike {}: struck where it read wrong ({wc} classes, {wr} roles), {:.1} s", k + 1, t.elapsed().as_secs_f64());
        if let Some(f) = dev {
            f(&sx, k + 1);
        }
    }
    let _ = std::fs::create_dir_all("data/strike");
    let s = Struck { pos_names: sx.pos_names.clone(), rel_names: sx.rel_names.clone(), pos: sx.pos.clone(), rel: sx.rel.clone() };
    let _ = std::fs::write(&path, serde_json::to_string(&s).unwrap());
    (sx, false)
}

/// `darma syntax-eval [strikes]`: grammar taught by UD's train split, watched on dev while striking, measured once on
/// the test split, against simple yardsticks: each word's commonest class (an unseen word: the commonest class of
/// all), and that class's commonest role
pub fn eval(seed: u64, strikes: usize) {
    let t0 = std::time::Instant::now();
    let train = load("data/raw/ud/en_ewt-ud-train.conllu");
    let dev = load("data/raw/ud/en_ewt-ud-dev.conllu");
    let test = load("data/raw/ud/en_ewt-ud-test.conllu");
    if train.is_empty() {
        return println!("UD not found (data/raw/ud)");
    }
    let mut by_word: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let mut by_pos: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let mut all: HashMap<String, u32> = HashMap::new();
    for t in train.iter().flatten() {
        *by_word.entry(t.word.clone()).or_default().entry(t.pos.clone()).or_default() += 1;
        *by_pos.entry(t.pos.clone()).or_default().entry(t.rel.clone()).or_default() += 1;
        *all.entry(t.pos.clone()).or_default() += 1;
    }
    let top = |m: &HashMap<String, u32>| m.iter().max_by_key(|x| (x.1, x.0)).map(|x| x.0.clone()).unwrap_or_default();
    let commonest = top(&all);
    let measure = |sx: &Syntax, set: &[Vec<Tok>], name: &str| {
        let read: Vec<(Vec<usize>, Vec<usize>)> = par_map(set, |s| {
            let tags = sx.read_classes(s);
            let roles = sx.read_roles(s, &tags);
            (tags, roles)
        });
        let (mut n, mut c, mut cb, mut un, mut uc, mut ucb, mut r, mut rb) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut rel_f: HashMap<&str, (usize, usize, usize)> = HashMap::new();
        for (s, (tags, roles)) in set.iter().zip(&read) {
            for (i, t) in s.iter().enumerate() {
                let got = &sx.pos_names[tags[i]];
                let base = by_word.get(&t.word).map(top).unwrap_or_else(|| commonest.clone());
                n += 1;
                c += (*got == t.pos) as usize;
                cb += (base == t.pos) as usize;
                if !by_word.contains_key(&t.word) {
                    un += 1;
                    uc += (*got == t.pos) as usize;
                    ucb += (base == t.pos) as usize;
                }
                let gr = &sx.rel_names[roles[i]];
                r += (*gr == t.rel) as usize;
                rb += by_pos.get(&base).map(top).is_some_and(|x| x == t.rel) as usize;
                for k in ["nsubj", "obj", "root"] {
                    let e = rel_f.entry(k).or_default();
                    e.0 += (t.rel == k && *gr == k) as usize;
                    e.1 += (*gr == k) as usize;
                    e.2 += (t.rel == k) as usize;
                }
            }
        }
        let p = |a: usize, b: usize| 100.0 * a as f32 / b.max(1) as f32;
        let f1 = |(tp, pr, g): (usize, usize, usize)| {
            let (a, b) = (tp as f32 / pr.max(1) as f32, tp as f32 / g.max(1) as f32);
            if a + b > 0.0 { 2.0 * a * b / (a + b) } else { 0.0 }
        };
        println!("  {name}: classes {:.1}% (yardstick {:.1}%); unseen words ({un}) {:.1}% (yardstick {:.1}%); roles {:.1}% (yardstick {:.1}%); role F1 doer {:.2}, done-to {:.2}, core {:.2}", p(c, n), p(cb, n), p(uc, un), p(ucb, un), p(r, n), p(rb, n), f1(rel_f["nsubj"]), f1(rel_f["obj"]), f1(rel_f["root"]));
    };
    let watch = |sx: &Syntax, _k: usize| measure(sx, &dev, "dev");
    let (mut sx, loaded) = strike_or_load(seed, strikes, &train, Some(&watch));
    if loaded {
        println!("loaded the {strikes} strikes struck before");
    }
    sx.prepare(&[&dev, &test]);
    measure(&sx, &test, "TEST");
    println!("all in {:.1} s", t0.elapsed().as_secs_f64());
}
