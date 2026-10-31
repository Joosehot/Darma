//! Hearing: which word each word completes (its core word), read in waves, taught by UD. Every pair of words in a
//! sentence is a wave of role-bound parts (their classes as Darma reads them, the direction and distance between
//! them, the small words themselves, whether a verb or a mark stands between), and the attachment, a register for
//! each of those parts (as the core keeps one per word, so they do not crowd one wave), holds what a word and its
//! core word look like: the first strike is the pairs people marked minus the pairs they did
//! not; each strike after strikes only where it reads wrong (the right pair added, the pair it chose taken away). A
//! sentence's tree is then the best whole tree by the attachment's resonance (Eisner's way: every word one core
//! word, no lines crossing), so the reading is always a tree.
use crate::syntax::{par_map, Syntax, Tok};
use crate::vector::HVec;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};

pub const HEAR_D: usize = 1024;
pub const BATCH: usize = 1000;
/// the parts of a pair's wave, each with its own register
pub const ROLES: usize = 10;

/// a marked sentence for hearing: its words (for the grammar) and each word's core word (0: the sentence's core)
pub struct Heard {
    pub toks: Vec<Tok>,
    pub heads: Vec<usize>,
}

pub fn load(path: &str) -> Vec<Heard> {
    let mut out = Vec::new();
    let (mut toks, mut heads) = (Vec::new(), Vec::new());
    let text = std::fs::read_to_string(path).unwrap_or_default();
    for line in text.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !toks.is_empty() {
                let n = toks.len();
                let h: Vec<usize> = std::mem::take(&mut heads);
                let t: Vec<Tok> = std::mem::take(&mut toks);
                if h.iter().all(|&x| x <= n) {
                    out.push(Heard { toks: t, heads: h });
                }
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
        toks.push(Tok { word: f[1].to_lowercase(), cap: f[1].chars().next().is_some_and(|c| c.is_uppercase()), pos: f[3].to_string(), rel: f[7].split(':').next().unwrap_or("").to_string() });
        heads.push(f[6].parse().unwrap_or(0));
    }
    out
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

pub fn bucket(d: usize) -> &'static str {
    match d {
        1 => "1",
        2 => "2",
        3..=4 => "3-4",
        5..=7 => "5-7",
        _ => "8+",
    }
}

pub struct Hear {
    seed: u64,
    atoms: std::sync::RwLock<HashMap<String, HVec>>,
    attach: Vec<HVec>,
    pub small: HashSet<String>,
}

impl Hear {
    fn atom(&self, name: &str) -> HVec {
        if let Some(a) = self.atoms.read().unwrap().get(name) {
            return a.clone();
        }
        let a = HVec::random(HEAR_D, &mut ChaCha8Rng::seed_from_u64(fnv(name) ^ self.seed));
        self.atoms.write().unwrap().insert(name.to_string(), a.clone());
        a
    }

    /// the wave of "word d completes word h" (h = 0: d is the sentence's core); words are 1-based, classes as read
    fn pair(&self, w: &[String], c: &[String], d: usize, h: usize) -> Vec<HVec> {
        self.pair_feats(w, c, d, h).into_iter().map(|(r, x)| self.atom(&format!("r:{r}")).bind(&self.atom(&format!("v:{x}")))).collect()
    }

    /// an atom by its name, made if it was never needed before (not kept)
    pub fn atom_fresh(&self, n: &str) -> HVec {
        self.atoms.read().unwrap().get(n).cloned().unwrap_or_else(|| HVec::random(HEAR_D, &mut ChaCha8Rng::seed_from_u64(fnv(n) ^ self.seed)))
    }

    /// a part's wave made without keeping its atoms
    pub fn bound_fresh(&self, role: &str, value: &str) -> HVec {
        let make = |n: &str| self.atoms.read().unwrap().get(n).cloned().unwrap_or_else(|| HVec::random(HEAR_D, &mut ChaCha8Rng::seed_from_u64(fnv(n) ^ self.seed)));
        make(&format!("r:{role}")).bind(&make(&format!("v:{value}")))
    }

    pub fn attach(&self) -> &[HVec] {
        &self.attach
    }

    /// the parts of "word d completes word h" (role, value), in the order of the attachment's registers
    pub fn pair_feats(&self, w: &[String], c: &[String], d: usize, h: usize) -> Vec<(&'static str, String)> {
        let cls = |k: usize| -> &str { if k == 0 { "ROOT" } else if k > c.len() { "END" } else { &c[k - 1] } };
        let sm = |k: usize| -> &str { if k == 0 || k > w.len() || !self.small.contains(&w[k - 1]) { "-" } else { &w[k - 1] } };
        let (dc, hc) = (cls(d), cls(h));
        let dir = if h == 0 { "root" } else if h < d { "L" } else { "R" };
        let dist = if h == 0 { "root" } else { bucket(d.abs_diff(h)) };
        let (lo, hi) = (d.min(h), d.max(h));
        let between = |p: &dyn Fn(&str) -> bool| h != 0 && (lo + 1..hi).any(|k| p(&c[k - 1]));
        let verb = between(&|x| x == "VERB");
        let punct = between(&|x| x == "PUNCT");
        let parts: Vec<(&'static str, String)> = vec![
            ("pair", format!("{dc}>{hc}")),
            ("pairdir", format!("{dc}>{hc} {dir}")),
            ("pairdist", format!("{dc}>{hc} {dir} {dist}")),
            ("dw", format!("{}>{hc} {dir}", sm(d))),
            ("hw", format!("{dc}>{} {dir}", sm(h))),
            ("words", format!("{}>{}", sm(d), sm(h))),
            ("hctx", format!("{dc}>{hc} {dir} {}|{}", if h == 0 { "-" } else { cls(h - 1) }, if h == 0 { "-" } else { cls(h + 1) })),
            ("dctx", format!("{dc}>{hc} {dir} {}|{}", cls(d - 1), cls(d + 1))),
            ("verb_between", format!("{dc}>{hc} {verb}")),
            ("punct_between", format!("{dc}>{hc} {punct}")),
        ];
        parts
    }

    fn score(&self, v: &[HVec]) -> f32 {
        v.iter().zip(&self.attach).map(|(a, b)| a.inner(b).re).sum()
    }

    /// every pair's resonance with the attachment: s[h][d] (h = 0: the core)
    fn scores(&self, w: &[String], c: &[String]) -> Vec<Vec<f32>> {
        let n = w.len();
        let mut s = vec![vec![f32::MIN; n + 1]; n + 1];
        for d in 1..=n {
            for h in 0..=n {
                if h != d {
                    s[h][d] = self.score(&self.pair(w, c, d, h));
                }
            }
        }
        s
    }

    /// each word's core word, the best of its own (not always a tree)
    pub fn greedy(&self, w: &[String], c: &[String]) -> Vec<usize> {
        let s = self.scores(w, c);
        (1..=w.len()).map(|d| (0..=w.len()).filter(|&h| h != d).max_by(|&a, &b| s[a][d].partial_cmp(&s[b][d]).unwrap()).unwrap()).collect()
    }

    /// the best whole tree (no lines crossing), by Eisner's way
    pub fn tree(&self, w: &[String], c: &[String]) -> Vec<usize> {
        eisner(&self.scores(w, c))
    }

    pub fn strike(train: &[Heard], classes: &[Vec<String>], seed: u64, strikes: usize, small: HashSet<String>) -> Hear {
        let mut h = Hear { seed, atoms: Default::default(), attach: vec![HVec::zero(HEAR_D); ROLES], small };
        let words: Vec<Vec<String>> = train.iter().map(|s| s.toks.iter().map(|t| t.word.clone()).collect()).collect();
        let idx: Vec<usize> = (0..train.len()).collect();
        let t = std::time::Instant::now();
        // the first strike: the marked pairs, minus the unmarked ones (each scaled to the marked ones' number)
        let parts: Vec<(Vec<HVec>, Vec<HVec>, usize, usize)> = par_map(&idx.chunks(BATCH).collect::<Vec<_>>(), |ch| {
            let (mut pos, mut neg, mut np, mut nn) = (vec![HVec::zero(HEAR_D); ROLES], vec![HVec::zero(HEAR_D); ROLES], 0, 0);
            for &k in ch.iter() {
                let n = words[k].len();
                for d in 1..=n {
                    for hh in 0..=n {
                        if hh == d {
                            continue;
                        }
                        let v = h.pair(&words[k], &classes[k], d, hh);
                        let into = if hh == train[k].heads[d - 1] { np += 1; &mut pos } else { nn += 1; &mut neg };
                        for (a, b) in into.iter_mut().zip(&v) {
                            a.add(b);
                        }
                    }
                }
            }
            (pos, neg, np, nn)
        });
        let (mut pos, mut neg, mut np, mut nn) = (vec![HVec::zero(HEAR_D); ROLES], vec![HVec::zero(HEAR_D); ROLES], 0, 0);
        for (p, q, a, b) in parts {
            for r in 0..ROLES {
                pos[r].add(&p[r]);
                neg[r].add(&q[r]);
            }
            np += a;
            nn += b;
        }
        for r in 0..ROLES {
            h.attach[r] = pos[r].clone();
            h.attach[r].add_scaled(&neg[r], -(np as f32) / nn.max(1) as f32);
        }
        println!("hearing strike 1: {np} marked pairs, {nn} unmarked, {:.1} s", t.elapsed().as_secs_f64());
        for k in 1..strikes {
            let t = std::time::Instant::now();
            let mut wrong = 0;
            for batch in idx.chunks(BATCH) {
                let strikes: Vec<Vec<(Vec<HVec>, Vec<HVec>)>> = par_map(batch, |&s| {
                    let got = h.greedy(&words[s], &classes[s]);
                    (1..=words[s].len()).filter(|&d| got[d - 1] != train[s].heads[d - 1]).map(|d| (h.pair(&words[s], &classes[s], d, train[s].heads[d - 1]), h.pair(&words[s], &classes[s], d, got[d - 1]))).collect()
                });
                for (right, chosen) in strikes.into_iter().flatten() {
                    for r in 0..ROLES {
                        h.attach[r].add(&right[r]);
                        h.attach[r].add_scaled(&chosen[r], -1.0);
                    }
                    wrong += 1;
                }
            }
            println!("hearing strike {}: struck where it read wrong ({wrong} words), {:.1} s", k + 1, t.elapsed().as_secs_f64());
        }
        h
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Struck {
    attach: Vec<HVec>,
    small: Vec<String>,
}

impl Hear {
    /// hearing struck from UD's train split on the classes the grammar reads (`strikes` passes), saved; or loaded
    pub fn strike_or_load(seed: u64, strikes: usize, sx: &Syntax) -> Hear {
        let path = format!("data/strike/hear_D{HEAR_D}_k{strikes}_s{seed}.json");
        if let Some(s) = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Struck>(&s).ok()) {
            return Hear { seed, atoms: Default::default(), attach: s.attach, small: s.small.into_iter().collect() };
        }
        let train = load("data/raw/ud/en_ewt-ud-train.conllu");
        let tr: Vec<Vec<Tok>> = train.iter().map(|s| s.toks.clone()).collect();
        let classes: Vec<Vec<String>> = par_map(&tr, |s| sx.read_classes(s).into_iter().map(|k| sx.pos_names[k].clone()).collect());
        let h = Hear::strike(&train, &classes, seed, strikes, small_words(&tr));
        let mut small: Vec<String> = h.small.iter().cloned().collect();
        small.sort();
        let _ = std::fs::create_dir_all("data/strike");
        let _ = std::fs::write(&path, serde_json::to_string(&Struck { attach: h.attach.clone(), small }).unwrap());
        h
    }

    /// the core word of each word of a sentence (1-based; 0: the sentence's core), its classes as read
    pub fn heads(&self, words: &[String], classes: &[String]) -> Vec<usize> {
        self.tree(words, classes)
    }
}

/// the 300 commonest words of the sentences: their own place matters ("not", "to", "of")
pub fn small_words(tr: &[Vec<Tok>]) -> HashSet<String> {
    let mut c: HashMap<String, u32> = HashMap::new();
    for t in tr.iter().flatten() {
        *c.entry(t.word.clone()).or_default() += 1;
    }
    let mut v: Vec<(String, u32)> = c.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(300).map(|x| x.0).collect()
}

/// Eisner's first-order projective decoding: the tree with the highest sum of s[head][dependent] (0 = the core)
pub fn eisner(s: &[Vec<f32>]) -> Vec<usize> {
    let n = s.len();
    // c[i][j][dir][complete]: dir 0 = head at j (left arc), 1 = head at i (right arc)
    let neg = f32::MIN / 4.0;
    let mut c = vec![vec![[[neg; 2]; 2]; n]; n];
    let mut b = vec![vec![[[0usize; 2]; 2]; n]; n];
    for i in 0..n {
        c[i][i] = [[0.0; 2]; 2];
    }
    for len in 1..n {
        for i in 0..n - len {
            let j = i + len;
            // incomplete: an arc between i and j
            for r in i..j {
                let base = c[i][r][1][1] + c[r + 1][j][0][1];
                let l = base + s[j][i];
                if l > c[i][j][0][0] {
                    c[i][j][0][0] = l;
                    b[i][j][0][0] = r;
                }
                let rr = base + s[i][j];
                if rr > c[i][j][1][0] {
                    c[i][j][1][0] = rr;
                    b[i][j][1][0] = r;
                }
            }
            // complete
            for r in i..j {
                let v = c[i][r][0][1] + c[r][j][0][0];
                if v > c[i][j][0][1] {
                    c[i][j][0][1] = v;
                    b[i][j][0][1] = r;
                }
            }
            for r in i + 1..=j {
                let v = c[i][r][1][0] + c[r][j][1][1];
                if v > c[i][j][1][1] {
                    c[i][j][1][1] = v;
                    b[i][j][1][1] = r;
                }
            }
        }
    }
    let mut heads = vec![0usize; n];
    fn back(i: usize, j: usize, dir: usize, comp: usize, b: &[Vec<[[usize; 2]; 2]>], heads: &mut [usize]) {
        if i == j {
            return;
        }
        let r = b[i][j][dir][comp];
        if comp == 0 {
            if dir == 0 { heads[i] = j } else { heads[j] = i }
            back(i, r, 1, 1, b, heads);
            back(r + 1, j, 0, 1, b, heads);
        } else if dir == 0 {
            back(i, r, 0, 1, b, heads);
            back(r, j, 0, 0, b, heads);
        } else {
            back(i, r, 1, 0, b, heads);
            back(r, j, 1, 1, b, heads);
        }
    }
    back(0, n - 1, 1, 1, &b, &mut heads);
    heads[1..].to_vec()
}

/// `darma hear-eval [strikes]`: hearing taught by UD's train split (on the classes Darma's grammar reads), measured
/// on its test split: each word's core word right (alone, and with its role right too), against a counted yardstick
/// (each class attaches to the nearest word of the class and on the side it most often attaches to)
pub fn eval(seed: u64, strikes: usize) {
    let t0 = std::time::Instant::now();
    let train = load("data/raw/ud/en_ewt-ud-train.conllu");
    let test = load("data/raw/ud/en_ewt-ud-test.conllu");
    if train.is_empty() {
        return println!("UD not found (data/raw/ud)");
    }
    let tr: Vec<Vec<Tok>> = train.iter().map(|s| s.toks.clone()).collect();
    let (mut sx, loaded) = crate::syntax::strike_or_load(seed, 3, &tr, None);
    let te: Vec<Vec<Tok>> = test.iter().map(|s| s.toks.clone()).collect();
    sx.prepare_for(&[&tr, &te]);
    let read = |sx: &Syntax, set: &[Vec<Tok>]| -> Vec<Vec<String>> { par_map(set, |s| sx.read_classes(s).into_iter().map(|k| sx.pos_names[k].clone()).collect()) };
    // a diagnosis only (DARMA_GOLD_CLASSES=1): people's classes instead of Darma's, to see what the classes cost
    let gold = std::env::var("DARMA_GOLD_CLASSES").is_ok();
    let people = |set: &[Vec<Tok>]| -> Vec<Vec<String>> { set.iter().map(|s| s.iter().map(|t| t.pos.clone()).collect()).collect() };
    let (ctr, cte) = if gold { (people(&tr), people(&te)) } else { (read(&sx, &tr), read(&sx, &te)) };
    if gold {
        println!("DIAGNOSIS: people's classes, not Darma's");
    }
    println!("the grammar {} and read the classes, {:.1} s", if loaded { "loaded" } else { "struck" }, t0.elapsed().as_secs_f64());
    let h = Hear::strike(&train, &ctr, seed, strikes, small_words(&tr));
    // the yardstick: each class's commonest (core class, side), counted on people's marks
    let mut counts: HashMap<String, HashMap<(String, bool), u32>> = HashMap::new();
    for s in &train {
        for (i, t) in s.toks.iter().enumerate() {
            let hd = s.heads[i];
            let key = if hd == 0 { ("ROOT".to_string(), false) } else { (s.toks[hd - 1].pos.clone(), hd - 1 < i) };
            *counts.entry(t.pos.clone()).or_default().entry(key).or_default() += 1;
        }
    }
    let yard = |c: &[String]| -> Vec<usize> {
        (0..c.len())
            .map(|i| {
                let Some((hc, left)) = counts.get(&c[i]).and_then(|m| m.iter().max_by_key(|x| (x.1, x.0.clone())).map(|x| x.0.clone())) else { return 0 };
                if hc == "ROOT" {
                    return 0;
                }
                let found = if left { (0..i).rev().find(|&k| c[k] == hc) } else { (i + 1..c.len()).find(|&k| c[k] == hc) };
                found.map_or(if left { i } else { (i + 2).min(c.len()) }, |k| k + 1)
            })
            .collect()
    };
    let idx: Vec<usize> = (0..test.len()).collect();
    let res: Vec<(usize, usize, usize, usize, usize)> = par_map(&idx, |&k| {
        let w: Vec<String> = test[k].toks.iter().map(|t| t.word.clone()).collect();
        let tree = h.tree(&w, &cte[k]);
        let greedy = h.greedy(&w, &cte[k]);
        let tags = sx.read_classes(&test[k].toks);
        let roles = sx.read_roles(&test[k].toks, &tags);
        let y = yard(&cte[k]);
        let g = &test[k].heads;
        let n = w.len();
        let right = (0..n).filter(|&i| tree[i] == g[i]).count();
        let labelled = (0..n).filter(|&i| tree[i] == g[i] && sx.rel_names[roles[i]] == test[k].toks[i].rel).count();
        (n, right, labelled, (0..n).filter(|&i| greedy[i] == g[i]).count(), (0..n).filter(|&i| y[i] == g[i]).count())
    });
    let s = |f: fn(&(usize, usize, usize, usize, usize)) -> usize| res.iter().map(f).sum::<usize>();
    let p = |a: usize| 100.0 * a as f32 / s(|r| r.0).max(1) as f32;
    println!("hearing (test, {} words): core word right {:.1}% (each word on its own, not a tree: {:.1}%); with its role right too {:.1}%; yardstick (counted: nearest word of the usual class on the usual side) {:.1}%", s(|r| r.0), p(s(|r| r.1)), p(s(|r| r.3)), p(s(|r| r.2)), p(s(|r| r.4)));
    let raw: (usize, usize, usize, usize, usize) = res.iter().fold((0, 0, 0, 0, 0), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.2, a.3 + r.3, a.4 + r.4));
    println!("  counts (words, tree right, labelled, alone right, yardstick right): {raw:?}");
    let short: Vec<_> = res.iter().filter(|r| r.0 <= 10).collect();
    let (sn, sr, sy) = short.iter().fold((0, 0, 0), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.4));
    println!("  sentences of 10 words or fewer ({}, {sn} words): core word right {:.1}% (yardstick {:.1}%)", short.len(), 100.0 * sr as f32 / sn.max(1) as f32, 100.0 * sy as f32 / sn.max(1) as f32);
    println!("all in {:.1} s", t0.elapsed().as_secs_f64());
}
