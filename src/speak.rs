//! Speech: grammar run backwards, in waves, taught by UD. What Darma means to say is a structure (a core word and
//! the words that complete it, each known by its role and class, in base form with its features); speaking is
//! putting it in order and bending its words.
//!   order: every word that completes another reads, by interference, its place beside it: how many places before or
//!          after (among the words completing the same one). The sentence is made from the tree, each word with its
//!          own completing words around it.
//!   bending: a word's base form and features read the change its ending takes (+ed, y -> ied, +s, or a whole form
//!          of its own: go -> went).
//! Both are struck as the grammar is: the first strike adds, later ones strike only where it reads wrong.
use crate::syntax::par_map;
use crate::vector::HVec;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

pub const SPEAK_D: usize = 1024;
pub const BATCH: usize = 1000;
/// the places a completing word can take: 1..=SIDE before or after its core word (further ones take the last)
pub const SIDE: i32 = 4;

#[derive(Clone)]
pub struct Word {
    pub form: String,
    pub lemma: String,
    pub pos: String,
    pub feats: String,
    pub head: usize,
    pub rel: String,
}

pub fn load(path: &str) -> Vec<Vec<Word>> {
    let mut out = Vec::new();
    let mut s: Vec<Word> = Vec::new();
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
        s.push(Word { form: f[1].to_lowercase(), lemma: f[2].to_lowercase(), pos: f[3].into(), feats: f[5].into(), head: f[6].parse().unwrap_or(0), rel: f[7].split(':').next().unwrap_or("").into() });
    }
    if !s.is_empty() {
        out.push(s);
    }
    // only sentences whose tree is whole (every head within the sentence)
    out.into_iter().filter(|s| s.iter().all(|w| w.head <= s.len())).collect()
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn ending(w: &str, n: usize) -> String {
    let c: Vec<char> = w.chars().collect();
    if c.len() >= n { c[c.len() - n..].iter().collect() } else { format!("^{}", w) }
}

/// the change from a base form to a form: how many letters off its end, and what is put on ("-1+ied")
fn change(lemma: &str, form: &str) -> String {
    let (a, b): (Vec<char>, Vec<char>) = (lemma.chars().collect(), form.chars().collect());
    let k = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    format!("-{}+{}", a.len() - k, b[k..].iter().collect::<String>())
}

fn apply(lemma: &str, ch: &str) -> String {
    let (cut, put) = ch[1..].split_once('+').unwrap_or(("0", ""));
    let a: Vec<char> = lemma.chars().collect();
    let n = cut.parse::<usize>().unwrap_or(0).min(a.len());
    a[..a.len() - n].iter().collect::<String>() + put
}

/// the sentence laid out from its tree, each word at its place beside its core word
pub fn lay_out(s: &[Word], pl: &[(i32, f32)]) -> Vec<usize> {
    fn lay(h: usize, s: &[Word], pl: &[(i32, f32)], out: &mut Vec<usize>) {
        let mut kids: Vec<usize> = (0..s.len()).filter(|&i| s[i].head == h + 1).collect();
        // by place; of two in one place, the more sure one nearer its core word
        kids.sort_by(|&a, &b| pl[a].0.cmp(&pl[b].0).then(if pl[a].0 < 0 { pl[a].1.partial_cmp(&pl[b].1).unwrap() } else { pl[b].1.partial_cmp(&pl[a].1).unwrap() }));
        for &k in kids.iter().filter(|&&k| pl[k].0 < 0) {
            lay(k, s, pl, out);
        }
        out.push(h);
        for &k in kids.iter().filter(|&&k| pl[k].0 > 0) {
            lay(k, s, pl, out);
        }
    }
    let mut out = Vec::new();
    if let Some(root) = (0..s.len()).find(|&i| s[i].head == 0) {
        lay(root, s, pl, &mut out);
    }
    out
}

/// the place a completing word was in: its rank among the words on its side of the same core word
pub fn gold_place(s: &[Word], i: usize) -> i32 {
    let h = s[i].head - 1;
    let side: Vec<usize> = (0..s.len()).filter(|&j| s[j].head == h + 1 && (j < h) == (i < h)).collect();
    let rank = if i < h { side.iter().filter(|&&j| j > i).count() as i32 + 1 } else { side.iter().filter(|&&j| j < i).count() as i32 + 1 };
    let r = rank.min(SIDE);
    if i < h { -r } else { r }
}

pub struct Speak {
    seed: u64,
    atoms: HashMap<String, HVec>,
    /// the places (-SIDE..-1, 1..SIDE), as prototypes
    places: Vec<HVec>,
    /// the changes of endings seen, each a prototype; and which changes each class of word was seen to take
    changes: Vec<String>,
    bends: Vec<HVec>,
    by_pos: HashMap<String, Vec<usize>>,
}

fn place_index(p: i32) -> usize {
    (if p < 0 { p + SIDE } else { p + SIDE - 1 }) as usize
}

fn place_of(i: usize) -> i32 {
    let i = i as i32;
    if i < SIDE { i - SIDE } else { i - SIDE + 1 }
}

/// the parts of a completing word's wave (role, value): its role and class, its core word's role and class, its base
/// form if it is a small word (one of the commonest), and how many words complete it
fn place_parts(s: &[Word], i: usize, small: &dyn Fn(&str) -> bool) -> Vec<(&'static str, String)> {
    let w = &s[i];
    let h = &s[w.head - 1];
    let mut p = vec![("rel", w.rel.clone()), ("pos", w.pos.clone()), ("hrel", h.rel.clone()), ("hpos", h.pos.clone()), ("pair", format!("{}>{}", w.rel, h.pos))];
    if small(&w.lemma) {
        p.push(("lemma", w.lemma.clone()));
    }
    let kids = s.iter().filter(|x| x.head == i + 1).count().min(3);
    p.push(("kids", kids.to_string()));
    p
}

fn bend_parts(w: &Word) -> Vec<(&'static str, String)> {
    let mut p = vec![("pos", w.pos.clone()), ("feats", w.feats.clone()), ("pf", format!("{} {}", w.pos, w.feats)), ("lemma", w.lemma.clone())];
    for n in 1..=3 {
        p.push((["e1", "e2", "e3"][n - 1], ending(&w.lemma, n)));
    }
    p
}

impl Speak {
    fn make(&self, name: &str) -> HVec {
        HVec::random(SPEAK_D, &mut ChaCha8Rng::seed_from_u64(fnv(name) ^ self.seed))
    }

    fn prepare(&mut self, names: Vec<String>) {
        let mut names = names;
        names.sort();
        names.dedup();
        names.retain(|n| !self.atoms.contains_key(n));
        let made = par_map(&names, |n| self.make(n));
        self.atoms.extend(names.into_iter().zip(made));
    }

    fn wave(&self, parts: &[(&'static str, String)]) -> HVec {
        let mut v = HVec::zero(SPEAK_D);
        for (r, x) in parts {
            let (rn, vn) = (format!("r:{r}"), format!("v:{x}"));
            let ra = self.atoms.get(&rn).cloned().unwrap_or_else(|| self.make(&rn));
            let va = self.atoms.get(&vn).cloned().unwrap_or_else(|| self.make(&vn));
            v.add(&ra.bind(&va));
        }
        v
    }

    fn best(protos: &[HVec], v: &HVec, among: Option<&[usize]>) -> (usize, f32) {
        let all: Vec<usize> = (0..protos.len()).collect();
        among.unwrap_or(&all).iter().map(|&k| (k, v.sim(&protos[k]))).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap_or((0, 0.0))
    }

    /// the place a completing word takes beside its core word, and how strongly it resonates there
    pub fn place(&self, s: &[Word], i: usize, small: &dyn Fn(&str) -> bool) -> (i32, f32) {
        let (k, r) = Self::best(&self.places, &self.wave(&place_parts(s, i, small)), None);
        (place_of(k), r)
    }

    /// the words in the order Darma would say them (indices into s)
    pub fn order(&self, s: &[Word], small: &dyn Fn(&str) -> bool) -> Vec<usize> {
        let pl: Vec<(i32, f32)> = (0..s.len()).map(|i| if s[i].head == 0 { (0, 0.0) } else { self.place(s, i, small) }).collect();
        lay_out(s, &pl)
    }

    /// a word bent: its base form with the change its class, features and ending read
    pub fn bend(&self, w: &Word) -> String {
        let among = self.by_pos.get(&w.pos).map(|v| v.as_slice());
        let (k, _) = Self::best(&self.bends, &self.wave(&bend_parts(w)), among);
        apply(&w.lemma, &self.changes[k])
    }

    pub fn strike(train: &[Vec<Word>], seed: u64, strikes: usize, small: &(dyn Fn(&str) -> bool + Sync)) -> Speak {
        let mut changes: Vec<String> = train.iter().flatten().map(|w| change(&w.lemma, &w.form)).collect();
        changes.sort();
        changes.dedup();
        let ci: HashMap<String, usize> = changes.iter().cloned().enumerate().map(|(i, c)| (c, i)).collect();
        let mut by_pos: HashMap<String, Vec<usize>> = HashMap::new();
        for w in train.iter().flatten() {
            by_pos.entry(w.pos.clone()).or_default().push(ci[&change(&w.lemma, &w.form)]);
        }
        for v in by_pos.values_mut() {
            v.sort();
            v.dedup();
        }
        let mut sp = Speak { seed, atoms: HashMap::new(), places: vec![HVec::zero(SPEAK_D); 2 * SIDE as usize], bends: vec![HVec::zero(SPEAK_D); changes.len()], changes, by_pos };
        let mut names = Vec::new();
        for s in train {
            for i in 0..s.len() {
                for (r, v) in bend_parts(&s[i]).into_iter().chain(if s[i].head > 0 { place_parts(s, i, small) } else { Vec::new() }) {
                    names.push(format!("r:{r}"));
                    names.push(format!("v:{v}"));
                }
            }
        }
        sp.prepare(names);
        let gold = gold_place;
        let t = std::time::Instant::now();
        for s in train {
            for i in 0..s.len() {
                let bw = sp.wave(&bend_parts(&s[i]));
                sp.bends[ci[&change(&s[i].lemma, &s[i].form)]].add(&bw);
                if s[i].head > 0 {
                    let pw = sp.wave(&place_parts(s, i, small));
                    sp.places[place_index(gold(s, i))].add(&pw);
                }
            }
        }
        println!("speech strike 1: every word added, {:.1} s", t.elapsed().as_secs_f64());
        for k in 1..strikes {
            let t = std::time::Instant::now();
            let (mut wp, mut wb) = (0, 0);
            for batch in train.chunks(BATCH) {
                let wrong: Vec<Vec<(bool, usize, usize, HVec)>> = par_map(batch, |s| {
                    let mut out = Vec::new();
                    for i in 0..s.len() {
                        let bw = sp.wave(&bend_parts(&s[i]));
                        let among = sp.by_pos.get(&s[i].pos).map(|v| v.as_slice());
                        let (got, _) = Self::best(&sp.bends, &bw, among);
                        let want = ci[&change(&s[i].lemma, &s[i].form)];
                        if got != want {
                            out.push((false, want, got, bw));
                        }
                        if s[i].head > 0 {
                            let pw = sp.wave(&place_parts(s, i, small));
                            let (got, _) = Self::best(&sp.places, &pw, None);
                            let want = place_index(gold(s, i));
                            if got != want {
                                out.push((true, want, got, pw));
                            }
                        }
                    }
                    out
                });
                for (place, want, got, v) in wrong.into_iter().flatten() {
                    let p = if place { &mut sp.places } else { &mut sp.bends };
                    p[want].add(&v);
                    p[got].add_scaled(&v, -1.0);
                    if place { wp += 1 } else { wb += 1 }
                }
            }
            println!("speech strike {}: struck where it read wrong ({wp} places, {wb} bendings), {:.1} s", k + 1, t.elapsed().as_secs_f64());
        }
        sp
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Struck {
    places: Vec<HVec>,
    changes: Vec<String>,
    bends: Vec<HVec>,
    by_pos: HashMap<String, Vec<usize>>,
}

impl Speak {
    /// speech struck from UD's train split, saved; or loaded
    pub fn strike_or_load(seed: u64, strikes: usize) -> (Speak, std::collections::HashSet<String>) {
        let train = load("data/raw/ud/en_ewt-ud-train.conllu");
        let small = small_lemmas(&train);
        let path = format!("data/strike/speak_D{SPEAK_D}_k{strikes}_s{seed}.json");
        if let Some(s) = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Struck>(&s).ok()) {
            return (Speak { seed, atoms: HashMap::new(), places: s.places, changes: s.changes, bends: s.bends, by_pos: s.by_pos }, small);
        }
        let sm = small.clone();
        let sp = Speak::strike(&train, seed, strikes, &move |w: &str| sm.contains(w));
        let _ = std::fs::create_dir_all("data/strike");
        let _ = std::fs::write(&path, serde_json::to_string(&Struck { places: sp.places.clone(), changes: sp.changes.clone(), bends: sp.bends.clone(), by_pos: sp.by_pos.clone() }).unwrap());
        (sp, small)
    }
}

/// the 200 commonest base forms: their own place matters ("not", "also", "very")
pub fn small_lemmas(train: &[Vec<Word>]) -> std::collections::HashSet<String> {
    let mut c: HashMap<String, u32> = HashMap::new();
    for w in train.iter().flatten() {
        *c.entry(w.lemma.clone()).or_default() += 1;
    }
    let mut v: Vec<(String, u32)> = c.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(200).map(|x| x.0).collect()
}

/// `darma speak-eval [strikes]`: speech taught by UD's train split, measured on its test split: the sentence put in
/// order from its tree (whole sentences right, word pairs in the right order), and words bent from base form and
/// features; against yardsticks: each role's commonest side and rank, and the base form said as it is
pub fn eval(seed: u64, strikes: usize) {
    let t0 = std::time::Instant::now();
    let train = load("data/raw/ud/en_ewt-ud-train.conllu");
    let test = load("data/raw/ud/en_ewt-ud-test.conllu");
    if train.is_empty() {
        return println!("UD not found (data/raw/ud)");
    }
    // small words: the 200 commonest base forms (their own place matters: "not", "also", "very")
    let mut c: HashMap<String, u32> = HashMap::new();
    for w in train.iter().flatten() {
        *c.entry(w.lemma.clone()).or_default() += 1;
    }
    let mut v: Vec<(String, u32)> = c.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let small_set: std::collections::HashSet<String> = v.into_iter().take(200).map(|x| x.0).collect();
    let small = |w: &str| small_set.contains(w);
    let sp = Speak::strike(&train, seed, strikes, &small);
    // the yardstick for order: each (role, core word's class)'s commonest place, counted (a small word: with its base form)
    let key = |s: &[Word], i: usize| format!("{} {} {}", s[i].rel, s[s[i].head - 1].pos, if small(&s[i].lemma) { s[i].lemma.as_str() } else { "" });
    let mut counts: HashMap<String, HashMap<i32, u32>> = HashMap::new();
    for s in &train {
        for i in 0..s.len() {
            if s[i].head > 0 {
                *counts.entry(key(s, i)).or_default().entry(gold_place(s, i)).or_default() += 1;
            }
        }
    }
    let counted = |s: &[Word]| -> Vec<usize> {
        let pl: Vec<(i32, f32)> = (0..s.len()).map(|i| if s[i].head == 0 { (0, 0.0) } else { counts.get(&key(s, i)).and_then(|m| m.iter().max_by_key(|x| (x.1, -x.0)).map(|x| (*x.0, *x.1 as f32))).unwrap_or((1, 0.0)) }).collect();
        lay_out(s, &pl)
    };
    let yard: Vec<(usize, usize, bool)> = par_map(&test, |s| {
        let o = counted(s);
        let pos: HashMap<usize, usize> = o.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let (mut pairs, mut right) = (0, 0);
        for a in 0..s.len() {
            for b in a + 1..s.len() {
                pairs += 1;
                right += (pos[&a] < pos[&b]) as usize;
            }
        }
        (pairs, right, o.iter().enumerate().all(|(k, &i)| k == i) && s.len() <= 10)
    });
    let seen: std::collections::HashSet<(String, String)> = train.iter().flatten().map(|w| (w.lemma.clone(), w.feats.clone())).collect();
    let results: Vec<(usize, usize, usize, bool, usize, usize, usize, usize, usize)> = par_map(&test, |s| {
        let o = sp.order(s, &small);
        let pos: HashMap<usize, usize> = o.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let n = s.len();
        let (mut pairs, mut right) = (0, 0);
        for a in 0..n {
            for b in a + 1..n {
                pairs += 1;
                right += (pos[&a] < pos[&b]) as usize;
            }
        }
        let whole = o.iter().enumerate().all(|(k, &i)| k == i);
        let (mut bent, mut bent_right, mut unseen, mut unseen_right, mut as_is) = (0, 0, 0, 0, 0);
        for w in s {
            let got = sp.bend(w);
            bent += 1;
            bent_right += (got == w.form) as usize;
            as_is += (w.lemma == w.form) as usize;
            if !seen.contains(&(w.lemma.clone(), w.feats.clone())) {
                unseen += 1;
                unseen_right += (got == w.form) as usize;
            }
        }
        (n, pairs, right, whole, bent, bent_right, unseen, unseen_right, as_is)
    });
    let sum = |f: fn(&(usize, usize, usize, bool, usize, usize, usize, usize, usize)) -> usize| results.iter().map(f).sum::<usize>();
    let p = |a: usize, b: usize| 100.0 * a as f32 / b.max(1) as f32;
    let short: Vec<_> = results.iter().filter(|r| r.0 <= 10).collect();
    println!("order (test, {} sentences): word pairs in the right order {:.1}% (chance 50%); whole sentences right {:.1}%; of sentences of 10 words or fewer ({}): {:.1}%", results.len(), p(sum(|r| r.2), sum(|r| r.1)), p(results.iter().filter(|r| r.3).count(), results.len()), short.len(), p(short.iter().filter(|r| r.3).count(), short.len()));
    println!("  yardstick (each role's commonest place, counted): word pairs {:.1}%; sentences of 10 words or fewer {:.1}%", p(yard.iter().map(|x| x.1).sum(), yard.iter().map(|x| x.0).sum()), p(yard.iter().filter(|x| x.2).count(), short.len()));
    println!("bending (test, {} words): right {:.1}% (the base form as it is {:.1}%); base form and features never seen together ({}): {:.1}%", sum(|r| r.4), p(sum(|r| r.5), sum(|r| r.4)), p(sum(|r| r.8), sum(|r| r.4)), sum(|r| r.6), p(sum(|r| r.7), sum(|r| r.6)));
    println!("all in {:.1} s", t0.elapsed().as_secs_f64());
}
