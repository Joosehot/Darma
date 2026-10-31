//! What Darma knows of things from reading the dictionary: each definition read for what the word is (the first
//! thing it names, as people marked things in UD: "car: a motor vehicle with four wheels" -> car is motor vehicle),
//! struck into a core of its own (one strike per definition; its own core so the owner's facts stay apart). "What is
//! a car?" is a measurement; "is a car a vehicle?" a chain of measurements (car is what? motor vehicle; vehicle is
//! what? ...), every step shown.
use crate::lexicon::Atoms;
use crate::quantum::{words_of, Core};
use crate::vector::HVec;
use crate::words::Read;
use std::collections::{HashMap, HashSet, VecDeque};

/// the dimension of the dictionary's core (each definition is short: few runs per register)
pub const KNOW_D: usize = 256;
/// how far a chain of "is" may go
pub const CHAIN: usize = 8;
pub const IS: &str = "is";

pub struct Know<'r> {
    r: &'r Read,
    core: Core<'static>,
    /// what each sense's definition says it is, in the definition's own words up to that ("a motor vehicle")
    said: HashMap<String, String>,
    /// how often people marked each word a noun in UD
    nouns: HashMap<String, f32>,
    pub facts: usize,
}

/// a sense's key in the core: the word and the sense's place in the dictionary ("plant#1")
fn key(w: &str, k: usize) -> String {
    format!("{w}#{k}")
}

impl<'r> Know<'r> {
    /// read: every noun sense of the words people used (DailyDialog, UD) and of the words those lead to; each sense
    /// one strike
    pub fn read(r: &'r Read, met: &dyn Fn(&str) -> bool) -> Option<Know<'r>> {
        let at: &'static Atoms = Box::leak(Box::new(Atoms::new(KNOW_D, 7)));
        let mut core = Core::new(at, None, 0.0);
        core.keyed = true;
        let nouns = crate::words::ud_nouns();
        if nouns.is_empty() {
            return None;
        }
        let mut k = Know { r, core, said: HashMap::new(), nouns, facts: 0 };
        let mut queue: VecDeque<String> = r.sense_words().filter(|w| met(w)).cloned().collect();
        queue.make_contiguous().sort();
        let mut seen: HashSet<String> = queue.iter().cloned().collect();
        while let Some(w) = queue.pop_front() {
            let Some(ss) = r.senses(&w) else { continue };
            for (i, d) in ss.iter().enumerate() {
                let Some((phrase, said)) = k.genus(d) else { continue };
                let kw = key(&w, i);
                let mut fact = vec![kw.clone(), IS.to_string()];
                fact.extend(phrase.iter().cloned());
                k.core.tell_keyed(&fact, &[kw.clone()]);
                k.said.insert(kw, said);
                k.facts += 1;
                for x in phrase {
                    if k.thing(&x) && seen.insert(x.clone()) {
                        queue.push_back(x);
                    }
                }
            }
        }
        // a measurement's winner must stand out (as the chat core: set by hand)
        k.core.threshold = 0.5;
        k.core.fid_threshold = 3.0;
        Some(k)
    }

    /// a thing: a word people marked a noun in UD at least half the time; a word UD never had is taken for a thing
    /// (rare words are mostly names of things; a rule set by hand)
    fn thing(&self, w: &str) -> bool {
        match self.nouns.get(w) {
            Some(s) => *s >= 0.5,
            None => w.chars().all(|c| c.is_alphabetic()),
        }
    }

    /// the heads of what a sense is: the last thing of each run of things ("motor vehicle" -> vehicle; "member of genus
    /// canis" -> member, canis): a thing before a thing describes it (as in the Composer)
    fn heads(&self, a: &str) -> Vec<String> {
        let w: Vec<&str> = a.split(' ').collect();
        (0..w.len()).filter(|&i| self.thing(w[i]) && !w.get(i + 1).is_some_and(|n| self.thing(n))).map(|i| w[i].to_string()).collect()
    }

    /// what a definition (as written) says the word is: the first run of things in it (within its first six words),
    /// with "of" and the run of things after it ("member of the genus canis"), in base forms; and the definition's
    /// words up to there
    fn genus(&self, d: &[String]) -> Option<(Vec<String>, String)> {
        let i = (0..d.len().min(6)).find(|&i| self.thing(&d[i]))?;
        let mut j = i;
        while j + 1 < d.len() && self.thing(&d[j + 1]) {
            j += 1;
        }
        if d.get(j + 1).is_some_and(|x| x == "of") {
            let s = j + 2 + d.get(j + 2).is_some_and(|x| !self.thing(x)) as usize;
            if s < d.len() && self.thing(&d[s]) {
                j = s;
                while j + 1 < d.len() && self.thing(&d[j + 1]) {
                    j += 1;
                }
            }
        }
        Some((d[i..=j].iter().filter(|x| self.thing(x) || *x == "of").map(|x| self.r.base(x)).collect(), d[..=j].join(" ")))
    }

    /// the sense of a word meant here: the one whose definition resonates most with what led to it (the context);
    /// without a context, the most used. Returns (the word in base form, the sense's place)
    pub fn sense(&self, w: &str, ctx: Option<&HVec>) -> Option<(String, usize)> {
        let w = if self.r.senses(w).is_some() { w.to_string() } else { self.r.base(w) };
        let ss = self.r.senses(&w)?;
        // another sense than the most used is taken only when it resonates clearly more: by three noise deviations
        // of a wave's likeness (1 / sqrt(D)), as a measurement must stand out
        let k = match ctx {
            None => 0,
            Some(c) => {
                let s: Vec<f32> = ss.iter().map(|d| self.r.wave(d).sim(c)).collect();
                let best = (0..s.len()).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap().then(b.cmp(&a))).unwrap_or(0);
                if s[best] - s[0] >= 3.0 / (crate::words::WORDS_D as f32).sqrt() { best } else { 0 }
            }
        };
        Some((w, k))
    }

    /// what a sense is: the core's answer (a measurement)
    fn is_what(&self, w: &str, k: usize) -> Option<String> {
        let q = vec![key(w, k), IS.to_string(), "what".to_string()];
        self.core.answer(&q, &self.core.candidates_for(&q)).0
    }

    /// "what is X" (its most used sense): the core's answer, and the definition's words for saying it
    pub fn what_is(&self, x: &str) -> Option<(String, String)> {
        let (w, k) = self.sense(x, None)?;
        let a = self.is_what(&w, k)?;
        Some((a, self.said.get(&key(&w, k)).cloned().unwrap_or_default()))
    }

    /// "is X a Y": X in the sense that resonates with Y, then what it is, by measurements, each word met taken in the
    /// sense its naming definition resonates with, until Y is met (the chain) or nothing more is known within CHAIN
    /// steps (None: it does not know)
    pub fn is_a(&self, x: &str, y: &str) -> Option<Vec<String>> {
        let y = self.r.base(y);
        let ctx = crate::words::Meaning::meaning(self.r, &y)?;
        let start = self.sense(x, Some(&ctx))?;
        let mut from: HashMap<(String, usize), ((String, usize), String)> = HashMap::new();
        let mut seen: HashSet<String> = HashSet::from([start.0.clone()]);
        let mut front = vec![start.clone()];
        for _ in 0..CHAIN {
            let mut next = Vec::new();
            for (w, k) in &front {
                let Some(a) = self.is_what(w, *k) else { continue };
                let step = format!("{w} ({}) is {a}", k + 1);
                let here = self.r.wave(&self.r.senses(w).map(|s| s[*k].clone()).unwrap_or_default());
                for z in self.heads(&a) {
                    if z == y {
                        let mut chain = vec![step.clone()];
                        let mut at = (w.clone(), *k);
                        while let Some((prev, s)) = from.get(&at) {
                            chain.push(s.clone());
                            at = prev.clone();
                        }
                        chain.reverse();
                        return Some(chain);
                    }
                    if seen.insert(z.clone()) {
                        if let Some(zk) = self.sense(&z, Some(&here)) {
                            from.insert(zk.clone(), ((w.clone(), *k), step.clone()));
                            next.push(zk);
                        }
                    }
                }
            }
            front = next;
            if front.is_empty() {
                break;
            }
        }
        None
    }
}

/// `darma know-test`: fixed questions written before the first run, against definitions shuffled (control)
pub fn test(seed: u64) {
    let t0 = std::time::Instant::now();
    let Some(r) = Read::read(seed) else { return println!("no dictionary") };
    let met = met_words();
    let what = ["dog", "car", "cat", "apple", "house", "teacher", "computer", "hello", "money", "music"];
    let yes = [("car", "vehicle"), ("dog", "animal"), ("cat", "animal"), ("apple", "fruit"), ("teacher", "person"), ("house", "building"), ("rose", "flower"), ("oak", "tree"), ("bus", "vehicle"), ("doctor", "person")];
    let no = [("car", "animal"), ("dog", "fruit"), ("apple", "person"), ("house", "animal"), ("teacher", "vehicle"), ("rose", "building"), ("bus", "flower"), ("oak", "person"), ("doctor", "fruit"), ("cat", "tree")];
    let shuffled = Read::read(seed).unwrap().shuffled();
    for (name, r) in [("read", &r), ("definitions shuffled (control)", &shuffled)] {
        let t = std::time::Instant::now();
        let Some(k) = Know::read(r, &met) else { return println!("data/heads.json missing") };
        println!("\n{name}: {} definitions struck in {:.1} s", k.facts, t.elapsed().as_secs_f64());
        for w in what {
            match k.what_is(w) {
                Some((a, said)) => println!("  what is a {w}? {a}   (read: \"{said}\")"),
                None => println!("  what is a {w}? I don't know."),
            }
        }
        let y = yes.iter().filter(|(a, b)| {
            let c = k.is_a(a, b);
            println!("  is a {a} a {b}? {}", c.as_ref().map_or("not as far as I know".to_string(), |c| format!("yes: {}", c.join("; "))));
            c.is_some()
        }).count();
        let n = no.iter().filter(|(a, b)| { let c = k.is_a(a, b); if let Some(c) = &c { println!("  WRONG yes: is a {a} a {b}? {}", c.join("; ")); } c.is_some() }).count();
        println!("  yes on true pairs {y} of {}; yes on false pairs {n} of {}", yes.len(), no.len());
    }
    println!("\nall in {:.1} s", t0.elapsed().as_secs_f64());
}

/// the words people used: DailyDialog's and UD's
pub fn met_words() -> impl Fn(&str) -> bool {
    let mut m: HashSet<String> = HashSet::new();
    for p in ["data/raw/dailydialog/train.json"] {
        if let Ok(s) = std::fs::read_to_string(p) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                let mut stack = vec![&v];
                while let Some(x) = stack.pop() {
                    match x {
                        serde_json::Value::String(t) => m.extend(words_of(t)),
                        serde_json::Value::Array(a) => stack.extend(a.iter()),
                        serde_json::Value::Object(o) => stack.extend(o.values()),
                        _ => {}
                    }
                }
            }
        }
    }
    m.extend(crate::words::ud_lemmas().into_keys());
    move |w: &str| m.contains(w)
}

/// `darma know-show <word>...`: how Darma read each sense of each word
pub fn show(seed: u64, words: &[String]) {
    let r = Read::read(seed).unwrap();
    let k = Know::read(&r, &met_words()).unwrap();
    for w in words {
        println!("{w}: thing {} (UD noun share {:?})", k.thing(w), k.nouns.get(w.as_str()));
        for (i, d) in r.senses(w).unwrap_or(&[]).iter().enumerate() {
            println!("  {}. {}  -> is {:?}", i + 1, d.join(" "), k.is_what(w, i));
        }
    }
}
