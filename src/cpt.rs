//! Conscious parallel thinking (the owner's name for it): an answer made new from understanding what was meant, not
//! guessed and not taken from anywhere. v0: what was said is heard into a structure (the grammar's classes and roles,
//! hearing's tree), and so is everything Darma knows (each remembered sentence heard the same way); thoughts then work
//! on the structures side by side, and the strongest are put together into what Darma says.
//!   recall:  which known sentence fills the question's structure: every part of the question but its hole found in
//!            it, in the same place of the tree (the answer is that sentence);
//!   nearest: when nothing fills it, what is known of the same one in the same way ("are you an AI?": what I am);
//! The workspace that chooses among them is in qa.rs (`Talk::think`).
use crate::hear::Hear;
use crate::speak::Speak;
use crate::syntax::{Syntax, Tok};
use std::collections::{HashMap, HashSet};

/// the question words: a hole in what is asked
pub const WH: [&str; 7] = ["what", "who", "where", "when", "why", "how", "which"];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub word: String,
    pub lemma: String,
    pub pos: String,
    pub rel: String,
    /// 1-based; 0: the sentence's core
    pub head: usize,
}

pub struct Cpt {
    pub sx: Syntax,
    pub hear: Hear,
    #[allow(dead_code)]
    pub speak: Speak,
    pub lemma: HashMap<(String, String), String>,
    pub small: HashSet<String>,
    cache: std::sync::Mutex<HashMap<String, Vec<Node>>>,
}

/// base forms as people marked them in UD, by word and class (the commonest lemma of each written form as that class:
/// "ai" as a verb is "be", from "ain't"; as a noun it is "ai")
fn ud_lemmas() -> HashMap<(String, String), String> {
    let mut c: HashMap<(String, String), HashMap<String, u32>> = HashMap::new();
    for name in ["data/raw/ud/en_ewt-ud-train.conllu", "data/raw/ud/en_ewt-ud-dev.conllu"] {
        for line in std::fs::read_to_string(name).unwrap_or_default().lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 4 && !f[0].contains('-') && !f[0].contains('.') {
                *c.entry((f[1].to_lowercase(), f[3].to_string())).or_default().entry(f[2].to_lowercase()).or_default() += 1;
            }
        }
    }
    c.into_iter().filter_map(|(w, m)| m.into_iter().max_by_key(|x| (x.1, x.0.clone())).map(|x| (w, x.0))).collect()
}

impl Cpt {
    /// the grammar, hearing and speech, struck before (or struck now and saved)
    pub fn load(seed: u64) -> Option<Cpt> {
        let train = crate::syntax::load("data/raw/ud/en_ewt-ud-train.conllu");
        if train.is_empty() {
            return None;
        }
        let (sx, _) = crate::syntax::strike_or_load(seed, 3, &train, None);
        let hear = Hear::strike_or_load(seed, 3, &sx);
        let (speak, _) = Speak::strike_or_load(seed, 3);
        let small = crate::hear::small_words(&train);
        Some(Cpt { sx, hear, speak, lemma: ud_lemmas(), small, cache: Default::default() })
    }

    /// a sentence heard: each word's class, role and core word. It is heard as UD's sentences are written: "I" with
    /// its capital and the sentence's mark at its end ("?" for a question), the mark then left out of the nodes
    pub fn hear(&self, w: &[String]) -> Vec<Node> {
        self.hear_as(w, ".")
    }

    pub fn hear_as(&self, w: &[String], mark: &str) -> Vec<Node> {
        let key = format!("{} {mark}", w.join(" "));
        if let Some(n) = self.cache.lock().unwrap().get(&key) {
            return n.clone();
        }
        let nodes = self.hear_fresh(w, mark);
        self.cache.lock().unwrap().insert(key, nodes.clone());
        nodes
    }

    /// a sentence heard without keeping it (for reading many at once, on every core)
    pub fn hear_fresh(&self, w: &[String], mark: &str) -> Vec<Node> {
        let mut toks: Vec<Tok> = w.iter().map(|x| Tok { word: x.clone(), cap: x == "i", pos: String::new(), rel: String::new() }).collect();
        toks.push(Tok { word: mark.to_string(), cap: false, pos: String::new(), rel: String::new() });
        let mut w: Vec<String> = w.to_vec();
        w.push(mark.to_string());
        let tags = self.sx.read_classes(&toks);
        let roles = self.sx.read_roles(&toks, &tags);
        let classes: Vec<String> = tags.iter().map(|&k| self.sx.pos_names[k].clone()).collect();
        let heads = if w.is_empty() { Vec::new() } else { self.hear.heads(&w, &classes) };
        let n = w.len() - 1;
        let nodes: Vec<Node> = (0..n).map(|i| Node { word: w[i].clone(), lemma: self.lemma.get(&(w[i].clone(), classes[i].clone())).cloned().unwrap_or_else(|| w[i].clone()), pos: classes[i].clone(), rel: self.sx.rel_names[roles[i]].clone(), head: if heads[i] > n { 0 } else { heads[i] } }).collect();
        nodes
    }

    /// how a heard sentence reads, for showing: each word with its role and its core word
    pub fn show(nodes: &[Node]) -> String {
        nodes.iter().map(|n| if n.head == 0 { format!("{}[core]", n.word) } else { format!("{}[{}→{}]", n.word, n.rel, nodes[n.head - 1].word) }).collect::<Vec<_>>().join(" ")
    }
}

/// a part of a heard sentence: where it stands in the tree (its role under the core, and its own role below that)
/// and its base form; with its place
fn parts(n: &[Node]) -> Vec<(String, String, usize)> {
    let top = |mut i: usize| -> usize {
        while n[i].head != 0 && n[n[i].head - 1].head != 0 {
            i = n[i].head - 1;
        }
        i
    };
    (0..n.len())
        .filter(|&i| !matches!(n[i].rel.as_str(), "punct" | "det"))
        .map(|i| {
            let key = if n[i].head == 0 {
                "core".to_string()
            } else if n[n[i].head - 1].head == 0 {
                n[i].rel.clone()
            } else {
                format!("{}/{}", n[top(i)].rel, n[i].rel)
            };
            (key, n[i].lemma.clone(), i)
        })
        .collect()
}

/// the hole of a question: its question word's place
pub fn hole(n: &[Node]) -> Option<usize> {
    n.iter().position(|x| WH.contains(&x.lemma.as_str()))
}

/// the words a question asks with: all but its hole, its marks, and a "do" that only carries the question (English
/// grammar: "do" stands for any doing, a rule set by hand)
pub fn asked_with(q: &[Node]) -> Vec<&str> {
    let h = hole(q);
    // (an article says nothing of its own: "a dog" and "dogs" ask the same)
    q.iter().enumerate().filter(|(i, x)| Some(*i) != h && x.lemma != "do" && !matches!(x.word.as_str(), "a" | "an" | "the") && x.lemma.chars().any(|c| c.is_alphanumeric())).map(|(_, x)| x.lemma.as_str()).collect()
}

/// does a known sentence answer a question, and how well its structure fits (for choosing among those that do)?
/// It answers when every word the question asks with is in it (hearing errs, so the words decide: an answer not about
/// what was asked would claim what is not so), the doer is the same one when both have one, and, for a question with
/// a hole, it says something more (what fills the hole). The fit: the share of the question's parts found at the
/// same place of the sentence's tree
pub fn fit(q: &[Node], f: &[Node]) -> Option<f32> {
    let fl: HashSet<&str> = f.iter().map(|x| x.lemma.as_str()).collect();
    let asked = asked_with(q);
    if asked.is_empty() || !asked.iter().all(|w| fl.contains(w)) {
        return None;
    }
    if let (Some(a), Some(b)) = (doer(q), doer(f)) {
        if a != b && !WH.contains(&a.as_str()) {
            return None;
        }
    }
    // for a question with a hole: something more, of the kind its question word asks for (English grammar, set by
    // hand: how - a state or a manner; who - someone; what - a thing or a doing; where - a place; when - a time;
    // why - a reason)
    let ql: HashSet<&str> = q.iter().map(|x| x.lemma.as_str()).collect();
    if let Some(h) = hole(q) {
        // (who and what: anything but a word that only joins or points; a name Darma has never met is read as anything)
        let joins = ["DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PUNCT"];
        let kind = |p: &str| -> bool {
            match q[h].lemma.as_str() {
                "how" => matches!(p, "ADJ" | "ADV"),
                "who" | "what" | "which" => !joins.contains(&p),
                "where" => matches!(p, "ADV" | "ADP" | "NOUN" | "PROPN"),
                "when" => matches!(p, "ADV" | "NUM" | "NOUN" | "ADP"),
                _ => matches!(p, "SCONJ" | "ADP" | "VERB" | "NOUN" | "ADJ"),
            }
        };
        // the filler stands at the sentence's core or right under it (not inside a name: "resonant" in "a resonant
        // quantum-spirited model" says no how)
        let near_core = |x: &Node| x.head == 0 || f[x.head - 1].head == 0;
        // how one is is not what one is
        if q[h].lemma == "how" && names_a_thing(f) {
            return None;
        }
        if !f.iter().any(|x| !ql.contains(x.lemma.as_str()) && x.lemma.chars().any(|c| c.is_alphanumeric()) && kind(&x.pos) && near_core(x)) {
            return None;
        }
    }
    let qp = parts(q);
    let fp: HashSet<(String, String)> = parts(f).into_iter().map(|(k, l, _)| (k, l)).collect();
    Some(qp.iter().filter(|(k, l, _)| fp.contains(&(k.clone(), l.clone()))).count() as f32 / qp.len().max(1) as f32)
}

/// does a sentence say what something is ("I am a program", "I am Darma": "be" with a thing or a name at its core)?
pub fn names_a_thing(n: &[Node]) -> bool {
    let core = n.iter().position(|x| x.head == 0);
    let thing = |x: &Node| matches!(x.pos.as_str(), "NOUN" | "PROPN");
    // "be a ..." always says what something is (English grammar, set by hand): hearing may read the rest wrong
    let be_a = n.windows(2).any(|p| p[0].lemma == "be" && matches!(p[1].word.as_str(), "a" | "an"));
    be_a || n.iter().any(|x| x.lemma == "be")
        && core.is_some_and(|c| n[c].pos != "VERB" && (thing(&n[c]) || n.iter().any(|x| x.head == c + 1 && thing(x))))
}

/// what is known of the same doer, when nothing answers: for "is it a ...?" what it is known to be (sentences naming
/// what the doer is); for anything else, a sentence of the same doer that shares a word with the question beyond the
/// commonest words. The share of the question's words it holds (None: not near)
pub fn near(q: &[Node], f: &[Node], small: &HashSet<String>) -> Option<f32> {
    let (a, b) = (doer(q)?, doer(f)?);
    if a != b {
        return None;
    }
    let fl: HashSet<&str> = f.iter().map(|x| x.lemma.as_str()).collect();
    let asked = asked_with(q);
    let share = asked.iter().filter(|w| fl.contains(*w)).count() as f32 / asked.len().max(1) as f32;
    let be_q = q.iter().any(|x| x.lemma == "be") && hole(q).is_none();
    if be_q {
        return names_a_thing(f).then_some(share);
    }
    asked.iter().any(|w| fl.contains(w) && !small.contains(*w) && *w != a).then_some(share)
}

/// the doer of a heard sentence (its subject under the core), if any
pub fn doer(n: &[Node]) -> Option<String> {
    n.iter().find(|x| x.head != 0 && n[x.head - 1].head == 0 && x.rel == "nsubj").map(|x| x.lemma.clone())
}

/// a known sentence said out: capitals where they belong (I, a name as hearing reads it, the first word), and a stop
pub fn say(n: &[Node]) -> String {
    let words: Vec<String> = n
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if x.word == "i" || x.pos == "PROPN" || i == 0 {
                let mut c = x.word.chars();
                c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
            } else {
                x.word.clone()
            }
        })
        .collect();
    format!("{}.", words.join(" "))
}

/// `darma hear <sentence>`: how Darma hears a sentence: each word's class, role and core word
pub fn show_cmd(words: &[String]) {
    let Some(c) = Cpt::load(1) else { return println!("UD not found") };
    let w = crate::quantum::words_of(&words.join(" "));
    for x in c.hear(&w) {
        println!("{:<12} {:<6} {:<8} -> {}", x.word, x.pos, x.rel, if x.head == 0 { "(core)".to_string() } else { w[x.head - 1].clone() });
    }
}
