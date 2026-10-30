//! The swim and the egg.
//!
//! The sperm: every way to read the sentence as (agent, action, patient), three different words of it. Each swims
//! with the strength memory gives it: how fit each word is for its role (its own role memory and its neighbourhood),
//! how often that order of roles was seen and how far apart the roles stood, all read by interference.
//! Without the egg (stage 3) the strongest sperm is the answer.
//!
//! The egg, three layers, each dearer than the one before and each one sifting:
//!   1. membrane: each word of a sperm is held against what the core knows of it (its role memory and
//!      neighbourhood); if any word fits its role worse than the learned threshold (set against chance: random
//!      readings of training sentences), the sperm does not get in;
//!   2. competition: a word with more than one sense has a sperm per sense; the sense whose context vector the
//!      sentence's words reinforce most wins (interference), and every candidate's score is logged;
//!   3. fertilisation: the winner is merged with what memory recalls of it, and Darma tests itself: does each role
//!      unbind back to its own word, with a margin over the runner-up above the learned threshold? If not, the next
//!      sperm is tried. If none passes, the answer is "en ymmärrä" (I do not understand).

use crate::lexicon::{tokens, Atoms, Lexicon};
use crate::memory::{normalised, order_of, structure, Memory};
use crate::vector::HVec;

pub const TOP: usize = 30;
pub struct Sperm {
    pub fill: [String; 3],
    pub pos: [usize; 3],
    pub swim: f32,
}

pub struct SenseChoice {
    pub word: String,
    pub scores: Vec<(String, f32)>,
    pub winner: String,
}

pub struct Path {
    pub sperm: Vec<Sperm>,
    pub kept_out: Vec<(String, f32)>,
    pub senses: Vec<SenseChoice>,
    pub tried: Vec<(String, bool, f32)>,
    pub answer: Option<[String; 3]>,
}

/// all readings of the sentence, strongest first (the swim)
pub fn swim(m: &Memory, at: &Atoms, words: &[String]) -> Vec<Sperm> {
    swim_within(m, at, words, usize::MAX)
}

/// the swim, with the roles at most `maxd` words apart (for reading long text quickly)
pub fn swim_within(m: &Memory, at: &Atoms, words: &[String], maxd: usize) -> Vec<Sperm> {
    let n = words.len();
    let fit: Vec<[f32; 4]> = (0..n).map(|i| m.fitness(at, words, i)).collect();
    let total = m.structures.len().max(1) as f32;
    let ord: Vec<f32> = crate::memory::ORDERS.iter().map(|o| (m.order_weight(at, o) / total).max(1e-3)).collect();
    let mut out = Vec::new();
    for a in 0..n {
        for v in 0..n {
            for p in 0..n {
                if a == v || v == p || a == p || words[a] == words[v] || words[v] == words[p] || words[a] == words[p] || a.abs_diff(v) > maxd || v.abs_diff(p) > maxd {
                    continue;
                }
                let o = crate::memory::ORDERS.iter().position(|x| *x == order_of([a, v, p])).unwrap();
                let lnp = |x: f32| x.max(1e-3).ln();
                let swim = lnp(fit[a][0]) + lnp(fit[v][1]) + lnp(fit[p][2]) + ord[o].ln()
                    + m.dist_share(at, 0, a.abs_diff(v)).ln() + m.dist_share(at, 1, v.abs_diff(p)).ln();
                out.push(Sperm { fill: [words[a].clone(), words[v].clone(), words[p].clone()], pos: [a, v, p], swim });
            }
        }
    }
    out.sort_by(|x, y| y.swim.partial_cmp(&x.swim).unwrap());
    out
}

/// competition between a word's senses: which sense's context the sentence reinforces (interference)
pub fn choose_sense(m: &Memory, at: &Atoms, lex: &Lexicon, words: &[String], i: usize) -> Option<SenseChoice> {
    let senses = lex.senses.get(&words[i])?;
    let c = m.context(at, words, i);
    let mut scores: Vec<(String, f32)> = senses.iter().map(|s| (s.clone(), m.contexts.get(s).map_or(0.0, |v| c.sim(v)))).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Some(SenseChoice { word: words[i].clone(), winner: scores[0].0.clone(), scores })
}

pub fn understand(m: &Memory, at: &Atoms, lex: &Lexicon, sentence: &str) -> Path {
    let words = tokens(sentence);
    let all = swim(m, at, &words);
    let sperm: Vec<Sperm> = all.into_iter().take(TOP).collect();
    let mut path = Path { sperm: Vec::new(), kept_out: Vec::new(), senses: Vec::new(), tried: Vec::new(), answer: None };
    // 2. competition for every word with senses (logged; the sense becomes part of the understanding)
    for i in 0..words.len() {
        if let Some(c) = choose_sense(m, at, lex, &words, i) {
            path.senses.push(c);
        }
    }
    // 1. membrane
    let mut inside: Vec<(usize, HVec)> = Vec::new();
    for (k, s) in sperm.iter().enumerate() {
        let st = normalised(structure(at, &s.fill));
        // compatible with what the core knows: every word fits the role it is given, better than chance does
        let c = m.fitness(at, &words, s.pos[0])[0].min(m.fitness(at, &words, s.pos[1])[1]).min(m.fitness(at, &words, s.pos[2])[2]);
        if c >= m.membrane {
            inside.push((k, st));
        } else {
            path.kept_out.push((s.fill.join(" / "), c));
        }
    }
    // 3. fertilisation, the strongest first
    for (k, st) in &inside {
        let (_, margin) = m.read_back(at, st, &sperm[*k].fill, &words, None);
        let pass = margin >= m.fertile;
        path.tried.push((sperm[*k].fill.join(" / "), pass, margin));
        if pass {
            path.answer = Some(sperm[*k].fill.clone());
            break;
        }
    }
    path.sperm = sperm;
    path
}

/// learning by reading: Darma reads plain text (no marks), sentence by sentence; where the strongest reading beats
/// the next by at least the learned margin and gets through the membrane, its pairs (who did what, what was done to
/// whom) are struck into memory. The margin is learned on the marked training sentences: the smallest margin at which
/// at least `precision` of the readings it lets through are right. Returns (sentences read, clauses struck, margin).
pub fn learn_by_reading(m: &mut Memory, at: &Atoms, clauses: &[crate::memory::Clause], text: &[String], precision: f32) -> (usize, usize, f32) {
    const MAXD: usize = 4;
    // the margin, from the training sentences whose clauses people marked
    let mut by: std::collections::BTreeMap<&str, Vec<[String; 3]>> = std::collections::BTreeMap::new();
    for c in clauses {
        by.entry(&c.sentence).or_default().push(c.gold());
    }
    let mut seen: Vec<(f32, bool)> = Vec::new();
    for (s, golds) in &by {
        let sw = swim_within(m, at, &tokens(s), MAXD);
        if sw.len() >= 2 {
            seen.push((sw[0].swim - sw[1].swim, golds.contains(&sw[0].fill)));
        }
    }
    seen.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (mut right, mut margin) = (0usize, f32::MAX);
    for (k, (mg, ok)) in seen.iter().enumerate() {
        right += *ok as usize;
        if k >= 20 && right as f32 / (k + 1) as f32 >= precision {
            margin = *mg;
        }
    }
    let (mut read, mut struck) = (0, 0);
    for para in text {
        for sent in para.split(['.', '!', '?']) {
            let words = tokens(sent);
            if words.len() < 3 || words.len() > 40 {
                continue;
            }
            read += 1;
            let sw = swim_within(m, at, &words, MAXD);
            if sw.len() < 2 || sw[0].swim - sw[1].swim < margin {
                continue;
            }
            let s = &sw[0];
            // only what it knows: every word of the reading met before in the marked sentences
            if !s.fill.iter().all(|w| m.word_roles.contains_key(w)) {
                continue;
            }
            let f = |r: usize| m.fitness(at, &words, s.pos[r])[r];
            if f(0).min(f(1)).min(f(2)) < m.membrane {
                continue;
            }
            let [a, v, p] = s.fill.clone();
            if std::env::var("DARMA_SHOW_READ").is_ok() && struck < 25 {
                eprintln!("  struck: {a} / {v} / {p}   <- {}", words.join(" "));
            }
            m.agents_of.entry(v.clone()).or_default().push(a.clone());
            m.patients_of.entry(v.clone()).or_default().push(p.clone());
            m.did.entry(a).or_default().push(v.clone());
            m.underwent.entry(p).or_default().push(v);
            struck += 1;
        }
    }
    (read, struck, margin)
}
