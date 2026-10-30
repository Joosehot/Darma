//! The swim and the egg.
//!
//! The sperm: every way to read the sentence as (agent, action, patient), three different words of it. Each swims
//! with the strength memory gives it: how often each word filled its role (read from the role bundles) and how
//! often that order of roles was seen (read from the order bundle), all by interference with the stored vectors.
//! Without the egg (stage 3) the strongest sperm is the answer.
//!
//! The egg, three layers, each dearer than the one before and each one sifting:
//!   1. membrane: a sperm's structure is held against every stored structure; if it is less like the most alike one
//!      than the learned threshold, it does not get in;
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
    let n = words.len();
    let w: Vec<[f32; 3]> = words.iter().map(|x| [0, 1, 2].map(|r| m.role_weight(at, x, r))).collect();
    let ord: Vec<f32> = crate::memory::ORDERS.iter().map(|o| m.order_weight(at, o)).collect();
    let mut out = Vec::new();
    for a in 0..n {
        for v in 0..n {
            for p in 0..n {
                if a == v || v == p || a == p || words[a] == words[v] || words[v] == words[p] || words[a] == words[p] {
                    continue;
                }
                let o = crate::memory::ORDERS.iter().position(|x| *x == order_of([a, v, p])).unwrap();
                let swim = (1.0 + w[a][0]).ln() + (1.0 + w[v][1]).ln() + (1.0 + w[p][2]).ln() + (1.0 + ord[o]).ln();
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
        let c = m.structures.iter().map(|x| st.sim(&x.0)).fold(f32::MIN, f32::max);
        if c >= m.membrane {
            inside.push((k, st));
        } else {
            path.kept_out.push((s.fill.join(" / "), c));
        }
    }
    // 3. fertilisation, the strongest first
    for (k, st) in &inside {
        let (ok, margin) = m.read_back(at, st, &sperm[*k].fill, &words, None);
        let pass = ok && margin >= m.fertile;
        path.tried.push((sperm[*k].fill.join(" / "), pass, margin));
        if pass {
            path.answer = Some(sperm[*k].fill.clone());
            break;
        }
    }
    path.sperm = sperm;
    path
}
