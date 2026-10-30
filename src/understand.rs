//! Understanding: what a line does (its act) and what it feels (its emotion), read from its state.
//!
//! People marked every turn of DailyDialog with an act (inform, question, directive, commissive) and an emotion (none,
//! anger, disgust, fear, happiness, sadness, surprise). Darma learns them in one pass, its own way: a line's state is
//! the sum of its runs of one and two words (each run the words' phase operators applied in turn, as in the core),
//! each weighed by how rare the run is; the states of all the turns marked with an act are superposed into that
//! act's prototype (and the same for each emotion), with what all turns share taken away. A new line is read by which
//! prototype it resonates with most. Nothing is copied: the marks teach what a line does, not what to answer.

use crate::lexicon::{tokens, Atoms};
use crate::quantum::Core;
use crate::vector::HVec;
use serde::Deserialize;
use std::collections::HashMap;

pub const ACTS: [&str; 4] = ["inform", "question", "directive", "commissive"];
pub const EMOTIONS: [&str; 7] = ["none", "anger", "disgust", "fear", "happiness", "sadness", "surprise"];

#[derive(Deserialize)]
struct Marked {
    dialog: Vec<String>,
    act: Vec<i64>,
    emotion: Vec<i64>,
}

pub struct Understanding<'a> {
    core: Core<'a>,
    idf: HashMap<String, f32>,
    acts: Vec<HVec>,
    emotions: Vec<HVec>,
    /// how common each act and emotion is among people's turns (log of its share): a line is read as the common
    /// kind unless another stands out (a weight PRIOR of it is added to each resonance)
    act_prior: Vec<f32>,
    emotion_prior: Vec<f32>,
    /// the chosen weights of commonness (acts, emotions)
    pub weights: (f32, f32),
}

/// how much the commonness of a kind weighs against its resonance: chosen on DailyDialog's validation split (the
/// test split is only measured), separately for acts and emotions, from these
pub const PRIORS: [f32; 7] = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08];

fn runs(w: &[String]) -> Vec<Vec<String>> {
    let mut r = Vec::new();
    for n in 1..=2 {
        for i in 0..w.len().saturating_sub(n - 1) {
            r.push(w[i..i + n].to_vec());
        }
    }
    r
}

/// a line's understanding: its act and emotion, with how strongly each prototype resonated
pub struct Reading {
    pub act: &'static str,
    pub emotion: &'static str,
    pub act_scores: Vec<(&'static str, f32)>,
    pub emotion_scores: Vec<(&'static str, f32)>,
}

impl<'a> Understanding<'a> {
    fn state(&self, w: &[String]) -> HVec {
        let mut s = HVec::zero(self.core.at.d);
        for r in runs(w) {
            let k = r.join(" ");
            if let Some(&f) = self.idf.get(&k) {
                s.add_scaled(&self.core.run(&r), f);
            }
        }
        crate::memory::normalised(s)
    }

    pub fn learn(at: &'a Atoms, path: &str) -> Option<Self> {
        let data: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let turns: Vec<(Vec<String>, usize, usize)> = data
            .iter()
            .flat_map(|d| d.dialog.iter().zip(&d.act).zip(&d.emotion).map(|((u, a), e)| (tokens(u), (*a as usize).saturating_sub(1).min(3), (*e as usize).min(6))))
            .filter(|t| !t.0.is_empty())
            .collect();
        let mut df: HashMap<String, u32> = HashMap::new();
        for (w, _, _) in &turns {
            let mut seen: Vec<String> = runs(w).into_iter().map(|r| r.join(" ")).collect();
            seen.sort();
            seen.dedup();
            for r in seen {
                *df.entry(r).or_default() += 1;
            }
        }
        let n = turns.len() as f32;
        let idf = df.into_iter().filter(|x| x.1 >= 2).map(|(r, c)| (r, (n / c as f32).ln())).collect();
        let mut u = Understanding { core: Core::new(at, None, 0.0), idf, acts: vec![HVec::zero(at.d); 4], emotions: vec![HVec::zero(at.d); 7], act_prior: vec![0.0; 4], emotion_prior: vec![0.0; 7], weights: (0.02, 0.02) };
        let mut all = HVec::zero(at.d);
        let (mut na, mut ne) = (vec![0f32; 4], vec![0f32; 7]);
        for (w, a, e) in &turns {
            let s = u.state(w);
            u.acts[*a].add(&s);
            u.emotions[*e].add(&s);
            all.add(&s);
            na[*a] += 1.0;
            ne[*e] += 1.0;
        }
        u.act_prior = na.iter().map(|k| (k / n).max(1e-6).ln()).collect();
        u.emotion_prior = ne.iter().map(|k| (k / n).max(1e-6).ln()).collect();
        // each prototype: the mean state of its turns, less the mean state of all turns (what every line shares)
        let mean = |v: &HVec, k: f32| HVec(v.0.iter().map(|c| c / k.max(1.0)).collect());
        let overall = mean(&all, n);
        for (p, k) in u.acts.iter_mut().zip(&na) {
            *p = mean(p, *k);
            p.add_scaled(&overall, -1.0);
        }
        for (p, k) in u.emotions.iter_mut().zip(&ne) {
            *p = mean(p, *k);
            p.add_scaled(&overall, -1.0);
        }
        Some(u)
    }

    pub fn read(&self, w: &[String]) -> Reading {
        let s = self.state(w);
        let score = |ps: &[HVec], pr: &[f32], wt: f32, names: &'static [&'static str]| -> Vec<(&'static str, f32)> {
            let mut v: Vec<(&'static str, f32)> = ps.iter().zip(pr).zip(names).map(|((p, l), n)| (*n, s.sim(p) + wt * l)).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v
        };
        let act_scores = score(&self.acts, &self.act_prior, self.weights.0, &ACTS);
        let emotion_scores = score(&self.emotions, &self.emotion_prior, self.weights.1, &EMOTIONS);
        Reading { act: act_scores[0].0, emotion: emotion_scores[0].0, act_scores, emotion_scores }
    }
}

fn confusion(u: &Understanding, path: &str) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let data: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut conf_a = vec![vec![0usize; 4]; 4];
    let mut conf_e = vec![vec![0usize; 7]; 7];
    for d in &data {
        for ((line, a), e) in d.dialog.iter().zip(&d.act).zip(&d.emotion) {
            let w = tokens(line);
            if w.is_empty() {
                continue;
            }
            let r = u.read(&w);
            let (ga, ge) = ((*a as usize).saturating_sub(1).min(3), (*e as usize).min(6));
            conf_a[ga][ACTS.iter().position(|x| *x == r.act).unwrap()] += 1;
            conf_e[ge][EMOTIONS.iter().position(|x| *x == r.emotion).unwrap()] += 1;
        }
    }
    (conf_a, conf_e)
}

/// (accuracy, mean F1, F1 per class)
fn scores(conf: &[Vec<usize>]) -> (f32, f32, Vec<f32>) {
    let k = conf.len();
    let total: usize = conf.iter().flatten().sum();
    let right: usize = (0..k).map(|i| conf[i][i]).sum();
    let f1s: Vec<f32> = (0..k)
        .map(|i| {
            let tp = conf[i][i] as f32;
            let pred: usize = (0..k).map(|j| conf[j][i]).sum();
            let gold: usize = conf[i].iter().sum();
            let p = if pred > 0 { tp / pred as f32 } else { 0.0 };
            let r = if gold > 0 { tp / gold as f32 } else { 0.0 };
            if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
        })
        .collect();
    (right as f32 / total.max(1) as f32, f1s.iter().sum::<f32>() / k as f32, f1s)
}

/// learn, choose the weights of commonness on the validation split (by mean F1), measure once on the test split
pub fn learn_and_choose<'a>(at: &'a Atoms) -> Option<Understanding<'a>> {
    let mut u = Understanding::learn(at, "data/raw/dailydialog/train_marked.json")?;
    let (mut best_a, mut best_e) = ((0.0f32, -1.0f32), (0.0f32, -1.0f32));
    for &wt in &PRIORS {
        u.weights = (wt, wt);
        let (ca, ce) = confusion(&u, "data/raw/dailydialog/validation_marked.json");
        let (fa, fe) = (scores(&ca).1, scores(&ce).1);
        if fa > best_a.1 {
            best_a = (wt, fa);
        }
        if fe > best_e.1 {
            best_e = (wt, fe);
        }
    }
    u.weights = (best_a.0, best_e.0);
    Some(u)
}

/// `darma understand-eval`: learned on DailyDialog's train split, the weights of commonness chosen on its validation
/// split, measured on its test split
pub fn eval(dim: usize, seed: u64) {
    let at = Atoms::new(dim, seed);
    let t0 = std::time::Instant::now();
    let Some(u) = learn_and_choose(&at) else { return println!("download DailyDialog first") };
    println!("learned in one pass and chose the weights on validation in {:.1} s, D {dim}: commonness weighs {} for acts, {} for emotions", t0.elapsed().as_secs_f64(), u.weights.0, u.weights.1);
    let (ca, ce) = confusion(&u, "data/raw/dailydialog/test_marked.json");
    for (name, conf, names) in [("acts", &ca, &ACTS[..]), ("emotions", &ce, &EMOTIONS[..])] {
        let (acc, mf, f1s) = scores(conf);
        let k = names.len();
        let total: usize = conf.iter().flatten().sum();
        let majority = conf.iter().map(|r| r.iter().sum::<usize>()).max().unwrap() as f32 / total as f32;
        let mf_major = 2.0 * majority / (1.0 + majority) / k as f32;
        println!("{name} (test): accuracy {:.1}% (the commonest alone {:.1}%), mean F1 {:.3} (the commonest alone {:.3})", 100.0 * acc, 100.0 * majority, mf, mf_major);
        println!("  F1 per class:{}", names.iter().zip(&f1s).zip(conf.iter()).map(|((n, f), r)| format!("  {n} {f:.2} (n {})", r.iter().sum::<usize>())).collect::<String>());
    }
}
