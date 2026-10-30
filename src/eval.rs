//! The measures (`darma eval`) and the baseline (bag of words + most common sense).
//!
//! Roles are measured on UD EWT's test sentences, read whole: a sentence counts as right when Darma's
//! (agent, action, patient) is one of the clauses people marked in it. The clause test (the three words alone, in
//! their order) and the nonsense test (the same three words reversed: "slogans chanted crowds") measure acceptance
//! and rejection on the same footing. Senses are measured on CoarseWSD-20's test sentences.

use crate::egg::{choose_sense, swim, understand};
use crate::lexicon::{tokens, Atoms, Lexicon};
use crate::memory::{normalised, structure, Clause, Memory, SenseCase};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

fn load<T: serde::de::DeserializeOwned>(p: &str) -> T {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|_| panic!("{p}: run prepare-data first"))).unwrap()
}

pub struct Data {
    pub train: Vec<Clause>,
    pub test: Vec<Clause>,
    pub senses_train: Vec<SenseCase>,
    pub senses_test: Vec<SenseCase>,
    pub lex: Lexicon,
}

pub fn data() -> Data {
    Data {
        train: load("data/train.json"),
        test: load("data/test.json"),
        senses_train: load("data/senses_train.json"),
        senses_test: load("data/senses_test.json"),
        lex: Lexicon::load("data/lexicon.json"),
    }
}

pub fn train(at: &Atoms, d: &Data) -> (Memory, f64) {
    let t = Instant::now();
    let m = Memory::train(at, &d.train, &d.senses_train, 5, 0.05);
    (m, t.elapsed().as_secs_f64())
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

pub fn run(dim: usize, seed: u64) {
    let d = data();
    let at = Atoms::new(dim, seed);
    let (m, train_s) = train(&at, &d);
    println!("trained in one pass: {} clauses, {} sense sentences, {:.2} s; membrane threshold {:.3}, fertilisation margin threshold {:.3}", d.train.len(), d.senses_train.len(), train_s, m.membrane, m.fertile);

    // the test sentences, read whole (several marked clauses may share one sentence)
    let mut by_sentence: BTreeMap<&str, Vec<[String; 3]>> = BTreeMap::new();
    for c in &d.test {
        by_sentence.entry(&c.sentence).or_default().push(c.gold());
    }
    let (mut right3, mut right_egg, mut said, mut base_right) = (0, 0, 0, 0);
    let mut micros = Vec::new();
    for (s, golds) in &by_sentence {
        let words = tokens(s);
        if let Some(best) = swim(&m, &at, &words).first() {
            right3 += golds.contains(&best.fill) as usize;
        }
        let t = Instant::now();
        let p = understand(&m, &at, &d.lex, s);
        micros.push(t.elapsed().as_micros() as f64);
        if let Some(a) = p.answer {
            said += 1;
            right_egg += golds.contains(&a) as usize;
        }
        // baseline: for each role, the sentence's word most often seen in that role (counts, no order, no vectors)
        let pick = |r: usize, not: &[&String]| words.iter().filter(|w| !not.contains(w)).max_by_key(|w| m.role_count[r].get(*w).copied().unwrap_or(0)).cloned();
        if let Some(v) = pick(1, &[]) {
            if let Some(a) = pick(0, &[&v]) {
                if let Some(p) = pick(2, &[&v, &a]) {
                    base_right += golds.contains(&[a, v, p]) as usize;
                }
            }
        }
    }
    let n = by_sentence.len();

    // clauses alone and their reversals (nonsense)
    let (mut clause_ok, mut nonsense_rejected, mut base_ns_rejected) = (0, 0, 0);
    let nonsense: Vec<String> = d.test.iter().map(|c| {
        let g = c.gold();
        format!("{} {} {}", g[2], g[1], g[0])
    }).take(30).collect();
    let (mut clause_right, mut clause_known) = (0, 0);
    for c in &d.test {
        let a = understand(&m, &at, &d.lex, &c.text).answer;
        clause_ok += a.is_some() as usize;
        clause_right += (a == Some(c.gold())) as usize;
        clause_known += c.gold().iter().enumerate().all(|(r, w)| m.role_count[r].contains_key(w)) as usize;
    }
    for s in &nonsense {
        nonsense_rejected += understand(&m, &at, &d.lex, s).answer.is_none() as usize;
        // the baseline rejects only when a word was never seen in the role its position gives
        let w = tokens(s);
        base_ns_rejected += (0..3).any(|r| !m.role_count[r].contains_key(&w[r])) as usize;
    }

    // order: the same three words, the roles swapped
    let mut order_sims = Vec::new();
    for c in &d.test {
        let g = c.gold();
        let a = normalised(structure(&at, &g));
        let b = normalised(structure(&at, &[g[2].clone(), g[1].clone(), g[0].clone()]));
        order_sims.push(a.sim(&b));
    }
    let order_mean = order_sims.iter().sum::<f32>() / order_sims.len() as f32;
    let order_below = order_sims.iter().filter(|&&x| x < m.membrane).count();

    // senses
    let (mut sense_right, mut base_sense_right) = (0, 0);
    let mut per_word: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let most: HashMap<&String, &String> = m.sense_count.iter().map(|(w, c)| (w, c.iter().max_by_key(|x| x.1).unwrap().0)).collect();
    for s in &d.senses_test {
        let words: Vec<String> = s.sentence.split(' ').map(String::from).collect();
        let mut lw = words.clone();
        lw[s.index] = s.word.clone();
        let got = choose_sense(&m, &at, &d.lex, &lw, s.index).map(|c| c.winner);
        let ok = got.as_deref() == Some(s.sense.as_str());
        let bok = most.get(&s.word).map(|x| **x == s.sense).unwrap_or(false);
        sense_right += ok as usize;
        base_sense_right += bok as usize;
        let e = per_word.entry(s.word.clone()).or_default();
        e.0 += ok as usize;
        e.1 += bok as usize;
        e.2 += 1;
    }

    let mean_us = micros.iter().sum::<f64>() / micros.len().max(1) as f64;
    println!("\n{:<44}{:>10}{:>12}", "measure", "Darma", "baseline");
    let row = |name: &str, a: f64, b: Option<f64>| println!("{:<44}{:>9.1}%{:>11}", name, a, b.map(|x| format!("{x:.1}%")).unwrap_or("-".into()));
    row(&format!("1. roles, whole sentence, without egg (n={n})"), pct(right3, n), None);
    row(&format!("1. roles, whole sentence, with egg (n={n})"), pct(right_egg, n), Some(pct(base_right, n)));
    row(&format!("   of the answered ({said} of {n})"), pct(right_egg, said), None);
    row(&format!("1. roles, clause alone (n={}; all 3 words seen in their roles: {clause_known})", d.test.len()), pct(clause_right, d.test.len()), None);
    row(&format!("2. sense chosen right (n={})", d.senses_test.len()), pct(sense_right, d.senses_test.len()), Some(pct(base_sense_right, d.senses_test.len())));
    row(&format!("3. nonsense rejected (n={})", nonsense.len()), pct(nonsense_rejected, nonsense.len()), Some(pct(base_ns_rejected, nonsense.len())));
    row(&format!("4. false rejections, sentences (n={n})"), pct(n - said, n), None);
    row(&format!("4. false rejections, clauses alone (n={})", d.test.len()), pct(d.test.len() - clause_ok, d.test.len()), None);
    println!("5. order: 'a v p' vs 'p v a' similarity {order_mean:.3} on average; {order_below} of {} below the membrane threshold {:.3}", order_sims.len(), m.membrane);
    println!("6. speed: {mean_us:.0} us a sentence through the egg; training {train_s:.2} s");
    println!("\nsense per word (Darma / most common sense, n):");
    for (w, (a, b, k)) in &per_word {
        print!("  {w} {:.0}/{:.0} ({k})", pct(*a, *k), pct(*b, *k));
    }
    println!();

    let res = json!({
        "dim": dim, "seed": seed, "train_seconds": train_s, "membrane_threshold": m.membrane, "fertile_threshold": m.fertile,
        "roles_sentences": n, "roles_without_egg": pct(right3, n), "roles_with_egg": pct(right_egg, n), "roles_answered": said,
        "roles_baseline": pct(base_right, n), "roles_clause_alone": pct(clause_right, d.test.len()), "clauses_all_words_seen_in_role": clause_known,
        "senses_n": d.senses_test.len(), "senses": pct(sense_right, d.senses_test.len()), "senses_baseline": pct(base_sense_right, d.senses_test.len()),
        "nonsense_n": nonsense.len(), "nonsense_rejected": pct(nonsense_rejected, nonsense.len()), "nonsense_rejected_baseline": pct(base_ns_rejected, nonsense.len()),
        "false_rejections_sentences": pct(n - said, n), "false_rejections_clauses": pct(d.test.len() - clause_ok, d.test.len()),
        "order_similarity_mean": order_mean, "order_below_threshold": order_below, "micros_per_sentence": mean_us,
        "nonsense_note": "nonsense = the test clauses' three words reversed (no ready-made nonsense set was found; nothing else invented)",
        "senses_note": "sense context vectors are bundled from CoarseWSD-20 train sentences by their marked sense; test marks are used only to measure",
    });
    std::fs::create_dir_all("results").unwrap();
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let p = format!("results/eval_{stamp}.json");
    std::fs::write(&p, serde_json::to_string_pretty(&res).unwrap()).unwrap();
    println!("written {p}");
}

pub fn ask(dim: usize, seed: u64, sentence: &str) {
    let d = data();
    let at = Atoms::new(dim, seed);
    let (m, _) = train(&at, &d);
    let p = understand(&m, &at, &d.lex, sentence);
    println!("words: {:?}", tokens(sentence));
    println!("sperm (the strongest {} of all readings, agent / action / patient):", p.sperm.len().min(8));
    for s in p.sperm.iter().take(8) {
        println!("  {:<40} swim {:.2}", s.fill.join(" / "), s.swim);
    }
    println!("membrane kept out {} (threshold {:.3}):", p.kept_out.len(), m.membrane);
    for (f, c) in p.kept_out.iter().take(5) {
        println!("  {f:<40} likeness {c:.3}");
    }
    for c in &p.senses {
        println!("competition for '{}': {} -> {}", c.word, c.scores.iter().map(|(s, v)| format!("{s} {v:.3}")).collect::<Vec<_>>().join(", "), c.winner);
    }
    println!("fertilisation (margin threshold {:.3}):", m.fertile);
    for (f, ok, mg) in p.tried.iter().take(8) {
        println!("  {f:<40} margin {mg:.3} {}", if *ok { "-> accepted" } else { "" });
    }
    match p.answer {
        Some(a) => println!("understood: who = {}, did what = {}, to whom/what = {}", a[0], a[1], a[2]),
        None => println!("en ymmärrä"),
    }
}
