//! Telling and asking (`darma qa`, `darma chat`): the quantum-minded core measured on real clauses.
//!
//! Darma is told the UD training clauses ("guerrillas killed engineer"), one addition to its state each. It is asked
//! the questions each clause answers ("who killed engineer", "guerrillas killed what"; the questions are the
//! clauses with one word replaced by who/what). The "en ymmärrä" threshold is learned from questions whose answer
//! Darma was never told (each clause taken back out of its state in turn): the winner's probability must reach
//! what 95% of such measurements do not.

use crate::eval::{data, train};
use crate::lexicon::Atoms;
use crate::quantum::{questions, words_of, Core};
use std::collections::HashSet;
use std::time::Instant;

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

pub fn run(dim: usize, seed: u64) {
    let d = data();
    let at = Atoms::new(dim, seed);
    // the meanings are learned only when they are measured (QA_MEANING=1): they take long at a large D
    let with_meaning = std::env::var("QA_MEANING").is_ok();
    let m = with_meaning.then(|| train(&at, &d).0);
    let limit: usize = std::env::var("QA_FACTS").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let facts: Vec<[String; 3]> = d.train.iter().take(limit).map(|c| c.gold()).collect();
    let mut cand: Vec<String> = facts.iter().flatten().cloned().collect::<HashSet<_>>().into_iter().collect();
    cand.sort();
    // a question is answerable when some told clause answers it
    let answerable = |facts: &[[String; 3]], skip: Option<usize>, q: &[String], who: bool| -> bool {
        facts.iter().enumerate().any(|(j, f)| Some(j) != skip && if who { f[1] == q[1] && f[2] == q[2] } else { f[0] == q[0] && f[1] == q[1] })
    };
    println!("told {} clauses; {} candidate words; D {dim}", facts.len(), cand.len());
    println!("{:<10}{:>11}{:>11}{:>11}{:>13}{:>13}{:>13}{:>10}", "meaning", "told:right", "told:wrong", "told:dunno", "untold:said", "reversed:said", "thresholds", "us/q");
    let betas: Vec<f32> = if with_meaning { vec![0.0, 1.0] } else { vec![0.0] };
    for beta in betas {
        let mut core = Core::new(&at, m.as_ref(), beta);
        let t0 = Instant::now();
        for f in &facts {
            core.tell(f);
        }
        let tell_s = t0.elapsed().as_secs_f64();
        // the threshold: questions whose answer was never told (each clause out of the state in turn)
        let mut absent = Vec::new();
        let mut absent_fid = Vec::new();
        for (i, f) in facts.iter().enumerate().step_by(3) {
            core.forget(f);
            for (k, (q, _)) in questions(f).iter().enumerate() {
                if !answerable(&facts, Some(i), q, k == 0) {
                    let ms = core.measure(q, &core.candidates_for(q));
                    // only measurements that were made (every word of the question told with something)
                    if ms.is_empty() {
                        continue;
                    }
                    let top = Core::peak(&ms);
                    absent_fid.push(ms[0].2);
                    if std::env::var("DARMA_DEBUG").is_ok() && top > 0.9 && absent.len() < 400 {
                        eprintln!("  untold {:?} -> {:?}", q, ms.iter().take(2).map(|x| (&x.0, x.1)).collect::<Vec<_>>());
                    }
                    absent.push(top);
                }
            }
            core.tell(f);
        }
        absent.sort_by(|a, b| a.partial_cmp(b).unwrap());
        core.threshold = absent[((absent.len() as f32 - 1.0) * 0.95) as usize];
        // the winner's own standing: learned on questions asked the wrong way round about the even-numbered clauses
        // ("who v a" when told "a v p"), measured below on the odd-numbered ones
        let mut wrong_way: Vec<f32> = Vec::new();
        for f in facts.iter().step_by(2) {
            let q = vec!["who".to_string(), f[1].clone(), f[0].clone()];
            if !answerable(&facts, None, &q, true) {
                let ms = core.measure(&q, &core.candidates_for(&q));
                if !ms.is_empty() {
                    wrong_way.push(ms[0].2);
                }
            }
        }
        let _ = absent_fid;
        wrong_way.sort_by(|a, b| a.partial_cmp(b).unwrap());
        core.fid_threshold = wrong_way[((wrong_way.len() as f32 - 1.0) * 0.95) as usize];
        // 1. told: does it answer what it was told?
        let (mut right, mut wrong, mut dunno) = (0, 0, 0);
        let t1 = Instant::now();
        let mut nq = 0;
        for f in &facts {
            for (q, a) in questions(f) {
                nq += 1;
                match core.answer(&q, &core.candidates_for(&q)).0 {
                    Some(x) if x == a => right += 1,
                    Some(_) => wrong += 1,
                    None => dunno += 1,
                }
            }
        }
        let us = t1.elapsed().as_micros() as f64 / nq as f64;
        // 2. untold: the test clauses' questions memory cannot answer: does it say "en ymmärrä"?
        let (mut said, mut n_untold) = (0, 0);
        for c in &d.test {
            for (k, (q, _)) in questions(&c.gold()).iter().enumerate() {
                if !answerable(&facts, None, q, k == 0) {
                    n_untold += 1;
                    said += core.answer(q, &core.candidates_for(q)).0.is_some() as usize;
                }
            }
        }
        // 3. order: told "a v p", asked "who v a" (who did it to the agent?): the answer is not told
        let (mut rsaid, mut n_rev) = (0, 0);
        for f in facts.iter().skip(1).step_by(2) {
            let q = vec!["who".to_string(), f[1].clone(), f[0].clone()];
            if !answerable(&facts, None, &q, true) {
                n_rev += 1;
                let (a, ms) = core.answer(&q, &core.candidates_for(&q));
                if a.is_some() && std::env::var("DARMA_DEBUG").is_ok() && rsaid < 12 {
                    eprintln!("  reversed {:?} (told {:?}) -> {:?}", q, f, ms.iter().take(2).map(|x| (&x.0, x.1)).collect::<Vec<_>>());
                }
                rsaid += a.is_some() as usize;
            }
        }
        // yes/no: told statements (should be yes), wrong-way statements about the odd clauses (should not be), and
        // the test clauses (untold: should not be). The standing a yes needs is learned on the even clauses' wrong-way
        // statements, as above.
        let mut neg: Vec<f32> = facts.iter().step_by(2).filter_map(|f| {
            let s = vec![f[2].clone(), f[1].clone(), f[0].clone()];
            (!facts.contains(&[s[0].clone(), s[1].clone(), s[2].clone()])).then(|| core.verify(&s).map(|x| x.1)).flatten()
        }).collect();
        neg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let yes_at = neg[((neg.len() as f32 - 1.0) * 0.95) as usize];
        let yes = |s: &[String]| core.verify(s).map_or(false, |x| x.1 > yes_at);
        let told_yes = facts.iter().filter(|f| yes(&f[..])).count();
        let wrong_way_st: Vec<Vec<String>> = facts.iter().skip(1).step_by(2).map(|f| vec![f[2].clone(), f[1].clone(), f[0].clone()]).filter(|s| !facts.contains(&[s[0].clone(), s[1].clone(), s[2].clone()])).collect();
        let wrong_yes = wrong_way_st.iter().filter(|s| yes(s)).count();
        let untold: Vec<[String; 3]> = d.test.iter().map(|c| c.gold()).filter(|g| !facts.contains(g)).collect();
        let untold_yes = untold.iter().filter(|g| yes(&g[..])).count();
        let untold_unknown = untold.iter().filter(|g| core.verify(&g[..]).is_none()).count();
        println!(
            "{:<10} yes/no: told said yes {:.1}%; wrong-way said yes {:.1}% (n={}); untold said yes {:.1}%, could not say {:.1}% (n={}); yes needs {:.2} sd",
            "", pct(told_yes, facts.len()), pct(wrong_yes, wrong_way_st.len()), wrong_way_st.len(), pct(untold_yes, untold.len()), pct(untold_unknown, untold.len()), untold.len(), yes_at
        );
        println!(
            "{:<10}{:>10.1}%{:>10.1}%{:>10.1}%{:>12.1}%{:>12.1}%{:>7.3}/{:.3}{:>8.0}",
            if beta > 0.0 { "with" } else { "without" },
            pct(right, nq), pct(wrong, nq), pct(dunno, nq), pct(said, n_untold), pct(rsaid, n_rev), core.threshold, core.fid_threshold, us
        );
        if beta == 0.0 {
            println!("   (telling {} clauses took {:.3} s; untold questions n={n_untold}, reversed n={n_rev}; chance per candidate {:.5})", facts.len(), tell_s, 1.0 / cand.len() as f32);
        }
    }
}

/// a conversation: a line with who/what is a question (measured), a line ending in "?" is a yes/no question, any
/// other line is told (added to the state). What was told is kept in `memory` (one sentence a line) and told again
/// when the conversation starts (the same state, exactly: telling is deterministic).
pub fn chat(dim: usize, seed: u64, beta: f32, memory: &str, fresh: bool) {
    let at = Atoms::new(dim, seed);
    let m = (beta > 0.0).then(|| train(&at, &data()).0);
    let mut core = Core::new(&at, m.as_ref(), beta);
    let mut known: Vec<String> = Vec::new();
    if fresh {
        let _ = std::fs::remove_file(memory);
    }
    let kept: Vec<String> = std::fs::read_to_string(memory).map(|s| s.lines().map(String::from).collect()).unwrap_or_default();
    for line in &kept {
        core.tell(&words_of(line));
    }
    // the copulas people marked in UD (seen at least 20 times): what says two things are the same
    let copulas: Vec<String> = std::fs::read_to_string("data/copulas.json")
        .ok()
        .and_then(|s| serde_json::from_str::<std::collections::BTreeMap<String, usize>>(&s).ok())
        .map(|m| m.into_iter().filter(|x| x.1 >= 20).map(|x| x.0).collect())
        .unwrap_or_default();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(memory).expect("memory file");
    println!("Darma listens ({} things remembered from before, in {memory}). Tell it things (\"the dog chased the cat\"); ask with who or what (\"who chased the cat\") or end with ? (\"the dog chased the cat?\"). Empty line ends.", kept.len());
    for line in std::io::stdin().lines() {
        let line = line.unwrap();
        let w = words_of(&line);
        if w.is_empty() {
            break;
        }
        if line.trim_end().ends_with('?') && !w.iter().any(|x| crate::quantum::HOLES.contains(&x.as_str())) {
            match core.verify_reasoned(&w, &copulas, 3.0) {
                None => println!("Darma: en tiedä      (some of these words I was never told)"),
                Some((true, why)) if why.is_empty() => println!("Darma: kyllä"),
                Some((true, why)) => println!("Darma: kyllä      (because {})", why.join("; ")),
                Some((false, _)) => println!("Darma: ei sen mukaan mitä tiedän"),
            }
        } else if w.iter().any(|x| crate::quantum::HOLES.contains(&x.as_str())) {
            // with few things told there is little noise to learn from: the winner must hold over half of the
            // question and stand out by half a match
            core.threshold = 0.5;
            core.fid_threshold = 3.0;
            match core.answer_reasoned(&w, &copulas) {
                (Some(x), why) if why.is_empty() => println!("Darma: {x}"),
                (Some(x), why) => println!("Darma: {x}      (because {})", why.join("; ")),
                (None, _) => println!("Darma: en ymmärrä"),
            }
        } else {
            core.tell(&w);
            use std::io::Write;
            let _ = writeln!(log, "{}", w.join(" "));
            // what can fill a hole: every word told, and every two words told together
            let mut fills: Vec<String> = w.clone();
            fills.extend(w.windows(2).map(|p| p.join(" ")));
            for x in fills {
                if !known.contains(&x) {
                    known.push(x);
                }
            }
            println!("Darma: (told: {} things)", core.told);
        }
    }
}
