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

/// a conversation with Darma: a line with who/what is a question (measured), a line ending in "?" is a yes/no
/// question, any other line is told (added to the state). What was told is kept in `memory` (one sentence a line)
/// and told again when the conversation starts (the same state, exactly: telling is deterministic).
pub struct Talk<'a> {
    pub core: Core<'a>,
    copulas: Vec<String>,
    log: std::fs::File,
    pub remembered: usize,
}

/// one reply: what Darma says, and on what it stood (empty when it stood on nothing more than the question)
pub struct Reply {
    pub said: String,
    pub why: String,
    pub kind: &'static str,
}

impl<'a> Talk<'a> {
    pub fn new(at: &'a Atoms, memory: &str, fresh: bool) -> Self {
        let mut core = Core::new(at, None, 0.0);
        // with few things told there is little noise to learn from: the winner must hold over half of the question
        // and stand out by half a match (set by hand, not learned)
        core.threshold = 0.5;
        core.fid_threshold = 3.0;
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
        let log = std::fs::OpenOptions::new().create(true).append(true).open(memory).expect("memory file");
        Talk { core, copulas, log, remembered: kept.len() }
    }

    pub fn line(&mut self, line: &str) -> Reply {
        let w = words_of(line);
        let hole = w.iter().any(|x| crate::quantum::HOLES.contains(&x.as_str()));
        let r = |said: &str, why: String, kind| Reply { said: said.to_string(), why, kind };
        if w.is_empty() {
            r("", String::new(), "none")
        } else if line.trim_end().ends_with('?') && !hole {
            match self.core.verify_reasoned(&w, &self.copulas, 3.0) {
                None => r("en tiedä", "some of these words I was never told".into(), "unknown"),
                Some((true, why)) if why.is_empty() => r("kyllä", String::new(), "yes"),
                Some((true, why)) => r("kyllä", format!("because {}", why.join("; ")), "yes"),
                Some((false, _)) => r("ei sen mukaan mitä tiedän", "I know these words, but not this of them".into(), "no"),
            }
        } else if hole {
            match self.core.answer_reasoned(&w, &self.copulas) {
                (Some(x), why) if why.is_empty() => r(&x, String::new(), "answer"),
                (Some(x), why) => r(&x, format!("because {}", why.join("; ")), "answer"),
                (None, _) => r("en ymmärrä", "nothing I was told stands out as the answer".into(), "unknown"),
            }
        } else {
            self.core.tell(&w);
            use std::io::Write;
            let _ = writeln!(self.log, "{}", w.join(" "));
            r("muistan", format!("{} things told", self.core.told), "told")
        }
    }
}

pub fn chat(dim: usize, seed: u64, memory: &str, fresh: bool) {
    let at = Atoms::new(dim, seed);
    let mut talk = Talk::new(&at, memory, fresh);
    println!("Darma listens ({} things remembered from before, in {memory}). Tell it things (\"the dog chased the cat\"); ask with who or what (\"who chased the cat\") or end with ? (\"the dog chased the cat?\"). Empty line ends.", talk.remembered);
    for line in std::io::stdin().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            break;
        }
        let rep = talk.line(&line);
        if rep.why.is_empty() || rep.kind == "told" {
            println!("Darma: {}", rep.said);
        } else {
            println!("Darma: {}      ({})", rep.said, rep.why);
        }
    }
}

/// the chat page (`darma serve`): the same conversation in a browser, at http://localhost:PORT
pub fn serve(dim: usize, seed: u64, memory: &str, port: u16) {
    use std::io::{Read, Write};
    let at = Atoms::new(dim, seed);
    let mut talk = Talk::new(&at, memory, false);
    let page = include_str!("chat.html");
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).expect("port in use");
    println!("Darma: http://localhost:{port}  ({} things remembered, in {memory})", talk.remembered);
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // the head, then as much body as it says
        let (head, body) = loop {
            let Ok(k) = s.read(&mut chunk) else { break (String::new(), String::new()) };
            if k == 0 {
                break (String::new(), String::new());
            }
            buf.extend_from_slice(&chunk[..k]);
            if let Some(p) = buf.windows(4).position(|x| x == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..p]).to_string();
                let len: usize = head.lines().find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0))).unwrap_or(0);
                while buf.len() < p + 4 + len {
                    let Ok(k) = s.read(&mut chunk) else { break };
                    if k == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..k]);
                }
                break (head, String::from_utf8_lossy(&buf[p + 4..(p + 4 + len).min(buf.len())]).to_string());
            }
        };
        let (ctype, out) = if head.starts_with("POST /say") {
            let rep = talk.line(body.trim());
            let j = serde_json::json!({"said": rep.said, "why": rep.why, "kind": rep.kind, "told": talk.core.told});
            ("application/json", j.to_string())
        } else if head.starts_with("GET /state") {
            ("application/json", serde_json::json!({"told": talk.core.told, "memory": memory}).to_string())
        } else {
            ("text/html; charset=utf-8", page.to_string())
        };
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}", out.len());
    }
}

/// the scale test (`darma qa-sentences`): Darma is told every UD training sentence whole (about 14,000 sentences,
/// 200,000 words), and asked about the marked clauses in their own words: "guerrillas killed a member" becomes "who
/// killed a member" (answer: guerrillas) and "guerrillas killed a what" (answer: member). An answer counts as right
/// when it holds the marked word. The thresholds are learned on the even-numbered clauses with their sentences taken
/// back out of the state (untold); the odd-numbered clauses are asked told. Untold: the test sentences' clauses.
pub fn sentences(dim: usize, seed: u64) {
    let d = data();
    let at = Atoms::new(dim, seed);
    let mut core = Core::new(&at, None, 0.0);
    let t0 = Instant::now();
    for s in &d.tagged {
        core.tell(&s.words);
    }
    println!("told {} sentences ({} words) in {:.1} s, D {dim}", d.tagged.len(), d.tagged.iter().map(|s| s.words.len()).sum::<usize>(), t0.elapsed().as_secs_f64());
    // a clause's two questions in its sentence's own words: (question, the marked answer word)
    let qs = |c: &crate::memory::Clause| -> Option<(Vec<String>, [(Vec<String>, String); 2])> {
        let w = words_of(&c.sentence);
        let g = c.gold();
        let a = w.iter().position(|x| *x == g[0])?;
        let p = (a..w.len()).find(|&i| w[i] == g[2])?;
        if p - a > 6 {
            return None;
        }
        let seg = w[a..=p].to_vec();
        let mut q1 = seg.clone();
        q1[0] = "who".into();
        let mut q2 = seg.clone();
        let last = q2.len() - 1;
        q2[last] = "what".into();
        Some((w, [(q1, g[0].clone()), (q2, g[2].clone())]))
    };
    // thresholds: untold questions (each even clause's sentence out of the state while it is asked)
    let (mut peaks, mut zs) = (Vec::new(), Vec::new());
    for c in d.train.iter().step_by(2) {
        let Some((w, both)) = qs(c) else { continue };
        core.forget(&w);
        for (q, _) in &both {
            let ms = core.measure(q, &core.candidates_for(q));
            if !ms.is_empty() {
                peaks.push(Core::peak(&ms));
                zs.push(ms[0].2);
            }
        }
        core.tell(&w);
    }
    let q95 = |mut v: Vec<f32>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f32 - 1.0) * 0.95) as usize]
    };
    core.threshold = q95(peaks);
    core.fid_threshold = q95(zs);
    let (mut right, mut wrong, mut dunno, mut n) = (0, 0, 0, 0);
    let t1 = Instant::now();
    for c in d.train.iter().skip(1).step_by(2) {
        let Some((_, both)) = qs(c) else { continue };
        for (q, a) in &both {
            n += 1;
            match core.answer(q, &core.candidates_for(q)).0 {
                Some(x) if x.split(' ').any(|p| p == a) => right += 1,
                Some(_) => wrong += 1,
                None => dunno += 1,
            }
        }
    }
    let us = t1.elapsed().as_micros() as f64 / n.max(1) as f64;
    let (mut said, mut nu) = (0, 0);
    for c in &d.test {
        let Some((_, both)) = qs(c) else { continue };
        for (q, _) in &both {
            nu += 1;
            said += core.answer(q, &core.candidates_for(q)).0.is_some() as usize;
        }
    }
    println!("told questions (n={n}): right {:.1}%, wrong {:.1}%, en ymmärrä {:.1}%; untold (test sentences, n={nu}) answered {:.1}%; thresholds {:.3} / {:.2} sd; {us:.0} us a question",
        pct(right, n), pct(wrong, n), pct(dunno, n), pct(said, nu), core.threshold, core.fid_threshold);
}

/// senses as measurements (`darma qa-senses`): the CoarseWSD-20 training sentences are told with the two-or-more-sense
/// word replaced by its marked sense ("cable that is swung by a crane_(machine) into the side"); a test sentence is
/// asked with the word as the hole, and the candidates are the word's senses: which sense does memory hold here?
pub fn senses(dim: usize, seed: u64) {
    let d = data();
    let at = Atoms::new(dim, seed);
    let mut core = Core::new(&at, None, 0.0);
    let t0 = Instant::now();
    let around = |s: &crate::memory::SenseCase, fill: &str| -> Vec<String> {
        let w: Vec<String> = s.sentence.split(' ').map(|x| x.to_lowercase()).collect();
        let lo = s.index.saturating_sub(4);
        let hi = (s.index + 5).min(w.len());
        let mut out: Vec<String> = w[lo..hi].to_vec();
        out[s.index - lo] = fill.to_string();
        out
    };
    for s in &d.senses_train {
        core.tell(&around(s, &s.sense));
    }
    println!("told {} sense-marked sentences (4 words each side of the word) in {:.1} s, D {dim}", d.senses_train.len(), t0.elapsed().as_secs_f64());
    let mut senses_of: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
    for (w, s) in d.lex.senses.iter() {
        senses_of.insert(w.as_str(), s.clone());
    }
    let (mut right, mut n, mut none) = (0, 0, 0);
    let mut per: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for s in &d.senses_test {
        let q = around(s, "what");
        let cands = senses_of.get(s.word.as_str()).cloned().unwrap_or_default();
        // the measurement: the sense with the highest fidelity here (no "en ymmärrä" threshold: one sense is chosen)
        let ms = core.measure(&q, &cands);
        let got = ms.first().filter(|x| x.1 > 0.0).map(|x| x.0.clone());
        n += 1;
        none += got.is_none() as usize;
        let ok = got.as_deref() == Some(s.sense.as_str());
        right += ok as usize;
        let e = per.entry(s.word.clone()).or_default();
        e.0 += ok as usize;
        e.1 += 1;
    }
    println!("sense chosen right {:.1}% (n={n}; no sense held at all {:.1}%)", pct(right, n), pct(none, n));
    println!("{}", per.iter().map(|(w, (a, b))| format!("{w} {:.0}%", pct(*a, *b))).collect::<Vec<_>>().join("  "));
}
