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
use std::collections::{HashMap, HashSet};
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
    /// every sentence told, and in how many of them each word was (what is rare is what tells things apart)
    said: Vec<Vec<String>>,
    df: HashMap<String, usize>,
    by_word: HashMap<String, Vec<usize>>,
    /// what Darma recalled lately (not said again soon), what it asked and is waiting for, what it asked before
    recent: Vec<usize>,
    pending: Option<Vec<String>>,
    asked: Vec<Vec<String>>,
    /// people's conversations: what they answered to what (None when DailyDialog is not downloaded)
    dialog: Option<crate::dialog::Dialog>,
    /// what Darma said last (the conversation so far, for the next match)
    last: Option<Vec<String>>,
    /// what Darma knows of itself
    pub me: crate::me::Me<'a>,
}

use crate::dialog::TALK_AT;

/// one reply: what Darma says, on what it stood, what it recalls of it, and what it asks back
pub struct Reply {
    pub said: String,
    pub why: String,
    pub kind: &'static str,
    pub recall: String,
    pub ask: String,
}

impl<'a> Talk<'a> {
    pub fn new(at: &'a Atoms, memory: &str, fresh: bool, seed_ud: bool) -> Self {
        let mut core = Core::new(at, None, 0.0);
        // with few things told there is little noise to learn from: the winner must hold over half of the question
        // and stand out by half a match (set by hand, not learned)
        core.threshold = 0.5;
        core.fid_threshold = 3.0;
        if fresh {
            let _ = std::fs::remove_file(memory);
        }
        // the copulas people marked in UD (seen at least 20 times): what says two things are the same
        let copulas: Vec<String> = std::fs::read_to_string("data/copulas.json")
            .ok()
            .and_then(|s| serde_json::from_str::<std::collections::BTreeMap<String, usize>>(&s).ok())
            .map(|m| m.into_iter().filter(|x| x.1 >= 20).map(|x| x.0).collect())
            .unwrap_or_default();
        let log = std::fs::OpenOptions::new().create(true).append(true).open(memory).expect("memory file");
        let mut talk = Talk { core, copulas, log, remembered: 0, said: Vec::new(), df: HashMap::new(), by_word: HashMap::new(), recent: Vec::new(), pending: None, asked: Vec::new(), dialog: crate::dialog::Dialog::load("data/raw/dailydialog/train.json"), last: None, me: crate::me::Me::new(at, &format!("{}_self.txt", memory.trim_end_matches(".txt")), fresh) };
        if seed_ud {
            // a memory to begin with: every UD training sentence (people's English, news and the web)
            for s in &data().tagged {
                talk.learn(&s.words);
            }
        }
        let kept: Vec<String> = std::fs::read_to_string(memory).map(|s| s.lines().map(String::from).collect()).unwrap_or_default();
        for line in &kept {
            talk.learn(&words_of(line));
        }
        talk.remembered = talk.said.len();
        let n = talk.said.len();
        talk.me.happened("remember", &format!("i remember {n} things"));
        talk
    }

    fn learn(&mut self, w: &[String]) {
        self.core.tell(w);
        let i = self.said.len();
        let mut u: Vec<&String> = w.iter().collect();
        u.sort();
        u.dedup();
        for x in u {
            *self.df.entry(x.clone()).or_default() += 1;
            self.by_word.entry(x.clone()).or_default().push(i);
        }
        self.said.push(w.to_vec());
    }

    /// how rare a word is among what was told (log of sentences over sentences with it)
    fn rarity(&self, x: &str) -> f32 {
        let n = self.said.len().max(1) as f32;
        (n / *self.df.get(x).unwrap_or(&1) as f32).ln()
    }

    /// recalling: the told sentence that shares most with what was just heard, each shared word weighed by how rare
    /// it is among what was told; a word told in more than half of the sentences ("the") does not count. Not the
    /// sentence just heard, not one recalled lately.
    fn recall(&mut self, w: &[String], not: Option<usize>) -> Option<String> {
        let mut score: HashMap<usize, f32> = HashMap::new();
        let mut u: Vec<&String> = w.iter().collect();
        u.sort();
        u.dedup();
        for x in u {
            let r = self.rarity(x);
            if r < std::f32::consts::LN_2 {
                continue;
            }
            for &i in self.by_word.get(x.as_str()).map(|v| v.as_slice()).unwrap_or(&[]) {
                *score.entry(i).or_default() += r;
            }
        }
        let best = score
            .into_iter()
            .filter(|(i, _)| Some(*i) != not && !self.recent.contains(i) && self.said[*i] != w)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(b.0.cmp(&a.0)))?;
        self.recent.push(best.0);
        if self.recent.len() > 5 {
            self.recent.remove(0);
        }
        Some(self.said[best.0].join(" "))
    }

    /// a sameness told: (what, what it is), from "X is a Y": the word before the copula and the last word
    fn sameness(&self, s: &[String]) -> Option<(String, String)> {
        let c = s.iter().position(|x| self.copulas.contains(x))?;
        if c == 0 || c + 1 >= s.len() {
            return None;
        }
        Some((s[c - 1].clone(), s[s.len() - 1].clone()))
    }

    /// wondering: where what was just heard and memory meet through a sameness ("rex is a dog" and "the dog
    /// chased the cat"), what would follow ("rex chased the cat") that Darma does not know yet, nor asked before
    fn wonder(&mut self, w: &[String]) -> Option<Vec<String>> {
        let mut pairs: Vec<((String, String), Vec<String>)> = Vec::new();
        if let Some(xy) = self.sameness(w) {
            for &i in self.by_word.get(&xy.1).map(|v| v.as_slice()).unwrap_or(&[]) {
                pairs.push((xy.clone(), self.said[i].clone()));
            }
        } else {
            for t in &self.said {
                if let Some(xy) = self.sameness(t) {
                    if w.contains(&xy.1) {
                        pairs.push((xy, w.to_vec()));
                    }
                }
            }
        }
        for ((x, y), t) in pairs {
            if self.sameness(&t).is_some() {
                continue;
            }
            // the thing it is, said of the other: X in the place of Y, with what was said after Y
            let Some(k) = t.iter().position(|v| *v == y) else { continue };
            if k + 1 >= t.len() {
                continue;
            }
            let mut st = vec![x.clone()];
            st.extend(t[k + 1..].iter().cloned());
            let known = self.core.verify(&st).is_some_and(|v| v.1 >= 3.0);
            if !known && !self.asked.contains(&st) {
                self.asked.push(st.clone());
                return Some(st);
            }
        }
        None
    }

    /// what people answered to the turn most like this one, if one is alike enough
    fn talk(&mut self, w: &[String]) -> Option<Reply> {
        let last = self.last.clone();
        let me = &self.me;
        let world = &self.core;
        let known = |x: &str| world.reg.contains_key(x) || me.core.reg.contains_key(x);
        let test = |ans: &str| me.names_a_stranger(ans, &known).or_else(|| me.contradicts(&words_of(ans)));
        let (ans, matched, s, let_go) = self.dialog.as_mut()?.reply_where(w, last.as_deref(), &test)?;
        let mut why = format!("people answered this to: \"{matched}\" ({s:.2})");
        if !let_go.is_empty() {
            why.push_str(&format!("; not as myself: {}", let_go.join("; ")));
        }
        (s >= TALK_AT).then(|| Reply { said: ans, why, kind: "talk", recall: String::new(), ask: String::new() })
    }

    pub fn line(&mut self, line: &str) -> Reply {
        let rep = self.line_inner(line);
        let said = if rep.ask.is_empty() { rep.said.clone() } else { rep.ask.clone() };
        self.last = Some(words_of(&said));
        // it tells itself what happened to it
        let n = self.said.len();
        self.me.happened("remember", &format!("i remember {n} things"));
        let heard = words_of(line).join(" ");
        if rep.kind == "told" && !heard.is_empty() {
            self.me.happened("told", &format!("i was told {heard}"));
        }
        if rep.kind == "unknown" && !heard.is_empty() {
            self.me.happened("unknown", &format!("i could not answer {heard}"));
        }
        if !rep.ask.is_empty() {
            self.me.happened("asked", &format!("i asked {}", words_of(&rep.ask).join(" ")));
        }
        rep
    }

    /// what the speaker says of themselves: kept in the world's memory as "you ..." ("my name is joose" -> "your
    /// name is joose"), and their questions of themselves ("what is my name") answered from it
    fn of_you(&mut self, asked: bool, w: &[String]) -> Reply {
        let r = |said: String, why: String, kind| Reply { said, why, kind, recall: String::new(), ask: String::new() };
        let hole = w.iter().any(|x| crate::quantum::HOLES.contains(&x.as_str()));
        let q = crate::me::to_you(&crate::me::statement_order(w));
        if hole {
            match self.core.answer_reasoned(&q, &self.copulas) {
                (Some(a), _) => r(q.iter().map(|x| if crate::quantum::HOLES.contains(&x.as_str()) { a.clone() } else { x.clone() }).collect::<Vec<_>>().join(" "), "from what you told me of yourself".into(), "answer"),
                // what the speaker is, only they can say: not as some person in a conversation answered
                (None, _) => r("en tiedä sitä sinusta".into(), format!("you have not told me: {}", q.join(" ")), "unknown"),
            }
        } else if asked {
            match self.core.verify(&q) {
                Some((_, z)) if z >= 3.0 => r("kyllä".into(), format!("you told me: {}", q.join(" ")), "yes"),
                Some(_) => r("ei sen mukaan mitä tiedän sinusta".into(), format!("you have not told me: {}", q.join(" ")), "no"),
                None => r("en tiedä sitä sinusta".into(), format!("you have not told me: {}", q.join(" ")), "unknown"),
            }
        } else if !matches!(w.first().map(|s| s.as_str()), Some("i" | "my")) {
            // the speaker spoke of themselves but did not say what they are ("tell me a joke"): talk
            self.talk(w).unwrap_or_else(|| r("en ymmärrä".into(), "nothing people said to this is like it".into(), "unknown"))
        } else {
            let s = crate::me::to_you(w);
            let i = self.said.len();
            self.learn(&s);
            use std::io::Write;
            let _ = writeln!(self.log, "{}", s.join(" "));
            let mut rep = r("muistan sen sinusta".into(), s.join(" "), "told");
            rep.recall = self.recall(&s, Some(i)).unwrap_or_default();
            rep
        }
    }

    /// what was said of Darma itself: kept in its self (told), or asked of its self (questions); a question of
    /// itself is answered only from what it knows of itself, never as some person in a conversation answered it
    fn of_me(&mut self, asked: bool, w: &[String]) -> Reply {
        let r = |said: String, why: String, kind| Reply { said, why, kind, recall: String::new(), ask: String::new() };
        let hole = w.iter().any(|x| crate::quantum::HOLES.contains(&x.as_str()));
        let q = crate::me::to_me(&crate::me::statement_order(w));
        if hole {
            // what or who it is (a question with a copula: "who are you", "where are you from") is answered from
            // itself only; what it does or thinks ("what do you think") may be answered as people did, never against
            // what it knows of itself
            let of_what_i_am = q.iter().any(|x| self.copulas.contains(x));
            match self.me.answer(&q, &self.copulas) {
                Some(s) => r(s, "from what I know of myself".into(), "answer"),
                None if !of_what_i_am => self.talk(w).unwrap_or_else(|| r("en tiedä sitä itsestäni".into(), format!("I know nothing of myself that answers \"{}\"", q.join(" ")), "unknown")),
                None => r("en tiedä sitä itsestäni".into(), format!("I know nothing of myself that answers \"{}\"", q.join(" ")), "unknown"),
            }
        } else if asked {
            // "are you ...": a question of what it is, answered from itself only; any other ("how are you?") is talk,
            // answered as people did, but never against what it knows of itself
            let of_what_i_am = w.first().is_some_and(|x| self.copulas.contains(x));
            match self.me.verify(&q, &self.copulas) {
                Some(true) => r("kyllä".into(), format!("I know of myself: {}", q.join(" ")), "yes"),
                Some(false) if of_what_i_am => r("ei sen mukaan mitä tiedän itsestäni".into(), format!("I do not know this of myself: {}", q.join(" ")), "no"),
                None if of_what_i_am => r("en tiedä sitä itsestäni".into(), format!("nobody has told me: {}", q.join(" ")), "unknown"),
                _ => self.talk(w).unwrap_or_else(|| r("en tiedä sitä itsestäni".into(), format!("nobody has told me: {}", q.join(" ")), "unknown")),
            }
        } else if matches!(w.first().map(|s| s.as_str()), Some("you" | "your")) {
            // said of Darma itself ("you are curious"): kept in its self; and it answers as people did, if that fits
            let s = crate::me::to_me(w);
            self.me.tell(&s);
            match self.talk(w) {
                Some(mut t) => {
                    t.why = format!("{}; I keep of myself: {}", t.why, s.join(" "));
                    t
                }
                None => r("muistan sen itsestäni".into(), s.join(" "), "told"),
            }
        } else {
            // spoken to Darma, but not saying what it is ("how you doing", "thank you"): talk
            self.talk(w).unwrap_or_else(|| r("en ymmärrä".into(), "nothing people said to this is like it, and it does not say what I am".into(), "unknown"))
        }
    }

    /// is this a question: it ends in "?", or opens with a word people's questions open with (learned)
    fn asks(&self, line: &str, w: &[String]) -> bool {
        line.trim_end().ends_with('?') || w.first().is_some_and(|f| self.dialog.as_ref().is_some_and(|d| d.opens_question(f)))
    }

    fn line_inner(&mut self, line: &str) -> Reply {
        let w = words_of(line);
        let hole = w.iter().any(|x| crate::quantum::HOLES.contains(&x.as_str()));
        let r = |said: &str, why: String, kind| Reply { said: said.to_string(), why, kind, recall: String::new(), ask: String::new() };
        // an answer to what Darma asked: "yes" tells it, "no" lets it go (these two words are a rule, not learned)
        if let Some(q) = self.pending.take() {
            match w.first().map(|s| s.as_str()) {
                Some("yes") => {
                    self.learn(&q);
                    use std::io::Write;
                    let _ = writeln!(self.log, "{}", q.join(" "));
                    let mut rep = r("kiitos, muistan sen", q.join(" "), "told");
                    rep.recall = self.recall(&q, None).unwrap_or_default();
                    return rep;
                }
                Some("no") => return r("selvä", format!("not: {}", q.join(" ")), "told"),
                _ => {}
            }
        }
        if w.is_empty() {
            r("", String::new(), "none")
        } else if crate::me::about_me(&w) {
            let asked = self.asks(line, &w);
            self.of_me(asked, &w)
        } else if crate::me::about_speaker(&w) {
            let asked = self.asks(line, &w);
            self.of_you(asked, &w)
        } else if self.asks(line, &w) && !hole {
            // what it was told answers first; if that cannot say, it answers as people did
            match self.core.verify_reasoned(&w, &self.copulas, 3.0) {
                Some((true, why)) if why.is_empty() => r("kyllä", String::new(), "yes"),
                Some((true, why)) => r("kyllä", format!("because {}", why.join("; ")), "yes"),
                Some((false, _)) => self.talk(&w).unwrap_or_else(|| r("ei sen mukaan mitä tiedän", "I know these words, but not this of them".into(), "no")),
                None => self.talk(&w).unwrap_or_else(|| r("en tiedä", "some of these words I was never told".into(), "unknown")),
            }
        } else if hole {
            match self.core.answer_reasoned(&w, &self.copulas) {
                (Some(x), why) if why.is_empty() => r(&x, String::new(), "answer"),
                (Some(x), why) => r(&x, format!("because {}", why.join("; ")), "answer"),
                (None, _) => self.talk(&w).unwrap_or_else(|| r("en ymmärrä", "nothing I was told stands out as the answer".into(), "unknown")),
            }
        } else {
            let i = self.said.len();
            self.learn(&w);
            use std::io::Write;
            let _ = writeln!(self.log, "{}", w.join(" "));
            // it answers as people did when what was said is like something they answered; else it says it remembers,
            // and what it recalls of it
            let mut rep = match self.talk(&w) {
                Some(t) => t,
                None => {
                    let mut rep = r("muistan", format!("{} things told", self.core.told), "told");
                    rep.recall = self.recall(&w, Some(i)).unwrap_or_default();
                    rep
                }
            };
            if let Some(q) = self.wonder(&w) {
                rep.ask = format!("{}?", q.join(" "));
                self.pending = Some(q);
            }
            rep
        }
    }
}

pub fn chat(dim: usize, seed: u64, memory: &str, fresh: bool) {
    let at = Atoms::new(dim, seed);
    let mut talk = Talk::new(&at, memory, fresh, false);
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
        if !rep.recall.is_empty() {
            println!("       ({})", rep.recall);
        }
        if !rep.ask.is_empty() {
            println!("Darma: {}", rep.ask);
        }
    }
}

/// the chat page (`darma serve`): the same conversation in a browser, at http://localhost:PORT
pub fn serve(dim: usize, seed: u64, memory: &str, port: u16, seed_ud: bool) {
    use std::io::{Read, Write};
    let at = Atoms::new(dim, seed);
    let mut talk = Talk::new(&at, memory, false, seed_ud);
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
            let j = serde_json::json!({"said": rep.said, "why": rep.why, "kind": rep.kind, "recall": rep.recall, "ask": rep.ask, "told": talk.core.told});
            ("application/json", j.to_string())
        } else if head.starts_with("GET /self") {
            ("application/json", serde_json::json!({"self": talk.me.all()}).to_string())
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
