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
    /// its understanding of what a line does and feels, and its grammar (for sentences of its own)
    own: Option<(crate::understand::Understanding<'a>, crate::compose::Composer)>,
    /// conscious parallel thinking: hearing, and thoughts on what is heard and known
    cpt: Option<crate::cpt::Cpt>,
    /// what it read in Wikipedia (struck before by `darma wiki-test`)
    wiki: Option<crate::wiki::Wiki>,
    /// its will: the stances it chooses itself, from what it lived and read
    will: crate::will::Will,
}

/// a social formula (a greeting, thanks, a goodbye) is a short line very like what people said: only then are
/// people's answers used (set by hand)
pub const SOCIAL_AT: f32 = 0.8;
pub const SOCIAL_WORDS: usize = 5;
/// what Darma says when it cannot answer by itself: it asks to be taught
pub const TEACH_ME: &str = "I don't understand that yet.";

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
        let mut talk = Talk { core, copulas, log, remembered: 0, said: Vec::new(), df: HashMap::new(), by_word: HashMap::new(), recent: Vec::new(), pending: None, asked: Vec::new(), dialog: crate::dialog::Dialog::load("data/raw/dailydialog/train_marked.json"), last: None, me: crate::me::Me::new(at, &format!("{}_self.txt", memory.trim_end_matches(".txt")), fresh), own: crate::understand::strike_or_load(at).map(|x| x.0).zip(crate::compose::Composer::learn("data/raw/dailydialog/train_marked.json")), cpt: crate::cpt::Cpt::load(1), wiki: crate::wiki::Wiki::load(), will: crate::will::Will::load(&format!("{}_will.json", memory.trim_end_matches(".txt"))) };
        // what it is doing, as it is: talking with whoever speaks to it (seen of itself, not told)
        talk.me.happened("doing", "i am talking with you");
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
            if r < std::f32::consts::LN_2 || self.cpt.as_ref().is_some_and(|c| c.small.contains(x)) {
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
    /// what strangers' words may still say for Darma (as Raven for Prim): a social formula (a short line very like a
    /// greeting, thanks or goodbye people said) answered as people answered it, or a feeling answered as people
    /// answer that feeling. Nothing else
    fn talk(&mut self, w: &[String]) -> Option<Reply> {
        let me = &self.me;
        let world = &self.core;
        let known = |x: &str| world.reg.contains_key(x) || me.core.reg.contains_key(x);
        let test = |ans: &str| me.names_a_stranger(ans, &known).or_else(|| me.contradicts(&words_of(ans)));
        // (feelings are not answered from the reading of emotions: it reads them too poorly, "I am Joose" as sadness)
        // a question of Darma that asks of a doing ("do you like music", "where do you live") is no formula: people's
        // answer would speak for Darma
        let question = self.dialog.as_ref().is_some_and(|d| w.first().is_some_and(|f| d.opens_question(f)));
        if crate::me::about_me(w) && question {
            if let Some(c) = self.cpt.as_ref() {
                let n = c.hear_as(&crate::me::to_me(w), "?");
                if n.iter().any(|x| x.pos == "VERB" && !matches!(x.lemma.as_str(), "be" | "do" | "go")) {
                    return None;
                }
            }
        }
        let last = self.last.clone();
        let d = self.dialog.as_mut()?;
        let short = w.iter().filter(|x| d.knows(x)).count() <= SOCIAL_WORDS;
        // a formula is answered as itself, whatever was said before (the conversation must not turn "hello" into
        // another turn's answer)
        let _ = last;
        let (ans, matched, s, _) = d.reply_where(w, None, None, &test)?;
        // only the formula itself: the answer's first sentence, and short ("Great, thanks." not the stranger's
        // "I was calling to change our meeting" after it)
        let end = ans.trim().find(['.', '!', '?']).map_or(ans.trim().len(), |k| k + 1);
        let ans = ans.trim()[..end].to_string();
        let formula = words_of(&ans).len() <= SOCIAL_WORDS && !words_of(&ans).is_empty();
        (short && formula && s >= SOCIAL_AT).then(|| Reply { said: ans, why: format!("a social formula; people answered this to \"{matched}\" ({s:.2})"), kind: "talk", recall: String::new(), ask: String::new() })
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
        // its will lives it: the things spoken of, and whether it could answer (a question it could not answer, one it
        // did; a stance asked is not counted)
        let things = self.things(&words_of(line));
        if !things.is_empty() && rep.kind != "will" {
            let answered = if rep.kind == "unknown" || rep.said.starts_with("I don't know") { Some(false) } else if self.asks(line, &words_of(line)) { Some(true) } else { None };
            self.will.lived(&things, answered);
        }
        rep
    }

    /// the speaker's mark on an answer: its will lives it too
    pub fn marked(&mut self, line: &str, good: bool) {
        let things = self.things(&words_of(line));
        if !things.is_empty() {
            self.will.marked(&things, good);
        }
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
                (None, _) => r("I don't know that about you yet.".into(), format!("you have not told me: {}", q.join(" ")), "unknown"),
            }
        } else if asked {
            match self.core.verify(&q) {
                Some((_, z)) if z >= 3.0 => r("Yes.".into(), format!("you told me: {}", q.join(" ")), "yes"),
                Some(_) => r("Not as far as I know.".into(), format!("you have not told me: {}", q.join(" ")), "no"),
                None => r("I don't know that about you yet.".into(), format!("you have not told me: {}", q.join(" ")), "unknown"),
            }
        } else {
            // what the speaker says of themselves begins where "i" or "my" begins ("hello darma, I am Joose": "i am
            // joose"); without it ("tell me a joke") it is only talk
            let Some(k) = w.iter().position(|x| x == "i" || x == "my") else {
                return self.talk(w).unwrap_or_else(|| r(TEACH_ME.into(), "nothing people said to this is like it".into(), "unknown"));
            };
            let s = crate::me::to_you(&w[k..]);
            let i = self.said.len();
            self.learn(&s);
            use std::io::Write;
            let _ = writeln!(self.log, "{}", s.join(" "));
            // it asks of what you did in a sentence of its own (its grammar); if it can make none, it answers as people
            // did; if nothing people said fits, it says it keeps it, and what it recalls of it
            if let Some((u, c)) = self.own.as_ref() {
                let me = &self.me;
                let world = &self.core;
                let known = |x: &str| world.reg.contains_key(x) || me.core.reg.contains_key(x);
                let test = |a: &str| me.names_a_stranger(a, &known).or_else(|| me.contradicts(&words_of(a)));
                if let Some((q, why)) = c.own_question(u, w, &test) {
                    return Reply { said: q, why: format!("{why}; I keep of you: {}", s.join(" ")), kind: "own", recall: String::new(), ask: String::new() };
                }
            }
            // what the speaker says of themselves is theirs: no one else's words answer it
            let mut rep = r("I'll remember that about you.".into(), s.join(" "), "told");
            rep.recall = self.recall(&s, Some(i)).unwrap_or_default();
            rep
        }
    }

    /// what was said of Darma itself: kept in its self (told), or asked of its self (questions); a question of
    /// itself is answered only from what it knows of itself, never as some person in a conversation answered it
    fn of_me(&mut self, asked: bool, w: &[String]) -> Reply {
        let mut rep = self.of_me_inner(asked, w);
        // not knowing something of itself, it says so and turns the question to the one asking, as people do
        if rep.kind == "unknown" && asked {
            if let Some(b) = self.dialog.as_ref().and_then(|d| d.turn_back.clone()) {
                rep.said = format!("I don't know yet. {b}");
                rep.why = format!("{}; so I turn it to you, as people do", rep.why);
            }
        }
        rep
    }

    fn of_me_inner(&mut self, asked: bool, w: &[String]) -> Reply {
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
                None if !of_what_i_am => self.talk(w).unwrap_or_else(|| r("I don't know that about myself yet.".into(), format!("I know nothing of myself that answers \"{}\"", q.join(" ")), "unknown")),
                None => r("I don't know that about myself yet.".into(), format!("I know nothing of myself that answers \"{}\"", q.join(" ")), "unknown"),
            }
        } else if asked {
            // "are you ...": a question of what it is, answered from itself only; any other ("how are you?") is talk,
            // answered as people did, but never against what it knows of itself
            let of_what_i_am = w.first().is_some_and(|x| self.copulas.contains(x)) && self.yes_no(w);
            match self.me.verify(&q, &self.copulas) {
                Some(true) => r("Yes.".into(), format!("I know of myself: {}", q.join(" ")), "yes"),
                Some(false) if of_what_i_am => r("Not as far as I know myself.".into(), format!("I do not know this of myself: {}", q.join(" ")), "no"),
                None if of_what_i_am => r("I don't know that about myself yet.".into(), format!("nobody has told me: {}", q.join(" ")), "unknown"),
                _ => self.talk(w).unwrap_or_else(|| r("I don't know that about myself yet.".into(), format!("nobody has told me: {}", q.join(" ")), "unknown")),
            }
        } else if matches!(w.first().map(|s| s.as_str()), Some("you" | "your")) {
            // said of Darma itself ("you are curious"): kept in its self; and it answers as people did, if that fits
            let s = crate::me::to_me(w);
            self.me.tell(&s);
            r("I'll remember that about myself.".into(), s.join(" "), "told")
        } else {
            // spoken to Darma, but not saying what it is ("how you doing", "thank you"): talk
            self.talk(w).unwrap_or_else(|| r(TEACH_ME.into(), "nothing people said to this is like it, and it does not say what I am".into(), "unknown"))
        }
    }

    /// is this a question: it ends in "?", or opens with a word people's questions open with (learned)
    fn asks(&self, line: &str, w: &[String]) -> bool {
        line.trim_end().ends_with('?') || w.first().is_some_and(|f| self.dialog.as_ref().is_some_and(|d| d.opens_question(f)))
    }

    /// is this a question people answer with yes or no (learned from what they answered)?
    fn yes_no(&self, w: &[String]) -> bool {
        w.first().map_or(true, |f| self.dialog.as_ref().map_or(true, |d| d.asks_yes_no(f)))
    }

    /// the known sentences a question may be answered from, in the question's own words (turned to Darma's side
    /// when asked of it): of itself, or what it was told (of the speaker, of the world)
    fn known_for(&self, w: &[String]) -> (Vec<String>, Vec<String>, &'static str) {
        if crate::me::about_me(w) {
            (crate::me::to_me(w), self.me.facts(), "me")
        } else if crate::me::about_speaker(w) {
            (crate::me::to_you(w), self.said.iter().map(|s| s.join(" ")).collect(), "you")
        } else {
            (w.to_vec(), self.said.iter().map(|s| s.join(" ")).collect(), "world")
        }
    }

    /// the things a line speaks of: its nouns as hearing reads them, in base form, beyond the commonest words and the
    /// ones speaking (I, you)
    fn things(&self, w: &[String]) -> Vec<String> {
        self.things_said(w).into_iter().map(|x| x.0).collect()
    }

    /// the things a line speaks of, each in base form (as a noun) and as it was said: its nouns, and what stands as a
    /// doer or done-to whatever hearing read its class as ("music" read as a describing word), beyond the commonest
    /// words and the ones speaking
    fn things_said(&self, w: &[String]) -> Vec<(String, String)> {
        let Some(c) = self.cpt.as_ref() else { return Vec::new() };
        let mut v: Vec<(String, String)> = c
            .hear(w)
            .into_iter()
            .filter(|x| (matches!(x.pos.as_str(), "NOUN" | "PROPN") || matches!(x.rel.as_str(), "obj" | "nsubj" | "obl") || x.head == 0) && !matches!(x.pos.as_str(), "PRON" | "DET" | "AUX" | "VERB" | "ADV" | "SCONJ" | "CCONJ" | "ADP" | "PART" | "PUNCT"))
            .filter(|x| !c.small.contains(&x.word) && !matches!(x.word.as_str(), "darma" | "i" | "you" | "me" | "yourself" | "myself"))
            .map(|x| (c.lemma_of(&x.word, "NOUN"), x.word))
            .collect();
        v.sort();
        v.dedup_by(|a, b| a.0 == b.0);
        v
    }

    /// its will asked or told: "do you like X", "what do you think about X" (asked of its stance), "you like X" (told
    /// a stance: no reason, it chooses itself). The verbs of liking and the "think about" frame are set by hand
    fn stance(&mut self, line: &str, w: &[String]) -> Option<Reply> {
        if !crate::me::about_me(w) {
            return None;
        }
        let liking = ["like", "love", "enjoy", "hate", "prefer"];
        let lemmas: Vec<String> = self.cpt.as_ref()?.hear(&crate::me::to_me(w)).into_iter().map(|x| x.lemma).collect();
        let think = lemmas.iter().any(|x| x == "think");
        let think_about = think && w.iter().any(|x| x == "about" || x == "of");
        // what it thinks of a claim ("do you think that's wrong"): it has no judgment of its own on it, and says so
        // (its drives weigh learning and talking, not right and wrong)
        if think && !think_about && self.asks(line, w) {
            return Some(Reply { said: "I don't know what I think about that yet.".into(), why: "my will has nothing of its own to judge that by".into(), kind: "will", recall: String::new(), ask: String::new() });
        }
        if !lemmas.iter().any(|x| liking.contains(&x.as_str())) && !think_about {
            return None;
        }
        // the thing: the last one said (in "do you like music" the one liked)
        let ts = self.things_said(w);
        let (mut thing, mut said_as) = w.iter().rev().find_map(|x| ts.iter().find(|t| t.1 == *x).cloned())?;
        // a name of two words ("albert einstein") when its article is one
        if let Some(i) = w.iter().rposition(|x| *x == said_as).filter(|&i| i > 0) {
            let two = format!("{} {}", w[i - 1], w[i]);
            if self.wiki.as_ref().is_some_and(|wk| wk.lead(&two).is_some()) {
                thing = two.clone();
                said_as = two.split(' ').map(|p| p.get(..1).map(|f| f.to_uppercase() + &p[1..]).unwrap_or_default()).collect::<Vec<_>>().join(" ");
            }
        }
        let asked = self.asks(line, w);
        // what it read of it, said inside its own sentence (a capital kept only for a name)
        let read = self.wiki.as_ref().and_then(|wk| wk.lead(&thing)).map(|r| {
            let first = r.split(' ').next().unwrap_or("").to_lowercase();
            let name = self.cpt.as_ref().is_some_and(|c| c.hear(&words_of(&r)).first().is_some_and(|x| x.pos == "PROPN")) && first != thing && crate::cpt::base_of(&first) != thing;
            let article = matches!(first.as_str(), "a" | "an" | "the");
            if !article && (name || said_as.chars().next().is_some_and(|c| c.is_uppercase())) { r } else { r.get(..1).map(|f| f.to_lowercase() + &r[1..]).unwrap_or(r) }
        });
        let (s, reasons, before) = self.will.choose(&thing, read.as_deref());
        let mut said = crate::will::say(&said_as, &s, &reasons, before.as_deref());
        if !asked {
            said = format!("I choose that myself. {said}");
        }
        Some(Reply { said, why: format!("my will: weighed {:.2} (curiosity, understanding, connection by my temperament {:.2}/{:.2}/{:.2})", s.weight, self.will.temperament[0], self.will.temperament[1], self.will.temperament[2]), kind: "will", recall: String::new(), ask: String::new() })
    }

    /// conscious parallel thinking, v0: the question heard into a structure, every known sentence heard the same
    /// way, and thoughts side by side: recall (the known sentences that fill the question's structure), nearest
    /// (what is known of the same one in the same way), ask back (when it knows nothing of it). The workspace takes the
    /// strongest; nothing when no thought has anything (the older ways answer then)
    fn think(&self, line: &str, w: &[String]) -> Option<Reply> {
        let cpt = self.cpt.as_ref()?;
        if !self.asks(line, w) {
            return None;
        }
        let (q, known, whose) = self.known_for(w);
        let qn = cpt.hear_as(&q, "?");
        let say = |f: &[crate::cpt::Node]| self.me.spell(&crate::cpt::say(f));
        // asked to tell of itself ("tell me something about yourself"): it says what it knows of itself, what it is
        // first ("tell" asks for what is known: a rule set by hand)
        if whose == "me" && qn.iter().any(|x| x.lemma == "tell") && q.iter().any(|x| x == "myself" || x == "i" || x == "me") {
            let mut fs: Vec<Vec<crate::cpt::Node>> = known.iter().map(|k| cpt.hear(&words_of(k))).collect();
            fs.sort_by_key(|f| !crate::cpt::names_a_thing(f));
            // each says something the ones before did not (a word beyond the commonest)
            let mut told: std::collections::HashSet<String> = Default::default();
            let mut said: Vec<String> = Vec::new();
            for f in &fs {
                let new: Vec<String> = f.iter().map(|x| x.lemma.clone()).filter(|l| !cpt.small.contains(l) && !told.contains(l)).collect();
                if new.is_empty() || said.len() == 3 {
                    continue;
                }
                told.extend(new);
                said.push(say(f));
            }
            return Some(Reply { said: said.join(" "), why: format!("heard: {}; asked to tell of myself: what I know of myself", crate::cpt::Cpt::show(&qn)), kind: "cpt", recall: String::new(), ask: String::new() });
        }
        let heard = crate::cpt::Cpt::show(&qn);
        let hole = crate::cpt::hole(&qn);
        let facts: Vec<(String, Vec<crate::cpt::Node>)> = known.iter().map(|k| (k.clone(), cpt.hear(&words_of(k)))).collect();
        let r = |said: String, why: String| Some(Reply { said, why, kind: "cpt", recall: String::new(), ask: String::new() });
        // recall: the known sentences that answer it, the best fitting first (at most two)
        // of those that answer: the best fitting, then the one saying least beyond the question (the most to the point)
        let mut full: Vec<(f32, &(String, Vec<crate::cpt::Node>))> = facts.iter().filter_map(|x| crate::cpt::fit(&qn, &x.1).map(|s| (s, x))).collect();
        full.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1 .1.len().cmp(&b.1 .1.len())));
        full.dedup_by(|a, b| a.1 .0 == b.1 .0);
        // a second answer only if it fits as well as the first
        if full.len() > 1 && (full[1].0 < full[0].0 || !crate::cpt::names_a_thing(&full[0].1 .1) || !crate::cpt::names_a_thing(&full[1].1 .1)) {
            full.truncate(1);
        }
        if !full.is_empty() {
            let said: Vec<String> = full.iter().take(2).map(|x| say(&x.1 .1)).collect();
            let yes = if hole.is_none() { "Yes. " } else { "" };
            return r(format!("{yes}{}", said.join(" ")), format!("heard: {heard}; recall: {}", full.iter().take(2).map(|x| format!("\"{}\" (fits {:.2})", x.1 .0, x.0)).collect::<Vec<_>>().join(", ")));
        }
        // nearest: what is known of the same doer in the same way
        let mut near: Vec<(f32, &(String, Vec<crate::cpt::Node>))> = facts.iter().filter_map(|x| crate::cpt::near(&qn, &x.1, &cpt.small).map(|s| (s, x))).collect();
        near.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1 .1.len().cmp(&b.1 .1.len())));
        near.dedup_by(|a, b| a.1 .0 == b.1 .0);
        if !near.is_empty() {
            let said: Vec<String> = near.iter().take(2).map(|x| say(&x.1 .1)).collect();
            let dunno = if whose == "me" { "I don't know that about myself." } else if whose == "you" { "I don't know that about you." } else { "I don't know." };
            return r(format!("{dunno} {}", said.join(" ")), format!("heard: {heard}; nothing I know answers it; nearest: what I know of {} in the same way", crate::cpt::doer(&qn).unwrap_or_default()));
        }
        // what it read: of the world, Wikipedia's sentences that answer it
        if whose == "world" {
            if let Some((_, f)) = self.wiki.as_ref().and_then(|wk| wk.answer(&qn, &cpt.small).into_iter().next().map(|(s, f)| (s, (f.text.clone(), wk.titles[f.title as usize].clone())))) {
                let yes = if hole.is_none() { "Yes. " } else { "" };
                return r(format!("{yes}{}", f.0), format!("heard: {heard}; I read it in Wikipedia (\"{}\")", f.1));
            }
        }
        // ask back: when it knows nothing of what was asked, it asks as people ask when they need more
        if whose != "me" {
            let back = self.dialog.as_ref().and_then(|d| d.ask_back.iter().find(|x| x.to_lowercase().contains("you")).or(d.ask_back.first()).cloned());
            if let Some(b) = back {
                return r(format!("I don't know. {b}"), format!("heard: {heard}; I know nothing of it, so I ask as people ask when they need more"));
            }
        }
        None
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
                    let mut rep = r("Thank you, I'll remember that.", q.join(" "), "told");
                    rep.recall = self.recall(&q, None).unwrap_or_default();
                    return rep;
                }
                Some("no") => return r("Okay.", format!("not: {}", q.join(" ")), "told"),
                _ => {}
            }
        }
        // a greeting at the start of a longer line ("hi I am Joose"): greeted back, the rest heard on its own; a name
        // the rest gives is said with the greeting
        if w.len() > 2 {
            for n in 1..=2 {
                if crate::me::about_me(&w[..n]) || crate::me::about_speaker(&w[..n]) || w[..n].iter().any(|x| crate::cpt::WH.contains(&x.as_str()) || self.copulas.contains(x)) {
                    break;
                }
                // a greeting: people's very words ("Hi.") answered as a formula
                let part = w[..n].join(" ");
                if let Some(g) = self.talk(&w[..n]).filter(|g| g.kind == "talk" && g.why.to_lowercase().contains(&format!("\"{part}"))) {
                    let rest = w[n..].join(" ");
                    let mut rep = self.line_inner(&rest);
                    let name = crate::me::about_speaker(&w[n..]).then_some(()).and(self.cpt.as_ref()).and_then(|c| {
                        let h = c.hear(&crate::me::to_you(&w[n..]));
                        let you = h.iter().any(|x| x.lemma == "you" || x.lemma == "your");
                        let be = h.iter().any(|x| x.lemma == "be");
                        h.iter().rev().find(|x| you && be && !matches!(x.pos.as_str(), "DET" | "ADP" | "AUX" | "CCONJ" | "SCONJ" | "PART" | "PUNCT" | "PRON" | "ADJ" | "VERB") && !c.small.contains(&x.word)).map(|x| {
                            let mut ch = x.word.chars();
                            ch.next().map(|f| f.to_uppercase().collect::<String>() + ch.as_str()).unwrap_or_default()
                        })
                    });
                    let gs = g.said.trim();
                    let (body, mark) = match gs.char_indices().last() {
                        Some((i, c)) if ".!?".contains(c) => (&gs[..i], &gs[i..]),
                        _ => (gs, "."),
                    };
                    let greet = match name {
                        Some(nm) => format!("{body}, {nm}{mark}"),
                        None => format!("{body}{mark}"),
                    };
                    rep.said = format!("{greet} {}", rep.said);
                    rep.why = format!("{}; then: {}", g.why, rep.why);
                    return rep;
                }
            }
        }
        if let Some(rep) = self.stance(line, &w) {
            return rep;
        }
        if let Some(rep) = self.think(line, &w) {
            return rep;
        }
        if w.is_empty() {
            r("", String::new(), "none")
        } else if crate::me::about_me(&w) {
            let asked = self.asks(line, &w);
            self.of_me(asked, &w)
        } else if crate::me::about_speaker(&w) {
            let asked = self.asks(line, &w);
            self.of_you(asked, &w)
        } else if self.asks(line, &w) && !hole && !self.yes_no(&w) {
            // a question people do not answer with yes or no ("how's it going?"): talk
            self.talk(&w).unwrap_or_else(|| r(TEACH_ME, "nothing people said to this is like it".into(), "unknown"))
        } else if self.asks(line, &w) && !hole {
            // what it was told answers first; if that cannot say, it answers as people did
            // in statement order ("is rex a dog" -> "rex is a dog"), as questions of you and of Darma are
            match self.core.verify_reasoned(&crate::me::statement_order(&w), &self.copulas, 3.0) {
                Some((true, why)) if why.is_empty() => r("Yes.", String::new(), "yes"),
                Some((true, why)) => r("Yes.", format!("because {}", why.join("; ")), "yes"),
                Some((false, _)) => self.talk(&w).unwrap_or_else(|| r("Not as far as I know.", "I know these words, but not this of them".into(), "no")),
                None => self.talk(&w).unwrap_or_else(|| r("I don't know.", "some of these words I was never told".into(), "unknown")),
            }
        } else if hole {
            match self.core.answer_reasoned(&w, &self.copulas) {
                (Some(x), why) if why.is_empty() => r(&x, String::new(), "answer"),
                (Some(x), why) => r(&x, format!("because {}", why.join("; ")), "answer"),
                (None, _) => self.talk(&w).unwrap_or_else(|| r(TEACH_ME, "nothing I was told stands out as the answer".into(), "unknown")),
            }
        } else {
            // a social formula or a feeling ("hello", "good night") is answered, not kept as a fact; anything else is
            // kept, and it says it remembers, and what it recalls of it
            if let Some(t) = self.talk(&w) {
                return t;
            }
            let i = self.said.len();
            self.learn(&w);
            use std::io::Write;
            let _ = writeln!(self.log, "{}", w.join(" "));
            let mut rep = r("I'll remember that.", format!("{} things told", self.core.told), "told");
            rep.recall = self.recall(&w, Some(i)).unwrap_or_default();
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
        } else if head.starts_with("POST /rate") {
            // a mark on an answer (sensible or not), kept for measuring: one line of JSON each
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("darma_ratings.jsonl") {
                let _ = writeln!(f, "{}", body.trim().replace('\n', " "));
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(body.trim()) {
                let g = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
                talk.marked(&g("line"), g("mark") == "good");
            }
            ("application/json", "{}".to_string())
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

/// `darma ratings`: the marks given on the page: the share of sensible answers, by how Darma made them
pub fn ratings() {
    let Ok(s) = std::fs::read_to_string("darma_ratings.jsonl") else { return println!("no marks yet (darma_ratings.jsonl)") };
    // the last mark of each answer counts (a mark can be changed)
    let mut last: HashMap<(String, String), (String, String)> = HashMap::new();
    for l in s.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(l) else { continue };
        let g = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
        last.insert((g("line"), g("said")), (g("kind"), g("mark")));
    }
    let mut by: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (_, (kind, mark)) in &last {
        let e = by.entry(kind.clone()).or_default();
        e.1 += 1;
        e.0 += (mark == "good") as usize;
    }
    let (g, n) = by.values().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    println!("{g} of {n} answers marked sensible ({:.0}%)", pct(g, n));
    let name = |k: &str| match k {
        "own" => "Darma's own sentence",
        "talk" => "a person's line",
        "answer" => "from its memory",
        "yes" | "no" => "yes/no from its memory",
        "told" => "it remembers",
        "unknown" => "it does not know",
        _ => "other",
    };
    for (k, (g, n)) in &by {
        println!("  {:<24} {g} of {n} ({:.0}%)", name(k), pct(*g, *n));
    }
    println!("not sensible:");
    for ((line, said), (kind, mark)) in last.iter().filter(|x| x.1 .1 == "bad").take(15) {
        println!("  {line:<40} -> {said}   [{}]", name(kind));
        let _ = mark;
    }
}
