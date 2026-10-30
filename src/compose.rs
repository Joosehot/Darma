//! Answering from understanding, in Darma's own words: what to answer is decided from what the line does, feels and
//! is about; how to say it comes from forms people use, filled with what Darma understood.
//!
//! Learned in one pass from DailyDialog's train split (its act and emotion marks, and the turns themselves):
//!   - how people answer each act (the act of the answer: a question is answered by informing, ...);
//!   - forms: an answer with the speaker's topic in it becomes a form with a place for a topic ("What color is your
//!     ___?" from "I bought a new car" -> "What color is your car?"), kept with the line it answered and the act and
//!     emotion of that line; it is kept as a form only if people filled it with at least two different topics. The
//!     topic of a line is its rarest thing (a word people marked in UD as a subject or object) that is rare at all
//!     (in fewer than one turn in a hundred);
//!   - how people answer a feeling: the short answers people gave to lines marked sad, happy, angry, ...
//! Answering: Darma reads the line (act, emotion, topic). A clear feeling is answered as people answer that feeling.
//! A line with a topic is answered with the form whose line is most like this one, among the forms of lines of the
//! same act and emotion, filled with this line's topic. Otherwise with what people most often answer that act.

use crate::lexicon::tokens;
use crate::understand::{Understanding, ACTS, EMOTIONS};
use serde::Deserialize;
use std::collections::HashMap;

pub const SLOT: &str = "___";
/// the places of a two-place form: what the line did (its verb) and what to (its object)
pub const DO: &str = "[DO]";
pub const WHAT: &str = "[WHAT]";

#[derive(Deserialize)]
struct Marked {
    dialog: Vec<String>,
    act: Vec<i64>,
    emotion: Vec<i64>,
}

pub struct Form {
    /// the answer, with the place of the topic marked
    pub words: Vec<String>,
    /// the line it answered (its words, the topic left out)
    source: Vec<String>,
    pub count: u32,
    /// how many different topics people filled it with (a form is a form only if it took more than one)
    pub topics: u32,
    /// for a two-place form: the shape of the verb it took (only a verb of that shape fills it)
    pub shape: &'static str,
}

/// a verb's shape, by its ending (the endings are written by hand): doing, done/did, does, do
pub fn shape(v: &str) -> &'static str {
    if v.ends_with("ing") {
        "ing"
    } else if v.ends_with("ed") {
        "ed"
    } else if v.ends_with('s') && !v.ends_with("ss") {
        "s"
    } else {
        "base"
    }
}

pub struct Composer {
    /// forms by (act, emotion) of the line answered
    forms: HashMap<(usize, usize), Vec<Form>>,
    /// two-place forms by (act, emotion) of the line answered: answers that took up the line's verb and its object
    /// ("Why do you [DO] [WHAT]?"), with how many different fillings people gave them
    forms2: HashMap<(usize, usize), Vec<Form>>,
    /// the words people marked as verbs in UD (at least twice)
    verbs: std::collections::HashSet<String>,
    /// Darma's grammar and the question beginnings people use (its own sentences are made with these)
    grammar: Option<(crate::grammar::Grammar, crate::grammar::Openings)>,
    /// short answers to each feeling, most common first (emotion index > 0)
    to_feeling: Vec<Vec<(String, u32)>>,
    /// short answers to each act with no topic, most common first
    to_act: Vec<Vec<(String, u32)>>,
    /// how people answer each act: counts of the answer's act
    pub act_after: Vec<Vec<u32>>,
    df: HashMap<String, u32>,
    turns: f32,
    /// the words people marked in UD as heads of noun phrases, with the share of their marks that were so: the things
    /// lines are about
    things: HashMap<String, f32>,
    /// every training turn's text (to measure how much of what Darma says is a copy)
    pub seen: std::collections::HashSet<String>,
}

fn act_of(a: i64) -> usize {
    (a as usize).saturating_sub(1).min(3)
}

fn tidy_words(w: &[String]) -> String {
    let mut s = w.join(" ");
    for (a, b) in [(" s ", "'s "), (" m ", "'m "), (" t ", "'t "), (" re ", "'re "), (" ll ", "'ll "), (" ve ", "'ve "), (" d ", "'d ")] {
        s = format!(" {s} ").replace(a, b).trim().to_string();
    }
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

impl Composer {
    pub fn learn(path: &str) -> Option<Composer> {
        let data: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut turns = 0f32;
        let mut seen = std::collections::HashSet::new();
        for d in &data {
            for u in &d.dialog {
                let mut w = tokens(u);
                seen.insert(w.join(" "));
                w.sort();
                w.dedup();
                turns += 1.0;
                for x in w {
                    *df.entry(x).or_default() += 1;
                }
            }
        }
        // the things: words people marked in UD as the head of a noun phrase, with the share of their marks that were so
        // (price: a head; rock in "rock-bottom price": a modifier)
        let things: HashMap<String, f32> = std::fs::read_to_string("data/heads.json")
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, (u32, u32)>>(&s).ok())
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, (h, _))| *h >= 2)
            .map(|(w, (h, m))| (w, h as f32 / (h + m) as f32))
            .collect();
        // the copulas people marked in UD are not what a line did ("is", "was")
        let copulas: std::collections::HashSet<String> = std::fs::read_to_string("data/copulas.json")
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, usize>>(&s).ok())
            .map(|m| m.into_keys().collect())
            .unwrap_or_default();
        let verbs: std::collections::HashSet<String> = std::fs::read_to_string("data/verbs.json")
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, u32>>(&s).ok())
            .unwrap_or_default()
            .into_iter()
            .filter(|x| x.1 >= 2)
            .map(|x| x.0)
            .filter(|v| !copulas.contains(v))
            .collect();
        let grammar = crate::grammar::Grammar::learn("data/ud_short.json").and_then(|g| {
            let o = g.openings("data/ud_short.json", path, &verbs)?;
            Some((g, o))
        });
        let mut c = Composer { grammar, forms2: HashMap::new(), verbs, forms: HashMap::new(), to_feeling: vec![Vec::new(); 7], to_act: vec![Vec::new(); 4], act_after: vec![vec![0; 4]; 4], df, turns, things, seen };
        let mut feel: Vec<HashMap<String, u32>> = vec![HashMap::new(); 7];
        let mut plain: Vec<HashMap<String, u32>> = vec![HashMap::new(); 4];
        let mut forms: HashMap<(usize, usize), HashMap<String, (Vec<String>, Vec<String>, u32, std::collections::HashSet<String>)>> = HashMap::new();
        let mut forms2: HashMap<(usize, usize), HashMap<String, (Vec<String>, Vec<String>, u32, std::collections::HashSet<String>)>> = HashMap::new();
        for d in &data {
            for k in 0..d.dialog.len().saturating_sub(1) {
                let (line, ans) = (tokens(&d.dialog[k]), tokens(&d.dialog[k + 1]));
                if line.is_empty() || ans.is_empty() {
                    continue;
                }
                let (a, e) = (act_of(d.act[k]), (d.emotion[k] as usize).min(6));
                c.act_after[a][act_of(d.act[k + 1])] += 1;
                if ans.len() <= 7 {
                    if e > 0 {
                        *feel[e].entry(tidy_words(&ans)).or_default() += 1;
                    }
                    *plain[a].entry(tidy_words(&ans)).or_default() += 1;
                }
                // a two-place form: the answer took up the line's verb (and its object, if it had one)
                if let Some((vi, pi)) = c.clause(&line) {
                    let (v, pw) = (line[vi].clone(), pi.map(|k| line[k].clone()));
                    let takes_v = ans.iter().filter(|x| **x == v).count() == 1;
                    let takes_p = pw.as_ref().map_or(false, |p| ans.iter().filter(|x| *x == p).count() == 1);
                    // both places taken up (what was done, and to what), in an answer of four words or more
                    if takes_v && takes_p && (4..=12).contains(&ans.len()) {
                        let words: Vec<String> = ans
                            .iter()
                            .map(|x| if *x == v { DO.to_string() } else if Some(x) == pw.as_ref() && takes_p { WHAT.to_string() } else { x.clone() })
                            .collect();
                        let source: Vec<String> = line.iter().enumerate().filter(|(k, _)| *k != vi && Some(*k) != pi).map(|(_, x)| x.clone()).collect();
                        let filling = format!("{v} {}", pw.clone().unwrap_or_default());
                        let e2 = forms2.entry((a, 0)).or_default().entry(words.join(" ")).or_insert((words, source, 0, Default::default()));
                        e2.2 += 1;
                        e2.3.insert(filling);
                    }
                }
                if let Some(t) = c.topic(&line) {
                    if ans.len() <= 12 && ans.iter().filter(|x| **x == t).count() == 1 {
                        let words: Vec<String> = ans.iter().map(|x| if *x == t { SLOT.to_string() } else { x.clone() }).collect();
                        let source: Vec<String> = line.iter().filter(|x| **x != t).cloned().collect();
                        let e2 = forms.entry((a, e)).or_default().entry(words.join(" ")).or_insert((words, source, 0, Default::default()));
                        e2.2 += 1;
                        e2.3.insert(t.clone());
                    }
                }
            }
        }
        let top = |m: HashMap<String, u32>| {
            let mut v: Vec<(String, u32)> = m.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v.truncate(20);
            v
        };
        c.to_feeling = feel.into_iter().map(top).collect();
        c.to_act = plain.into_iter().map(top).collect();
        // a two-place form needs no second filling to be a form: with both what was done and to what taken out, it
        // already holds no one line's content ("What kind of [WHAT] do you [DO]?")
        c.forms2 = forms2
            .into_iter()
            .map(|(k, m)| (k, m.into_values().map(|(words, source, count, ts)| {
                let v = ts.iter().next().map(|f| f.split(' ').next().unwrap_or("").to_string()).unwrap_or_default();
                Form { words, source, count, topics: ts.len() as u32, shape: shape(&v) }
            }).collect()))
            .collect();
        c.forms = forms
            .into_iter()
            .map(|(k, m)| (k, m.into_values().filter(|x| x.3.len() >= 2).map(|(words, source, count, ts)| Form { words, source, count, topics: ts.len() as u32, shape: "" }).collect()))
            .collect();
        Some(c)
    }

    fn rarity(&self, w: &str) -> f32 {
        (self.turns / *self.df.get(w).unwrap_or(&1) as f32).ln()
    }

    /// a line's topic: its rarest thing (a word people marked as a subject or object), met at least twice, if that
    /// word is in fewer than one turn in a hundred
    pub fn topic(&self, line: &[String]) -> Option<String> {
        // the most head-like of the rare things (mostly a head, not a word that modifies one): rarity times its share
        // of marks as a head
        // a thing right before another thing modifies it ("rock" in "rock bottom price"): the head comes last
        let head_like = |x: &String| self.things.get(x).is_some_and(|h| *h >= 0.5);
        line.iter()
            .enumerate()
            .filter(|(i, x)| head_like(x) && !line.get(i + 1).is_some_and(|n| head_like(n)) && self.df.get(*x).is_some_and(|c| *c >= 2) && self.rarity(x) >= 100f32.ln())
            .map(|(_, x)| x)
            .max_by(|a, b| (self.rarity(a) * self.things[*a]).partial_cmp(&(self.rarity(b) * self.things[*b])).unwrap())
            .cloned()
    }

    /// a line's clause: its first verb (a word people marked as a verb, met at least twice) and the first head of a
    /// noun phrase after it within four words (its object), if any: positions in the line
    pub fn clause(&self, line: &[String]) -> Option<(usize, Option<usize>)> {
        let vi = line.iter().position(|x| self.verbs.contains(x) && self.df.get(x).is_some_and(|c| *c >= 2))?;
        let head_like = |x: &String| self.things.get(x).is_some_and(|h| *h >= 0.5);
        let pi = (vi + 1..(vi + 5).min(line.len())).find(|&k| head_like(&line[k]) && !line.get(k + 1).is_some_and(|n| head_like(n)));
        Some((vi, pi))
    }

    fn likeness(&self, a: &[String], b: &[String]) -> f32 {
        let w = |x: &String| self.rarity(x);
        let shared: f32 = a.iter().filter(|x| b.contains(x)).map(|x| w(x).powi(2)).sum();
        let na: f32 = a.iter().map(|x| w(x).powi(2)).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| w(x).powi(2)).sum::<f32>().sqrt();
        if na * nb == 0.0 { 0.0 } else { shared / (na * nb) }
    }

    /// Darma's own question of what the speaker says they did (and to what), made by its grammar; None if the line
    /// holds no such clause or the grammar makes nothing of it
    pub fn own_question(&self, u: &Understanding, line: &[String], test: &dyn Fn(&str) -> Option<String>) -> Option<(String, String)> {
        let r = u.read(line);
        let a = ACTS.iter().position(|x| *x == r.act).unwrap();
        let (vi, pi) = self.clause(line)?;
        let pi = pi?;
        let (g, o) = self.grammar.as_ref()?;
        if !line[..vi].iter().any(|x| x == "i" || x == "we") {
            return None;
        }
        for (q, b, n, with) in g.ask_with(o, &line[vi], &line[pi], "you", a).iter().take(10) {
            if test(q).is_none() {
                return Some((q.clone(), format!("read: {} ({}); what you did: \"{}\" to \"{}\"; my own question, in my grammar: a beginning people use after a {} (\"{b} ...\", weighs {n}{}), the verb bent to fit", r.act, r.emotion, line[vi], line[pi], r.act, if *with { "" } else { ", never with this verb" })));
            }
        }
        None
    }

    /// what Darma answers, how it decided (act and emotion read, topic, how it chose), every candidate it let go
    /// and why; `test` lets go of what goes against Darma's character
    pub fn answer(&self, u: &Understanding, d: &crate::dialog::Dialog, line: &[String], test: &dyn Fn(&str) -> Option<String>) -> Option<(String, String)> {
        let r = u.read(line);
        let a = ACTS.iter().position(|x| *x == r.act).unwrap();
        let e = EMOTIONS.iter().position(|x| *x == r.emotion).unwrap();
        let after = &self.act_after[a];
        let answer_act = ACTS[(0..4).max_by_key(|&i| after[i]).unwrap()];
        let head = format!("read: {} ({}), people answer it by {}", r.act, r.emotion, answer_act);
        // a feeling (not happiness: people's commonest answer to it is a thank-you, a stock that misfires): answered
        // as people answer that feeling
        if e > 0 && EMOTIONS[e] != "happiness" {
            if let Some((s, n)) = self.to_feeling[e].iter().find(|(s, _)| test(s).is_none()) {
                return Some((s.clone(), format!("{head}; to {} people most often say this ({n} times)", r.emotion)));
            }
        }
        // a clause (what the line did, and to what), said by the speaker of themselves: Darma asks of it in a sentence
        // of its own, made by its grammar (a beginning people use after a line of this act, the verb bent to fit, the
        // object with its article)
        if let (Some((vi, Some(pi))), Some((g, o))) = (self.clause(line), self.grammar.as_ref()) {
            if line[..vi].iter().any(|x| x == "i" || x == "we") {
                for (q, b, n, with) in g.ask_with(o, &line[vi], &line[pi], "you", a).iter().take(10) {
                    if test(q).is_none() {
                        return Some((q.clone(), format!("{head}; what you did: \"{}\" to \"{}\"; my own question: a beginning people use after a {} (\"{b} ...\", weighs {n}{}), the verb bent to fit", line[vi], line[pi], r.act, if *with { "" } else { ", never with this verb" })));
                    }
                }
            }
        }
        // (the two-place forms seen once are not used: they carry their first conversation's leftovers)
        if let (Some((vi, pi)), false) = (self.clause(line), true) {
            let rest: Vec<String> = line.iter().enumerate().filter(|(k, _)| *k != vi && Some(*k) != pi).map(|(_, x)| x.clone()).collect();
            let here = d.meaning_state(&rest);
            let vshape = shape(&line[vi]);
            let pool: Vec<&Form> = self.forms2.get(&(a, 0)).map(|v| v.iter().filter(|f| f.shape == vshape && (pi.is_some() || !f.words.iter().any(|x| x == WHAT))).collect()).unwrap_or_default();
            let mut best: Vec<(f32, &Form)> = pool.iter().map(|f| (here.sim(&d.meaning_state(&f.source)).max(0.0), *f)).collect();
            best.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            for (s, f) in best.iter().take(20) {
                let said = tidy_words(&f.words.iter().map(|x| if x == DO { line[vi].clone() } else if x == WHAT { line[pi.unwrap()].clone() } else { x.clone() }).collect::<Vec<_>>());
                if *s > 0.0 && test(&said).is_none() {
                    return Some((said, format!("{head}; what it did: \"{}\"{}; a form people answered such lines with ({s:.2}, {} fillings): \"{}\"", line[vi], pi.map(|k| format!(", to \"{}\"", line[k])).unwrap_or_default(), f.topics, f.words.join(" "))));
                }
            }
        }
        // a topic: the form whose line is most like this one, filled with the topic
        if let Some(t) = self.topic(line) {
            let rest: Vec<String> = line.iter().filter(|x| **x != t).cloned().collect();
            let pool: Vec<&Form> = self.forms.get(&(a, e)).map(|v| v.iter().collect()).unwrap_or_default();
            // the form whose line is most like this one; a form taken with many topics is more surely a form
            // the form whose line resonates most in meaning with this one (the topic left out of both)
            let here = d.meaning_state(&rest);
            let mut best: Vec<(f32, &Form)> = pool.iter().map(|f| (here.sim(&d.meaning_state(&f.source)).max(0.0), *f)).collect();
            best.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            for (s, f) in best.iter().take(20) {
                let said = tidy_words(&f.words.iter().map(|x| if x == SLOT { t.clone() } else { x.clone() }).collect::<Vec<_>>());
                if test(&said).is_none() {
                    return Some((said, format!("{head}; topic \"{t}\"; a form people answered such lines with ({s:.2}): \"{}\"", f.words.join(" "))));
                }
            }
        }
        // else what people most often answer this act
        let (s, n) = self.to_act[a].iter().find(|(s, _)| test(s).is_none())?;
        Some((s.clone(), format!("{head}; no topic; to a {} people most often say this ({n} times)", r.act)))
    }
}

/// `darma compose-eval`: DailyDialog's test dialogues; Darma's own answer to each turn against the person's real
/// answer (shared words, F1), and how many of its answers are word for word some training turn (copies)
pub fn eval(dim: usize, seed: u64) {
    let at = crate::lexicon::Atoms::new(dim, seed);
    let Some((u, _)) = crate::understand::strike_or_load(&at) else { return println!("download DailyDialog first") };
    let c = Composer::learn("data/raw/dailydialog/train_marked.json").unwrap();
    let dl = crate::dialog::Dialog::load("data/raw/dailydialog/train_marked.json").unwrap();
    let n2: usize = c.forms2.values().map(|v| v.len()).sum();
    println!("two-place forms: {n2}; some: {}", {
        let mut all: Vec<&Form> = c.forms2.values().flatten().collect();
        all.sort_by(|a, b| b.topics.cmp(&a.topics));
        all.iter().take(8).map(|f| format!("\"{}\" ({})", f.words.join(" "), f.topics)).collect::<Vec<_>>().join(", ")
    });
    println!("how people answer each act (share of the answer's act):");
    for (i, row) in c.act_after.iter().enumerate() {
        let n: u32 = row.iter().sum();
        println!("  {:<11} -> {}", ACTS[i], ACTS.iter().zip(row).map(|(a, k)| format!("{a} {:.0}%", 100.0 * *k as f32 / n as f32)).collect::<Vec<_>>().join(", "));
    }
    let mut test: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string("data/raw/dailydialog/test_marked.json").unwrap()).unwrap();
    // a short run: the first 150 test dialogues
    test.truncate(150);
    let f1 = |a: &[String], b: &[String]| -> f32 {
        let common = a.iter().filter(|x| b.contains(x)).count() as f32;
        if common == 0.0 { 0.0 } else { let p = common / a.len() as f32; let r = common / b.len() as f32; 2.0 * p * r / (p + r) }
    };
    let (mut n, mut f, mut copies, mut with_topic, mut feeling, mut two) = (0usize, 0f32, 0usize, 0usize, 0usize, 0usize);
    // fluency: the share of the word pairs of Darma's sentences that people's turns hold (DailyDialog train)
    let mut pairs: std::collections::HashSet<String> = std::collections::HashSet::new();
    {
        let tr: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string("data/raw/dailydialog/train_marked.json").unwrap()).unwrap();
        for d in &tr {
            for u in &d.dialog {
                let w = tokens(u);
                for p in w.windows(2) {
                    pairs.insert(p.join(" "));
                }
            }
        }
    }
    let (mut seen_pairs, mut all_pairs, mut own) = (0usize, 0usize, 0usize);
    let mut shown = 0;
    for d in &test {
        for k in 0..d.dialog.len().saturating_sub(1) {
            let (line, real) = (tokens(&d.dialog[k]), tokens(&d.dialog[k + 1]));
            if line.is_empty() || real.is_empty() {
                continue;
            }
            let Some((ans, why)) = c.answer(&u, &dl, &line, &|_| None) else { continue };
            let aw = tokens(&ans);
            n += 1;
            f += f1(&aw, &real);
            copies += c.seen.contains(&aw.join(" ")) as usize;
            with_topic += why.contains("topic \"") as usize;
            two += why.contains("what it did") as usize;
            own += why.contains("my own question") as usize;
            for p in aw.windows(2) {
                all_pairs += 1;
                seen_pairs += pairs.contains(&p.join(" ")) as usize;
            }
            feeling += (why.contains("people most often say this") && !why.contains("no topic")) as usize;
            if shown < 8 && n % 120 == 1 {
                shown += 1;
                println!("\n  said:   {}\n  Darma:  {ans}\n  person: {}\n  ({why})", crate::dialog::tidy(&d.dialog[k]), crate::dialog::tidy(&d.dialog[k + 1]));
            }
        }
    }
    println!("
{n} test turns answered: F1 with the person's real answer {:.3}; {:.1}% word for word a training turn; {:.1}% Darma's own grammar question, {:.1}% a two-place form, {:.1}% a one-place form, {:.1}% a feeling; fluency: {:.1}% of its word pairs are in people's turns", f / n as f32, 100.0 * copies as f32 / n as f32, 100.0 * own as f32 / n as f32, 100.0 * two as f32 / n as f32, 100.0 * with_topic as f32 / n as f32, 100.0 * feeling as f32 / n as f32, 100.0 * seen_pairs as f32 / all_pairs.max(1) as f32);
}
