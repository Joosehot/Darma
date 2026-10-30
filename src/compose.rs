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
}

pub struct Composer {
    /// forms by (act, emotion) of the line answered
    forms: HashMap<(usize, usize), Vec<Form>>,
    /// short answers to each feeling, most common first (emotion index > 0)
    to_feeling: Vec<Vec<(String, u32)>>,
    /// short answers to each act with no topic, most common first
    to_act: Vec<Vec<(String, u32)>>,
    /// how people answer each act: counts of the answer's act
    pub act_after: Vec<Vec<u32>>,
    df: HashMap<String, u32>,
    turns: f32,
    /// the words people marked in UD as a verb's subject or object, with the share of their uses that were so: the
    /// things lines are about
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
        // the things: words people marked in UD as a subject or an object, at least twice
        let mut things_n: HashMap<String, (u32, u32)> = HashMap::new();
        if let Ok(s) = std::fs::read_to_string("data/roles_train.json") {
            if let Ok(tagged) = serde_json::from_str::<Vec<crate::memory::Tagged>>(&s) {
                for tg in tagged {
                    for (w, r) in tg.words.iter().zip(&tg.roles) {
                        let e = things_n.entry(w.clone()).or_default();
                        e.1 += 1;
                        if r == "A" || r == "P" {
                            e.0 += 1;
                        }
                    }
                }
            }
        }
        let things = things_n.into_iter().filter(|x| x.1 .0 >= 2).map(|(w, (a, n))| (w, a as f32 / n as f32)).collect();
        let mut c = Composer { forms: HashMap::new(), to_feeling: vec![Vec::new(); 7], to_act: vec![Vec::new(); 4], act_after: vec![vec![0; 4]; 4], df, turns, things, seen };
        let mut feel: Vec<HashMap<String, u32>> = vec![HashMap::new(); 7];
        let mut plain: Vec<HashMap<String, u32>> = vec![HashMap::new(); 4];
        let mut forms: HashMap<(usize, usize), HashMap<String, (Vec<String>, Vec<String>, u32, std::collections::HashSet<String>)>> = HashMap::new();
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
        c.forms = forms
            .into_iter()
            .map(|(k, m)| (k, m.into_values().filter(|x| x.3.len() >= 2).map(|(words, source, count, ts)| Form { words, source, count, topics: ts.len() as u32 }).collect()))
            .collect();
        Some(c)
    }

    fn rarity(&self, w: &str) -> f32 {
        (self.turns / *self.df.get(w).unwrap_or(&1) as f32).ln()
    }

    /// a line's topic: its rarest thing (a word people marked as a subject or object), met at least twice, if that
    /// word is in fewer than one turn in a hundred
    pub fn topic(&self, line: &[String]) -> Option<String> {
        // the most thing-like of the rare things: rarity times the share of its uses as a subject or object
        line.iter()
            .filter(|x| self.things.contains_key(*x) && self.df.get(*x).is_some_and(|c| *c >= 2) && self.rarity(x) >= 100f32.ln())
            .max_by(|a, b| (self.rarity(a) * self.things[*a]).partial_cmp(&(self.rarity(b) * self.things[*b])).unwrap())
            .cloned()
    }

    fn likeness(&self, a: &[String], b: &[String]) -> f32 {
        let w = |x: &String| self.rarity(x);
        let shared: f32 = a.iter().filter(|x| b.contains(x)).map(|x| w(x).powi(2)).sum();
        let na: f32 = a.iter().map(|x| w(x).powi(2)).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| w(x).powi(2)).sum::<f32>().sqrt();
        if na * nb == 0.0 { 0.0 } else { shared / (na * nb) }
    }

    /// what Darma answers, how it decided (act and emotion read, topic, how it chose), every candidate it let go
    /// and why; `test` lets go of what goes against Darma's character
    pub fn answer(&self, u: &Understanding, line: &[String], test: &dyn Fn(&str) -> Option<String>) -> Option<(String, String)> {
        let r = u.read(line);
        let a = ACTS.iter().position(|x| *x == r.act).unwrap();
        let e = EMOTIONS.iter().position(|x| *x == r.emotion).unwrap();
        let after = &self.act_after[a];
        let answer_act = ACTS[(0..4).max_by_key(|&i| after[i]).unwrap()];
        let head = format!("read: {} ({}), people answer it by {}", r.act, r.emotion, answer_act);
        // a feeling: answered as people answer that feeling
        if e > 0 {
            if let Some((s, n)) = self.to_feeling[e].iter().find(|(s, _)| test(s).is_none()) {
                return Some((s.clone(), format!("{head}; to {} people most often say this ({n} times)", r.emotion)));
            }
        }
        // a topic: the form whose line is most like this one, filled with the topic
        if let Some(t) = self.topic(line) {
            let rest: Vec<String> = line.iter().filter(|x| **x != t).cloned().collect();
            let pool: Vec<&Form> = self.forms.get(&(a, e)).map(|v| v.iter().collect()).unwrap_or_default();
            // the form whose line is most like this one; a form taken with many topics is more surely a form
            let mut best: Vec<(f32, &Form)> = pool.iter().map(|f| (self.likeness(&rest, &f.source) * (1.0 + (f.topics as f32).ln()), *f)).collect();
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
    let Some(u) = crate::understand::learn_and_choose(&at) else { return println!("download DailyDialog first") };
    let c = Composer::learn("data/raw/dailydialog/train_marked.json").unwrap();
    println!("how people answer each act (share of the answer's act):");
    for (i, row) in c.act_after.iter().enumerate() {
        let n: u32 = row.iter().sum();
        println!("  {:<11} -> {}", ACTS[i], ACTS.iter().zip(row).map(|(a, k)| format!("{a} {:.0}%", 100.0 * *k as f32 / n as f32)).collect::<Vec<_>>().join(", "));
    }
    let test: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string("data/raw/dailydialog/test_marked.json").unwrap()).unwrap();
    let f1 = |a: &[String], b: &[String]| -> f32 {
        let common = a.iter().filter(|x| b.contains(x)).count() as f32;
        if common == 0.0 { 0.0 } else { let p = common / a.len() as f32; let r = common / b.len() as f32; 2.0 * p * r / (p + r) }
    };
    let (mut n, mut f, mut copies, mut with_topic, mut feeling) = (0usize, 0f32, 0usize, 0usize, 0usize);
    let mut shown = 0;
    for d in &test {
        for k in 0..d.dialog.len().saturating_sub(1) {
            let (line, real) = (tokens(&d.dialog[k]), tokens(&d.dialog[k + 1]));
            if line.is_empty() || real.is_empty() {
                continue;
            }
            let Some((ans, why)) = c.answer(&u, &line, &|_| None) else { continue };
            let aw = tokens(&ans);
            n += 1;
            f += f1(&aw, &real);
            copies += c.seen.contains(&aw.join(" ")) as usize;
            with_topic += why.contains("topic \"") as usize;
            feeling += (why.contains("people most often say this") && !why.contains("no topic")) as usize;
            if shown < 10 && n % 700 == 1 {
                shown += 1;
                println!("\n  said:   {}\n  Darma:  {ans}\n  person: {}\n  ({why})", crate::dialog::tidy(&d.dialog[k]), crate::dialog::tidy(&d.dialog[k + 1]));
            }
        }
    }
    println!("\n{n} test turns answered: shares words with the person's real answer F1 {:.3}; {:.1}% of the answers are word for word a training turn; {:.1}% filled a form with the line's topic; {:.1}% answered a feeling", f / n as f32, 100.0 * copies as f32 / n as f32, 100.0 * with_topic as f32 / n as f32, 100.0 * feeling as f32 / n as f32);
}
