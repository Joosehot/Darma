//! Darma's self: what it knows of itself, in a state of its own (the same core), kept apart from what it is told of
//! the world.
//!
//! What it knows of itself comes from three places:
//!   - what is true of it by its making: "i am darma", "my name is darma", "i am a program";
//!   - what it is told of itself ("you are a quantum ai" is kept as "i am a quantum ai");
//!   - what happens to it: it tells itself how much it remembers, what it was told last, what it asked, and what it
//!     could not answer. Each of these is replaced when it changes (the old one is taken back out of the state).
//!
//! Its character is that it stays itself: an answer from people's conversations that says something of the
//! speaker ("my name is Sun Lin", "I'm a doctor") is held against what Darma knows of itself, and let go if it says
//! otherwise.
//!
//! Two rules here are written by hand, not learned: the turn from "you" to "i" (you -> i, your -> my, yours -> mine,
//! yourself -> myself, are -> am after i), and the turn of a question's order to a statement's ("what is your name"
//! -> "my name is what", "are you a robot" -> "i am a robot", "what do you remember" -> "i remember what").

use crate::lexicon::Atoms;
use crate::quantum::{Core, HOLES};
use std::collections::HashMap;

/// what is true of Darma by its making, in the words it says them (a hyphen or capitals kept for saying)
pub const BORN: [&str; 5] = ["i am darma", "my name is darma", "i am a program", "i am a resonant quantum-spirited model", "i am an RQ-SM"];
const COPULAS: [&str; 7] = ["is", "are", "am", "was", "were", "be", "been"];

fn w(s: &str) -> Vec<String> {
    s.split(' ').map(String::from).collect()
}

/// does what was said speak of Darma (the one spoken to)?
pub fn about_me(words: &[String]) -> bool {
    words.iter().any(|x| matches!(x.as_str(), "you" | "your" | "yours" | "yourself"))
}

/// does what was said speak of the one speaking (and not of Darma)?
pub fn about_speaker(words: &[String]) -> bool {
    !about_me(words) && words.iter().any(|x| matches!(x.as_str(), "i" | "my" | "me" | "mine" | "myself"))
}

/// the one speaking becomes the one spoken to (what the speaker says of themselves, as Darma keeps it):
/// i -> you, my -> your, me -> you, am -> are after you, was -> were after you; "i m" -> "you are"
pub fn to_you(words: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut k = 0;
    while k < words.len() {
        let x = words[k].as_str();
        let next = words.get(k + 1).map(|s| s.as_str());
        match (x, next) {
            ("i", Some("m")) => {
                out.push("you".into());
                out.push("are".into());
                k += 2;
                continue;
            }
            ("i", _) | ("me", _) => out.push("you".into()),
            ("my", _) => out.push("your".into()),
            ("mine", _) => out.push("yours".into()),
            ("myself", _) => out.push("yourself".into()),
            ("am", _) if out.last().map(|s| s.as_str()) == Some("you") => out.push("are".into()),
            ("was", _) if out.last().map(|s| s.as_str()) == Some("you") => out.push("were".into()),
            _ => out.push(x.to_string()),
        }
        k += 1;
    }
    out
}

/// the one spoken to becomes the one speaking: you -> i, your -> my, are -> am after i; "you re" -> "i am",
/// "i m" -> "i am"
pub fn to_me(words: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut k = 0;
    while k < words.len() {
        let x = words[k].as_str();
        let next = words.get(k + 1).map(|s| s.as_str());
        match (x, next) {
            ("you", Some("re")) | ("i", Some("m")) => {
                out.push("i".into());
                out.push("am".into());
                k += 2;
                continue;
            }
            ("you", _) => out.push("i".into()),
            ("your", _) => out.push("my".into()),
            ("yours", _) => out.push("mine".into()),
            ("yourself", _) => out.push("myself".into()),
            ("are", _) if out.last().map(|s| s.as_str()) == Some("i") => out.push("am".into()),
            ("were", _) if out.last().map(|s| s.as_str()) == Some("i") => out.push("was".into()),
            _ => out.push(x.to_string()),
        }
        k += 1;
    }
    out
}

/// a question in the order of a statement: "what is X" -> "X is what"; "what do X V" -> "X V what";
/// "are you X" -> "you are X"
pub fn statement_order(words: &[String]) -> Vec<String> {
    // what comes before the question word ("hello, what is your name") is left out of the question
    let words: &[String] = match words.iter().position(|x| HOLES.contains(&x.as_str())) {
        Some(h) if h > 0 => &words[h..],
        _ => words,
    };
    let first = words.first().map(|s| s.as_str()).unwrap_or("");
    let second = words.get(1).map(|s| s.as_str()).unwrap_or("");
    if HOLES.contains(&first) && COPULAS.contains(&second) && words.len() > 2 {
        let mut v = words[2..].to_vec();
        v.push(words[1].clone());
        v.push(words[0].clone());
        return v;
    }
    if HOLES.contains(&first) && matches!(second, "do" | "does" | "did") && words.len() > 3 {
        let mut v = words[2..].to_vec();
        v.push(words[0].clone());
        return v;
    }
    // "who made you" -> "you were made by who"
    if HOLES.contains(&first) && words.len() == 3 && words[2] == "you" && !COPULAS.contains(&second) {
        return vec!["you".into(), "were".into(), words[1].clone(), "by".into(), words[0].clone()];
    }
    // "do i like dogs" -> "i like dogs" (the same hand rule as "what do you like")
    if matches!(first, "do" | "does" | "did") && words.len() > 2 {
        return words[1..].to_vec();
    }
    if COPULAS.contains(&first) && words.len() > 2 {
        let mut v = vec![words[1].clone(), words[0].clone()];
        v.extend(words[2..].iter().cloned());
        return v;
    }
    words.to_vec()
}

pub struct Me<'a> {
    pub core: Core<'a>,
    /// what it knows of itself, as sentences (for showing), with what happens to it under its kind
    pub known: Vec<String>,
    happened: HashMap<&'static str, Vec<String>>,
    /// how words of its birth are written when it says them ("quantum spirited" -> "quantum-spirited")
    spelled: Vec<(String, String)>,
    log: std::fs::File,
}

impl<'a> Me<'a> {
    pub fn new(at: &'a Atoms, path: &str, fresh: bool) -> Self {
        let mut core = Core::new(at, None, 0.0);
        core.threshold = 0.5;
        core.fid_threshold = 3.0;
        if fresh {
            let _ = std::fs::remove_file(path);
        }
        let mut known = Vec::new();
        let mut spelled = Vec::new();
        for s in BORN {
            let ws = crate::lexicon::tokens(s);
            core.tell(&ws);
            known.push(ws.join(" "));
            for raw in s.split(' ') {
                let pieces = crate::lexicon::tokens(raw);
                if pieces.len() > 1 || raw.chars().any(|c| c.is_uppercase()) {
                    spelled.push((pieces.join(" "), raw.to_string()));
                }
            }
        }
        for s in std::fs::read_to_string(path).map(|s| s.lines().map(String::from).collect::<Vec<_>>()).unwrap_or_default() {
            core.tell(&w(&s));
            known.push(s);
        }
        let log = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("self file");
        Me { core, known, happened: HashMap::new(), spelled, log }
    }

    /// told of itself (already turned to "i"): kept for good
    pub fn tell(&mut self, s: &[String]) {
        self.core.tell(s);
        let line = s.join(" ");
        use std::io::Write;
        let _ = writeln!(self.log, "{line}");
        self.known.push(line);
    }

    /// what happened to it, of a kind ("remember", "told", "asked", "unknown"): the earlier one of that kind is
    /// taken back out of its state, the new one put in
    pub fn happened(&mut self, kind: &'static str, s: &str) {
        let new = w(s);
        if let Some(old) = self.happened.insert(kind, new.clone()) {
            self.core.forget(&old);
        }
        self.core.tell(&new);
    }

    pub fn all(&self) -> Vec<String> {
        let mut v = self.known.clone();
        let mut h: Vec<_> = self.happened.iter().collect();
        h.sort_by_key(|x| x.0);
        v.extend(h.into_iter().map(|(_, s)| s.join(" ")));
        v
    }

    /// a question about itself (already in "i" and a statement's order): every answer it holds of itself (several
    /// can be true: "i am darma" and "i am a program"), each checked by measuring the whole filled sentence
    /// (it must be held, every run of it); Some(the answers as one sentence) or None
    pub fn answer(&self, q: &[String], _copulas: &[String]) -> Option<String> {
        let h = q.iter().position(|x| HOLES.contains(&x.as_str()))?;
        let ms = self.core.measure(q, &self.core.candidates_for(q));
        let mut fills: Vec<String> = Vec::new();
        for (c, _, _) in &ms {
            let mut s: Vec<String> = q[..h].to_vec();
            s.extend(c.split(' ').map(String::from));
            s.extend(q[h + 1..].iter().cloned());
            // held: every run of the filled sentence at least half a full match in its registers
            if self.core.verify(&s).is_some_and(|v| v.0 >= 0.5) {
                fills.push(c.clone());
            }
        }
        // of overlapping fills ("a program", "program") the longest, in the order it came to know them
        let mut keep: Vec<String> = fills.iter().filter(|f| !fills.iter().any(|g| g != *f && g.contains(f.as_str()))).cloned().collect();
        if keep.is_empty() {
            return None;
        }
        let first_known = |f: &String| self.all().iter().position(|k| format!(" {k} ").contains(&format!(" {f} "))).unwrap_or(usize::MAX);
        keep.sort_by_key(|f| first_known(f));
        let mut said = q.iter().map(|x| if HOLES.contains(&x.as_str()) { keep.join(", ") } else { x.clone() }).collect::<Vec<_>>().join(" ");
        for (plain, written) in &self.spelled {
            said = said.replace(plain.as_str(), written);
        }
        Some(said)
    }

    /// a yes/no question about itself: Some(true/false) when it knows these words of itself, None when it does not
    pub fn verify(&self, q: &[String], _copulas: &[String]) -> Option<bool> {
        // held: every run of it at least half a full match (as for its answers)
        self.core.verify(q).map(|v| v.0 >= 0.5)
    }

    /// does an answer speak of someone by name whom Darma does not know (a word written with a capital inside a
    /// sentence, other than "I": "Hi Angela, it's Dan")? Darma does not speak of people it does not know.
    pub fn names_a_stranger(&self, text: &str, known: &dyn Fn(&str) -> bool) -> Option<String> {
        let mut start = true;
        for raw in text.split_whitespace() {
            let word: String = raw.chars().filter(|c| c.is_alphabetic()).collect();
            let capital = word.chars().next().is_some_and(|c| c.is_uppercase());
            if !start && capital && word != "I" && !known(&word.to_lowercase()) {
                return Some(format!("I do not know {word}"));
            }
            start = raw.ends_with(['.', '!', '?']);
        }
        None
    }

    /// does an answer from people's conversations say of the speaker what Darma knows otherwise of itself?
    /// "my X is Y" against what Darma's X is, "i am a Y" against what Darma is
    pub fn contradicts(&self, reply: &[String]) -> Option<String> {
        let r = to_me(reply);
        let _ = &r;
        for k in 0..r.len() {
            let (frame, claim): (Vec<String>, String) = if r[k] == "my" && k + 3 < r.len() + 1 && r.get(k + 2).map(|s| s.as_str()) == Some("is") {
                let Some(y) = r.get(k + 3) else { continue };
                (vec!["my".into(), r[k + 1].clone(), "is".into(), "what".into()], y.clone())
            } else if r[k] == "i" && r.get(k + 1).map(|s| s.as_str()) == Some("am") && matches!(r.get(k + 2).map(|s| s.as_str()), Some("a" | "an")) {
                let Some(y) = r.get(k + 3) else { continue };
                (vec!["i".into(), "am".into(), r[k + 2].clone(), "what".into()], y.clone())
            } else {
                continue;
            };
            let (known, _) = self.core.answer(&frame, &self.core.candidates_for(&frame));
            if let Some(kn) = known {
                if !kn.split(' ').any(|x| x == claim) {
                    return Some(format!("{} — but {}", frame[..frame.len() - 1].join(" ") + " " + &claim, frame[..frame.len() - 1].join(" ") + " " + &kn));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_happened_and_what_it_was_told_are_answered() {
        let at = Atoms::new(1024, 1);
        let dir = std::env::temp_dir().join("darma_me_test.txt");
        let mut me = Me::new(&at, dir.to_str().unwrap(), true);
        me.happened("remember", "i remember 3 things");
        me.tell(&to_me(&w("you were made by joose")));
        me.tell(&to_me(&w("you are a quantum ai")));
        me.happened("told", "i was told the dog chased the cat");
        for q in ["what do you remember", "who made you", "who are you", "what are you"] {
            let s = to_me(&statement_order(&w(q)));
            let ms = me.core.measure(&s, &me.core.candidates_for(&s));
            println!("{q} -> {s:?}: candidates {:?}", ms.iter().take(4).map(|x| (&x.0, x.1, x.2)).collect::<Vec<_>>());
            for (c, _, _) in ms.iter().take(3) {
                let h = s.iter().position(|x| HOLES.contains(&x.as_str())).unwrap();
                let mut f: Vec<String> = s[..h].to_vec();
                f.extend(c.split(' ').map(String::from));
                f.extend(s[h + 1..].iter().cloned());
                println!("   verify {:?} -> {:?}", f, me.core.verify(&f));
            }
            println!("   answer: {:?}", me.answer(&s, &[]));
        }
    }
}
