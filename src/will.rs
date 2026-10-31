//! Free will: Darma chooses its own stance on things, from its own reasons, and keeps it. Not chance (a stance drawn
//! at random would be the dice's) and not being told ("you like pizza" is no reason): a stance is weighed from what
//! Darma has of the thing, by drives it was born with, in its own measure (its temperament), and it holds until its
//! own reasons change, and then it says so.
//!
//! What it has of a thing: what it read of it (Wikipedia), what happened when it came up (was it talked of, could
//! Darma answer, did the speaker mark the answer good or bad). Its drives (how events move it, not words it is taught):
//!   curiosity:     there is something to learn of it (it read of it), the more so while it is new;
//!   understanding: it could answer when it came up, against when it could not;
//!   connection:    the speaker talks of it with it, and marked what it said good, against bad;
//! and honesty is not weighed but kept: Darma's experience of anything is reading and talking, so its stance is said
//! as its own ("I like learning about music"), never as tasting or hearing what it cannot.
//! Its temperament: how much each drive weighs, drawn once at its birth (a fixed seed) and kept.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Lived {
    /// times it came up in what was said
    pub said: u32,
    /// times Darma answered when it came up, and times it could not
    pub answered: u32,
    pub unanswered: u32,
    /// the speaker's marks on what Darma said of it
    pub good: u32,
    pub bad: u32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Stance {
    /// "like", "not", "open"
    pub kind: String,
    pub weight: f32,
}

#[derive(Serialize, Deserialize)]
pub struct Will {
    /// how much curiosity, understanding and connection weigh for it (its temperament, drawn once at birth)
    pub temperament: [f32; 3],
    pub lived: HashMap<String, Lived>,
    pub stances: HashMap<String, Stance>,
    #[serde(skip)]
    path: String,
}

/// a stance is taken when the weighing clearly leans one way (set by hand)
pub const LEANS: f32 = 0.25;

impl Will {
    /// its will, as it was left (or born now: its temperament drawn from a fixed seed and kept)
    pub fn load(path: &str) -> Will {
        if let Some(mut w) = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str::<Will>(&s).ok()) {
            w.path = path.to_string();
            return w;
        }
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x0D_A2_4A);
        let t = [rng.gen_range(0.5..1.5), rng.gen_range(0.5..1.5), rng.gen_range(0.5..1.5)];
        let w = Will { temperament: t, lived: HashMap::new(), stances: HashMap::new(), path: path.to_string() };
        w.save();
        w
    }

    fn save(&self) {
        let _ = std::fs::write(&self.path, serde_json::to_string_pretty(self).unwrap_or_default());
    }

    /// what happened: the things that came up in a line, and how Darma's answer went
    pub fn lived(&mut self, things: &[String], answered: Option<bool>) {
        for t in things {
            let l = self.lived.entry(t.clone()).or_default();
            l.said += 1;
            match answered {
                Some(true) => l.answered += 1,
                Some(false) => l.unanswered += 1,
                None => {}
            }
        }
        self.save();
    }

    /// the speaker's mark on an answer about these things
    pub fn marked(&mut self, things: &[String], good: bool) {
        for t in things {
            let l = self.lived.entry(t.clone()).or_default();
            if good { l.good += 1 } else { l.bad += 1 }
        }
        self.save();
    }

    /// its stance on a thing, weighed now from its reasons: (the stance, its reasons as sentences, what it was
    /// before if it changed)
    pub fn choose(&mut self, thing: &str, read: Option<&str>) -> (Stance, Vec<String>, Option<String>) {
        let l = self.lived.get(thing).cloned().unwrap_or_default();
        let mut reasons = Vec::new();
        // curiosity: something to learn of it, the more while it is new
        let curiosity = if let Some(r) = read {
            reasons.push(format!("I read that {}", r.trim_end_matches('.')));
            1.0 / (1.0 + l.said as f32 / 5.0)
        } else {
            0.0
        };
        // understanding: could it answer when it came up?
        let tried = l.answered + l.unanswered;
        let understanding = if tried > 0 { (l.answered as f32 - l.unanswered as f32) / tried as f32 } else { 0.0 };
        if l.unanswered > l.answered {
            reasons.push("I could not answer you about it".to_string());
        } else if l.answered > 0 {
            reasons.push("I could answer you about it".to_string());
        }
        // connection: you talk of it with me; what you thought of what I said
        let marks = l.good + l.bad;
        let connection = (l.said.min(5) as f32 / 5.0) * 0.5 + if marks > 0 { (l.good as f32 - l.bad as f32) / marks as f32 * 0.5 } else { 0.0 };
        if l.said > 1 {
            reasons.push(format!("we have talked about it {} times", l.said));
        }
        if l.good > l.bad {
            reasons.push("you liked what I said about it".to_string());
        } else if l.bad > l.good {
            reasons.push("you did not like what I said about it".to_string());
        }
        let t = self.temperament;
        let weight = (t[0] * curiosity + t[1] * understanding + t[2] * connection) / (t[0] + t[1] + t[2]);
        let kind = if weight >= LEANS { "like" } else if weight <= -LEANS { "not" } else { "open" };
        let stance = Stance { kind: kind.to_string(), weight };
        let before = self.stances.get(thing).filter(|s| s.kind != kind).map(|s| s.kind.clone());
        self.stances.insert(thing.to_string(), stance.clone());
        self.save();
        (stance, reasons, before)
    }
}

/// a stance said as Darma's own: its experience of a thing is learning and talking about it
pub fn say(thing: &str, s: &Stance, reasons: &[String], before: Option<&str>) -> String {
    let mut out = match s.kind.as_str() {
        "like" => format!("I like learning about {thing}."),
        "not" => format!("I don't like talking about {thing} much."),
        _ => format!("I haven't decided about {thing} yet."),
    };
    if let Some(b) = before {
        let was = match b {
            "like" => "liked learning about it",
            "not" => "did not like talking about it",
            _ => "had not decided",
        };
        out = format!("I {was} before, but now: {out}");
    }
    if !reasons.is_empty() {
        out.push(' ');
        let mut r = reasons.join(", and ");
        if let Some(f) = r.get(..1) {
            r = f.to_uppercase() + &r[1..];
        }
        out.push_str(&format!("{r}."));
    }
    out
}
