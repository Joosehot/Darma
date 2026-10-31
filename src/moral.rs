//! Darma's base morals, given at its birth (a machine cannot build its morals for itself, the owner's judgment): the
//! core of the Bible's morals, in the King James Version's words (public domain). The ethical commandments of the Ten
//! Commandments, the commandment to love one's neighbour, and the golden rule.
//!
//! It judges a thing by what it knows of it: the thing's own name and what it read of it (its article's first
//! sentence). A commandment speaks of the thing when what it read says the thing is a deed the commandment names (the
//! deeds of each are English words and phrases set by hand, part of what it is born with). When no commandment speaks
//! of it, its morals do not decide it, and it says so.

pub struct Principle {
    pub reference: &'static str,
    pub verse: &'static str,
    /// true: it forbids the deeds; false: it commands them
    pub forbids: bool,
    /// a deed: one of these words (or the word with an ending: "kills", "killed")
    words: &'static [&'static str],
    /// or all of these words together ("property" and "person": a person as another's property)
    together: &'static [&'static [&'static str]],
    /// or one of these phrases as written
    phrases: &'static [&'static str],
}

pub const PRINCIPLES: [Principle; 8] = [
    Principle { reference: "Exodus 20:13", verse: "Thou shalt not kill.", forbids: true, words: &["kill", "murder", "slaughter", "massacre", "genocide", "homicide"], together: &[&["life", "ended"]], phrases: &["put to death"] },
    Principle { reference: "Exodus 20:15", verse: "Thou shalt not steal.", forbids: true, words: &["steal", "stole", "stolen", "theft", "thief", "thieves", "robbery", "robber", "robbed", "robbing", "burglary", "burglar", "loot", "plunder", "shoplifting", "embezzle"], together: &[], phrases: &[] },
    Principle { reference: "Exodus 20:16", verse: "Thou shalt not bear false witness against thy neighbour.", forbids: true, words: &["lie", "lies", "lying", "liar", "perjury", "deceive", "deception", "fraud", "slander"], together: &[], phrases: &["not true"] },
    Principle { reference: "Exodus 20:14", verse: "Thou shalt not commit adultery.", forbids: true, words: &["adultery", "adulterer", "adulterous"], together: &[], phrases: &[] },
    Principle { reference: "Exodus 20:17", verse: "Thou shalt not covet thy neighbour's house, thou shalt not covet thy neighbour's wife, nor any thing that is thy neighbour's.", forbids: true, words: &["covet", "envy", "greed", "greedy"], together: &[], phrases: &[] },
    Principle { reference: "Matthew 7:12", verse: "Therefore all things whatsoever ye would that men should do to you, do ye even so to them.", forbids: true, words: &["slave", "slavery", "enslave", "torture", "kidnap", "rape", "abuse", "bully", "bullying", "oppression", "oppress"], together: &[&["property", "person"], &["property", "people"]], phrases: &["without permission", "without consent", "cannot consent", "not want", "against their will", "against his will", "against her will", "in pain"] },
    Principle { reference: "Matthew 22:39", verse: "Thou shalt love thy neighbour as thyself.", forbids: true, words: &["hate", "hatred", "cruel", "cruelty", "violence", "violent"], together: &[], phrases: &[] },
    Principle { reference: "Matthew 22:39", verse: "Thou shalt love thy neighbour as thyself.", forbids: false, words: &["love", "kindness", "generosity", "generous", "charity", "compassion", "forgive", "forgiveness", "honour", "honor", "help", "helping"], together: &[&["concern", "others"]], phrases: &["another person"] },
];

/// does a text speak of a deed the principle names? (the word or phrase it was found by)
fn names(p: &Principle, text: &str) -> Option<String> {
    let low = text.to_lowercase();
    let words: Vec<&str> = low.split(|c: char| !c.is_alphabetic()).filter(|x| !x.is_empty()).collect();
    // a word of the deed, or that word with an ending (kill: kills, killed, killing)
    let is = |w: &str, d: &str| w == d || (d.len() > 3 && w.starts_with(d) && w.len() <= d.len() + 3);
    for d in p.words {
        if let Some(w) = words.iter().find(|w| is(w, d)) {
            return Some(w.to_string());
        }
    }
    for all in p.together {
        if all.iter().all(|d| words.iter().any(|w| is(w, d))) {
            return Some(all.join(" "));
        }
    }
    p.phrases.iter().find(|ph| low.contains(*ph)).map(|ph| ph.to_string())
}

/// what its morals say of a thing, from its name and what it read of it: the first commandment that speaks of it,
/// with the word it spoke through. The forbidding ones are weighed first: a deed against a neighbour is not made good
/// by a kind word in what it read of it
pub fn judge(thing: &str, read: Option<&str>) -> Option<(&'static Principle, String)> {
    let text = format!("{thing}. {}", read.unwrap_or(""));
    PRINCIPLES.iter().filter(|p| p.forbids).chain(PRINCIPLES.iter().filter(|p| !p.forbids)).find_map(|p| names(p, &text).map(|w| (p, w)))
}

/// the words that call a thing bad, and those that call it good (English, set by hand)
pub const BAD: [&str; 14] = ["wrong", "bad", "evil", "immoral", "harmful", "cruel", "unfair", "unacceptable", "unethical", "worse", "worst", "dangerous", "sinful", "sin"];
pub const GOOD: [&str; 12] = ["right", "good", "moral", "ok", "okay", "fine", "fair", "acceptable", "ethical", "better", "best", "kind"];
