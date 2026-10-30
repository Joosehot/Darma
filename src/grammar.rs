//! Darma's grammar, learned from the sentences people parsed in UD English EWT: the orders of word kinds that many
//! sentences share (patterns), the words people put in each place of a pattern, and how each word bends (lemma and
//! features to form). A sentence of Darma's own is a pattern with its places filled: the kinds of words that carry
//! content (a verb, a noun) with what Darma has to say, the others with what people most often put there.
//!
//! Nothing here is a sentence someone said: a pattern is a kind order seen in at least MIN_SEEN sentences, and its
//! small words are the commonest in each place over all of them.

use serde::Deserialize;
use std::collections::HashMap;

/// a pattern must be seen in at least this many sentences to be one
pub const MIN_SEEN: u32 = 5;
/// the kinds of word that carry content (filled by what Darma has to say); every other kind is filled by the
/// commonest word of that place
const CONTENT: [&str; 2] = ["VERB", "NOUN"];
/// kinds a question pattern may hold besides those (no names, numbers or adjectives: they carry content Darma does
/// not have)
const SMALL: [&str; 9] = ["PRON", "AUX", "DET", "ADP", "PART", "ADV", "SCONJ", "CCONJ", "PUNCT"];

#[derive(Deserialize)]
pub struct Tok {
    pub form: String,
    pub lemma: String,
    pub upos: String,
    pub feats: String,
}

pub struct Pattern {
    pub kinds: Vec<String>,
    pub seen: u32,
    /// for each place: the words people put there, and for a verb place the features it bore
    words: Vec<HashMap<String, u32>>,
    feats: Vec<HashMap<String, u32>>,
}

pub struct Grammar {
    pub patterns: Vec<Pattern>,
    /// (lemma, features) -> the forms people wrote, with how often
    forms: HashMap<(String, String), HashMap<String, u32>>,
    /// form -> its lemma (the commonest)
    lemma_of: HashMap<String, String>,
}

fn top(m: &HashMap<String, u32>) -> Option<&String> {
    m.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map(|x| x.0)
}

impl Grammar {
    pub fn learn(path: &str) -> Option<Grammar> {
        let sents: Vec<Vec<Tok>> = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        let mut pats: HashMap<String, Pattern> = HashMap::new();
        let mut forms: HashMap<(String, String), HashMap<String, u32>> = HashMap::new();
        let mut lemmas: HashMap<String, HashMap<String, u32>> = HashMap::new();
        for s in &sents {
            for t in s {
                *forms.entry((t.lemma.clone(), t.feats.clone())).or_default().entry(t.form.clone()).or_default() += 1;
                *lemmas.entry(t.form.clone()).or_default().entry(t.lemma.clone()).or_default() += 1;
            }
            let kinds: Vec<String> = s.iter().map(|t| t.upos.clone()).collect();
            let p = pats.entry(kinds.join(" ")).or_insert_with(|| Pattern { kinds: kinds.clone(), seen: 0, words: vec![HashMap::new(); kinds.len()], feats: vec![HashMap::new(); kinds.len()] });
            p.seen += 1;
            for (i, t) in s.iter().enumerate() {
                *p.words[i].entry(t.form.clone()).or_default() += 1;
                *p.feats[i].entry(t.feats.clone()).or_default() += 1;
            }
        }
        // how words bend, over all parsed sentences (not only the short ones)
        if let Some(all) = std::fs::read_to_string("data/bends.json").ok().and_then(|s| serde_json::from_str::<Vec<(String, String, String, u32)>>(&s).ok()) {
            for (form, lemma, feats, n) in all {
                *forms.entry((lemma.clone(), feats)).or_default().entry(form.clone()).or_default() += n;
                *lemmas.entry(form).or_default().entry(lemma).or_default() += n;
            }
        }
        let mut patterns: Vec<Pattern> = pats.into_values().filter(|p| p.seen >= MIN_SEEN).collect();
        patterns.sort_by(|a, b| b.seen.cmp(&a.seen));
        let lemma_of = lemmas.into_iter().filter_map(|(f, m)| top(&m).cloned().map(|l| (f, l))).collect();
        Some(Grammar { patterns, forms, lemma_of })
    }

    /// a verb bent to the features a place wants: its lemma's form with those features, if people wrote one
    fn bend(&self, verb: &str, feats: &str) -> Option<String> {
        let lemma = self.lemma_of.get(verb).cloned().unwrap_or_else(|| verb.to_string());
        self.forms.get(&(lemma, feats.to_string())).and_then(top).cloned()
    }

    /// questions Darma can ask of what someone did and to what: every question pattern (ending in "?") with one verb
    /// and one noun after it and only small words besides, filled; commonest pattern first. Each comes with its pattern.
    pub fn ask_about(&self, verb: &str, noun: &str) -> Vec<(String, String, u32)> {
        let mut out = Vec::new();
        for p in &self.patterns {
            let n_verb = p.kinds.iter().filter(|k| *k == "VERB").count();
            let n_noun = p.kinds.iter().filter(|k| *k == "NOUN").count();
            let vi = p.kinds.iter().position(|k| k == "VERB");
            let ni = p.kinds.iter().position(|k| k == "NOUN");
            let small = p.kinds.iter().all(|k| CONTENT.contains(&k.as_str()) || SMALL.contains(&k.as_str()));
            let asks = top(p.words.last().unwrap()).is_some_and(|w| w == "?");
            if n_verb != 1 || n_noun != 1 || !small || !asks || vi > ni {
                continue;
            }
            let (vi, ni) = (vi.unwrap(), ni.unwrap());
            let Some(vf) = top(&p.feats[vi]).and_then(|f| self.bend(verb, f)) else { continue };
            let words: Vec<String> = (0..p.kinds.len())
                .map(|i| if i == vi { vf.clone() } else if i == ni { noun.to_string() } else { top(&p.words[i]).cloned().unwrap_or_default() })
                .collect();
            out.push((sentence(&words), p.kinds.join(" "), p.seen));
        }
        out
    }
}

/// words as a sentence: marks against the word before them, the first letter up
pub fn sentence(words: &[String]) -> String {
    let mut s = String::new();
    for w in words {
        let mark = w.chars().all(|c| !c.is_alphanumeric());
        let clitic = w.starts_with('\'') || w == "n't";
        if !s.is_empty() && !mark && !clitic {
            s.push(' ');
        }
        s.push_str(w);
    }
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// questions in parts: how people begin a question up to its verb ("why do you", "what do you", "have you ever"),
/// learned from DailyDialog's questions: the small words before the first verb of a question. A small word is one
/// whose kind people most often marked in UD as a small kind (pronoun, auxiliary, determiner, ...). Each beginning
/// keeps how often it was said, the verbs said after it, and the shape the verb had there.
pub struct Openings {
    /// beginning -> (how often, the verbs' lemmas after it with how often, the verb forms' features with how often)
    pub begins: HashMap<String, (u32, HashMap<String, u32>, HashMap<String, u32>)>,
    /// how often each beginning answered a line of each act (inform, question, directive, commissive)
    pub after_act: HashMap<String, [u32; 4]>,
    /// how often each beginning's verb was followed by a word of content (an object) and how often the question ended
    /// at the verb
    pub takes_object: HashMap<String, (u32, u32)>,
    /// the article people put before a noun ("a car"; none before "football"), when they mostly put one
    article: HashMap<String, String>,
    small: std::collections::HashSet<String>,
    feats_of: HashMap<String, String>,
}

impl Grammar {
    /// the commonest features of a form, as people marked them
    fn feats(&self, form: &str) -> Option<String> {
        let mut best: Option<(&String, u32)> = None;
        for ((_, f), m) in &self.forms {
            if let Some(&n) = m.get(form) {
                if best.map_or(true, |b| n > b.1) {
                    best = Some((f, n));
                }
            }
        }
        best.map(|b| b.0.clone())
    }

    pub fn openings(&self, ud: &str, dd: &str, verbs: &std::collections::HashSet<String>) -> Option<Openings> {
        // small words: their commonest kind in UD is a small kind
        let sents: Vec<Vec<Tok>> = serde_json::from_str(&std::fs::read_to_string(ud).ok()?).ok()?;
        let mut kinds: HashMap<String, HashMap<String, u32>> = HashMap::new();
        for s in &sents {
            for t in s {
                *kinds.entry(t.form.clone()).or_default().entry(t.upos.clone()).or_default() += 1;
            }
        }
        let small: std::collections::HashSet<String> = kinds.iter().filter(|(_, m)| top(m).is_some_and(|k| SMALL.contains(&k.as_str()))).map(|(w, _)| w.clone()).collect();
        #[derive(Deserialize)]
        struct Marked {
            dialog: Vec<String>,
            act: Vec<i64>,
        }
        let dialogs: Vec<Marked> = serde_json::from_str(&std::fs::read_to_string(dd).ok()?).ok()?;
        let mut begins: HashMap<String, (u32, HashMap<String, u32>, HashMap<String, u32>)> = HashMap::new();
        let mut before: HashMap<String, HashMap<String, u32>> = HashMap::new();
        let mut feats_of: HashMap<String, String> = HashMap::new();
        let mut after_act: HashMap<String, [u32; 4]> = HashMap::new();
        let mut takes_object: HashMap<String, (u32, u32)> = HashMap::new();
        for d in &dialogs {
            for (k, u) in d.dialog.iter().enumerate() {
                let prev_act = (k > 0 && d.act[k - 1] > 0).then(|| (d.act[k - 1] as usize - 1).min(3));
                let w = crate::lexicon::tokens(u);
                for k in 1..w.len() {
                    let prev = if matches!(w[k - 1].as_str(), "a" | "an" | "the") { w[k - 1].clone() } else { String::new() };
                    *before.entry(w[k].clone()).or_default().entry(prev).or_default() += 1;
                }
                for s in crate::dialog::sentences(u) {
                    if !s.trim_end().ends_with('?') {
                        continue;
                    }
                    let w = crate::lexicon::tokens(&s);
                    let Some(vi) = w.iter().position(|x| verbs.contains(x)) else { continue };
                    if vi == 0 || vi > 5 || !w[..vi].iter().all(|x| small.contains(x)) {
                        continue;
                    }
                    let f = feats_of.entry(w[vi].clone()).or_insert_with(|| self.feats(&w[vi]).unwrap_or_default()).clone();
                    let lemma = self.lemma_of.get(&w[vi]).cloned().unwrap_or_else(|| w[vi].clone());
                    if let Some(pa) = prev_act {
                        after_act.entry(w[..vi].join(" ")).or_default()[pa] += 1;
                    }
                    // an object: a word of content right after the verb (not a small word: "going to do" is not)
                    // (a small word after the verb says neither: "can you tell me ...")
                    let e = takes_object.entry(w[..vi].join(" ")).or_default();
                    if vi + 1 == w.len() {
                        e.1 += 1;
                    } else if !small.contains(&w[vi + 1]) {
                        e.0 += 1;
                    }
                    let e = begins.entry(w[..vi].join(" ")).or_default();
                    e.0 += 1;
                    *e.1.entry(lemma).or_default() += 1;
                    *e.2.entry(f).or_default() += 1;
                }
            }
        }
        // a noun takes the article people most often put before it, if they put one (a, an or the) at least two times
        // in five
        let article = before
            .into_iter()
            .filter_map(|(n, m)| {
                let total: u32 = m.values().sum();
                let with: u32 = m.iter().filter(|(a, _)| !a.is_empty()).map(|x| x.1).sum();
                let (a, _) = m.iter().filter(|(a, _)| !a.is_empty()).max_by(|x, y| x.1.cmp(y.1))?;
                (total >= 5 && with * 5 >= total * 2).then(|| (n, a.clone()))
            })
            .collect();
        begins.retain(|_, e| e.0 >= MIN_SEEN);
        Some(Openings { begins, after_act, takes_object, article, small, feats_of })
    }

    /// questions of what someone did and to what, from the beginnings people use; a beginning people used with this
    /// very verb is preferred; commonest first. (question, its beginning, how often it was said, used with this verb)
    /// `to` is whom the question is to ("you": a beginning that does not hold it is not asked)
    pub fn ask_with(&self, o: &Openings, verb: &str, noun: &str, to: &str, after: usize) -> Vec<(String, String, u32, bool)> {
        let lemma = self.lemma_of.get(verb).cloned().unwrap_or_else(|| verb.to_string());
        let mut out: Vec<(String, String, u32, bool)> = Vec::new();
        for (b, (n, vs, fs)) in &o.begins {
            if !b.split(' ').any(|x| x == to) {
                continue;
            }
            // how often people said this beginning with this very verb (a beginning never said with it counts for
            // little)
            let with = vs.get(&lemma).copied().unwrap_or(0);
            let n = if with > 0 { with } else { (n / 50).max(1) };
            // weighed by how often people opened so after a line of this act (the share of its answers it is)
            let aa = o.after_act.get(b).copied().unwrap_or_default();
            let share = (aa[after] as f32 + 1.0) / (aa.iter().sum::<u32>() as f32 + 4.0);
            let n = ((n as f32) * share * 100.0).round() as u32;
            let Some(f) = top(fs) else { continue };
            let Some(v) = self.bend(verb, f) else { continue };
            let mut words: Vec<String> = b.split(' ').map(String::from).collect();
            words.push(v);
            // a beginning whose questions mostly end at the verb asks for the object itself ("what are you reading?")
            let (with_obj, without) = o.takes_object.get(b).copied().unwrap_or((1, 0));
            if with_obj >= without {
                if let Some(a) = o.article.get(noun).filter(|a| !a.is_empty()) {
                    words.push(a.clone());
                }
                words.push(noun.to_string());
            }
            words.push("?".into());
            out.push((sentence(&words), b.clone(), n, with > 0));
        }
        out.sort_by(|a, b| b.3.cmp(&a.3).then(b.2.cmp(&a.2)));
        let _ = (&o.small, &o.feats_of);
        out
    }
}
