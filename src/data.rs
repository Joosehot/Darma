//! The data (`darma prepare-data`): nothing is invented here. Two downloaded, human-annotated sets are read and
//! written into the forms Darma uses.
//!
//! Roles: Universal Dependencies English EWT (CC BY-SA 4.0), real English sentences whose subjects and objects people
//! marked. A clause is kept when a verb has a noun subject before it and a noun object after it (active voice, no
//! passive auxiliary). Its three words, in the sentence's order, are what Darma reads; the marked lemmas are the gold
//! roles (for measuring only). EWT's own split is kept: train + dev for training, test for testing.
//!
//! Senses: CoarseWSD-20 (CC BY-NC 4.0), Wikipedia sentences for 20 two-or-more-sense nouns (bank, crane, mole,
//! pitcher, seal, spring, ...), each marked with its sense by people. Its own train/test split is kept.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

#[derive(Serialize, Clone)]
pub struct Clause {
    /// the whole sentence it comes from
    pub sentence: String,
    /// the three words Darma reads, in the sentence's order
    pub text: String,
    pub agent: String,
    pub action: String,
    pub patient: String,
}

#[derive(Serialize, Clone)]
pub struct SenseCase {
    pub word: String,
    pub sentence: String,
    /// which token of the sentence is the word
    pub index: usize,
    pub sense: String,
}

/// a UD sentence's words with the role each one plays, as people marked it: A (a verb's noun subject), V (a verb
/// with a subject or an object), P (a verb's noun object), - (no role)
#[derive(Serialize)]
pub struct Tagged {
    pub words: Vec<String>,
    pub roles: Vec<String>,
}

#[derive(Serialize)]
struct Word {
    word: String,
    pos: String,
    senses: Vec<String>,
}

struct Tok {
    form: String,
    lemma: String,
    upos: String,
    head: usize,
    rel: String,
}

fn clauses(path: &Path) -> Vec<Clause> {
    let text = fs::read_to_string(path).expect("UD file");
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let mut sent = String::new();
        let mut toks: Vec<Tok> = Vec::new();
        for line in block.lines() {
            if let Some(s) = line.strip_prefix("# text = ") {
                sent = s.to_string();
            } else if !line.starts_with('#') {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 8 || f[0].contains('-') || f[0].contains('.') {
                    continue;
                }
                toks.push(Tok { form: f[1].to_lowercase(), lemma: f[2].to_lowercase(), upos: f[3].into(), head: f[6].parse().unwrap_or(0), rel: f[7].into() });
            }
        }
        for (vi, v) in toks.iter().enumerate() {
            if v.upos != "VERB" {
                continue;
            }
            let kids: Vec<(usize, &Tok)> = toks.iter().enumerate().filter(|(_, t)| t.head == vi + 1).collect();
            let subj = kids.iter().find(|(_, t)| t.rel == "nsubj" && t.upos == "NOUN");
            let obj = kids.iter().find(|(_, t)| t.rel == "obj" && t.upos == "NOUN");
            let passive = kids.iter().any(|(_, t)| t.rel == "aux:pass" || t.rel == "nsubj:pass");
            if let (Some(&(si, s)), Some(&(oi, o))) = (subj, obj) {
                let alpha = |t: &Tok| t.lemma.chars().all(|c| c.is_ascii_alphabetic());
                if !passive && si < vi && vi < oi && alpha(s) && alpha(v) && alpha(o) {
                    out.push(Clause {
                        sentence: sent.clone(),
                        text: format!("{} {} {}", s.form, v.form, o.form),
                        agent: s.lemma.clone(),
                        action: v.lemma.clone(),
                        patient: o.lemma.clone(),
                    });
                }
            }
        }
    }
    out
}

fn tagged(path: &Path) -> Vec<Tagged> {
    let text = fs::read_to_string(path).expect("UD file");
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let toks: Vec<Tok> = block
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|line| {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 8 || f[0].contains('-') || f[0].contains('.') {
                    return None;
                }
                Some(Tok { form: f[1].to_lowercase(), lemma: f[2].to_lowercase(), upos: f[3].into(), head: f[6].parse().unwrap_or(0), rel: f[7].into() })
            })
            .collect();
        let verb = |h: usize| h > 0 && toks.get(h - 1).is_some_and(|t| t.upos == "VERB");
        let mut words = Vec::new();
        let mut roles = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            // only letters, split as Darma splits a sentence (so the words line up with what it reads)
            let role = if t.rel == "nsubj" && t.upos == "NOUN" && verb(t.head) {
                "A"
            } else if t.rel == "obj" && t.upos == "NOUN" && verb(t.head) {
                "P"
            } else if t.upos == "VERB" && toks.iter().any(|k| k.head == i + 1 && (k.rel == "nsubj" || k.rel == "obj")) {
                "V"
            } else {
                "-"
            };
            for piece in t.form.split(|c: char| !c.is_alphabetic()).filter(|x| !x.is_empty()) {
                words.push(piece.to_string());
                roles.push(role.to_string());
            }
        }
        if !words.is_empty() {
            out.push(Tagged { words, roles });
        }
    }
    out
}

fn senses(dir: &Path, split: &str) -> (Vec<SenseCase>, BTreeMap<String, Vec<String>>) {
    let mut cases = Vec::new();
    let mut names = BTreeMap::new();
    let mut words: Vec<_> = fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.file_name().into_string().unwrap()).collect();
    words.sort();
    for w in words {
        let d = dir.join(&w);
        let map: BTreeMap<String, String> = serde_json::from_str(&fs::read_to_string(d.join("classes_map.txt")).unwrap()).unwrap();
        let mut list: Vec<(usize, String)> = map.iter().map(|(k, v)| (k.parse().unwrap(), v.clone())).collect();
        list.sort();
        names.insert(w.clone(), list.iter().map(|x| x.1.clone()).collect());
        let data = fs::read_to_string(d.join(format!("{split}.data.txt"))).unwrap();
        let gold = fs::read_to_string(d.join(format!("{split}.gold.txt"))).unwrap();
        for (line, g) in data.lines().zip(gold.lines()) {
            let (i, s) = line.split_once('\t').unwrap();
            let g: usize = g.trim().parse().unwrap();
            cases.push(SenseCase { word: w.clone(), sentence: s.to_string(), index: i.parse().unwrap(), sense: list[g].1.clone() });
        }
    }
    (cases, names)
}

pub fn prepare(dir: &Path) -> std::io::Result<()> {
    let raw = dir.join("raw");
    let mut train = clauses(&raw.join("ud/en_ewt-ud-train.conllu"));
    train.extend(clauses(&raw.join("ud/en_ewt-ud-dev.conllu")));
    let test = clauses(&raw.join("ud/en_ewt-ud-test.conllu"));
    let (wsd_train, names) = senses(&raw.join("coarsewsd"), "train");
    let (wsd_test, _) = senses(&raw.join("coarsewsd"), "test");
    // the lexicon: every word the role clauses use (one sense each: the data marks no senses for them) and the
    // CoarseWSD words with their senses
    let mut nouns: BTreeSet<String> = train.iter().chain(&test).flat_map(|c| [c.agent.clone(), c.patient.clone()]).collect();
    let verbs: BTreeSet<String> = train.iter().chain(&test).map(|c| c.action.clone()).collect();
    for w in names.keys() {
        nouns.remove(w);
    }
    let mut lex: Vec<Word> = names.iter().map(|(w, s)| Word { word: w.clone(), pos: "noun".into(), senses: s.clone() }).collect();
    lex.extend(nouns.iter().map(|w| Word { word: w.clone(), pos: "noun".into(), senses: vec![format!("{w}#1")] }));
    lex.extend(verbs.iter().map(|w| Word { word: w.clone(), pos: "verb".into(), senses: vec![format!("{w}#1")] }));
    let put = |name: &str, s: String| fs::write(dir.join(name), s);
    put("lexicon.json", serde_json::to_string_pretty(&lex).unwrap())?;
    let mut tg = tagged(&raw.join("ud/en_ewt-ud-train.conllu"));
    tg.extend(tagged(&raw.join("ud/en_ewt-ud-dev.conllu")));
    put("roles_train.json", serde_json::to_string(&tg).unwrap())?;
    println!("role-marked training sentences (UD EWT train + dev): {}, {} words", tg.len(), tg.iter().map(|t| t.words.len()).sum::<usize>());
    put("train.json", serde_json::to_string_pretty(&train).unwrap())?;
    put("test.json", serde_json::to_string_pretty(&test).unwrap())?;
    put("senses_train.json", serde_json::to_string_pretty(&wsd_train).unwrap())?;
    put("senses_test.json", serde_json::to_string_pretty(&wsd_test).unwrap())?;
    let seen: BTreeSet<&str> = train.iter().map(|c| c.text.as_str()).collect();
    println!(
        "roles (UD EWT): train {} clauses, test {} ({} of them also in train); senses (CoarseWSD-20): {} words, train {}, test {}; lexicon {} words ({} with two or more senses)",
        train.len(),
        test.len(),
        test.iter().filter(|c| seen.contains(c.text.as_str())).count(),
        names.len(),
        wsd_train.len(),
        wsd_test.len(),
        lex.len(),
        names.len()
    );
    Ok(())
}
