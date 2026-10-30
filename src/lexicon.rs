//! The atoms: every word, role and sense gets its own random phase vector, made from its name and the seed (so the
//! same word always has the same vector and nothing is stored but the seed). The lexicon file says which words have
//! more than one sense: those are the words with more than one sperm.

use crate::vector::HVec;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct Atoms {
    pub d: usize,
    seed: u64,
    cache: RefCell<HashMap<String, HVec>>,
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

impl Atoms {
    pub fn new(d: usize, seed: u64) -> Self {
        Atoms { d, seed, cache: RefCell::new(HashMap::new()) }
    }

    pub fn get(&self, name: &str) -> HVec {
        if let Some(v) = self.cache.borrow().get(name) {
            return v.clone();
        }
        let v = HVec::random(self.d, &mut ChaCha8Rng::seed_from_u64(fnv(name) ^ self.seed));
        self.cache.borrow_mut().insert(name.to_string(), v.clone());
        v
    }

    pub fn word(&self, w: &str) -> HVec {
        self.get(&format!("w:{w}"))
    }

    pub fn role(&self, r: usize) -> HVec {
        self.get(["role:AGENT", "role:ACTION", "role:PATIENT"][r])
    }
}

#[derive(Deserialize)]
pub struct Entry {
    pub word: String,
    pub pos: String,
    pub senses: Vec<String>,
}

pub struct Lexicon {
    pub senses: HashMap<String, Vec<String>>,
}

impl Lexicon {
    pub fn load(path: &str) -> Self {
        let e: Vec<Entry> = serde_json::from_str(&std::fs::read_to_string(path).expect("data/lexicon.json")).unwrap();
        let mut senses = HashMap::new();
        for x in e {
            if x.senses.len() > 1 {
                senses.insert(x.word, x.senses);
            }
        }
        Lexicon { senses }
    }
}

/// words of a sentence: lower case, letters only
pub fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphabetic()).filter(|t| !t.is_empty()).map(String::from).collect()
}
