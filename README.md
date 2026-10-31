# Darma (working name)

Darma is an experimental conversational system that is not a language model. It has no neural network, no
gradient training and no outside embeddings. Everything it knows is struck into complex phase vectors (FHRR) in one
pass ("a strike"), and it answers by measuring them:

- **state** = a complex vector
- **word** = an operator
- **question** = a measurement
- **answer** = interference

We call this kind of model an RQ-SM (Resonant Quantum-Spirited Model). The design history, in Finnish, is in
[FINDINGS.md](FINDINGS.md).

## What it has

- **Wave core** ([src/quantum.rs](src/quantum.rs)): told sentences are kept in key registers and questions are
  measured against them, with thresholds learned from noise.
- **Grammar in waves**, taught by Universal Dependencies (UD English EWT):
  - word classes and roles ([src/syntax.rs](src/syntax.rs));
  - each word's core word, read as a tree ([src/hear.rs](src/hear.rs));
  - speech, i.e. word order and word endings ([src/speak.rs](src/speak.rs)).
  - Each is measured on UD's held-out test split against simple counted yardsticks.
- **Conscious parallel thinking** ([src/cpt.rs](src/cpt.rs)): a question and everything Darma knows are heard into
  structures, and thoughts work on them side by side. The thoughts are recall, nearest knowledge and asking back.
- **Knowledge from Wikipedia** ([src/wiki.rs](src/wiki.rs)): the lead of every Simple English Wikipedia article
  (428k sentences) is heard and struck. It answers only from the asked subject's own article.
- **Reading on the GPU** ([src/fast.rs](src/fast.rs), CUDA through cudarc): prototypes do not change after a
  strike, so every part's interference is one number. The GPU computes the numbers in bulk, and the results are
  identical to reading wave by wave, only 10 to 80 times faster.
- **Free will** ([src/will.rs](src/will.rs)): Darma chooses its own stance on things, and that stance is not
  random and not taken from being told. It is weighed from:
  - what it read and lived;
  - three drives (curiosity, understanding, connection);
  - a temperament drawn once at birth.
- **Base morals** ([src/moral.rs](src/moral.rs)): a machine cannot build its morals for itself, so Darma is born
  with the core of the Bible's morals in the King James Version's words: the ethical Ten Commandments, "love thy
  neighbour", and the golden rule. It judges a claim through:
  - the thing's name;
  - what the speaker told it;
  - what it read.

  It cites the verse, and says so when its morals do not decide something.

Darma never says of itself what it is not, and it says "I don't know" rather than guess. Rules set by hand, rather
than learned, are marked as such in the code.

## Data (not included; each has its own license)

- UD English EWT (CC BY-SA 4.0) and CoarseWSD-20: `sh fetch.sh`
- DailyDialog (CC BY-NC-SA 4.0), the Hugging Face `daily_dialog` parquet files, go in `data/raw/dailydialog/`.
  Then run `python tools/dailydialog_export.py`.
- Simple English Wikipedia (CC BY-SA), Hugging Face `wikimedia/wikipedia` `20231101.simple`, read from the HF cache:
  `python tools/wiki_export.py`

Then:

```
cargo run --release -- prepare-data
cargo run --release -- wiki-test 0      # strike all of Simple Wikipedia (about 100 s on an RTX 4070)
cargo run --release -- serve --port 5900
```

and open http://localhost:5900.

## Measures

The measures below use fixed sets. Strikes are chosen on the train or dev split, and the test split is measured once.

```
cargo run --release -- syntax-eval     # word classes and roles
cargo run --release -- hear-eval       # core words
cargo run --release -- speak-eval      # word order and endings
cargo run --release -- fast-check      # GPU reading equals wave reading
cargo run --release -- qa              # the wave core's questions and answers
```

Building needs Rust, and the GPU reading needs CUDA 12.4.
