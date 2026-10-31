//! Reading many sentences at once, with the GPU. Reading is interference: a place's wave (a sum of parts, each a role
//! bound to a value) against prototype waves. Once struck, the prototypes do not change, so each part's interference
//! with each prototype is one number that never changes: the GPU makes all the numbers a body of text needs at once
//! (every part's wave against every prototype, one product of two matrices), and reading is then adding numbers up.
//! The same reading as wave by wave (checked by `darma fast-check`), only faster.
use crate::cpt::{Cpt, Node};
use crate::syntax::{par_map, Syntax, Tok};
use crate::vector::HVec;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

const KERNELS: &str = r#"
extern "C" __global__ void dense(const float* b, const float* p, float* out, int n, int k, int d2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * k) return;
    const float* x = b + (size_t)(i / k) * d2;
    const float* y = p + (size_t)(i % k) * d2;
    float s = 0.0f;
    for (int j = 0; j < d2; j++) s += x[j] * y[j];
    out[i] = s;
}
extern "C" __global__ void indexed(const float* b, const float* p, const unsigned int* idx, float* out, int n, int d2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* x = b + (size_t)i * d2;
    const float* y = p + (size_t)idx[i] * d2;
    float s = 0.0f;
    for (int j = 0; j < d2; j++) s += x[j] * y[j];
    out[i] = s;
}
"#;

pub struct Gpu {
    dev: Arc<CudaDevice>,
}

/// a wave as real numbers: re, im, re, im ... (so Re<a, b> is a plain dot product)
fn flat(v: &HVec, out: &mut Vec<f32>) {
    for c in &v.0 {
        out.push(c.re);
        out.push(c.im);
    }
}

/// parts made into waves and sent at most this many at once
const CHUNK: usize = 20000;

impl Gpu {
    pub fn new() -> Option<Gpu> {
        let dev = CudaDevice::new(0).ok()?;
        dev.load_ptx(cudarc::nvrtc::compile_ptx(KERNELS).ok()?, "darma", &["dense", "indexed"]).ok()?;
        Some(Gpu { dev })
    }

    /// Re<b_i, p_j> for every wave b_i and prototype p_j: n x k
    pub fn dense(&self, b: &[HVec], p: &[HVec]) -> Vec<f32> {
        let (n, k) = (b.len(), p.len());
        if n == 0 {
            return Vec::new();
        }
        let d2 = 2 * b[0].0.len();
        let (mut bf, mut pf) = (Vec::with_capacity(n * d2), Vec::with_capacity(k * d2));
        b.iter().for_each(|v| flat(v, &mut bf));
        p.iter().for_each(|v| flat(v, &mut pf));
        let bd = self.dev.htod_copy(bf).unwrap();
        let pd = self.dev.htod_copy(pf).unwrap();
        let mut od = self.dev.alloc_zeros::<f32>(n * k).unwrap();
        let f = self.dev.get_func("darma", "dense").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems((n * k) as u32), (&bd, &pd, &mut od, n as i32, k as i32, d2 as i32)) }.unwrap();
        self.dev.dtoh_sync_copy(&od).unwrap()
    }

    /// Re<b_i, p_{idx_i}> for every wave b_i: n
    pub fn indexed(&self, b: &[HVec], p: &[HVec], idx: &[u32]) -> Vec<f32> {
        let n = b.len();
        if n == 0 {
            return Vec::new();
        }
        let d2 = 2 * b[0].0.len();
        let (mut bf, mut pf) = (Vec::with_capacity(n * d2), Vec::with_capacity(p.len() * d2));
        b.iter().for_each(|v| flat(v, &mut bf));
        p.iter().for_each(|v| flat(v, &mut pf));
        let bd = self.dev.htod_copy(bf).unwrap();
        let pd = self.dev.htod_copy(pf).unwrap();
        let id = self.dev.htod_copy(idx.to_vec()).unwrap();
        let mut od = self.dev.alloc_zeros::<f32>(n).unwrap();
        let f = self.dev.get_func("darma", "indexed").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&bd, &pd, &id, &mut od, n as i32, d2 as i32)) }.unwrap();
        self.dev.dtoh_sync_copy(&od).unwrap()
    }
}

/// a part's numbers against each prototype, made on the GPU the first time the part is met
struct Table {
    k: usize,
    rows: HashMap<String, Vec<f32>>,
}

impl Table {
    fn new(k: usize) -> Table {
        Table { k, rows: HashMap::new() }
    }

    /// make the numbers of the parts not met before: their waves on every core, the products on the GPU
    fn fill(&mut self, gpu: &Gpu, keys: Vec<String>, atom: &(dyn Fn(&str) -> HVec + Sync), protos: &[HVec]) {
        let mut keys: Vec<String> = keys.into_iter().filter(|k| !self.rows.contains_key(k)).collect();
        keys.sort();
        keys.dedup();
        let atoms = atoms_of(&keys, atom);
        for ch in keys.chunks(CHUNK) {
            let waves: Vec<HVec> = par_map(ch, |k| { let (r, v) = unkey(k); atoms[r].bind(&atoms[v]) });
            let out = gpu.dense(&waves, protos);
            for (i, k) in ch.iter().enumerate() {
                self.rows.insert(k.clone(), out[i * self.k..(i + 1) * self.k].to_vec());
            }
        }
    }

    fn best(&self, keys: &[String]) -> usize {
        let mut s = vec![0f32; self.k];
        for k in keys {
            if let Some(r) = self.rows.get(k) {
                for (a, b) in s.iter_mut().zip(r) {
                    *a += b;
                }
            }
        }
        (0..self.k).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap_or(0)
    }
}

/// the atoms of the parts' roles and values, each made once, on every core
fn atoms_of(keys: &[String], atom: &(dyn Fn(&str) -> HVec + Sync)) -> HashMap<String, HVec> {
    let mut names: Vec<String> = keys.iter().flat_map(|k| { let (r, v) = unkey(k); [r.to_string(), v.to_string()] }).collect();
    names.sort();
    names.dedup();
    let made = par_map(&names, |n| atom(n));
    names.into_iter().zip(made).collect()
}

fn key(r: &str, v: &str) -> String {
    format!("r:{r}\u{1}v:{v}")
}

fn unkey(k: &str) -> (&str, &str) {
    k.split_once('\u{1}').unwrap_or((k, ""))
}

/// prototypes as unit waves (a place's reading divides by each prototype's length; its own length is the same for all)
fn unit(p: &[HVec]) -> Vec<HVec> {
    p.iter().map(|v| {
        let n = v.norm().max(1e-9);
        HVec(v.0.iter().map(|c| c / n).collect())
    }).collect()
}

/// every pair part there can be, numbered: its value is made of a few small alphabets (the classes as hearing writes
/// them, the side, the distance, the small words), so all of them are made at once on the GPU and a pair's score is ten
/// numbers looked up by index (no words to write or find)
pub struct PairTable {
    cls: HashMap<String, usize>,
    sm: HashMap<String, usize>,
    a: usize,
    s: usize,
    /// per part: its numbers, by index
    num: Vec<Vec<f32>>,
}

const DIRS: [&str; 3] = ["root", "L", "R"];
const DISTS: [&str; 6] = ["root", "1", "2", "3-4", "5-7", "8+"];

impl PairTable {
    fn build(gpu: &Gpu, cpt: &Cpt) -> PairTable {
        let mut cls: Vec<String> = cpt.sx.pos_names.clone();
        cls.extend(["ROOT", "END", "-"].map(String::from));
        let mut sm: Vec<String> = cpt.hear.small.iter().cloned().collect();
        sm.sort();
        sm.push("-".into());
        let (a, s) = (cls.len(), sm.len());
        let tf = ["false", "true"];
        // the values of each part, in index order (written exactly as hearing writes them)
        let mut vals: Vec<Vec<String>> = vec![Vec::new(); 10];
        for dc in &cls {
            for hc in &cls {
                vals[0].push(format!("{dc}>{hc}"));
                for dir in DIRS {
                    vals[1].push(format!("{dc}>{hc} {dir}"));
                    for dist in DISTS {
                        vals[2].push(format!("{dc}>{hc} {dir} {dist}"));
                    }
                    for l in &cls {
                        for r in &cls {
                            vals[6].push(format!("{dc}>{hc} {dir} {l}|{r}"));
                        }
                    }
                }
                for b in tf {
                    vals[8].push(format!("{dc}>{hc} {b}"));
                    vals[9].push(format!("{dc}>{hc} {b}"));
                }
            }
        }
        vals[7] = vals[6].clone();
        for c in &cls {
            for w in &sm {
                for dir in DIRS {
                    vals[4].push(format!("{c}>{w} {dir}"));
                }
            }
        }
        for w in &sm {
            for c in &cls {
                for dir in DIRS {
                    vals[3].push(format!("{w}>{c} {dir}"));
                }
            }
            for w2 in &sm {
                vals[5].push(format!("{w}>{w2}"));
            }
        }
        let roles: Vec<&str> = cpt.hear.pair_feats(&["x".to_string()], &["X".to_string()], 1, 0).into_iter().map(|x| x.0).collect();
        let mut num = Vec::new();
        for (r, vs) in roles.iter().zip(&vals) {
            let ra = cpt.hear.atom_fresh(&format!("r:{r}"));
            let mut out = Vec::with_capacity(vs.len());
            for ch in vs.chunks(CHUNK) {
                let waves: Vec<HVec> = par_map(ch, |v| ra.bind(&cpt.hear.atom_fresh(&format!("v:{v}"))));
                let ix = vec![roles.iter().position(|x| x == r).unwrap() as u32; ch.len()];
                out.extend(gpu.indexed(&waves, cpt.hear.attach(), &ix));
            }
            num.push(out);
        }
        PairTable { cls: cls.into_iter().enumerate().map(|(i, c)| (c, i)).collect(), sm: sm.into_iter().enumerate().map(|(i, w)| (w, i)).collect(), a, s, num }
    }

    /// the score of "word d completes word h" (1-based; h = 0: the core), classes and small words as numbers
    fn score(&self, c: &[usize], w: &[usize], d: usize, h: usize, root: usize, end: usize, dash: usize, sdash: usize) -> f32 {
        let (a, s) = (self.a, self.s);
        let cls = |k: usize| if k == 0 { root } else if k > c.len() { end } else { c[k - 1] };
        let sm = |k: usize| if k == 0 || k > w.len() { sdash } else { w[k - 1] };
        let (dc, hc) = (cls(d), cls(h));
        let dir = if h == 0 { 0 } else if h < d { 1 } else { 2 };
        let dist = if h == 0 { 0 } else { DISTS.iter().position(|x| *x == crate::hear::bucket(d.abs_diff(h))).unwrap() };
        let (lo, hi) = (d.min(h), d.max(h));
        let (verb_k, punct_k) = (self.cls.get("VERB").copied(), self.cls.get("PUNCT").copied());
        let verb = h != 0 && (lo + 1..hi).any(|k| Some(c[k - 1]) == verb_k);
        let punct = h != 0 && (lo + 1..hi).any(|k| Some(c[k - 1]) == punct_k);
        let pair = dc * a + hc;
        let (hl, hr) = if h == 0 { (dash, dash) } else { (cls(h - 1), cls(h + 1)) };
        let (dl, dr) = (cls(d - 1), cls(d + 1));
        self.num[0][pair]
            + self.num[1][pair * 3 + dir]
            + self.num[2][(pair * 3 + dir) * 6 + dist]
            + self.num[3][(sm(d) * a + hc) * 3 + dir]
            + self.num[4][(dc * s + sm(h)) * 3 + dir]
            + self.num[5][sm(d) * s + sm(h)]
            + self.num[6][((pair * 3 + dir) * a + hl) * a + hr]
            + self.num[7][((pair * 3 + dir) * a + dl) * a + dr]
            + self.num[8][pair * 2 + verb as usize]
            + self.num[9][pair * 2 + punct as usize]
    }
}

/// the reader with its tables, kept across batches of sentences
pub struct Reader {
    gpu: Gpu,
    classes: Table,
    roles: Table,
    pairs: PairTable,
}

impl Reader {
    pub fn new(cpt: &Cpt) -> Option<Reader> {
        let (pos, rel) = cpt.sx.protos();
        let gpu = Gpu::new()?;
        let t = std::time::Instant::now();
        let pairs = PairTable::build(&gpu, cpt);
        eprintln!("  every pair part made on the GPU ({} numbers) in {:.1} s", pairs.num.iter().map(|v| v.len()).sum::<usize>(), t.elapsed().as_secs_f64());
        Some(Reader { gpu, classes: Table::new(pos.len()), roles: Table::new(rel.len()), pairs })
    }

    /// sentences heard as Cpt::hear_fresh hears them (the mark at the end, "I" with its capital)
    pub fn read(&mut self, cpt: &Cpt, sents: &[Vec<String>], mark: &str) -> Vec<Vec<Node>> {
        let sx: &Syntax = &cpt.sx;
        let (pos, rel) = sx.protos();
        let (pos, rel) = (unit(pos), unit(rel));
        let toks: Vec<Vec<Tok>> = sents.iter().map(|w| {
            let mut t: Vec<Tok> = w.iter().map(|x| Tok { word: x.clone(), cap: x == "i", pos: String::new(), rel: String::new() }).collect();
            t.push(Tok { word: mark.to_string(), cap: false, pos: String::new(), rel: String::new() });
            t
        }).collect();
        let bound = |n: &str| sx.atom_pub(n);
        let tp = std::time::Instant::now();
        let prof = std::env::var("DARMA_PROFILE").is_ok();
        let lap = |s: &str| if prof { eprintln!("    {s}: {:.2} s", tp.elapsed().as_secs_f64()) };
        // classes
        let cf: Vec<Vec<Vec<String>>> = par_map(&toks, |s| (0..s.len()).map(|i| Syntax::class_feats(s, i).into_iter().map(|(r, v)| key(&r, &v)).collect()).collect());
        self.classes.fill(&self.gpu, cf.iter().flatten().flatten().cloned().collect(), &bound, &pos);
        lap("class parts and numbers");
        let tags: Vec<Vec<usize>> = par_map(&cf, |s| s.iter().map(|p| self.classes.best(p)).collect());
        lap("classes read");
        // roles
        let idx: Vec<usize> = (0..toks.len()).collect();
        let rf: Vec<Vec<Vec<String>>> = par_map(&idx, |&k| (0..toks[k].len()).map(|i| sx.role_feats(&toks[k], i, &tags[k]).into_iter().map(|(r, v)| key(&r, &v)).collect()).collect());
        self.roles.fill(&self.gpu, rf.iter().flatten().flatten().cloned().collect(), &bound, &rel);
        lap("role parts and numbers");
        let roles: Vec<Vec<usize>> = par_map(&rf, |s| s.iter().map(|p| self.roles.best(p)).collect());
        // core words: every pair scored from the pair table, the best tree
        let classes: Vec<Vec<String>> = tags.iter().map(|t| t.iter().map(|&k| sx.pos_names[k].clone()).collect()).collect();
        let pt = &self.pairs;
        let (root, end, dash, sdash) = (pt.cls["ROOT"], pt.cls["END"], pt.cls["-"], pt.sm["-"]);
        let heads: Vec<Vec<usize>> = par_map(&idx, |&k| {
            let c: Vec<usize> = classes[k].iter().map(|x| pt.cls[x]).collect();
            let w: Vec<usize> = toks[k].iter().map(|x| pt.sm.get(&x.word).copied().unwrap_or(sdash)).collect();
            let n = c.len();
            let mut s = vec![vec![f32::MIN; n + 1]; n + 1];
            for d in 1..=n {
                for h in 0..=n {
                    if h != d {
                        s[h][d] = pt.score(&c, &w, d, h, root, end, dash, sdash);
                    }
                }
            }
            crate::hear::eisner(&s)
        });
        lap("trees");
        // the nodes, as Cpt::hear_fresh makes them (the mark left out)
        (0..sents.len()).map(|k| {
            let w = &sents[k];
            let n = w.len();
            (0..n).map(|i| Node { word: w[i].clone(), lemma: cpt.lemma.get(&(w[i].clone(), classes[k][i].clone())).cloned().unwrap_or_else(|| w[i].clone()), pos: classes[k][i].clone(), rel: sx.rel_names[roles[k][i]].clone(), head: if heads[k][i] > n { 0 } else { heads[k][i] } }).collect()
        }).collect()
    }
}

/// `darma fast-check`: the GPU reading against the wave-by-wave reading on UD's test sentences: how many words get
/// the same class, role and core word, and how long each takes
pub fn check() {
    let Some(cpt) = Cpt::load(1) else { return println!("UD not found") };
    let test = crate::syntax::load("data/raw/ud/en_ewt-ud-test.conllu");
    let sents: Vec<Vec<String>> = test.iter().take(500).map(|s| s.iter().map(|t| t.word.clone()).collect()).collect();
    let t0 = std::time::Instant::now();
    let slow: Vec<Vec<Node>> = par_map(&sents, |w| cpt.hear_fresh(w, "."));
    let ts = t0.elapsed().as_secs_f64();
    let Some(mut r) = Reader::new(&cpt) else { return println!("no GPU (CUDA)") };
    let t1 = std::time::Instant::now();
    let fast = r.read(&cpt, &sents, ".");
    let tf = t1.elapsed().as_secs_f64();
    let t2 = std::time::Instant::now();
    let _ = r.read(&cpt, &sents, ".");
    let tf2 = t2.elapsed().as_secs_f64();
    let (mut n, mut c, mut ro, mut h) = (0, 0, 0, 0);
    for (a, b) in slow.iter().zip(&fast) {
        for (x, y) in a.iter().zip(b) {
            n += 1;
            c += (x.pos == y.pos) as usize;
            ro += (x.rel == y.rel) as usize;
            h += (x.head == y.head) as usize;
        }
    }
    let p = |a: usize| 100.0 * a as f32 / n.max(1) as f32;
    println!("{} sentences, {n} words: the same class {:.2}%, role {:.2}%, core word {:.2}%", sents.len(), p(c), p(ro), p(h));
    println!("wave by wave (every core) {ts:.2} s; GPU {tf:.2} s (making the numbers), {tf2:.2} s once they are made");
}
