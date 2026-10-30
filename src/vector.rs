//! Phase vectors (FHRR, Fourier Holographic Reduced Representations): the quantum-minded but classical core.
//!
//! An atom is D complex numbers of length 1 with random phases. Binding (entanglement) multiplies element by
//! element, so phases add; unbinding multiplies by the conjugate, so they subtract back. A superposition is a sum:
//! where phases agree the sum grows, where they disagree it cancels (interference). Similarity is the real part of
//! the normalised inner product, so 1 means the same, 0 means unrelated, and -1 means opposite.

use num_complex::Complex32;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::TAU;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HVec(pub Vec<Complex32>);

impl HVec {
    pub fn zero(d: usize) -> Self {
        HVec(vec![Complex32::new(0.0, 0.0); d])
    }

    /// a new atom: unit phasors with random phases
    pub fn random(d: usize, rng: &mut ChaCha8Rng) -> Self {
        HVec((0..d).map(|_| Complex32::from_polar(1.0, rng.gen::<f32>() * TAU)).collect())
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// binding: element-wise product (phases add)
    pub fn bind(&self, o: &HVec) -> HVec {
        HVec(self.0.iter().zip(&o.0).map(|(a, b)| a * b).collect())
    }

    /// unbinding: product with the other's conjugate (phases subtract); exact for unit phasors
    pub fn unbind(&self, o: &HVec) -> HVec {
        HVec(self.0.iter().zip(&o.0).map(|(a, b)| a * b.conj()).collect())
    }

    /// superposition, in place
    pub fn add(&mut self, o: &HVec) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            *a += b;
        }
    }

    pub fn add_scaled(&mut self, o: &HVec, w: f32) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            *a += b * w;
        }
    }

    pub fn bundle(vs: &[&HVec]) -> HVec {
        let mut s = HVec::zero(vs[0].dim());
        for v in vs {
            s.add(v);
        }
        s
    }

    /// a step in the state's order: every component moves one place on (a permutation, so unitary)
    pub fn shift(&self) -> HVec {
        let n = self.0.len();
        HVec((0..n).map(|i| self.0[(i + n - 1) % n]).collect())
    }

    /// the phases alone: every component brought to length 1 (zero stays zero)
    pub fn phase(&self) -> HVec {
        HVec(self.0.iter().map(|c| { let r = c.norm(); if r > 0.0 { c / r } else { *c } }).collect())
    }

    /// the complex inner product <self, o>
    pub fn inner(&self, o: &HVec) -> Complex32 {
        self.0.iter().zip(&o.0).map(|(a, b)| a * b.conj()).sum()
    }

    pub fn norm(&self) -> f32 {
        self.0.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt()
    }

    /// similarity: Re<a, b> / (|a| |b|), the interference of the two
    pub fn sim(&self, o: &HVec) -> f32 {
        let dot: f32 = self.0.iter().zip(&o.0).map(|(a, b)| (a * b.conj()).re).sum();
        let n = self.norm() * o.norm();
        if n == 0.0 {
            0.0
        } else {
            dot / n
        }
    }
}

/// measurement: the nearest known atom (cleanup memory), with its similarity and the runner-up's
pub fn cleanup<'a, K: Clone>(q: &HVec, atoms: impl Iterator<Item = (K, &'a HVec)>) -> Option<(K, f32, f32)> {
    let (mut best, mut b1, mut b2) = (None, f32::MIN, f32::MIN);
    for (k, v) in atoms {
        let s = q.sim(v);
        if s > b1 {
            b2 = b1;
            b1 = s;
            best = Some(k);
        } else if s > b2 {
            b2 = s;
        }
    }
    best.map(|k| (k, b1, b2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const D: usize = 1024;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(7)
    }

    #[test]
    fn unbind_gives_back_the_bound() {
        let mut r = rng();
        let (a, b) = (HVec::random(D, &mut r), HVec::random(D, &mut r));
        let s = a.bind(&b).unbind(&a).sim(&b);
        println!("unbind(bind(a, b), a) ~ b: similarity {s:.6}");
        assert!(s > 0.95);
    }

    #[test]
    fn random_atoms_are_nearly_orthogonal() {
        let mut r = rng();
        let v: Vec<HVec> = (0..200).map(|_| HVec::random(D, &mut r)).collect();
        let mut sims = Vec::new();
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                sims.push(v[i].sim(&v[j]).abs());
            }
        }
        let mean = sims.iter().sum::<f32>() / sims.len() as f32;
        let max = sims.iter().cloned().fold(0.0, f32::max);
        println!("{} pairs of random atoms: mean |similarity| {mean:.4}, largest {max:.4} (1/sqrt(D) = {:.4})", sims.len(), 1.0 / (D as f32).sqrt());
        assert!(mean < 0.05 && max < 0.2);
    }

    #[test]
    fn a_superposition_gives_back_its_parts() {
        let mut r = rng();
        let atoms: Vec<HVec> = (0..200).map(|_| HVec::random(D, &mut r)).collect();
        let roles: Vec<HVec> = (0..3).map(|_| HVec::random(D, &mut r)).collect();
        let parts = [17usize, 42, 123];
        let mut s = HVec::zero(D);
        for (role, &p) in roles.iter().zip(&parts) {
            s.add(&role.bind(&atoms[p]));
        }
        for (role, &p) in roles.iter().zip(&parts) {
            let (k, s1, s2) = cleanup(&s.unbind(role), atoms.iter().enumerate()).unwrap();
            println!("role -> atom {k} (wanted {p}): similarity {s1:.3}, runner-up {s2:.3}");
            assert_eq!(k, p);
            assert!(s1 - s2 > 0.2);
        }
    }

    #[test]
    fn interference_reinforces_and_cancels() {
        let mut r = rng();
        let a = HVec::random(D, &mut r);
        let mut same = a.clone();
        same.add(&a);
        let opposite = HVec(a.0.iter().map(|c| -c).collect());
        let mut cancel = a.clone();
        cancel.add(&opposite);
        println!("a + a: norm {:.1} (|a| = {:.1}); a + (-a): norm {:.3}", same.norm(), a.norm(), cancel.norm());
        assert!((same.norm() - 2.0 * a.norm()).abs() < 1e-2);
        assert!(cancel.norm() < 1e-3);
    }

    #[test]
    fn how_many_bound_pairs_a_superposition_holds() {
        // capacity: n role-filler pairs in one sum, each read back through cleanup over 200 atoms
        let mut r = rng();
        let atoms: Vec<HVec> = (0..200).map(|_| HVec::random(D, &mut r)).collect();
        for n in [3usize, 10, 30, 60, 100] {
            let roles: Vec<HVec> = (0..n).map(|_| HVec::random(D, &mut r)).collect();
            let fill: Vec<usize> = (0..n).map(|_| r.gen_range(0..atoms.len())).collect();
            let mut s = HVec::zero(D);
            for (ro, &f) in roles.iter().zip(&fill) {
                s.add(&ro.bind(&atoms[f]));
            }
            let right = roles.iter().zip(&fill).filter(|(ro, &f)| cleanup(&s.unbind(ro), atoms.iter().enumerate()).unwrap().0 == f).count();
            println!("{n:>3} pairs in one superposition: {right}/{n} read back right");
        }
    }
}
