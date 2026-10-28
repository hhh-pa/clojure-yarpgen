// Licensed under the Apache License, Version 2.0.
//! Seeded RNG (SplitMix64). Ports YARPGen's utils.h probability helpers
//! (`Probability<T>` + `discrete_distribution`) with no dependencies.

use std::time::{SystemTime, UNIX_EPOCH};

/// Deterministic 64-bit RNG.
#[derive(Clone, Debug)]
pub struct Rng {
    seed: u64,
    state: u64,
}

impl Rng {
    /// Create an RNG. Seed 0 means "pick a random seed"; the effective
    /// seed is retained for logging and reproduction.
    pub fn new(seed: u64) -> Self {
        let eff = if seed == 0 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9e3779b97f4a7c15);
            nanos ^ (std::process::id() as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        } else {
            seed
        };
        Rng {
            seed: eff,
            state: eff,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Derive a child RNG (for mutation state) by salting the
    /// seed. Deterministic but not statistically independent: unlike
    /// [`Rng::new`], a derived zero is remapped instead of meaning
    /// "random", keeping mutations reproducible for every salt.
    pub fn fork(&self, salt: u64) -> Self {
        let eff = self.seed ^ salt.wrapping_mul(0x94d049bb133111eb);
        let eff = if eff == 0 { 1 } else { eff };
        Rng {
            seed: eff,
            state: eff,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        const DIV: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DIV
    }

    /// Random value in `[lo, hi]` inclusive.
    pub fn gen_range(&mut self, lo: i64, hi: i64) -> i64 {
        assert!(lo <= hi, "empty range {lo}..={hi}");
        if lo == hi {
            return lo;
        }
        let span = (hi as u64).wrapping_sub(lo as u64).wrapping_add(1);
        (lo as i128 + (self.next_u64() as u128 % span as u128) as i128) as i64
    }

    pub fn gen_usize(&mut self, n: usize) -> usize {
        assert!(n > 0, "cannot pick from empty collection");
        (self.next_u64() % n as u64) as usize
    }

    pub fn gen_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// True with probability `p`.
    pub fn roll(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Uniform random element.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.gen_usize(items.len())]
    }

    /// Weighted pick from `(value, weight)` pairs. Ports the
    /// `discrete_distribution` over `Probability<T>` parts.
    pub fn pick_weighted<T: Clone>(&mut self, weights: &[(T, u32)]) -> T {
        let total: u64 = weights.iter().map(|(_, w)| *w as u64).sum();
        assert!(total > 0, "empty weight table");
        let mut x = self.next_u64() % total;
        for (v, w) in weights {
            let w = *w as u64;
            if x < w {
                return v.clone();
            }
            x -= w;
        }
        weights.last().unwrap().0.clone()
    }

    /// Randomize a weight table in place, preserving keys and order.
    /// Ports `--param-shuffle` behaviour.
    pub fn shuffle_weights<T>(&mut self, weights: &mut [(T, u32)]) {
        for (_, w) in weights.iter_mut() {
            *w = self.gen_range(1, 100) as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(
            (0..10).map(|_| a.next_u64()).collect::<Vec<_>>(),
            (0..10).map(|_| b.next_u64()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn weighted_pick_respects_zero_weights() {
        let mut rng = Rng::new(7);
        let table = vec![("a", 0), ("b", 100)];
        for _ in 0..50 {
            assert_eq!(rng.pick_weighted(&table), "b");
        }
    }
}
