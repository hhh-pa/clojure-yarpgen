// Licensed under the Apache License, Version 2.0.
//! Differential oracle: boost-style hash combine over output variables.
//! Ports YARPGen's program.cpp checksum emission
//! (`*seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2)`, seed starts 0).

use crate::sexp::Sexp;

pub const GOLDEN: i64 = 0x9e3779b9;

/// Fold one 64-bit value into the running seed (wrapping long math,
/// matching Clojure's `unchecked-*` ops).
pub fn combine(seed: i64, v: i64) -> i64 {
    seed ^ v.wrapping_add(GOLDEN).wrapping_add(
        seed.wrapping_shl(6)
            .wrapping_add(((seed as u64) >> 2) as i64),
    )
}

/// Checksum over `(name, value)` outputs in sorted-name order.
pub fn checksum(outputs: &[(String, i64)]) -> i64 {
    let mut sorted = outputs.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted.iter().fold(0i64, |s, (_, v)| combine(s, *v))
}

/// S-expression for the emitted driver: reduces `combine` over the
/// values returned by `(gen-func/outputs)`, which the emitter writes in
/// sorted-name order to match [`checksum`].
pub fn checksum_sexp() -> Sexp {
    let combine_fn = Sexp::list(vec![
        Sexp::sym("fn"),
        Sexp::vec(vec![Sexp::sym("s"), Sexp::sym("v")]),
        Sexp::list(vec![
            Sexp::sym("unchecked-long"),
            Sexp::list(vec![
                Sexp::sym("bit-xor"),
                Sexp::sym("s"),
                Sexp::list(vec![
                    Sexp::sym("unchecked-add"),
                    Sexp::list(vec![
                        Sexp::sym("unchecked-add"),
                        Sexp::list(vec![Sexp::sym("long"), Sexp::sym("v")]),
                        Sexp::int(GOLDEN),
                    ]),
                    Sexp::list(vec![
                        Sexp::sym("unchecked-add"),
                        Sexp::list(vec![
                            Sexp::sym("bit-shift-left"),
                            Sexp::sym("s"),
                            Sexp::int(6),
                        ]),
                        Sexp::list(vec![
                            Sexp::sym("unsigned-bit-shift-right"),
                            Sexp::sym("s"),
                            Sexp::int(2),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]);
    Sexp::list(vec![
        Sexp::sym("reduce"),
        combine_fn,
        Sexp::int(0),
        Sexp::list(vec![Sexp::sym("gen-func/outputs")]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_matches_boost_sequence() {
        // Hand-computed: seed=0, v=1.
        // t = 1 + 0x9e3779b9 + (0<<6) + (0>>2) = 0x9e3779ba; 0 ^ t = t.
        assert_eq!(combine(0, 1), 0x9e3779bai64);
        // Order matters.
        assert_ne!(combine(combine(0, 1), 2), combine(combine(0, 2), 1));
    }

    #[test]
    fn checksum_sorts_by_name() {
        let a = vec![("b".to_string(), 2), ("a".to_string(), 1)];
        let b = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        assert_eq!(checksum(&a), checksum(&b));
        assert_eq!(checksum(&b), combine(combine(0, 1), 2));
    }

    #[test]
    fn driver_sexp_mentions_outputs() {
        assert!(checksum_sexp().flat().contains("gen-func/outputs"));
    }
}
