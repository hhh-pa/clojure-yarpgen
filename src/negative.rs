// Licensed under the Apache License, Version 2.0.
//! Negative suite: programs that MUST fail to compile or run, each with
//! an expected error. Probes hard compiler limits (arity, recur rules,
//! nesting depth, type hints) rather than optimizer behavior.

use crate::cli::NegativeKind;
use crate::rng::Rng;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// One must-fail case.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub content: String,
    pub expected_error: String,
    pub description: String,
}

fn arity_cases(rng: &mut Rng) -> Vec<Case> {
    let n = rng.gen_range(21, 30);
    let params: Vec<String> = (0..n).map(|i| format!("a{i}")).collect();
    let sum = params.join(" ");
    vec![
        Case {
            name: "arity-too-many-params".to_string(),
            content: format!("(defn too-many [{}] (+ {}))", params.join(" "), sum),
            expected_error: "more than 20 params".to_string(),
            description: format!("{n} fixed params, limit is 20"),
        },
        Case {
            name: "arity-double-variadic".to_string(),
            content: "(defn two-rest ([a & r1] r1) ([a & r2] r2))".to_string(),
            expected_error: "variadic".to_string(),
            description: "two variadic overloads in one defn".to_string(),
        },
    ]
}

fn recur_cases() -> Vec<Case> {
    vec![
        Case {
            name: "recur-non-tail".to_string(),
            content: "(defn non-tail [n] (if (zero? n) 0 (+ 1 (recur (dec n)))))"
                .to_string(),
            expected_error: "tail position".to_string(),
            description: "recur in non-tail position".to_string(),
        },
        Case {
            name: "recur-across-try".to_string(),
            content: "(defn across-try [n] (loop [i n] (if (zero? i) i (try (recur (dec i)) (catch Exception _ i)))))"
                .to_string(),
            expected_error: "across try".to_string(),
            description: "recur across a try boundary".to_string(),
        },
        Case {
            name: "recur-arity-mismatch".to_string(),
            content: "(defn arity-mismatch [n] (loop [i n j 0] (if (zero? i) j (recur (dec i)))))"
                .to_string(),
            expected_error: "Mismatched argument count".to_string(),
            description: "recur with wrong arg count".to_string(),
        },
    ]
}

fn nesting_cases(rng: &mut Rng, max_nesting: i64) -> Vec<Case> {
    let depth = if max_nesting > 0 {
        max_nesting as usize
    } else {
        rng.gen_range(2000, 8000) as usize
    };
    let mut deep = "0".to_string();
    for _ in 0..depth {
        deep = format!("(+ 1 {deep})");
    }
    let ladder_n = depth.min(4000);
    let mut ladder = "0".to_string();
    for k in 0..ladder_n {
        ladder = format!("(let [x{k} {ladder}] x{k})");
    }
    vec![
        Case {
            name: format!("nesting-depth-{depth}"),
            content: format!("(defn deep [] {deep})"),
            expected_error: "StackOverflowError".to_string(),
            description: format!("expression nested {depth} deep"),
        },
        Case {
            name: "nesting-let-ladder".to_string(),
            content: ladder,
            expected_error: "StackOverflowError".to_string(),
            description: "deeply nested let chain".to_string(),
        },
    ]
}

fn hint_cases() -> Vec<Case> {
    vec![
        Case {
            name: "hints-primitive-variadic".to_string(),
            content: "(defn prim-rest [^long x & more] more)".to_string(),
            expected_error: "variadic".to_string(),
            description: "primitive hint combined with variadic args".to_string(),
        },
        Case {
            name: "hints-ambiguous-literal".to_string(),
            content: "(defn hint-literal [] (let [^String s 42] (.length s)))".to_string(),
            expected_error: "Can't type hint a local".to_string(),
            description: "wrong hint: String-hinted long literal (compile-time)".to_string(),
        },
    ]
}

/// Generate negative cases for the requested kind.
pub fn generate_cases(rng: &mut Rng, kind: NegativeKind, max_nesting: i64) -> Vec<Case> {
    match kind {
        NegativeKind::None => vec![],
        NegativeKind::Arity => arity_cases(rng),
        NegativeKind::Recur => recur_cases(),
        NegativeKind::Nesting => nesting_cases(rng, max_nesting),
        NegativeKind::Hints => hint_cases(),
        NegativeKind::All => {
            let mut v = arity_cases(rng);
            v.extend(recur_cases());
            v.extend(nesting_cases(rng, max_nesting));
            v.extend(hint_cases());
            v
        }
    }
}

/// Write each case as `kebab-case-NNN.clj` plus a `manifest.edn`.
/// Returns the written file paths.
pub fn write_cases(out_dir: &Path, seed: u64, cases: &[Case]) -> io::Result<Vec<PathBuf>> {
    fs::create_dir_all(out_dir)?;
    let mut paths = Vec::with_capacity(cases.len());
    let mut manifest = String::from("[");
    for (i, case) in cases.iter().enumerate() {
        let fname = format!("{}-{:03}.clj", case.name, i + 1);
        let path = out_dir.join(&fname);
        fs::write(
            &path,
            format!(
                ";; Negative case, seed {seed}: {}\n;; Expected error: {}\n{}\n",
                case.description, case.expected_error, case.content
            ),
        )?;
        manifest.push_str(&format!(
            "{{:file \"{fname}\" :expected-error \"{}\" :description \"{}\"}}\n ",
            case.expected_error, case.description
        ));
        paths.push(path);
    }
    manifest.push(']');
    fs::write(out_dir.join("manifest.edn"), manifest)?;
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kinds_produce_cases() {
        let mut rng = Rng::new(11);
        for kind in [
            NegativeKind::Arity,
            NegativeKind::Recur,
            NegativeKind::Nesting,
            NegativeKind::Hints,
            NegativeKind::All,
        ] {
            let cases = generate_cases(&mut rng, kind, 0);
            assert!(!cases.is_empty(), "{kind:?} produced no cases");
            for c in &cases {
                assert!(!c.expected_error.is_empty());
                assert!(!c.content.is_empty());
            }
        }
        assert!(generate_cases(&mut rng, NegativeKind::None, 0).is_empty());
    }

    #[test]
    fn max_nesting_caps_depth_case() {
        let mut rng = Rng::new(11);
        let cases = generate_cases(&mut rng, NegativeKind::Nesting, 50);
        assert!(cases[0].name.contains("depth-50"));
    }
}
