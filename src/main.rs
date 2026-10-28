// Licensed under the Apache License, Version 2.0.
//! clojure-yarpgen: random Clojure program generator for compiler
//! stress testing, in the spirit of Intel YARPGen.

mod cli;
mod context;
mod emitter;
mod expr;
mod hash;
mod mutate;
mod negative;
mod policy;
mod rng;
mod runner;
mod sexp;
mod stats;
mod stmt;
mod types;

use cli::{MutateMode, NegativeKind, Options, Std};
use context::GenCtx;
use emitter::{Input, Output, Program};
use policy::Policy;
use rng::Rng;
use std::path::PathBuf;
use std::process::ExitCode;
use types::{CType, Value};

/// Generate one program: inputs, outputs, populated scopes, hash.
/// Ports YARPGen's `ProgramGenerator` constructor flow.
fn generate_program(seed: u64, options: &Options, policy: &Policy, rng: &mut Rng) -> Program {
    let mut genctx = GenCtx::new();
    let n_scalars = rng.gen_range(policy.scalar_inputs_min, policy.scalar_inputs_max);
    let mut inputs = Vec::new();
    for i in 0..n_scalars {
        let typ = *rng.pick(&policy.int_types);
        let c = types::random_const(rng, policy, typ);
        genctx.add_input(&format!("in-{i}"), typ, c.value.clone());
        inputs.push(Input {
            name: format!("in-{i}"),
            typ,
            value: c.value,
        });
    }
    for i in 0..2 {
        let v = types::random_vec(rng, policy);
        genctx.add_input(&format!("vec-{i}"), CType::Vec, v.value.clone());
        inputs.push(Input {
            name: format!("vec-{i}"),
            typ: CType::Vec,
            value: v.value,
        });
    }
    for i in 0..2 {
        let m = types::random_map(rng);
        genctx.add_input(&format!("map-{i}"), CType::Map, m.value.clone());
        inputs.push(Input {
            name: format!("map-{i}"),
            typ: CType::Map,
            value: m.value,
        });
    }
    for i in 0..2 {
        let r = types::random_record(rng);
        genctx.add_input(&format!("rec-{i}"), CType::Record, r.value.clone());
        inputs.push(Input {
            name: format!("rec-{i}"),
            typ: CType::Record,
            value: r.value,
        });
    }
    let mut outputs = Vec::new();
    for i in 0..n_scalars {
        let c = types::random_const(rng, policy, CType::Long);
        let init = c.value.as_long();
        genctx.add_output(&format!("out-{i}"), CType::Long, Value::Long(init));
        outputs.push(Output {
            name: format!("out-{i}"),
            init,
            final_value: init,
        });
    }
    // Every top-level statement must fit in one emitted function;
    // oversized units are repaired per statement so one big loop nest
    // can't nuke the whole scope's control-flow diversity.
    let (final_ctx, stmts) =
        stmt::populate_top_scope(rng, policy, genctx, options.allow_exc_in_dead);
    for o in outputs.iter_mut() {
        o.final_value = final_ctx
            .lookup(&o.name)
            .map(|v| v.as_long())
            .unwrap_or(o.init);
    }
    let pairs: Vec<(String, i64)> = outputs
        .iter()
        .map(|o| (o.name.clone(), o.final_value))
        .collect();
    let mut program = Program {
        seed,
        inputs,
        outputs,
        stmts,
        expected_hash: hash::checksum(&pairs),
        check_algo: options.check_algo,
        mutation_seed: None,
    };
    if options.mutate != MutateMode::None {
        let mut_seed = if options.mutation_seed == 0 {
            rng.next_u64()
        } else {
            options.mutation_seed
        };
        let mut mut_rng = rng.fork(mut_seed);
        program = mutate::mutate_program(
            &program,
            &mut mut_rng,
            policy,
            options.allow_exc_in_dead,
            options.mutate,
        );
        program.mutation_seed = Some(mut_seed);
    }
    program
}

/// Generate one program and emit it to `dir`. Returns the program
/// so callers can report on it.
fn generate_to_dir(options: &Options, seed: u64, dir: &std::path::Path) -> Option<Program> {
    let mut rng = Rng::new(seed);
    let policy = policy::make_policy(options, &mut rng);
    let program = generate_program(seed, options, &policy, &mut rng);
    if let Err(e) = emitter::emit_test(dir, &program, options, &mut rng) {
        eprintln!("error: failed to emit test: {e}");
        return None;
    }
    Some(program)
}

/// Generation flags the negative path ignores (it uses only seed,
/// out-dir, and max-nesting): warn rather than silently drop them.
/// `--count` keeps its own message below.
fn ignored_negative_flags(options: &Options) -> Vec<&'static str> {
    let d = Options::default();
    let mut v = Vec::new();
    if options.std != d.std {
        v.push("--std");
    }
    if options.check_algo != d.check_algo {
        v.push("--check-algo");
    }
    if options.emit_pragmas != d.emit_pragmas {
        v.push("--emit-pragmas");
    }
    if options.param_shuffle != d.param_shuffle {
        v.push("--param-shuffle");
    }
    if options.expl_loop_param != d.expl_loop_param {
        v.push("--expl-loop-param");
    }
    if options.mutate != d.mutate {
        v.push("--mutate");
    }
    if options.mutation_seed != d.mutation_seed {
        v.push("--mutation-seed");
    }
    if options.allow_exc_in_dead != d.allow_exc_in_dead {
        v.push("--allow-exc-in-dead");
    }
    if options.max_array_dims != d.max_array_dims {
        v.push("--max-array-dims");
    }
    if options.simple_loops != d.simple_loops {
        v.push("--simple-loops");
    }
    if options.unchecked_math != d.unchecked_math {
        v.push("--unchecked-math");
    }
    if options.dg_headers != d.dg_headers {
        v.push("--dg-headers");
    }
    if options.max_fn_cost != d.max_fn_cost {
        v.push("--max-fn-cost");
    }
    if options.top_stmts != d.top_stmts {
        v.push("--top-stmts");
    }
    v
}

fn run_generate(options: &Options) -> ExitCode {
    if options.negative != NegativeKind::None {
        if options.count != 1 {
            eprintln!("warning: --count is ignored with --negative (one suite)");
        }
        let ignored = ignored_negative_flags(options);
        if !ignored.is_empty() {
            eprintln!("warning: {} ignored with --negative", ignored.join(", "));
        }
        let mut rng = Rng::new(options.seed);
        let seed = rng.seed();
        let cases = negative::generate_cases(&mut rng, options.negative, options.max_nesting);
        let dir = PathBuf::from(&options.out_dir);
        match negative::write_cases(&dir, seed, &cases) {
            Ok(paths) => {
                println!("seed={seed} dir={} cases={}", dir.display(), paths.len());
                for p in &paths {
                    println!("  {}", p.display());
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: failed to write negative suite: {e}");
                ExitCode::FAILURE
            }
        }
    } else {
        let base_seed = Rng::new(options.seed).seed();
        let count = options.count.max(1);
        for i in 0..count {
            let seed = base_seed.wrapping_add(i as u64);
            let dir = if count == 1 {
                PathBuf::from(&options.out_dir)
            } else {
                PathBuf::from(&options.out_dir).join(format!("seed-{seed}"))
            };
            let program = match generate_to_dir(options, seed, &dir) {
                Some(p) => p,
                None => return ExitCode::FAILURE,
            };
            let st = stats::program_stats(&program);
            let mut line = format!(
                "seed={seed} dir={} hash={} stmts={} exprs={}",
                dir.display(),
                program.expected_hash,
                st.get("stmts").unwrap_or(&0),
                st.get("exprs").unwrap_or(&0)
            );
            if let Some(m) = program.mutation_seed {
                line.push_str(&format!(" mut-seed={m}"));
            }
            println!("{line}");
        }
        ExitCode::SUCCESS
    }
}

/// Options fuzz cannot serve, with the user-facing reason.
fn fuzz_reject_reason(options: &Options) -> Option<&'static str> {
    if options.negative != NegativeKind::None {
        Some("fuzz does not support --negative (different oracle)")
    } else if options.std == Std::Cljs {
        Some("fuzz does not support --std cljs (execution is JVM-only)")
    } else {
        None
    }
}

/// Fuzz pre-dispatch decision, resolved before any generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FuzzGate {
    Help,
    Version,
    Errors,
    Rejected,
    Go,
}

fn fuzz_gate(parsed: &cli::Parsed) -> FuzzGate {
    if parsed.help {
        FuzzGate::Help
    } else if parsed.version {
        FuzzGate::Version
    } else if !parsed.errors.is_empty() {
        FuzzGate::Errors
    } else if fuzz_reject_reason(&parsed.options).is_some() {
        FuzzGate::Rejected
    } else {
        FuzzGate::Go
    }
}

/// Unattended fuzz loop: generate `--count` programs from `-s` into
/// per-seed subdirs and verify each across all backends.
fn run_fuzz(args: &[String]) -> ExitCode {
    let parsed = cli::parse_args(args);
    match fuzz_gate(&parsed) {
        FuzzGate::Help => {
            print_full_help();
            return ExitCode::SUCCESS;
        }
        FuzzGate::Version => {
            println!("clojure-yarpgen {}", cli::VERSION);
            return ExitCode::SUCCESS;
        }
        FuzzGate::Errors => {
            for e in &parsed.errors {
                eprintln!("error: {e}");
            }
            return ExitCode::from(2);
        }
        FuzzGate::Rejected => {
            if let Some(reason) = fuzz_reject_reason(&parsed.options) {
                eprintln!("error: {reason}");
            }
            return ExitCode::from(2);
        }
        FuzzGate::Go => {}
    }
    let options = parsed.options;
    let classpath = match runner::find_clojure_classpath() {
        Some(cp) => cp,
        None => {
            eprintln!("error: no Clojure jar found; set CLOJURE_JAR to a clojure.jar path");
            return ExitCode::FAILURE;
        }
    };
    let base_seed = Rng::new(options.seed).seed();
    println!("fuzz base-seed={base_seed} dir={}", options.out_dir);
    let count = options.count.max(1);
    let mut passed = 0;
    let mut failed: Vec<u64> = Vec::new();
    for i in 0..count {
        let seed = base_seed.wrapping_add(i as u64);
        let dir = PathBuf::from(&options.out_dir).join(format!("seed-{seed}"));
        if generate_to_dir(&options, seed, &dir).is_none() {
            failed.push(seed);
            continue;
        }
        match runner::verify(&dir, &classpath, std::time::Duration::from_secs(120), None) {
            Some(verdict) => {
                if verdict.pass {
                    passed += 1;
                    println!("[{seed}] PASS hash={}", verdict.expected);
                } else {
                    failed.push(seed);
                    println!("[{seed}] FAIL dir={}", dir.display());
                    runner::report(&verdict);
                }
            }
            None => {
                failed.push(seed);
                println!("[{seed}] ERROR dir={} (no expected.edn)", dir.display());
            }
        }
    }
    println!("fuzz: {passed} pass {} fail of {count}", failed.len());
    if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Verify argument dispatch: help/version short-circuit before
/// any directory is treated as a test.
#[derive(Clone, Debug, PartialEq, Eq)]
enum VerifyAction {
    Help,
    Version,
    Usage,
    Run { dir: PathBuf, only: Vec<String> },
}

fn verify_action(args: &[String]) -> VerifyAction {
    if args.is_empty() {
        VerifyAction::Usage
    } else if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        VerifyAction::Help
    } else if args.len() == 1 && (args[0] == "--version" || args[0] == "-v") {
        VerifyAction::Version
    } else {
        VerifyAction::Run {
            dir: PathBuf::from(&args[0]),
            only: args[1..].to_vec(),
        }
    }
}

/// Backend names in `only` that match no known backend. Unknown
/// names must error: silently filtering them yields zero runs and a
/// bare FAIL that reads exactly like a hash mismatch.
fn unknown_backends(only: &[String]) -> Vec<String> {
    let known: Vec<&str> = runner::backends().iter().map(|b| b.name).collect();
    only.iter()
        .filter(|n| !known.contains(&n.as_str()))
        .cloned()
        .collect()
}

fn run_verify(args: &[String]) -> ExitCode {
    let (dir, only) = match verify_action(args) {
        VerifyAction::Help => {
            print_full_help();
            return ExitCode::SUCCESS;
        }
        VerifyAction::Version => {
            println!("clojure-yarpgen {}", cli::VERSION);
            return ExitCode::SUCCESS;
        }
        VerifyAction::Usage => {
            eprintln!("Usage: clojure-yarpgen verify <test-dir> [backend...]");
            return ExitCode::from(2);
        }
        VerifyAction::Run { dir, only } => (dir, only),
    };
    let unknown = unknown_backends(&only);
    if !unknown.is_empty() {
        let known: Vec<&str> = runner::backends().iter().map(|b| b.name).collect();
        eprintln!(
            "error: unknown backend '{}' (expected one of: {})",
            unknown.join("', '"),
            known.join(", ")
        );
        return ExitCode::from(2);
    }
    let classpath = match runner::find_clojure_classpath() {
        Some(cp) => cp,
        None => {
            eprintln!("error: no Clojure jar found; set CLOJURE_JAR to a clojure.jar path");
            return ExitCode::FAILURE;
        }
    };
    match runner::verify(
        &dir,
        &classpath,
        std::time::Duration::from_secs(120),
        if only.is_empty() { None } else { Some(&only) },
    ) {
        Some(verdict) => {
            if runner::report(&verdict) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        None => {
            eprintln!(
                "error: {} is not a generated test (no expected.edn)",
                dir.display()
            );
            ExitCode::FAILURE
        }
    }
}

fn print_full_help() {
    print!("{}", cli::usage());
    println!(
        "Commands:\n  verify DIR [backend...]  run a generated test across backends\n  fuzz [options]           generate --count tests and verify each"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("verify") {
        return run_verify(&args[1..]);
    }
    if args.first().map(|s| s.as_str()) == Some("fuzz") {
        return run_fuzz(&args[1..]);
    }
    let parsed = cli::parse_args(&args);
    if parsed.help {
        print_full_help();
        return ExitCode::SUCCESS;
    }
    if parsed.version {
        println!("clojure-yarpgen {}", cli::VERSION);
        return ExitCode::SUCCESS;
    }
    if !parsed.errors.is_empty() {
        for e in &parsed.errors {
            eprintln!("error: {e}");
        }
        eprintln!("{}", cli::usage());
        return ExitCode::from(2);
    }
    run_generate(&parsed.options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn fuzz_gate_routes_help_version_errors() {
        // --help/--version must short-circuit: running a random fuzz
        // (and writing seed dirs) on a help flag is a side effect.
        assert_eq!(
            fuzz_gate(&cli::parse_args(&words(&["--help"]))),
            FuzzGate::Help
        );
        assert_eq!(
            fuzz_gate(&cli::parse_args(&words(&["--version"]))),
            FuzzGate::Version
        );
        assert_eq!(
            fuzz_gate(&cli::parse_args(&words(&["--bogus", "1"]))),
            FuzzGate::Errors
        );
        assert_eq!(
            fuzz_gate(&cli::parse_args(&words(&["--std", "cljs"]))),
            FuzzGate::Rejected
        );
        assert_eq!(fuzz_gate(&cli::parse_args(&words(&[]))), FuzzGate::Go);
    }

    #[test]
    fn verify_action_routes_help_version_usage() {
        assert_eq!(verify_action(&words(&[])), VerifyAction::Usage);
        assert_eq!(verify_action(&words(&["--help"])), VerifyAction::Help);
        assert_eq!(verify_action(&words(&["-v"])), VerifyAction::Version);
        match verify_action(&words(&["dir", "default"])) {
            VerifyAction::Run { dir, only } => {
                assert_eq!(dir, PathBuf::from("dir"));
                assert_eq!(only, vec!["default".to_string()]);
            }
            other => panic!("expected Run, got {other:?}"),
        }
    }

    #[test]
    fn unknown_backends_are_reported() {
        assert!(unknown_backends(&[]).is_empty());
        assert!(unknown_backends(&words(&["default", "c1-only", "direct-link", "bb"])).is_empty());
        assert_eq!(unknown_backends(&words(&["c1"])), vec!["c1".to_string()]);
        assert_eq!(
            unknown_backends(&words(&["default", "--help"])),
            vec!["--help".to_string()]
        );
    }

    #[test]
    fn negative_ignores_generation_flags() {
        use crate::cli::{CheckAlgo, MutateMode};
        assert!(ignored_negative_flags(&Options::default()).is_empty());
        // Used flags (seed, out-dir, max-nesting, count) never warn.
        let used = Options {
            seed: 7,
            out_dir: "x".to_string(),
            max_nesting: 3,
            count: 9,
            ..Options::default()
        };
        assert!(ignored_negative_flags(&used).is_empty());
        let ignored = Options {
            mutate: MutateMode::All,
            check_algo: CheckAlgo::Asserts,
            top_stmts: 50,
            ..Options::default()
        };
        assert_eq!(
            ignored_negative_flags(&ignored),
            vec!["--check-algo", "--mutate", "--top-stmts"]
        );
    }

    #[test]
    fn fuzz_rejects_negative_and_cljs() {
        let negative = Options {
            negative: NegativeKind::All,
            ..Options::default()
        };
        assert!(fuzz_reject_reason(&negative).is_some());
        let cljs = Options {
            std: Std::Cljs,
            ..Options::default()
        };
        assert!(fuzz_reject_reason(&cljs).is_some());
        assert!(fuzz_reject_reason(&Options::default()).is_none());
    }
}
