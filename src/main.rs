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

use cli::{MutateMode, NegativeKind, Options};
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
    // repopulate on oversized statements, with a size-guaranteed
    // straight-line scope as the final fallback.
    let (final_ctx, stmts) = {
        let mut result = None;
        for _ in 0..3 {
            let (ctx, stmts) =
                stmt::populate_scope(rng, policy, genctx.clone(), options.allow_exc_in_dead, 0);
            if stmts
                .iter()
                .all(|s| stats::stmt_cost(s) <= emitter::MAX_STMT_COST)
            {
                result = Some((ctx, stmts));
                break;
            }
        }
        result.unwrap_or_else(|| {
            stmt::populate_stub_scope(rng, policy, genctx, options.allow_exc_in_dead, 0)
        })
    };
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

fn run_generate(options: &Options) -> ExitCode {
    if options.negative != NegativeKind::None {
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

/// Unattended fuzz loop: generate `--count` programs from `-s` into
/// per-seed subdirs and verify each across all backends.
fn run_fuzz(args: &[String]) -> ExitCode {
    let parsed = cli::parse_args(args);
    if !parsed.errors.is_empty() {
        for e in &parsed.errors {
            eprintln!("error: {e}");
        }
        return ExitCode::from(2);
    }
    let options = parsed.options;
    if options.negative != NegativeKind::None {
        eprintln!("error: fuzz does not support --negative (different oracle)");
        return ExitCode::from(2);
    }
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

fn run_verify(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: clojure-yarpgen verify <test-dir> [backend...]");
        return ExitCode::from(2);
    }
    let dir = PathBuf::from(&args[0]);
    let only: Vec<String> = args[1..].to_vec();
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
        print!("{}", cli::usage());
        println!(
            "Commands:\n  verify DIR [backend...]  run a generated test across backends\n  fuzz [options]           generate --count tests and verify each"
        );
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
