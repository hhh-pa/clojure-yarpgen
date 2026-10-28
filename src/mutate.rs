// Licensed under the Apache License, Version 2.0.
//! Program mutation. Ports YARPGen's `--mutate`/`--mutation-seed`: an
//! auxiliary RNG re-rolls assignment right-hand sides (exprs) and,
//! for `All`, loop bounds too.

use crate::cli::{Density, MutateMode};
use crate::context::GenCtx;
use crate::emitter::{Output, Program};
use crate::expr;
use crate::hash;
use crate::policy::{self, Policy};
use crate::rng::Rng;
use crate::stmt::{self, Stmt};

fn mutate_stmt(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    mode: MutateMode,
    s: &Stmt,
) -> (GenCtx, Stmt) {
    match s {
        Stmt::Assign { target, .. } => {
            let rhs = expr::generate_safe(rng, policy, genctx, false, exc_mode);
            let mut next = genctx.clone();
            next.update_value(target, rhs.ir().value.clone());
            (
                next,
                Stmt::Assign {
                    target: target.clone(),
                    expr: rhs,
                },
            )
        }
        Stmt::If { cond, then, els } => {
            let (_, t) = mutate_scope(rng, policy, &genctx.with_taken(true), exc_mode, mode, then);
            let (_, e) = mutate_scope(rng, policy, &genctx.with_taken(false), exc_mode, mode, els);
            (
                genctx.clone(),
                Stmt::If {
                    cond: cond.clone(),
                    then: t,
                    els: e,
                },
            )
        }
        Stmt::Loop {
            var,
            start,
            end,
            step,
            body,
        } => {
            let (next, body) = mutate_scope(rng, policy, genctx, exc_mode, mode, body);
            let (start, end, step) = if mode == MutateMode::All {
                let (s, e, st) = policy::loop_bounds(rng, policy);
                (s, e, st)
            } else {
                (*start, *end, *step)
            };
            (
                next,
                Stmt::Loop {
                    var: var.clone(),
                    start,
                    end,
                    step,
                    body,
                },
            )
        }
    }
}

fn mutate_scope(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    mode: MutateMode,
    stmts: &[Stmt],
) -> (GenCtx, Vec<Stmt>) {
    let mut ctx = genctx.clone();
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        if rng.roll(0.5) {
            let (next, s2) = mutate_stmt(rng, policy, &ctx, exc_mode, mode, s);
            ctx = next;
            out.push(s2);
        } else {
            out.push(s.clone());
        }
    }
    (ctx, out)
}

/// Mutate a generated program. Returns the program with fresh
/// statements, final values, and hash.
pub fn mutate_program(
    program: &Program,
    rng: &mut Rng,
    policy: &Policy,
    exc_mode: Density,
    mode: MutateMode,
) -> Program {
    let mut genctx = GenCtx::new();
    for i in &program.inputs {
        genctx.add_input(&i.name, i.typ, i.value.clone());
    }
    for o in &program.outputs {
        genctx.add_output(
            &o.name,
            crate::types::CType::Long,
            crate::types::Value::Long(o.init),
        );
    }
    // Mutated right-hand sides may fault under evolved loop values;
    // retry with fresh rolls, shrinking on budget pressure, and fall
    // back to the original program when nothing validates.
    let mut attempt = 0;
    let (stmts, final_ctx) = loop {
        let (_, mut stmts) = mutate_scope(rng, policy, &genctx, exc_mode, mode, &program.stmts);
        let mut trial = genctx.clone();
        let mut ok = false;
        for _ in 0..6 {
            let mut ctx = genctx.clone();
            match stmt::simulate_bounded_whole(&mut ctx, &stmts) {
                Ok(()) => {
                    // Re-rolled right-hand sides must not push a single
                    // statement past one function's size cap.
                    if stmts
                        .iter()
                        .all(|s| crate::stats::stmt_cost(s) <= crate::emitter::MAX_STMT_COST)
                    {
                        trial = ctx;
                        ok = true;
                    }
                    break;
                }
                Err(stmt::SimErr::Budget) => {
                    let mut shrunk = false;
                    for s in stmts.iter_mut() {
                        shrunk |= stmt::shrink_loop_bounds(s);
                    }
                    if !shrunk {
                        break;
                    }
                }
                Err(stmt::SimErr::Fault) => break,
            }
        }
        if ok {
            break (stmts, trial);
        }
        attempt += 1;
        if attempt >= 3 {
            let mut ctx = genctx.clone();
            stmt::simulate_stmts(&mut ctx, &program.stmts);
            break (program.stmts.clone(), ctx);
        }
    };
    let outputs: Vec<Output> = program
        .outputs
        .iter()
        .map(|o| Output {
            name: o.name.clone(),
            init: o.init,
            final_value: final_ctx
                .lookup(&o.name)
                .map(|v| v.long_value())
                .unwrap_or(o.init),
        })
        .collect();
    let pairs: Vec<(String, i64)> = outputs
        .iter()
        .map(|o| (o.name.clone(), o.final_value))
        .collect();
    Program {
        seed: program.seed,
        inputs: program.inputs.clone(),
        outputs,
        stmts,
        expected_hash: hash::checksum(&pairs),
        check_algo: program.check_algo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::Input;
    use crate::expr::Expr;
    use crate::types::{CType, Ir, Value};

    fn sample() -> Program {
        Program {
            seed: 9,
            inputs: vec![Input {
                name: "in-0".to_string(),
                typ: CType::Long,
                value: Value::Long(1),
            }],
            outputs: vec![Output {
                name: "out-0".to_string(),
                init: 0,
                final_value: 0,
            }],
            stmts: vec![Stmt::Assign {
                target: "out-0".to_string(),
                expr: Expr::Const(Ir::long(2)),
            }],
            expected_hash: 0,
            check_algo: crate::cli::CheckAlgo::Hash,
        }
    }

    #[test]
    fn mutation_keeps_program_coherent() {
        let mut rng = Rng::new(3);
        let policy = Policy::default();
        let m = mutate_program(&sample(), &mut rng, &policy, Density::None, MutateMode::All);
        assert_eq!(m.stmts.len(), 1);
        assert_eq!(m.outputs.len(), 1);
        let pairs = vec![("out-0".to_string(), m.outputs[0].final_value)];
        assert_eq!(m.expected_hash, hash::checksum(&pairs));
    }
}
