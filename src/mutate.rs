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
use std::collections::BTreeSet;

fn mutate_stmt(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    mode: MutateMode,
    depth: i64,
    s: &Stmt,
) -> (GenCtx, Stmt) {
    match s {
        Stmt::Assign {
            target, target_id, ..
        } => {
            // Output ids are stable: generation and mutation seed the
            // context with inputs then outputs in the same order, and
            // assignment targets are always outputs.
            debug_assert_eq!(genctx.id_of(target), Some(*target_id));
            let rhs = expr::generate_safe(rng, policy, genctx, false, exc_mode);
            let mut next = genctx.clone();
            next.update_value_by_id(*target_id, rhs.ir().value.clone());
            (
                next,
                Stmt::Assign {
                    target: target.clone(),
                    target_id: *target_id,
                    expr: rhs,
                },
            )
        }
        Stmt::If { cond, then, els } => {
            let (_, t) = mutate_scope(
                rng,
                policy,
                &genctx.with_taken(true),
                exc_mode,
                mode,
                depth,
                then,
            );
            let (_, e) = mutate_scope(
                rng,
                policy,
                &genctx.with_taken(false),
                exc_mode,
                mode,
                depth,
                els,
            );
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
            let (next, body) = mutate_scope(rng, policy, genctx, exc_mode, mode, depth + 1, body);
            let (start, end, step) = if mode == MutateMode::All {
                policy::loop_bounds_at_depth(rng, policy, depth)
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
        Stmt::Dotimes { var, end, body } => {
            let (next, body) = mutate_scope(rng, policy, genctx, exc_mode, mode, depth + 1, body);
            let end = if mode == MutateMode::All {
                policy::loop_end_at_depth(rng, policy, depth)
            } else {
                *end
            };
            (
                next,
                Stmt::Dotimes {
                    var: var.clone(),
                    end,
                    body,
                },
            )
        }
        Stmt::Let { bindings, body } => {
            // Re-roll binding right-hand sides against the outer
            // context. Locals stay unbound here (as with loop bodies),
            // so fresh right-hand sides conservatively avoid them;
            // kept subtrees re-validate by id, and whole-program
            // simulation re-binds everything in order.
            let mut new_bindings = Vec::with_capacity(bindings.len());
            for (name, rhs) in bindings {
                let rhs = if rng.roll(0.5) {
                    expr::generate_safe(rng, policy, genctx, false, exc_mode)
                } else {
                    rhs.clone()
                };
                new_bindings.push((name.clone(), rhs));
            }
            let (next, body) = mutate_scope(rng, policy, genctx, exc_mode, mode, depth, body);
            (
                next,
                Stmt::Let {
                    bindings: new_bindings,
                    body,
                },
            )
        }
        Stmt::Recur {
            var,
            start,
            end,
            step,
            accs,
            body,
        } => {
            let (next, body) = mutate_scope(rng, policy, genctx, exc_mode, mode, depth + 1, body);
            let (start, end, step) = if mode == MutateMode::All {
                policy::loop_bounds_at_depth(rng, policy, depth)
            } else {
                (*start, *end, *step)
            };
            // Re-roll accumulator inits and updates like let
            // bindings: conservatively blind to locals, validated by
            // whole-program simulation.
            let mut new_accs = Vec::with_capacity(accs.len());
            for (name, init, update) in accs {
                let init = if rng.roll(0.5) {
                    expr::generate_safe(rng, policy, genctx, false, exc_mode)
                } else {
                    init.clone()
                };
                let update = if rng.roll(0.5) {
                    expr::generate_safe(rng, policy, genctx, false, exc_mode)
                } else {
                    update.clone()
                };
                new_accs.push((name.clone(), init, update));
            }
            (
                next,
                Stmt::Recur {
                    var: var.clone(),
                    start,
                    end,
                    step,
                    accs: new_accs,
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
    depth: i64,
    stmts: &[Stmt],
) -> (GenCtx, Vec<Stmt>) {
    let mut ctx = genctx.clone();
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        if rng.roll(0.5) {
            let (next, s2) = mutate_stmt(rng, policy, &ctx, exc_mode, mode, depth, s);
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
    let atoms: BTreeSet<String> = program.outputs.iter().map(|o| o.name.clone()).collect();
    let mut attempt = 0;
    let (stmts, final_ctx) = loop {
        let (_, mut stmts) = mutate_scope(rng, policy, &genctx, exc_mode, mode, 0, &program.stmts);
        let mut trial = genctx.clone();
        let mut ok = false;
        for _ in 0..6 {
            let mut ctx = genctx.clone();
            match stmt::simulate_bounded_whole(&mut ctx, &stmts) {
                Ok(()) => {
                    // Re-rolled right-hand sides must not push a single
                    // statement past one function's size caps.
                    if stmts.iter().all(|s| stmt::stmt_fits(s, &atoms)) {
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
                .map(|v| v.as_long())
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
        mutation_seed: program.mutation_seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::Input;
    use crate::expr::Expr;
    use crate::types::{CType, Ir, Value};

    fn sample() -> Program {
        // Mirror seed_context order (inputs then outputs) so the
        // recorded id matches the mutation context's numbering.
        let mut ctx = GenCtx::new();
        ctx.add_input("in-0", CType::Long, Value::Long(1));
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
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
                target_id: oid,
                expr: Expr::Const(Ir::long(2)),
            }],
            expected_hash: 0,
            check_algo: crate::cli::CheckAlgo::Hash,
            mutation_seed: None,
        }
    }

    fn nested_loop_sample() -> Program {
        // Outer (depth 0) + inner (depth 1) loop over one output.
        let mut ctx = GenCtx::new();
        ctx.add_input("in-0", CType::Long, Value::Long(1));
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let body = Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: Expr::BinOp {
                op: crate::types::BinOp::Add,
                left: Box::new(Expr::Var {
                    id: oid,
                    name: "out-0".to_string(),
                    ir: Ir::long(0),
                }),
                right: Box::new(Expr::Const(Ir::long(1))),
                ir: Ir::long(1),
            },
        };
        let inner = Stmt::Loop {
            var: "i_in".to_string(),
            start: 0,
            end: 64,
            step: 1,
            body: vec![body],
        };
        let outer = Stmt::Loop {
            var: "i_out".to_string(),
            start: 0,
            end: 4,
            step: 1,
            body: vec![inner],
        };
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
                final_value: 256,
            }],
            stmts: vec![outer],
            expected_hash: 0,
            check_algo: crate::cli::CheckAlgo::Hash,
            mutation_seed: None,
        }
    }

    fn inner_end(program: &Program) -> i64 {
        match &program.stmts[0] {
            Stmt::Loop { body, .. } => match &body[0] {
                Stmt::Loop { end, .. } => *end,
                other => panic!("expected inner loop, got {other:?}"),
            },
            other => panic!("expected outer loop, got {other:?}"),
        }
    }

    #[test]
    fn mutation_scales_re_rolled_bounds_by_depth() {
        // With a constant 64 policy, unscaled re-rolls keep the inner
        // (depth-1) end at 64; depth-scaled ones halve it to 32. Some
        // seed must visit the inner loop, and none may produce a
        // third value.
        let policy = Policy {
            iter_end_min: 64,
            iter_end_max: 64,
            ..Policy::default()
        };
        let mut saw_scaled = false;
        for seed in 1..20u64 {
            let mut rng = Rng::new(seed);
            let m = mutate_program(
                &nested_loop_sample(),
                &mut rng,
                &policy,
                Density::None,
                MutateMode::All,
            );
            let end = inner_end(&m);
            assert!(
                end == 64 || end == 32,
                "seed {seed}: inner end {end}, expected 32 or 64"
            );
            saw_scaled |= end == 32;
        }
        assert!(saw_scaled, "no seed applied depth-scaled bounds");
    }

    #[test]
    fn mutation_keeps_let_program_coherent() {
        // A let block survives mutation with values re-simulated:
        // binding right-hand sides re-roll, locals re-bind by id.
        let mut ctx = GenCtx::new();
        ctx.add_input("in-0", CType::Long, Value::Long(1));
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let lid = ctx.add_local("l", CType::Long, Value::Long(2));
        let program = Program {
            seed: 9,
            inputs: vec![Input {
                name: "in-0".to_string(),
                typ: CType::Long,
                value: Value::Long(1),
            }],
            outputs: vec![Output {
                name: "out-0".to_string(),
                init: 0,
                final_value: 2,
            }],
            stmts: vec![Stmt::Let {
                bindings: vec![("l".to_string(), Expr::Const(Ir::long(2)))],
                body: vec![Stmt::Assign {
                    target: "out-0".to_string(),
                    target_id: oid,
                    expr: Expr::Var {
                        id: lid,
                        name: "l".to_string(),
                        ir: Ir::long(2),
                    },
                }],
            }],
            expected_hash: 0,
            check_algo: crate::cli::CheckAlgo::Hash,
            mutation_seed: None,
        };
        let mut rng = Rng::new(3);
        let policy = Policy::default();
        let m = mutate_program(&program, &mut rng, &policy, Density::None, MutateMode::All);
        assert_eq!(m.stmts.len(), 1);
        let pairs = vec![("out-0".to_string(), m.outputs[0].final_value)];
        assert_eq!(m.expected_hash, hash::checksum(&pairs));
    }

    #[test]
    fn mutation_keeps_recur_program_coherent() {
        // A recur loop survives mutation with values re-simulated:
        // bounds, inits and updates re-roll, locals re-bind by id.
        let mut ctx = GenCtx::new();
        ctx.add_input("in-0", CType::Long, Value::Long(1));
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        // Sim binds the counter then the accumulator after the two
        // pre-registered vars, so the accumulator reads id 3.
        let program = Program {
            seed: 9,
            inputs: vec![Input {
                name: "in-0".to_string(),
                typ: CType::Long,
                value: Value::Long(1),
            }],
            outputs: vec![Output {
                name: "out-0".to_string(),
                init: 0,
                final_value: 7,
            }],
            stmts: vec![Stmt::Recur {
                var: "r".to_string(),
                start: 0,
                end: 2,
                step: 1,
                accs: vec![(
                    "ra".to_string(),
                    Expr::Const(Ir::long(5)),
                    Expr::Const(Ir::long(7)),
                )],
                body: vec![Stmt::Assign {
                    target: "out-0".to_string(),
                    target_id: oid,
                    expr: Expr::Var {
                        id: crate::context::VarId(3),
                        name: "ra".to_string(),
                        ir: Ir::long(5),
                    },
                }],
            }],
            expected_hash: 0,
            check_algo: crate::cli::CheckAlgo::Hash,
            mutation_seed: None,
        };
        let mut rng = Rng::new(3);
        let policy = Policy::default();
        let m = mutate_program(&program, &mut rng, &policy, Density::None, MutateMode::All);
        assert_eq!(m.stmts.len(), 1);
        let pairs = vec![("out-0".to_string(), m.outputs[0].final_value)];
        assert_eq!(m.expected_hash, hash::checksum(&pairs));
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
