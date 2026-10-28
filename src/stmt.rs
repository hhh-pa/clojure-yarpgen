// Licensed under the Apache License, Version 2.0.
//! Statement skeleton generation and population. Ports YARPGen's
//! stmt.cpp `ScopeStmt::generateStructure` + populate, with loop bodies
//! simulated at generation time so every variable's final value is
//! known exactly (enables a precomputed expected hash).

use crate::cli::Density;
use crate::context::GenCtx;
use crate::expr::{self, Expr};
use crate::policy::{self, AssignKind, Policy, StructKind};
use crate::rng::Rng;
use crate::sexp::Sexp;
use crate::types::{self, BinOp, CType, Ir, Value};
use std::collections::BTreeSet;

/// Statement AST.
#[derive(Clone, Debug)]
pub enum Stmt {
    Assign {
        target: String,
        expr: Expr,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
    },
    Loop {
        var: String,
        start: i64,
        end: i64,
        step: i64,
        body: Vec<Stmt>,
    },
}

/// Evaluate an expression against current context bindings.
fn live_value(genctx: &GenCtx, node: &Expr) -> Ir {
    match node {
        Expr::Const(ir) => ir.clone(),
        Expr::Var { name, .. } => match genctx.lookup(name) {
            Some(v) => Ir::new(v.typ, v.value, crate::types::ExKind::NoEx),
            None => Ir::new(CType::Long, Value::Long(0), crate::types::ExKind::Uninit),
        },
        Expr::UnOp { op, arg, .. } => types::eval_unop(*op, &live_value(genctx, arg)),
        Expr::BinOp {
            op, left, right, ..
        } => types::eval_binop(*op, &live_value(genctx, left), &live_value(genctx, right)),
        Expr::Nth { coll, idx, .. } => {
            let c = live_value(genctx, coll);
            let i = live_value(genctx, idx);
            match (&c.value, i.value.as_long()) {
                (Value::Vec(items), k) if i.is_ok() && k >= 0 && (k as usize) < items.len() => {
                    Ir::long(items[k as usize])
                }
                _ => Ir::new(CType::Long, Value::Long(0), crate::types::ExKind::Oob),
            }
        }
        Expr::NthDefault {
            coll, idx, default, ..
        } => {
            let c = live_value(genctx, coll);
            let i = live_value(genctx, idx);
            // The index subtree always executes, so its faults propagate
            // even though a default exists.
            if !i.is_ok() {
                return Ir::new(CType::Long, Value::Long(0), i.ex);
            }
            let k = i.value.as_long();
            // Clojure narrows the index with toIntExact (checked) in
            // BOTH nth arities: out-of-int-range always throws.
            if k < i64::from(i32::MIN) || k > i64::from(i32::MAX) {
                return Ir::new(CType::Long, Value::Long(0), crate::types::ExKind::Oob);
            }
            match &c.value {
                Value::Vec(items) if k >= 0 && (k as usize) < items.len() => {
                    Ir::long(items[k as usize])
                }
                _ => live_value(genctx, default),
            }
        }
        Expr::IdxMod { idx, len, .. } => {
            let i = live_value(genctx, idx);
            if i.is_ok() {
                Ir::long(types::smod_floor(i.value.as_long(), *len))
            } else {
                Ir::new(CType::Long, Value::Long(0), i.ex)
            }
        }
    }
}

/// Evaluation budget for one loop simulation. Deeply nested loops
/// with wide bounds would otherwise multiply simulation cost
/// exponentially; when the budget is exhausted the caller shrinks the
/// bounds (or falls back to straight-line code) and retries, so the
/// emitted program always matches the simulated values exactly.
pub const SIM_BUDGET: u64 = 20_000;
/// Budget for validating a whole program at once (mutation, tests).
/// Larger because it covers every top-level statement in one pass.
pub const SIM_BUDGET_WHOLE: u64 = 500_000;

/// Bounded-simulation failure: budget exhaustion (shrink and retry)
/// or a value fault such as loop-carried overflow (regenerate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimErr {
    Budget,
    Fault,
}

/// Thread generation-time values through one statement, charging
/// expression evaluations against `budget`. In strict mode any
/// evaluation whose ExKind is not `NoEx` (overflow, division by zero,
/// out-of-bounds) fails with [`SimErr::Fault`]; lenient mode applies
/// values unconditionally for straight-line code.
fn try_simulate_stmt(
    genctx: &mut GenCtx,
    stmt: &Stmt,
    budget: &mut u64,
    strict: bool,
) -> Result<(), SimErr> {
    match stmt {
        Stmt::Assign { target, expr } => {
            *budget = budget.checked_sub(1).ok_or(SimErr::Budget)?;
            let v = live_value(genctx, expr);
            if strict && !v.is_ok() {
                return Err(SimErr::Fault);
            }
            genctx.update_value(target, v.value);
            genctx.stmt_count += 1;
            Ok(())
        }
        Stmt::If { cond, then, els } => {
            *budget = budget.checked_sub(1).ok_or(SimErr::Budget)?;
            let c = live_value(genctx, cond);
            if strict && !c.is_ok() {
                return Err(SimErr::Fault);
            }
            let branch = if types::coerce_bool(&c) { then } else { els };
            try_simulate_stmts(genctx, branch, budget, strict)
        }
        Stmt::Loop {
            var,
            start,
            end,
            step,
            body,
        } => {
            let mut i = *start;
            while i < *end {
                genctx.add_local(var, CType::Long, Value::Long(i));
                try_simulate_stmts(genctx, body, budget, strict)?;
                i += *step;
            }
            Ok(())
        }
    }
}

fn try_simulate_stmts(
    genctx: &mut GenCtx,
    stmts: &[Stmt],
    budget: &mut u64,
    strict: bool,
) -> Result<(), SimErr> {
    for s in stmts {
        try_simulate_stmt(genctx, s, budget, strict)?;
    }
    Ok(())
}

/// Thread generation-time values through a statement sequence.
/// Lenient and unbounded: used for straight-line code (exact by
/// construction) where dead branches may intentionally hold faults.
pub fn simulate_stmts(genctx: &mut GenCtx, stmts: &[Stmt]) {
    let mut budget = u64::MAX;
    let _ = try_simulate_stmts(genctx, stmts, &mut budget, false);
}

/// Strict bounded simulation: `Ok` guarantees fault-free values within
/// budget. Used to validate loops, whose bodies execute repeatedly
/// with evolving values.
pub fn simulate_bounded(genctx: &mut GenCtx, stmts: &[Stmt]) -> Result<(), SimErr> {
    let mut budget = SIM_BUDGET;
    try_simulate_stmts(genctx, stmts, &mut budget, true)
}

/// Strict whole-program simulation with the larger program budget.
/// Used where an entire scope is validated in one pass.
pub fn simulate_bounded_whole(genctx: &mut GenCtx, stmts: &[Stmt]) -> Result<(), SimErr> {
    let mut budget = SIM_BUDGET_WHOLE;
    try_simulate_stmts(genctx, stmts, &mut budget, true)
}

/// Halve every loop bound in a statement tree (minimum trip count 2).
/// Returns false when nothing was shrinkable.
pub fn shrink_loop_bounds(stmt: &mut Stmt) -> bool {
    match stmt {
        Stmt::Loop { end, body, .. } => {
            let mut shrunk = false;
            if *end > 2 {
                *end = (*end / 2).max(2);
                shrunk = true;
            }
            for b in body.iter_mut() {
                shrunk |= shrink_loop_bounds(b);
            }
            shrunk
        }
        Stmt::If { then, els, .. } => {
            let mut shrunk = false;
            for b in then.iter_mut().chain(els.iter_mut()) {
                shrunk |= shrink_loop_bounds(b);
            }
            shrunk
        }
        Stmt::Assign { .. } => false,
    }
}

/// Outcome of validating a freshly populated loop.
enum LoopSim {
    /// Fault-free within budget; carries the post-loop context.
    Ok(GenCtx),
    /// Loop-carried fault (or irreducible cost): repopulate the body.
    Fault,
}

/// Simulate a freshly populated loop, shrinking its bounds on budget
/// exhaustion. A value fault means the body misbehaves under evolved
/// values, so the caller must regenerate the body instead.
fn simulate_loop_or_shrink(pre_ctx: &GenCtx, stmt: &mut Stmt, creation_count: i64) -> LoopSim {
    for _ in 0..6 {
        let mut ctx = pre_ctx.clone();
        match simulate_bounded(&mut ctx, std::slice::from_ref(stmt)) {
            Ok(()) => {
                ctx.stmt_count = creation_count;
                ctx.loop_depth = pre_ctx.loop_depth;
                ctx.locals = pre_ctx.locals.clone();
                return LoopSim::Ok(ctx);
            }
            Err(SimErr::Budget) => {
                if !shrink_loop_bounds(stmt) {
                    break;
                }
            }
            Err(SimErr::Fault) => return LoopSim::Fault,
        }
    }
    LoopSim::Fault
}

/// Assignment or reduction (ASSIGN 70 / REDUCTION 30) to a writable var.
fn gen_assign(rng: &mut Rng, policy: &Policy, genctx: &GenCtx, exc_mode: Density) -> Option<Stmt> {
    let writable = genctx.writable_vars();
    if writable.is_empty() {
        return None;
    }
    let entry = rng.pick(&writable);
    let target = entry.name.clone();
    let kind: AssignKind = rng.pick_weighted(&policy.assign_distr);
    if kind == AssignKind::Reduction && entry.typ != CType::Vec {
        let op = *rng.pick(&[
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
            BinOp::Min,
            BinOp::Max,
        ]);
        let rhs = expr::generate_safe(rng, policy, genctx, false, exc_mode);
        let self_expr = Expr::Var {
            name: target.clone(),
            ir: Ir::new(entry.typ, entry.value.clone(), crate::types::ExKind::NoEx),
        };
        let ir = types::eval_binop(op, self_expr.ir(), rhs.ir());
        if ir.is_ok() {
            return Some(Stmt::Assign {
                target,
                expr: Expr::BinOp {
                    op,
                    left: Box::new(self_expr),
                    right: Box::new(rhs),
                    ir,
                },
            });
        }
        return Some(Stmt::Assign { target, expr: rhs });
    }
    Some(Stmt::Assign {
        target,
        expr: expr::generate_safe(rng, policy, genctx, false, exc_mode),
    })
}

fn gen_if(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
) -> (GenCtx, Vec<Stmt>) {
    let raw_cond = expr::generate_safe(rng, policy, genctx, true, exc_mode);
    // Scalar conditions are constant-true in Clojure (0 is truthy);
    // wrap them so branches genuinely vary.
    let cond = if raw_cond.ir().typ == CType::Bool {
        raw_cond
    } else {
        let zero = Expr::Const(Ir::long(0));
        let ir = types::eval_binop(BinOp::Ne, raw_cond.ir(), zero.ir());
        Expr::BinOp {
            op: BinOp::Ne,
            left: Box::new(raw_cond),
            right: Box::new(zero),
            ir,
        }
    };
    let cond_val = types::coerce_bool(cond.ir());
    let mut then_ctx = genctx.with_taken(cond_val);
    then_ctx.if_depth += 1;
    let mut else_ctx = genctx.with_taken(!cond_val);
    else_ctx.if_depth += 1;
    let has_else = rng.roll(policy.else_branch_prob);
    let (then_ctx, then) = populate_scope(rng, policy, then_ctx, exc_mode, depth + 1);
    let (else_ctx, els) = if has_else {
        populate_scope(rng, policy, else_ctx, exc_mode, depth + 1)
    } else {
        (else_ctx, vec![])
    };
    let next = if cond_val { then_ctx } else { else_ctx };
    (next, vec![Stmt::If { cond, then, els }])
}

/// Populate a straight-line scope (assignments only). Used when a loop
/// cannot be simulated within budget even after shrinking, and as the
/// size-guaranteed fallback for oversized statements.
pub fn populate_stub_scope(
    rng: &mut Rng,
    policy: &Policy,
    genctx: GenCtx,
    exc_mode: Density,
    depth: i64,
) -> (GenCtx, Vec<Stmt>) {
    let mut stub_policy = policy.clone();
    stub_policy.struct_distr = vec![(StructKind::Stub, 1)];
    populate_scope(rng, &stub_policy, genctx, exc_mode, depth)
}

/// A single countable loop (`doseq` over `range`). The body is populated
/// once with the loop variable bound, then simulated across all
/// iterations so downstream values stay exact.
fn gen_loop(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
    tag: &str,
) -> (GenCtx, Vec<Stmt>) {
    for _ in 0..4 {
        let (start, end, step) = policy::loop_bounds(rng, policy);
        let var = format!("i_{tag}_{}", genctx.loop_depth);
        let mut body_ctx = genctx.clone();
        body_ctx.add_local(&var, CType::Long, Value::Long(start));
        body_ctx.loop_depth += 1;
        let (after_body, body) = populate_scope(rng, policy, body_ctx, exc_mode, depth + 1);
        let mut stmt = Stmt::Loop {
            var,
            start,
            end,
            step,
            body,
        };
        match simulate_loop_or_shrink(genctx, &mut stmt, after_body.stmt_count) {
            LoopSim::Ok(final_ctx) => return (final_ctx, vec![stmt]),
            LoopSim::Fault => continue,
        }
    }
    populate_stub_scope(rng, policy, genctx.clone(), exc_mode, depth)
}

/// A true nest of loops: each level's body is the next loop, and the
/// innermost body is a populated scope.
fn gen_loop_nest(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
    tag: &str,
) -> (GenCtx, Vec<Stmt>) {
    let n = rng
        .gen_range(
            policy.loop_nest_depth_min,
            policy.loop_nest_depth_max.min(policy.loop_nest_depth_limit),
        )
        .max(1);

    struct NestBuilder<'a> {
        rng: &'a mut Rng,
        policy: &'a Policy,
        exc_mode: Density,
        tag: &'a str,
        levels: i64,
    }

    impl NestBuilder<'_> {
        fn build(&mut self, genctx: &GenCtx, depth: i64, level: i64) -> (GenCtx, Stmt) {
            let (start, end, step) = policy::loop_bounds(self.rng, self.policy);
            let var = format!("i_{}_nest{level}", self.tag);
            let mut body_ctx = genctx.clone();
            body_ctx.add_local(&var, CType::Long, Value::Long(start));
            body_ctx.loop_depth += 1;
            if level == self.levels - 1 {
                let (after, body) =
                    populate_scope(self.rng, self.policy, body_ctx, self.exc_mode, depth + 1);
                (
                    after,
                    Stmt::Loop {
                        var,
                        start,
                        end,
                        step,
                        body,
                    },
                )
            } else {
                let (after, inner) = self.build(&body_ctx, depth + 1, level + 1);
                (
                    after,
                    Stmt::Loop {
                        var,
                        start,
                        end,
                        step,
                        body: vec![inner],
                    },
                )
            }
        }
    }

    for _ in 0..4 {
        let mut builder = NestBuilder {
            rng: &mut *rng,
            policy,
            exc_mode,
            tag,
            levels: n,
        };
        let (c2, stmt) = builder.build(genctx, depth, 0);
        let mut stmt = stmt;
        match simulate_loop_or_shrink(genctx, &mut stmt, c2.stmt_count) {
            LoopSim::Ok(final_ctx) => return (final_ctx, vec![stmt]),
            LoopSim::Fault => continue,
        }
    }
    populate_stub_scope(rng, policy, genctx.clone(), exc_mode, depth)
}

fn assign_or(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
) -> (GenCtx, Vec<Stmt>) {
    let stmt = gen_assign(rng, policy, genctx, exc_mode).unwrap_or(Stmt::If {
        cond: Expr::Const(Ir::boolean(true)),
        then: vec![],
        els: vec![],
    });
    let mut next = genctx.clone();
    simulate_stmts(&mut next, std::slice::from_ref(&stmt));
    (next, vec![stmt])
}

fn populate_one(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
    tag: &str,
) -> (GenCtx, Vec<Stmt>) {
    match rng.pick_weighted(&policy.struct_distr) {
        StructKind::LoopSeq => {
            if genctx.loop_depth < policy.loop_nest_depth_limit {
                gen_loop(rng, policy, genctx, exc_mode, depth, tag)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::LoopNest => {
            if genctx.loop_depth < policy.loop_nest_depth_limit {
                gen_loop_nest(rng, policy, genctx, exc_mode, depth, tag)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::IfElse => {
            if genctx.if_depth < policy.if_else_limit {
                gen_if(rng, policy, genctx, exc_mode, depth)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::Stub => assign_or(rng, policy, genctx, exc_mode),
    }
}

/// Populate one scope with a policy-sized statement sequence.
pub fn populate_scope(
    rng: &mut Rng,
    policy: &Policy,
    genctx: GenCtx,
    exc_mode: Density,
    depth: i64,
) -> (GenCtx, Vec<Stmt>) {
    let n = policy::scope_size(rng, policy);
    let mut ctx = genctx;
    let mut stmts = Vec::new();
    let mut k = 0;
    while k < n && ctx.stmt_count < policy.stmt_num_limit {
        let tag = format!("{depth}_{k}_{}", ctx.stmt_count);
        let (next, s) = populate_one(rng, policy, &ctx, exc_mode, depth, &tag);
        ctx = next;
        stmts.extend(s);
        k += 1;
    }
    (ctx, stmts)
}

/// Lower a statement to a Clojure s-expression. Mutable outputs are
/// atoms; loop variables and inputs are plain values.
pub fn to_sexp(stmt: &Stmt, atoms: &BTreeSet<String>) -> Sexp {
    match stmt {
        Stmt::Assign { target, expr } => Sexp::list(vec![
            Sexp::sym("reset!"),
            Sexp::sym(target),
            expr::to_sexp(expr, atoms),
        ]),
        Stmt::If { cond, then, els } => {
            let c = expr::to_sexp(cond, atoms);
            let mut t = vec![Sexp::sym("do")];
            t.extend(then.iter().map(|s| to_sexp(s, atoms)));
            if els.is_empty() {
                Sexp::list(vec![Sexp::sym("when"), c, Sexp::list(t)])
            } else {
                let mut e = vec![Sexp::sym("do")];
                e.extend(els.iter().map(|s| to_sexp(s, atoms)));
                Sexp::list(vec![Sexp::sym("if"), c, Sexp::list(t), Sexp::list(e)])
            }
        }
        Stmt::Loop {
            var,
            start,
            end,
            step,
            body,
        } => {
            let mut items = vec![
                Sexp::sym("doseq"),
                Sexp::vec(vec![
                    Sexp::sym(var),
                    Sexp::list(vec![
                        Sexp::sym("range"),
                        Sexp::int(*start),
                        Sexp::int(*end),
                        Sexp::int(*step),
                    ]),
                ]),
            ];
            items.extend(body.iter().map(|s| to_sexp(s, atoms)));
            Sexp::list(items)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_ctx() -> GenCtx {
        let mut c = GenCtx::new();
        c.add_input("in-0", CType::Long, Value::Long(3));
        c.add_output("out-0", CType::Long, Value::Long(0));
        c
    }

    #[test]
    fn populated_scopes_simulate_cleanly() {
        for seed in 1..20u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let (ctx, stmts) = populate_scope(&mut rng, &policy, seeded_ctx(), Density::None, 0);
            assert!(!stmts.is_empty());
            assert!(ctx.stmt_count > 0);
            let mut sim = seeded_ctx();
            simulate_stmts(&mut sim, &stmts);
            assert_eq!(
                sim.lookup("out-0").unwrap().long_value(),
                ctx.lookup("out-0").unwrap().long_value()
            );
        }
    }

    fn collect_targets(stmts: &[Stmt], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::Assign { target, .. } => out.push(target.clone()),
                Stmt::If { then, els, .. } => {
                    collect_targets(then, out);
                    collect_targets(els, out);
                }
                Stmt::Loop { body, .. } => collect_targets(body, out),
            }
        }
    }

    #[test]
    fn assign_targets_are_outputs_only() {
        // Loop variables are doseq-bound plain values; emitting
        // (reset! loop-var ...) would throw at runtime.
        for seed in 1..30u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let (ctx, stmts) = populate_scope(&mut rng, &policy, seeded_ctx(), Density::None, 0);
            let mut targets = Vec::new();
            collect_targets(&stmts, &mut targets);
            assert!(!targets.is_empty(), "seed {seed}: no assignments");
            for t in &targets {
                assert!(
                    ctx.ext_out.contains_key(t),
                    "seed {seed}: assignment to non-output {t}"
                );
            }
        }
    }

    #[test]
    fn populated_scopes_are_strict_clean() {
        // Strict re-simulation must succeed: no loop-carried overflow,
        // division by zero, or out-of-bounds on any iteration.
        for seed in 1..30u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let (_, stmts) = populate_scope(&mut rng, &policy, seeded_ctx(), Density::None, 0);
            let mut sim = seeded_ctx();
            assert_eq!(
                simulate_bounded_whole(&mut sim, &stmts),
                Ok(()),
                "seed {seed}: strict simulation fault"
            );
        }
    }

    #[test]
    fn strict_simulation_catches_growing_reduction() {
        // (* out-0 out-0) from 2 overflows within 10 iterations; the
        // strict checker must report Fault, not silently wrap.
        let mut ctx = seeded_ctx();
        ctx.update_value("out-0", Value::Long(2));
        let square = Expr::BinOp {
            op: BinOp::Mul,
            left: Box::new(Expr::Var {
                name: "out-0".to_string(),
                ir: Ir::long(2),
            }),
            right: Box::new(Expr::Var {
                name: "out-0".to_string(),
                ir: Ir::long(2),
            }),
            ir: Ir::long(4),
        };
        let prog = vec![Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 10,
            step: 1,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                expr: square,
            }],
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn strict_simulation_catches_division_by_zero() {
        let mut ctx = seeded_ctx();
        let div = Expr::BinOp {
            op: BinOp::Div,
            left: Box::new(Expr::Const(Ir::long(1))),
            right: Box::new(Expr::Var {
                name: "out-0".to_string(),
                ir: Ir::long(0),
            }),
            ir: Ir::long(0),
        };
        let prog = vec![Stmt::Assign {
            target: "out-0".to_string(),
            expr: div,
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn strict_simulation_catches_huge_nth_index() {
        // Clojure narrows nth indices via toIntExact (checked) even
        // when a default is present, so an out-of-int-range index must
        // fault in strict simulation.
        let mut ctx = GenCtx::new();
        ctx.add_input("v", CType::Vec, Value::Vec(vec![1, 2, 3]));
        ctx.add_output("out-0", CType::Long, Value::Long(0));
        let nth = Expr::NthDefault {
            coll: Box::new(Expr::Var {
                name: "v".to_string(),
                ir: Ir::new(
                    CType::Vec,
                    Value::Vec(vec![1, 2, 3]),
                    crate::types::ExKind::NoEx,
                ),
            }),
            idx: Box::new(Expr::Const(Ir::long(1 << 40))),
            default: Box::new(Expr::Const(Ir::long(9))),
            ir: Ir::long(9),
        };
        let prog = vec![Stmt::Assign {
            target: "out-0".to_string(),
            expr: nth,
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn loop_emission_shape() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let s = Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 10,
            step: 1,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                expr: Expr::Const(Ir::long(1)),
            }],
        };
        assert_eq!(
            to_sexp(&s, &atoms).flat(),
            "(doseq [i (range 0 10 1)] (reset! out-0 1))"
        );
    }
}
