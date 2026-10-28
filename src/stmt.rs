// Licensed under the Apache License, Version 2.0.
//! Statement skeleton generation and population. Ports YARPGen's
//! stmt.cpp `ScopeStmt::generateStructure` + populate, with loop bodies
//! simulated at generation time so every variable's final value is
//! known exactly (enables a precomputed expected hash).

use crate::cli::Density;
use crate::context::{GenCtx, VarId};
use crate::expr::{self, Expr};
use crate::policy::{self, AssignKind, Policy, StructKind};
use crate::rng::Rng;
use crate::sexp::Sexp;
use crate::types::{self, BinOp, CType, Ir, Value};
use std::collections::BTreeSet;

/// True when a subscript index lands inside a collection on every
/// platform: the bounds check runs in `i64` so a huge index can
/// never truncate into range through `as usize` (32-bit).
fn k_in_range(k: i64, len: usize) -> bool {
    k >= 0 && k < len as i64
}

/// A destructuring pattern: sequential positions are an implied
/// 0-based prefix, while map patterns carry one explicit key per
/// name (`{local :key}` keeps local names unique — `:keys` would
/// force same-name bindings and collide across scopes).
#[derive(Clone, Debug, PartialEq)]
pub enum DestrPattern {
    Seq,
    Map(Vec<String>),
}

/// A destructuring source: a small literal, or an immutable
/// vector/map/record input by name (never an atom: inputs are plain
/// defs).
#[derive(Clone, Debug, PartialEq)]
pub enum DestrSource {
    VecLit(Vec<i64>),
    MapLit(Vec<(String, i64)>),
    Var(String),
}

/// Statement AST.
#[derive(Clone, Debug)]
pub enum Stmt {
    Assign {
        target: String,
        target_id: VarId,
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
    /// `(dotimes [var end] ...)`: identical trips to a unit-step loop
    /// from 0, but a different macroexpansion path through the compiler.
    Dotimes {
        var: String,
        end: i64,
        body: Vec<Stmt>,
    },
    /// Sequential `(let [name rhs ...] ...)`: later bindings see
    /// earlier ones (Clojure `let*` semantics). Bindings are plain
    /// long locals, truncated when the block exits like loop
    /// variables. Deliberately unhinted pending a binding-hint probe.
    Let {
        bindings: Vec<(String, Expr)>,
        body: Vec<Stmt>,
    },
    /// A counted `(loop [i start, acc init ...] ...)` with a tail
    /// `(recur ...)`: same trip shape as [`Stmt::Loop`] but through
    /// `LoopExpr` codegen instead of `doseq` expansion. Each
    /// accumulator carries an init expression (evaluated once at
    /// entry) and an update expression (evaluated per iteration
    /// after the body); all recur arguments evaluate under
    /// pre-rebind values and rebind atomically, exactly like the
    /// runtime. Locals are unhinted, as with `let`.
    Recur {
        var: String,
        start: i64,
        end: i64,
        step: i64,
        accs: Vec<(String, Expr, Expr)>,
        body: Vec<Stmt>,
    },
    /// A `(let [pattern source] ...)` with a sequential or map
    /// destructuring pattern over an immutable source. Bound values
    /// are constants (positions of a known vector, present keys of a
    /// known map or record), so simulation re-binds them without
    /// evaluation. Patterns stay flat and total: over-long patterns
    /// and absent keys would bind nil, which the model cannot
    /// represent.
    Destructure {
        pattern: DestrPattern,
        names: Vec<String>,
        source: DestrSource,
        values: Vec<i64>,
        body: Vec<Stmt>,
    },
}

/// Evaluate an expression against current context bindings.
fn live_value(genctx: &GenCtx, node: &Expr) -> Ir {
    match node {
        Expr::Const(ir) => ir.clone(),
        Expr::Var { id, .. } => Ir::new(
            genctx.var_type(*id),
            genctx.var_value(*id).clone(),
            crate::types::ExKind::NoEx,
        ),
        Expr::UnOp { op, arg, .. } => types::eval_unop(*op, &live_value(genctx, arg)),
        Expr::BinOp {
            op, left, right, ..
        } => types::eval_binop(*op, &live_value(genctx, left), &live_value(genctx, right)),
        Expr::Nth { coll, idx, .. } => {
            let c = live_value(genctx, coll);
            let i = live_value(genctx, idx);
            // Compare in i64 before casting: `k as usize` truncates
            // on 32-bit targets, turning out-of-bounds into hits.
            match &c.value {
                Value::Vec(items) if i.is_ok() && k_in_range(i.value.as_long(), items.len()) => {
                    Ir::long(items[i.value.as_long() as usize])
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
                Value::Vec(items) if k_in_range(k, items.len()) => Ir::long(items[k as usize]),
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
        Expr::Assoc { coll, key, val, .. } => {
            let c = live_value(genctx, coll);
            if !c.is_ok() {
                return Ir::new(CType::Map, Value::Map(Default::default()), c.ex);
            }
            let v = live_value(genctx, val);
            if !v.is_ok() {
                return Ir::new(CType::Map, Value::Map(Default::default()), v.ex);
            }
            match c.value {
                Value::Map(mut items) => {
                    items.insert(key.clone(), v.value.as_long());
                    Ir::new(CType::Map, Value::Map(items), crate::types::ExKind::NoEx)
                }
                // Assoc of an existing field stays a record (probed);
                // generation never assocs new keys onto records.
                Value::Record(mut items) => {
                    items.insert(key.clone(), v.value.as_long());
                    Ir::new(
                        CType::Record,
                        Value::Record(items),
                        crate::types::ExKind::NoEx,
                    )
                }
                _ => Ir::new(
                    CType::Map,
                    Value::Map(Default::default()),
                    crate::types::ExKind::Oob,
                ),
            }
        }
        Expr::Record { args, .. } => {
            // Args run in REC_FIELDS order; both constructors agree
            // (probed), so the form needs no modeling.
            let mut items = std::collections::BTreeMap::new();
            for (f, a) in types::REC_FIELDS.iter().zip(args.iter()) {
                let v = live_value(genctx, a);
                if !v.is_ok() {
                    return Ir::new(CType::Record, Value::Record(Default::default()), v.ex);
                }
                items.insert(f.to_string(), v.value.as_long());
            }
            Ir::new(
                CType::Record,
                Value::Record(items),
                crate::types::ExKind::NoEx,
            )
        }
        Expr::Get {
            coll, key, default, ..
        } => {
            let c = live_value(genctx, coll);
            if !c.is_ok() {
                return Ir::new(CType::Long, Value::Long(0), c.ex);
            }
            match &c.value {
                // Records read exactly like maps; generation only
                // asks present fields, so no default form exists.
                Value::Map(items) | Value::Record(items) => match items.get(key) {
                    Some(n) => Ir::long(*n),
                    // The default always evaluates, so its faults
                    // propagate exactly like NthDefault's.
                    None => match default {
                        Some(d) => live_value(genctx, d),
                        // Unreachable by construction (absent keys
                        // always carry a default), but strict
                        // simulation must fault rather than invent a
                        // nil the model cannot represent.
                        None => Ir::new(CType::Long, Value::Long(0), crate::types::ExKind::Oob),
                    },
                },
                _ => Ir::new(CType::Long, Value::Long(0), crate::types::ExKind::Oob),
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
        Stmt::Assign {
            target_id, expr, ..
        } => {
            *budget = budget.checked_sub(1).ok_or(SimErr::Budget)?;
            let v = live_value(genctx, expr);
            if strict && !v.is_ok() {
                return Err(SimErr::Fault);
            }
            genctx.update_value_by_id(*target_id, v.value);
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
            // Truncation mirrors population exactly: without it, stale
            // loop variables would shift later ids out of alignment.
            let snap = genctx.snapshot_len();
            let mut i = *start;
            while i < *end {
                genctx.add_local(var, CType::Long, Value::Long(i));
                try_simulate_stmts(genctx, body, budget, strict)?;
                i += *step;
            }
            genctx.truncate_to(snap);
            Ok(())
        }
        Stmt::Dotimes { var, end, body } => {
            let snap = genctx.snapshot_len();
            let mut i = 0;
            while i < *end {
                genctx.add_local(var, CType::Long, Value::Long(i));
                try_simulate_stmts(genctx, body, budget, strict)?;
                i += 1;
            }
            genctx.truncate_to(snap);
            Ok(())
        }
        Stmt::Let { bindings, body } => {
            let snap = genctx.snapshot_len();
            for (name, rhs) in bindings {
                *budget = budget.checked_sub(1).ok_or(SimErr::Budget)?;
                let v = live_value(genctx, rhs);
                if strict && !v.is_ok() {
                    return Err(SimErr::Fault);
                }
                genctx.add_local(name, CType::Long, v.value);
            }
            try_simulate_stmts(genctx, body, budget, strict)?;
            genctx.truncate_to(snap);
            Ok(())
        }
        Stmt::Recur {
            var,
            start,
            end,
            step,
            accs,
            body,
        } => {
            let snap = genctx.snapshot_len();
            genctx.add_local(var, CType::Long, Value::Long(*start));
            for (name, init, _) in accs {
                *budget = budget.checked_sub(1).ok_or(SimErr::Budget)?;
                let v = live_value(genctx, init);
                if strict && !v.is_ok() {
                    return Err(SimErr::Fault);
                }
                genctx.add_local(name, CType::Long, v.value);
            }
            let mut i = *start;
            while i < *end {
                try_simulate_stmts(genctx, body, budget, strict)?;
                // All recur arguments evaluate under pre-rebind
                // values (the body's outputs included: emission
                // evaluates them after the body too), then rebind
                // atomically. Updating in place here would let a
                // later accumulator observe an earlier one's new
                // value, which the runtime never does.
                let mut new_vals = Vec::with_capacity(accs.len());
                for (_, _, update) in accs.iter() {
                    let v = live_value(genctx, update);
                    if strict && !v.is_ok() {
                        return Err(SimErr::Fault);
                    }
                    new_vals.push(v.value);
                }
                i += *step;
                genctx.add_local(var, CType::Long, Value::Long(i));
                for ((name, _, _), val) in accs.iter().zip(new_vals) {
                    genctx.add_local(name, CType::Long, val);
                }
            }
            genctx.truncate_to(snap);
            Ok(())
        }
        Stmt::Destructure {
            names,
            values,
            body,
            ..
        } => {
            // Sources are immutable with constant values: re-bind the
            // same names in order (mirroring population exactly, so
            // ids realign) and simulate the body. No evaluation, so
            // no fault is possible here in either mode.
            let snap = genctx.snapshot_len();
            for (name, v) in names.iter().zip(values.iter()) {
                *budget = budget.checked_sub(1).ok_or(SimErr::Budget)?;
                genctx.add_local(name, CType::Long, Value::Long(*v));
            }
            try_simulate_stmts(genctx, body, budget, strict)?;
            genctx.truncate_to(snap);
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

/// Halve every loop bound in a statement tree (minimum end bound 2).
/// Returns false when nothing was shrinkable.
pub fn shrink_loop_bounds(stmt: &mut Stmt) -> bool {
    match stmt {
        Stmt::Loop { end, body, .. } | Stmt::Dotimes { end, body, .. } => {
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
        Stmt::Let { body, .. } => {
            let mut shrunk = false;
            for b in body.iter_mut() {
                shrunk |= shrink_loop_bounds(b);
            }
            shrunk
        }
        Stmt::Recur { end, body, .. } => {
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
        Stmt::Destructure { body, .. } => {
            let mut shrunk = false;
            for b in body.iter_mut() {
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
                ctx.restore_scope_from(pre_ctx);
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
    let target_id = *rng.pick(&writable);
    let target = genctx.var_name(target_id).to_string();
    let typ = genctx.var_type(target_id);
    let kind: AssignKind = rng.pick_weighted(&policy.assign_distr);
    if kind == AssignKind::Reduction && typ != CType::Vec {
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
            id: target_id,
            name: target.clone(),
            ir: Ir::new(
                typ,
                genctx.var_value(target_id).clone(),
                crate::types::ExKind::NoEx,
            ),
        };
        let ir = types::eval_binop(op, self_expr.ir(), rhs.ir());
        if ir.is_ok() {
            return Some(Stmt::Assign {
                target,
                target_id,
                expr: Expr::BinOp {
                    op,
                    left: Box::new(self_expr),
                    right: Box::new(rhs),
                    ir,
                },
            });
        }
        return Some(Stmt::Assign {
            target,
            target_id,
            expr: rhs,
        });
    }
    Some(Stmt::Assign {
        target,
        target_id,
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
    let mut next = if cond_val { then_ctx } else { else_ctx };
    // Scope the depth: branches see +1 while populating, but the
    // increment must not leak to later siblings (loops restore via
    // restore_scope_from; ifs restore here).
    next.if_depth = genctx.if_depth;
    (next, vec![Stmt::If { cond, then, els }])
}

/// A sequential `let` block over a populated body. Bindings are
/// generated against the progressively extended context (later ones
/// see earlier ones); values thread by re-evaluation, leniently in
/// dead code exactly like assignments.
fn gen_let(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
    tag: &str,
) -> (GenCtx, Vec<Stmt>) {
    let mut ctx = genctx.clone();
    let snap = ctx.snapshot_len();
    let mut bindings = Vec::new();
    for k in 0..rng.gen_range(1, 3) {
        let rhs = expr::generate_safe(rng, policy, &ctx, false, exc_mode);
        let name = format!("l_{tag}_{k}");
        let v = live_value(&ctx, &rhs);
        ctx.add_local(&name, CType::Long, v.value);
        bindings.push((name, rhs));
    }
    ctx.let_depth += 1;
    let (mut after, body) = populate_scope(rng, policy, ctx, exc_mode, depth + 1);
    after.truncate_to(snap);
    after.let_depth = genctx.let_depth;
    (after, vec![Stmt::Let { bindings, body }])
}

/// A counted `loop`/`recur` block: counter plus 0-2 accumulators,
/// each with an entry init and a per-iteration update. Bodies that
/// fault or blow the budget under evolved values regenerate a few
/// times, then degrade to a stub scope, exactly like `doseq` loops.
fn gen_recur(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
    tag: &str,
) -> (GenCtx, Vec<Stmt>) {
    for _ in 0..4 {
        let (start, end, step) = policy::loop_bounds_at_depth(rng, policy, genctx.loop_depth);
        let var = format!("r_{tag}_{}", genctx.loop_depth);
        let mut body_ctx = genctx.clone();
        body_ctx.loop_depth += 1;
        body_ctx.add_local(&var, CType::Long, Value::Long(start));
        let mut accs = Vec::new();
        for k in 0..rng.gen_range(0, 2) {
            let init = expr::generate_safe(rng, policy, &body_ctx, false, exc_mode);
            let update = expr::generate_safe(rng, policy, &body_ctx, false, exc_mode);
            let name = format!("ra_{tag}_{k}");
            let v = live_value(&body_ctx, &init);
            body_ctx.add_local(&name, CType::Long, v.value);
            accs.push((name, init, update));
        }
        let (after_body, body) = populate_scope(rng, policy, body_ctx, exc_mode, depth + 1);
        let mut stmt = Stmt::Recur {
            var,
            start,
            end,
            step,
            accs,
            body,
        };
        match simulate_loop_or_shrink(genctx, &mut stmt, after_body.stmt_count) {
            LoopSim::Ok(final_ctx) => return (final_ctx, vec![stmt]),
            LoopSim::Fault => continue,
        }
    }
    populate_stub_scope(rng, policy, genctx.clone(), exc_mode, depth)
}

/// A `let` with a destructuring pattern over an immutable source:
/// a sequential prefix of a known vector, or present keys of a known
/// map. Sources are small literals or vector/map inputs; patterns
/// never overrun (names fit the length, keys are present), so no nil
/// can arise. Values are constants, so no validation is needed —
/// population binds and simulates exactly once.
fn gen_destructure(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    exc_mode: Density,
    depth: i64,
    tag: &str,
) -> (GenCtx, Vec<Stmt>) {
    let seq = rng.roll(0.5);
    let cands: Vec<VarId> = genctx
        .all_vars()
        .into_iter()
        .filter(|id| {
            let t = genctx.var_type(*id);
            // Records destructure as maps (same `get` lookups).
            if seq {
                t == CType::Vec
            } else {
                t == CType::Map || t == CType::Record
            }
        })
        .collect();
    // Prefer a var source when one with a non-empty value exists
    // (inputs always qualify); otherwise use a small literal. The
    // length check keeps over-long patterns impossible by
    // construction: names always fit the source.
    let mut var_pick: Option<VarId> = None;
    if !cands.is_empty() && rng.roll(0.5) {
        let id = *rng.pick(&cands);
        let len = match genctx.var_value(id) {
            Value::Vec(items) => items.len(),
            Value::Map(items) | Value::Record(items) => items.len(),
            _ => 0,
        };
        if len > 0 {
            var_pick = Some(id);
        }
    }
    let (pattern, names, source, values) = if seq {
        let items: Vec<i64> = match var_pick {
            Some(id) => match genctx.var_value(id) {
                // Only the destructured prefix (≤ 3) is ever read;
                // cloning a 64K-element input to keep it is waste.
                Value::Vec(items) => items.iter().take(3).copied().collect(),
                // Unreachable: length-checked above.
                _ => vec![0, 0],
            },
            None => (0..rng.gen_range(2, 4))
                .map(|_| rng.gen_range(-100, 100))
                .collect(),
        };
        let n = rng.gen_range(1, items.len().min(3) as i64) as usize;
        let names: Vec<String> = (0..n).map(|k| format!("d_{tag}_{k}")).collect();
        let values = items[..n].to_vec();
        let source = match var_pick {
            Some(id) => DestrSource::Var(genctx.var_name(id).to_string()),
            None => DestrSource::VecLit(items),
        };
        (DestrPattern::Seq, names, source, values)
    } else {
        let items: Vec<(String, i64)> = match var_pick {
            Some(id) => match genctx.var_value(id) {
                Value::Map(items) | Value::Record(items) => {
                    items.iter().map(|(k, v)| (k.clone(), *v)).collect()
                }
                // Unreachable: length-checked above.
                _ => vec![("ka".to_string(), 0)],
            },
            None => match types::random_map(rng).value {
                Value::Map(items) => items.into_iter().collect(),
                // Unreachable: random_map always yields a map.
                _ => vec![("ka".to_string(), 0)],
            },
        };
        // Random key subset, Fisher-Yates prefix over a few keys.
        let mut order: Vec<usize> = (0..items.len()).collect();
        for i in (1..order.len()).rev() {
            let j = rng.gen_usize(i + 1);
            order.swap(i, j);
        }
        let n = rng.gen_range(1, items.len().min(3) as i64) as usize;
        let mut names = Vec::with_capacity(n);
        let mut keys = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for (k, oi) in order[..n].iter().enumerate() {
            names.push(format!("d_{tag}_{k}"));
            keys.push(items[*oi].0.clone());
            values.push(items[*oi].1);
        }
        let source = match var_pick {
            Some(id) => DestrSource::Var(genctx.var_name(id).to_string()),
            None => DestrSource::MapLit(items),
        };
        (DestrPattern::Map(keys), names, source, values)
    };
    let mut ctx = genctx.clone();
    let snap = ctx.snapshot_len();
    for (name, v) in names.iter().zip(values.iter()) {
        ctx.add_local(name, CType::Long, Value::Long(*v));
    }
    ctx.let_depth += 1;
    let (mut after, body) = populate_scope(rng, policy, ctx, exc_mode, depth + 1);
    after.truncate_to(snap);
    after.let_depth = genctx.let_depth;
    (
        after,
        vec![Stmt::Destructure {
            pattern,
            names,
            source,
            values,
            body,
        }],
    )
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
        // Half of all loops use dotimes: same trips as a unit-step
        // doseq from 0, different macroexpansion stress.
        let dotimes = rng.roll(0.5);
        let (start, end, step) = if dotimes {
            (
                0,
                policy::loop_end_at_depth(rng, policy, genctx.loop_depth),
                1,
            )
        } else {
            policy::loop_bounds_at_depth(rng, policy, genctx.loop_depth)
        };
        let var = format!("i_{tag}_{}", genctx.loop_depth);
        let mut body_ctx = genctx.clone();
        body_ctx.add_local(&var, CType::Long, Value::Long(start));
        body_ctx.loop_depth += 1;
        let (after_body, body) = populate_scope(rng, policy, body_ctx, exc_mode, depth + 1);
        let mut stmt = if dotimes {
            Stmt::Dotimes { var, end, body }
        } else {
            Stmt::Loop {
                var,
                start,
                end,
                step,
                body,
            }
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
    // The nest builder does not re-check the limit per level, so
    // cap its levels by the remaining total-depth budget: the depth
    // limit is a true bound on the control path, not a suggestion.
    let remaining = (policy.loop_nest_depth_limit
        - (genctx.loop_depth + genctx.if_depth + genctx.let_depth))
        .max(1);
    let n = rng
        .gen_range(
            policy.loop_nest_depth_min,
            policy.loop_nest_depth_max.min(policy.loop_nest_depth_limit),
        )
        .clamp(1, remaining);

    struct NestBuilder<'a> {
        rng: &'a mut Rng,
        policy: &'a Policy,
        exc_mode: Density,
        tag: &'a str,
        levels: i64,
    }

    impl NestBuilder<'_> {
        fn build(&mut self, genctx: &GenCtx, depth: i64, level: i64) -> (GenCtx, Stmt) {
            let dotimes = self.rng.roll(0.5);
            let (start, end, step) = if dotimes {
                (
                    0,
                    policy::loop_end_at_depth(self.rng, self.policy, genctx.loop_depth),
                    1,
                )
            } else {
                policy::loop_bounds_at_depth(self.rng, self.policy, genctx.loop_depth)
            };
            let var = format!("i_{}_nest{level}", self.tag);
            let mut body_ctx = genctx.clone();
            body_ctx.add_local(&var, CType::Long, Value::Long(start));
            body_ctx.loop_depth += 1;
            let wrap = |var: String, body: Vec<Stmt>| {
                if dotimes {
                    Stmt::Dotimes { var, end, body }
                } else {
                    Stmt::Loop {
                        var,
                        start,
                        end,
                        step,
                        body,
                    }
                }
            };
            if level == self.levels - 1 {
                let (after, body) =
                    populate_scope(self.rng, self.policy, body_ctx, self.exc_mode, depth + 1);
                (after, wrap(var, body))
            } else {
                let (after, inner) = self.build(&body_ctx, depth + 1, level + 1);
                (after, wrap(var, vec![inner]))
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
    // Depth limits bound the TOTAL control path: bytecode density
    // compounds with nesting, so a loop inside an if inside a let
    // counts every level no matter the kind.
    let total_depth = genctx.loop_depth + genctx.if_depth + genctx.let_depth;
    match rng.pick_weighted(&policy.struct_distr) {
        StructKind::LoopSeq => {
            if total_depth < policy.loop_nest_depth_limit {
                gen_loop(rng, policy, genctx, exc_mode, depth, tag)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::LoopNest => {
            if total_depth < policy.loop_nest_depth_limit {
                gen_loop_nest(rng, policy, genctx, exc_mode, depth, tag)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::IfElse => {
            if total_depth < policy.if_else_limit {
                gen_if(rng, policy, genctx, exc_mode, depth)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::Let => {
            if total_depth < policy.let_nest_limit {
                gen_let(rng, policy, genctx, exc_mode, depth, tag)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::Recur => {
            if total_depth < policy.loop_nest_depth_limit {
                gen_recur(rng, policy, genctx, exc_mode, depth, tag)
            } else {
                assign_or(rng, policy, genctx, exc_mode)
            }
        }
        StructKind::Destructure => {
            // Binding forms share let's depth budget.
            if total_depth < policy.let_nest_limit {
                gen_destructure(rng, policy, genctx, exc_mode, depth, tag)
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
    let n = policy::scope_size(rng, policy, depth);
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

/// Maximum rendered flat chars for a single top-level statement,
/// which becomes one JVM method. Cost units undercount `doseq`/`if`
/// macroexpansion badly (control-dense methods broke the 64KB
/// ceiling at ~2400 and even ~1250 chars, the latter 6-deep), so
/// rendered size guards what units cannot. Sized below the smallest
/// observed failure with margin for denser shapes; natural tapered
/// loops render in the hundreds of chars. Multi-statement chunks
/// need no such bar: the 60-unit function budget already holds them
/// near ~400 chars.
pub const MAX_STMT_SOURCE: usize = 800;

/// True when a top-level statement fits one emitted function: under
/// both the cost-unit bar and the rendered-source bar.
pub fn stmt_fits(s: &Stmt, atoms: &BTreeSet<String>) -> bool {
    crate::stats::stmt_cost(s) <= crate::emitter::MAX_STMT_COST
        && to_sexp(s, atoms).flat().len() <= MAX_STMT_SOURCE
}

/// Populate the top-level scope with per-statement size repair. Each
/// populate unit is retried until every yielded statement fits one
/// function; a unit that never fits degrades to one stub scope.
/// Repairing per statement (not per scope) keeps one oversized loop
/// nest from nuking the whole scope's control-flow diversity.
pub fn populate_top_scope(
    rng: &mut Rng,
    policy: &Policy,
    genctx: GenCtx,
    exc_mode: Density,
) -> (GenCtx, Vec<Stmt>) {
    let atoms: BTreeSet<String> = genctx.output_names().into_iter().collect();
    let n = policy::scope_size(rng, policy, 0);
    let mut ctx = genctx;
    let mut stmts = Vec::new();
    let mut k = 0;
    while k < n && ctx.stmt_count < policy.stmt_num_limit {
        let tag = format!("0_{k}_{}", ctx.stmt_count);
        let mut placed = false;
        for _ in 0..3 {
            let (next, s) = populate_one(rng, policy, &ctx, exc_mode, 0, &tag);
            if s.iter().all(|x| stmt_fits(x, &atoms)) {
                ctx = next;
                stmts.extend(s);
                placed = true;
                break;
            }
        }
        if !placed {
            let (next, s) = populate_stub_scope(rng, policy, ctx, exc_mode, 0);
            debug_assert!(s.iter().all(|x| stmt_fits(x, &atoms)));
            ctx = next;
            stmts.extend(s);
        }
        k += 1;
    }
    (ctx, stmts)
}

/// Lower a statement to a Clojure s-expression. Mutable outputs are
/// atoms; loop variables and inputs are plain values.
pub fn to_sexp(stmt: &Stmt, atoms: &BTreeSet<String>) -> Sexp {
    match stmt {
        Stmt::Assign { target, expr, .. } => {
            debug_assert!(
                atoms.contains(target),
                "assignment target {target} is not an atom"
            );
            Sexp::list(vec![
                Sexp::sym("reset!"),
                Sexp::sym(target),
                expr::to_sexp(expr, atoms),
            ])
        }
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
        // Deliberately unhinted: dotimes rebinds through its own
        // expansion, so binding metadata has murkier semantics than
        // the probed doseq case.
        Stmt::Dotimes { var, end, body } => {
            let mut items = vec![
                Sexp::sym("dotimes"),
                Sexp::vec(vec![Sexp::sym(var), Sexp::int(*end)]),
            ];
            items.extend(body.iter().map(|s| to_sexp(s, atoms)));
            Sexp::list(items)
        }
        Stmt::Let { bindings, body } => {
            let mut bind = Vec::with_capacity(bindings.len() * 2);
            for (name, rhs) in bindings {
                bind.push(Sexp::sym(name));
                bind.push(expr::to_sexp(rhs, atoms));
            }
            let mut items = vec![Sexp::sym("let"), Sexp::vec(bind)];
            items.extend(body.iter().map(|s| to_sexp(s, atoms)));
            Sexp::list(items)
        }
        Stmt::Recur {
            var,
            start,
            end,
            step,
            accs,
            body,
        } => {
            // (loop [i start, acc init, ...]
            //   (if (< i end)
            //     (do body... (recur (+ i step) update...))
            //     nil))
            // The recur sits last in the do, so it is in tail
            // position; arity matches the binding vector exactly.
            let mut bind = Vec::with_capacity(2 + accs.len() * 2);
            bind.push(Sexp::sym(var));
            bind.push(Sexp::int(*start));
            for (name, init, _) in accs {
                bind.push(Sexp::sym(name));
                bind.push(expr::to_sexp(init, atoms));
            }
            let mut recur_args = vec![Sexp::list(vec![
                Sexp::sym("+"),
                Sexp::sym(var),
                Sexp::int(*step),
            ])];
            for (_, _, update) in accs {
                recur_args.push(expr::to_sexp(update, atoms));
            }
            let mut tail = vec![Sexp::sym("do")];
            tail.extend(body.iter().map(|s| to_sexp(s, atoms)));
            tail.push(Sexp::list({
                let mut r = vec![Sexp::sym("recur")];
                r.extend(recur_args);
                r
            }));
            Sexp::list(vec![
                Sexp::sym("loop"),
                Sexp::vec(bind),
                Sexp::list(vec![
                    Sexp::sym("if"),
                    Sexp::list(vec![Sexp::sym("<"), Sexp::sym(var), Sexp::int(*end)]),
                    Sexp::list(tail),
                    Sexp::sym("nil"),
                ]),
            ])
        }
        Stmt::Destructure {
            pattern,
            names,
            source,
            body,
            ..
        } => {
            let src = match source {
                DestrSource::VecLit(items) => {
                    Sexp::vec(items.iter().map(|n| Sexp::int(*n)).collect())
                }
                DestrSource::MapLit(pairs) => {
                    let mut flat = Vec::with_capacity(pairs.len() * 2);
                    for (k, v) in pairs {
                        flat.push(Sexp::sym(&format!(":{k}")));
                        flat.push(Sexp::int(*v));
                    }
                    Sexp::map(flat)
                }
                DestrSource::Var(name) => Sexp::sym(name),
            };
            let pat = match pattern {
                DestrPattern::Seq => Sexp::vec(names.iter().map(|n| Sexp::sym(n)).collect()),
                DestrPattern::Map(keys) => {
                    let mut flat = Vec::with_capacity(names.len() * 2);
                    for (n, k) in names.iter().zip(keys.iter()) {
                        flat.push(Sexp::sym(n));
                        flat.push(Sexp::sym(&format!(":{k}")));
                    }
                    Sexp::map(flat)
                }
            };
            let mut items = vec![Sexp::sym("let"), Sexp::vec(vec![pat, src])];
            items.extend(body.iter().map(|s| to_sexp(s, atoms)));
            Sexp::list(items)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shuffled_policy(rng: &mut Rng) -> Policy {
        let mut p = Policy::default();
        rng.shuffle_weights(&mut p.struct_distr);
        rng.shuffle_weights(&mut p.assign_distr);
        rng.shuffle_weights(&mut p.node_distr);
        p
    }

    #[test]
    fn oversized_statements_fail_stmt_fits() {
        // A bushy loop renders past the per-statement source bar, so
        // raw scopes must go through populate_top_scope repair: size
        // enforcement lives there, not in populate_scope.
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let ctx = seeded_ctx();
        let oid = ctx.id_of("out-0").unwrap();
        let body: Vec<Stmt> = (0..60)
            .map(|i| Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Const(Ir::long(i)),
            })
            .collect();
        let big = Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 2,
            step: 1,
            body,
        };
        assert!(!stmt_fits(&big, &atoms));
        let small = Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: Expr::Const(Ir::long(1)),
        };
        assert!(stmt_fits(&small, &atoms));
    }

    #[test]
    fn top_scope_repair_caps_every_statement() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        for seed in 1..12u64 {
            let mut rng = Rng::new(seed);
            let policy = shuffled_policy(&mut rng);
            let (_, stmts) = populate_top_scope(&mut rng, &policy, seeded_ctx(), Density::None);
            for s in &stmts {
                assert!(
                    stmt_fits(s, &atoms),
                    "seed {seed}: oversized stmt escaped repair"
                );
            }
        }
    }

    #[test]
    fn depths_are_scoped_not_cumulative() {
        // if_depth must reset when an if-scope exits (like loop_depth
        // does via restore_scope_from); otherwise if/loop picks dry
        // up for later siblings once ~5 ifs accumulate.
        for seed in 1..20u64 {
            let mut rng = Rng::new(seed);
            let policy = shuffled_policy(&mut rng);
            let (ctx, _) = populate_top_scope(&mut rng, &policy, seeded_ctx(), Density::None);
            assert_eq!(ctx.if_depth, 0, "seed {seed}: if_depth leaked");
            assert_eq!(ctx.loop_depth, 0, "seed {seed}: loop_depth leaked");
            assert_eq!(ctx.let_depth, 0, "seed {seed}: let_depth leaked");
        }
    }

    #[test]
    fn top_scope_repair_preserves_control_flow() {
        // Whole-scope repopulation degraded ~85% of shuffled programs
        // to straight-line assigns; per-statement repair must keep
        // loops and branches abundant.
        let mut control = 0;
        for seed in 1..=10u64 {
            let mut rng = Rng::new(seed);
            let policy = shuffled_policy(&mut rng);
            let (_, stmts) = populate_top_scope(&mut rng, &policy, seeded_ctx(), Density::None);
            control += stmts
                .iter()
                .filter(|s| !matches!(s, Stmt::Assign { .. }))
                .count();
        }
        assert!(
            control >= 6,
            "only {control} control-flow tops over 10 programs"
        );
    }

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
                sim.lookup("out-0").unwrap().as_long(),
                ctx.lookup("out-0").unwrap().as_long()
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
                Stmt::Loop { body, .. } | Stmt::Dotimes { body, .. } => collect_targets(body, out),
                Stmt::Let { body, .. } => collect_targets(body, out),
                Stmt::Recur { body, .. } => collect_targets(body, out),
                Stmt::Destructure { body, .. } => collect_targets(body, out),
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
                let id = ctx.id_of(t).expect("target interned");
                assert!(
                    ctx.writable_vars().contains(&id),
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
        let oid = ctx.id_of("out-0").unwrap();
        ctx.update_value_by_id(oid, Value::Long(2));
        let var = || Expr::Var {
            id: oid,
            name: "out-0".to_string(),
            ir: Ir::long(2),
        };
        let square = Expr::BinOp {
            op: BinOp::Mul,
            left: Box::new(var()),
            right: Box::new(var()),
            ir: Ir::long(4),
        };
        let prog = vec![Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 10,
            step: 1,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: square,
            }],
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn strict_simulation_catches_division_by_zero() {
        let mut ctx = seeded_ctx();
        let oid = ctx.id_of("out-0").unwrap();
        let div = Expr::BinOp {
            op: BinOp::Div,
            left: Box::new(Expr::Const(Ir::long(1))),
            right: Box::new(Expr::Var {
                id: oid,
                name: "out-0".to_string(),
                ir: Ir::long(0),
            }),
            ir: Ir::long(0),
        };
        let prog = vec![Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: div,
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn huge_nth_index_never_truncates_into_range() {
        // (1<<32)+1 truncates to 1 under `as usize` on 32-bit
        // targets; the bounds check must run in i64 so it still
        // faults there. (Passes either way on 64-bit, where the
        // cast is exact.)
        let mut ctx = GenCtx::new();
        ctx.add_input("v", CType::Vec, Value::Vec(vec![1, 2, 3]));
        ctx.add_output("out-0", CType::Long, Value::Long(0));
        let vid = ctx.id_of("v").unwrap();
        let oid = ctx.id_of("out-0").unwrap();
        let nth = Expr::Nth {
            coll: Box::new(Expr::Var {
                id: vid,
                name: "v".to_string(),
                ir: Ir::new(
                    CType::Vec,
                    Value::Vec(vec![1, 2, 3]),
                    crate::types::ExKind::NoEx,
                ),
            }),
            idx: Box::new(Expr::Const(Ir::long((1 << 32) + 1))),
            ir: Ir::long(0),
        };
        let prog = vec![Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: nth,
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn zero_trip_loops_skip_the_body() {
        // end 0: the exit test fails at entry, so even a body that
        // would fault never executes — strict simulation must pass
        // and outputs keep their entry values. Recur accumulators
        // likewise keep their inits.
        let mut ctx = seeded_ctx();
        let oid = ctx.id_of("out-0").unwrap();
        let fault = Expr::BinOp {
            op: BinOp::Div,
            left: Box::new(Expr::Const(Ir::long(1))),
            right: Box::new(Expr::Const(Ir::long(0))),
            ir: Ir::long(0),
        };
        let mk = |expr: Expr| Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 0,
            step: 1,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr,
            }],
        };
        assert_eq!(simulate_bounded(&mut ctx, &[mk(fault)]), Ok(()));
        let dot = Stmt::Dotimes {
            var: "j".to_string(),
            end: 0,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Const(Ir::long(99)),
            }],
        };
        assert_eq!(simulate_bounded(&mut ctx, &[dot]), Ok(()));
        assert_eq!(ctx.lookup("out-0").unwrap().as_long(), 0);
        let rec = Stmt::Recur {
            var: "r".to_string(),
            start: 0,
            end: 0,
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
                    id: ctx.id_of("out-0").unwrap(),
                    name: "out-0".to_string(),
                    ir: Ir::long(0),
                },
            }],
        };
        assert_eq!(simulate_bounded(&mut ctx, &[rec]), Ok(()));
        // Counter and accumulator locals leave scope with the loop.
        assert!(ctx.id_of("r").is_none());
        assert!(ctx.id_of("ra").is_none());
    }

    fn map_ctx() -> (GenCtx, crate::context::VarId, crate::context::VarId) {
        let mut ctx = GenCtx::new();
        let items: std::collections::BTreeMap<String, i64> =
            [("ka".to_string(), 1), ("kb".to_string(), 2)]
                .into_iter()
                .collect();
        ctx.add_input("m", CType::Map, Value::Map(items));
        ctx.add_output("out-0", CType::Long, Value::Long(0));
        let mid = ctx.id_of("m").unwrap();
        let oid = ctx.id_of("out-0").unwrap();
        (ctx, mid, oid)
    }

    fn map_var(mid: crate::context::VarId) -> Expr {
        Expr::Var {
            id: mid,
            name: "m".to_string(),
            ir: Ir::new(
                CType::Map,
                Value::Map(
                    [("ka".to_string(), 1), ("kb".to_string(), 2)]
                        .into_iter()
                        .collect(),
                ),
                crate::types::ExKind::NoEx,
            ),
        }
    }

    #[test]
    fn simulation_threads_get_and_assoc() {
        // Present-key get, absent-key get-with-default, and read
        // through one assoc all simulate to their runtime values.
        let (mut ctx, mid, oid) = map_ctx();
        let get = Expr::Get {
            coll: Box::new(map_var(mid)),
            key: "kb".to_string(),
            default: None,
            kw: true,
            ir: Ir::long(2),
        };
        assert_eq!(live_value(&ctx, &get).value, Value::Long(2));
        let miss = Expr::Get {
            coll: Box::new(map_var(mid)),
            key: "kc".to_string(),
            default: Some(Box::new(Expr::Const(Ir::long(9)))),
            kw: false,
            ir: Ir::long(9),
        };
        assert_eq!(live_value(&ctx, &miss).value, Value::Long(9));
        // Keyword-invoke and plain get agree by construction (same
        // node, different emission).
        let plain = Expr::Get {
            coll: Box::new(map_var(mid)),
            key: "kb".to_string(),
            default: None,
            kw: false,
            ir: Ir::long(2),
        };
        assert_eq!(live_value(&ctx, &plain), live_value(&ctx, &get));
        let assoc = Expr::Assoc {
            coll: Box::new(map_var(mid)),
            key: "kc".to_string(),
            val: Box::new(Expr::Const(Ir::long(7))),
            ir: Ir::long(0),
        };
        let through = Expr::Get {
            coll: Box::new(assoc),
            key: "kc".to_string(),
            default: None,
            kw: false,
            ir: Ir::long(7),
        };
        let prog = vec![Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: through,
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Ok(()));
        assert_eq!(ctx.lookup("out-0").unwrap().as_long(), 7);
    }

    #[test]
    fn simulation_threads_records() {
        // Fresh construction evaluates args in field order; reads
        // hit fields; assoc of an existing field stays a record with
        // the overwritten value.
        let ctx = seeded_ctx();
        let rec = Expr::Record {
            args: vec![
                Expr::Const(Ir::long(1)),
                Expr::Const(Ir::long(2)),
                Expr::Const(Ir::long(3)),
            ],
            map_ctor: false,
            ir: Ir::long(0),
        };
        let v = live_value(&ctx, &rec);
        assert_eq!(v.typ, CType::Record);
        match &v.value {
            Value::Record(items) => {
                assert_eq!(items.get("fa"), Some(&1));
                assert_eq!(items.get("fb"), Some(&2));
                assert_eq!(items.get("fc"), Some(&3));
            }
            other => panic!("expected record, got {other:?}"),
        }
        let get = Expr::Get {
            coll: Box::new(rec),
            key: "fb".to_string(),
            default: None,
            kw: true,
            ir: Ir::long(2),
        };
        assert_eq!(live_value(&ctx, &get).value, Value::Long(2));
        // map_ctor agrees with positional (probed): same node shape
        // modulo the flag evaluates identically.
        let map_built = Expr::Record {
            args: vec![
                Expr::Const(Ir::long(4)),
                Expr::Const(Ir::long(5)),
                Expr::Const(Ir::long(6)),
            ],
            map_ctor: true,
            ir: Ir::long(0),
        };
        let assoc = Expr::Assoc {
            coll: Box::new(map_built),
            key: "fa".to_string(),
            val: Box::new(Expr::Const(Ir::long(9))),
            ir: Ir::long(0),
        };
        let av = live_value(&ctx, &assoc);
        assert_eq!(av.typ, CType::Record);
        match &av.value {
            Value::Record(items) => {
                assert_eq!(items.get("fa"), Some(&9));
                assert_eq!(items.len(), 3);
            }
            other => panic!("expected record, got {other:?}"),
        }
    }

    #[test]
    fn strict_simulation_faults_bare_map_miss() {
        // Absent key without a default would read nil at runtime;
        // the model has no nil, so strict simulation faults rather
        // than inventing a value. Unreachable by construction.
        let (mut ctx, mid, oid) = map_ctx();
        let miss = Expr::Get {
            coll: Box::new(map_var(mid)),
            key: "kc".to_string(),
            default: None,
            kw: false,
            ir: Ir::long(0),
        };
        let prog = vec![Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: miss,
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
        let vid = ctx.id_of("v").unwrap();
        let oid = ctx.id_of("out-0").unwrap();
        let nth = Expr::NthDefault {
            coll: Box::new(Expr::Var {
                id: vid,
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
            target_id: oid,
            expr: nth,
        }];
        assert_eq!(simulate_bounded(&mut ctx, &prog), Err(SimErr::Fault));
    }

    #[test]
    fn loop_emission_shape() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let mut ctx = GenCtx::new();
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let s = Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 10,
            step: 1,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Const(Ir::long(1)),
            }],
        };
        assert_eq!(
            to_sexp(&s, &atoms).flat(),
            "(doseq [i (range 0 10 1)] (reset! out-0 1))"
        );
    }

    #[test]
    fn let_emission_shape() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let mut ctx = GenCtx::new();
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let s = Stmt::Let {
            bindings: vec![
                ("l1".to_string(), Expr::Const(Ir::long(1))),
                ("l2".to_string(), Expr::Const(Ir::long(2))),
            ],
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Const(Ir::long(3)),
            }],
        };
        assert_eq!(
            to_sexp(&s, &atoms).flat(),
            "(let [l1 1 l2 2] (reset! out-0 3))"
        );
    }

    #[test]
    fn destructure_emission_shapes() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let mut ctx = GenCtx::new();
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let body = || {
            vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Const(Ir::long(1)),
            }]
        };
        let seq = Stmt::Destructure {
            pattern: DestrPattern::Seq,
            names: vec!["d_0".to_string(), "d_1".to_string()],
            source: DestrSource::Var("vec-0".to_string()),
            values: vec![7, 8],
            body: body(),
        };
        assert_eq!(
            to_sexp(&seq, &atoms).flat(),
            "(let [[d_0 d_1] vec-0] (reset! out-0 1))"
        );
        let lit = Stmt::Destructure {
            pattern: DestrPattern::Seq,
            names: vec!["d_0".to_string()],
            source: DestrSource::VecLit(vec![5, 6]),
            values: vec![5],
            body: body(),
        };
        assert_eq!(
            to_sexp(&lit, &atoms).flat(),
            "(let [[d_0] [5 6]] (reset! out-0 1))"
        );
        let map = Stmt::Destructure {
            pattern: DestrPattern::Map(vec!["ka".to_string(), "kc".to_string()]),
            names: vec!["d_0".to_string(), "d_1".to_string()],
            source: DestrSource::Var("map-0".to_string()),
            values: vec![1, 3],
            body: body(),
        };
        assert_eq!(
            to_sexp(&map, &atoms).flat(),
            "(let [{d_0 :ka d_1 :kc} map-0] (reset! out-0 1))"
        );
        let mlit = Stmt::Destructure {
            pattern: DestrPattern::Map(vec!["kb".to_string()]),
            names: vec!["d_0".to_string()],
            source: DestrSource::MapLit(vec![("kb".to_string(), 2)]),
            values: vec![2],
            body: body(),
        };
        assert_eq!(
            to_sexp(&mlit, &atoms).flat(),
            "(let [{d_0 :kb} {:kb 2}] (reset! out-0 1))"
        );
    }

    #[test]
    fn destructure_binds_values_and_restores_scope() {
        // Bound values are visible in the body and locals leave scope
        // on exit, exactly like let bindings.
        let ctx = seeded_ctx();
        let oid = ctx.id_of("out-0").unwrap();
        // The destructure binds "d" as the next interned id.
        let did = VarId(ctx.snapshot_len() as u32);
        let s = Stmt::Destructure {
            pattern: DestrPattern::Seq,
            names: vec!["d".to_string()],
            source: DestrSource::VecLit(vec![42, 43]),
            values: vec![42],
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Var {
                    id: did,
                    name: "d".to_string(),
                    ir: Ir::long(42),
                },
            }],
        };
        let mut sim = ctx;
        assert_eq!(simulate_bounded(&mut sim, &[s]), Ok(()));
        assert_eq!(sim.lookup("out-0").unwrap().as_long(), 42);
        assert!(sim.id_of("d").is_none());
    }

    fn destructure_ctx() -> GenCtx {
        let mut ctx = seeded_ctx();
        ctx.add_input("vec-0", CType::Vec, Value::Vec(vec![1, 2, 3]));
        let mut m = std::collections::BTreeMap::new();
        m.insert("ka".to_string(), 4);
        m.insert("kc".to_string(), 6);
        ctx.add_input("map-0", CType::Map, Value::Map(m));
        let mut r = std::collections::BTreeMap::new();
        r.insert("fa".to_string(), 7);
        r.insert("fb".to_string(), 8);
        r.insert("fc".to_string(), 9);
        ctx.add_input("rec-0", CType::Record, Value::Record(r));
        ctx
    }

    fn collect_destr_sources(stmts: &[Stmt], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::Destructure { source, body, .. } => {
                    match source {
                        DestrSource::Var(name) => out.push(name.clone()),
                        DestrSource::VecLit(_) => out.push("<veclit>".to_string()),
                        DestrSource::MapLit(_) => out.push("<maplit>".to_string()),
                    }
                    // Bodies nest under the same forced policy.
                    collect_destr_sources(body, out);
                }
                Stmt::If { then, els, .. } => {
                    collect_destr_sources(then, out);
                    collect_destr_sources(els, out);
                }
                Stmt::Loop { body, .. }
                | Stmt::Dotimes { body, .. }
                | Stmt::Let { body, .. }
                | Stmt::Recur { body, .. } => collect_destr_sources(body, out),
                Stmt::Assign { .. } => {}
            }
        }
    }

    #[test]
    fn destructure_populates_cleanly() {
        // A destructure-only policy yields destructuring blocks over
        // literals and input vars (vectors, maps, and records); every
        // yielded statement fits one function and simulates cleanly
        // under strict bounds.
        let mut saw_rec_source = false;
        for seed in 1..30u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy {
                struct_distr: vec![(StructKind::Destructure, 1)],
                ..Policy::default()
            };
            let ctx = destructure_ctx();
            let atoms: BTreeSet<String> = ctx.output_names().into_iter().collect();
            let (final_ctx, stmts) = populate_top_scope(&mut rng, &policy, ctx, Density::None);
            assert!(!stmts.is_empty(), "seed {seed}: no statements");
            for s in &stmts {
                assert!(stmt_fits(s, &atoms), "seed {seed}: oversized {s:?}");
            }
            let mut sources = Vec::new();
            collect_destr_sources(&stmts, &mut sources);
            saw_rec_source |= sources.iter().any(|n| n == "rec-0");
            let mut sim = destructure_ctx();
            assert_eq!(
                simulate_bounded_whole(&mut sim, &stmts),
                Ok(()),
                "seed {seed}: strict simulation fault"
            );
            assert_eq!(sim.lookup("out-0"), final_ctx.lookup("out-0"));
            assert_eq!(final_ctx.let_depth, 0, "seed {seed}: let_depth leaked");
        }
        assert!(saw_rec_source, "no destructure ever sourced rec-0");
    }

    #[test]
    fn let_bindings_are_sequential() {
        // Clojure let is let*: later bindings see earlier ones. The
        // simulation must thread the same order (pre-interned ids
        // stay aligned, as with loop variables).
        let mut base = seeded_ctx();
        base.add_local("a", CType::Long, Value::Long(0));
        base.add_local("b", CType::Long, Value::Long(0));
        let oid = base.id_of("out-0").unwrap();
        let aid = base.id_of("a").unwrap();
        let bid = base.id_of("b").unwrap();
        let avar = || Expr::Var {
            id: aid,
            name: "a".to_string(),
            ir: Ir::long(0),
        };
        let s = Stmt::Let {
            bindings: vec![
                ("a".to_string(), Expr::Const(Ir::long(1))),
                (
                    "b".to_string(),
                    Expr::BinOp {
                        op: BinOp::Add,
                        left: Box::new(avar()),
                        right: Box::new(Expr::Const(Ir::long(1))),
                        ir: Ir::long(2),
                    },
                ),
            ],
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Var {
                    id: bid,
                    name: "b".to_string(),
                    ir: Ir::long(0),
                },
            }],
        };
        let mut sim = base;
        simulate_stmts(&mut sim, &[s]);
        assert_eq!(sim.lookup("out-0").unwrap().as_long(), 2);
    }

    #[test]
    fn recur_emission_shape() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let mut ctx = GenCtx::new();
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let s = Stmt::Recur {
            var: "i".to_string(),
            start: 0,
            end: 9,
            step: 3,
            accs: vec![(
                "a".to_string(),
                Expr::Const(Ir::long(1)),
                Expr::Const(Ir::long(2)),
            )],
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Const(Ir::long(5)),
            }],
        };
        assert_eq!(
            to_sexp(&s, &atoms).flat(),
            "(loop [i 0 a 1] (if (< i 9) (do (reset! out-0 5) (recur (+ i 3) 2)) nil))"
        );
    }

    #[test]
    fn recur_accs_update_atomically() {
        // All recur arguments evaluate under pre-rebind values: with
        // a:=b, b:=a over two trips, the simulation must swap (final
        // out 12), not copy (which would give 22).
        let mut base = seeded_ctx();
        base.add_local("i", CType::Long, Value::Long(0));
        base.add_local("a", CType::Long, Value::Long(0));
        base.add_local("b", CType::Long, Value::Long(0));
        let oid = base.id_of("out-0").unwrap();
        let aid = base.id_of("a").unwrap();
        let bid = base.id_of("b").unwrap();
        let avar = || Expr::Var {
            id: aid,
            name: "a".to_string(),
            ir: Ir::long(0),
        };
        let bvar = || Expr::Var {
            id: bid,
            name: "b".to_string(),
            ir: Ir::long(0),
        };
        let s = Stmt::Recur {
            var: "i".to_string(),
            start: 0,
            end: 2,
            step: 1,
            accs: vec![
                ("a".to_string(), Expr::Const(Ir::long(1)), bvar()),
                ("b".to_string(), Expr::Const(Ir::long(2)), avar()),
            ],
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(avar()),
                    right: Box::new(Expr::BinOp {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Const(Ir::long(10))),
                        right: Box::new(bvar()),
                        ir: Ir::long(0),
                    }),
                    ir: Ir::long(0),
                },
            }],
        };
        let mut sim = base;
        simulate_stmts(&mut sim, &[s]);
        assert_eq!(sim.lookup("out-0").unwrap().as_long(), 12);
    }

    #[test]
    fn recur_counter_trips_match_loop() {
        let mut ctx = seeded_ctx();
        ctx.add_local("i", CType::Long, Value::Long(0));
        let oid = ctx.id_of("out-0").unwrap();
        let s = Stmt::Recur {
            var: "i".to_string(),
            start: 0,
            end: 9,
            step: 3,
            accs: vec![],
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::Var {
                        id: oid,
                        name: "out-0".to_string(),
                        ir: Ir::long(0),
                    }),
                    right: Box::new(Expr::Const(Ir::long(1))),
                    ir: Ir::long(1),
                },
            }],
        };
        let mut sim = ctx;
        simulate_stmts(&mut sim, &[s]);
        assert_eq!(sim.lookup("out-0").unwrap().as_long(), 3);
    }

    #[test]
    fn dotimes_emission_shape() {
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let mut ctx = GenCtx::new();
        let oid = ctx.add_output("out-0", CType::Long, Value::Long(0));
        let iid = ctx.add_local("i", CType::Long, Value::Long(0));
        let s = Stmt::Dotimes {
            var: "i".to_string(),
            end: 10,
            body: vec![Stmt::Assign {
                target: "out-0".to_string(),
                target_id: oid,
                expr: Expr::Var {
                    id: iid,
                    name: "i".to_string(),
                    ir: Ir::long(0),
                },
            }],
        };
        assert_eq!(
            to_sexp(&s, &atoms).flat(),
            "(dotimes [i 10] (reset! out-0 i))"
        );
    }

    #[test]
    fn dotimes_matches_unit_loop_trips() {
        // Pre-intern the loop variable: simulation upserts the same id
        // on loop entry, so hand-built references stay aligned.
        let mut base = seeded_ctx();
        base.add_local("i", CType::Long, Value::Long(0));
        let oid = base.id_of("out-0").unwrap();
        let iid = base.id_of("i").unwrap();
        let body = vec![Stmt::Assign {
            target: "out-0".to_string(),
            target_id: oid,
            expr: Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Var {
                    id: oid,
                    name: "out-0".to_string(),
                    ir: Ir::long(0),
                }),
                right: Box::new(Expr::Var {
                    id: iid,
                    name: "i".to_string(),
                    ir: Ir::long(0),
                }),
                ir: Ir::long(0),
            },
        }];
        let doseq = Stmt::Loop {
            var: "i".to_string(),
            start: 0,
            end: 7,
            step: 1,
            body: body.clone(),
        };
        let dotimes = Stmt::Dotimes {
            var: "i".to_string(),
            end: 7,
            body,
        };
        let mut a = base.clone();
        simulate_stmts(&mut a, &[doseq]);
        let mut b = base;
        simulate_stmts(&mut b, &[dotimes]);
        // 0+1+2+3+4+5+6
        assert_eq!(a.lookup("out-0").unwrap().as_long(), 21);
        assert_eq!(b.lookup("out-0").unwrap().as_long(), 21);
    }
}
