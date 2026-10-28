// Licensed under the Apache License, Version 2.0.
//! Arithmetic expression generator with generation-time evaluation and
//! exception-driven rebuilding. Ports YARPGen's expr.cpp
//! `ArithmeticExpr::create` + propagateType/evaluate/rebuild.

use crate::cli::Density;
use crate::context::{GenCtx, VarId};
use crate::policy::{NodeKind, Policy};
use crate::rng::Rng;
use crate::sexp::Sexp;
use crate::types::{self, BinOp, CType, ExKind, Ir, UnOp, Value};

const MAX_REBUILD_ATTEMPTS: i32 = 8;

/// Expression AST. Every node carries its generation-time value.
#[derive(Clone, Debug)]
pub enum Expr {
    Const(Ir),
    Var {
        id: VarId,
        name: String,
        ir: Ir,
    },
    UnOp {
        op: UnOp,
        arg: Box<Expr>,
        ir: Ir,
    },
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
        ir: Ir,
    },
    Nth {
        coll: Box<Expr>,
        idx: Box<Expr>,
        ir: Ir,
    },
    NthDefault {
        coll: Box<Expr>,
        idx: Box<Expr>,
        default: Box<Expr>,
        ir: Ir,
    },
    /// Floored-modulo index wrap. Emits `(mod idx len)`; always in
    /// range for a non-empty collection, so its IR is always clean.
    IdxMod {
        idx: Box<Expr>,
        len: i64,
        ir: Ir,
    },
}

impl Expr {
    pub fn ir(&self) -> &Ir {
        match self {
            Expr::Const(ir) => ir,
            Expr::Var { ir, .. } => ir,
            Expr::UnOp { ir, .. } => ir,
            Expr::BinOp { ir, .. } => ir,
            Expr::Nth { ir, .. } => ir,
            Expr::NthDefault { ir, .. } => ir,
            Expr::IdxMod { ir, .. } => ir,
        }
    }
}

/// Dead-code exception opt-in. Ports `--allow-ub-in-dc`: allowed only
/// when the context is not taken.
pub fn allow_exc(rng: &mut Rng, policy: &Policy, taken: bool, mode: Density) -> bool {
    if taken {
        return false;
    }
    match mode {
        Density::All => true,
        Density::Some => {
            let total = policy.ub_allow_weight + policy.ub_deny_weight;
            rng.roll(f64::from(policy.ub_allow_weight) / f64::from(total))
        }
        Density::None => false,
    }
}

/// With `similar_operators_prob`, restrict binops to one family.
fn similar_op_pool(rng: &mut Rng, policy: &Policy) -> Option<Vec<BinOp>> {
    if rng.roll(policy.similar_operators_prob) {
        let fams = types::op_families();
        Some(rng.pick(&fams).clone())
    } else {
        None
    }
}

/// Const or variable leaf. ConstUse forces constants; otherwise reuse
/// rolls decide between an existing variable and a fresh constant.
fn leaf(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    want_bool: bool,
    const_only: bool,
) -> Expr {
    let const_only = const_only || rng.roll(policy.const_use_prob);
    let vars: Vec<VarId> = genctx
        .all_vars()
        .into_iter()
        .filter(|id| {
            let typ = genctx.var_type(*id);
            if want_bool {
                typ == CType::Bool
            } else {
                typ.is_scalar()
            }
        })
        .collect();
    let use_var = !const_only && !vars.is_empty() && rng.roll(policy.reuse_prob);
    if use_var {
        let id = *rng.pick(&vars);
        Expr::Var {
            id,
            name: genctx.var_name(id).to_string(),
            ir: Ir::new(
                genctx.var_type(id),
                genctx.var_value(id).clone(),
                ExKind::NoEx,
            ),
        }
    } else {
        let typ = if want_bool {
            CType::Bool
        } else {
            // Scalar operands must never be boolean: Clojure arithmetic
            // on booleans throws, so pick from the integral types only.
            let integrals: Vec<CType> = policy
                .int_types
                .iter()
                .copied()
                .filter(|t| *t != CType::Bool)
                .collect();
            if integrals.is_empty() {
                CType::Long
            } else {
                *rng.pick(&integrals)
            }
        };
        Expr::Const(types::random_const(rng, policy, typ))
    }
}

fn gen_unary(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    depth: i64,
    pool: &Option<Vec<BinOp>>,
) -> Expr {
    let op = [
        UnOp::Neg,
        UnOp::Neg,
        UnOp::BNot,
        UnOp::Inc,
        UnOp::Dec,
        UnOp::Abs,
    ][rng.gen_usize(6)];
    let arg = generate(rng, policy, genctx, depth + 1, false, pool);
    let ir = types::eval_unop(op, arg.ir());
    Expr::UnOp {
        op,
        arg: Box::new(arg),
        ir,
    }
}

fn gen_binary(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    depth: i64,
    want_bool: bool,
    pool: &Option<Vec<BinOp>>,
) -> Expr {
    let op = if let Some(p) = pool {
        if want_bool {
            *rng.pick(p)
        } else {
            // A compare pool must not leak booleans into scalar
            // positions; fall back to scalar operators instead.
            let scalars: Vec<BinOp> = p.iter().copied().filter(|o| !o.is_compare()).collect();
            if scalars.is_empty() {
                *rng.pick(BinOp::scalar())
            } else {
                *rng.pick(&scalars)
            }
        }
    } else if want_bool {
        *rng.pick(BinOp::compares())
    } else {
        *rng.pick(BinOp::scalar())
    };
    let left = generate(rng, policy, genctx, depth + 1, false, &None);
    let right = generate(rng, policy, genctx, depth + 1, false, &None);
    let ir = types::eval_binop(op, left.ir(), right.ir());
    Expr::BinOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
        ir,
    }
}

/// Vector subscript. Ports YARPGen's subscript policy: the index is
/// rebuilt with a floored `mod` wrap, or a default is added, when
/// generation-time evaluation reports out-of-bounds.
fn gen_nth(rng: &mut Rng, policy: &Policy, genctx: &GenCtx, depth: i64) -> Expr {
    let vecs: Vec<VarId> = genctx
        .all_vars()
        .into_iter()
        .filter(|id| genctx.var_type(*id) == CType::Vec)
        .collect();
    if vecs.is_empty() {
        return generate(rng, policy, genctx, depth, false, &None);
    }
    let v = *rng.pick(&vecs);
    let coll = Expr::Var {
        id: v,
        name: genctx.var_name(v).to_string(),
        ir: Ir::new(CType::Vec, genctx.var_value(v).clone(), ExKind::NoEx),
    };
    // The index subtree always executes, so a faulty raw index must be
    // replaced (a default would only mask it). Leaves are always clean.
    let mut idx = generate(rng, policy, genctx, depth + 1, false, &None);
    if !idx.ir().is_ok() {
        idx = leaf(rng, policy, genctx, false, false);
    }
    let items = match genctx.var_value(v) {
        Value::Vec(items) => items.clone(),
        _ => return Expr::Const(types::random_const(rng, policy, CType::Long)),
    };
    let wrap_coll = coll.clone();
    let wrap = |idx: Expr, i: i64| {
        let m = items.len() as i64;
        let j = types::smod_floor(i, m);
        let wrapped = Expr::IdxMod {
            idx: Box::new(idx),
            len: m,
            ir: Ir::long(j),
        };
        Expr::Nth {
            coll: Box::new(wrap_coll.clone()),
            idx: Box::new(wrapped),
            ir: Ir::long(items[j as usize]),
        }
    };
    let i = idx.ir().value.as_long();
    let in_int_range = i >= i64::from(i32::MIN) && i <= i64::from(i32::MAX);
    if idx.ir().is_ok() && in_int_range && i >= 0 && (i as usize) < items.len() {
        let val = items[i as usize];
        Expr::Nth {
            coll: Box::new(coll),
            idx: Box::new(idx),
            ir: Ir::long(val),
        }
    } else if !in_int_range {
        // Out-of-int-range narrows via toIntExact (checked) in BOTH nth
        // arities, so only the mod wrap is sound here.
        wrap(idx, i)
    } else if rng.roll(0.5) {
        let d = types::random_const(rng, policy, CType::Long);
        Expr::NthDefault {
            coll: Box::new(coll),
            idx: Box::new(idx),
            default: Box::new(Expr::Const(d.clone())),
            ir: d,
        }
    } else {
        wrap(idx, i)
    }
}

/// Generate a random expression AST of at most `max_arith_depth`.
pub fn generate(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    depth: i64,
    want_bool: bool,
    pool: &Option<Vec<BinOp>>,
) -> Expr {
    if depth >= policy.max_arith_depth {
        return leaf(rng, policy, genctx, want_bool, pool.is_some());
    }
    match rng.pick_weighted(&policy.node_distr) {
        NodeKind::Leaf => leaf(rng, policy, genctx, want_bool, pool.is_some()),
        NodeKind::Unary => {
            if want_bool {
                leaf(rng, policy, genctx, want_bool, pool.is_some())
            } else {
                gen_unary(rng, policy, genctx, depth, pool)
            }
        }
        NodeKind::Binary => {
            if !want_bool && rng.roll(policy.nth_prob) {
                gen_nth(rng, policy, genctx, depth)
            } else {
                gen_binary(rng, policy, genctx, depth, want_bool, pool)
            }
        }
    }
}

/// Swap a faulting operator for a safer one. Ports the rebuild rules in
/// expr.cpp (ADD<->SUB, MUL->DIV, DIV/MOD ZeroDiv->MUL else SUB,
/// NEGATE->PLUS).
fn rebuild_binop(op: BinOp, ex: ExKind) -> BinOp {
    use BinOp::*;
    if ex == ExKind::ZeroDiv {
        return match op {
            Div | Mod => Mul,
            _ => Sub,
        };
    }
    match op {
        Add => Sub,
        Sub => Add,
        Mul => Div,
        Div | Mod => Sub,
        _ => op,
    }
}

fn rebuild_unop(op: UnOp, _ex: ExKind) -> UnOp {
    match op {
        UnOp::Neg | UnOp::Inc | UnOp::Dec => UnOp::Pos,
        other => other,
    }
}

fn re_eval(expr: Expr) -> Expr {
    match expr {
        Expr::BinOp {
            op, left, right, ..
        } => {
            let ir = types::eval_binop(op, left.ir(), right.ir());
            Expr::BinOp {
                op,
                left,
                right,
                ir,
            }
        }
        Expr::UnOp { op, arg, .. } => {
            let ir = types::eval_unop(op, arg.ir());
            Expr::UnOp { op, arg, ir }
        }
        other => other,
    }
}

/// Single rebuild step: swap the operator, else the caller regenerates.
fn rebuild_node(rng: &mut Rng, policy: &Policy, node: Expr) -> Expr {
    match node {
        Expr::BinOp {
            op,
            left,
            right,
            ir,
        } => {
            let op = rebuild_binop(op, ir.ex);
            re_eval(Expr::BinOp {
                op,
                left,
                right,
                ir,
            })
        }
        Expr::UnOp { op, arg, ir } => {
            let op = rebuild_unop(op, ir.ex);
            re_eval(Expr::UnOp { op, arg, ir })
        }
        // Belt and braces: gen_nth builds clean tops by construction,
        // so a faulty Nth only arrives via a faulty index, for which a
        // default is sound exactly when the index is int-ranged (the
        // common OOB case). Out-of-int-range indices cannot occur here.
        Expr::Nth { coll, idx, .. } => {
            let d = types::random_const(rng, policy, CType::Long);
            Expr::NthDefault {
                coll,
                idx,
                default: Box::new(Expr::Const(d.clone())),
                ir: d,
            }
        }
        other => other,
    }
}

/// Generate an expression whose top-level ExKind is `NoEx`. Rebuilds at
/// depth 0 until clean; falls back to a safe constant. When dead-code
/// exceptions are allowed, the faulty node is kept as-is.
pub fn generate_safe(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    want_bool: bool,
    exc_mode: Density,
) -> Expr {
    let pool = similar_op_pool(rng, policy);
    // Roll the dead-code allowance once per expression: rolling per
    // attempt would make persistently faulty positions near-certain
    // to be allowed under Density::Some.
    let mut allowed: Option<bool> = None;
    let mut node = generate(rng, policy, genctx, 0, want_bool, &pool);
    for _ in 0..MAX_REBUILD_ATTEMPTS {
        if node.ir().is_ok() {
            return node;
        }
        let allow = *allowed.get_or_insert_with(|| allow_exc(rng, policy, genctx.taken, exc_mode));
        if allow {
            return node;
        }
        node = rebuild_node(rng, policy, node);
        if node.ir().is_ok() {
            return node;
        }
        node = generate(rng, policy, genctx, 0, want_bool, &pool);
    }
    if want_bool {
        Expr::Const(Ir::boolean(true))
    } else {
        Expr::Const(Ir::long(0))
    }
}

/// Lower an expression AST to a Clojure s-expression. Names in
/// `atoms` (mutable outputs) are emitted as `(deref name)`.
pub fn to_sexp(node: &Expr, atoms: &std::collections::BTreeSet<String>) -> Sexp {
    match node {
        Expr::Const(ir) => match &ir.value {
            Value::Bool(b) => Sexp::bool(*b),
            _ => Sexp::int(ir.value.as_long()),
        },
        Expr::Var { name, .. } => {
            if atoms.contains(name) {
                Sexp::list(vec![Sexp::sym("deref"), Sexp::sym(name)])
            } else {
                Sexp::sym(name)
            }
        }
        Expr::UnOp { op, arg, .. } => {
            let s = match op {
                UnOp::Neg => "-",
                UnOp::BNot => "bit-not",
                UnOp::Inc => "inc",
                UnOp::Dec => "dec",
                UnOp::Abs => "abs",
                UnOp::Pos => "+",
            };
            Sexp::list(vec![Sexp::sym(s), to_sexp(arg, atoms)])
        }
        Expr::BinOp {
            op, left, right, ..
        } => Sexp::list(vec![
            Sexp::sym(op.sym()),
            to_sexp(left, atoms),
            to_sexp(right, atoms),
        ]),
        Expr::Nth { coll, idx, .. } => Sexp::list(vec![
            Sexp::sym("nth"),
            to_sexp(coll, atoms),
            to_sexp(idx, atoms),
        ]),
        Expr::NthDefault {
            coll, idx, default, ..
        } => Sexp::list(vec![
            Sexp::sym("nth"),
            to_sexp(coll, atoms),
            to_sexp(idx, atoms),
            to_sexp(default, atoms),
        ]),
        Expr::IdxMod { idx, len, .. } => {
            Sexp::list(vec![Sexp::sym("mod"), to_sexp(idx, atoms), Sexp::int(*len)])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_vars() -> GenCtx {
        let mut c = GenCtx::new();
        c.add_output("out-0", CType::Long, Value::Long(5));
        c.add_input("in-0", CType::Long, Value::Long(-3));
        c
    }

    #[test]
    fn safe_exprs_are_clean_across_seeds() {
        for seed in 1..50u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let ctx = ctx_with_vars();
            for want_bool in [false, true] {
                let e = generate_safe(&mut rng, &policy, &ctx, want_bool, Density::None);
                assert!(e.ir().is_ok(), "seed {seed} bool={want_bool}: {:?}", e.ir());
            }
        }
    }

    fn contains_bool(node: &Expr) -> bool {
        if node.ir().typ == CType::Bool {
            return true;
        }
        match node {
            Expr::Const(_) | Expr::Var { .. } => false,
            Expr::UnOp { arg, .. } => contains_bool(arg),
            Expr::BinOp { left, right, .. } => contains_bool(left) || contains_bool(right),
            Expr::Nth { coll, idx, .. } => contains_bool(coll) || contains_bool(idx),
            Expr::NthDefault {
                coll, idx, default, ..
            } => contains_bool(coll) || contains_bool(idx) || contains_bool(default),
            Expr::IdxMod { idx, .. } => contains_bool(idx),
        }
    }

    #[test]
    fn scalar_exprs_contain_no_bools() {
        // Clojure arithmetic on booleans throws; scalar expressions
        // must be bool-free at every node, not just the top.
        for seed in 1..50u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let ctx = ctx_with_vars();
            let e = generate_safe(&mut rng, &policy, &ctx, false, Density::None);
            assert!(!contains_bool(&e), "seed {seed}: bool in scalar expr");
        }
    }

    fn all_clean(node: &Expr) -> bool {
        if !node.ir().is_ok() {
            return false;
        }
        match node {
            Expr::Const(_) | Expr::Var { .. } => true,
            Expr::UnOp { arg, .. } => all_clean(arg),
            Expr::BinOp { left, right, .. } => all_clean(left) && all_clean(right),
            Expr::Nth { coll, idx, .. } => all_clean(coll) && all_clean(idx),
            Expr::NthDefault {
                coll, idx, default, ..
            } => all_clean(coll) && all_clean(idx) && all_clean(default),
            Expr::IdxMod { idx, .. } => all_clean(idx),
        }
    }

    #[test]
    fn safe_exprs_are_clean_at_every_node() {
        // Faults propagate, so a clean top implies a clean tree; pin
        // that directly since runtime executes every node.
        for seed in 1..50u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let ctx = ctx_with_vars();
            for want_bool in [false, true] {
                let e = generate_safe(&mut rng, &policy, &ctx, want_bool, Density::None);
                assert!(all_clean(&e), "seed {seed} bool={want_bool}: faulty node");
            }
        }
    }

    #[test]
    fn index_wrap_emits_floored_mod() {
        use std::collections::BTreeSet;
        let atoms = BTreeSet::new();
        let e = Expr::IdxMod {
            idx: Box::new(Expr::Const(Ir::long(-61))),
            len: 7,
            ir: Ir::long(2),
        };
        assert_eq!(to_sexp(&e, &atoms).flat(), "(mod -61 7)");
    }

    #[test]
    fn emission_derefs_atoms() {
        use std::collections::BTreeSet;
        let atoms: BTreeSet<String> = ["out-0".to_string()].into_iter().collect();
        let ctx = ctx_with_vars();
        let e = Expr::Var {
            id: ctx.id_of("out-0").unwrap(),
            name: "out-0".to_string(),
            ir: Ir::long(1),
        };
        assert_eq!(to_sexp(&e, &atoms).flat(), "(deref out-0)");
        let f = Expr::Var {
            id: ctx.id_of("in-0").unwrap(),
            name: "in-0".to_string(),
            ir: Ir::long(1),
        };
        assert_eq!(to_sexp(&f, &atoms).flat(), "in-0");
    }
}
