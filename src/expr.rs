// Licensed under the Apache License, Version 2.0.
//! Arithmetic expression generator with generation-time evaluation and
//! exception-driven rebuilding. Ports YARPGen's expr.cpp
//! `ArithmeticExpr::create` + propagateType/evaluate/rebuild.

use crate::cli::Density;
use crate::context::{GenCtx, VarId};
use crate::policy::{NodeKind, Policy};
use crate::rng::Rng;
use crate::sexp::Sexp;
use crate::types::{self, BinOp, CType, ExKind, Ir, SeqPred, UnOp, Value};
use std::collections::BTreeMap;

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
    /// Map read. Present keys emit `(get coll :k)` or the
    /// keyword-invoke form `(:k coll)` (same value, different
    /// codegen); absent keys emit `(get coll :k default)`. A bare
    /// miss returns nil, which the model cannot represent, so
    /// absent keys always carry a default.
    Get {
        coll: Box<Expr>,
        key: String,
        default: Option<Box<Expr>>,
        kw: bool,
        ir: Ir,
    },
    /// One persistent update. Emits `(assoc coll :k val)`; the IR is
    /// the whole updated map.
    Assoc {
        coll: Box<Expr>,
        key: String,
        val: Box<Expr>,
        ir: Ir,
    },
    /// Fresh record construction. Emits `(->R0 a b c)` or the
    /// `(map->R0 {:fa a :fb b :fc c})` form (same value, different
    /// constructor codegen); the IR is the whole record. Appears only
    /// as a Get/Assoc collection: records are not scalars.
    Record {
        args: Vec<Expr>,
        map_ctor: bool,
        ir: Ir,
    },
    /// A `(range start end step)` pipeline source. Non-empty by
    /// construction (end > start, step > 0); the IR is the
    /// materialized vector.
    SeqRange {
        start: i64,
        end: i64,
        step: i64,
        ir: Ir,
    },
    /// A realized seq pipeline: stages left to right over the source,
    /// then a scalar terminator. Sources are small (ranges, vecs of
    /// at most 64), every stage is total, and First/Last/Min/Max
    /// forms are gated on non-empty finals — so the node is clean by
    /// construction and loop-invariant (immutable sources, closed
    /// combinators; only take/drop counts re-evaluate).
    SeqPipe {
        source: Box<Expr>,
        stages: Vec<SeqStage>,
        term: SeqTerm,
        ir: Ir,
    },
}

/// One order-preserving pipeline stage. `Take`/`Drop` counts are
/// scalar subtrees (total for every long: negatives clamp); `Doall`
/// forces intermediate realization and is value-transparent.
#[derive(Clone, Debug)]
pub enum SeqStage {
    Map(UnOp),
    Filter(SeqPred),
    Take(Box<Expr>),
    Drop(Box<Expr>),
    Doall,
}

/// How a pipeline becomes a scalar. `Reduce` folds a small constant
/// seed (Mul excluded: products of small ints overflow); `Apply` has
/// no seed, so Min/Max forms only emit over non-empty finals.
#[derive(Clone, Debug)]
pub enum SeqTerm {
    Count,
    First,
    Last,
    Reduce { op: BinOp, init: Box<Expr> },
    Apply(BinOp),
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
            Expr::Get { ir, .. } => ir,
            Expr::Assoc { ir, .. } => ir,
            Expr::Record { ir, .. } => ir,
            Expr::SeqRange { ir, .. } => ir,
            Expr::SeqPipe { ir, .. } => ir,
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
            // positions; fall back to one fixed scalar family so
            // similarity still holds.
            let scalars: Vec<BinOp> = p.iter().copied().filter(|o| !o.is_compare()).collect();
            if scalars.is_empty() {
                *rng.pick(&[BinOp::Add, BinOp::Sub])
            } else {
                *rng.pick(&scalars)
            }
        }
    } else if want_bool {
        *rng.pick(BinOp::compares())
    } else {
        *rng.pick(BinOp::scalar())
    };
    // The pool propagates to the whole subtree (subscript index,
    // map value/default, and seq count/seed trees excepted:
    // gen_nth/gen_get/gen_seq roll those separately).
    let left = generate(rng, policy, genctx, depth + 1, false, pool);
    let right = generate(rng, policy, genctx, depth + 1, false, pool);
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
    // Borrow, never clone: only one element and the length are
    // read, and inputs can span 64K elements.
    let items = match genctx.var_value(v) {
        Value::Vec(items) => items,
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

/// A fresh record construction with clean scalar args in field
/// order. Probed: `(->R0 a b c)` and `(map->R0 {:fa a :fb b :fc c})`
/// agree exactly. Faulty args become leaves (every argument always
/// evaluates), so the node is clean by construction.
fn fresh_record(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    depth: i64,
) -> (Expr, BTreeMap<String, i64>) {
    let mut args = Vec::with_capacity(types::REC_FIELDS.len());
    let mut values = BTreeMap::new();
    for f in types::REC_FIELDS {
        let mut arg = generate(rng, policy, genctx, depth + 1, false, &None);
        if !arg.ir().is_ok() {
            arg = leaf(rng, policy, genctx, false, false);
        }
        values.insert(f.to_string(), arg.ir().value.as_long());
        args.push(arg);
    }
    let ir = Ir::new(CType::Record, Value::Record(values.clone()), ExKind::NoEx);
    (
        Expr::Record {
            args,
            map_ctor: rng.roll(0.5),
            ir,
        },
        values,
    )
}

/// Record read. The collection is a record input or a fresh
/// construction; reads hit present fields via `get` or
/// keyword-invoke. Records are total (all fields always present), so
/// no default form exists; assoc wraps use existing fields only,
/// which provably stay records.
fn gen_record_get(
    rng: &mut Rng,
    policy: &Policy,
    genctx: &GenCtx,
    depth: i64,
    records: &[VarId],
) -> Expr {
    let (base, items) = if rng.roll(0.4) {
        fresh_record(rng, policy, genctx, depth)
    } else {
        let v = *rng.pick(records);
        match genctx.var_value(v) {
            Value::Record(items) => {
                let var = Expr::Var {
                    id: v,
                    name: genctx.var_name(v).to_string(),
                    ir: Ir::new(CType::Record, Value::Record(items.clone()), ExKind::NoEx),
                };
                (var, items.clone())
            }
            _ => fresh_record(rng, policy, genctx, depth),
        }
    };
    let (coll, items) = if rng.roll(0.3) {
        let key = (*rng.pick(&types::REC_FIELDS)).to_string();
        let mut val = generate(rng, policy, genctx, depth + 1, false, &None);
        if !val.ir().is_ok() {
            val = leaf(rng, policy, genctx, false, false);
        }
        let n = val.ir().value.as_long();
        let mut updated = items;
        updated.insert(key.clone(), n);
        let assoc = Expr::Assoc {
            coll: Box::new(base),
            key,
            val: Box::new(val),
            ir: Ir::new(CType::Record, Value::Record(updated.clone()), ExKind::NoEx),
        };
        (assoc, updated)
    } else {
        (base, items)
    };
    // Total by construction (inputs hold all fields, assoc only
    // overwrites), so a present field always exists.
    let present: Vec<String> = items.keys().cloned().collect();
    let key = rng.pick(&present).clone();
    let val = items[&key];
    Expr::Get {
        coll: Box::new(coll),
        key,
        default: None,
        kw: rng.roll(0.5),
        ir: Ir::long(val),
    }
}

/// Map read. Ports the subscript policy to maps: the present key
/// emits `get` or keyword-invoke, the absent key emits `get` with a
/// default, and the collection sometimes passes through one `assoc`
/// first (persistent-update codegen on the read path). Clean by
/// construction: every argument always evaluates, so faulty
/// subtrees are replaced with leaves, never masked.
fn gen_get(rng: &mut Rng, policy: &Policy, genctx: &GenCtx, depth: i64) -> Expr {
    let maps: Vec<VarId> = genctx
        .all_vars()
        .into_iter()
        .filter(|id| genctx.var_type(*id) == CType::Map)
        .collect();
    let records: Vec<VarId> = genctx
        .all_vars()
        .into_iter()
        .filter(|id| genctx.var_type(*id) == CType::Record)
        .collect();
    if maps.is_empty() && records.is_empty() {
        return generate(rng, policy, genctx, depth, false, &None);
    }
    // Record reads with record_prob (share control, and a bisection
    // knob); without map vars, records are the only game.
    if !records.is_empty() && (maps.is_empty() || rng.roll(policy.record_prob)) {
        return gen_record_get(rng, policy, genctx, depth, &records);
    }
    let v = *rng.pick(&maps);
    let items = match genctx.var_value(v) {
        Value::Map(items) => items.clone(),
        _ => return Expr::Const(types::random_const(rng, policy, CType::Long)),
    };
    let base = Expr::Var {
        id: v,
        name: genctx.var_name(v).to_string(),
        ir: Ir::new(CType::Map, Value::Map(items.clone()), ExKind::NoEx),
    };
    let (coll, items) = if rng.roll(0.3) {
        let key = (*rng.pick(&types::MAP_KEYS)).to_string();
        let mut val = generate(rng, policy, genctx, depth + 1, false, &None);
        if !val.ir().is_ok() {
            val = leaf(rng, policy, genctx, false, false);
        }
        let n = val.ir().value.as_long();
        let mut updated = items;
        updated.insert(key.clone(), n);
        let assoc = Expr::Assoc {
            coll: Box::new(base),
            key,
            val: Box::new(val),
            ir: Ir::new(CType::Map, Value::Map(updated.clone()), ExKind::NoEx),
        };
        (assoc, updated)
    } else {
        (base, items)
    };
    let absent: Vec<&str> = types::MAP_KEYS
        .iter()
        .copied()
        .filter(|k| !items.contains_key(*k))
        .collect();
    if !absent.is_empty() && rng.roll(0.3) {
        let key = (*rng.pick(&absent)).to_string();
        let mut default = generate(rng, policy, genctx, depth + 1, false, &None);
        if !default.ir().is_ok() {
            default = leaf(rng, policy, genctx, false, false);
        }
        let ir = default.ir().clone();
        Expr::Get {
            coll: Box::new(coll),
            key,
            default: Some(Box::new(default)),
            kw: false,
            ir,
        }
    } else {
        // Map inputs always hold 2-4 keys and assoc only adds, so a
        // present key always exists.
        let present: Vec<String> = items.keys().cloned().collect();
        let key = rng.pick(&present).clone();
        let val = items[&key];
        Expr::Get {
            coll: Box::new(coll),
            key,
            default: None,
            kw: rng.roll(0.5),
            ir: Ir::long(val),
        }
    }
}

/// Run pipeline stages over materialized items. Shared by
/// construction (evaluator reads stored IRs) and live simulation
/// (evaluator re-evaluates): one implementation, no drift. Element
/// and count faults propagate with their kind.
pub(crate) fn run_seq_stages(
    items: Vec<i64>,
    stages: &[SeqStage],
    eval: &dyn Fn(&Expr) -> Ir,
) -> Result<Vec<i64>, ExKind> {
    let mut items = items;
    for s in stages {
        match s {
            SeqStage::Map(op) => {
                let mut out = Vec::with_capacity(items.len());
                for el in items {
                    let r = types::eval_unop(*op, &Ir::long(el));
                    if !r.is_ok() {
                        return Err(r.ex);
                    }
                    out.push(r.value.as_long());
                }
                items = out;
            }
            SeqStage::Filter(p) => items.retain(|el| p.eval(*el)),
            SeqStage::Take(n) => {
                let k = seq_count(n, eval)?;
                items.truncate(k.clamp(0, items.len() as i64) as usize);
            }
            SeqStage::Drop(n) => {
                let k = seq_count(n, eval)?;
                let d = k.clamp(0, items.len() as i64) as usize;
                items.drain(..d);
            }
            SeqStage::Doall => {}
        }
    }
    Ok(items)
}

fn seq_count(n: &Expr, eval: &dyn Fn(&Expr) -> Ir) -> Result<i64, ExKind> {
    let nv = eval(n);
    if nv.is_ok() {
        Ok(nv.value.as_long())
    } else {
        Err(nv.ex)
    }
}

/// Apply a scalar terminator to final items. First/Last and
/// Min/Max-apply over an empty final fault Oob: unreachable by
/// construction (gated), so strict simulation rejects rather than
/// inventing the nil the runtime would produce.
pub(crate) fn apply_seq_term(items: &[i64], term: &SeqTerm, eval: &dyn Fn(&Expr) -> Ir) -> Ir {
    let oob = || Ir::new(CType::Long, Value::Long(0), ExKind::Oob);
    match term {
        SeqTerm::Count => Ir::long(items.len() as i64),
        SeqTerm::First => items.first().map_or_else(oob, |n| Ir::long(*n)),
        SeqTerm::Last => items.last().map_or_else(oob, |n| Ir::long(*n)),
        SeqTerm::Reduce { op, init } => {
            let seed = eval(init);
            if !seed.is_ok() {
                return Ir::new(CType::Long, Value::Long(0), seed.ex);
            }
            let mut acc = seed;
            for el in items {
                acc = types::eval_binop(*op, &acc, &Ir::long(*el));
                if !acc.is_ok() {
                    return Ir::new(CType::Long, Value::Long(0), acc.ex);
                }
            }
            acc
        }
        SeqTerm::Apply(op) => {
            if items.is_empty() {
                // `(apply + [])` is `(+)` is 0 (probed); every other
                // op needs an element (min/max of nothing throw).
                if *op == BinOp::Add {
                    return Ir::long(0);
                }
                return oob();
            }
            let mut it = items.iter();
            // Non-empty, checked above.
            let mut acc = Ir::long(*it.next().unwrap());
            for el in it {
                acc = types::eval_binop(*op, &acc, &Ir::long(*el));
                if !acc.is_ok() {
                    return Ir::new(CType::Long, Value::Long(0), acc.ex);
                }
            }
            acc
        }
    }
}

/// Sub-expressions buried in stages/terms (take/drop counts, reduce
/// seeds): the walker/cost/collector recursion set.
pub(crate) fn stage_subexprs(s: &SeqStage) -> Vec<&Expr> {
    match s {
        SeqStage::Take(n) | SeqStage::Drop(n) => vec![n],
        _ => vec![],
    }
}

pub(crate) fn term_subexprs(t: &SeqTerm) -> Vec<&Expr> {
    match t {
        SeqTerm::Reduce { init, .. } => vec![init],
        _ => vec![],
    }
}

/// Maximum vector-input length a pipeline may realize. Bounds model
/// cost (a 64K input would charge 64K evaluations per simulation
/// step); larger vectors fall back to range sources.
const MAX_SEQ_SOURCE_LEN: usize = 64;

/// A scalar subtree that always evaluates: raw generation with a
/// faulty root replaced by a leaf (leaves are always clean).
fn clean_subexpr(rng: &mut Rng, policy: &Policy, genctx: &GenCtx, depth: i64) -> Expr {
    let mut e = generate(rng, policy, genctx, depth + 1, false, &None);
    if !e.ir().is_ok() {
        e = leaf(rng, policy, genctx, false, false);
    }
    e
}

/// A `(range start end step)` source with at most ~48 elements: big
/// enough to cross seq-chunk boundaries, tiny for the model.
fn seq_range_source(rng: &mut Rng) -> (Expr, Vec<i64>) {
    let start = rng.gen_range(0, 8);
    let end = rng.gen_range(start + 2, 48);
    let step = rng.gen_range(1, 3);
    let mut items = Vec::new();
    let mut i = start;
    while i < end {
        items.push(i);
        i += step;
    }
    let ir = Ir::new(CType::Vec, Value::Vec(items.clone()), ExKind::NoEx);
    (
        Expr::SeqRange {
            start,
            end,
            step,
            ir,
        },
        items,
    )
}

/// A scalar terminator, gated on the final items: First/Last need an
/// element, Min/Max-apply need one too (else Add's identity).
fn pick_seq_term(rng: &mut Rng, final_items: &[i64]) -> SeqTerm {
    match rng.gen_usize(5) {
        0 => SeqTerm::Count,
        1 if !final_items.is_empty() => SeqTerm::First,
        2 if !final_items.is_empty() => SeqTerm::Last,
        3 => SeqTerm::Reduce {
            op: *rng.pick(&[BinOp::Add, BinOp::Min, BinOp::Max]),
            init: Box::new(Expr::Const(Ir::long(rng.gen_range(-10, 10)))),
        },
        4 => {
            let op = *rng.pick(&[BinOp::Add, BinOp::Min, BinOp::Max]);
            SeqTerm::Apply(if final_items.is_empty() && op != BinOp::Add {
                BinOp::Add
            } else {
                op
            })
        }
        _ => SeqTerm::Count,
    }
}

/// A realized seq pipeline over a small vector input or a range
/// literal: 1-3 order-preserving stages plus a scalar terminator.
/// Total by construction (bounded values, clamped counts, gated
/// empties), so the node is always clean and loop-invariant.
fn gen_seq(rng: &mut Rng, policy: &Policy, genctx: &GenCtx, depth: i64) -> Expr {
    let small_vecs: Vec<VarId> = genctx
        .all_vars()
        .into_iter()
        .filter(|id| {
            genctx.var_type(*id) == CType::Vec
                && matches!(genctx.var_value(*id), Value::Vec(items) if items.len() <= MAX_SEQ_SOURCE_LEN)
        })
        .collect();
    let (source, items) = if !small_vecs.is_empty() && rng.roll(0.5) {
        let v = *rng.pick(&small_vecs);
        match genctx.var_value(v) {
            Value::Vec(items) => {
                let var = Expr::Var {
                    id: v,
                    name: genctx.var_name(v).to_string(),
                    ir: Ir::new(CType::Vec, Value::Vec(items.clone()), ExKind::NoEx),
                };
                (var, items.clone())
            }
            _ => seq_range_source(rng),
        }
    } else {
        seq_range_source(rng)
    };
    let mut stages = Vec::new();
    for _ in 0..rng.gen_range(1, 3) {
        match rng.gen_usize(4) {
            0 => stages.push(SeqStage::Map(*rng.pick(&[
                UnOp::Neg,
                UnOp::BNot,
                UnOp::Inc,
                UnOp::Dec,
                UnOp::Abs,
            ]))),
            1 => stages.push(SeqStage::Filter(*rng.pick(&[
                SeqPred::Odd,
                SeqPred::Even,
                SeqPred::Pos,
                SeqPred::Neg,
                SeqPred::Zero,
            ]))),
            2 => stages.push(SeqStage::Take(Box::new(clean_subexpr(
                rng, policy, genctx, depth,
            )))),
            _ => stages.push(SeqStage::Drop(Box::new(clean_subexpr(
                rng, policy, genctx, depth,
            )))),
        }
    }
    if rng.roll(0.4) {
        stages.insert(rng.gen_usize(stages.len() + 1), SeqStage::Doall);
    }
    let (term, ir) = match run_seq_stages(items, &stages, &|e| e.ir().clone()) {
        // Unreachable: stages are total over bounded values and
        // counts are leaf-clean. Emit a valid Count node carrying
        // the fault so generate_safe regenerates (or dead code keeps
        // a well-formed tree).
        Err(ex) => (SeqTerm::Count, Ir::new(CType::Long, Value::Long(0), ex)),
        Ok(final_items) => {
            let term = pick_seq_term(rng, &final_items);
            let ir = apply_seq_term(&final_items, &term, &|e| e.ir().clone());
            (term, ir)
        }
    };
    Expr::SeqPipe {
        source: Box::new(source),
        stages,
        term,
        ir,
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
    // A pooled top-level leaf is a constant; below the top, pooled
    // subtrees may reference variables (the pool constrains operator
    // choice, not leaf kind).
    let top_const = pool.is_some() && depth == 0;
    if depth >= policy.max_arith_depth {
        return leaf(rng, policy, genctx, want_bool, top_const);
    }
    match rng.pick_weighted(&policy.node_distr) {
        NodeKind::Leaf => leaf(rng, policy, genctx, want_bool, top_const),
        NodeKind::Unary => {
            if want_bool {
                leaf(rng, policy, genctx, want_bool, top_const)
            } else {
                gen_unary(rng, policy, genctx, depth, pool)
            }
        }
        NodeKind::Binary => {
            if !want_bool && rng.roll(policy.nth_prob) {
                gen_nth(rng, policy, genctx, depth)
            } else if !want_bool && rng.roll(policy.map_prob) {
                gen_get(rng, policy, genctx, depth)
            } else if !want_bool && rng.roll(policy.seq_prob) {
                gen_seq(rng, policy, genctx, depth)
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
fn rebuild_node(node: Expr) -> Expr {
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
        // Unreachable in practice: gen_nth replaces faulty indices
        // with clean leaves, so Nth tops are clean by construction and
        // only BinOp/UnOp nodes arrive here faulty. Pass through
        // unchanged (the caller regenerates) rather than converting
        // to a default, which would be unsound for out-of-int-range
        // indices: Clojure narrows via toIntExact in both arities.
        Expr::Nth { coll, idx, ir } => Expr::Nth { coll, idx, ir },
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
        node = rebuild_node(node);
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
        Expr::UnOp { op, arg, .. } => Sexp::list(vec![Sexp::sym(op.sym()), to_sexp(arg, atoms)]),
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
        Expr::Get {
            coll,
            key,
            default,
            kw,
            ..
        } => {
            let k = Sexp::sym(&format!(":{key}"));
            if *kw {
                Sexp::list(vec![k, to_sexp(coll, atoms)])
            } else if let Some(d) = default {
                Sexp::list(vec![
                    Sexp::sym("get"),
                    to_sexp(coll, atoms),
                    k,
                    to_sexp(d, atoms),
                ])
            } else {
                Sexp::list(vec![Sexp::sym("get"), to_sexp(coll, atoms), k])
            }
        }
        Expr::Assoc { coll, key, val, .. } => Sexp::list(vec![
            Sexp::sym("assoc"),
            to_sexp(coll, atoms),
            Sexp::sym(&format!(":{key}")),
            to_sexp(val, atoms),
        ]),
        Expr::Record { args, map_ctor, .. } => {
            // Args run in REC_FIELDS order, matching the declared
            // field vector positionally.
            if *map_ctor {
                let mut flat = Vec::with_capacity(args.len() * 2);
                for (f, a) in types::REC_FIELDS.iter().zip(args.iter()) {
                    flat.push(Sexp::sym(&format!(":{f}")));
                    flat.push(to_sexp(a, atoms));
                }
                Sexp::list(vec![
                    Sexp::sym(&format!("map->{}", types::RECORD_NAME)),
                    Sexp::map(flat),
                ])
            } else {
                let mut items = vec![Sexp::sym(&format!("->{}", types::RECORD_NAME))];
                items.extend(args.iter().map(|a| to_sexp(a, atoms)));
                Sexp::list(items)
            }
        }
        Expr::SeqRange {
            start, end, step, ..
        } => Sexp::list(vec![
            Sexp::sym("range"),
            Sexp::int(*start),
            Sexp::int(*end),
            Sexp::int(*step),
        ]),
        Expr::SeqPipe {
            source,
            stages,
            term,
            ..
        } => {
            let mut form = to_sexp(source, atoms);
            for s in stages {
                form = match s {
                    SeqStage::Map(op) => {
                        Sexp::list(vec![Sexp::sym("map"), Sexp::sym(op.sym()), form])
                    }
                    SeqStage::Filter(p) => {
                        Sexp::list(vec![Sexp::sym("filter"), Sexp::sym(p.sym()), form])
                    }
                    SeqStage::Take(n) => {
                        Sexp::list(vec![Sexp::sym("take"), to_sexp(n, atoms), form])
                    }
                    SeqStage::Drop(n) => {
                        Sexp::list(vec![Sexp::sym("drop"), to_sexp(n, atoms), form])
                    }
                    SeqStage::Doall => Sexp::list(vec![Sexp::sym("doall"), form]),
                };
            }
            match term {
                SeqTerm::Count => Sexp::list(vec![Sexp::sym("count"), form]),
                SeqTerm::First => Sexp::list(vec![Sexp::sym("first"), form]),
                SeqTerm::Last => Sexp::list(vec![Sexp::sym("last"), form]),
                SeqTerm::Reduce { op, init } => Sexp::list(vec![
                    Sexp::sym("reduce"),
                    Sexp::sym(op.sym()),
                    to_sexp(init, atoms),
                    form,
                ]),
                SeqTerm::Apply(op) => {
                    Sexp::list(vec![Sexp::sym("apply"), Sexp::sym(op.sym()), form])
                }
            }
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
            Expr::Get { coll, default, .. } => {
                contains_bool(coll) || default.as_ref().is_some_and(|d| contains_bool(d))
            }
            Expr::Assoc { coll, val, .. } => contains_bool(coll) || contains_bool(val),
            Expr::Record { args, .. } => args.iter().any(contains_bool),
            Expr::SeqRange { .. } => false,
            Expr::SeqPipe {
                source,
                stages,
                term,
                ..
            } => {
                contains_bool(source)
                    || stages.iter().flat_map(stage_subexprs).any(contains_bool)
                    || term_subexprs(term).iter().any(|e| contains_bool(e))
            }
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
            Expr::Get { coll, default, .. } => {
                all_clean(coll) && default.as_ref().is_none_or(|d| all_clean(d))
            }
            Expr::Assoc { coll, val, .. } => all_clean(coll) && all_clean(val),
            Expr::Record { args, .. } => args.iter().all(all_clean),
            Expr::SeqRange { .. } => true,
            Expr::SeqPipe {
                source,
                stages,
                term,
                ..
            } => {
                all_clean(source)
                    && stages.iter().flat_map(stage_subexprs).all(all_clean)
                    && term_subexprs(term).iter().all(|e| all_clean(e))
            }
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
    fn rebuild_passes_nth_through_unchanged() {
        // gen_nth tops are clean by construction, so rebuild never
        // sees a faulty Nth; if it ever does, passthrough (the caller
        // regenerates) is the sound answer, since a default would not
        // save an out-of-int-range index.
        let nth = Expr::Nth {
            coll: Box::new(Expr::Const(Ir::new(
                CType::Vec,
                Value::Vec(vec![1]),
                ExKind::NoEx,
            ))),
            idx: Box::new(Expr::Const(Ir::long(1 << 40))),
            ir: Ir::new(CType::Long, Value::Long(0), ExKind::Oob),
        };
        match rebuild_node(nth) {
            Expr::Nth { ir, .. } => assert_eq!(ir.ex, ExKind::Oob),
            other => panic!("rebuild must not convert Nth, got {other:?}"),
        }
    }

    fn collect_binops(node: &Expr, out: &mut Vec<BinOp>) {
        // Subscript index trees and map value/default/ctor/count
        // trees roll outside the pool; skip them.
        match node {
            Expr::BinOp {
                op, left, right, ..
            } => {
                out.push(*op);
                collect_binops(left, out);
                collect_binops(right, out);
            }
            Expr::UnOp { arg, .. } => collect_binops(arg, out),
            Expr::Const(_) | Expr::Var { .. } => {}
            Expr::Nth { .. }
            | Expr::NthDefault { .. }
            | Expr::IdxMod { .. }
            | Expr::Get { .. }
            | Expr::Assoc { .. }
            | Expr::Record { .. }
            | Expr::SeqRange { .. }
            | Expr::SeqPipe { .. } => {}
        }
    }

    #[test]
    fn similar_pool_restricts_whole_subtree() {
        // A similar-operators pool restricts operator choice in the
        // whole subtree (subscript index trees excepted). Compare
        // pools in scalar positions fall back to Add/Sub so no
        // boolean leaks into arithmetic. Tested on generate directly:
        // post-hoc rebuild swaps may legitimately leave the family.
        let policy = Policy::default();
        let mut saw_ops = false;
        for seed in 1..20u64 {
            let mut rng = Rng::new(seed);
            let mut ctx = ctx_with_vars();
            // Without a vector/map, gen_nth/gen_get take their
            // unpooled fallback paths; the real paths roll index and
            // value trees outside the pool (skipped by the
            // collector) instead.
            ctx.add_input("v-0", CType::Vec, Value::Vec(vec![7, -2, 5]));
            ctx.add_input(
                "m-0",
                CType::Map,
                Value::Map(
                    [("ka".to_string(), 7), ("kb".to_string(), -2)]
                        .into_iter()
                        .collect(),
                ),
            );
            for fam in types::op_families() {
                let compares_only = fam.iter().all(|o| o.is_compare());
                let pool = Some(fam.clone());
                // Scalar tree: pool family, or Add/Sub for a compare pool.
                let e = generate(&mut rng, &policy, &ctx, 0, false, &pool);
                let mut ops = Vec::new();
                collect_binops(&e, &mut ops);
                saw_ops |= !ops.is_empty();
                let allowed: Vec<BinOp> = if compares_only {
                    vec![BinOp::Add, BinOp::Sub]
                } else {
                    fam.clone()
                };
                assert!(
                    ops.iter().all(|o| allowed.contains(o)),
                    "seed {seed} scalar: {ops:?} escapes {allowed:?}"
                );
                // Boolean tree: top compare with scalar Add/Sub below
                // for a compare pool, else the pool family throughout.
                let e = generate(&mut rng, &policy, &ctx, 0, true, &pool);
                let mut ops = Vec::new();
                collect_binops(&e, &mut ops);
                saw_ops |= !ops.is_empty();
                let ok = if compares_only {
                    ops.iter()
                        .all(|o| o.is_compare() || *o == BinOp::Add || *o == BinOp::Sub)
                } else {
                    ops.iter().all(|o| fam.contains(o))
                };
                assert!(ok, "seed {seed} bool: {ops:?} escapes pool");
            }
        }
        assert!(saw_ops, "no binops generated at all");
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

    fn map_var(ctx: &GenCtx) -> Expr {
        let id = ctx.id_of("map-0").unwrap();
        Expr::Var {
            id,
            name: "map-0".to_string(),
            ir: Ir::new(CType::Map, ctx.var_value(id).clone(), ExKind::NoEx),
        }
    }

    #[test]
    fn map_forms_emit_get_invoke_default_and_assoc() {
        use std::collections::BTreeMap;
        use std::collections::BTreeSet;
        let atoms = BTreeSet::new();
        let mut ctx = ctx_with_vars();
        let items: BTreeMap<String, i64> = [("ka".to_string(), 1), ("kb".to_string(), 2)]
            .into_iter()
            .collect();
        ctx.add_input("map-0", CType::Map, Value::Map(items));
        let get = Expr::Get {
            coll: Box::new(map_var(&ctx)),
            key: "ka".to_string(),
            default: None,
            kw: false,
            ir: Ir::long(1),
        };
        assert_eq!(to_sexp(&get, &atoms).flat(), "(get map-0 :ka)");
        let kw = Expr::Get {
            coll: Box::new(map_var(&ctx)),
            key: "kb".to_string(),
            default: None,
            kw: true,
            ir: Ir::long(2),
        };
        assert_eq!(to_sexp(&kw, &atoms).flat(), "(:kb map-0)");
        let miss = Expr::Get {
            coll: Box::new(map_var(&ctx)),
            key: "kc".to_string(),
            default: Some(Box::new(Expr::Const(Ir::long(9)))),
            kw: false,
            ir: Ir::long(9),
        };
        assert_eq!(to_sexp(&miss, &atoms).flat(), "(get map-0 :kc 9)");
        let assoc = Expr::Assoc {
            coll: Box::new(map_var(&ctx)),
            key: "kc".to_string(),
            val: Box::new(Expr::Const(Ir::long(7))),
            ir: Ir::new(CType::Map, Value::Map(BTreeMap::new()), ExKind::NoEx),
        };
        assert_eq!(to_sexp(&assoc, &atoms).flat(), "(assoc map-0 :kc 7)");
    }

    fn contains_get(node: &Expr) -> bool {
        match node {
            Expr::Get { .. } | Expr::Assoc { .. } => true,
            Expr::Const(_) | Expr::Var { .. } => false,
            Expr::UnOp { arg, .. } => contains_get(arg),
            Expr::BinOp { left, right, .. } => contains_get(left) || contains_get(right),
            Expr::Nth { coll, idx, .. } => contains_get(coll) || contains_get(idx),
            Expr::NthDefault {
                coll, idx, default, ..
            } => contains_get(coll) || contains_get(idx) || contains_get(default),
            Expr::IdxMod { idx, .. } => contains_get(idx),
            Expr::Record { args, .. } => args.iter().any(contains_get),
            Expr::SeqRange { .. } => false,
            Expr::SeqPipe {
                source,
                stages,
                term,
                ..
            } => {
                contains_get(source)
                    || stages.iter().flat_map(stage_subexprs).any(contains_get)
                    || term_subexprs(term).iter().any(|e| contains_get(e))
            }
        }
    }

    fn contains_record(node: &Expr) -> bool {
        match node {
            Expr::Record { .. } => true,
            Expr::Const(_) | Expr::Var { .. } => false,
            Expr::UnOp { arg, .. } => contains_record(arg),
            Expr::BinOp { left, right, .. } => contains_record(left) || contains_record(right),
            Expr::Nth { coll, idx, .. } => contains_record(coll) || contains_record(idx),
            Expr::NthDefault {
                coll, idx, default, ..
            } => contains_record(coll) || contains_record(idx) || contains_record(default),
            Expr::IdxMod { idx, .. } => contains_record(idx),
            Expr::Get { coll, default, .. } => {
                contains_record(coll) || default.as_ref().is_some_and(|d| contains_record(d))
            }
            Expr::Assoc { coll, val, .. } => contains_record(coll) || contains_record(val),
            Expr::SeqRange { .. } => false,
            Expr::SeqPipe {
                source,
                stages,
                term,
                ..
            } => {
                contains_record(source)
                    || stages.iter().flat_map(stage_subexprs).any(contains_record)
                    || term_subexprs(term).iter().any(|e| contains_record(e))
            }
        }
    }

    fn contains_seq(node: &Expr) -> bool {
        match node {
            Expr::SeqRange { .. } | Expr::SeqPipe { .. } => true,
            Expr::Const(_) | Expr::Var { .. } => false,
            Expr::UnOp { arg, .. } => contains_seq(arg),
            Expr::BinOp { left, right, .. } => contains_seq(left) || contains_seq(right),
            Expr::Nth { coll, idx, .. } => contains_seq(coll) || contains_seq(idx),
            Expr::NthDefault {
                coll, idx, default, ..
            } => contains_seq(coll) || contains_seq(idx) || contains_seq(default),
            Expr::IdxMod { idx, .. } => contains_seq(idx),
            Expr::Get { coll, default, .. } => {
                contains_seq(coll) || default.as_ref().is_some_and(|d| contains_seq(d))
            }
            Expr::Assoc { coll, val, .. } => contains_seq(coll) || contains_seq(val),
            Expr::Record { args, .. } => args.iter().any(contains_seq),
        }
    }

    #[test]
    fn map_reads_are_clean_across_seeds() {
        // With map inputs present, generation must actually emit
        // reads (else this is vacuous) and every tree stays clean.
        let mut saw_get = false;
        for seed in 1..100u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let mut ctx = ctx_with_vars();
            let m = types::random_map(&mut rng);
            ctx.add_input("map-0", CType::Map, m.value.clone());
            let e = generate_safe(&mut rng, &policy, &ctx, false, Density::None);
            assert!(e.ir().is_ok(), "seed {seed}: {:?}", e.ir());
            assert!(all_clean(&e), "seed {seed}: faulty node under Get");
            assert!(!contains_bool(&e), "seed {seed}: bool under Get");
            saw_get |= contains_get(&e);
        }
        assert!(saw_get, "no map reads generated in 99 seeds");
    }

    #[test]
    fn record_ctors_emit_positional_and_map_forms() {
        use std::collections::BTreeSet;
        let atoms = BTreeSet::new();
        let args = || {
            vec![
                Expr::Const(Ir::long(1)),
                Expr::Const(Ir::long(2)),
                Expr::Const(Ir::long(3)),
            ]
        };
        let pos = Expr::Record {
            args: args(),
            map_ctor: false,
            ir: Ir::long(0),
        };
        assert_eq!(to_sexp(&pos, &atoms).flat(), "(->R0 1 2 3)");
        let map = Expr::Record {
            args: args(),
            map_ctor: true,
            ir: Ir::long(0),
        };
        assert_eq!(
            to_sexp(&map, &atoms).flat(),
            "(map->R0 {:fa 1 :fb 2 :fc 3})"
        );
    }

    #[test]
    fn record_reads_are_clean_across_seeds() {
        // With a record input present, generation must actually emit
        // fresh constructions (else this is vacuous) and every tree
        // stays clean and bool-free.
        let mut saw_record = false;
        let mut saw_get = false;
        for seed in 1..100u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let mut ctx = ctx_with_vars();
            let r = types::random_record(&mut rng);
            ctx.add_input("rec-0", CType::Record, r.value.clone());
            let e = generate_safe(&mut rng, &policy, &ctx, false, Density::None);
            assert!(e.ir().is_ok(), "seed {seed}: {:?}", e.ir());
            assert!(all_clean(&e), "seed {seed}: faulty node under record");
            assert!(!contains_bool(&e), "seed {seed}: bool under record");
            saw_record |= contains_record(&e);
            saw_get |= contains_get(&e);
        }
        assert!(saw_record, "no record constructions in 99 seeds");
        assert!(saw_get, "no record reads generated in 99 seeds");
    }

    #[test]
    fn seq_pipes_emit_nested_stages() {
        use std::collections::BTreeSet;
        let atoms = BTreeSet::new();
        let src = Expr::SeqRange {
            start: 0,
            end: 6,
            step: 2,
            ir: Ir::long(0),
        };
        let pipe = Expr::SeqPipe {
            source: Box::new(src),
            stages: vec![
                SeqStage::Map(UnOp::Inc),
                SeqStage::Filter(SeqPred::Odd),
                SeqStage::Take(Box::new(Expr::Const(Ir::long(2)))),
            ],
            term: SeqTerm::Reduce {
                op: BinOp::Add,
                init: Box::new(Expr::Const(Ir::long(5))),
            },
            ir: Ir::long(0),
        };
        assert_eq!(
            to_sexp(&pipe, &atoms).flat(),
            "(reduce + 5 (take 2 (filter odd? (map inc (range 0 6 2)))))"
        );
        let applied = Expr::SeqPipe {
            source: Box::new(Expr::SeqRange {
                start: 0,
                end: 3,
                step: 1,
                ir: Ir::long(0),
            }),
            stages: vec![SeqStage::Doall],
            term: SeqTerm::Apply(BinOp::Min),
            ir: Ir::long(0),
        };
        assert_eq!(
            to_sexp(&applied, &atoms).flat(),
            "(apply min (doall (range 0 3 1)))"
        );
    }

    #[test]
    fn seq_pipes_are_clean_across_seeds() {
        // Pipelines need no inputs at all (range sources always
        // work), so generation must emit them and every tree stays
        // clean and bool-free.
        let mut saw_seq = false;
        for seed in 1..100u64 {
            let mut rng = Rng::new(seed);
            let policy = Policy::default();
            let ctx = ctx_with_vars();
            let e = generate_safe(&mut rng, &policy, &ctx, false, Density::None);
            assert!(e.ir().is_ok(), "seed {seed}: {:?}", e.ir());
            assert!(all_clean(&e), "seed {seed}: faulty node under SeqPipe");
            assert!(!contains_bool(&e), "seed {seed}: bool under SeqPipe");
            saw_seq |= contains_seq(&e);
        }
        assert!(saw_seq, "no seq pipelines generated in 99 seeds");
    }

    #[test]
    fn seq_runner_matches_hand_values() {
        // The shared construction/simulation runner, direct: map,
        // filter, take, and every terminator shape.
        let run = |items: Vec<i64>, stages: Vec<SeqStage>, term: SeqTerm| {
            run_seq_stages(items, &stages, &|e| e.ir().clone())
                .map(|fin| apply_seq_term(&fin, &term, &|e| e.ir().clone()))
        };
        let c = |n: i64| Box::new(Expr::Const(Ir::long(n)));
        // (reduce + 5 (take 2 (filter odd? (map inc [0 1 2 3]))))
        // = 5 + (1+3) = 9.
        let stages = vec![
            SeqStage::Map(UnOp::Inc),
            SeqStage::Filter(SeqPred::Odd),
            SeqStage::Take(c(2)),
        ];
        let term = SeqTerm::Reduce {
            op: BinOp::Add,
            init: c(5),
        };
        assert_eq!(
            run(vec![0, 1, 2, 3], stages, term).unwrap().value,
            Value::Long(9)
        );
        // Empty finals: count is 0, first/last fault Oob (gated away
        // at construction), apply-+ is its identity.
        let drop_all = vec![SeqStage::Drop(c(99))];
        let v = run(vec![1], drop_all.clone(), SeqTerm::Count).unwrap();
        assert_eq!(v.value, Value::Long(0));
        let f = run(vec![1], drop_all.clone(), SeqTerm::First).unwrap();
        assert_eq!(f.ex, ExKind::Oob);
        let a = run(vec![1], drop_all, SeqTerm::Apply(BinOp::Add)).unwrap();
        assert_eq!(a.value, Value::Long(0));
        // Faulty take counts propagate with their kind.
        let bad = Expr::BinOp {
            op: BinOp::Div,
            left: Box::new(Expr::Const(Ir::long(1))),
            right: Box::new(Expr::Const(Ir::long(0))),
            ir: Ir::new(CType::Long, Value::Long(0), ExKind::ZeroDiv),
        };
        let err = run_seq_stages(vec![1, 2], &[SeqStage::Take(Box::new(bad))], &|e| {
            e.ir().clone()
        });
        assert_eq!(err, Err(ExKind::ZeroDiv));
    }
}
