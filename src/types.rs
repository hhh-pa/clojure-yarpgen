// Licensed under the Apache License, Version 2.0.
//! Value model: types, ranges, and the exception-kind lattice.
//! Ports YARPGen's data.h/type.h/ir_value.h. The no-UB guarantee becomes
//! a deterministic+exception-free guarantee: every generated expression is
//! evaluated at generation time and rebuilt until its ExKind is `NoEx`.

use crate::policy::Policy;
use crate::rng::Rng;

/// Exception kinds. Port of the reachable subset of YARPGen's `UBKind`
/// enum. The Clojure subset needs no shift/cast/recur faults: shift
/// amounts are clamped at generation, and emitted code contains no
/// casts or `recur` (those live in the negative suite instead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExKind {
    NoEx,
    Uninit,
    ArithOvf,
    ZeroDiv,
    Oob,
}

/// Scalar types plus generation-time vectors. All integrals are emitted
/// as Clojure longs; the width drives special constants and overflow
/// detection so runtime promotion to BigInt never happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CType {
    Bool,
    Int,
    UInt,
    Long,
    ULong,
    Vec,
}

impl CType {
    pub fn range(self) -> (i64, i64) {
        match self {
            CType::Bool => (0, 1),
            CType::Int => (i32::MIN as i64, i32::MAX as i64),
            CType::UInt => (0, u32::MAX as i64),
            CType::Long => (i64::MIN, i64::MAX),
            CType::ULong => (0, i64::MAX),
            CType::Vec => (0, 0),
        }
    }

    pub fn is_scalar(self) -> bool {
        !matches!(self, CType::Vec | CType::Bool)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Long(i64),
    Bool(bool),
    Vec(Vec<i64>),
}

impl Value {
    pub fn as_long(&self) -> i64 {
        match self {
            Value::Long(n) => *n,
            Value::Bool(b) => i64::from(*b),
            Value::Vec(v) => v.len() as i64,
        }
    }
}

/// A type + value + exception-kind triple. Port of `IRValue`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ir {
    pub typ: CType,
    pub value: Value,
    pub ex: ExKind,
}

impl Ir {
    pub fn new(typ: CType, value: Value, ex: ExKind) -> Self {
        Ir { typ, value, ex }
    }
    pub fn long(n: i64) -> Self {
        Ir::new(CType::Long, Value::Long(n), ExKind::NoEx)
    }
    pub fn boolean(b: bool) -> Self {
        Ir::new(CType::Bool, Value::Bool(b), ExKind::NoEx)
    }
    fn ovf() -> Self {
        Ir::new(CType::Long, Value::Long(0), ExKind::ArithOvf)
    }
    pub fn is_ok(&self) -> bool {
        self.ex == ExKind::NoEx
    }
}

fn fits_long(n: i128) -> bool {
    n >= i128::from(i64::MIN) && n <= i128::from(i64::MAX)
}

pub fn sadd(a: &Ir, b: &Ir) -> Ir {
    let r = a.value.as_long() as i128 + b.value.as_long() as i128;
    if fits_long(r) {
        Ir::long(r as i64)
    } else {
        Ir::ovf()
    }
}

pub fn ssub(a: &Ir, b: &Ir) -> Ir {
    let r = a.value.as_long() as i128 - b.value.as_long() as i128;
    if fits_long(r) {
        Ir::long(r as i64)
    } else {
        Ir::ovf()
    }
}

pub fn smul(a: &Ir, b: &Ir) -> Ir {
    let r = a.value.as_long() as i128 * b.value.as_long() as i128;
    if fits_long(r) {
        Ir::long(r as i64)
    } else {
        Ir::ovf()
    }
}

pub fn sdiv(a: &Ir, b: &Ir) -> Ir {
    let (x, y) = (a.value.as_long(), b.value.as_long());
    if y == 0 {
        Ir::new(CType::Long, Value::Long(0), ExKind::ZeroDiv)
    } else if x == i64::MIN && y == -1 {
        Ir::ovf()
    } else {
        Ir::long(x / y)
    }
}

pub fn smod(a: &Ir, b: &Ir) -> Ir {
    let (x, y) = (a.value.as_long(), b.value.as_long());
    if y == 0 {
        Ir::new(CType::Long, Value::Long(0), ExKind::ZeroDiv)
    } else if x == i64::MIN && y == -1 {
        Ir::ovf()
    } else {
        Ir::long(x % y)
    }
}

pub fn sneg(a: &Ir) -> Ir {
    let x = a.value.as_long();
    if x == i64::MIN {
        Ir::ovf()
    } else {
        Ir::long(-x)
    }
}

/// Normalize a shift amount exactly like the JVM: `<<`/`>>` on longs
/// mask the count to 6 bits. (Probed: `(bit-shift-left 1 64)` is `1`,
/// `(bit-shift-left 1 -1)` is `Long/MIN_VALUE`.) Note this is JVM
/// semantics; a future CLJS execution backend would need its own
/// 32-bit model for shift-heavy programs.
fn norm_shift(y: i64) -> u32 {
    (y & 63) as u32
}

pub fn sshl(a: &Ir, b: &Ir) -> Ir {
    Ir::long(
        a.value
            .as_long()
            .wrapping_shl(norm_shift(b.value.as_long())),
    )
}

pub fn sshr(a: &Ir, b: &Ir) -> Ir {
    Ir::long(
        a.value
            .as_long()
            .wrapping_shr(norm_shift(b.value.as_long())),
    )
}

pub fn sand(a: &Ir, b: &Ir) -> Ir {
    Ir::long(a.value.as_long() & b.value.as_long())
}

pub fn sor(a: &Ir, b: &Ir) -> Ir {
    Ir::long(a.value.as_long() | b.value.as_long())
}

pub fn sxor(a: &Ir, b: &Ir) -> Ir {
    Ir::long(a.value.as_long() ^ b.value.as_long())
}

pub fn snot(a: &Ir) -> Ir {
    Ir::long(!a.value.as_long())
}

/// Floored modulo for index wrapping. Matches Clojure `(mod x m)` for
/// `m > 0`: always in `[0, m)`, unlike truncated `rem`.
pub fn smod_floor(x: i64, m: i64) -> i64 {
    x.rem_euclid(m)
}

pub fn smin(a: &Ir, b: &Ir) -> Ir {
    Ir::long(a.value.as_long().min(b.value.as_long()))
}

pub fn smax(a: &Ir, b: &Ir) -> Ir {
    Ir::long(a.value.as_long().max(b.value.as_long()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    Min,
    Max,
    Lt,
    Gt,
    Eq,
    Ne,
    Le,
    Ge,
}

impl BinOp {
    pub fn scalar() -> &'static [BinOp] {
        use BinOp::*;
        &[Add, Sub, Mul, Div, Mod, Shl, Shr, And, Or, Xor, Min, Max]
    }
    pub fn compares() -> &'static [BinOp] {
        use BinOp::*;
        &[Lt, Gt, Eq, Ne, Le, Ge]
    }
    pub fn is_compare(self) -> bool {
        matches!(
            self,
            BinOp::Lt | BinOp::Gt | BinOp::Eq | BinOp::Ne | BinOp::Le | BinOp::Ge
        )
    }
    /// Emitted Clojure symbol.
    pub fn sym(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "quot",
            BinOp::Mod => "rem",
            BinOp::Shl => "bit-shift-left",
            BinOp::Shr => "bit-shift-right",
            BinOp::And => "bit-and",
            BinOp::Or => "bit-or",
            BinOp::Xor => "bit-xor",
            BinOp::Min => "min",
            BinOp::Max => "max",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Eq => "=",
            BinOp::Ne => "not=",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    BNot,
    /// Identity `(+ x)`; the rebuild target of a faulting negation.
    Pos,
}

pub fn eval_binop(op: BinOp, a: &Ir, b: &Ir) -> Ir {
    use BinOp::*;
    // Faults propagate: a faulty operand poisons the whole tree, so a
    // deeply nested overflow can never hide behind a placeholder value.
    if a.ex != ExKind::NoEx || b.ex != ExKind::NoEx {
        let ex = if a.ex != ExKind::NoEx { a.ex } else { b.ex };
        let (typ, value) = if op.is_compare() {
            (CType::Bool, Value::Bool(false))
        } else {
            (CType::Long, Value::Long(0))
        };
        return Ir::new(typ, value, ex);
    }
    match op {
        Add => sadd(a, b),
        Sub => ssub(a, b),
        Mul => smul(a, b),
        Div => sdiv(a, b),
        Mod => smod(a, b),
        Shl => sshl(a, b),
        Shr => sshr(a, b),
        And => sand(a, b),
        Or => sor(a, b),
        Xor => sxor(a, b),
        Min => smin(a, b),
        Max => smax(a, b),
        Lt | Gt | Eq | Ne | Le | Ge => {
            let (x, y) = (a.value.as_long(), b.value.as_long());
            let r = match op {
                Lt => x < y,
                Gt => x > y,
                Eq => x == y,
                Ne => x != y,
                Le => x <= y,
                _ => x >= y,
            };
            Ir::boolean(r)
        }
    }
}

pub fn eval_unop(op: UnOp, a: &Ir) -> Ir {
    if a.ex != ExKind::NoEx {
        return Ir::new(CType::Long, Value::Long(0), a.ex);
    }
    match op {
        UnOp::Neg => sneg(a),
        UnOp::BNot => snot(a),
        UnOp::Pos => a.clone(),
    }
}

/// Operator families for the SimilarOperators sub-policy.
pub fn op_families() -> Vec<Vec<BinOp>> {
    use BinOp::*;
    vec![
        vec![Add, Sub],
        vec![Mul, Div, Mod],
        vec![And, Or, Xor, Shl, Shr],
        vec![Lt, Gt, Eq, Ne, Le, Ge],
        vec![Min, Max],
    ]
}

/// YARPGen-style special constants: zero, min, max, bit-blocks,
/// end-bit patterns.
pub fn special_consts(typ: CType) -> Vec<i64> {
    let (lo, hi) = typ.range();
    let mut out = vec![
        0,
        1,
        -1,
        2,
        lo,
        hi,
        hi.saturating_sub(1),
        0x5555_5555,
        0xFF,
        0xFFFF,
        0xFFFF_FFFF,
        1 << 31,
        1 << 32,
        1 << 62,
    ];
    if fits_long(0x5555_5555_5555_5555) {
        out.push(0x5555_5555_5555_5555);
    }
    out.into_iter().map(|v| v.clamp(lo, hi)).collect()
}

/// Random constant: special value with `special_const_prob`, else a
/// small literal (small values keep downstream ops overflow-free).
pub fn random_const(rng: &mut Rng, policy: &Policy, typ: CType) -> Ir {
    if typ == CType::Bool {
        return Ir::boolean(rng.gen_bool());
    }
    if rng.roll(policy.special_const_prob) {
        let cands = special_consts(typ);
        let v = *rng.pick(&cands);
        let (lo, hi) = typ.range();
        Ir::new(typ, Value::Long(v.clamp(lo, hi)), ExKind::NoEx)
    } else {
        Ir::new(typ, Value::Long(rng.gen_range(-100, 100)), ExKind::NoEx)
    }
}

/// Random small vector constant.
pub fn random_vec(rng: &mut Rng, policy: &Policy) -> Ir {
    let len = rng.gen_range(2, policy.max_vector_len.max(2)) as usize;
    Ir::new(
        CType::Vec,
        Value::Vec((0..len).map(|_| rng.gen_range(-100, 100)).collect()),
        ExKind::NoEx,
    )
}

/// Clojure truthiness for branch conditions: only `false` and `nil`
/// are falsey. Note `0` and `[]` are truthy, unlike in C.
pub fn coerce_bool(v: &Ir) -> bool {
    match &v.value {
        Value::Bool(b) => *b,
        Value::Long(_) | Value::Vec(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_flagged_not_wrapped() {
        let a = Ir::long(i64::MAX);
        let b = Ir::long(1);
        assert_eq!(sadd(&a, &b).ex, ExKind::ArithOvf);
        assert_eq!(ssub(&Ir::long(i64::MIN), &b).ex, ExKind::ArithOvf);
        assert_eq!(smul(&a, &a).ex, ExKind::ArithOvf);
    }

    #[test]
    fn div_by_zero_flagged() {
        assert_eq!(sdiv(&Ir::long(1), &Ir::long(0)).ex, ExKind::ZeroDiv);
        assert_eq!(smod(&Ir::long(1), &Ir::long(0)).ex, ExKind::ZeroDiv);
        assert_eq!(
            sdiv(&Ir::long(i64::MIN), &Ir::long(-1)).ex,
            ExKind::ArithOvf
        );
    }

    #[test]
    fn div_matches_rust_semantics() {
        assert_eq!(sdiv(&Ir::long(7), &Ir::long(-3)).value, Value::Long(-2));
        assert_eq!(smod(&Ir::long(7), &Ir::long(-3)).value, Value::Long(1));
    }

    #[test]
    fn shifts_match_jvm_masking() {
        // Probed against clojure 1.12 on a JVM: counts mask to 6 bits.
        let one = Ir::long(1);
        assert_eq!(sshl(&one, &Ir::long(64)).value, Value::Long(1));
        assert_eq!(sshl(&one, &Ir::long(65)).value, Value::Long(2));
        assert_eq!(sshl(&one, &Ir::long(-1)).value, Value::Long(i64::MIN));
        assert_eq!(sshr(&Ir::long(256), &Ir::long(33)).value, Value::Long(0));
        assert_eq!(sshr(&Ir::long(-256), &Ir::long(33)).value, Value::Long(-1));
        assert_eq!(
            sshl(&Ir::long(3), &Ir::long(100)).value,
            Value::Long(206158430208)
        );
    }

    #[test]
    fn floored_mod_matches_clojure_mod() {
        // Clojure (mod x m) for m > 0 is always in [0, m).
        assert_eq!(smod_floor(-61, 7), 2);
        assert_eq!(smod_floor(-5, 7), 2);
        assert_eq!(smod_floor(7, 7), 0);
        assert_eq!(smod_floor(0, 5), 0);
        for x in -200..200 {
            for m in [2, 3, 7, 8] {
                let j = smod_floor(x, m);
                assert!((0..m).contains(&j), "mod {x} {m} = {j}");
            }
        }
    }

    #[test]
    fn faults_propagate_through_parents() {
        let ovf = smul(&Ir::long(1 << 62), &Ir::long(-23));
        assert_eq!(ovf.ex, ExKind::ArithOvf);
        // A clean parent of a faulty child must stay faulty.
        let parent = eval_binop(BinOp::Sub, &ovf, &Ir::long(1));
        assert_eq!(parent.ex, ExKind::ArithOvf);
        let wrapped = eval_unop(UnOp::Neg, &ovf);
        assert_eq!(wrapped.ex, ExKind::ArithOvf);
        // Compares propagate with a boolean result type.
        let zdiv = sdiv(&Ir::long(1), &Ir::long(0));
        let cmp = eval_binop(BinOp::Lt, &Ir::long(1), &zdiv);
        assert_eq!(cmp.ex, ExKind::ZeroDiv);
        assert_eq!(cmp.typ, CType::Bool);
    }

    #[test]
    fn special_consts_stay_in_range() {
        for t in [CType::Int, CType::UInt, CType::Long, CType::ULong] {
            let (lo, hi) = t.range();
            for v in special_consts(t) {
                assert!((lo..=hi).contains(&v), "{t:?} const {v} out of range");
            }
        }
    }
}
