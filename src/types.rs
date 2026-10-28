// Licensed under the Apache License, Version 2.0.
//! Value model: types, ranges, and the exception-kind lattice.
//! Ports YARPGen's data.h/type.h/ir_value.h. The no-UB guarantee becomes
//! a deterministic+exception-free guarantee: every generated expression is
//! evaluated at generation time and rebuilt until its ExKind is `NoEx`.

use crate::policy::Policy;
use crate::rng::Rng;
use std::collections::BTreeMap;

/// Exception kinds. Port of the reachable subset of YARPGen's `UBKind`
/// enum. The Clojure subset needs no shift/cast/recur faults: shift
/// amounts are clamped at generation, and emitted code contains no
/// casts or `recur` (those live in the negative suite instead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExKind {
    NoEx,
    ArithOvf,
    ZeroDiv,
    Oob,
}

/// Scalar types plus generation-time vectors. All integrals are emitted
/// as Clojure longs; the width drives special constants and overflow
/// detection. Checked long arithmetic throws on overflow (probed:
/// `addExact`/`multiplyExact`/`incrementExact`), so overflow is a
/// fault to rebuild, exactly like division by zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CType {
    Bool,
    Int,
    UInt,
    Long,
    ULong,
    Vec,
    Map,
    Record,
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
            CType::Map => (0, 0),
            CType::Record => (0, 0),
        }
    }

    pub fn is_scalar(self) -> bool {
        !matches!(self, CType::Vec | CType::Bool | CType::Map | CType::Record)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Long(i64),
    Bool(bool),
    Vec(Vec<i64>),
    Map(BTreeMap<String, i64>),
    Record(BTreeMap<String, i64>),
}

impl Value {
    pub fn as_long(&self) -> i64 {
        match self {
            Value::Long(n) => *n,
            Value::Bool(b) => i64::from(*b),
            Value::Vec(v) => v.len() as i64,
            Value::Map(m) => m.len() as i64,
            Value::Record(m) => m.len() as i64,
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
    let (x, y) = (a.value.as_long(), b.value.as_long());
    // Probed: non-folded `(- x Long/MIN_VALUE)` ALWAYS throws via an
    // intermediate negateExact — even `(- -5 MIN)`, which fits — while
    // fully constant `(- -5 MIN)` folds correctly. Since foldedness is
    // invisible here, any MIN subtrahend is a fault. (Siblings `+ *`
    // `quot` `rem` with MIN operands behave per plain math.)
    if y == i64::MIN {
        return Ir::ovf();
    }
    let r = x as i128 - y as i128;
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
        // Probed: JVM idiv only throws on /0, so MIN/-1 wraps to MIN
        // (JLS 15.17.2) instead of faulting. Rust would panic here,
        // hence the explicit arm.
        Ir::long(i64::MIN)
    } else {
        Ir::long(x / y)
    }
}

pub fn smod(a: &Ir, b: &Ir) -> Ir {
    let (x, y) = (a.value.as_long(), b.value.as_long());
    if y == 0 {
        Ir::new(CType::Long, Value::Long(0), ExKind::ZeroDiv)
    } else if x == i64::MIN && y == -1 {
        // Like quot, rem cannot overflow: MIN % -1 == 0, no throw.
        Ir::long(0)
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

/// Probed: `(inc Long/MAX_VALUE)` throws (incrementExact).
pub fn sinc(a: &Ir) -> Ir {
    let x = a.value.as_long();
    if x == i64::MAX {
        Ir::ovf()
    } else {
        Ir::long(x + 1)
    }
}

/// Probed: `(dec Long/MIN_VALUE)` throws (decrementExact).
pub fn sdec(a: &Ir) -> Ir {
    let x = a.value.as_long();
    if x == i64::MIN {
        Ir::ovf()
    } else {
        Ir::long(x - 1)
    }
}

/// Probed: `(abs Long/MIN_VALUE)` wraps to MIN_VALUE, never throws.
pub fn sabs(a: &Ir) -> Ir {
    Ir::long(a.value.as_long().wrapping_abs())
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
    Inc,
    Dec,
    Abs,
    /// Identity `(+ x)`; the rebuild target of faulting unary ops.
    Pos,
}

impl UnOp {
    /// Emitted Clojure symbol, shared by scalar and `map` emission.
    pub fn sym(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::BNot => "bit-not",
            UnOp::Inc => "inc",
            UnOp::Dec => "dec",
            UnOp::Abs => "abs",
            UnOp::Pos => "+",
        }
    }
}

/// Total single-arg predicates for `filter` stages. All total on
/// longs; probed identical on JVM Clojure and bb, including
/// negatives (`odd?` of -3 is true) and zero (even, neither pos nor
/// neg).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqPred {
    Odd,
    Even,
    Pos,
    Neg,
    Zero,
}

impl SeqPred {
    pub fn sym(self) -> &'static str {
        match self {
            SeqPred::Odd => "odd?",
            SeqPred::Even => "even?",
            SeqPred::Pos => "pos?",
            SeqPred::Neg => "neg?",
            SeqPred::Zero => "zero?",
        }
    }

    pub fn eval(self, x: i64) -> bool {
        match self {
            SeqPred::Odd => x % 2 != 0,
            SeqPred::Even => x % 2 == 0,
            SeqPred::Pos => x > 0,
            SeqPred::Neg => x < 0,
            SeqPred::Zero => x == 0,
        }
    }
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
        UnOp::Inc => sinc(a),
        UnOp::Dec => sdec(a),
        UnOp::Abs => sabs(a),
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

/// Shared keyword vocabulary (without the `:`) for map inputs. All
/// maps draw keys from this set, so every key is present in some map
/// and absent in others: reads roll present keys (plain `get` or
/// keyword-invoke) and absent keys (`get` with a default).
pub const MAP_KEYS: [&str; 4] = ["ka", "kb", "kc", "kd"];

/// Random small map constant: 2-4 vocabulary keys with small values.
pub fn random_map(rng: &mut Rng) -> Ir {
    let mut keys = MAP_KEYS.to_vec();
    // Fisher-Yates prefix: take a random-sized random subset.
    for i in (1..keys.len()).rev() {
        let j = rng.gen_usize(i + 1);
        keys.swap(i, j);
    }
    let n = rng.gen_range(2, MAP_KEYS.len() as i64) as usize;
    let map: BTreeMap<String, i64> = keys[..n]
        .iter()
        .map(|k| (k.to_string(), rng.gen_range(-100, 100)))
        .collect();
    Ir::new(CType::Map, Value::Map(map), ExKind::NoEx)
}

/// The one record type every program defines: `(defrecord R0 [fa fb
/// fc])`, emitted in init.clj. Fixed fields keep positional
/// construction arity-correct by construction; every record value
/// holds exactly these keys (sorted order doubles as field order).
pub const RECORD_NAME: &str = "R0";
/// Field vocabulary, alphabetical so BTreeMap order matches the
/// declared field vector.
pub const REC_FIELDS: [&str; 3] = ["fa", "fb", "fc"];

/// Random record constant: all three fields with small values.
pub fn random_record(rng: &mut Rng) -> Ir {
    let map: BTreeMap<String, i64> = REC_FIELDS
        .iter()
        .map(|k| (k.to_string(), rng.gen_range(-100, 100)))
        .collect();
    Ir::new(CType::Record, Value::Record(map), ExKind::NoEx)
}

/// Clojure truthiness for branch conditions: only `false` and `nil`
/// are falsey. Note `0` and `[]` are truthy, unlike in C.
pub fn coerce_bool(v: &Ir) -> bool {
    match &v.value {
        Value::Bool(b) => *b,
        // Probed: even `{}` is truthy (records are maps).
        Value::Long(_) | Value::Vec(_) | Value::Map(_) | Value::Record(_) => true,
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
    }

    #[test]
    fn minus_min_subtrahend_always_faults() {
        // Probed: `(- x MIN)` throws via negateExact for non-folded x
        // even when the result fits, so the model faults on any MIN
        // subtrahend. Only fully constant trees fold around it.
        assert_eq!(
            ssub(&Ir::long(-5), &Ir::long(i64::MIN)).ex,
            ExKind::ArithOvf
        );
        assert_eq!(ssub(&Ir::long(0), &Ir::long(i64::MIN)).ex, ExKind::ArithOvf);
        // Other MIN positions follow plain math.
        assert_eq!(
            ssub(&Ir::long(i64::MIN), &Ir::long(-5)).value,
            Value::Long(i64::MIN + 5)
        );
        assert_eq!(
            sadd(&Ir::long(5), &Ir::long(i64::MIN)).value,
            Value::Long(i64::MIN + 5)
        );
        assert_eq!(
            smul(&Ir::long(0), &Ir::long(i64::MIN)).value,
            Value::Long(0)
        );
    }

    #[test]
    fn quot_min_by_neg_one_wraps() {
        // Probed on Clojure 1.12.0 and 1.12.6: JVM idiv only throws
        // on /0 (JLS 15.17.2); MIN/-1 wraps to MIN as a Long.
        let q = sdiv(&Ir::long(i64::MIN), &Ir::long(-1));
        assert_eq!(q.ex, ExKind::NoEx);
        assert_eq!(q.value, Value::Long(i64::MIN));
    }

    #[test]
    fn rem_min_by_neg_one_is_zero() {
        assert_eq!(
            smod(&Ir::long(i64::MIN), &Ir::long(-1)),
            Ir::long(0),
            "rem of MIN by -1 is 0"
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
    fn inc_dec_abs_match_probed_edges() {
        assert_eq!(sinc(&Ir::long(i64::MAX)).ex, ExKind::ArithOvf);
        assert_eq!(sinc(&Ir::long(41)).value, Value::Long(42));
        assert_eq!(sdec(&Ir::long(i64::MIN)).ex, ExKind::ArithOvf);
        assert_eq!(sdec(&Ir::long(42)).value, Value::Long(41));
        assert_eq!(sabs(&Ir::long(-5)).value, Value::Long(5));
        assert_eq!(sabs(&Ir::long(i64::MIN)).value, Value::Long(i64::MIN));
        assert!(sabs(&Ir::long(i64::MIN)).is_ok());
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

    #[test]
    fn random_map_draws_vocab_keys() {
        // 2-4 vocabulary keys with small values, never empty.
        for seed in 1..50u64 {
            let mut rng = Rng::new(seed);
            let m = random_map(&mut rng);
            assert_eq!(m.typ, CType::Map);
            assert!(m.is_ok());
            match &m.value {
                Value::Map(items) => {
                    assert!((2..=MAP_KEYS.len()).contains(&items.len()));
                    for (k, v) in items {
                        assert!(MAP_KEYS.contains(&k.as_str()), "key {k} off-vocabulary");
                        assert!((-100..=100).contains(v));
                    }
                }
                other => panic!("expected map value, got {other:?}"),
            }
        }
    }

    #[test]
    fn maps_are_not_scalar_but_truthy() {
        assert!(!CType::Map.is_scalar());
        let m = Ir::new(CType::Map, Value::Map(BTreeMap::new()), ExKind::NoEx);
        assert!(coerce_bool(&m));
    }

    #[test]
    fn seq_preds_match_probed_table() {
        // Probed byte-identical on JVM Clojure 1.12.6 and bb: rows
        // are (odd? even? pos? neg? zero?).
        let table: &[(i64, [bool; 5])] = &[
            (-3, [true, false, false, true, false]),
            (-2, [false, true, false, true, false]),
            (-1, [true, false, false, true, false]),
            (0, [false, true, false, false, true]),
            (1, [true, false, true, false, false]),
            (2, [false, true, true, false, false]),
        ];
        use SeqPred::{Even, Neg, Odd, Pos, Zero};
        for (x, want) in table {
            let got = [
                Odd.eval(*x),
                Even.eval(*x),
                Pos.eval(*x),
                Neg.eval(*x),
                Zero.eval(*x),
            ];
            assert_eq!(&got, want, "x={x}");
        }
    }

    #[test]
    fn random_record_holds_all_fields() {
        // Total by construction: every field present, so positional
        // construction is always arity-correct.
        for seed in 1..50u64 {
            let mut rng = Rng::new(seed);
            let r = random_record(&mut rng);
            assert_eq!(r.typ, CType::Record);
            assert!(r.is_ok());
            match &r.value {
                Value::Record(items) => {
                    assert_eq!(items.len(), REC_FIELDS.len());
                    for f in REC_FIELDS {
                        let v = items
                            .get(f)
                            .unwrap_or_else(|| panic!("seed {seed}: missing {f}"));
                        assert!((-100..=100).contains(v));
                    }
                }
                other => panic!("expected record value, got {other:?}"),
            }
        }
        assert!(!CType::Record.is_scalar());
        // Sorted order must match the declared field vector, which
        // doubles as positional-constructor order.
        let mut sorted = REC_FIELDS.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, REC_FIELDS);
    }
}
