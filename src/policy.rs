// Licensed under the Apache License, Version 2.0.
//! Generation policy: probability tables and structural limits.
//! Ports YARPGen's gen_policy.h/gen_policy.cpp defaults.

use crate::cli::{Density, Options};
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructKind {
    LoopSeq,
    LoopNest,
    IfElse,
    Let,
    Recur,
    Stub,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignKind {
    Assign,
    Reduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Leaf,
    Unary,
    Binary,
}

/// Hard ceiling for `--max-array-dims`: vector literals are built
/// and emitted elementwise, so uncapped lengths OOM. Subscript
/// stress is length-independent (indices wrap by `mod`), so nothing
/// is lost past this.
pub const MAX_VECTOR_LEN: i64 = 65536;

/// Effective generation policy for one program.
#[derive(Clone, Debug)]
pub struct Policy {
    pub stmt_num_limit: i64,
    pub loop_nest_depth_min: i64,
    pub loop_nest_depth_max: i64,
    pub loop_nest_depth_limit: i64,
    pub if_else_limit: i64,
    pub let_nest_limit: i64,
    pub scope_stmt_min: i64,
    pub scope_stmt_max: i64,
    pub top_stmts: i64,
    pub iter_end_min: i64,
    pub iter_end_max: i64,
    pub iter_step_min: i64,
    pub iter_step_max: i64,
    pub max_arith_depth: i64,
    pub scalar_inputs_min: i64,
    pub scalar_inputs_max: i64,
    pub struct_distr: Vec<(StructKind, u32)>,
    pub else_branch_prob: f64,
    pub assign_distr: Vec<(AssignKind, u32)>,
    pub node_distr: Vec<(NodeKind, u32)>,
    pub similar_operators_prob: f64,
    pub const_use_prob: f64,
    pub special_const_prob: f64,
    pub reuse_prob: f64,
    pub nth_prob: f64,
    pub map_prob: f64,
    pub int_types: Vec<crate::types::CType>,
    pub max_vector_len: i64,
    pub ub_allow_weight: u32,
    pub ub_deny_weight: u32,
}

impl Default for Policy {
    fn default() -> Self {
        use crate::types::CType;
        Policy {
            stmt_num_limit: 1000,
            loop_nest_depth_min: 2,
            loop_nest_depth_max: 3,
            loop_nest_depth_limit: 5,
            if_else_limit: 5,
            let_nest_limit: 5,
            scope_stmt_min: 2,
            scope_stmt_max: 5,
            top_stmts: 0,
            iter_end_min: 10,
            iter_end_max: 25,
            iter_step_min: 1,
            iter_step_max: 4,
            max_arith_depth: 4,
            scalar_inputs_min: 10,
            scalar_inputs_max: 20,
            struct_distr: vec![
                (StructKind::LoopSeq, 10),
                (StructKind::LoopNest, 10),
                (StructKind::IfElse, 10),
                (StructKind::Let, 10),
                (StructKind::Recur, 10),
                (StructKind::Stub, 50),
            ],
            else_branch_prob: 0.2,
            assign_distr: vec![(AssignKind::Assign, 70), (AssignKind::Reduction, 30)],
            node_distr: vec![
                (NodeKind::Leaf, 30),
                (NodeKind::Unary, 20),
                (NodeKind::Binary, 50),
            ],
            similar_operators_prob: 0.1,
            const_use_prob: 0.1,
            special_const_prob: 0.3,
            reuse_prob: 0.3,
            nth_prob: 0.15,
            map_prob: 0.15,
            int_types: vec![
                CType::Bool,
                CType::Int,
                CType::UInt,
                CType::Long,
                CType::ULong,
            ],
            max_vector_len: 8,
            ub_allow_weight: 30,
            ub_deny_weight: 70,
        }
    }
}

/// Build the effective policy from CLI options: simple-loops profile,
/// explicit loop bounds, nesting/vector caps, and param shuffling.
pub fn make_policy(options: &Options, rng: &mut Rng) -> Policy {
    use crate::types::CType;
    let mut p = Policy::default();
    let simple = match options.simple_loops {
        Density::All => true,
        Density::Some => rng.roll(0.5),
        Density::None => false,
    };
    if simple {
        p.loop_nest_depth_min = 1;
        p.loop_nest_depth_max = 2;
        p.loop_nest_depth_limit = 2;
        p.max_arith_depth = 2;
        p.iter_end_min = 32;
        p.iter_end_max = 64;
        p.int_types = vec![CType::Int, CType::UInt];
    }
    if options.expl_loop_param {
        p.iter_end_min = 1;
        p.iter_end_max = 64;
        p.iter_step_min = 1;
        p.iter_step_max = 8;
    }
    if options.max_nesting > 0 {
        // Explicit nesting guard wins over profiles: cap loop/if
        // depth, clamping the nest min/max so the depth roll in
        // gen_loop_nest keeps a valid (non-inverted) range.
        let l = options.max_nesting.max(1);
        p.loop_nest_depth_limit = l;
        p.if_else_limit = l;
        p.let_nest_limit = l;
        p.loop_nest_depth_min = p.loop_nest_depth_min.min(l);
        p.loop_nest_depth_max = p.loop_nest_depth_max.clamp(p.loop_nest_depth_min, l);
    }
    if options.top_stmts > 0 {
        p.top_stmts = options.top_stmts;
    }
    if options.max_array_dims > 0 {
        p.max_vector_len = options.max_array_dims.min(MAX_VECTOR_LEN);
    }
    if options.param_shuffle {
        rng.shuffle_weights(&mut p.struct_distr);
        rng.shuffle_weights(&mut p.assign_distr);
        rng.shuffle_weights(&mut p.node_distr);
    }
    p
}

/// Depth-scaled loop bounds: the end range halves per nesting depth
/// (floored at 2), so the trip-count product of deep nests stays
/// simulable. Depth-0 loops keep the full policy range; trip counts
/// barely matter for compiler stress, structure does.
pub fn loop_bounds_at_depth(rng: &mut Rng, policy: &Policy, depth: i64) -> (i64, i64, i64) {
    let shift = depth.clamp(0, 3) as u32;
    let lo = (policy.iter_end_min >> shift).max(2);
    let hi = (policy.iter_end_max >> shift).max(lo);
    (
        0,
        rng.gen_range(lo, hi),
        rng.gen_range(policy.iter_step_min, policy.iter_step_max),
    )
}

/// Depth-scaled end bound for `dotimes`.
pub fn loop_end_at_depth(rng: &mut Rng, policy: &Policy, depth: i64) -> i64 {
    let shift = depth.clamp(0, 3) as u32;
    let lo = (policy.iter_end_min >> shift).max(2);
    let hi = (policy.iter_end_max >> shift).max(lo);
    rng.gen_range(lo, hi)
}

/// Statement count for one scope: the `--top-stmts` override at
/// depth 0, else a policy roll. (A populate step can yield several
/// statements when a loop degrades to a stub scope, so the top scope
/// may exceed the override.) Nested scopes taper with depth:
/// breadth compounds with nesting into single statements too big
/// for one 64KB method, while depth alone keeps the nesting stress.
pub fn scope_size(rng: &mut Rng, policy: &Policy, depth: i64) -> i64 {
    if depth == 0 && policy.top_stmts > 0 {
        policy.top_stmts
    } else if depth == 0 {
        rng.gen_range(policy.scope_stmt_min, policy.scope_stmt_max)
    } else {
        rng.gen_range(1, (policy.scope_stmt_max - depth).max(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_yarpgen_limits() {
        let p = Policy::default();
        assert_eq!(p.stmt_num_limit, 1000);
        assert_eq!(p.max_arith_depth, 4);
        // LoopSeq, LoopNest, IfElse, Let, Recur, Stub.
        assert_eq!(p.struct_distr.len(), 6);
        assert_eq!(p.let_nest_limit, 5);
    }

    #[test]
    fn top_stmts_overrides_depth_zero_only() {
        let mut rng = Rng::new(5);
        let policy = Policy {
            top_stmts: 40,
            ..Policy::default()
        };
        assert_eq!(scope_size(&mut rng, &policy, 0), 40);
        for _ in 0..20 {
            // Depth 1 is not overridden; it tapers to 1..=4.
            assert!((1..=4).contains(&scope_size(&mut rng, &policy, 1)));
        }
        let plain = Policy::default();
        for _ in 0..20 {
            assert!((2..=5).contains(&scope_size(&mut rng, &plain, 0)));
        }
    }

    #[test]
    fn nested_scope_sizes_taper_with_depth() {
        // Breadth compounds with nesting into oversize methods, so
        // nested scopes narrow while staying non-empty.
        let mut rng = Rng::new(5);
        let policy = Policy::default();
        for _ in 0..50 {
            assert!((1..=4).contains(&scope_size(&mut rng, &policy, 1)));
            assert!((1..=3).contains(&scope_size(&mut rng, &policy, 2)));
            assert!((1..=2).contains(&scope_size(&mut rng, &policy, 3)));
            assert!((1..=2).contains(&scope_size(&mut rng, &policy, 9)));
        }
    }

    #[test]
    fn depth_scaled_bounds_shrink_with_depth() {
        let mut rng = Rng::new(5);
        let policy = Policy::default();
        let (_, hi0, _) = loop_bounds_at_depth(&mut rng, &policy, 0);
        assert!((10..=25).contains(&hi0));
        for _ in 0..50 {
            let (_, deep, _) = loop_bounds_at_depth(&mut rng, &policy, 4);
            assert!((2..=3).contains(&deep), "deep end {deep} out of 2..=3");
        }
    }

    #[test]
    fn max_array_dims_is_capped() {
        // Gigantic vector lengths would OOM building or emitting the
        // literal; the flag clamps to a sane maximum instead.
        let mut rng = Rng::new(1);
        let opts = Options {
            max_array_dims: 10_000_000_000,
            param_shuffle: false,
            ..Options::default()
        };
        let p = make_policy(&opts, &mut rng);
        assert_eq!(p.max_vector_len, crate::policy::MAX_VECTOR_LEN);
        let opts = Options {
            max_array_dims: 100,
            param_shuffle: false,
            ..Options::default()
        };
        let p = make_policy(&opts, &mut rng);
        assert_eq!(p.max_vector_len, 100);
    }

    #[test]
    fn max_nesting_caps_loop_and_if_depth() {
        let mut rng = Rng::new(1);
        let opts = Options {
            max_nesting: 1,
            param_shuffle: false,
            ..Options::default()
        };
        let p = make_policy(&opts, &mut rng);
        assert_eq!(p.loop_nest_depth_limit, 1);
        assert_eq!(p.if_else_limit, 1);
        assert_eq!(p.let_nest_limit, 1);
        // Nest min/max stay a valid range for the depth roll.
        assert!(p.loop_nest_depth_min <= p.loop_nest_depth_max.min(p.loop_nest_depth_limit));
        let opts = Options {
            max_nesting: 3,
            param_shuffle: false,
            ..Options::default()
        };
        let p = make_policy(&opts, &mut rng);
        assert_eq!(p.loop_nest_depth_limit, 3);
        assert_eq!(p.loop_nest_depth_min, 2);
        assert_eq!(p.loop_nest_depth_max, 3);
    }

    #[test]
    fn simple_profile_narrows_depth_and_types() {
        let mut rng = Rng::new(1);
        let opts = Options {
            simple_loops: Density::All,
            param_shuffle: false,
            ..Options::default()
        };
        let p = make_policy(&opts, &mut rng);
        assert_eq!(p.loop_nest_depth_limit, 2);
        assert_eq!(p.max_arith_depth, 2);
        assert_eq!(p.int_types.len(), 2);
    }
}
