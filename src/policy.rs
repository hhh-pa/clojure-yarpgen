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

/// Effective generation policy for one program.
#[derive(Clone, Debug)]
pub struct Policy {
    pub stmt_num_limit: i64,
    pub loop_nest_depth_min: i64,
    pub loop_nest_depth_max: i64,
    pub loop_nest_depth_limit: i64,
    pub if_else_limit: i64,
    pub scope_stmt_min: i64,
    pub scope_stmt_max: i64,
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
    pub int_types: Vec<crate::types::CType>,
    pub max_nesting: i64,
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
            scope_stmt_min: 2,
            scope_stmt_max: 5,
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
                (StructKind::Stub, 70),
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
            int_types: vec![
                CType::Bool,
                CType::Int,
                CType::UInt,
                CType::Long,
                CType::ULong,
            ],
            max_nesting: 64,
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
        p.max_nesting = options.max_nesting;
    }
    if options.max_array_dims > 0 {
        p.max_vector_len = options.max_array_dims;
    }
    if options.param_shuffle {
        rng.shuffle_weights(&mut p.struct_distr);
        rng.shuffle_weights(&mut p.assign_distr);
        rng.shuffle_weights(&mut p.node_distr);
    }
    p
}

/// Random `[start end step]` loop bounds per policy.
pub fn loop_bounds(rng: &mut Rng, policy: &Policy) -> (i64, i64, i64) {
    (
        0,
        rng.gen_range(policy.iter_end_min, policy.iter_end_max),
        rng.gen_range(policy.iter_step_min, policy.iter_step_max),
    )
}

/// Random statement count for one scope.
pub fn scope_size(rng: &mut Rng, policy: &Policy) -> i64 {
    rng.gen_range(policy.scope_stmt_min, policy.scope_stmt_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_yarpgen_limits() {
        let p = Policy::default();
        assert_eq!(p.stmt_num_limit, 1000);
        assert_eq!(p.max_arith_depth, 4);
        assert_eq!(p.struct_distr.len(), 4);
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
