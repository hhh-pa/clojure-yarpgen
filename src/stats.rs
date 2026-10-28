// Licensed under the Apache License, Version 2.0.
//! Program statistics. Ports YARPGen's statistics collection.

use crate::emitter::Program;
use crate::expr::Expr;
use crate::stmt::Stmt;
use std::collections::BTreeMap;

/// Count AST nodes in a generated program.
pub fn program_stats(program: &Program) -> BTreeMap<String, i64> {
    let mut counts = BTreeMap::new();
    counts.insert("inputs".to_string(), program.inputs.len() as i64);
    counts.insert("outputs".to_string(), program.outputs.len() as i64);
    counts.insert("seed".to_string(), program.seed as i64);
    for s in &program.stmts {
        walk_stmt(s, &mut counts, 1);
    }
    counts
}

fn bump(counts: &mut BTreeMap<String, i64>, key: &str) {
    *counts.entry(key.to_string()).or_insert(0) += 1;
}

fn walk_expr(e: &Expr, counts: &mut BTreeMap<String, i64>) {
    bump(counts, "exprs");
    match e {
        Expr::Const(_) => bump(counts, "consts"),
        Expr::Var { .. } => bump(counts, "var-refs"),
        Expr::UnOp { arg, .. } => {
            bump(counts, "unops");
            walk_expr(arg, counts);
        }
        Expr::BinOp {
            op, left, right, ..
        } => {
            bump(counts, "binops");
            bump(
                counts,
                match op {
                    crate::types::BinOp::Add => "adds",
                    crate::types::BinOp::Sub => "subs",
                    crate::types::BinOp::Mul => "muls",
                    crate::types::BinOp::Div => "divs",
                    _ => "other-binops",
                },
            );
            walk_expr(left, counts);
            walk_expr(right, counts);
        }
        Expr::Nth { coll, idx, .. } => {
            bump(counts, "subscripts");
            walk_expr(coll, counts);
            walk_expr(idx, counts);
        }
        Expr::NthDefault {
            coll, idx, default, ..
        } => {
            bump(counts, "subscripts");
            walk_expr(coll, counts);
            walk_expr(idx, counts);
            walk_expr(default, counts);
        }
        Expr::IdxMod { idx, .. } => {
            bump(counts, "index-wraps");
            walk_expr(idx, counts);
        }
    }
}

fn walk_stmt(s: &Stmt, counts: &mut BTreeMap<String, i64>, depth: i64) {
    bump(counts, "stmts");
    let max = counts.entry("max-depth".to_string()).or_insert(0);
    *max = (*max).max(depth);
    match s {
        Stmt::Assign { expr, .. } => {
            bump(counts, "assigns");
            walk_expr(expr, counts);
        }
        Stmt::If { cond, then, els } => {
            bump(counts, "ifs");
            walk_expr(cond, counts);
            for t in then {
                walk_stmt(t, counts, depth + 1);
            }
            for e in els {
                walk_stmt(e, counts, depth + 1);
            }
        }
        Stmt::Loop { body, .. } => {
            bump(counts, "loops");
            for b in body {
                walk_stmt(b, counts, depth + 1);
            }
        }
    }
}

/// Bytecode-size proxy: control flow is weighted up because `doseq`
/// macroexpansion is large relative to its AST size. Used to chunk
/// `test-fn` so no single method nears the 64KB JVM limit
/// ("Method code too large!").
pub fn stmt_cost(s: &Stmt) -> u64 {
    match s {
        Stmt::Assign { expr, .. } => 1 + expr_cost(expr),
        Stmt::If { cond, then, els } => {
            4 + expr_cost(cond)
                + then.iter().map(stmt_cost).sum::<u64>()
                + els.iter().map(stmt_cost).sum::<u64>()
        }
        Stmt::Loop { body, .. } => 8 + body.iter().map(stmt_cost).sum::<u64>(),
    }
}

pub fn expr_cost(e: &Expr) -> u64 {
    1 + match e {
        Expr::Const(_) | Expr::Var { .. } => 0,
        Expr::UnOp { arg, .. } => expr_cost(arg),
        Expr::BinOp { left, right, .. } => expr_cost(left) + expr_cost(right),
        Expr::Nth { coll, idx, .. } => expr_cost(coll) + expr_cost(idx),
        Expr::NthDefault {
            coll, idx, default, ..
        } => expr_cost(coll) + expr_cost(idx) + expr_cost(default),
        Expr::IdxMod { idx, .. } => expr_cost(idx),
    }
}
