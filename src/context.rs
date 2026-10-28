// Licensed under the Apache License, Version 2.0.
//! Generation contexts and symbol tables. Ports YARPGen's context.h
//! `GenCtx`/`PopulateCtx`, including taken-tracking
//! (`then = taken && cond`) for live vs dead branch analysis.

use crate::types::{CType, Value};
use std::collections::BTreeMap;

/// One tracked variable: external input, external output, or local.
#[derive(Clone, Debug)]
pub struct Var {
    pub name: String,
    pub typ: CType,
    pub value: Value,
}

impl Var {
    pub fn long_value(&self) -> i64 {
        self.value.as_long()
    }
}

/// Generation context: symbol tables, counters, and the taken flag.
#[derive(Clone, Debug)]
pub struct GenCtx {
    pub ext_in: BTreeMap<String, Var>,
    pub ext_out: BTreeMap<String, Var>,
    pub locals: BTreeMap<String, Var>,
    pub taken: bool,
    pub stmt_count: i64,
    pub loop_depth: i64,
    pub if_depth: i64,
}

impl GenCtx {
    pub fn new() -> Self {
        GenCtx {
            ext_in: BTreeMap::new(),
            ext_out: BTreeMap::new(),
            locals: BTreeMap::new(),
            taken: true,
            stmt_count: 0,
            loop_depth: 0,
            if_depth: 0,
        }
    }

    pub fn add_input(&mut self, name: &str, typ: CType, value: Value) {
        self.ext_in.insert(
            name.to_string(),
            Var {
                name: name.to_string(),
                typ,
                value,
            },
        );
    }

    pub fn add_output(&mut self, name: &str, typ: CType, value: Value) {
        self.ext_out.insert(
            name.to_string(),
            Var {
                name: name.to_string(),
                typ,
                value,
            },
        );
    }

    pub fn add_local(&mut self, name: &str, typ: CType, value: Value) {
        self.locals.insert(
            name.to_string(),
            Var {
                name: name.to_string(),
                typ,
                value,
            },
        );
    }

    /// Look up a variable across all tables, plus the hidden `zero`.
    pub fn lookup(&self, name: &str) -> Option<Var> {
        if name == "zero" {
            return Some(Var {
                name: "zero".to_string(),
                typ: CType::Long,
                value: Value::Long(0),
            });
        }
        self.locals
            .get(name)
            .or_else(|| self.ext_out.get(name))
            .or_else(|| self.ext_in.get(name))
            .cloned()
    }

    /// Update a variable's generation-time value wherever it lives.
    pub fn update_value(&mut self, name: &str, value: Value) {
        if let Some(v) = self.locals.get_mut(name) {
            v.value = value;
        } else if let Some(v) = self.ext_out.get_mut(name) {
            v.value = value;
        } else if let Some(v) = self.ext_in.get_mut(name) {
            v.value = value;
        }
    }

    /// All visible variables.
    pub fn all_vars(&self) -> Vec<&Var> {
        self.ext_in
            .values()
            .chain(self.ext_out.values())
            .chain(self.locals.values())
            .collect()
    }

    /// Variables that assignment may target: external outputs only.
    /// Locals are `doseq`-bound loop variables (plain values, not atoms)
    /// and must never be `reset!`.
    pub fn writable_vars(&self) -> Vec<&Var> {
        self.ext_out.values().collect()
    }

    /// Derive a branch context: `taken && flag`.
    pub fn with_taken(&self, flag: bool) -> Self {
        let mut c = self.clone();
        c.taken = self.taken && flag;
        c
    }
}

impl Default for GenCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_prefers_locals() {
        let mut c = GenCtx::new();
        c.add_input("x", CType::Long, Value::Long(1));
        c.add_local("x", CType::Long, Value::Long(2));
        assert_eq!(c.lookup("x").unwrap().long_value(), 2);
    }

    #[test]
    fn hidden_zero_always_visible() {
        let c = GenCtx::new();
        assert_eq!(c.lookup("zero").unwrap().long_value(), 0);
    }

    #[test]
    fn taken_tracks_conjunction() {
        let c = GenCtx::new();
        assert!(c.with_taken(true).taken);
        assert!(!c.with_taken(false).taken);
        assert!(!c.with_taken(false).with_taken(true).taken);
    }
}
