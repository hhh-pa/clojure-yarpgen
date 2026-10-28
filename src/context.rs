// Licensed under the Apache License, Version 2.0.
//! Generation contexts and symbol tables. Ports YARPGen's context.h
//! `GenCtx`/`PopulateCtx`, including taken-tracking
//! (`then = taken && cond`) for live vs dead branch analysis.
//!
//! Variables are interned to integer [`VarId`]s at creation. Hot paths
//! (expression evaluation, assignment during simulation) index flat
//! vectors; name probes happen only on scope operations. This keeps
//! simulation of deep loop nests fast: string-keyed maps dominated
//! the profile before interning.

use crate::types::{CType, Value};
use std::collections::BTreeMap;

/// Interned variable identity. Stable within one program across
/// context clones. Ids ARE reused after scope truncation, which is
/// sound because reuse mirrors lexical scope: an id always denotes
/// the innermost live binding, simulation visits taken paths only,
/// and mutation never interns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

/// Generation context: interned symbol table plus counters and the
/// taken flag. Cloning is cheap (small flat vectors); scope exit
/// truncates back to a snapshot length.
#[derive(Clone, Debug)]
pub struct GenCtx {
    names: Vec<String>,
    index: BTreeMap<String, VarId>,
    typs: Vec<CType>,
    values: Vec<Value>,
    /// Output ids. Outputs are registered before any local, so scope
    /// truncation never touches them.
    outputs: Vec<VarId>,
    pub taken: bool,
    pub stmt_count: i64,
    pub loop_depth: i64,
    pub if_depth: i64,
    pub let_depth: i64,
}

impl GenCtx {
    pub fn new() -> Self {
        GenCtx {
            names: Vec::new(),
            index: BTreeMap::new(),
            typs: Vec::new(),
            values: Vec::new(),
            outputs: Vec::new(),
            taken: true,
            stmt_count: 0,
            loop_depth: 0,
            if_depth: 0,
            let_depth: 0,
        }
    }

    fn intern(&mut self, name: &str, typ: CType, value: Value) -> VarId {
        if let Some(id) = self.index.get(name) {
            let id = *id;
            self.values[id.0 as usize] = value;
            self.typs[id.0 as usize] = typ;
            return id;
        }
        let id = VarId(self.names.len() as u32);
        self.names.push(name.to_string());
        self.index.insert(name.to_string(), id);
        self.typs.push(typ);
        self.values.push(value);
        id
    }

    pub fn add_input(&mut self, name: &str, typ: CType, value: Value) -> VarId {
        self.intern(name, typ, value)
    }

    pub fn add_output(&mut self, name: &str, typ: CType, value: Value) -> VarId {
        let id = self.intern(name, typ, value);
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
        id
    }

    /// Bind a local (loop variable or `let` binding). Rebinding an
    /// existing name (every loop iteration during simulation)
    /// updates in place.
    pub fn add_local(&mut self, name: &str, typ: CType, value: Value) -> VarId {
        self.intern(name, typ, value)
    }

    pub fn id_of(&self, name: &str) -> Option<VarId> {
        self.index.get(name).copied()
    }

    pub fn var_name(&self, id: VarId) -> &str {
        &self.names[id.0 as usize]
    }

    pub fn var_type(&self, id: VarId) -> CType {
        self.typs[id.0 as usize]
    }

    pub fn var_value(&self, id: VarId) -> &Value {
        &self.values[id.0 as usize]
    }

    /// Name-based lookup for cold paths (final values, tests).
    /// Hot paths use ids directly.
    pub fn lookup(&self, name: &str) -> Option<Value> {
        self.id_of(name).map(|id| self.var_value(id).clone())
    }

    /// Hot-path update by id. O(1), no probing or cloning.
    pub fn update_value_by_id(&mut self, id: VarId, value: Value) {
        self.values[id.0 as usize] = value;
    }

    /// All visible variables.
    pub fn all_vars(&self) -> Vec<VarId> {
        (0..self.names.len() as u32).map(VarId).collect()
    }

    /// Variables that assignment may target: external outputs only.
    /// Locals (loop variables, `let` bindings) are plain values, not
    /// atoms, and must never be `reset!`.
    pub fn writable_vars(&self) -> Vec<VarId> {
        self.outputs.clone()
    }

    /// Output names, for atom rendering when measuring statements.
    pub fn output_names(&self) -> Vec<String> {
        self.outputs
            .iter()
            .map(|id| self.names[id.0 as usize].clone())
            .collect()
    }

    /// Derive a branch context: `taken && flag`.
    pub fn with_taken(&self, flag: bool) -> Self {
        let mut c = self.clone();
        c.taken = self.taken && flag;
        c
    }

    /// Drop everything interned after `other`'s snapshot: locals
    /// leaving scope. Outputs predate all locals and survive.
    pub fn restore_scope_from(&mut self, other: &Self) {
        self.truncate_to(other.names.len());
        self.outputs.truncate(other.outputs.len());
        self.loop_depth = other.loop_depth;
    }

    /// Current table length: a snapshot for [`GenCtx::truncate_to`].
    pub fn snapshot_len(&self) -> usize {
        self.names.len()
    }

    /// Drop entries at and past `len`, restoring loop-exit scope.
    pub fn truncate_to(&mut self, len: usize) {
        debug_assert!(len <= self.names.len());
        for name in self.names.drain(len..) {
            self.index.remove(&name);
        }
        self.typs.truncate(len);
        self.values.truncate(len);
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
    fn lookup_round_trip() {
        let mut c = GenCtx::new();
        c.add_input("x", CType::Long, Value::Long(1));
        c.add_output("y", CType::Long, Value::Long(2));
        assert_eq!(c.lookup("x").unwrap().as_long(), 1);
        assert_eq!(c.lookup("y").unwrap().as_long(), 2);
        assert!(c.lookup("missing").is_none());
    }

    #[test]
    fn rebind_updates_in_place() {
        let mut c = GenCtx::new();
        let a = c.add_local("i", CType::Long, Value::Long(0));
        let b = c.add_local("i", CType::Long, Value::Long(5));
        assert_eq!(a, b);
        assert_eq!(c.var_value(a), &Value::Long(5));
        assert_eq!(c.all_vars().len(), 1);
    }

    #[test]
    fn writables_are_outputs_only() {
        let mut c = GenCtx::new();
        c.add_input("in-0", CType::Long, Value::Long(0));
        c.add_output("out-0", CType::Long, Value::Long(0));
        c.add_local("i", CType::Long, Value::Long(0));
        let w = c.writable_vars();
        assert_eq!(w.len(), 1);
        assert_eq!(c.var_name(w[0]), "out-0");
    }

    #[test]
    fn scope_restore_drops_locals() {
        let mut c = GenCtx::new();
        c.add_output("out-0", CType::Long, Value::Long(0));
        let pre = c.clone();
        c.add_local("i", CType::Long, Value::Long(3));
        assert_eq!(c.all_vars().len(), 2);
        c.restore_scope_from(&pre);
        assert_eq!(c.all_vars().len(), 1);
        assert!(c.id_of("i").is_none());
        assert_eq!(c.lookup("out-0").unwrap().as_long(), 0);
    }

    #[test]
    fn taken_tracks_conjunction() {
        let c = GenCtx::new();
        assert!(c.with_taken(true).taken);
        assert!(!c.with_taken(false).taken);
        assert!(!c.with_taken(false).with_taken(true).taken);
    }
}
