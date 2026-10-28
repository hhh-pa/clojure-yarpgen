// Licensed under the Apache License, Version 2.0.
//! Minimal s-expression builder and pretty printer for emitted Clojure code.

/// An s-expression: the emission target for every generated form.
#[derive(Clone, Debug, PartialEq)]
pub enum Sexp {
    Sym(String),
    Int(i64),
    Bool(bool),
    Str(String),
    List(Vec<Sexp>),
    /// `^long` type-hinted symbol (the `--emit-pragmas` analog).
    Hinted(String),
}

impl Sexp {
    pub fn sym(s: &str) -> Self {
        Sexp::Sym(s.to_string())
    }
    pub fn int(n: i64) -> Self {
        Sexp::Int(n)
    }
    pub fn bool(b: bool) -> Self {
        Sexp::Bool(b)
    }
    pub fn str(s: &str) -> Self {
        Sexp::Str(s.to_string())
    }
    pub fn list(items: Vec<Sexp>) -> Self {
        Sexp::List(items)
    }
    /// `[a b c]` prints as a vector; represented as a list with a marker.
    pub fn vec(items: Vec<Sexp>) -> Self {
        let mut v = vec![Sexp::Sym("__vec__".to_string())];
        v.extend(items);
        Sexp::List(v)
    }
    /// `{k v ...}` prints as a map; alternating keys and values with
    /// a marker, mirroring [`Sexp::vec`].
    pub fn map(items: Vec<Sexp>) -> Self {
        let mut v = vec![Sexp::Sym("__map__".to_string())];
        v.extend(items);
        Sexp::List(v)
    }

    /// Compact one-line rendering.
    pub fn flat(&self) -> String {
        match self {
            Sexp::Sym(s) => s.clone(),
            Sexp::Int(n) => n.to_string(),
            Sexp::Bool(b) => b.to_string(),
            Sexp::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            Sexp::Hinted(s) => format!("^long {s}"),
            Sexp::List(items) => {
                if items.first() == Some(&Sexp::Sym("__vec__".to_string())) {
                    let inner: Vec<String> = items[1..].iter().map(|x| x.flat()).collect();
                    format!("[{}]", inner.join(" "))
                } else if items.first() == Some(&Sexp::Sym("__map__".to_string())) {
                    let inner: Vec<String> = items[1..].iter().map(|x| x.flat()).collect();
                    format!("{{{}}}", inner.join(" "))
                } else {
                    let inner: Vec<String> = items.iter().map(|x| x.flat()).collect();
                    format!("({})", inner.join(" "))
                }
            }
        }
    }

    /// Pretty rendering with a right margin; long lists break one element
    /// per line. Deterministic for a fixed AST.
    pub fn pretty(&self) -> String {
        self.render(0, 100)
    }

    fn render(&self, indent: usize, margin: usize) -> String {
        let pad = " ".repeat(indent);
        match self {
            Sexp::List(items) => {
                let flat = self.flat();
                if indent + flat.len() <= margin || items.len() <= 1 {
                    return format!("{pad}{flat}");
                }
                let is_vec = items.first() == Some(&Sexp::Sym("__vec__".to_string()));
                let is_map = items.first() == Some(&Sexp::Sym("__map__".to_string()));
                let (open, close, elems) = if is_vec {
                    ("[", "]", &items[1..])
                } else if is_map {
                    ("{", "}", &items[1..])
                } else {
                    ("(", ")", &items[..])
                };
                let mut out = format!("{pad}{open}{}", elems[0].flat());
                for e in &elems[1..] {
                    out.push('\n');
                    out.push_str(&e.render(indent + 2, margin));
                }
                out.push_str(close);
                out
            }
            _ => format!("{pad}{}", self.flat()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_round_trip() {
        let e = Sexp::list(vec![
            Sexp::sym("+"),
            Sexp::int(1),
            Sexp::list(vec![Sexp::sym("*"), Sexp::int(2), Sexp::int(3)]),
        ]);
        assert_eq!(e.flat(), "(+ 1 (* 2 3))");
    }

    #[test]
    fn vec_prints_with_brackets() {
        let v = Sexp::vec(vec![Sexp::int(1), Sexp::int(2)]);
        assert_eq!(v.flat(), "[1 2]");
    }

    #[test]
    fn map_prints_with_braces() {
        let m = Sexp::map(vec![
            Sexp::sym(":ka"),
            Sexp::int(1),
            Sexp::sym(":kb"),
            Sexp::int(2),
        ]);
        assert_eq!(m.flat(), "{:ka 1 :kb 2}");
    }

    #[test]
    fn pretty_breaks_long_lists() {
        let long = Sexp::list(
            std::iter::once(Sexp::sym("do"))
                .chain((0..100).map(Sexp::int))
                .collect(),
        );
        let out = long.pretty();
        assert!(out.contains('\n'));
        assert!(out.starts_with("(do"));
    }
}
