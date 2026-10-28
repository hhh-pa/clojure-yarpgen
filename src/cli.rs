// Licensed under the Apache License, Version 2.0.
//! Command-line options. Ports YARPGen's options.h/options.cpp:
//! long `--arg=value` plus short `-s`/`-o`/`-h`/`-v` forms.

pub const VERSION: &str = "0.1.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Std {
    Clj,
    Cljs,
    Cljc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckAlgo {
    Hash,
    Asserts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Density {
    None,
    Some,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutateMode {
    None,
    Exprs,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnOff {
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegativeKind {
    None,
    Arity,
    Recur,
    Nesting,
    Hints,
    All,
}

/// Parsed command-line options.
#[derive(Clone, Debug)]
pub struct Options {
    pub seed: u64,
    pub out_dir: String,
    pub std: Std,
    pub check_algo: CheckAlgo,
    pub emit_pragmas: Density,
    pub param_shuffle: bool,
    pub expl_loop_param: bool,
    pub mutate: MutateMode,
    pub mutation_seed: u64,
    pub allow_exc_in_dead: Density,
    pub max_array_dims: i64,
    pub max_nesting: i64,
    pub simple_loops: Density,
    pub unchecked_math: OnOff,
    pub dg_headers: bool,
    pub negative: NegativeKind,
    pub count: i64,
    pub max_fn_cost: u64,
    pub top_stmts: i64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            seed: 0,
            out_dir: ".".to_string(),
            std: Std::Clj,
            check_algo: CheckAlgo::Hash,
            emit_pragmas: Density::Some,
            param_shuffle: true,
            expl_loop_param: false,
            mutate: MutateMode::None,
            mutation_seed: 0,
            allow_exc_in_dead: Density::None,
            max_array_dims: 0,
            max_nesting: 0,
            simple_loops: Density::None,
            unchecked_math: OnOff::Off,
            dg_headers: false,
            negative: NegativeKind::None,
            count: 1,
            max_fn_cost: 0,
            top_stmts: 0,
        }
    }
}

pub struct Parsed {
    pub options: Options,
    pub errors: Vec<String>,
    pub help: bool,
    pub version: bool,
}

pub fn usage() -> &'static str {
    "Usage: clojure-yarpgen [options]\n\
     Generate random Clojure programs that stress the compiler.\n\n\
     \x20 -h, --help              show this help\n\
     \x20 -v, --version           show version\n\
     \x20 -s, --seed N            random seed (0 = random, default 0)\n\
     \x20 -o, --out-dir DIR       output directory (default .)\n\
     \x20 --std clj|cljs|cljc     target dialect (default clj)\n\
     \x20 --check-algo hash|asserts\n\
     \x20                         oracle kind (default hash)\n\
     \x20 --emit-pragmas none|some|all\n\
     \x20                         loop-binding hint density (default some)\n\
     \x20 --param-shuffle true|false\n\
     \x20                         randomize policy weights (default true)\n\
     \x20 --expl-loop-param true|false\n\
     \x20                         explore wide loop bounds (default false)\n\
     \x20 --mutate none|exprs|all mutate a generated program (default none)\n\
     \x20 --mutation-seed N       mutation seed (0 = random, default 0)\n\
     \x20 --allow-exc-in-dead none|some|all\n\
     \x20                         allow exceptions in dead branches (default none)\n\
     \x20 --max-array-dims N      vector length (0 = default 8, capped at 65536)\n\
     \x20 --max-nesting N         loop/if depth cap, negative-suite depth (0 = defaults)\n\
     \x20 --simple-loops none|some|all\n\
     \x20                         analyzer-friendly profile (default none)\n\
     \x20 --unchecked-math on|off compile test-fn unchecked (default off)\n\
     \x20 --dg-headers true|false emit DejaGNU dg directives (default false)\n\
     \x20 --negative none|arity|recur|nesting|hints|all\n\
     \x20                         negative (must-fail) suite (default none)\n\
     \x20 --count N               programs to generate (default 1)\n\
     \x20 --max-fn-cost N         cost units per function, 0 = default 60\n\
     \x20                         (raise toward the 64KB ceiling deliberately)\n\
     \x20 --top-stmts N           top-level statements, 0 = policy roll 2-5\n"
}

fn parse_density(s: &str) -> Option<Density> {
    match s {
        "none" => Some(Density::None),
        "some" => Some(Density::Some),
        "all" => Some(Density::All),
        _ => None,
    }
}

fn set_option(opts: &mut Options, key: &str, val: &str, errors: &mut Vec<String>) {
    match key {
        "seed" => match val.parse::<u64>() {
            Ok(n) => opts.seed = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --seed")),
        },
        "mutation-seed" => match val.parse::<u64>() {
            Ok(n) => opts.mutation_seed = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --mutation-seed")),
        },
        "max-fn-cost" => match val.parse::<u64>() {
            Ok(n) => opts.max_fn_cost = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --max-fn-cost")),
        },
        "max-array-dims" => match val.parse::<i64>() {
            Ok(n) => opts.max_array_dims = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --max-array-dims")),
        },
        "max-nesting" => match val.parse::<i64>() {
            Ok(n) => opts.max_nesting = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --max-nesting")),
        },
        "count" => match val.parse::<i64>() {
            Ok(n) => opts.count = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --count")),
        },
        "top-stmts" => match val.parse::<i64>() {
            Ok(n) => opts.top_stmts = n,
            Err(_) => errors.push(format!("invalid integer '{val}' for --top-stmts")),
        },
        "out-dir" => opts.out_dir = val.to_string(),
        "std" => match val {
            "clj" => opts.std = Std::Clj,
            "cljs" => opts.std = Std::Cljs,
            "cljc" => opts.std = Std::Cljc,
            _ => errors.push(format!("invalid value '{val}' for --std")),
        },
        "check-algo" => match val {
            "hash" => opts.check_algo = CheckAlgo::Hash,
            "asserts" => opts.check_algo = CheckAlgo::Asserts,
            _ => errors.push(format!("invalid value '{val}' for --check-algo")),
        },
        "emit-pragmas" => match parse_density(val) {
            Some(d) => opts.emit_pragmas = d,
            None => errors.push(format!("invalid value '{val}' for --emit-pragmas")),
        },
        "mutate" => match val {
            "none" => opts.mutate = MutateMode::None,
            "exprs" => opts.mutate = MutateMode::Exprs,
            "all" => opts.mutate = MutateMode::All,
            _ => errors.push(format!("invalid value '{val}' for --mutate")),
        },
        "allow-exc-in-dead" => match parse_density(val) {
            Some(d) => opts.allow_exc_in_dead = d,
            None => errors.push(format!("invalid value '{val}' for --allow-exc-in-dead")),
        },
        "simple-loops" => match parse_density(val) {
            Some(d) => opts.simple_loops = d,
            None => errors.push(format!("invalid value '{val}' for --simple-loops")),
        },
        "unchecked-math" => match val {
            "on" => opts.unchecked_math = OnOff::On,
            "off" => opts.unchecked_math = OnOff::Off,
            _ => errors.push(format!("invalid value '{val}' for --unchecked-math")),
        },
        "negative" => match val {
            "none" => opts.negative = NegativeKind::None,
            "arity" => opts.negative = NegativeKind::Arity,
            "recur" => opts.negative = NegativeKind::Recur,
            "nesting" => opts.negative = NegativeKind::Nesting,
            "hints" => opts.negative = NegativeKind::Hints,
            "all" => opts.negative = NegativeKind::All,
            _ => errors.push(format!("invalid value '{val}' for --negative")),
        },
        "param-shuffle" => match val {
            "true" => opts.param_shuffle = true,
            "false" => opts.param_shuffle = false,
            _ => errors.push(format!("invalid value '{val}' for --param-shuffle")),
        },
        "expl-loop-param" => match val {
            "true" => opts.expl_loop_param = true,
            "false" => opts.expl_loop_param = false,
            _ => errors.push(format!("invalid value '{val}' for --expl-loop-param")),
        },
        "dg-headers" => match val {
            "true" => opts.dg_headers = true,
            "false" => opts.dg_headers = false,
            _ => errors.push(format!("invalid value '{val}' for --dg-headers")),
        },
        _ => errors.push(format!("unknown option --{key}")),
    }
}

/// Parse argv into options, errors, and help/version flags.
pub fn parse_args(args: &[String]) -> Parsed {
    let mut options = Options::default();
    let mut errors = Vec::new();
    let mut help = false;
    let mut version = false;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "-h" || a == "--help" {
            help = true;
            i += 1;
        } else if a == "-v" || a == "--version" {
            version = true;
            i += 1;
        } else if a == "-s" || a == "-o" {
            let key = if a == "-s" { "seed" } else { "out-dir" };
            if i + 1 >= args.len() {
                errors.push(format!("{a} requires a value"));
                i += 1;
            } else {
                set_option(&mut options, key, &args[i + 1], &mut errors);
                i += 2;
            }
        } else if let Some(body) = a.strip_prefix("--") {
            if let Some((k, v)) = body.split_once('=') {
                set_option(&mut options, k, v, &mut errors);
                i += 1;
            } else if i + 1 >= args.len() {
                errors.push(format!("{a} requires a value"));
                i += 1;
            } else {
                set_option(&mut options, body, &args[i + 1], &mut errors);
                i += 2;
            }
        } else {
            errors.push(format!("unexpected argument '{a}'"));
            i += 1;
        }
    }
    Parsed {
        options,
        errors,
        help,
        version,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(words: &[&str]) -> Parsed {
        parse_args(&words.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn defaults_hold() {
        let p = parse(&[]);
        assert!(p.errors.is_empty());
        assert_eq!(p.options.seed, 0);
        assert!(p.options.param_shuffle);
    }

    #[test]
    fn short_and_long_forms() {
        let p = parse(&["-s", "42", "--out-dir=/tmp/x", "--check-algo", "asserts"]);
        assert!(p.errors.is_empty());
        assert_eq!(p.options.seed, 42);
        assert_eq!(p.options.out_dir, "/tmp/x");
        assert_eq!(p.options.check_algo, CheckAlgo::Asserts);
    }

    #[test]
    fn invalid_values_reported() {
        let p = parse(&["--std", "java", "--seed", "abc", "--bogus", "1"]);
        assert_eq!(p.errors.len(), 3);
    }
}
