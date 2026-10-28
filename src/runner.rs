// Licensed under the Apache License, Version 2.0.
//! Differential runner library: run one generated test across a backend
//! matrix (JIT tiers) and compare every observed hash to the
//! generator-precomputed expected hash. Ports YARPGen's run_gen.py
//! TestRun flow. Used programmatically; see `main.rs --verify`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// One execution backend. Disagreement between backends signals a
/// miscompilation of the generated program: default JIT (C2) vs
/// C1-only vs direct-linked calls vs the Babashka interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Jvm,
    Bb,
}

#[derive(Clone, Debug)]
pub struct Backend {
    pub name: &'static str,
    pub kind: BackendKind,
    pub jvm_opts: Vec<&'static str>,
}

pub fn backends() -> Vec<Backend> {
    vec![
        Backend {
            name: "default",
            kind: BackendKind::Jvm,
            jvm_opts: vec![],
        },
        Backend {
            name: "c1-only",
            kind: BackendKind::Jvm,
            jvm_opts: vec!["-XX:TieredStopAtLevel=1"],
        },
        Backend {
            name: "direct-link",
            kind: BackendKind::Jvm,
            jvm_opts: vec!["-Dclojure.compiler.direct-linking=true"],
        },
        Backend {
            name: "bb",
            kind: BackendKind::Bb,
            jvm_opts: vec![],
        },
    ]
}

/// True when Babashka is installed and runs: gates the `bb` backend.
/// The probe answer is process-constant, so it runs once and caches;
/// fuzz calls `verify` per seed and must not respawn `bb` every time.
pub fn bb_available() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(bb_probe)
}

/// One uncached `bb --version` probe. A short deadline keeps a wedged
/// `bb` from hanging verification.
fn bb_probe() -> bool {
    let mut child = match Command::new("bb")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => return false,
        }
    }
}

/// Backends to run: `only` filters by name; `bb` joins the default
/// set only when installed, but an explicit `bb` request always
/// attempts (failing honestly if the binary is missing).
pub fn active_backends(only: Option<&[String]>, bb_present: bool) -> Vec<Backend> {
    let explicit_bb = only
        .map(|names| names.iter().any(|n| n == "bb"))
        .unwrap_or(false);
    backends()
        .into_iter()
        .filter(|b| {
            let named = only
                .map(|names| names.iter().any(|n| n == b.name))
                .unwrap_or(true);
            named && (b.kind != BackendKind::Bb || bb_present || explicit_bb)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub backend: String,
    pub exit: i32,
    pub hash: Option<i64>,
    pub seed: Option<i64>,
    pub torture_pass: bool,
    pub output: String,
    pub error: String,
    pub timed_out: bool,
}

#[derive(Clone, Debug)]
pub struct Expected {
    pub seed: u64,
    pub expected_hash: i64,
    pub check_algo: String,
    pub std: String,
}

/// Parse the `expected.edn` manifest. The file has a fixed shape:
/// `{:seed N :expected-hash H :check-algo :hash :std :clj}`; the
/// `:std` key defaults to `clj` for manifests written before it.
pub fn read_expected(dir: &Path) -> Option<Expected> {
    let text = fs::read_to_string(dir.join("expected.edn")).ok()?;
    let seed = field(&text, ":seed")?.parse().ok()?;
    let expected_hash = field(&text, ":expected-hash")?.parse().ok()?;
    let check_algo = field(&text, ":check-algo")?
        .trim_start_matches(':')
        .to_string();
    let std = field(&text, ":std")
        .map(|s| s.trim_start_matches(':').to_string())
        .unwrap_or_else(|| "clj".to_string());
    Some(Expected {
        seed,
        expected_hash,
        check_algo,
        std,
    })
}

fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let start = text.find(key)? + key.len();
    let rest = text[start..].trim_start();
    Some(
        rest.split([' ', '\n', '\r', '}', '\t'])
            .next()
            .unwrap_or(""),
    )
}

/// Run one generated test directory in a child JVM. File names
/// follow the manifest's `:std` so `--std cljc` tests resolve; CLJS
/// has no JVM execution backend and fails with a clear message.
pub fn run_test(dir: &Path, classpath: &str, backend: &Backend, timeout: Duration) -> RunResult {
    // Absolute paths: joined file names must survive current_dir(dir).
    let owned;
    let dir = match dir.canonicalize() {
        Ok(abs) => {
            owned = abs;
            &owned
        }
        Err(_) => dir,
    };
    let std = read_expected(dir)
        .map(|e| e.std)
        .unwrap_or_else(|| "clj".to_string());
    if std == "cljs" {
        return RunResult {
            backend: backend.name.to_string(),
            exit: -1,
            hash: None,
            seed: None,
            torture_pass: false,
            output: String::new(),
            error: "execution is JVM-only; cannot verify .cljs output".to_string(),
            timed_out: false,
        };
    }
    let dialect = match std.as_str() {
        "cljc" => crate::cli::Std::Cljc,
        _ => crate::cli::Std::Clj,
    };
    let paths = crate::emitter::output_paths(dir, dialect);
    let mut cmd = if backend.kind == BackendKind::Bb {
        bb_command(dir, &paths)
    } else {
        let mut cmd = Command::new("java");
        for opt in &backend.jvm_opts {
            cmd.arg(opt);
        }
        cmd.arg("-cp")
            .arg(classpath)
            .arg("clojure.main")
            .arg("-i")
            .arg(&paths.init)
            .arg("-i")
            .arg(&paths.func)
            .arg("-i")
            .arg(&paths.driver)
            .arg("-e")
            .arg("(gen-driver/-main)")
            .current_dir(dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd
    };
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return RunResult {
                backend: backend.name.to_string(),
                exit: -1,
                hash: None,
                seed: None,
                torture_pass: false,
                output: String::new(),
                error: format!("spawn failed: {e}"),
                timed_out: false,
            };
        }
    };
    // Drain both pipes on background threads from the start: a verbose
    // compiler crash can otherwise fill a pipe buffer, block the child,
    // and turn into a bogus 120s timeout.
    let out_handle = child.stdout.take().map(|mut out| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = out.read_to_string(&mut buf);
            buf
        })
    });
    let err_handle = child.stderr.take().map(|mut err| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = err.read_to_string(&mut buf);
            buf
        })
    });
    let deadline = Instant::now() + timeout;
    let exit = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status.code().unwrap_or(-1)),
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    // Reap the killed child: dropping without wait
                    // leaks a zombie until the fuzzer itself exits.
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => break None,
        }
    };
    let output = out_handle.and_then(|h| h.join().ok()).unwrap_or_default();
    let error = err_handle.and_then(|h| h.join().ok()).unwrap_or_default();
    let (hash, seed) = parse_hash_line(&output);
    let torture_pass = output.lines().any(|l| l.trim() == "TORTURE-PASS");
    RunResult {
        backend: backend.name.to_string(),
        exit: exit.unwrap_or(-1),
        hash,
        seed,
        torture_pass,
        output,
        error,
        timed_out: exit.is_none(),
    }
}

/// Babashka invocation: loads the three test files and runs the
/// driver, exactly like the JVM `-i` chain. Needs no classpath.
fn bb_command(dir: &Path, paths: &crate::emitter::EmitPaths) -> Command {
    let script = format!(
        "(load-file \"{}\") (load-file \"{}\") (load-file \"{}\") (gen-driver/-main)",
        paths.init.display(),
        paths.func.display(),
        paths.driver.display()
    );
    let mut cmd = Command::new("bb");
    cmd.arg("-e")
        .arg(script)
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Parse the driver's `"<hash> <seed>"` line. The last two-number
/// line wins: the driver's println comes after any JVM chatter.
fn parse_hash_line(output: &str) -> (Option<i64>, Option<i64>) {
    let mut found = (None, None);
    for line in output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            if let (Ok(h), Ok(s)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                found = (Some(h), Some(s));
            }
        }
    }
    found
}

#[derive(Debug)]
pub struct Verdict {
    pub pass: bool,
    pub expected: i64,
    pub runs: Vec<RunResult>,
    pub skipped: Vec<String>,
}

/// Run `dir` across backends and compare to the expected hash.
pub fn verify(
    dir: &Path,
    classpath: &str,
    timeout: Duration,
    only: Option<&[String]>,
) -> Option<Verdict> {
    let expected = read_expected(dir)?;
    // Probe only for the default matrix: an explicit backend list
    // never needs the answer (`active_backends` always attempts an
    // explicitly named `bb`, and ignores `bb` otherwise).
    let bb_present = only.is_none() && bb_available();
    let selected = active_backends(only, bb_present);
    let skipped = if only.is_none() && !bb_present {
        vec!["bb backend skipped (not installed)".to_string()]
    } else {
        Vec::new()
    };
    let runs: Vec<RunResult> = selected
        .iter()
        .map(|b| run_test(dir, classpath, b, timeout))
        .collect();
    let pass = if expected.check_algo == "asserts" {
        !runs.is_empty() && runs.iter().all(|r| r.exit == 0 && r.torture_pass)
    } else {
        !runs.is_empty()
            && runs.iter().all(|r| r.exit == 0)
            && runs.iter().all(|r| r.hash == Some(expected.expected_hash))
            && runs.iter().all(|r| r.seed == Some(expected.seed as i64))
    };
    Some(Verdict {
        pass,
        expected: expected.expected_hash,
        runs,
        skipped,
    })
}

/// Print a human-readable verification report. Returns pass/fail.
pub fn report(verdict: &Verdict) -> bool {
    for s in &verdict.skipped {
        println!("note: {s}");
    }
    for r in &verdict.runs {
        println!(
            "[{}] exit={} hash={} timed-out={}",
            r.backend,
            r.exit,
            r.hash
                .map(|h| h.to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.timed_out
        );
        if r.exit != 0 && !r.output.trim().is_empty() {
            println!("--- stdout ---\n{}", r.output.trim());
        }
        if !r.error.trim().is_empty() {
            println!("--- stderr ---\n{}", r.error.trim());
        }
    }
    println!(
        "expected={} {}",
        verdict.expected,
        if verdict.pass { "PASS" } else { "FAIL" }
    );
    verdict.pass
}

/// Locate a Clojure classpath for the child JVM: `CLOJURE_JAR` (or a
/// full `CLOJURE_CP`) wins, else the newest `clojure-*.jar` plus its
/// spec jars under `~/.m2`.
pub fn find_clojure_classpath() -> Option<String> {
    if let Ok(cp) = std::env::var("CLOJURE_CP") {
        return Some(cp);
    }
    if let Ok(jar) = std::env::var("CLOJURE_JAR") {
        return Some(jar);
    }
    let m2 = dirs_m2();
    let clojure = newest_jar(&m2.join("clojure"))?;
    // Missing spec jars yield a classpath that dies in
    // ExceptionInInitializerError; fail discovery instead, with the
    // CLOJURE_JAR/CLOJURE_CP escape hatch intact. Releases before
    // 1.9 predate the spec dependency and run standalone.
    let mut cp = vec![clojure.clone()];
    for lib in ["spec.alpha", "core.specs.alpha"] {
        match newest_jar(&m2.join(lib)) {
            Some(jar) => cp.push(jar),
            None if clojure_needs_specs(&clojure) => return None,
            None => {}
        }
    }
    Some(
        std::env::join_paths(cp.iter())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|_| {
                cp.iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(":")
            }),
    )
}

/// Numeric version components from a jar path, for ordering cached
/// artifacts: `.../1.12.0/clojure-1.12.0.jar` sorts after
/// `.../1.9.0/clojure-1.9.0.jar`, which a plain string sort gets
/// wrong ("9" > "1").
fn jar_version(path: &Path) -> Vec<u64> {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let core = name.rsplit('-').next().unwrap_or("");
    let core = core
        .strip_suffix("SNAPSHOT")
        .unwrap_or(core)
        .trim_end_matches(['.', '-']);
    core.split('.')
        .map(|p| p.parse::<u64>().unwrap_or(0))
        .collect()
}

fn jar_cmp(a: &Path, b: &Path) -> std::cmp::Ordering {
    jar_version(a).cmp(&jar_version(b)).then_with(|| a.cmp(b))
}

/// True when a Clojure release needs the spec jars beside it:
/// spec.alpha and core.specs.alpha became mandatory in 1.9.
/// Unparseable names stay lenient (specs optional, as before).
fn clojure_needs_specs(jar: &Path) -> bool {
    jar_version(jar) >= vec![1, 9]
}

fn newest_jar(dir: &Path) -> Option<PathBuf> {
    let mut jars: Vec<PathBuf> = Vec::new();
    collect_jars(dir, &mut jars);
    jars.sort_by(|a, b| jar_cmp(a, b));
    jars.pop()
}

fn dirs_m2() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".m2/repository/org/clojure")
}

fn collect_jars(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_jars(&path, out);
        } else if path.extension().map(|e| e == "jar").unwrap_or(false)
            && !path.to_string_lossy().contains("sources")
            && !path.to_string_lossy().contains("javadoc")
        {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hash_line() {
        assert_eq!(parse_hash_line("12345 42\n"), (Some(12345), Some(42)));
        assert_eq!(parse_hash_line("no numbers here"), (None, None));
        assert_eq!(parse_hash_line("-7 9"), (Some(-7), Some(9)));
        // The driver's println comes last; earlier chatter loses.
        assert_eq!(parse_hash_line("9 9\n3 4"), (Some(3), Some(4)));
    }

    #[test]
    fn parses_expected_manifest() {
        let dir = std::env::temp_dir().join("cy-expected-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("expected.edn"),
            "{:seed 42 :expected-hash -5 :check-algo :hash :std :cljc}\n",
        )
        .unwrap();
        let e = read_expected(&dir).unwrap();
        assert_eq!(e.seed, 42);
        assert_eq!(e.expected_hash, -5);
        assert_eq!(e.check_algo, "hash");
        assert_eq!(e.std, "cljc");
        // Manifests written before :std default to clj.
        fs::write(
            dir.join("expected.edn"),
            "{:seed 42 :expected-hash -5 :check-algo :hash}\n",
        )
        .unwrap();
        assert_eq!(read_expected(&dir).unwrap().std, "clj");
        let _ = fs::remove_dir_all(&dir);
    }

    fn names(backends: &[Backend]) -> Vec<&str> {
        backends.iter().map(|b| b.name).collect()
    }

    #[test]
    fn active_backends_gate_bb_on_availability() {
        // Default set: bb joins only when installed.
        assert_eq!(
            names(&active_backends(None, true)),
            vec!["default", "c1-only", "direct-link", "bb"]
        );
        assert_eq!(
            names(&active_backends(None, false)),
            vec!["default", "c1-only", "direct-link"]
        );
        // Explicit bb always attempts (failing honestly if missing).
        let bb = vec!["bb".to_string()];
        assert_eq!(names(&active_backends(Some(&bb), false)), vec!["bb"]);
        assert_eq!(names(&active_backends(Some(&bb), true)), vec!["bb"]);
        // Other filters never drag bb along.
        let jvm = vec!["default".to_string()];
        assert_eq!(names(&active_backends(Some(&jvm), true)), vec!["default"]);
    }

    #[test]
    fn bb_command_loads_all_three_files() {
        let dir = PathBuf::from("/tmp/t");
        let paths = crate::emitter::output_paths(&dir, crate::cli::Std::Clj);
        let cmd = bb_command(&dir, &paths);
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert_eq!(args[0], "-e");
        assert!(args[1].contains("init.clj"), "{}", args[1]);
        assert!(args[1].contains("func.clj"), "{}", args[1]);
        assert!(args[1].contains("driver.clj"), "{}", args[1]);
        assert!(args[1].contains("(gen-driver/-main)"), "{}", args[1]);
        assert_eq!(cmd.get_current_dir(), Some(dir.as_path()));
    }

    #[test]
    fn spec_requirement_follows_clojure_1_9() {
        assert!(!clojure_needs_specs(&PathBuf::from(
            "/m2/clojure/1.8.0/clojure-1.8.0.jar"
        )));
        assert!(clojure_needs_specs(&PathBuf::from(
            "/m2/clojure/1.9.0/clojure-1.9.0.jar"
        )));
        assert!(clojure_needs_specs(&PathBuf::from(
            "/m2/clojure/1.12.0/clojure-1.12.0.jar"
        )));
        assert!(!clojure_needs_specs(&PathBuf::from("/m2/clojure.jar")));
    }

    #[test]
    fn jar_ordering_compares_versions_numerically() {
        use std::cmp::Ordering;
        let v19 = PathBuf::from("/m2/clojure/1.9.0/clojure-1.9.0.jar");
        let v112 = PathBuf::from("/m2/clojure/1.12.0/clojure-1.12.0.jar");
        assert_eq!(jar_cmp(&v19, &v112), Ordering::Less);
        assert_eq!(jar_cmp(&v112, &v19), Ordering::Greater);
        let spec = PathBuf::from("/m2/spec.alpha/0.5.238/spec.alpha-0.5.238.jar");
        assert_eq!(jar_version(&spec), vec![0, 5, 238]);
    }
}
