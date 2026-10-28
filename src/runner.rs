// Licensed under the Apache License, Version 2.0.
//! Differential runner library: run one generated test across a backend
//! matrix (JIT tiers) and compare every observed hash to the
//! generator-precomputed expected hash. Ports YARPGen's run_gen.py
//! TestRun flow. Used programmatically; see `main.rs --verify`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// One execution backend. Disagreement between backends signals a
/// miscompilation of the generated program: default JIT (C2) vs
/// C1-only vs direct-linked calls.
#[derive(Clone, Debug)]
pub struct Backend {
    pub name: &'static str,
    pub jvm_opts: Vec<&'static str>,
}

pub fn backends() -> Vec<Backend> {
    vec![
        Backend {
            name: "default",
            jvm_opts: vec![],
        },
        Backend {
            name: "c1-only",
            jvm_opts: vec!["-XX:TieredStopAtLevel=1"],
        },
        Backend {
            name: "direct-link",
            jvm_opts: vec!["-Dclojure.compiler.direct-linking=true"],
        },
    ]
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
}

/// Parse the `expected.edn` manifest. The file has a fixed shape:
/// `{:seed N :expected-hash H :check-algo :hash}`.
pub fn read_expected(dir: &Path) -> Option<Expected> {
    let text = fs::read_to_string(dir.join("expected.edn")).ok()?;
    let seed = field(&text, ":seed")?.parse().ok()?;
    let expected_hash = field(&text, ":expected-hash")?.parse().ok()?;
    let check_algo = field(&text, ":check-algo")?
        .trim_start_matches(':')
        .to_string();
    Some(Expected {
        seed,
        expected_hash,
        check_algo,
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

/// Run one generated test directory in a child JVM.
pub fn run_test(dir: &Path, classpath: &str, backend: &Backend, timeout: Duration) -> RunResult {
    let mut cmd = Command::new("java");
    for opt in &backend.jvm_opts {
        cmd.arg(opt);
    }
    cmd.arg("-cp")
        .arg(classpath)
        .arg("clojure.main")
        .arg("-i")
        .arg("init.clj")
        .arg("-i")
        .arg("func.clj")
        .arg("-i")
        .arg("driver.clj")
        .arg("-e")
        .arg("(gen-driver/-main)")
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
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

/// Parse the driver's `"<hash> <seed>"` line.
fn parse_hash_line(output: &str) -> (Option<i64>, Option<i64>) {
    for line in output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            if let (Ok(h), Ok(s)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                return (Some(h), Some(s));
            }
        }
    }
    (None, None)
}

#[derive(Debug)]
pub struct Verdict {
    pub pass: bool,
    pub expected: i64,
    pub runs: Vec<RunResult>,
}

/// Run `dir` across backends and compare to the expected hash.
pub fn verify(
    dir: &Path,
    classpath: &str,
    timeout: Duration,
    only: Option<&[String]>,
) -> Option<Verdict> {
    let expected = read_expected(dir)?;
    let selected: Vec<Backend> = backends()
        .into_iter()
        .filter(|b| {
            only.map(|names| names.iter().any(|n| n == b.name))
                .unwrap_or(true)
        })
        .collect();
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
    })
}

/// Print a human-readable verification report. Returns pass/fail.
pub fn report(verdict: &Verdict) -> bool {
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
    let mut cp = vec![clojure];
    for lib in ["spec.alpha", "core.specs.alpha"] {
        if let Some(jar) = newest_jar(&m2.join(lib)) {
            cp.push(jar);
        }
    }
    Some(
        cp.iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(":"),
    )
}

fn newest_jar(dir: &Path) -> Option<PathBuf> {
    let mut jars: Vec<PathBuf> = Vec::new();
    collect_jars(dir, &mut jars);
    jars.sort();
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
    }

    #[test]
    fn parses_expected_manifest() {
        let dir = std::env::temp_dir().join("cy-expected-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("expected.edn"),
            "{:seed 42 :expected-hash -5 :check-algo :hash}\n",
        )
        .unwrap();
        let e = read_expected(&dir).unwrap();
        assert_eq!(e.seed, 42);
        assert_eq!(e.expected_hash, -5);
        assert_eq!(e.check_algo, "hash");
        let _ = fs::remove_dir_all(&dir);
    }
}
