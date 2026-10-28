# clojure-yarpgen

A random Clojure program generator for compiler stress testing, in the
spirit of [Intel YARPGen](https://github.com/intel/yarpgen). It emits
deterministic, exception-free Clojure programs, runs them across JVM
JIT tiers, and compares every observed checksum against a
generator-precomputed expected hash. A companion negative suite probes
hard compiler limits (arity, `recur` rules, nesting depth) with
programs that must fail.

The tool itself is written in Rust and has zero dependencies. The
generated programs need only Clojure on the JVM to run.

## Quickstart

```sh
cargo build --release
./target/release/clojure-yarpgen -s 42 -o out/smoke
./target/release/clojure-yarpgen verify out/smoke
```

`verify` runs the test on every backend (default JIT and C1-only) and
prints `PASS` when all observed hashes equal the expected hash:

```
[default] exit=0 hash=-2585397846746023539 timed-out=false
[c1-only] exit=0 hash=-2585397846746023539 timed-out=false
expected=-2585397846746023539 PASS
```

`verify` locates Clojure via `CLOJURE_CP`, `CLOJURE_JAR`, or the newest
jars under `~/.m2`.

## What it generates

Each test directory holds four files, mirroring YARPGen's layout:

| File           | Contents                                                     |
| -------------- | ------------------------------------------------------------ |
| `init.clj`     | input `def`s: scalars plus a few vectors                      |
| `func.clj`     | output atoms, `test-fn-N` chunk functions, `outputs` reader  |
| `driver.clj`   | runs `test-fn`, prints `<hash> <seed>` (or asserts values)   |
| `expected.edn` | `{:seed … :expected-hash … :check-algo …}` manifest           |

`test-fn` is loop nests (`doseq` over `range`), `if`/`when`
branches, and atom assignments over integer arithmetic, bit
operations, comparisons, `min`/`max`, and vector `nth`. Top-level
statements are chunked into sequential `test-fn-N` functions so no
single method nears the 64KB JVM limit. Example fragment:

```clojure
(reset! out-1 (min (deref out-1) (nth vec-1 (mod (bit-not 60) 7))))
(doseq [^long i_0_1_1_nest0 (range 0 4 2)]
  (doseq [i_0_1_1_nest1 (range 0 4 4)]
    (reset! out-10 (* (deref out-10) -62))))
```

## Generation model

Like YARPGen, every variable's type and value is tracked at
generation time:

- **Evaluate, then rebuild.** Each expression is evaluated while it is
  built. Faults (long overflow, division by zero, out-of-bounds,
  out-of-int-range `nth` indices) trigger operator swaps (`+`↔`-`,
  `*`→`quot`, faulty negation→identity, index→floored `mod` wrap or
  `nth` default) and regeneration until the whole tree is clean.
  Faults propagate from any nested node to the root.
- **Loops are simulated.** Bodies execute in the model across all
  iterations with evolving values, so loop-carried overflow fails
  validation and regenerates the body. A simulation budget shrinks
  pathological bounds first; the emitted bounds always equal the
  simulated ones.
- **Branches track taken-ness.** `if` conditions are evaluated at
  generation time; the live branch threads values forward. Both
  branches are still populated.
- **JVM-exact semantics.** The model matches probed Clojure/JVM
  behavior: `0` is truthy, shifts mask to 6 bits, `rem`/`quot`
  truncate, `nth` narrows checked via `toIntExact` in both arities.

### Default policy limits

Ported from YARPGen's probability tables:

| Knob                 | Default                                  |
| -------------------- | ---------------------------------------- |
| statements           | ≤ 1000, scopes of 2–5                    |
| loop nests           | depth 2–3 (cap 5), bounds 10–25, step 1–4|
| `if`/`else`          | depth ≤ 5, else-branch 20%               |
| expression depth     | ≤ 4                                      |
| inputs/outputs       | 10–20 scalars + 2 vectors                |
| assign vs reduction  | 70 / 30                                  |
| SimilarOperators     | 10% single-family expressions            |
| ConstUse             | 10% all-const leaves                     |
| special constants    | 30% (0, min/max, bit patterns)           |

`--param-shuffle=false` freezes the weights; `--simple-loops=all`
selects a flat, analyzer-friendly profile.

## Oracle

- `--check-algo hash` (default): boost-style hash combine over the
  outputs, printed with the seed. Every backend must reproduce the
  precomputed hash exactly — a mismatch is either a miscompilation or
  a generator soundness bug (file it).
- `--check-algo asserts`: the driver compares each output to its
  expected value and prints `TORTURE-PASS` or `ERROR: value mismatch`.

The backend matrix is `{default JIT, C1-only
(-XX:TieredStopAtLevel=1)}`: disagreement between JIT tiers signals a
JIT bug in the generated shape. `verify DIR [backend...]` selects a
subset.

## CLI reference

```
-s, --seed N            random seed (0 = random, default 0)
-o, --out-dir DIR       output directory (default .)
--std clj|cljs|cljc     file dialect (default clj; execution is JVM-only in v0.1)
--check-algo hash|asserts
--emit-pragmas none|some|all   loop-binding ^long hint density (default some)
--param-shuffle true|false     (default true)
--expl-loop-param true|false   wide loop bounds 1-64 (default false)
--mutate none|exprs|all        re-roll RHS exprs (+ loop bounds for all)
--mutation-seed N
--allow-exc-in-dead none|some|all  keep faults in dead branches (default none)
--max-array-dims N      vector-length cap (0 = default 8)
--max-nesting N         nesting guard (0 = default 64)
--simple-loops none|some|all
--unchecked-math on|off compile test-fn with *unchecked-math* (default off)
--dg-headers true|false DejaGNU dg directives for torture-suite interop
--negative none|arity|recur|nesting|hints|all  negative suite (default none)
--count N               programs to generate as seed-N subdirs (default 1)
verify DIR [backend...] run a generated test across backends
```

`--unchecked-math on` is a generator self-check: emitted programs
avoid overflow entirely, so checked and unchecked runs must agree —
disagreement means the generator let an overflow through.

## Negative suite

`--negative KIND` writes must-fail programs plus a `manifest.edn`
with expected errors:

| Kind    | Cases                                                        |
| ------- | ------------------------------------------------------------ |
| `arity` | >20 params, double variadic overloads                        |
| `recur` | non-tail `recur`, `recur` across `try`, arity mismatch       |
| `nesting`| deep expression/`let` ladders (expect `StackOverflowError`) |
| `hints` | primitive+variadic, wrongly-hinted locals                    |
| `all`   | everything                                                   |

## Reproducing a failure

Every artifact records its seed:

```sh
./target/release/clojure-yarpgen -s 1234 -o out/repro
./target/release/clojure-yarpgen verify out/repro
```

Same seed + same flags = byte-identical output. Bisect by re-running
with `--param-shuffle=false`, `--simple-loops=all`, or a targeted
`--mutate` seed.

## Guarantees and known limits

Valid-suite programs are deterministic and exception-free **on the
JVM**; the model pins JVM semantics (6-bit shift masking, checked
`nth` narrowing, truthiness). Cross-dialect (Clojure vs ClojureScript)
differential testing is out of scope for v0.1: `--std cljs|cljc`
currently selects the file extension only, and shift-heavy programs
would need a 32-bit shift model to compare across dialects.

## Credits

Design follows [YARPGen](https://github.com/intel/yarpgen) (Livinskii
et al., OOPSLA 2020 and PLDI 2023) and the differential-testing
approach of [Csmith](https://github.com/csmith-project/csmith) (Yang
et al., PLDI 2011), adapted to Clojure's semantics: atoms for mutable
outputs, `doseq` loops, no-UB as exception-freedom, and the JVM method
size ceiling as a first-class generation constraint.
