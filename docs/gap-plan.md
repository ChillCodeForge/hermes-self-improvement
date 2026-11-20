# Gap plan: from foundations to a complete improvement loop

Status date: 2026-11-15. Baseline: `main` at `fea6f9b`, 169 passing tests
(`scripts/docker-test.sh`).

This plan compares the full target specification (closed improvement loop,
benchmark-growth loop, strict Generator/Evaluator/Promoter separation) with the
code that exists today and lists the work that closes each gap. The README
"Phased delivery" table stays the phase-level roadmap; this document is the
work-level breakdown. Items are marked **done** only when code and tests exist.

## Decisions taken for this plan

| Topic | Decision | Reason |
|---|---|---|
| Package layout | Keep the existing flat `hermes_improve/*.py` modules; add new modules beside them instead of moving everything into sub-packages. | A mass move would break every open branch and give no behavioral gain. The spec layout is marked "approximate". |
| Benchmark schema | Add a Pydantic model as the published schema (JSON Schema export, YAML/JSON loading) while keeping the existing strict semantic validation as the authority. | Spec asks for Pydantic + JSON/YAML; existing checks (provenance, confidence, leakage) are stricter than a type schema. |
| Config | YAML files under `config/` (`promotion.yaml`, `scoring.yaml`, `taxonomy.yaml`, `models.yaml`) loaded by a single `config.py`; code defaults stay identical to today's constants. | Thresholds must be configurable but the Generator must not be able to change them, so config is read from the protected checkout, not from candidates. |
| Test platform | POSIX only (fcntl, Unix users, symlinks). On Windows dev boxes use `scripts/docker-test.sh`; CI runs on `ubuntu-latest`. | Matches the OS-isolation target. |
| External optimizers | GEPA / DSPy / hermes-agent-self-evolution are optional extras behind adapters with a pinned version check; the built-in strategy works without them. | Experimental external APIs must not leak into the core. |
| Qwen | No model name hardcoded; all roles configured through `QWEN_*` env or `config/models.yaml`. Live endpoint tests are opt-in. | No guaranteed local endpoint in CI. |

## Gap matrix

| Spec area | Exists today | Gap → work package |
|---|---|---|
| §8 Production brain repo layout | skills-only promoter | `brain/` tree for all artifact areas, seeded baseline skill → **WP-A** |
| §9 Git workflow | worktree + branch per skill candidate | `candidate/<experiment-id>` naming, diff hash + parent baseline for every type → **WP-A** |
| §10 Candidate types | enum + auto-promotion set for skills | target-path mapping for all 12 types, manual approval gate (signed operator approval) for non-auto types, configurable unlock → **WP-A** |
| §11–13 Benchmark store/schema/eligibility | JSON store, splits incl. SHADOW, eligibility enum | Pydantic schema + YAML, reproducibility score 0–1 from the nine factors, eligibility classifier → **WP-B** |
| §15 Dedup | exact + lexical near-dup | MinHash/TF-IDF semantic clustering, cluster-consistent split assignment → **WP-B** |
| §16 Holdout rotation | spent registry | operator-managed generation rotation (issue #7) → **WP-B** |
| §17 Suites | suite membership check | SMOKE/CORE/FULL builders, impact-aware selection by candidate target/capabilities → **WP-B** |
| Curated corpus | 1 schema fixture | 20–50 deterministic command benchmarks with fixtures across categories → **WP-B** |
| §19 Verifiers | 5 verifiers | Command, UnitTest, IntegrationTest, Build, Lint, Typecheck, UnchangedPath, GitDiff, HTTPResponse, MCPHealth + spec-driven registry → **WP-C** |
| §18/21 Evaluation engine | paired evaluator, Docker runner | run-count policy per suite (det=1, smoke 1–2, core 3, full/holdout 5), Hermes `batch_runner` runner adapter, full metric capture → **WP-C** |
| §22 Judge | A/B swap function | evaluator integration only when no deterministic verifier, uncertainty propagation, invalid JSON handling → **WP-C** |
| §24 Token metrics | tokens per run | tokens_per_success, tool_calls_per_success, latency_per_success, failed calls, retries, context size in reports and policy → **WP-C** |
| §25 Iterative evolution | `run_evolution` API | `evolve` CLI, generations with smoke→core→full escalation, stop criteria, strategy interface, GEPA adapter, version-pinned SelfEvolutionAdapter → **WP-D** |
| §27 Dreaming → candidate | proposals selectable as metadata | Optimizer-role content generation from a selected proposal (untrusted-evidence framing), max 5 per run → **WP-D** |
| §6 Curator | caller-supplied signals | live read-only connector to Hermes Curator state, `curator status` CLI, CuratorAdapter mutation → candidate → **WP-E** |
| §29 Monitoring | stable/rollback recommendation | PROVISIONAL status on promote, shadow/historical replay evidence, rollback execution path via promoter → **WP-E** |
| §35 Observability | stats in pieces | `hermes-improve report` covering Hermes/Improvement/Benchmarks/Dreaming/Qwen/Production → **WP-E** |
| §36 Laya | — | decision-history table + recording at eligibility/risk/severity/suite/priority decision points → **WP-E** |
| §32–33 OS isolation | UID checks, docker sandbox | `scripts/provision.sh` (users, groups, modes, ACLs), network allowlist for sandbox, tests that run as root in a container and prove the boundaries → **WP-F** |
| §38–39 Tests / E2E | many unit tests | full 16-step automated E2E → **WP-G**: `tests/e2e/test_full_improvement_loop.py` on brain-consuming `skill_procedure` benchmarks |
| §40 Docs | 7 docs | `dreaming.md`, `git-workflow.md`, `promotion.md`, `laya-future.md`; refresh all others to the implemented state → each WP + **WP-G** |
| §4 Discovery | initial notes | verified versions/APIs of local Hermes, self-evolution, GEPA, DSPy, Qwen stack → **WP-0** |

## Execution

Work packages A–F run in parallel on separate branches/worktrees with disjoint
primary file ownership. Shared files (`cli.py`, `store.py`, `models.py`,
`README.md`) receive additive changes only and are merged in order
0 → A → B → C → D → E → F, then WP-G adds the E2E test and the documentation
pass. Every merge must keep the full suite green in the Linux test container.

## Real external blockers (expected)

- No live Hermes runtime is exercised in CI; the batch_runner adapter is tested
  against the real `batch_runner.py` interface from discovery, not a live model.
- No guaranteed local Qwen endpoint; Qwen-dependent paths are tested against a
  local OpenAI-compatible stub server, live use is opt-in.
- Real sealed-holdout data and real production identities are operator-provisioned.

## Outcome (2026-11-20)

All work packages are merged into `main`, and every package passed GitHub
Actions. Skipped tests are limited to the optional `gepa` extra and
Docker-image tests. Since #19, CI runs the
suite twice: on the plain runner (root tests skipped) and as root in the
Linux test container via `scripts/docker-test.sh`, where a skipped
`root_required` test fails the job.

| Package | PR | Status |
|---|---|---|
| Plan + container test runner | #8 | **done** |
| WP-0 Discovery | #9 | **done** |
| WP-E Curator, monitoring, report, Laya | #10 | **done** |
| WP-A Brain, config, candidate types, manual gate, promotion | #11 | **done** |
| WP-F OS isolation, provisioning | #12 | **done** (root CI job added in #19) |
| WP-C Verifiers, run policy, judge, metrics, Hermes runner, sandbox | #13 | **done** |
| WP-D Evolution loop, GEPA adapter, Dreaming → candidate | #14 | **done** |
| WP-B Schema, eligibility, clustering, suites, holdout rotation, replay, corpus | #15 | **done** (closes #7) |
| WP-H §38 test matrix, evolve isolation, CLI completeness | #16 | **done** |
| WP-G Brain-consuming benchmarks, 16-step E2E | #17 | **done** |
| Root CI job, CODEOWNERS team + branch protection | #19, #21 | **done** |
| Curator archive/delete/consolidate as gated removal candidates | #22 | **done** |
| Semantic clustering layer (embeddings) | #23 | **done** |
| Spec audit fixes: §23 policy order/generalization/token gate, §16 finalist-only holdout, §26 ledger schema + migrations, §35/§36 report and Laya decisions, §29 session feed | #24 | **done** |
| Test layout `tests/{unit,integration,security,e2e}` (§37) | #25 | **done** |
| Token metering proxy for `HermesBatchRunner` (§24) | #26 | **done** |
| Isolated lab findings: prompts as package data, JSON mode for Qwen roles, per-stage Dreaming queue, proxy forwards backend probes | #27 | **done** |
| Hermes runner: workspace mounted via task `cwd`, reasoning discards reported | #28 | **done** |
| Hermes runner: `AIAgent` evaluation driver with the `skills` toolset (batch_runner never enabled skills) | #29 | **done** |
| Sandbox and Hermes runner forward the Docker daemon selection (`DOCKER_HOST`, `DOCKER_CONTEXT`, TLS vars); unreachable daemon reported as such (lab F2) | #31 | **done** |
| Test suite hermetic against a provisioned `/srv/hermes-improve` layout and operator `HERMES_*`/`QWEN_*` variables (lab F4) | #32 | **done** |
| LIGHT Dreaming clusters errors deterministically with the lexical layers of `clustering.py` | #33 | **done** |
| `evolve` selects the screening runner like `evaluate` (docker, local, hermes) | #34 | **done** |
| SMOKE 20-task minimum as an advisory warning (deliberate deviation, see below) | #35 | **done** |
| Batch ingestion drops all-zero `tool_stats` rows (REM payload 66,866 to 5,296 characters; lab F12) | #37 | **done** |
| Judge: transport errors fail closed, per-order rationales kept in the signed report, no decision rows from an aborted `evaluate` (lab F25, F26, F29) | #38 | **done** |
| Generated skills need fenced Hermes front matter with `name` = target directory; report breaks observations down by source (lab F15, F22) | #39 | **done** |
| Monitoring evidence scoped to the current activation; SHADOW excluded from the experiment benchmark version and `evaluate` fails fast on a version mismatch (lab F17, F18) | #40 | **done** |
| Dreaming outcome signal: tool error text from tool turns, operator outcome labels, REM weighs labelled outcomes (lab F11) | #42 | **done** |
| `benchmarks replay` for `hermes_agent` benchmarks through the selected runner; partial trajectory on timeout (lab F14, F16) | #43 | **done** |
| Judge reads declared file deliverables; judged tasks out of the quality means, judge-only gains need signed approval; HARD rule for judge benchmarks (lab F24, F27, F28) | #44 | **done** |
| Monitoring against the candidate arm per run; latency alone only warns; blocked rollbacks as release events; one-line CLI errors (lab F19, F20, F21, F23) | #45 | **done** |

A spec-versus-code audit after #23 found gaps in §16, §23, §24, §26, §29,
§35 and §36 that the table above had marked done; #24 closes them. #24 made
missing token usage block promotion; #26 supplies the measurement for
Hermes runs: a loopback proxy between batch_runner and the model endpoint
reads the `usage` of every response and marks a run measured only when every
call reported usage and no model traffic bypassed it. The Docker and local
command runners run no model and stay unmeasured, so model-free benchmark sets
still need `require_token_measurement: false`.

Open items:

- Isolated lab run (2026-11-17: no internet, DinD daemon, Hermes v0.22.2,
  local Qwen3 1.7B) found four defects, which #27 fixes. Prompts were
  missing from non-editable installs. Qwen roles sent no JSON mode. LIGHT
  Dreaming consumed the REM queue. The metering proxy hid Ollama from
  Hermes' context-window check. The two smaller items are closed as well:
  #31 forwards the Docker daemon selection to the sandbox client (F2), and
  #32 keeps the test suite from reading a provisioned
  `/srv/hermes-improve` layout (F4).
- Live runs (2026-11-18) with Hermes v0.22.2 and Qwen3-4B-Instruct-2507 found
  two runner defects in the Hermes path; #28 and #29 fix them. The workspace
  was not mounted into the Docker sandbox. More seriously, `batch_runner`
  never enables the `skills` toolset, so evaluated skills were invisible to
  the agent. With the `AIAgent` driver from #29 the first complete live
  promotion passed. The setup was isolated: no internet, a DinD daemon and a
  local Qwen on Ollama at 64K context. SMOKE, CORE and FULL plus the sealed
  holdout (5 runs per task) passed: quality +33.3, holdout +33.3, tokens +8.7%
  measured by the proxy. `promote` then deployed `brain-v1` (PROVISIONAL) with
  a signed manifest. In every candidate run the agent called
  `skill_view("lab-facts")`.
- Low-priority audit items, both closed. LIGHT Dreaming now masks volatile
  tokens (paths, ids, numbers) and clusters errors with the exact,
  fingerprint and MinHash layers of `clustering.py`, without a model (#33).
  The SMOKE 20-task minimum is a deliberate deviation: it warns on stderr
  and in the signed evaluation metrics but never blocks (#35). SMOKE is
  non-deciding staged evidence (FULL and the sealed holdout decide), impact
  narrowing legitimately shrinks it below 20, and the bundled corpus has
  only 14 SMOKE tasks.

- Live self-improvement, monitoring and judge runs (2026-11-18 to
  2026-11-20, same isolated lab; details and findings F11 to F29 in the lab's
  `results/FINDINGS.md`):
  - Closed loop without a hand-written candidate: 12 real production
    sessions (2/12 correct: the agent read files and counted in its head),
    ingest, Dreaming LIGHT/REM/DEEP and `candidate generate`, all live. The
    four generated candidates had no effect on SMOKE, and two `evolve` rounds
    with the Hermes runner (qwen strategy) produced 4 valid variants without
    a single improved run. Nothing was promoted, which is correct, but the
    system did not improve itself. Main cause: Dreaming has no outcome
    signal. Sessions carry only the prompt, tool counters and
    `final_status`, so REM read wrong answers as efficient successes and
    proposed generic tool-minimisation skills.
  - Monitoring and rollback: a degraded `brain-v2` was detected by real
    Hermes replays and rolled back to `brain-v1` with
    `monitor evaluate --apply`; `deploy/current`, `main`, tags, release events
    and `status` were consistent afterwards. The reactivated release was still
    flagged by old evidence; #40 fixes that.
  - Judge: a subjective benchmark was graded live (candidate preferred,
    confidence 0.98) and stored in the signed evaluation; the policy does not
    use positive verdicts.

All findings from these live runs (F11 to F29) are fixed: #42 gives
Dreaming an outcome signal (tool error text from tool turns, operator outcome
labels via `sessions label` / `sessions import-labels`, REM treats a completed
but incorrect session as a failure). #43 replays `hermes_agent` benchmarks to
produce `replay_evidence` and keeps a partial trajectory for timed-out runs.
#44 lets the judge read declared file deliverables, keeps judged tasks out of
the quality means and routes judge-only gains to signed operator approval,
and defines when a judge benchmark is HARD. #45 compares monitoring with the
candidate arm's per-run evaluation evidence, makes latency alone a warning
instead of a rollback reason, records blocked rollbacks as release events and
prints one-line errors for expected refusals.

Remaining limits:

- None of #42 to #45 has been re-run in the live lab. In particular, whether
  outcome labels let REM and DEEP produce a useful candidate is unproven.
- No component labels session outcomes automatically; an operator (or an
  external verifier) has to attach them.
- Release manifests written before #45 have no `monitoring_reference`; for
  them the monitor still compares the success rate against the old baseline
  arm.

Outside what this repository can prove on its own:

- CI does not exercise the live Hermes runtime, a live Qwen endpoint or live
  GEPA runs (see "Real external blockers"). The Hermes path, the metering
  proxy, Dreaming REM/DEEP and promotion were verified manually in an
  isolated live setup (Hermes v0.22.2, local Qwen3). That setup is not
  reproducible in CI. Live GEPA optimization was not run: one qwen `evolve`
  round with 4 variants took about 1.8 h on the lab GPU, and GEPA's default
  budget would need more than a day.
- Semantic clustering (layer 5) needs a configured embedding endpoint
  (`QWEN_EMBEDDING_*`); it is covered by a fake embedder and a local HTTP stub
  but not exercised against a live Qwen3-Embedding server, and the default
  cosine threshold (0.85) is not calibrated on real data. Without an endpoint
  clustering stays lexical and reports say so explicitly.
