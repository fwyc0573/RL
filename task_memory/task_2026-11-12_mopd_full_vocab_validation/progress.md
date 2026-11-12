## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record planning evidence, environment observations, and the pending core acceptance answer. |
| 2026-11-12 | Recover execution state; record current cloud assets, dependency setup, and next executable steps. |
| 2026-11-12 | Resume after input preparation; record parallel actor environment installation and remaining runtime prerequisites. |
| 2026-11-12 | Verify Torch program-file identity and record 124 passing numerical/routing tests plus recipe validation. |

# Progress

## Current status

| Phase | Status | Evidence |
| --- | --- | --- |
| A | completed | requirements.md, design.md, plan.md; source/model/data and current full-loss implementation inspected. |
| Q | completed | User answered `continue` to the concrete acceptance question. |
| B | in-progress | Input preparation and recipe validation PASS; Torch imports pass; actor dependencies and CUDA toolkit remain incomplete. |
| C | in-progress | 58 full-loss, 17 kernel, and 49 teacher-routing tests PASS; one CUDA routing test skipped; actual-model observability remains pending. |
| D | pending | No GPU diagnostic job submitted. |
| E | pending | No main training or resume performed. |
| F | pending | Final acceptance and completion archive outstanding. |

## 2026-11-12 — source and implementation investigation

- Read target `AGENTS.md`, MOPD algorithm documentation, the full-vocabulary recipe and parents, and the source task's model/data records before choosing the implementation route.
- Invoked `grill-with-docs` by reading its local SKILL.md and loading `grilling` and `domain-modeling`; no callable Skill tool is exposed in this session. Loaded glossary format, repository documentation conventions, build guidance, StepMind worker guidance for the planned run, and `check-progress` for the phase checkpoint.
- Target Git working tree was clean on `mopd-sys`, HEAD `d51b2842`. The source worktree's current HEAD is `6536accb03f0e494f136232d33b49e7f948f42c6`; its unrelated untracked files were not edited.
- Source recipe confirms initial R1-Distill-Qwen-1.5B student, five teacher paths/revisions, equal source weights, and a historical 200-update full-data mode. The manifest confirms 54,600 training prompts and 320 separate probe prompts across five groups.
- The historical asset check tested matching token IDs on one string. That evidence is insufficient to prove whole-vocabulary equivalence. DeepCoder's chat template differs; B owns current full tokenizer/config verification and common tokenized-input checks.
- Inspected `OnPolicyDistillationFullConfig`, `_opd_full_call`, native payload reconstruction tests, Single-Controller DebugWriter, NemoGymDataset, and the checkpoint functional test. Full-vocabulary loss replaces sampled-token optimization and the ordinary PG mismatch metric set. C owns missing observability; no bug has yet been established by execution.
- The existing full-vocabulary shell suite contains cleanup commands. It was read only; future verification must reuse its checks without executing unauthorized deletions.
- Host date was `2026-11-12T02:12:07+08:00`; `python3 --version` returned Python 3.10.6. Bare `python` returned `command not found`. `pyproject.toml` requires Python `>=3.13.14,<3.14`, so B must select a compatible NeMo-RL runtime rather than reuse the historical StepTron environment.
- A bounded public documentation search returned no usable result; all decisions here rest on inspected local primary artifacts. No external API behavior is claimed verified by that search.

## Changes and their purposes

- `requirements.md`: preserve the raw request, explicit authorization, and one pending acceptance question.
- `design.md`: make exact inputs, existing mechanisms, numerical hazards and runtime prerequisites reviewable.
- `plan.md`: define finite milestones and acceptance, including the reference checks needed to distinguish a real fix from hidden error.
- Root `GLOSSARY.md`: distinguish full-vocabulary distillation, sampled-token feedback, train/generation mismatch and policy staleness without implementation details. No ADR is warranted for the reversible run choices.
- Verification method for this phase: static source and document consistency checks only. GPU training, current cloud availability and numerical acceptance remain pending.

## Pending handoff

Phase-A validation is recorded in `test_report_2026-11-12_plan_sources.md`: source group counts, 54,600 training/320 probe rows, document links and source premises passed. `review.md` records the required check-progress self-check: PASS 1 / TODO 6, no detected drift. Applied its route: finish the local handoff, ask only Q, then advance B/C after the answer; do not expand the investigation or claim runtime validation.

1. The core answer is recorded. Continue execution without requesting the same approval again.
2. Advance B and C: verify runtime/current assets and prepare the minimal recipe plus focused numerical/metric checks.
3. Execute D, then E; diagnose actual failures, make scoped fixes and rerun affected checks.
4. Re-read requirements, reconcile every acceptance item, and write the final report and English archive. Do not mark the overall task complete at the planning checkpoint.

## 2026-11-12 — execution preparation and recovery

- User confirmed execution with `continue`. Planning commit: `843d679b`; branch remains `mopd-sys`. Requirements/plan/progress changes are task-owned. No GPU job has been submitted and no numerical test has passed yet.
- Read-only cloud inspection through `ssh shai-ycfeng` confirmed all six model weights. All have Qwen2 architecture, hidden size 1536 and vocabulary size 151936. Complete token-to-ID and added-token mappings match. Raw merges use two JSON representations; normalized code/if merges match the student. Complete normalized validation, including stem, remains pending. Code/if chat templates differ, so all roles will consume the student's tokenization.
- Data sources are conversation lists, each containing two message objects; metadata is on the first message. Counts are 54,600 training conversations and 320 probe conversations. The initial inspection assumed dictionary rows and failed; reading the actual schema resolved that inspection error. Conversion must retain source/group identity and remove old assistant answers from fresh-rollout inputs.
- Local raw evidence: `/data/ycfeng/tmp/nemo-mopd-20261112/model_metadata.json` and `personal_h800_commitments.json`. Personal nonterminal H800 commitments were 74 GPUs at `2026-11-11T18:29:23Z`; this is not a measurement of total free group quota. Do not alter those unrelated jobs.
- Project requires Python 3.13.14 and pinned Torch 2.13.0/vLLM 0.29.0. Local Python 3.10 is unsuitable. Direct corporate PyPI mirror responds; public PyPI through the proxy timed out. Old uv 0.11.14 lacks Python 3.13.14 in its catalog. Installed isolated uv 0.13.7 from the corporate mirror at `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv`; Python download is now running. No project version requirement was lowered.
- Main submodules checked out successfully: Automodel `b916107a5ce8e77a3f18682f8b0eaa46a4b54389`, Gym `5911d628c744d0bf43fe1d9ddf3bda72d59b0f9a`, Megatron-Bridge `7e9dcd609edf2bb9478d3fa4d9b9f8ee3875c16e`. Nested Megatron-LM checkout completed at `3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a`.
- Local `/data` has about 41 GiB available; keep environment copies minimal and place run artifacts in the authorized personal cloud prefix. Reuse the existing Python RJob launcher from `../verl/tests/manual/agent_opd/rjob_run.py` instead of creating a second submitter.
- Loaded session-memory guidance on recovery. Its factual checkpoint content is retained here under the user's task-record convention, avoiding a second competing session archive.

### Resume next actions

1. Finish Python/base environment and verify imports; prepare native recipe/data conversion while dependencies install.
2. Complete asset/split checks, then run existing full-vocabulary CPU tests and a small real-model GPU diagnostic.
3. Diagnose measured parity/gradient failures before the five-teacher 200-update run; record native checkpoint/resume and refit evidence.

## 2026-11-12 — first concrete recipe and input execution

- Added the five-teacher full-vocabulary recipe under `examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2cp4-pack-single-controller-fullvocab.yaml`. It uses TP2/CP4, 200 steps, 256 fresh trajectories/update, 65,536 tokens, source Adam settings, native save/restore, strict taskset teacher routing, and native single-agent-turn Gym orchestration. Recipe validation is still pending.
- Added `tests/manual/mopd/prepare_inputs.py`: verifies pinned data-manifest SHA256 values, checks train/probe/reserve separation, removes historical assistant answers, preserves source IDs, and materializes equal group sampling slots. All 54,600 source training rows are retained; math's 10,600-row pool is cycled for 400 additional slots to match the other four 11,000-row pools. No source file is edited.
- Python 3.13.14 installed with uv 0.13.7 after switching the download to direct networking. CUDA 13.2 nvcc already exists in the mounted verl environment. Base `uv sync --locked --no-dev --group test` progressed at low download throughput; exported the same immutable requirements with `uv export --frozen --no-dev --group test --no-emit-project` and switched transport to the corporate mirror with `uv pip sync`. The project lock and dependency requirements are unchanged. Base environment path: `/data/ycfeng/envs/nemo-mopd-py313`.
- Submitted 0-GPU input preparation as `exp-1112-023909-432726` using the existing personal Python launcher (local source `/data/ycfeng`, NFS `100.96.128.195:/data/ycfeng:/data/ycfeng`, creator `i-fengyicheng`, H800/codesign, 8 CPU/32 GB). This is an asset preparation job, not training.
- That job failed with user command exit 1 at `2026-11-11T18:40:39Z`: `ValueError: code tokenizer differs from student: ['post_processor']`. Incident logs prove that the command ran after a transient ConfigMap event. No OOM/preemption evidence explains this application exception. Platform finished at 18:40:42Z; launcher cleanup followed at 18:40:44.995Z. Raw RCA is `/data/ycfeng/tmp/nemo-mopd-20261112/input-rca`; live/historical events and incident logs succeeded, audit requests failed and do not establish an initiating manual actor.
- Follow-up read confirmed code/stem/if use TemplateProcessing to add BOS 151646, while student uses ByteLevel. All normalized merges and decoders agree. The validator incorrectly required identical independent text rendering despite the planned common student tokenizer. It now checks vocabulary/BPE semantics and records post-processors/templates separately. Existing `create_teacher_worker_groups` passes the common tokenizer to every teacher; actual same-token runtime verification remains required. This is a validator correction, not evidence that numerical training parity already passes.

## 2026-11-12 — second input failure and recovery checkpoint

- Second preparation job `exp-1112-024352-999672` failed with `KeyError: 'chat_template'` at `2026-11-11T18:44:43Z`; native phase Failed at 18:44:47Z, launcher cleanup at 18:44:51Z. Creator `i-fengyicheng`, node `gpu-h800-0285.host.platform.shaipower.com`, replica exit1/restart0. This is a business exception, not evidence of preemption/OOM. IF has no embedded template; the metadata reader now records an optional embedded template and a separate `chat_template.jinja` when present.
- Exact-object RCA: `/data/ycfeng/tmp/nemo-mopd-20261112/input-bpe-rca/`. First incident-log query returned empty; subsequent native replica-log query returned the full traceback. Audit queries failed. There is no evidence assigning an initiating manual actor; later system stop and launcher cleanup are recorded separately.
- Base installation remains alive (exec session 34651); immutable exports for mcore+nemo_gym, vllm+nemo_gym, and nemo_gym are prepared from `uv.lock`. Native per-actor venv caching supports mounted prebuilt environments; no core runtime edits are needed for this.
- Refreshed `review.md` with check-progress after recovery: PASS2/TODO5, no drift found. Applied route: finish actual preparation/import checks, then existing numerical tests and H800 diagnostics; do not continue broad investigation or claim completion.

## 2026-11-12 — input preparation PASS

- Job `exp-1112-025041-145201` and its replica are Succeeded, exit0/restart0. Six model vocabulary/BPE checks, 11 source SHA256/count checks, subset-aware source identities, and exact-prompt split checks passed. Outputs: 55,000 training slots from all 54,600 source rows, plus 320 probes. Exact execution and evidence are in `test_report_2026-11-12_input_preparation.md`.
- Before this submission, the source-only overlap check exposed independently numbered DeepCoder subsets. Including `subset` removed 21 false matches; no source training rows were dropped. IF uses a separate chat_template.jinja, now recorded by the metadata reader.
- Base download diagnosis: PyTorch-hosted package variants were selected ahead of the mirror because the earlier extra index had higher priority. Giving the mirror explicit priority accelerated the Nvidia downloads, then the locked Triton hash correctly rejected the different PyPI wheel. Retain the locked PyTorch wheel URLs for torch/torchvision/triton; download them with aria2 while mirror-only locked packages install. No lockfile, package version, or hash check was relaxed. Failed installer logs remain in the task tmp directory.
- Native entrypoint is `examples/run_grpo_single_controller.py`. The inherited `force_on_policy_ratio=false` keeps the separate policy-logprob forward enabled, so native `log_full_train_data` should carry real prev_logprobs; verify actual emitted columns in D.

## 2026-11-12 — runtime recovery at 12:25 HKT

- Recovered the live aria2 process (exec session 54981) rather than restarting it. Torch and Triton are still incomplete; Torchvision completed. Full base imports and all numerical tests remain unverified.
- Base environment has 265 installed non-Torch packages, including Ray 2.58.0, pytest 9.1.1, Pydantic 2.13.5, OmegaConf 2.3.0, and Transformers 5.16.1. Python is 3.13.14 and uv is 0.13.7. The repository lockfile and dependency versions are unchanged.
- Created three persistent actor venvs under `/data/ycfeng/envs/nemo-mopd-actors/` using the native fully qualified actor names. Split immutable exported requirements into prebuild and deferred groups; deferred packages still must be installed before imports or GPU submission. All original hashes and versions are preserved.
- The first actor install used an unverified Artifactory PyPI URL and failed resolution (aiobotocore/ai-dynamo/aiohttp versions unavailable). Corrected transport to the handbook's `http://mirrors.i.basemind.com/pypi/simple/`, the existing successful base mirror. Parallel install sessions: MCore 84198, vLLM 47453, Gym 26327; logs are `*-mirror-prebuild.log` in the task tmp directory.
- A bounded original PyTorch hostname range request remains slow. The next download hypothesis is egress-route throughput, tested with a 1 MiB request through the existing local proxy without changing its shared configuration. Do not vary package versions or disable hashes to address transport latency.
- Current `/data` free space is 246 GiB at 12:21 HKT; earlier 37–106 GiB readings are historical. Use shared hardlinks and keep training outputs in the personal cloud prefix.

## 2026-11-12 — Torch recovery and CPU checks

- The corporate PyPI Torch 2.13.0 wheel contains runtime version `2.13.0+cu130`. Its 12,015 non-dist-info files match every hash and size in the official locked CUDA 13.0 wheel RECORD; no missing, extra, or changed program files. The downloaded wheel SHA256 is `a3a9a21312872af8a26950b2c15680335a386a1f56ed03e780653d78b9607e9e`. Installed this byte-identical program distribution in base, MCore, and Gym environments; project dependency files remain unchanged. Distribution metadata differs and is recorded, not claimed identical.
- The ordinary PyPI Triton 3.7.1 wheel differs in seven program files, including native libraries. It was not substituted. Continue the existing official locked wheel download; do not repeat mirror exploration. Background exec session 64244 owns this download, and 82064 owns the vLLM prebuild installation.
- Base imports pass with Python 3.13.14 and Torch 2.13.0+cu130. Existing `test_opd_full.py`: 58 passed in 90.01 s; selected full-vocabulary kernels: 17 passed in 9.89 s; teacher-worker suite: 49 passed, one CUDA-only skip in 25.73 s. The new recipe passes `test_shipped_config`: one test and five subtests, with three inapplicable backend subchecks skipped. Exact commands and limitations are in `test_report_2026-11-12_cpu_validation.md`.
- The earlier claimed CUDA 13.2 nvcc path is no longer present. A current scoped search found no nvcc under the persistent env/toolchain directories. B now owns a persistent CUDA 13.0 toolkit before any GPU extension build. No GPU run may install dependencies on its normal startup path.
- MCore/Gym editable installs completed, but actor imports and deferred runtime packages are not yet verified. Build isolation fetched another Torch version into a cache; no compiled ABI compatibility is assumed from install success. Gym native input/schema tokenization is still pending.
- Next actions: finish the exact Triton and vLLM downloads; obtain the CUDA 13.0 compiler and build required actor extensions against the installed Torch; verify imports and Gym inputs, then submit actual H800 diagnostics. The 200-update workflow, real-model parity, refit, and resume remain unverified.

## 2026-11-12 — actor preparation and native Ray diagnosis

- Gym actor imports pass. MCore first required the already-pinned Lens and TransferQueue packages; installing those advanced import to TE. Loading the existing CUDA 13.0 compatibility library on the CPU host then exposed the actual missing TE PyTorch extension. Its build remains pending the compiler; no GPU was consumed.
- Downloading NVIDIA CUDA 13.0.2 redistribution components with manifest SHA256 checks into `/data/ycfeng/toolchain/cuda-13.0.2`. The compiler is 13.0.88; this matches Torch's CUDA 13.0 major/minor. NVCC archive and four small components are downloaded and verified; library components remain in flight, so the toolkit is not yet declared ready.
- A controlled 1 MiB range request to the same FlashInfer wheel measured 52,271 bytes/s through local proxy 7890 versus 515,081 bytes/s through company proxy 3128. Continued task-owned Triton/vLLM transfers on the faster measured route, retaining aria2 partial state. This changes transport only. Native uv debug reports streaming two FlashInfer wheels because range metadata requests are unsupported.
- The first CPU Ray command selected a GCS port inside Ray's worker-port range and was rejected before startup. Using 26379 removed that configuration conflict. Native raylet/GCS logs then showed `Illegal header key` and SIGABRT while initializing metrics export. The inherited agent OTEL header key was `Authorization`; excluding inherited `OTEL_*` variables from the task subprocess made Ray startup succeed. No header value was printed or archived.
- Local prefetch Ray head is at `100.96.128.195:26379`, four CPU slots and 256 MiB object store, with task-local temp paths. The native prefetch invocation initially triggered automatic uv repository packaging; applied Dockerfile's existing `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` and restarted only that task-owned prefetch driver. Current prefetch log: `/data/ycfeng/tmp/nemo-mopd-20261112/gym-prefetch-2.log`. Keep this setting in the worker environment.
- Input converter Ruff checks, formatting check, recipe validation, and scoped diff checks pass. Commit the verified input/configuration step; its commit does not imply completion of B, C, or training acceptance.
