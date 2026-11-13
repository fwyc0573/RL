## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Complete pooled rollout-statistic RED/GREEN and actual step18 length replay; keep formal training active. |
| 2026-11-14 | Verify corrected resume seven-worker runtime and the sole step17 checkpoint. |
| 2026-11-14 | Record actual expanded-resume failure, native CPU reproduction and corrected preparation. |
| 2026-11-14 | Record actual resumed overflow attempts and preserve the eight-unit TQ restore layout for expanded generation. |
| 2026-11-14 | Resume with explicit latest-checkpoint-only storage constraint; suspend stale resume arguments until retention is verified. |
| 2026-11-14 | Recover failed formal terminal state, zero-update business metrics and exact rank3 preemption. |
| 2026-11-13 | Reject smaller GPU layouts with platform and native evidence; complete current input identity terminal verification. |
| 2026-11-13 | Recover the existing formal job at15:36HKT and preserve its queued state and unfinished acceptance checks. |
| 2026-11-13 | Accept completed five-teacher PP4 GPU save/resume, finite checkpoint, and full diagnostic parity. |
| 2026-11-13 | Recover failed gradient probes, CPU correction, actual queue state, and formal FP16 preflight. |
| 2026-11-13 | Recover actual decode controls, padding RCA, and ordinary native replay frontier. |
| 2026-11-12 | Recover the live model-reference job and record verified Gym and CUDA environment preparation. |
| 2026-11-12 | Record planning evidence, environment observations, and the pending core acceptance answer. |
| 2026-11-12 | Recover execution state; record current cloud assets, dependency setup, and next executable steps. |
| 2026-11-12 | Resume after input preparation; record parallel actor environment installation and remaining runtime prerequisites. |
| 2026-11-12 | Verify Torch program-file identity and record 124 passing numerical/routing tests plus recipe validation. |
| 2026-11-12 | Record verified native preparation and the automatic CPU import/submission chain. |
| 2026-11-12 | Recover TQ serialization failure and native empty-batch diagnostic. |

# Progress

## Current status

| Phase | Status | Evidence |
| --- | --- | --- |
| A | completed | requirements.md, design.md, plan.md; source/model/data and current full-loss implementation inspected. |
| Q | completed | User answered `continue` to the concrete acceptance question. |
| B | completed | Current assets/data/tokenizers, recipe, exact locked dependencies and four runtime imports verified; see native-runtime-versions.json. |
| C | completed | CPU/HF references PASS; native token/mask/version fields prepared. Actual native parity remains phase D. |
| D | completed | FP16 TP2/PP4 five-teacher diagnostic: five successful updates, finite native state and save/resume; recorded long-context parity checks PASS. |
| E | in-progress | Formal exp-1113-174329-659007 Failed after rank3 Preempted. Eleven attempts, zero successful updates; native step10 resume/state/gradient diagnosis pending. |
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

## 2026-11-12 — recovery after the model-reference submission

- Committed the verified input/configuration and CPU-test step as `c2cfe5f0` (signed off); no push. The only new uncommitted executable is `tests/manual/mopd/check_model_payloads.py`.
- Native Gym prefetch completed successfully (`gym-prefetch-2.log`: `Succeeded: 1`). Its environments reside under `/data/ycfeng/envs/nemo-mopd-gym-servers`, with interpreter links into the persistent Python toolchain. Use the existing `env.nemo_gym.skip_venv_if_present=true` runtime override and `NEMO_GYM_VENV_DIR` for GPU runs.
- All 320 probes passed native `MaterializedTask[SingleAgentTurnTaskInput]`, `NemoGymDataset`, processor, and student prompt tokenization checks. Every group has 64 rows. Prompt length ranges: math 42–297, code 323–1420, stem 31–178, logic 22–6663, if 18–2684. Evidence: `/data/ycfeng/tmp/nemo-mopd-20261112/gym-schema-tokenization.json` and `gym-schema-tokenization-3.log`. This is input validation only. Earlier local attempts encountered an incomplete download and Transformers 5's default dictionary return; explicitly requesting `return_dict=False` corrected the diagnostic.
- CUDA 13.0.2 toolkit is assembled under `/data/ycfeng/toolchain/cuda-13.0.2` (nvcc 13.0.88). The exact NVIDIA nvvm wheel supplied the component whose archive download failed. TE 2.18.0 PyTorch extension compiled against installed Torch without build isolation in 2m43s (`mcore-te-build-3.log`). MCore import now fails at missing locked Triton (`mcore-import-4.log`); the extension build alone does not establish an operational actor.
- vLLM prebuild/runtime installation completed, including FlashInfer wheels, Torch, Torchvision, Lens, TransferQueue, editable RL/Gym, and torch-c-dlpack-ext. Its actor import remains unverified. Official Triton download remains alive in exec session `21056`; about 135/188 MiB at the latest check, with low network throughput. Do not substitute the nonidentical ordinary PyPI wheel or alter the lock.
- Initial-checkpoint HF numerical control was submitted as `exp-1112-125639-935155`, 1 H800 / 8 CPU / 48 GB, creator `i-fengyicheng`, NFS `100.96.128.195:/data/ycfeng:/data/ycfeng`. Replica `exp-1112-125639-935155-cfcd2084` is Running on `gpu-h800-0582`; worker reports H800 81559 MiB, driver 570.124.06, mounted Python 3.13.14, Torch 2.13.0+cu130, and successful BF16 CUDA arithmetic. Launcher session `50542` must remain alive. Snapshot: `model-reference-inspection-1.log` and `/data/ycfeng/tmp/exp-1112-125639-935155.log`.
- The HF control compares all six initial checkpoints' direct logits, reconstructed hidden-state/head outputs, and native full-vocabulary loss/gradient kernel. It does not cover Megatron, vLLM, cross-worker transport, training, refit, or resume. Its cloud outputs are under this task's `reference/` prefix; the subsequent completed results are recorded below.

### Current next actions

1. Finish runtime imports and inspect the live numerical control; preserve first failures and perform exact-job RCA when required.
2. Run native self-distillation and five-teacher diagnostics with same-token trainer/vLLM comparisons, then repair demonstrated causes.
3. Complete the 200-update run with 256 fresh trajectories/update, actual long-context coverage, native checkpoint/resume, and parity checks at initial/refit/resume boundaries.

- B network hypothesis, before execution: local host egress is throttling the exact Triton wheel (about 8 KiB/s, over one hour remaining). A zero-GPU worker will try the same locked URL via direct worker egress, capped at 900 seconds with a 40-second low-speed limit. A fast transfer and matching SHA256 would establish a working download route; slow/error output will reject that route without changing dependencies. This preparation job installs nothing and leaves the existing partial download untouched.

- B import dependencies: native Megatron checkpoint code imports `modelopt.torch.opt.plugins` and `nvidia_resiliency_ext.checkpointing` directly. Installing their pinned revisions exposed a concrete CUPTI development-layout error: the build requires `libcupti.so`, while the installed NVIDIA wheel supplies `libcupti.so.13`. Added the ordinary unversioned linker symlink in the assembled toolkit and set the build's documented `CUDA_PATH`. Retry prediction: CUPTI discovery passes without disabling the extension or changing its source. Original traceback is in `mcore-checkpoint-deps-install.log`.
- `uv pip check` reports Torch's cuDNN/NCCL/NVSHMEM metadata pins differ from installed versions; all three installed versions are intentional repository `override-dependencies` in `pyproject.toml`, not new substitutions. Missing Triton and required checkpoint packages remain B work. Unused video/quantization-only extras do not establish this text-only runtime's validity; actual actor imports and training remain required.

## 2026-11-12 — HF control completed; native training still pending

- Six-model HF reference PASS: reconstructed/direct selected and full-distribution logprob errors are zero; maximum loss absolute error 2.384185791e-7; maximum gradient element error 2.980232239e-8. Full-distribution BF16/FP32 head MAE 0.035309–0.050630 is a precision control, not trainer/vLLM parity. Exact results and limits: `test_report_2026-11-12_model_reference.md`.
- `exp-1112-125639-935155` and exact replica Succeeded; exit0/restart0. Exact Replica CRD read replaces forbidden core Pod read (HTTP403). Creator, image digest, local NFS and business marker are retained in `evidence/model-reference-terminal.json`. The launcher ended normally; no native training job is currently active.
- vLLM worker, AsyncEngineArgs and AsyncLLM imports succeeded, with missing-Triton warnings. MCore remains blocked on the locked Triton; neither actual vLLM engine initialization nor MCore runtime is established by those imports.
- Checkpoint dependencies are installed at the locked revisions: nvidia-modelopt0.46.0 commit43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a and nvidia-resiliency-ext0.7.0 commit ebf0352a82ca5699148ea10bec2b8eedd121e382. `uv --no-config pip install --no-deps` prevented the project override from substituting modelopt0.48.0. Verified package identities: `evidence/mcore-checkpoint-deps-identity.json`.
- The previous Triton download timed out, preserving partial state. Session5561 is continuing the same immutable wheel with SHA25659688b9b924f887316dcd0fae9e8cbe697ee1d1f6ab386723982cc9ecdebee01; about77% at14:25HKT. Zero-GPU worker `exp-1112-130400-081046` succeeded as a command but produced no wheel; that route did not resolve the dependency. Do not repeat it or substitute the different PyPI program files.
- Next mainline action while download progresses: prepare the smallest RJob/Ray adapter for the existing Single-Controller entrypoint. Reuse native actor venvs and the prebuilt Gym environment. No dependency installation belongs in normal GPU startup. Diagnose actual imports after Triton completion, then execute D.

## 2026-11-12 — native launch preparation

- The native Single-Controller entrypoint `--help` passes in the mounted base environment, without starting Ray or training. The full actor runtime is still pending Triton.
- Added the smallest worker-side RJob/Ray adapter at `tests/manual/mopd/run_on_rjob.py`: platform replica rank/count, native Ray startup, resource readiness, existing Single-Controller command, prebuilt actor/Gym environments, per-job logs/checkpoints, and bounded final Ray log retention. It contains no alternate trainer or dependency installer. Runtime checks on GPU are pending; do not count preparation as a training PASS.
- Selected native `raw_logprobs` for trainer/vLLM parity. Temperature1/top-p1/top-k=-1 remain unchanged. Agent-owned artifacts are segregated from native training outcome; log archival is a soft gate.
- Static vocabulary finding: Qwen2 model vocab151936, Bridge provider divisor128 and TP2 would pad to152064. The full-loss kernel normalizes all supplied logits and does not exclude padded vocabulary entries. Set the existing model provider override `make_vocab_size_divisible_by=64` explicitly for student and teachers: divisor64×TP2=128 divides151936, preserving exactly the model vocabulary without extra logits. Runtime shape/distribution verification remains D work; no core loss change or tolerance change is claimed.
- Initial HF-to-Megatron conversion requires a shared checkpoint directory. The adapter uses the existing `NRL_MEGATRON_CHECKPOINT_DIR` under this task's personal cloud prefix, instead of worker-local HF caches; other high-frequency caches remain worker-local. This also preserves source model directories.

## 2026-11-12 — native preparation verified at 15:10 HKT

- B/C completed preparation slice: CLI help, personal local submit dry-run, final recipe validator, 263 native failure/timeout/replacement/trace tests, and direct auxiliary write-error isolation all pass. Exact commands, logs, expected/actual values, and limits are in `test_report_2026-11-12_native_preparation.md`. Launch scripts remain uncommitted pending actual native GPU execution.
- The original raw-logprob and vocabulary-divisor validators both finished successfully (1 test, 5 subtests each). A later invocation exposed inherited OTEL gRPC metadata; excluding OTEL reproduced the earlier environment repair. No library behavior changed.
- Final recipe now uses the existing zero-lookahead in-order sampler and native trace. Formal isolated failures use bounded native retry/replacement settings, retaining the 256-trajectories/update requirement. The validator caught the default 600s watchdog below the 7200s rollout deadline; selecting the existing 9000s watchdog fixed that configuration conflict. Short-run overrides retain 300s rollout/router and 600s watchdog.
- Worker adapter now treats runtime/command/replica/finished evidence writes and Ray log retention as soft gates. The training subprocess writes console output to the platform; auxiliary cloud stdout-file creation is no longer on the training path. Required Ray head startup rendezvous remains explicit.
- Personal 24-H800 self-distillation submission arguments are retained at `evidence/self-diagnostic-submit-args.json`. Dry-run confirms 3x8GPU, 32CPU/node, 192GB/node, local `/data/ycfeng`, codesign, explicit company image and both existing RDMA resources; no job was created. Five aliases point to the initial student only for this numerical diagnostic. The required five-teacher run is unchanged.
- Triton remains the one download prerequisite. Session5561 continues its immutable wheel; a separate CPU-only waiter (session8655) will check the exact SHA256 and install it into all four environments only after aria2 removes its partial marker. It neither downloads another copy nor submits GPUs. Its receipts are `finish-locked-triton.log` and `triton-install-results.json` under the task tmp directory.
- Actual training updates remain 0. Next: locked-wheel installation -> required actor imports -> authorized 24-GPU native diagnostic -> five-teacher diagnostic/fixes -> 200-update checkpoint/resume run.

- At15:25HKT, the sole Triton continuation had181/188MiB. The installation waiter now uses the task's Python3.13 (session8351); system Python3.10 lacks hashlib.file_digest, detected and corrected before the checksum step. CPU-only automatic readiness chain session55518 waits for validated installation receipts, probes base/MCore/vLLM/Gym required imports with60s deadlines, and submits the prepared personal24-GPU native diagnostic only if all imports pass. It keeps the local StepMind launcher alive. No GPU job has been submitted yet. Scripts/logs: `launch_native_when_ready.py`, `launch-native-when-ready.log`, `native-import-results.json` and `self-diagnostic-launch.log` under the task tmp directory. Preserve first failed imports and repair locally; do not duplicate the chain.

## 2026-11-12 — locked runtime completed; native import deadline

- At15:34HKT the exact Triton wheel completed, aria2 checksum passed, and the install waiter independently verified SHA25659688b9b924f887316dcd0fae9e8cbe697ee1d1f6ab386723982cc9ecdebee01. All four persistent environment installs exited0. Evidence: `/data/ycfeng/tmp/nemo-mopd-20261112/triton-install-results.json` and `finish-locked-triton.log`. The download and install waiters are complete.
- Native base/vLLM/Gym imports passed in Python3.13.14, Torch2.13.0+cu130, Triton3.7.1. The MCore probe exceeded the preparation script's60s deadline; the automatic chain stopped before GPU submission. Original exception: `launch-native-when-ready.log`; partial import output: `native-import-megatron-locked-triton.log`. Observed at50s:26s accumulated CPU, ~1.1GiB RSS, no dependency traceback. This is not native training evidence; updates remain0.
- Before another probe, ranked hypotheses: H1 cold NFS/module imports exceed60s; a warmed, timed module-by-module probe finishes without source/dependency changes. H2 an import-time CUDA extension build is active; process children/compiler activity or a stack dump identifies it. H3 a module waits indefinitely; repeated timed stack dumps remain at the same wait without CPU progress. Capture remaining child state before repeating; do not allocate GPUs until the required imports finish.
- Native scheduler inspection found that resume checks its saved horizon. Keep the formal run configured for200steps from its initial launch; perform the planned interruption after an early native checkpoint publishes, then restore that same configuration/checkpoint. Do not launch a10-step scheduler and change its horizon at resume.
- Personal commitment read at07:30:01Z is retained in `personal-commitments-pre-native.json`; it lists current commitments across GPU types, not free codesign quota. `systemd-run --user` lacked a bus and system scope required interactive authentication; the bounded100-object read used timeout60s and a30GiB address-space limit instead. No unrelated job was changed.

- Timed MCore import PASS in77.731s, including worker, MCore, TE, ModelOpt plugins and resiliency checkpoint imports. Stack dumps show successive module source compilation/loading, ruling out a fixed wait and showing why the60s preparation deadline failed. No dependency or source change was required. Actual four-environment receipts: `native-import-results.json`; the MCore receipt references `mcore-import-timed-1.log`. Reuse these passed probes and submit directly; do not rerun the stopped60s chain.

- B/C preparation complete: current assets/data/tokenizer, native recipe, four intended runtime imports, focused CPU and real-model HF references, and native token/mask/version observation fields are verified or prepared. Native execution/parity remains D, not a B/C PASS inference.
- Submitted the24-H800 native self-distillation diagnostic as `exp-1112-153814-140170` at15:38:14HKT, using the unchanged prepared arguments. Local launcher session21760 must remain alive. Source `/data/ycfeng`, H800/codesign,3x8GPU,32CPU and192GB/node, both existing RDMA requests. Exact creator/source/runtime and terminal checks remain pending. This is a5-update control; the required production five teachers are unchanged.

- At the latest exact native-job query, queue `codesign-default` reported H800 remaining2, request24 and queue rank2. Creator/source/resources are correct; do not stop unrelated jobs or duplicate the native workflow. While it queues, submit an independent2-GPU existing test slice: `tests/unit/distributed/test_distributed_logprob.py::test_student_teacher_divergence_kernels` (4cases, TP1/TP2, reverse-KL/cross-entropy). Its actual distinct-teacher CUDA/NCCL checks cover cross-vocabulary normalization that CPU-neutralized collectives and a zero-divergence self-control cannot expose. This is an additional D numerical prerequisite, not a replacement training workflow. Expected:4PASS/0skip, unchanged existing rtol1e-4/atol1e-4 forward and gradient assertions.

- Independent2-GPU reference `exp-1112-154509-699035` started on gpu-h800-0401 with correct personal creator/local NFS, H800 driver570.124.06, Python3.13.14 and Torch2.13.0+cu130. It failed in the native pytest Ray fixture before numerical cases: JUnit1setup-error/0test failures, AF_UNIX socket path110bytes exceeds107. Exact Replica UID/owner/image/node and Error exit1/restart0 at07:46:02Z are retained in `evidence/tp2-reference-failure.json`; RJob Failed07:46:04Z preceded launcher cleanup07:46:09Z.
- RCA collector obtained exact objects, live/historical events, logs and existing-DeepScan reads; both all-operation audits timed out20s and are unavailable, not empty. The earlier missing-ConfigMap event was07:45:25Z, before successful worker start07:45:39Z. The retained JUnit setup traceback establishes the business failure; absent audited manual stop actor/reason/time stayUNKNOWN. Raw queries: `/data/ycfeng/tmp/nemo-mopd-20261112/rca-tp2-reference-1/`.
- Narrow repair: shorten only the2-GPU test worker TMPDIR prefix to `/tmp/nrl-<job>`. Direct CPU Ray socket validator reproduces the old error and accepts the new path (114versus99bytes with7-digit PID); the existing native adapter path is105bytes and needs no change. Preserve107-byte native limit and all test assertions. Resubmit only this failed numerical slice with the same personal creator, source, image and2-GPU shape.

## 2026-11-12 — resumed native diagnostic monitoring

- Re-read requirements, plan, design, progress, review and actual launcher/process state. Reused the existing phase-D native job; no duplicate native submission. At15:56HKT personal Python inspection confirmed exp-1112-153814-140170 Pending, creator i-fengyicheng and NFS100.96.128.195:/data/ycfeng:/data/ycfeng. Launcher session21760 and PID329008 remain alive.
- Re-ran only the failed TP numerical slice using the previously validated short TMPDIR. Exact command: /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_tp2_reference.py, with the runbook personal/proxy environment and PYTHONPATH=/data/ycfeng/steptron. New job exp-1112-155645-933258; launcher session91878; log tp2-reference-launch-2.log. Preserved prior command receipt as tp2-reference-command-attempt-1.json before resubmission. The expected result remains4PASS/0skip; native kernel tolerances unchanged.
- At15:58HKT the TP replacement was Pending with correct creator/local source. Inspect the existing two jobs while keeping both launchers alive. Runtime preparation PASS does not imply any native optimizer update or trainer/vLLM parity PASS.
- Applied check-progress S5 correction: updated the stale Current status table above and the preparation report's historical runtime/submission statements. Mainline remains D -> E -> F; no new acceptance requirement or source change.

## 2026-11-12 — successful optimizer update observation

- D/E observation gap: split Megatron computes update_successful, but previously did not return it to TQPolicy/controller; step_trained alone cannot prove200 successful updates. Added optimizer_update_successful through the existing result/metric path. TQPolicy combines every reported DP replica leader with logical AND. Gain: distinguish an accepted loop step from an actual optimizer update; existing metrics do not provide this fact. Optimizer, scheduler, counters and failure handling are unchanged.
- Reproduced missing-metric KeyError with the new fault control, then38 TQ/step-metric checks passed in8.56s. MCore's two True/False result/pre-hook controls passed in41.12s with explicitly limited CPU device and collective mocks; actual GPU execution is unverified. First wrong-marker invocation ran0 tests; first unmocked MCore invocation on this CPU host failed at CUDA allocation before optimizer execution. Retained those errors; did not change production CUDA behavior or repair unrelated tests.
- Pinned Ruff0.9.9 check/format and scoped diff pass. The first uv tool invocation tried uncached public metadata and timed out; reused the already-downloaded exact0.9.9 executable through uv run --no-sync. No dependency installation or further download is needed.
- Evidence, exact commands and limits: test_report_2026-11-12_optimizer_updates.md. Raw logs: optimizer-update-status-{red,green,mcore,mcore-2,mcore-cpu-control}.log under the task tmp directory.
- At16:07HKT both existing GPU jobs remained Pending with personal creator/current local NFS. Launchers21760 and91878 remain alive. The mainline is still D -> E -> F; actual successful updates remain0.

- Signed commit3d467da9 records the scoped optimizer observation and regression report; no push. Ran check-progress after this verified deliverable: PASS4/BLOCKED1/TODO2, no drift; applied S5 record correction. Both queued native jobs remain the D frontier; do not extend already-working support tooling or promote E before D evidence.

## 2026-11-12 — diagnostic queue scheduling correction

- Observed before the change: both exact jobs remain Pending; the native job has waited about40 minutes, queue rank2 and H800 remaining0. Both launchers are alive. No native optimizer update has occurred. Current receipts are under /data/ycfeng/tmp/nemo-mopd-20261112/native-job-watch-1/.
- Hypothesis for this scheduling attempt: the existing ResourceConfig.preemptible=True selects the SDK global queue and does not require unused codesign quota. The handbook and pinned brainpp.struct source confirm that route; actual scheduling still requires a submitted job. Preserve H800/codesign, personal creator, local source, model identities, GPU shapes, numerical assertions and existing runtime.
- Prepare the submitter's existing ResourceConfig selection as --preemptible and validate its dry-run before stopping/replacing only these two never-started jobs. Use rjob_stop.sh, record the planned reason, verify terminal state, then submit replacements once. Preserve earlier receipts; do not keep duplicate nonterminal runs or change unrelated jobs. If this queue also waits, retain the new exact jobs rather than repeat scheduling changes.

## 2026-11-12 — native startup failure, correction, and TP interruption

- The status changed during scheduling preparation: original self job was approved08:18:01Z, reached MOPD_RAY_READY with3nodes/24H800, then Failed08:19:22Z. Head gpu-h800-0086 used the expected Python3.13.14, Torch2.13.0/CUDA13 runtime and source. Its user command exited1 at08:19:21Z before model initialization: Hydra rejected env.nemo_gym.skip_venv_if_present because the key was absent from the recipe's struct config.
- Local CPU reproduction preserved the same OverridesError. Changed only the adapter injection to +env.nemo_gym.skip_venv_if_present=true. Complete native overrides now parse, resolve and pass MasterConfig and validate_single_controller_config. The first green probe imported a nonexistent omegaconf_utils module; corrected the probe import to the actual nemo_rl.utils.config without dependency/environment changes. Logs: native-overrides-red.log, native-overrides-green.log, native-overrides-green-2.log. Pinned Ruff check/format PASS.
- Agent operational error: the dry-run state guard rejected the assumption that both old jobs were Pending, but that shell command lacked set -e and continued into rjob_stop.sh. It did not stop the already Failed native job. It stopped TP job exp-1112-155645-933258 at08:20:26Z after its worker started08:20:16Z and phase became Running. Original TP numerical checks remainUNVERIFIED; this is an agent interruption, not a kernel failure or preemption. The local stop audit and server stopped-by label both identify i-fengyicheng. Preserve this error and use set -euo pipefail before every dependent shell mutation.
- Exact-object RCA completed for both cases: live/historical Events, incident logs and existing-DeepScan reads succeeded; no existing DeepScan flows. Both all-operation audit queries timed out18s in each case, so audit coverage is unavailable, not empty. The native Hydra traceback establishes its business failure; initiating manual-stop fields there remainUNKNOWN, and controller cleanup is subsequent. Selected evidence: evidence/native-startup-and-tp-interruption.json; raw directories rca-self-native-1 and rca-tp2-interrupted-2 under the task tmp root.
- After confirming both originals terminal and local override validation PASS, submitted exactly one replacement per slice. Native exp-1112-162239-053672 uses launcher17124; TP exp-1112-162239-776626 uses launcher28672. Both are personal H800/codesign with original GPU shapes/source/image; preemptible=yes and quotagroup-global-default verified08:22:58Z. Native and TP launch logs: self-diagnostic-launch-2.log and tp2-reference-launch-3.log. Read-only monitor81077 writes native-job-watch-2/. No duplicate nonterminal jobs or model substitutions.
- Successful optimizer updates remain0. The queue change establishes configuration only; actual self-control, TP4PASS/0skip, trainer/vLLM parity, five-teacher training, long context and200-update resume remain pending.

- At16:23:58HKT both replacement RJobs were Running; self has3active replicas and TP1. Global queue approvals08:23:01Z/08:23:17Z. Worker TP log confirms2H800/driver570.124.06. Applied check-progress after startup/override correction: PASS4/TODO3; D7 identified the stale-state stop error and correction is active. Current frontline remains actual D execution; no more queue substitutions or support-tool expansion.

## 2026-11-12 — TP reference passed; native preemption established

- TP job exp-1112-162239-776626 reached Succeeded; cloud pytest.xml records4 tests,0errors/0failures/0skips,53.510s. TP1/TP2 reverse-KL and cross-entropy loss/gradient cases passed existing rtol/atol1e-4. This is the tensor-kernel reference, not actual Qwen model parity or vocab-shape evidence. Do not rerun it.
- Native replacement exp-1112-162239-053672 Failed08:24:12Z. Initiating replica is c81e728d on gpu-h800-0326, not logical head cfcd2084. Its native Kubernetes Event is explicit: Preempted at08:24:09Z by high priority pod shai-core/exp-1112-082354-535509-client-cfcd2084 after terminating grace period (source addon-agent-podfreezer). Container exited14308:24:10Z; other replicas were stopped later by the RJob fail policy. This establishes infrastructure preemption; no numerical or environment cause is inferred.
- RCA first collected logical-head objects, then the actual failed initiating replica. Exact owner/replica UIDs, live Event Pod UID and incident timing are retained in evidence/native-preemption-2.json. Live/historical Events, logs and existing-DeepScan reads succeeded; audits timed out18s. No existing DeepScan. Full deleted Pod owner object was not returned, so that binding is a stated limit. Missing initiating manual-stop fields remainUNKNOWN. Raw: rca-self-native-2/ and rca-self-native-2-initiator/.
- Route correction follows new preemption evidence: retain TP PASS and resubmit only the native5-update control once as nonpreemptible through the sameH800/codesign source/image/settings, with the already-validated Hydra prefix. Keep its launcher alive even if regular quota queues it. Do not retry the global queue or modify numerical thresholds. Native successful updates remain0.

## 2026-11-12 — resume checkpoint at 16:31 HKT

- Actual state reconciliation: self diagnostic `exp-1112-162755-471889` is the sole nonterminal native validation job. Launcher76679 and read-only monitor52990 remain alive. Personal creator, nonpreemptible codesign quota, 24H800 request and current local NFS source are verified in monitor receipts. At08:30:12Z it was Pending, queue rank2, remaining H800 quota0. Retain this exact job; do not submit duplicates or switch back to the preemptible queue.
- TP terminal collection completed: job/replica Succeeded, ownerUID match, container exit0/restart0 and business marker visible. Successful cloud JUnit was read and copied into selected evidence; four reference cases PASS with0skips in53.510s. Report: `test_report_2026-11-12_distributed_reference.md`. This settles only the synthetic TP reference slice; native optimizer updates remain0.
- Independent D preparation while queued: inspect native untruncated train dump, policy logprob forward and per-sample weight tags; keep same-token/same-weight parity bound0.02 nats/token. Recipe force_on_policy_ratio=false preserves the separate prev_logprobs forward; do not treat full-loss missing mismatch metrics as equality. Inspect native wall-clock checkpoint stop/resume without changing the200-step scheduler horizon.

- At 2026-11-12T16:38:28+08:00, both subsequent candidate configurations passed `MasterConfig` plus `validate_single_controller_config`: seven-node five-teacher5-update/32-prompt/128-generation control and200-step/256-prompt/65536-token main candidate with native checkpoint early-stop. Both retain five distinct approved teachers and full-vocabulary hidden-state loss. These are CPU preparation only and no jobs were submitted. Commands: `validate_next_native_configs.py` in the task tmp root via the pinned Python3.13 uv environment; PASS markers in `next-native-configs-validation.log`. The45-minute early checkpoint deadline is a candidate pending measured startup/step durations, not an accepted runtime bound. First probe-file creation mistakenly executed project imports using host Python3.10, causing ModuleNotFoundError before configuration validation; corrected file creation and used the existing mounted3.13 environment, with no dependency repair/install.
- Signed commit61d6fb68 archives the4PASS distributed reference, terminal evidence and reconciled status. Live native state remains Pending at20261112T083813Z; selected receipt `evidence/self-diagnostic-current-job.json`. Launcher76679 and monitor52990 are retained.

- At2026-11-12T16:43:58+08:00, native exp-1112-162755-471889 is Running. Quota approval08:40:25Z; replica logs confirm MOPD_RAY_READY for3nodes/24H800. Keep launcher76679/monitor52990; actual model initialization, optimizer updates and parity are still pending. Current logs are retained in native-job-watch-3 under the task tmp root.
- Independent long-input preparation: copied only initial student tokenizer/config files through authorized read-only personal cloud access, not model weights. Current Transformers apply_chat_template defaults return_dict=True; the first CPU length probe counted BatchEncoding fields and failed its assertion. Signature/result-type probe established that cause; explicit return_dict=False yields5 materialized validation tasks, one per teacher group, each65504 prompt tokens with32-token generation limit and65536 ceiling. Model config max_position_embeddings131072; tokenizer model_max_length16384 emits a warning but is not GPU-length validation. Original failing and corrected logs preserved as long-context-input-preparation.log and long-context-input-preparation-2.log. Input identity: evidence/long-context-input-preparation.json. Synthetic inputs are only for the long-context test; production data pools are unchanged. GPU coverage remainsUNVERIFIED.

## 2026-11-12 — resumed vLLM NCCL startup diagnosis

- Re-read requirements, plan, design, review and progress; verified launcher76679 and monitor52990 remain alive. At08:50:24Z the exact nonpreemptible native job exp-1112-162755-471889 and all3replicas were Running. No finished.json, model refit, optimizer update or parity evidence exists. Do not duplicate this job or reuse the prior preemption cause.
- Native initiating business error at08:45:17Z: vLLM0.29.0 RayWorkerProc pid3623 on10.191.9.11 / gpu-h800-0187 fails NCCLLibrary.ncclGetUniqueId with RuntimeError NCCL invalid usage. The ActorHandleNotFoundError at08:45:18Z is subsequent cleanup, and optional nixl_ep undefined-symbol warning has not been shown causal.
- Runtime receipts show head gpu-h800-0326 uses NCCL_SOCKET_IFNAME=ens20np0, while generation node gpu-h800-0187 is provisioned with ens22np0 and a different NCCL_IB_HCA ordering. Hypothesis under investigation: driver environment forwarding replaces node-specific platform settings. Verify actor environment and same-library unique-ID control before editing.
- Read-only live dashboard nodes/log reads succeeded. First local Ray read probe stalled connecting despite TCP9489/HTTP9380 access; interrupted only that CPU probe, preserving the GPU launcher, then retried bounded with proxy variables removed for the Ray probe. Original log retained in live-node-nccl-probe.log. This is inspection failure, not a second training failure.

- Same-node control completed through the existing Ray dashboard on the approved RJob; no new GPU worker was allocated. The provisioned gpu-h800-0187 interface list contains ens22np0 and no ens20np0. All8live outer generation actors inherited head ens20np0. The exact vLLM NCCLLibrary / NCCL2.30.7 call returns1 with `Bootstrap : no socket interface found` for ens20np0, and0 / NCCL_UNIQUE_ID_PASS for ens22np0. Evidence: evidence/native-nccl-interface-control.json. This establishes node-specific environment forwarding as the initiating NCCL cause.
- Narrow launcher correction: retain all platform NCCL variables on every replica/Ray daemon, omit only NCCL_SOCKET_IFNAME and NCCL_IB_HCA from the native head driver subprocess. Native Ray environment snapshots then allow workers to inherit their own node's provisioned values. No NCCL transport disable, library installation, GPU layout change or tolerance change. Full native rerun remains necessary.
- Original job Failed08:55:01Z; head user command Error/exit1/restart0 at08:54:59Z; other replicas Stopped08:55:03/04Z by subsequent fail policy. Local launcher observed failed status and cleanup at08:55:06Z. No agent manual stop was requested. Native finished.json records failed. The intended native forwarding A/B read was not submitted because the dashboard had already closed; retain that connection-refused error as a probe limitation, not a passing check.
- Exact-object preterminal RCA covered live/historical Events, incident logs and existing DeepScan; no existing DeepScan. Audits timed out18s. Startup ConfigMap event at08:41:26Z precedes the successful worker start and is not the08:45:17Z cause. Evidence boundaries and terminal follow-up are in test_report_2026-11-12_native_nccl.md.
- Pinned Ruff check/format PASS. Revalidated both prepared subsequent configs after enabling the five-teacher native checkpoint/save_period5; five-teacher-hidden and main-early-checkpoint PASS in next-native-configs-validation-2.log. No GPU dependency installation was performed.
- After personal exact-job/owner/terminal checks and CPU validation, submitted only the affected5-update24H800 native control as exp-1112-165702-849504 at08:57:02Z, with original nonpreemptible codesign resources/model settings and corrected launcher. Launcher42091 must remain alive. Do not duplicate it or promote200-step acceptance before model execution evidence.

## 2026-11-12 — resumed corrected native execution at 17:07 HKT

- Reconciled requirements, plan, progress, design, review, task-owned edits and the live sessions. Launcher42091 and monitor68803 are alive; no additional GPU job was submitted. Exact job exp-1112-165702-849504 moved from Pending to Starting at09:06:33Z; queue approval09:06:26.831489121Z. Creator i-fengyicheng, nonpreemptible codesign and NFS100.96.128.195:/data/ycfeng:/data/ycfeng match the requested setup. Initial model/refit, successful optimizer updates and parity remain unverified.
- Old-job archive follow-up: read the exact rank-0 worker error files and finished.json through bounded personal read-only cloud access. The archived worker error files contain no traceback; finished.json confirms failed. This does not recover the final native head-driver exception. Keep that missing final traceback as a collection limitation; the separately reproduced NCCL initiating error remains the established business root cause. Raw summary: /data/ycfeng/tmp/nemo-mopd-20261112/old-native-head-error-summary.json.
- Native DebugWriter emits identical token_ids, token_loss_mask, generation_logprobs, prev_logprobs and metadata for each row. Its dump precedes training and measures the separate trainer logprob forward. Full-loss with reference-KL coefficient0 does not compute sampled-token logprobs during the training forward, so those dumps alone cannot verify the actual training forward. Do not change the objective or claim that missing mismatch metrics mean equality. Inspect real output before selecting the smallest additional comparison.
- The native node environment read script is prepared under the task tmp directory for the existing dashboard only; it performs no GPU allocation, model mutation or dependency installation. Check actual actors and native engine/refit before promoting five-teacher/main execution.

## 2026-11-12 — architecture environment failure and scoped repair

- Recovered exact terminal evidence for exp-1112-165702-849504 (UID134dafb7-6caa-4a76-96c1-ac0a4ff6e4ef). Head user command Error/exit1/restart0 at09:12:51Z; job Failed09:12:55Z, with subsequent any-task fail-policy cleanup. Launcher42091 exited1 after cleanup17:12:56.753HKT; monitor68803 exited0. No agent manual stop was requested. Evidence: evidence/self-diagnostic-4-terminal.json. Native successful optimizer updates remain0.
- Incident query_range09:11–09:14Z returned355rows belowlimit3000 and recovered the initiating17:12:47HKT traceback: setup_single_controller -> _build_trainer -> TQPolicy -> Policy.__init__, lm_policy.py:328, RuntimeError TORCH_CUDA_ARCH_LIST is not set. Live/historical job/replica Events read successfully; the09:07:18Z ConfigMap warning precedes successful execution and is not this failure. Existing DeepScan count0. Both all-operation audits timed out18s; missing initiating manual-stop actor/reason/request time remains UNKNOWN.
- Previous NCCL fix progressed through4 TP2 vLLM engines, weights, KV cache and CUDA graphs. Outer/inner actor network settings matched their generation node. This run used ens20np0 on all3nodes, so it does not independently verify the heterogeneous-node forwarding case. No initial refit, trainer forward, parity or updates passed.
- CPU native-guard reproduction used actual Policy.__init__ with a zero-size cluster and the launcher's actual pre-cloud prologue, mocking only CUDA hardware discovery. First probe failed due to an omitted OmegaConf resolver registration; corrected probe reproduced the exact architecture error (exit2). Raw commands/scripts: /data/ycfeng/tmp/nemo-mopd-20261112/probe_cuda_arch_guard.py and cuda-arch-guard-red-2.log. This is diagnostic execution, not a new persistent unit test.
- Scoped repair in tests/manual/mopd/run_on_rjob.py: discover allocated CUDA device capabilities and set TORCH_CUDA_ARCH_LIST only when absent, before Ray startup. Keep explicit inherited values. Record capabilities and final value in existing runtime receipts. Gain: satisfy native Megatron compilation setup from actual hardware without hardcoded architecture, new flags, dependency installation or core-guard changes. Expected: CPU native guard passes; actual H800 receipts show9.0 and trainer creation proceeds. GPU rerun remains required.

- CPU architecture verification PASS: absent variable reproduces native error, actual worker prologue yields9.0 and reaches deliberate zero-cluster guard, explicit9.0+PTX retained. Personal submission dry-run PASS. Corrected two probe-only errors (resolver registration and environment restoration); retained logs. Ruff initially queried public PyPI and timed out; reuse of cached corporate-index Ruff0.9.9 with --offline passed check/format without dependency installation. Report: test_report_2026-11-12_cuda_arch_environment.md. Exact personal old-job read confirms Failed before one affected resubmission.
- check-progress self-check after environment repair: mainline remains D; A/Q/B/C PASS, D FAILED native execution pending rerun, E/F TODO. S5 stale Starting state corrected to observed Failed. All D1–D7 CLEAR: repair advances actual trainer startup; no helper-framework expansion, unrelated optional-library fix or duplicate allocation. Next: one corrected self-control -> actual native trainer/refit/parity -> five-teacher/long-context ->200updates/native resume. GPU promotion remains conditional on actual control evidence.

- At17:30:36.877HKT submitted only the corrected24H800 self-control as exp-1112-173036-554443. Launcher71916 remains alive. Same5-update/32prompt/full-vocabulary/8192diagnostic and explicit company image/local NFS/personal codesign configuration. Worker now derives the missing architecture environment. Do not duplicate the allocation; actual trainer/refit/parity remainsUNVERIFIED.

- Actual corrected job Running09:36:06Z, UIDe7d21741-1753-4bec-8dd8-70d133f6f998; source100.96.128.195:/data/ycfeng:/data/ycfeng and personal codesign confirmed. All3worker receipts show actual capability[[9,0]] and architecture9.0. Read-only Ray job mopd-native-architecture-read-20261112-5 SUCCEEDED;12generation actors/processes retained9.0. Native Megatron constructor passed and8trainer workers initialized in68.25s. Thus the demonstrated missing-architecture guard is fixed on GPU; weight loading/refit/training remainUNVERIFIED. No secondGPUallocation or dependency installation.
- check-progress after GPU constructor milestone: still frontierD, native workflow remainsTODO/0updates despite the verified scoped repair. NoD1–D7/S1–S6 issue introduced; proceed with the existing native job rather than broadening tests or reopening inputs. Re-read before-steptron-run for the later formal phase; existing263failure/trace checks remain valid preparation, actual accounting remainsrequired.

- By09:41:44Z native student HF conversion/model setup/init_complete succeeded. Actual padded-vocabulary log:151936semantic tokens,0dummy tokens. Selfteacher worker initialization is in progress and reuses the completed native student conversion. Teacher/trainer architecture propagation also confirmed in the read-only node receipt; all8trainer actors retain9.0. Still0successful updates and no published train dump. Signed scoped launcher commit: f92b7685. Full acceptance remainsD/E/F outstanding.


## 2026-11-12 — self-control business and numerical evidence

- Collected pending analysis session98582 and launcher71916/monitor32270: all exit0. Exact job exp-1112-173036-554443 (UIDe7d21741-1753-4bec-8dd8-70d133f6f998) Succeeded09:44:05Z, all3replicas succeeded and native replica receipts match; controller5/5, trainer_version5. Backend cleanup Job Stop is logged after succeeded09:44:09Z, not an initiating interruption. No manual stop.
- Native TB confirms optimizer_update_successful=1 at each of5steps; skipped_step=0,32 valid samples/4096 response tokens each, constant LR~1e-6/WD0.01 and zero version lag. Loss/fullRKL values0.000237406/0.000450706/0.000456192/0.000513847/0.000554322; grad norms0.0349325/0.0496490/0.0642046/0.0666772/0.0567912, all finite and below clipping1.0. Decomposition max error9.20e-7; per-token RKL min>=-3.10e-7. Self residual remains a numerical-control question, not fully explained healthPASS.
-160actual samples/20480response tokens in five dumps match prompt prefix, response masks and version0–4. Independent scorer minus vLLM overall MAE0.014616026/signedmean-0.000690514/p950.0843612/p990.1227255/max0.3456368. All step means0.013099–0.016504<=0.02. IF-group mean0.0219348, code0.0168807, logic0.0123190, math0.00991985, stem0.0125118. No threshold change or token filtering. Investigate BF16/batch-shape/attention or conversion tails before formal promotion.
- All five group aliases exercised but teacher_model_unique=1, since this control uses the initial student as frozen selfteacher. Distinct-teacher coverage is pending. Dump scorer forward is distinct from actual training forward; the latter remainsUNVERIFIED.
- Raw downloads and analysis remain in /data/ycfeng/tmp/nemo-mopd-20261112/native-5-analysis-inputs, native-5-{scalars,parity}.json, analyze_native_control_5.py. Selected terminal, metrics and parity are evidence/self-native-5-{terminal,training-metrics,parity}.json. Native EventAccumulator initially returned no tags because copied filenames do not contain tfevents; direct EventFileLoader recovered2331records/378numeric tags without redownload.
- Mainline D: five-teacher5-update diagnostic can exercise independent teacher/routing/checkpoint capabilities while same-weight discrepancy investigation proceeds; E200updates remains gated on resolvedD. Do not repeat completed CPU/TP/HF controls.

- Submitted one distinct-five-teacher diagnostic at17:53:10HKT: exp-1112-175309-039353,56H800/codesign,7×8,5updates/32prompts/128generation/native checkpoint every5. Personal launcher74027 stays alive. Dry-runPASS; exact args retained in tasktmp/five-teacher-diagnostic-submit-args.json. No formal200-update promotion.

## 2026-11-12 — native-forward control preparation at 18:08 HKT

- Reconciled exact five-teacher job exp-1112-175309-039353: Pending10:06:04Z, creator i-fengyicheng, nonpreemptible/codesign, queue rank2, remaining H80047 versus request56. Launcher74027 and monitor15762 are alive. Retain this job; no duplicate submission.
- The test-local observer initially inherited the Ray ActorClass rather than MegatronPolicyWorkerImpl. Corrected before GPU launch, copying the original runtime_env decorator. Actual MCore interpreter import/builder FQN check PASS in native-observer-import.log; Ruff check PASS. This establishes interfaces/import only, not model execution. Inspected native packing/loss/postprocessor call sites: keyword arguments match; DP training schema retains generation_logprobs; teacher config deep-copies student env_vars.
- D hypothesis controls, recorded before launch: (1) BF16 head output rounding: paired FP32-output heads should reduce same-weight trainer/vLLM MAE and tails; teacher direct/reconstructed comparison reveals whether reconstruction has matching precision. (2) Packing/batch shape: actual training minus independent scorer remains nonzero when the same-backend head is unchanged. (3) Trunk/attention or checkpoint conversion: differences remain after paired head control and are larger than same-forward reconstruction errors. No numerical threshold changes or token filtering.
- Prepared two24-H800/3-node, nonpreemptible native5-update/32-prompt controls, BF16 and paired FP32 heads with the explicit teacher override. Both keep initial student as selfteacher to isolate numerical paths, so these do not satisfy distinct-teacher training. The observer wraps native loss prep and teacher postprocessing, retains original results and gradients, and writes detached training/scorer/generation LP plus all TP/CP logit-shard statistics. No core trainer/objective change. They use separate personal output prefixes native-bf16-observed and native-fp32-observed. Current56-GPU distinct-teacher run retains the ordinary native entrypoint.

- After both CPU config/paired-head validation and personal launcher dry-run PASS, submitted BF16 observed exp-1112-181156-140223 (launcher83836) and paired-FP32 observed exp-1112-181232-890845 (launcher44927). Read-only monitor77515 tracks both. At10:13:23Z BF16 was Running and FP32 Starting; creator/source/nonpreemptible/codesign receipts are in native-observed-job-watch/. Total requested current slices104H800; these are independent numerical controls, not retries of the56-GPU distinct-teacher job.
- D long-context work has no numerical-head prerequisite: prepared one additional24H800 native1-update control with the existing5 synthetic rows (65504 prompt tokens,32 generated tokens,65536 ceiling), BF16/selfteacher, ordinary entrypoint and unchanged native dump/trace. Source is the500KB temporary validation JSONL on the existing local source mount; SHA256 bc162cf4696b1aefc828124671b4e2e1174cb02ef7e8f7dfe1f705845e6a525e. Production pools unchanged. It measures actual sequence lengths, memory, positive loss-token coverage and successful native execution; it does not establish long-context coverage for every distinct teacher. All4independent requested slices would total128H800. Validate before submission.

- Long-control config and personal dry-run PASS (native-long-context-config-validation.log / native-long-context-dry-run.log); submitted exactly one exp-1112-181524-408383, launcher16259, read-only monitor68778. All4active task slices request128H800 in total. Native200-update main run remains unsubmitted. Current job identities/sessions/source/quota receipts: evidence/native-controls-current-jobs.json.
- Repeated final Ruff invocation initially missed the existing cached corporate index identity; setting the same UV_INDEX_URL as the earlier successful call restored offline Ruff0.9.9, with4files unchanged and all checks passing. No install or network retry. CPU native-observer import, native paired-head configs and both dry-runs PASS; GPU forward measurements remain pending.


## 2026-11-12 — actual forward controls completed and projection defect isolated

- BF16 exp-1112-181156-140223 and paired-FP32 exp-1112-181232-890845 Succeeded; launcher83836/44927 and monitor77515 exit0. Each5/5successful optimizer updates, zero skipped, finite losses/gradients. Selected terminal/TB/parity evidence and test_report_2026-11-12_native_forward_precision.md now retain actual results.
- Actual train/scorer max2.38e-7/3.57e-7; trainer/vLLM MAE0.01458138/0.00989524. FP32 every group passes0.02; IF drops0.0220399→0.0168749. No tolerance changes or token filtering. Same-forward teacher reconstruction: BF16 all1280rankrecords exact; FP32 maxlogit0.125/mean0.00253643 and sampledLPMAE0.00945954. Native reconstruction loses the configured FP32 output through BF16 weight-typed matmul. Repair must preserve the teacher's own output precision across load/offload/evict, not infer it from the student.
- First analyzer assertion ignored sequence padding (762valid versus768padded); corrected valid slicing only after checking full array lengths.160/160token identities, masks and native scorer/generation values match. Optional runtime-rank-0.json was absent during tar collection; all business observations/5dumps/TB/finished files were recovered. Original stderr retained.
- Prepared real loader→reconstruction/eviction regression in existing test_megatron_worker.py before the fix. First pytest command collected0 because mcore tests are excluded by default; adding-m mcore reaches unrelated global Ray fixture then aborts with GRPC_CALL_ERROR_INVALID_METADATA. Preserve both logs. Run the pure checkpoint-loader regression with--noconftest, without changing Ray/core behavior or GPU jobs.
- Latest monitored five-teacher job remainsPending10:26:24Z; long-contextRunning10:26:34Z. No duplicate submissions, new dependencies or manual stops. Production200updates still unsubmitted.

## 2026-11-12 — precision repair, long-context failure, and affected GPU revalidation

- Correct loader→projection regression reproduced FP32-to-BF16 rounding before the fix. Store each teacher's configured logit dtype and reload its head in that dtype after eviction; mixed teacher projections promote only the scatter buffer, preserving each teacher's own matmul precision. Gain: hidden-state reconstruction matches the teacher output contract without inferring precision from the student. No objective or threshold change.
- CPU regressions PASS: 9 loader/lifecycle/off-last-PP checks and all 59 full-loss checks. Logs: tasktmp/teacher-head-precision-green.log and teacher-head-projection-regression-2.log. The mixed-head test first failed the old dtype guard, then exposed advanced-index source/destination dtype requirements; the final cast after each teacher's matmul preserves its rounding and passes. The unrelated global Ray fixture abort is retained and not repaired; pure checkpoint checks use --noconftest and -m mcore. Corrected observer import and both native configs PASS before launch.
- BF16 long-context exp-1112-181524-408383 Succeeded; five actual rows each contain 65,504 prompt + 32 response = 65,536 tokens. One successful update, skipped_step=0, valid sequences=5, valid tokens=160; loss=0 and grad norm=0 are finite. Independent scorer/vLLM MAE=0.1361867711, p95=0.320202446, p99=2.77063072, max=6.68006516. All five per-sample means exceed 0.02. Raw finite_pass=false includes defined singleton stddev NaN (experience/metric_utils.py:47), not a NaN loss or gradient. Evidence: native-long-context-parity.json; no formal promotion.
- Recorded long-context hypotheses before revalidation: head rounding (paired FP32 should remove most error if causal), long-position attention/RoPE (error remains in FP32 and grows with context), CP/packing (same tokens with CP1/unpacked should reduce it if causal). Test paired heads first, then isolate remaining paths using existing native forwards. Short-sequence PASS cannot settle the 65,536-token criterion.
- Both new dry-run sessions27571/2164 exited0 with personal H800/codesign and local mount payload. First local submission invocation lacked steptron on PYTHONPATH and exited before importing ResourceConfig or creating a job; retained native-fp32-corrected-submit.log. Reusing the handbook PYTHONPATH resolves it without environment installation.
- Submitted exactly two affected controls at18:39HKT: corrected short FP32 exp-1112-183916-289473, launcher36686, and long FP32 exp-1112-183922-688559, launcher78627. Read-only monitor59834: short Running and long Pending at10:39:45Z. Each24H800; original distinct-teacher56H800 job exp-1112-175309-039353 and launcher74027/monitor15762 remain alive. Total requested current slices104H800. Exact args and commands are tasktmp/native-fp32-corrected-submit-args.json, native-long-fp32-submit-args.json and submit_native_{fp32_corrected,long_fp32}.py. Main200updates remains unsubmitted.
- Corrected short control adds detached full-logprob, native reverse-KL loss and probe-gradient errors on every TP/CP rank. GPU observations are still UNVERIFIED. The probe does not connect to native model parameters and is restricted to this24GPU diagnostic; retain ordinary entrypoints for the56GPU production diagnostic.

## 2026-11-12 — fixed-token CP replay prepared

- Both affected FP32 jobs are Running at10:46:12Z; initial model/vLLM/selfteacher construction is proceeding. The original56GPU distinct-teacher queue job remains intact.
- D CP hypothesis can be tested independently using the five recorded BF16 long trajectories, without new generation or teacher inference. Added tests/manual/mopd/replay_logprobs.py: ordinary Policy.get_logprobs, optimizer/reference disabled, same checkpoint/config/IDs/mask, sequential CP4 and CP1 in one8GPU allocation. CP4 uses8GPUs; CP1 uses2GPUs at DP1 so no data-parallel or batch-shape change. All samples are scored one at a time under unchanged sequence packing.
- Prediction before submission: CP4 replay should match recorded native scorer outputs; if CP1 reduces the 0.13619 generation gap substantially while CP4 reproduces it, CP is causal. If CP1 and CP4 remain close but both disagree with vLLM, investigate shared long-context attention/RoPE. Existing raw masks and response tokens remain unchanged. Replay compares sampled LP only; it is not another trainer or a five-teacher/main-run replacement.
- CPU actual script import/help exit0 and Ruff0.9.9 PASS; personal8GPU dry-run payload PASS. First offline Ruff call selected the default cache, which lacks this tool; setting the already established task UV_CACHE_DIR resolves it without dependency/network changes. Prepared native-long-cp-replay-submit-args.json and submit_native_long_cp_replay.py; no job created by dry-run.

## 2026-11-12 — corrected precision GPU verification completed

- Corrected FP32 exp-1112-183916-289473 Succeeded; native5updates/trainer_version5 and all5TB successful-update counters=1, skipped0. Collected13,854,720bytes via read-only SSH, exit0; native finished.json included. All160actual forwards/20,480response tokens and1280rank records complete. Trainer/vLLM MAE0.009914506, everygroup≤0.02; trainer/scorer max2.68e-7. Reconstructed/direct LP MAE3.15e-6; full-logit max1.1444e-4/fullLPmax1.4496e-4; loss error5.5732e-7. See native_forward_precision report and corrected evidence.
- Raw all-position SUM probe gradient max1.6684e-5 exceeds an unnormalized1e-5 assertion. Preserve it. Inspected native token_level_loss=true and finish_train_step1/global_valid_toks correction; all5updates denominator4096. Fixed-teacher KL has independent positions, so response masking cannot increase coordinate error; normalized native logit-gradient error upper bound4.0733e-9 passes existing1e-5. This is a bound, not observed parameter-gradient equality; reference-zero relative error remains undefined. No model/mask/normalizer/tolerance change. Projection residual is FP32 GEMM accumulation, no BF16-output precision loss remains.
- Loss range8.52e-5–2.11e-4 and grad norm0.02514–0.06578 finite, clipping1.0; no native training-health anomaly in this control. Five distinct teachers and long parity still pending.
- Fixed-token replay submitted once as exp-1112-184750-688596 at18:47:50HKT,8H800, launcher67340, monitor92156. It is Running at10:52:10Z. LongFP32 exp-1112-183922-688559 alsoRunning; original distinct-teacher exp-1112-175309-039353 retained. Total requested current slices88H800 after corrected short completion. No duplicate allocations or200update submission.

## 2026-11-12 — resumption, actual terminal reconciliation, and CP replay repair

- FP32 long exp-1112-183922-688559 is Succeeded at10:52:10Z, all3replicas succeeded, native finished.json agrees and SC completed1/1update. Worker stacks/engine exits follow business completion and are normal cleanup, not initiating failure. Collected its native dump/TB/finished via read-only SSH; selected evidence now records160response tokens, each65,536total, loss1.50832e-8, grad norm0.00406362, successful1/skipped0.
- FP32 long scorer/vLLM MAE0.1310551181, p950.31538687,p992.74242068,max6.69283390; all5sample means exceed0.02. Paired FP32 heads do not resolve the long-context discrepancy. Preserve the threshold and existing mask; CP replay is the next independent hypothesis test.
- CP replay exp-1112-184750-688596 Failed10:53:14Z. Exact incident-bound RCA collector retained objects/live+historical Events/logs/existing DeepScan, exit1 because both all-operation audit reads timed out. Actual initiating assertion10:53:06Z: train_iters missing; native exit1 at10:53:10Z precedes launcher cleanup10:53:16Z. Earlier ConfigMap wait resolved before worker start; no existing DeepScan and no positive preemption/OOM evidence returned. Stop actor/reason/request-time remainUNKNOWN. Raw collector: tasktmp/rca-cp-replay-init-1; selected evidence/native-cp-replay-init-rca.json.
- Reused _maybe_inject_megatron_train_iters in replay_logprobs.py. CPU actual native _validate_training_config reproduced the original assertion, then passed CP4 andCP1 after preparation, train_iters1. Actual base-runtime CLI help and Ruff check/format PASS. Exact commands: tasktmp/check_replay_native_config.py using mounted MCore interpreter through uv run --no-sync; output replay-native-config-red-green.log. No core guard, dependency, mask or numerical tolerance change.
- Submitted exactly one affected8H800 rerun exp-1112-185915-382589 at18:59:16HKT, launcher53211, read-only monitor73684. New output prefix native-long-cp-replay-2 retains first failure. Original distinct-five-teacher exp-1112-175309-039353 remainsPending, launcher74027/monitor15762 alive, latest queue remaining48/request56. No duplicate five-teacher job or200-update submission.
- check-progress applied: A/Q/B/C PASS, D/E/F TODO; S5 stale FP32-long status corrected in plan/progress/review. Independent five-teacher workflow remains queued while CP diagnosis proceeds; no broader tests or unrelated Ray/NIXL environment repair.

- At11:04:34Z exact personal API reads confirmed five-teacher exp-1112-175309-039353 Running with7active replicas (UID84a2ec3c-9eac-40b4-b0c9-bc88c1810f70), CP2 replay exp-1112-185915-382589 Pending (UIDf22d72a5-0b0e-486e-8759-0bcc59685147). Both local NFS source100.96.128.195:/data/ycfeng:/data/ycfeng. Five-teacher math init completed; code loading.
- Verified observer/entrypoint slice committed with sign-off as fbb114fe, after final Ruff check/format PASS and prior actual GPU forward verification. Replay remains uncommitted until GPU scoring succeeds. check-progress after this deliverable retains Dfrontier; continue actual five-teacher workflow and queued CP2, without repeated CPU tests.

- Before submitting the next independent long-context control: fused RoPE is a remaining numerical hypothesis distinct from CP. If its angular/cos-sin application explains the discrepancy, CP1 with existing apply_rope_fusion=false should change fixed-token logprobs toward vLLM while CP1/fused retains the gap. The2GPU replay retains checkpoint, BF16 head, packing, DP1, token IDs and masks; it only changes the existing fusion setting. Store response logprobs for160positions so independent variants can be compared without another baseline run. No core behavior fix is claimed before measurements.

- Unfused-RoPE control dry-run and replay Ruff PASS; submitted2H800 exp-1112-190831-616219 at19:08:31HKT, launcher76780. This is an independent fixed-token CP1 comparison, not another training run. Existing five-teacher56GPU and CP8GPU jobs remain intact; total current task requests66H800. Exact args: tasktmp/native-long-unfused-rope-replay-submit-args.json.

- Formal resume entrypoint prepared at tests/manual/mopd/resume_training.py: two separate ordinary controller subprocesses in the same allocation; initial native checkpoint deadline followed by identical horizon and deadline=null on resume. No auxiliary validator blocks either native stage. Actual base-runtime help and Ruff PASS; formal GPU execution remainsUNVERIFIED and script uncommitted. This reuses tests/functional/grpo_checkpoint_single_controller.sh native timeout/save/resume route without its hard auxiliary checks or cleanup commands.

## 2026-11-12 — five distinct teachers and fixed-token CP replay completed

- Five-teacher exp-1112-175309-039353 Succeeded11:11:24Z, all7replicas, native5/5successful optimizer updates and zero skipped. All5teachers exercised at every update,32validsamples/4096response tokens perupdate, versionlag0, loss0.111503–0.171106 and pre-clippinggradnorm1.777639–3.039701. Native optimizer computes this norm before maxnorm1.0 clipping; no arbitrary scaling change.160samples/20480response tokens agree in IDs/masks/version. Overall independent scorer/vLLMMAE0.014508 passes but IFgroup0.022422 fails0.02; preserve groupFAIL pending pairedFP32 revalidation.
- Native step5 saved model/optimizer/scheduler, training_info, dataloader, sampler/replacement/recovery and data-plane metadata. training_info current_step5/trainer_version5/consumed160/total_valid_tokens20480. Read-onlyselectedtensor comparison initial→step5 covers144896BF16elements:429changed, absolutechangemax7.629395e-6, allselectedvaluesfinite. Finalnormunchanged; fullmodel/optimizerfinite scan and actualreload remainUNVERIFIED. RemotePython3.12initialmetadata firstfailed pathlib._local; resolved analysis by localnativePython3.13metadata reads and exactremote byte ranges (no model or environmentfix).
- CP replay exp-1112-185915-382589 Succeeded; CP4matches recordedscorer exactly160/160tokens. CP4vLLMMAE0.13618677 vsCP1 0.12153486, p99/maxrecorded in evidence/native-long-cp-replay.json. CP1minusCP4MAE0.04371640/max0.72641373. CPaffectsnumerics but CP1 alone doesnotmeet0.02. Head/CPcontrols have narrowed remaininglongattention/RoPEpath.
- UnfusedRoPEexp-1112-190831-616219 Failed11:10:13Z beforemodelinit: argparse interpreted trailing policy.megatron_cfg.apply_rope_fusion=false asCPinteger. Nativeexit2 andlaunchercleanup followthiserror. ScopedRCAobjects/events/incidentlogs/deepscan readscompleted; allauditqueries45stimeout; manualstopactor/reason/requesttimeUNKNOWN. Reorderedthisoverridebefore --dump; actualscriptCLI RED/GREEN reachesinit_ray aftercorrectedconfig. No corecode or tolerancechange. Affected2GPUrerunexp-1112-191624-359257 submittedonce19:16:24HKT, launcher67067/monitor54661, nowRunning.
- Independentfive-teacherFP32save/resume diagnostic exp-1112-191947-030955 submitted19:19:47HKT,56H800, launcher95310/monitor94771. CPUactualnativevalidator andpersonaldry-runPASS. Keep5-step schedulerhorizon fromstart; deadline00:00:00:01 oninitialstage thennull onresume. Allpairedfp32_lm_head overrides matchverifiedshortcontrol. This isshortsave/resume verification, notformal200-update promotion. Temporaryconfigcheck initiallyusedwrongvalidatorimport anddictaccess forBaseModels; finalchecksinspectedactualtypes andpassed. Do notextendthisharness.

## 2026-11-12 — RoPE control completed; next attention hypothesis

- CP1 unfused RoPE replay exp-1112-191624-359257 Succeeded. Its same-token vLLM MAE is 0.10059409, versus fused CP1 0.12153486. All five sample means still exceed 0.02. This excludes disabling RoPE fusion as a sufficient remedy. Native numerical result, not merely successful exit, controls this verdict.
- Before the next native replay: if the selected attention implementation explains the long-context error, forcing the existing flash backend and flash_attention_version=4 should materially reduce CP1/fused same-token response errors; unchanged error excludes this implementation choice as a sufficient cause. Local pinned vLLM selects FA3 by default on H800 when supported, while Megatron config defaults to FA2 with auto backend selection; the MCore environment has FA4 installed. Keep tokens, checkpoint, head dtype, mask, TP2/DP1/CP1, and RoPE fusion unchanged. Prepare one 2-GPU FA4 replay. If this fails numerically, next compare vLLM prompt logprobs against its own retained generation logprobs to isolate prefill/decode history.
- Ruff initially failed offline resolution through the PATH uv. Using the existing pinned task uv at /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv and the known corporate cache/index passes. No dependency download or installation.


## 2026-11-12 — resume recovery and attention controls submitted

- Existing FP32 save/resume `exp-1112-191947-030955` remains Running through 11:32Z, launcher95310 and monitor94771 alive. Actual log still says initial phase and is building the final teacher; no resume or native update is claimed yet. Shared recipe is unchanged during this two-stage execution.
- Collected completed CP1 unfused RoPE: MAE0.1005940875, p950.24005744, p990.51850528, max5.26047039 across all160 response tokens. All5sample means fail0.02. Selected evidence/native-long-unfused-rope-replay.json retains response outputs; this excludes disabling fusion alone as a sufficient remedy.
- Submitted one FA4 fixed-token CP1/fused replay as `exp-1112-193156-178680` at19:31:56HKT (2H800, launcher27944). Submitted one native vLLM prompt scoring replay as `exp-1112-193212-322978` at19:32:12HKT (2H800, launcher26006). Its actual CPU CLI/config and personal dry-run PASS. Native synchronous prompt scoring uses the recorded65536 tokens, with engine capacity65537 solely for one unused new output token; formal ceiling remains65536. Ray executor preserves worker platform NCCL settings.
- Monitor59262 uses watch_attention_and_generation_replays.py and output attention-and-generation-replay-watch. Personal creator, codesign, current-machine NFS100.96.128.195:/data/ycfeng:/data/ycfeng verified. Latest FA4Starting/vLLMPending; queue reports insufficient current quota, so retain both launchers and do not duplicate. Requested task allocation60GPUs including56GPU resume.
- Final Ruff0.9.9 check and format PASS for replay_logprobs.py, replay_generation_logprobs.py and resume_training.py. An offline --with invocation missed the cached index identity; executing the existing cached Ruff binary through pinned uv run --no-sync succeeds without installation. Only GPU-verified replay_logprobs.py is ready for a scoped code commit; native resume and vLLM scripts remain unverified pending actual execution.

- Verified fixed-token replay and selected five-teacher/CP/RoPE evidence committed with sign-off as1b61f3b9. Required check-progress checkpoint applied in review.md: A/Q/B/C PASS; D/E/F TODO. S5 stale RoPE/initial status corrected; no D1–D7 drift, no broadened testing or objective change. Next actual frontier is collecting FA4/vLLM replay while native resume continues.


## 2026-11-12 — FA4 result, actual resume, and formal sample preparation

- FA4 `exp-1112-193156-178680` completed native replay and finished.json; launcher27944 exit0. All160 same response tokens remain finite, MAE0.1298855153, p950.19516401, p992.84165693, max6.74194527; all5sample means fail0.02. Backend-version switch is not a sufficient remedy. No further FA4 baseline needed.
- FP32 save/resume actually entered resume: step_1 saved, MOPD_CHECKPOINT_PHASE resume and dataloader restore observed. Native student loaded step_1/policy/weights at11:37Z. Its first32samples/4096response tokens have scorer/vLLMMAE0.0101784620 and everygroup≤0.02 (IF0.0157096689). Initial step1 metadata is current_step1/trainer_version1/consumed32/valid4096; resumed updates and optimizer/scheduler/data-state acceptance remain pending. Raw step1 parity: tasktmp/five-teacher-fp32-resume-parity.json.
- Native vLLM prompt replay still initializing. Actual synchronous engine chooses max_num_batched_tokens16384; recorded async engine uses2048. This diagnostic compares native synchronous prompt scoring against original decode scores, including that native default difference. It cannot attribute any observed gap exclusively to prompt-vs-decode; match2048 via existing vllm_kwargs before a subsequent isolation if needed. No shared recipe change while resume runs.
- Loaded before-steptron-run for reusable formal sample-preparation requirements. Existing native trace links group/prompt IDs, attempt number, environment instance, run timestamp; native token/data-plane metadata links weightversion/step/sample. Timeout/data/infra outcomes are classified at existing sample boundaries and replacement preserves full batch size. Existing focused tests cover success, timeout cancellation, dropped-attempt trace, replacement success and fraction1 floor.
- CPU preparation PASS:8tests,72deselected in13.51s with native basePython3.13 via pinned uv run --no-sync python -m pytest --noconftest; exact full command/raw log is tasktmp/formal-sample-failure-preflight-3.log. Before that systemd-run --user failed missing bus, and system scope needed interactive authentication; retained both native errors then applied30GiB ulimit andtimeout60s to the focused CPU invocation. No host permissions or services changed. No production code change or new tests required for this preparation.
- Remaining long-context hypotheses after head/CP/RoPE/backend controls: (1) vLLM prefill/decode and chunked-prefill history differ—native prompt replay should expose an internal gap if causal; (2) RoPE frequency or application rounding differs between engines—same-input per-layer q/k and cached cos/sin should show the first divergence if causal; (3) BF16 GEMM/residual/norm accumulation differs—layerwise errors should arise outside RoPE and increase through the trunk. Follow the actual prompt-replay evidence before introducing another probe.


- Completed vLLM default prompt replay: all5samples/160tokens, prompt-minus-original-generationMAE0.0368034811, p990.28300462/max0.99954462; every samplemean>0.02. It also disagrees with originalscorer, especiallysample4. Same-engine scoring history/defaults contribute error, but do not explain the whole native gap. Before next control: if prefill chunk shape2048vs16384 is responsible, matching existing vllm_kwargs.max_num_batched_tokens=2048 should move prompt scores toward originalgeneration; otherwise this default mismatch is insufficient. KeepBF16,weights,mask,positions,TP2,FA3 andprefixcache unchanged. Prepare one2GPUmatched2048run using the alreadyGPU-verified replay script. Formal run waits forD.


## 2026-11-12 — recovered terminal results and layer comparison preparation

- Refreshed existing sessions: FP32 resume exp-1112-191947-030955 Succeeded11:43:50Z with7replicas; matched2048 vLLM replay exp-1112-194130-498485 Succeeded11:45:09Z. Both launchers exited0. No job resubmission or stop.
- FP32 save/resume completed160samples/20480response tokens through step5; scorer/vLLMMAE0.00977465355, p990.08889551, max0.44571996. Every step/group passes0.02, IF0.01607742233. Actual training-forward parity is not observed in this particular run. TB and native state/finite inspection remain in progress.
- Matched2048 prompt replay MAE0.03742300957, p990.35544111, max0.91518831:FAIL. Matching prefill chunk does not resolve the engine-internal scoring difference.
- Before layer comparison: if RoPE/attention causes the long discrepancy, embedding should match and the first attention layer should diverge; if later residual/norm/MLP accumulation dominates, early layers should agree substantially better than later layers. Capture identical response prediction positions (mask indices minus1), embedding,28decoder layers and final norm for the original five65536-token samples. Native retainsCP1/TP2/fused; vLLM usesTP2/FA3/chunk2048 and requires eager hooks, whose logprob behavior must be compared separately. No core fix is claimed.
- Corrected unsubmitted native observer metadata: capture positions are per-sample Python lists, survive actual sequence-packing sharding, and are removed before native microbatch construction. Native actual CLI/config reaches init_ray; generation actual CLI/config reaches engine boundary; pinnedRuff0.9.9check/format and both personal submit dry-runsPASS. These are preparation checks, GPU hooks remainUNVERIFIED.

- Completed native resume state audit: TBsuccessful5/skipped0 atsteps1–5,loss0.10995886–0.16735572,gradnorm1.69602823–2.98565817,LR1e-6/versionlag0. NativeoptimizerbothTPgroupsAdamstep5; schedulernum_steps160/decay_steps160,train_stateconsumed160,establishcontinuedstateafter4resumedupdates. ActualTQemptyboundaryvalidatedand32replacementreserverestored. Fullread-onlycheckpointscan8files/1801chunks/7108614656elementsPASS:1777088000modeland5331526656optimzervaluesfinite. Exactreport:test_report_2026-11-12_native_resume.md.
- Native layer job exp-1112-194949-717142 Succeeded11:52:33Z and captured30stagesat32predictionpositionsforeachofthe5samples. vLLMlayerjob exp-1112-194949-673462 Failed11:52:31Z:callableMsgpackserializationTypeError atapply_model11:52:21Z precedesenginecleanup. NativevLLMworker.py:654setsVLLM_ALLOW_INSECURE_SERIALIZATION=1; thediagnosticomittedthisexistingworkerinitialization. Restorethatsameenvironmentsettinginthehookentrypoint; no model/runtime-dependencychange. IncidentRCAobjects/live+historicalEvents/logs/DeepScanqueriescomplete; bothall-operationaudits18stimeout,stopactor/reason/timeUNKNOWN. Nativehookstatesretained; onlyaffectedvLLMprobeiseligibleforrerunafteractualserializationRED/GREEN.

## 2026-11-12 — layer comparison resumed at 20:02 HKT

- Recovered launcher26264 and monitor84411 for exp-1112-195948-825359; existing rerun Running12:01:55Z. No duplicate job or formal200-update run submitted. Native layer launcher46502 exit0; first generation-layer launcher93820 exit1 matches its recorded serialization traceback.
- Native capture scorer control session72945 exit0:160 response tokens versus prior CP1/fused replay, MAE0.0/max0.0. Hook observation preserves the native scorer result for the actual long-context inputs. Raw captures: /data/ycfeng/tmp/nemo-mopd-20261112/long-native-layers-inputs/.
- Next D step: collect eager generation-layer results, independently quantify its logprob behavior, then compare identical global prediction positions across30stages. Diagnose the first measured divergence before changing native behavior; unchanged0.02 criterion.

- Generation-layer rerun Succeeded12:02:35Z; launcher26264/monitor84411 exit0. Five samples/30stages/32positions and replay LP retained at tasktmp/long-generation-layers-2-inputs and personal cloud long-generation-layers-2/exp-1112-195948-825359. Optional runtime-rank-0.json/resolved-config.yaml absent during tar; business artifacts recovered. First CPU comparison overthreaded small tensors; interrupted only analysisPID830191 and reran withOMP/MKL/OPENBLAS4 under60s, not a GPU lifecycle change.
- Eager prompt versus matched2048 compiled prompt MAE0.03340532/max0.61597967; eager versus originalgeneration MAE0.04009260:FAIL. Do not treat eager as the originalgeneration. First-layer relativeRMSE0.03269–0.04731; selected later states differ increasingly, particularlysample884048b5.
- Diagnosed observation defect: generation embedding was read by a post-hook after fused residual/norm storage mutation; its RMS~0.61 versus true nativeembedding~0.032 proves the capturedstage is not comparable. Exclude thatstage from inference; retain rawcomparison. Change to pre-hook, snapshot before mutation. Actual pinnedvLLM get_rope reuses onecachedmodule, so current-layer identity restricts rotaryhooks to intendedlayers0/1.
- Before next4GPU pairedcapture: (1) embedding/checkpoint conversion cause predicts differing correctedembedding or QKV; (2) RoPE cause predicts close pre-rotationQKV but sharply differing post-rotationQ/K or actualcos/sin; (3) attention accumulation cause predicts matchingQ/K but larger coreattentionoutputdifference. Capture first2layers only at the same32globalpredictionpositions; compare nativefusedCP1/TP2 with eagerFA3/chunk2048. No productionbehavior or tolerancechange.

- Capture-boundary CPU preparation: first invocation omitted projectPYTHONPATH and failedimporttests.manual; corrected invocationPASS with existingbasePython3.13. Nativeobserver import firstomitted the existingCUDAcompatloaderpath; preservederrorlibcuda.so.1 and reran with the worker's alreadyvalidatedLD_LIBRARY_PATH. No dependency installation. Bothpersonal2GPUdry-runsPASS.

- Nativeobserver actualpinnedruntimeimportPASS after restoringexistingCUDAcompatpath;captureboundaryPASS andbothdry-runsPASS. Submittedpairedcaptures20:09:33HKT: nativeexp-1112-200933-746577 (launcher80034),generationexp-1112-200933-693626 (launcher40877),2H800each. Keepbothlaunchersalive; noformalrun. Exactargs/script/logs use long-{native,generation}-attention-layers in tasktmp.

- Corrected archive filename fromrun_on_rjob.py: actualruntime receipt isruntime-0.json, successfullyreadfromtheexistinggenerationlayerjob. Originaltarrequestedruntime-rank-0.json, whichwasanincorrectinspectionfilename, notproofofmissingworkerreceipt. ActualruntimePython3.13.14/Torch2.13.0+cu130/CUDA13.0 and2H800/capability9.0 confirmed.

- PairedattentioncapturesSucceeded12:12:41Z(native)/12:12:37Z(generation);launchers80034/40877 andmonitor71499 exit0. Correctedembeddingexactall5samples. AllfirstlayerQKV,preRoPEQ/KexactforbothTPranks;afterRoPEQmax0.5/Kmax2.0,attentionrelativeRMSE0.03661–0.16842. NativeFP32angletrigcasttoBF16matchesvLLMcacheexactly. NativeactualrotationmatchesFP32trigformula;vLLMactualrotationmatchesBF16cacheformulaexactly. FirstdifferenceisRoPEcos/sinprecision,notvocab/token/weightconversionorfirstprojection. Selectedcapturecomparison:tasktmp/attention-layer-comparison.json.
- Beforeprecisionintervention: usevLLM'salreadyimplementedFlashInferrotaryoperation,whichrequiresFP32cos_sin_cacheandretainsBF16Q/K,viaitsnativeapply_modelAPIinthetest-onlyeagerprobe. RecomputefullFP32cachefromthesamemodulefrequencies,avoidingupcastofalreadyroundedBF16cache. Prediction: firstlayerpostRoPEQ/Kalignwithnative; ifcoreattention/endLPstilldiffer,theearliestcauseisconfirmedbutinsufficient. CompareagainstnativeCP1scorerandretainoriginalgenerationdifferenceasaseparatehistoricalcontrol. Noproductionpatchorformalrunyet.

- FP32 RoPE preflight completed using the actual pinned vLLM RotaryEmbedding: rebuilding the FP32 cache preserves device and its BF16 projection equals the original cache exactly; selecting the existing FlashInfer operation passed. CLI reaches the original engine boundary with no test-only flag leaking into Hydra. First CPU VllmConfig auto-device inference failed on this CPU host; rerun explicitly selects DeviceConfig(device="cpu"). Logs: fp32-rope-preflight.log (original) and fp32-rope-preflight-2.log (PASS). No dependency repair or GPU allocation was used for that check.

- Submitted one FP32 RoPE intervention as exp-1112-201719-338060 at20:17:19HKT,2H800/codesign, launcher57505. Keep the launcher alive. Exact wrapper submit_long_generation_fp32_rope.py and long-generation-fp32-rope-submit-args.json; no formal200-update submission. Next compare its Q/K and all160 response logprobs to the saved nativeCP1 scorer.


## 2026-11-12 — FP32 RoPE terminal reconciliation and remaining MLP hypothesis

- D: exp-1112-201719-338060 Succeeded12:20:27Z; launcher57505/monitor59780 exit0. No duplicate submission or formal200-update run. Selected results archived in native-long-fp32-rope-{terminal,parity,layer-comparison}.json and RoPE report. First post-RoPE Q/K MAE/max0, cos/sin max5.96e-8; attention relativeRMSE0.000335–0.000630. Overall160-token nativeCP1/eager-vLLMMAE0.03176861 stillFAIL0.02; four of five samplemeansFAIL, p990.21216241/max0.50518227. First divergence repaired only in the test-only vLLM route, not production.
- D pre-execution hypotheses ranked from inspected code and observed first-layer remainder: (1) unfused native SwiGLU rounds its BF16 SiLU intermediate; existing bias_activation_fusion=true should reduce MLP/layer/final-LP difference if causal. (2) residual/RMSNorm arithmetic differs; capture first remaining inputs/outputs if fusion is insufficient. (3) row-projection or attention reduction differences accumulate over28layers; matching activation with continued projection/layer differences would support this. Next is one2H800 nativeCP1/TP2 replay against retained corrected vLLM output; checkpoint/tokens/masks/head/RoPE unchanged. No recipe or production schema change yet.
- check-progress after resumed key-step reconciliation: A/Q/B/C PASS; D/E/F TODO. D1–D7 CLEAR: measurement narrows actual precision cause, one native fusion comparison advances D, no broadened harness or repeated excluded controls. S5 stale intervention status corrected in plan/report/review. E200updates remains a real dependency on unresolved D parity, not idle independent work.

- D SwiGLU preflight: pinnedRuff0.9.9 check/formatPASS; actual native observer import, CLI/config and personal dry-runPASS. Initial CPU checks used an incorrect actor directory and omitted NEMO_RL_VENV_DIR; restored the existing /data/ycfeng/envs/nemo-mopd-actors paths before submission, no dependency change. Submitted exactlyone2H800 exp-1112-202910-954148 at20:29:10HKT; launcher74729 staysalive. Exact wrapper/args: submit_long_native_fused_swiglu.py and long-native-fused-swiglu-submit-args.json in tasktmp. Native capture retains first2MLP projection/activation/output rows to check the actual fusion arithmetic. No formalrun or sharedrecipe change.

- D SwiGLU comparison completed: exp-1112-202910-954148 Succeeded12:31:26Z/launcher74729exit0;160tokensMAE0.03172934FAIL. Nativefusionactivation matchesFP32formulaexactly,but first-layerMAE worsens0.0009077–0.0010146. Do not promotefusion. Terminal/layer/activation/parity retained; scalar/report values distinguished from unobservedvLLMactivation.
- D coreRoPErepair prepared through existingpatches.py and hf_overrides.rope_parameters.fp32_rope; worker/promptreplay request itbeforeenginecreation. Sixreal-factorychecksPASS;affected12CPUchecksPASS,39deselected. OriginalfactoryRED preserved. GPUcore/compiled/freshgeneration remainsUNVERIFIED; sourcefix uncommittedpendingactualGPUcheck.
- check-progress afterSwiGLU milestone: A/Q/B/CPASS;D/E/FTODO. D1–D7CLEAR: single-variablefusionmeasurement excludes hypothesis;nextpairedFP32head control advancesremainingD afterRoPErepair. S5plan previously saidfusionpending,now corrected. Noformalrun,noobjective/mask/tolerancechange. Nextpairednative/vLLMFP32heads withretainedcapture,thenactualcompiled/freshworkflow.

- D pairedFP32/coreRoPEprobes submitted afteractualCLI/configanddry-runPASS: nativeexp-1112-203938-713091(launcher69409),eagergenerationexp-1112-203938-677167(launcher90553),compiledgenerationexp-1112-203959-100754(launcher65065),2H800each. Keepall3launchersalive;formalrununsubmitted. NativeonlyenablesexistingFP32head;generationusescompleteoriginalropeparametersplusfp32_rope=trueandexistingFP32head. Eagertrunkcapturediagnosesremainingdifference;compiledresultsettlesactualdefaultmodeparityseparately. InitialgenerationCPUCLIusedbaseenvwithoutvLLM;actualexistingvLLMactorenvPASS(core-fp32-generation-cli-2.log),noinstallation.

## 2026-11-12 — core precision results recovered and frequency regression diagnosed

- All3 paired core probes Succeeded, launchers69409/90553/65065exit0; no live GPU launcher. Native/eager160-tokenMAE1.9457531214FAIL; native/compiled1.9473593901FAIL; eager/compiled0.0238099017. First QKV remains exact, RoPE cache/Q/K differ strongly. Archived terminal/parity/layer/MLP evidence in native-long-core-fp32-*.json.
- Root cause of enlarged core result: submitted rope_theta1000000 differs from original student/native10000. Actual captured cache verifies1000000 (MAE1.33e-6); native angle verifies10000 withinCPU/GPUFP32max0.0002441. Correct the affected overrides using original config; retain native capture and all existing precision/mask settings. This is an execution configuration error, not evidence that cache repair failed at the original frequency.
- CPU analysis position assertion was caused by native sequence-parallel attention projection/MLP output rows. Merged these by global position while keeping TP feature shards paired. analyze_core_fp32.py exit0/core-fp32-comparison-2.log; no GPU rerun needed. Frequency diagnostic originally asserted bitwiseCPU/GPUangles and failed; actual roundoff preserved in selected JSON.
- D pre-execution prediction: correcting only submitted frequency should restore exact first-layerQ/K and earlier endLPprecision regime. Both corrected2H800 personal dry-runsPASS; native rerun unnecessary. Eager andcompiled remain separate comparisons. Noformalrun, no tolerance change.

- check-progress恢复checkpoint完成：A/Q/B/CPASS,D/E/FTODO；S5旧paired状态已纠正，D1–D7CLEAR。修正frequency作业只提交一次：eagerexp-1112-205326-305203(launcher63645)，compiledexp-1112-205326-529652(launcher57389)，各2H800。保活两个launcher；不提交formalrun。

- D corrected-frequency eagerexp-1112-205326-305203 andcompiledexp-1112-205326-529652bothSucceeded; captures/LP/finished/runtime fetched throughboundedreadonlySSH. analyze_core_fp32_theta.py exit0. FirstlayerQ/Kmax0;cachemax5.96e-8;firstattentionrelativeRMSE0.000335–0.000630. EagerMAE0.02657703/p990.22286058/max0.51532555FAIL;compiledMAE0.03850816/p990.42302819/max0.61054474FAIL. Eager/compiledMAE0.02780242. Both existingFP32heads insufficient. Selectednative-long-core-fp32-theta-*.json retainsall160positions.
- Actualeager vLLMSwiGLU equalsBF16intermediateformula exactly;FP32formula differs. This supports notpromoting nativefusion. FirstMLPprojectionMAE~0.000274/firstMLPoutput~0.00058–0.00063;firstattentionprojection~0.000105–0.000140. Nativeattention notexactdespiteexactQ/K.
- D rankedpre-executionhypotheses: (1) native defaultFA2/auto backend versusvLLMFA3 remainingattentionarithmetic; forceexistingnativeflashbackend+FA3 shouldreducecoreattentionandendLPdifference. (2) residual/RMSNorm rounding; ifattentionmatchesbutMLPprojectionstilldiffers,inspectactualpostattentionnormandresidualinputs. (3) TPprojection/reduction differences; ifactivationmatchesbutprojectionfails,comparecorrespondingTPshards/reduction. NativeFA4beforeRoPErepairdidnotsettlecorrectedcombination. Nextsinglebackendcontrol2H800keepsnativeCP1/TP2/FP32head/originalfrequency/tokens/masks/unfusedSwiGLUandretainsfirsttwoattention/MLPcaptures. Noformalrun.

- VerifiedcoreRoPEfirstQ/Kmilestone：removedduplicate test-only--fp32-rope/use_fp32_rope afterfactoryGPUequivalence. Existingproductionroute isnowtheonly retained precision route. Broaderlongparity remainsFAIL; noformalrun.

- NativeFA3 preparation failedbeforeGPUsubmission: actualMCoreconfig/backendenvaccepted3,butnoindependentFA3package in nativeactorenv (`long-native-fa3-preflight.log`); noFA3RJob wascreated. ExistingFA4packageavailableinbothnativeandvLLM, pinnedvLLMinterfaceexplicitlysupportsSM9.x. ContinuebackendhypothesiswithpairedFA4capture,notaninstallationdetour. ExistingnativeFA4beforeFP32RoPEandBF16headdoesnotsettlethismatchedcorrectedarithmeticcase. Exactnewargsretainallpriorweights/tokens/masks/cache/headsettingsandchangeonlytheattentionbackendsetting.

- Corecachefinalregression12PASS/39deselected/11.50s;Ruff0.9.9check+format8filesPASS. check-progressafterthisverifiedsubstepretainsA/Q/B/CPASS,D/E/FTODO;S5updatedtocompletedoriginalfrequencyLPFAIL. NoD1–D7drift;nextmatchingbackendmeasurementclearsD. NativeFA4preparationPASS;vLLMFA4isvendored,firstincorrectindependentmodulepreflighterrorretainedandactualmodulepreflightinprogress.

- ProductionFP32RoPE/capture stepcommittedwithsignoffas d9b967b8;firstcommitattemptmettransientindex.lock,read-onlyinspectionfounditalreadyremovedandnoliveGitmutationprocess,oneunchangedretrySucceeded. No lock deletion orhistorychange. Existingrecipeeditsremainunstaged. Latestcheck-progressretainsDlongLPfrontier;committingverifiedfirstdivergencerepairdoesnotmarkDcomplete.
- BothFA4nativeandvendoredvLLMconfig/importpreflightsPASSandpersonal2GPUdry-runsPASS. Submittedonceat21:03:11HKT: nativeexp-1112-210311-532182(launcher61605), generationexp-1112-210311-662995(launcher94116),2H800each. Keepbothlaunchersalive. Exactwrappers/args/logs:long-native-fp32-head-fa4 andlong-generation-core-fp32-fa4 in tasktmp. NoFA3job,noformal200-update submission.

- NativeFA4 exp-1112-210311-532182 Failed13:05:00Z;childexit2at13:04:57Z precedeslaunchercleanup13:05:01.782Z. Exactincident-boundRCAobject/Events/historical/logs/DeepScanqueriescomplete;bothall-operationauditqueries18stimeout,actor/reason/requesttimeUNKNOWN. Platformretainschildargv/exit2;exactCPUentrypointreproducesargparseinvalidintwhenappendedoverridefollowsnargs+context-parallel-sizes. Moveoverridesbefore--dump/--context-parallel-sizes;CPUoriginalRED(exit2)andcorrectedGREEN(NATIVE_FA4_CLI_CONFIG_PASS,exit0)observed. ReportprimaryparsertextasCPUobservation;platforminitiatingtextunavailable. Noengineorprecisionchange.
- vLLMFA4 exp-1112-210311-662995 Succeeded13:07:06Zsnapshot;retaincompletedcapture. Submittedonlyaffectednative2GPUrerunusingcorrectedargv wrapper submit_long_native_fp32_head_fa4_2.py (launcher5080),newcloudprefixlong-native-fp32-head-layers-fa4-2. Keepalive;do notduplicatevLLMorformalrun.

- NativecorrectedFA4jobidentity exp-1112-210903-042205 created21:09:03HKT,launcher5080alive. CompletedvLLMFA4capturesretrievedatlong-generation-core-fp32-layers-theta-fa4-inputs. HistoricalnativeFA2vsnewvLLMFA4MAE0.02268349/max0.50479650stillFAIL;thisisnotmatchedFA4result. NativeFA4 comparisonawaited.


## 2026-11-12 — matched FA4 result and remaining normalization boundary

- Corrected native FA4 exp-1112-210903-042205 Succeeded13:13:51Z; launcher5080/monitor59626 exit0. Actual2H800/Python3.13.14/Torch2.13.0+cu130/CUDA13.0, personal creator/codesign/localNFS100.96.128.195 verified. Retrieved exact native captures, replay LP, runtime-0.json and finished.json; paired completed vLLMFA4 capture reused without rerun.
- Matched FA4/CP1/TP2/FP32head/FP32RoPE at original theta10000:160-token MAE0.01905496458, signedmean0.00493660311, p990.09266241074, max0.11156654358. Overall mean passes0.02;3/5samplemeans exceed0.02 (0.02007044/0.02000784/0.02069354), so group/tail investigation remains outstanding and D is not promoted. Firstlayer embedding/QKV/Q/K/coreattention/attentionprojection all exact. Two firstlayer MLPs/layeroutputs exact; three have sparse projection errors MAE0.8e-6–1.7e-6/max0.0078125. Thus matched attention removes its observed firstlayer remainder; earliest retained difference is after attentionprojection and beforeMLPprojection. Selected evidence native-long-matched-fa4-*.json.
- Analyze command: OMP/MKL/OPENBLAS4, PYTHONPATH currentrepo, existing CUDAcompatLD_LIBRARY_PATH, baseUVenv, timeout60s, uv--no-sync python withRLIMIT_AS30GiB running analyze_matched_fa4.py. systemd-run--user and systemscope rejected because host lacks a bus; preserved logs1/2 and bounded successful log3. Only32positions/sample read.
- Ranked pre-execution hypotheses: (1) residual/RMSNorm arithmetic: matching attentionprojection with sparse MLPprojection difference predicts differing postattentionnorm input/output; capture real nativeTE apply_normalization and vLLMnorm boundaries without changing returned tensors. (2) TP GEMM/reduction: identical actualnorm outputs with differingMLPprojection would support projection arithmetic. (3) compiledkernel arithmetic: compare matchedFA4 compiledprompt replay against retainednativeFA4 and eagervLLM, keeping weights/tokens/head/cache/backend unchanged. Formal200updates notsubmitted.
- CompiledFA4 CLI boundary and personal2GPUdry-run passed. Submitted only compiled variant exp-1112-211826-090194, launcher34455; keep alive. Exact wrapper submit_long_generation_core_fp32_compiled_fa4.py/argsJSON in tasktmp. Firsttwo norm capture additions are diagnostic-only and pending nativeimport/boundary/GPUchecks; do not use these workers forformaltraining.

- Norm observation nativeactualimport, exactCLI-to-init_ray, generationCLI-to-engine, inputmutation/outputselection boundary and personaldry-runsPASS. Ruff0.9.9check/formatPASS via existingcachedbinary; firsttoolresolution lacked cachedindex and exitedbeforechecks, no dependency install. Submitted once each nativeexp-1112-212103-283781(launcher15358) andgenerationexp-1112-212103-266417(launcher9966),2H800each. Keeplaunchersalive. Captures only add observed firsttwo RMSNorm inputs/outputs; no native/vLLM computation/returned tensors changed. Formalrun stillunsubmitted.
- check-progress after matchedFA4milestone: A/Q/B/CPASS,D/E/FTODO. D1–D7CLEAR: backendhypothesis advancedfirstattention toexact; remainingnorm/projection hypotheses are recorded beforeexecution, compiledvariant advances independentmodeacceptance. S5plan/review staleFA4status updated; noS1–S4/S6defect. Nextcollectactualnormboundary/compiledresult, applycausefix onlywhen measured, thenfreshrollout/refit andformal200updates.

## 2026-11-12 — final runtime preflight and normalization reconciliation

- RecoveredCPU sessions21274/53704: all3FINAL_RUNTIME_CONFIG_PASS markers and all3personal dry-runs completedexit0. Retrievedcompletedlaunchers34455/15358/9966andmonitors98749/99203exit0. No repeatedGPUsubmission. Compiled/normplatformterminalrecordsandanalysisJSONarchived under evidence/native-long-matched-fa4-*.
- Norm observers preserveall160LPexactly. Actualfirst-layernormalizedinput/roundedresidualexact;3samples have1–2BF16outputelements differing. Firstremainingnormboundary is observed; its causal propagationtofinaltail is inferred. CompiledFA4 MAE0.03365744FAIL. No newTEbackend/dependency repair; proceedwithalreadyexisting eagerFA4/FP32RoPE/FP32heads/TP2CP1recipe.
- Final independent checks:24H800 selfteacher5updates/32fresh/8192 withdiagnosticNativeForwardWorker;6H8003x2selfteacher1actual65536-tokenupdate/5samples usingordinaryentrypoint;56H800fiveoriginalteachers5updates/32fresh/8192 withordinarynative save/resume. CPUresolvedconfigvalidationanddry-runPASS. The24GPUobserver isexcludedfromformalrun;32+auxiliaryanalysesremain non-blocking. Nextsubmit these86GPUs onceeachandkeepalllaunchersalive. Formal200updatesnot submitted.
- check-progress invokedafterrecoverymilestone: A/Q/B/CPASS,D/E/FTODO. D1–D7CLEAR: capturednormresultssettlethefirstremainingboundary;compiledfailureselectseager;threeindependentactualworkflowchecks advanceD. S5oldpendingnorm/compiledwordingcorrectedinplan/progress/report/review. No acceptance/scopemodification. Mainline: finalworkflowchecks -> formal200successfulupdates/256fresh/65536 -> finalparity/finite/report.

- Finalworkflowchecks submitted once throughpersonal localRJobBackend at21:34HKT:24GPU exp-1112-213427-594083 launcher61127;6GPU exp-1112-213433-270314 launcher67499;56GPU exp-1112-213439-246833 launcher73755. Keepall3launchersalive. Exactscripts/args/logs final-* under tasktmp. Noformal200-update job yet.

- Dfinal-longtraining exp-1112-213433-270314 Failed before native training:wait_cluster rejectsduplicateNodeManagerAddress; actualreplicas0/2sharegpu-h800-0320. BusinessValueError precedeslauncher cleanup21:37:11HKT;6GPUchecks settle no training criterion. ExactRCAcollector launched(52936),rawfinal-long-training-rca/. Correctonlyresource/layout:3x8GPUs forcesdistinctfullnodes;batch8matchesDP4,source5original65536casesretainedandcycled;sameobjective/precision/tolerance.24GPUaffectedrerunprepared,beforeGPUsubmissionCPUvalidationrequired. Other24/56GPUjobscontinue.
- Formal200/256/65536recipeprepared butnot submitted. Full native optimizer/data-plane checkpointing and7200/9000deadlines restored,ordinaryresumeentrypoint,48hourouterdeadline,allcheckpoints retained (keep_top_k/ft_keep_latest_k=null). FirstCPUpreflight failedmissingHydra+prefix foroptionalft_keep_latest_k;correctedactualschemausage. Secondcheck failedtestscript dictaccessofGRPOConfig;changedtoactualBaseModelattributes,withnoproductbehaviorchange.

- CorrectedlongCP1/DP4config/dry-runPASS; submittedonce exp-1112-213949-129130 at21:39:49HKT,launcher70958,monitor27627. Existing24/56GPUsremainalive; active requestedallocation104GPUs. Current56GPUreplicasareunscheduled(coschedulingGPUgap24),notafailednativeinit; retainqueue/launcher. Formalconfigpreflight3/dry-run2PASSwith200/256/65536restored,nosubmission. Runtime/fullCLI/evidenceandplacementRCArecordedinfinal_workflowreport.
- ObservationcoveragepredictionfromNativeForwardWorker:actualtraining/teacher_same_forwardwrittenonlyglobalrank0;atDP4maycoveronlyoneshard. Analyzeobservedsubsethonestlyandretainallglobalindependentscorercomparisons;missingobservationsarenotnumericalfailuresandcannotstophealthy56GPUtraining.
- Verifiednormalizationcapture/evidencestepcommittedwithsignoffas9826c435;recipesremainunstageduntilselectedruntimeactualverification.


## 2026-11-12 — TQ singleton serialization failure recovered

- D self-observed exp-1112-213427-594083 Failed13:43:02Z after fresh rollouts, before any observed successful optimizer update. Actual simple storage put reported `self.stride(-1) must be 1 to view Int as Byte ... got 0` for all three units. Launcher61127 and RCA collector1394 exit1 recovered. Object/owner/creator verified; exact live/historical Events, incident logs and existing DeepScan queries completed; both all-operation audits timed out. Manual stop actor/reason/request time UNKNOWN. Selected evidence: evidence/final-self-observed-storage-rca.json; raw final-self-observed-rca/. Shutdown logs follow the business error.
- D CPU RED: OMP/MKL/OPENBLAS4, timeout60s, Python3.13.14, Torch2.13.0+cu130, RLIMIT_AS30GiB, real MsgpackEncoder.encode(TensorDict(input_lengths=int32 singleton view with stride0)). Command `/data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/repro_tq_zero_stride.py` deterministically produces the exact RuntimeError in4.25s; tq-zero-stride-original-red.log retained. A scalar and ordinary singleton both encode correctly; cloning the zero-stride singleton with memory_format=contiguous_format yields stride1. Exact failed field/producer still unobserved.
- Ranked hypotheses recorded before further probes: (1) contiguous singleton broadcast view reaches serializer unchanged; clone at byte-encoding boundary should preserve values and fix it. (2) jagged unbind metadata encoder has the same byte-view defect; corresponding zero-stride row should also reproduce. (3) SSD encoding independently shares the restriction; only relevant if SSD is enabled. Actual runs have SSD disabled, so that optional path does not block this run. No site-packages edits or exception fallback.
- Existing final-long-training-2 exp-1112-213949-129130 Running13:46:04Z and final-five-teacher-resume exp-1112-213439-246833 Running13:46:37Z. Launchers70958/73755 and monitors27627/19889 remain alive. No duplicate submission; formal200 remains unsubmitted.
- check-progress resumed milestone: A/Q/B/C PASS4; D/E/F TODO3. D1–D7 CLEAR: deterministic actual encoder RED directly advances D; two independent workflow jobs remain active; no backend installation or new objective. S5 HIT: plan/review still describe final self pending and jobs awaiting placement. BranchB correction: record actual failed TQ path and both Running states; no agreed scope/tolerance changes. Next D root-cause serialization repair + affected regression, in parallel actual long/resume outputs; then formal200/256/65536.

- D TQ root-mechanism fix: existing adapter installs wrappers for both ordinary tensor and nested-row metadata byte encoders. Contiguous tensors whose flattened stride is not1 receive a contiguous-format clone; ordinary unit-stride inputs stay zero-copy. No input values/dtypes/shapes changed, no site-packages edits, no exception suppression. Persistent real-adapter/storage-selector regression created before the fix. Removing only these wrappers in the CPU test process reproduces9FAIL/1PASS; corrected regression10PASS/5.68s; affected scalar-schema/wire-object checks25PASS/12.88s. Ruff0.9.9 check/format2filesPASS. Raw upstream reproduction remains the retained RED; importing the adapter supplies the production repair. Original GPU scenario still pending.
- D actual long-job reconciliation: exp-1112-213949-129130 Succeeded13:49:57Z, launcher70958/monitor27627exit0, but logs/exp_001 contain no training metrics and rollout trace contains only4setup/sync/restore events. Native source dataloader uses batch_size8/drop_last=True for a5row source. Do not claim a65536 training update. Selected evidence: evidence/final-long-training-no-update.json; raw runtime/command/trace/TB retained in final-long-training-2-artifacts/.
- D affected long correction uses existing dataset repeat=2 (10rows), shuffle=false and split_validation_size0.0. Native CPU StatefulDataLoader observes1batchof8, all5original inputs plus first3repeated; sourcefileunchanged. No alternate dataset/loader. Existing temporary long-context-training-batch8.jsonl is unused and outside deliverables. Configvalidation2PASS and both personaldry-runsPASS. Formaltraining remains the original five training pools.
- D submitted once at21:55HKT: self-observed-2 exp-1112-215538-183707,24H800,launcher86892; long-training-3 exp-1112-215539-050439,24H800,launcher19420. Keep both alive. Existing56H800 exp-1112-213439-246833 Running13:55:35Z withlauncher73755/monitor19889 alive; noduplicate. Three nonterminal requests total104GPUs. Formal200 remains unsubmitted.


## 2026-11-12 — resumed state and first final-five-teacher result

- Original zero-stride CPU reproduction after importing the actual adapter completed exit0 with ZERO_STRIDE_WIRE_PASS; session99208 collected. Exact script/log: /data/ycfeng/tmp/nemo-mopd-20261112/{repro_tq_zero_stride.py,tq-zero-stride-original-green.log}. GPU original scenario remains pending; no broader acceptance claim.
- Three launchers73755/86892/19420 and monitors19889/90724 verified alive. Latest inspected2026-11-12T14:00:53Z: five-teacher and self-observed Running; long-training-3 Pending with request24/remaining17H800. Preserve current queue; no resubmission. Personal creator/localNFS100.96.128.195 and codesign retained.
- Five-teacher step1:32valid samples/4096response tokens; grad_norm1.7595814466, trainer_version1/policy_lag0. Native checkpoint step1 consumed_samples32/current_step1/sampler_dispatch_index1/next_nemo_gym_task_index64. Resume log observes model reload and native TQ restore, but subsequent optimizer updates are not yet verified.
- Independent scorer minus vLLM generation on identical tokens/versions: MAE0.01001796739, signedmean-0.00041293, p990.09298755/max0.27509165. All five group means<=0.02, no token/mask/version issues. Exact per-sample tail and group counts in evidence/final-five-teacher-step1-parity.json; this is not an actual-training-forward measurement.
- Read-only artifact collection uses existing exact paths and bounded SSH; temporary collection script initially had shell/string quoting errors and then incorrectly expected an idx field in the source rows. Corrected to source line index as already verified for NemoGymDataset. These were local auxiliary errors and never propagated to or stopped training. Stop extending this collector once the required artifacts are available.


## 2026-11-12 — final short native workflows completed

- Five-teacher exp-1112-213439-246833 Succeeded14:05:25Z,7replicas; self exp-1112-215538-183707 Succeeded14:05:19Z,3replicas. Both local launchers73755/86892 exit0 collected. Long exp-1112-215539-050439 now Running14:07:19Z; launcher19420/monitor90724 remain alive. No extra submission.
- Five-teacher:5successful updates,0skipped,160samples/20480response tokens. Loss[0.114023,0.140388,0.174867,0.160583,0.160291]; gradnorm[1.75958,2.87954,2.66155,2.86080,2.95542]. Native optimizer returns the norm before clipping; clipping1.0 remains active. FP32 entropy/cross-entropy decomposition maximum2.3841858e-6 is a measured per-position arithmetic statistic; full-loss/reference checks retain their existing tolerances. No nonfinite objective/grad observed.
- Native resume contents: optimizer param-group step1→5; scheduler num_steps32→160 with unchanged decay horizon160. Native step5 counters consumed160/current_step5/trainer_version5/validtokens20480/sampler_dispatch_index5/next_gym_task224. Checkpoint finite scan covers all1801tensorchunks/8files,1777088000model elements+5331526656optimizer elements,allfinite. Read-only scan/inspect exact scripts and logs final-* under tasktmp; remoteTorch2.5.1, timeout60s/file,15GiB/file,2parallel reads.
- Five-teacher same-version scorer/vLLM MAE0.0096118060,p990.08671216,max0.33361328; every step/group mean<=0.02. Four of160sample means exceed0.02, documented in exact tail table. Saved/reloaded model output comparison is represented by continued scorer/vLLM same-version checks, not a bitwise saved-weight replay.
- TQ original GPU scenario now passes5successful selfteacherupdates/0skipped. Actualtraining/scorer is exact on90observed samples/11520response tokens; actualtraining/vLLMMAE0.0086732236, everygroup<=0.02. Remaining70samples have no rank0training observation; this is an explicit coverage limit. All160global scorer/vLLM samples independently passoverallMAE0.0096954580;8samplemeans exceed0.02 and remain visible. Initialselfloss-6.6883e-10/gradnorm1.0841e-5; laterfrozen-teacherloss0.000153–0.000206 andgradnorm0.0348–0.0557,allfinite.
- Fullteacher reconstruction covers160calls/1280shards(all8ranks),logitMAE4.1325452e-6/max1.2207031e-4, sampledLPMAE2.9453349e-6/max6.29425e-5. Distribution/loss/gradient results in evidence/final-self-observed-subset-parity.json. Rawunnormalizedall-positiongradienterror1.6964563e-5 exceeds1e-5,retainedFAIL; native response-mask/token-normalization yields derivedupperbound4.14174e-9<=1e-5. Referencegradient0 meansrelativeerrorundefined; no arbitrary scale, objective or tolerance change. This follows the previously inspected native normalization proof, not an observed parameter-gradient equality.
- Formal200updates remains unsubmitted until the actual65536trainingupdate passes. TQcode step is nowGPUverified and readyfor signedscopedcommit; selectedCP1recipe still waitsforlongtrainingacceptance.

- TQproduction/test stepcommittedwithsignoffas e8fcef02. Firstcommitattempt met transientindex.lock; inspectionfoundlockalreadyabsentandonlyour2filesstaged; unchangedretrySucceeded. No lockfiledelete/historyrewrite.
- Native selectedweightdelta initialstudent→step5:144896norm/biaselementsacross113chunks,422elementschangedin85chunks,maxabsdelta7.6293945e-6,allfinite. Selectedevidence final-five-teacher-weight-delta.json. FirstreadonlyremoteanalysiscouldnotunpicklePython3.13PathmetadataonPython3.10(pathlib._local); metadataonly78938bytesfetchedandparsedonmountedPython3.13,thenexacttensorbytechunksreadremotelywithTorch2.5.1. No runtimepackageorproductionchange.


## 2026-11-12 — recovered final long-training failure

- Exact job exp-1112-215539-050439 is Failed at14:12:49Z. Launcher19420 exit1 and monitor90724 exit0 collected; terminal inspector25052 exit0 collected. No new submission or stop. Formal200 remains unsubmitted.
- Head log records Megatron allocation OOM at22:12:04HKT: requested19914555392bytes, available17435197440bytes, total85029158912bytes. This precedes vLLM SIGTERM22:12:10HKT; the shutdown signal is cleanup evidence. Exact operation and business traceback remain under investigation.
- Five-teacher and selfteacher short run outcomes remain verified as recorded. Selected original-job logs, Ray worker traces, TensorBoard and partial native dump will establish the actual failure stage before any correction.
- First record-update invocation used tasktmp as cwd and failed before modifying any file; this update uses the absolute task path.


## 2026-11-12 — long backward RCA and TP4 correction

- Complete final head platform log now returns the initiating RayTaskError(OutOfMemoryError): native student train_microbatch_presharded -> reverse-KL backward -> model_utils.py:2677 empty_like(student_logits). Request18.55GiB; GPU4free16.36GiB, allocated44.12GiB/reserved-unallocated14.10GiB. This is a full-sequence gradient allocation, not teacher payload or a generation SIGTERM. Raw exp-1112-215539-050439-cfcd2084-final-full.log retained in tasktmp; selected traceback and RCA are in evidence/final-long-training-3-*.
- Exact-object/live+historicalEvents/incidentlogs/existingDeepScan succeeded. Bothall-operationaudit reads timedout15s; no automaticretry. Manualstopactor/reason/requesttime UNKNOWN. Headcontainerexit1 at14:12:21Z precedes platformFailed14:12:24Z. TBhas0successfulupdates.
- Existingparallelism correction: trainerTP4/CP1/DP2 on8GPUs, teachersandgenerationretainTP2. Full151936vocabulary,FP32heads,reverseKL,responsemask/token-normalization,originalfive65536inputswithnativerepeat2andbatch8areunchanged. Nativegradientperrank9957277696bytes versus19914555392 atTP2. No coregradient/scaling/fallbackchange.
- TightCPUallocation fixture reaches the actual empty_like call using meta logits and injects only the incident per-allocation budget. TP2REDexit1 at19914555392bytes; TP4budgetPASSexit0 at9957277696bytes. This verifies allocation shape, not aggregateGPUpeak,numericsortraining. Exact repro_full_kl_grad_allocation.py and full-kl-grad-allocation-{tp2-red,tp4-budget}.log in tasktmp. NativeTP4schema andpersonaldry-runPASS. Actualoriginalbatch8/65536 rerun is required.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 1｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A / Q / B / C | PASS | Source, acceptance, runtime and reference preparation recorded |
| D | FAILED | Short workflows verified; 65536 backward OOM remains under GPU verification |
| E | TODO | Formal200updates and independent saved-checkpoint replay outstanding |
| F | TODO | Final reconciliation and archive require E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** e8fcef02 short-workflow milestone to resumed RCA｜**当前前沿：** D long-training and independent saved-checkpoint replay preparation

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Bounded original-job collection stopped at actual traceback |
| D2 细节反复重试 | CLEAR | Placement, empty batch and OOM are distinct measured failures |
| D3 支线问题修补 | CLEAR | Work addresses original65536acceptance |
| D4 可并行任务停滞 | CLEAR | Saved-checkpoint replay preparation resumes while GPUlongcheck runs |
| D5 超范围镀金 | CLEAR | ExistingTPsetting only, no new coreinterface |
| D6 分析空转 | CLEAR | Traceback produced a concreteTP4verification |
| D7 路线违背 | CLEAR | Formalrunstillwaitsforactuallongtraining acceptance |

**文档检查：** S5 plan.md/report Running contradicted the terminalFailed; corrected directly. S1/S2/S3/S4/S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：先修正状态文档，随后继续。

**文档更新（先执行）：**
1. plan.md/report status: record exactFailedjob,0verifiedupdates and RCAinprogress（可直接执行，已完成）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：原batch8/65536在TP4实际训练 → successfulupdate、finiteloss/grad、same-tokenparity。
2. 并行：保存权重的独立replay准备 ‖ 正式TP4config准备；正式提交仍等待D。

## 4. 最值得关注的 3 点

1. 原失败已定位到studentfull-KLbackward第二张全序列gradient的分配。
2. TP4只验证了allocation大小和CPU配置；GPU验收仍需真实update。
3. 正式200update和独立保存输出replay仍未完成。

- TP4original65536training submitted once as exp-1112-222130-343023,24H800,launcher79179. NativeCPUvalidation32985exit0,allocationRED9435exit1 and budgetGREEN1379exit0,personaldry-run52363exit0. Keep launcher alive. FormalTP4 and savedstep1/step2tokensreplay preparation proceed independently.

- Independent native saved-step1 replay submitted once as exp-1112-222427-715476,4H800,launcher89330. ActualCPUentrypoint reaches init_ray after parsing32savedstep2samples; everymetadataweight_version=1. InitialimportmissingprojectPYTHONPATH fixed to the existingworkerprojectpath; thenargparsenargs+requiredcontext-parallel-sizeslastafterallHydraoverrides. FinalactualCLIpreflightandpersonaldry-runPASSbeforeGPUsubmission. No productparserchange.
- LongTP4jobRunning14:24:03Z; Ray confirms3nodes/24GPUs. Creatorpersonal/codesign/localNFS100.96.128.195verified. FormalTP4nativeCPU200/256/65536checkanddry-runPASS; stillunsubmitted.

## 2026-11-12 — TP4 long workflow complete, consistency rejected

- Completed exp-1112-222130-343023: native execution and platform Succeeded, launcher79179/monitor3799 exit0. Actual8rows each65504prompt+32response=65536tokens.1successfulupdate,0skipped,256validresponse tokens; loss0.0050317468 andpreclipgradnorm70.8461304 arefinite. scorer/vLLMMAE0.0685276996FAIL0.02,p950.15791428,p990.84881840,max3.0723002. All8samplemeansFAIL; original5longcasesareallmath selfteacher. TP4 is not promoted. Evidence final-long-training-4-{parity,training-metrics}.json.
- Savedstep1 replay exp-1112-222427-715476 Failed during validate_model_paths: step1 was deleted by native retention after step5 save(save_period5/keep_top_k3/noftretention); actual checkpointforward never ran. Native retention code and cloud path absence verified. Containerexit1 14:27:52Z -> Failed14:27:54Z -> launchercleanup14:27:58Z. RCA6primaryqueries succeeded,2allaudits timeout15s;manualstopactor/reason/requesttimeUNKNOWN. Formalretentionnullalreadypreventsdeletion. Exact same-weight savedoutputreplay is E acceptance using retainedformalstep1 andversion1/step2tokens; do not rerun missingpath.
- Pre-execution predictions for D: (1) TP2/CP1 replay of identical TP4longtokens shouldreduceMAEto<=0.02 ifTP4numerics dominate; (2) TP2/CP2 controls CPshape independently using same tokens; (3) nativeTP2/PP2/CP1 preserves TP2 projection arithmetic and halves laststage decoderactivation/param burden, potentiallyclearing18.55GiBbackwardallocation. ActualPP2support inspectedinexistingMegatronpipeline+fullteacherheadsetup; CPUconfig/nativeboundaryanddryrunrequiredbeforeGPU. IfTP2replaystillFAIL, investigatefreshgeneration/batch/normdifference, retainingceilings. Noformalrunsubmitted.

- CPU preparation passed for the actual TP2 replay CLI (same masks and recorded tokens, stop at init_ray), MasterConfig/SingleController validation, and native Megatron training checks for CP1/CP2 and PP2. First local check accessed dict-valued `native.data` as an attribute; corrected only the preflight. Successful command uses `uv run --no-sync python preflight_final_tp2_parallel.py` in the pinned Megatron actor environment; logs `final-tp2-parallel-preflight-2.log`. Both personal dry-runs exited 0.
- Submitted once at 22:36 HKT: TP2 replay `exp-1112-223646-775620`, 8 H800, launcher6526; TP2/PP2 long workflow `exp-1112-223647-213583`, 24 H800, launcher63849. Keep both launchers alive. No formal 200-update job exists yet.

- TP4 terminal verification: all three exact replicas Succeeded, container exit0/restart0; creator and current-machine NFS source verified. Native GPU monitor observed trainer memory maxima 43.79–44.91 in `ray/node.0.gpu.*.mem_gb`, teacher 28.05, generation 69.34–69.43. Seven samples at the configured 10-second interval do not guarantee transient peak capture. See evidence/final-long-training-4-observed-memory.json.
- Formal TP2/PP2 candidate configuration is prepared independently for the unchanged 200/256/65536 contract and full checkpoint retention. It remains unsubmitted until D passes.

- Required PP2 checkpoint replay preparation found a reachable allocation bug: `replay_logprobs.py` computed TP×CP workers and omitted PP. The actual CLI boundary for TP2/PP2 produced RED (`requires4workers; actual[2]`); multiplying the existing PP size now gives GREEN `[4]`. Added TP/PP widths to the existing replay result so the tested mesh is explicit. Ruff0.9.9 lint/format pass. No new flag, trainer, loss, mask or tolerance. GPU verification is the same recorded 65,536-token batch in a separate 4-H800 PP2 replay; commit waits for its actual result. Logs: pp2-replay-allocation-{red,green}.log; script: tasktmp/check_replay_pp2_allocation.py.

- Submitted PP2 replay once: `exp-1112-224238-102778`,4H800,launcher74702. Both prior jobs remain Running; no duplicate submission. PP2 replay uses original TP4 recorded tokens at weight_version0 and initial weights, not the deleted step1.

## 2026-11-12 — TP2/PP2 long workflow and identical-token reconciliation

- Recovered existing launchers without resubmission. exp-1112-223646-775620 Failed after completing CP1 replay; CP2 reaches TE attention and raises no available dot-product backend. Exact six RCA source queries completed, both all-operation audits timed out15s; initiating manual-stop actor/reason/time UNKNOWN. CP1 result is retained, not discarded with the later failure.
- TP2/PP1/CP1 and TP2/PP2/CP1 replay response outputs are identical on all8original65536 sequences (256response tokens). Both scorer/vLLM MAE0.05935666217FAIL, below TP4's0.06852769961 but still above0.02. PP2 allocation fix exercised real4-worker scoring; platform terminal verification still being reconciled.
- Native TP2/PP2/CP1 exp-1112-223647-213583 recorded1successfulupdate/0skipped,8validseqs/256validtokens,loss-8.178703936e-9 and preclippinggradnorm0.00336871529, finite. Its scorer outputs/error reproduce the same TP2 result. TP4's70.84613gradnorm is associated with its different studentTP4/teacherTP2 arithmetic; causal attribution requires the forthcoming layer comparisons.
- Hypotheses before next probes: H1 vLLM incremental decode differs from fixed-token prefill; predict vLLM prompt replay differs from recorded generation and narrows native error. H2 native/vLLM norm/projection kernels differ even for matched prefill; predict first hidden-state divergence before attention despite exact inputs. H3 long-position attention differs with batch/KV-cache layout; predict input/QKV/RoPE boundaries agree until attention output. Use existing two2GPU layer entrypoints on originalTP4tokens, unchanged weights/precision/0.02. These advance D only; formal200 remains unsubmitted.

- Exact terminalobjects: TP2replayFailed/container1; PP2trainingSucceeded/all3containers0; PP2replaySucceeded/container0. Native launchers63849/74702exit0,failed6526exit1. PP2replayallocation/scoring fix signedcommitaf17b989. check-progress: A/Q/B/CPASS,D/E/FTODO; D1–D7CLEAR; S5oldTP4next-action corrected. Nativeprefill2GPUsubmittedonceexp-1112-225028-947427,launcher62305; keepalive. GenerationCPUpreflight encountered root-ownedpatch_lockPermissionError; verify installedRoPEpatch marker and mockonlyfilepatchoperation atCPUboundary, withoutchangingworker path or dependencies.

- GenerationprefillCPU boundary secondcheckPASS after confirmingtheexistinginstalledRoPEpatch; workermodificationpathunchanged. Bothpersonal dry-runsexit0. Generationprefill2GPUsubmittedonceexp-1112-225203-504770,launcher44738; nativeprefill2GPUexp-1112-225028-947427launcher62305 remainsalive. Exactsubmitargs savedinevidence/final-long-{generation,native}-prefill-submit-args.json. Noformalrun yet.

- Nativeprefillexp-1112-225028-947427 Failed at22:53:41HKT with actualRayTaskError(KeyError)MOPD_LAYER_OUTPUT_DIR in LayerOutputWorker.get_logprobs after its scorerforward. Exactincidentlogs confirmthemissingexistingCLIenvsetting; no modelattention/backwardfailure. Six RCA readscompleted; audits unavailable15stimeout. Correctonlyrequiredconfigenvdestination to ${logger.log_dir}/layer-outputs; strengthenCPUpreflightatthePolicyboundarytocheckresolvedactualactorenv. Originalbusinessfailureevidenceretained; existinggenerationprobecontinues.
- AdditionalH3 controlpreparation: installedvLLMflash_attn.py:587–600 usesadaptiveSplitKV by default and forcesnum_splits1 for VLLM_BATCH_INVARIANT; nativefullprefillusesunpagedselfattention. Ifshape-dependentattentionreductiondriveserror, theexistingbatchinvariantmode shouldreduceprefill/decode/nativeerror. Themodealsochangeslinearmatmul/RMSNorm; attribute onlytheoverallconfiguredmodeuntillayercontrolsettleswhichboundary. No numericalthreshold change.
- CPUactualengineboundaryRED proved replay_generation_logprobs.py ignored existingvllm_cfg.env_vars. Addedtheexistingmapto os.environ beforevLLMimport/engine; GREEN requiredbeforeGPUvariant. Thisdiagnosticconfigfixdoesnotmodifytheformalworkerimplementation, whichalreadyhonorsthemap.

- Same8tokens vLLMprefill completed: nativeTP2scorer/vLLMprefill MAE0.01883321955overallPASS(max0.12449;3/8samplemeans>0.02). vLLMprefill/recordedactualgeneration MAE0.05840253056FAIL(max1.10977), matchingthelargeobservednative/generationgap. Thisnarrowsprimarydifference to vLLMprefillversus generation, ratherthanPP layout; nofilteringorrawlogprobscalechange.
- Nativecapturecorrectedconfig/actualPolicyenvboundaryPASSandpersonal dry-runexit0; submittedonce2GPUexp-1112-225719-755352launcher21542. vLLMexistingbatchinvariant config/environmentboundaryGREEN andruff0.9.9check/formatPASS; submittedonce2GPUexp-1112-225732-223406launcher62934. Keeplaunchersalive; monitorscopeexactjobs. Nativeprefillreplaywithcapture mustreproducetheprior0.05935666resultbeforeusinglayerobservations. Formal200unsubmitted.

- Nativecapturesecondprobeexp-1112-225719-755352Failedbeforeforward: actualargparseconsumedtrailingHydraoverrideascontext-parallelint. Launcher21542exit1. Causeis tasklaunchargumentorder,notbackendnumerics. OriginalCPUpreflightusedarebuiltvectorandmissedthisfault; correctedpreflightnowusesactualsubmissiontailthroughPolicyboundary. BothCLIenvandargorderexplicitlyvalidated,thenonce-onlynative2GPUexp-1112-230040-904461submittedlauncher9781. No furthercapturetoolchangesunlessanewobservedfailure requiresit.
- ActualfreshlongbatchinvariantcontrolCPU nativeconfigPASS andpersonaldry-runPASS,unchanged8/65536/TP2PP2/selfteacher/fullvocabulary. Submittedonce24H800exp-1112-225944-064229launcher51331. Existinginvariantprompt2GPUexp-1112-225732-223406launcher62934continues; noformal200job yet.
- check-progress keyprefillmilestone: A/Q/B/CPASS,D/E/FTODO. D1HIT: thetwoauxiliarylaunchfailuresconsumedGPUtimewhiletheworkingreplayalreadyestablishedprefill/decodegap. Correctionapplied: useactualCLIvectorbeforeanyfurthercaptureallocation,andadvancefreshnativeinvariantcontrolinparallel. D2–D7CLEAR; S5oldplanfrontierupdatedtoactualfreshinvariantcheck. No scope or acceptancechange.

## 2026-11-12 — resumed invariant result and complete layer capture

- `exp-1112-225732-223406` and `exp-1112-230040-904461` are Succeeded, each replica/container exit0/restart0. Launcher62934exit0; native launcher9781 reached successful terminal. Creator i-fengyicheng and local source verified. The 24H800 `exp-1112-225944-064229` is Starting because gang placement is unschedulable (member gap2); no business artifacts yet. Keep launcher51331 and monitor69544 alive, no resubmission.
- Existing batch invariance is rejected for promotion: original8sequences/256responsepositions, nativeTP2 versus invariantprefill MAE0.24694790764, p951.2586863, max6.8822734; invariantprefill versus originalgeneration MAE0.25645986102, max6.7738152. Unchanged ceiling0.02FAIL. This is a measured control outcome, not a repair. Script environment fix is validated: configured environment reached actual engine and altered executed arithmetic, despite rejecting this configuration.
- Native observer exactly reproduces prior TP2/PP1 and PP2 scorer outputs. Captures cover all8samples/twoTP ranks; first-layer QKV, post-RoPEQ/K and attention are exact. Three samples first diverge in post-attention RMSNorm; two others first diverge in second-layer input RMSNorm. Same rounded residual inputs agree. Sparse BF16 normalization differences propagate; no causal full-model intervention yet. Raw selected32positions only, not fullprompt coverage. Layer comparisons use actual Python3.13.14 through uv--no-sync, OMP/MKL/OPENBLAS4, timeout60s.

- Next D hypothesis before execution: native vLLM `SamplingParams.__post_init__` sets skip_reading_prefix_cache=true whenever prompt_logprobs is requested; ordinary generation uses prefix caching. The long inputs share prefixes, and prefix hits alter executed prefill shapes. Predict that setting only existing vllm_cfg.enable_prefix_caching=false in the original8x65536 TP2/PP2/CP1 native training reduces actual scorer/generation MAE to<=0.02. A continuing FAIL rules out prefix reuse as a sufficient repair. Preserve full masks, weights, head/RoPE/FA4, loss and tolerance. Native V1 source rejects trace replay (requires Model Runner V2); do not allocate for that unrelated runner change.

- Native no-prefix long control CPU config initially rejected a missing YAML key; corrected to existing Hydra `+` addition before GPU. MasterConfig/native shape check and personaldryrunPASS, then submitted once24H800 `exp-1112-231515-172060`, launcher2948. Current codesign inventory readHTTP200 covers123machines,25usableGPUrequests/twofullyusable8GPUmachines; this is neither H800type filtering nor verified personalquota. Both24GPUgangjobsremain monitored, no duplicate.
- Independent D GEMM hypothesis before execution: PyTorch F.linear decode M1 and prefill M2048/65536 may select different BF16 accumulation/reduction paths. On identical captured first-layernorm rows and actual checkpointTP2QKVshards, measure M1/32/2048/65536 against each other, nativecapturedQKV andFP32reference. If disabling allow_bf16_reduced_precision_reduction removes the shapegap, investigate it as a causalprecisionrepair; otherwise reject this switch. OneH800purekernelprobe, no training/objective change; no arbitraryfactors.

## 2026-11-12 — exact current-control recovery and check-progress

- Recovered launchers: invariant 51331 exit1 after actual platform failure; no-prefix launcher2948 remains alive. Previous read-only monitor69544 exit0 after observing Failed. Reused its tasktmp inspector with explicit job names and started no-prefix monitor22111, exact Running at15:18:34Z. No resubmission and no formal200run.
- Invariant exp-1112-225944-064229 failed before first update during initial refit communicator warmup in StatelessProcessGroup.init_nccl_communicator -> broadcast; actual nccl.bindings.nccl.NCCLError InternalError(3). Launcher cleanup23:16:23.535HKT follows actual23:16:04HKT business traceback. No training dumps/updates or numerical verdict exist. Exact-object incident RCA collector3797 completed six primary reads; both all-operation audits failed on bounded timeout. Initiating manual-stop actor/reason/requesttime remain UNKNOWN. Do not retry rejected invariant numerics solely because this native control failed.
- Retained final-long-attention-layer-comparison.json, final-long-norm-comparison.json and no-prefix exact args/resolved config in evidence/. Capture coverage remains eight samples, two TP ranks and32responsepredictionpositions per sample, not fullprompt coverage.
- check-progress: A/Q/B/CPASS, D/E/FTODO; D1–D7CLEAR; S5 stale invariant frontier corrected in plan/review. Current mainline is existing no-prefix actual training plus bounded CPU GEMM-input verification. The previous capture tooling failures remain in history and are not reclassified as product errors.
- Initial remote Ray-log read had a tasktmp script quoting SyntaxError; corrected raw script returned0 and found no NCCL WARN records in retained Ray stderr/out files. This limited absence is not evidence that NCCL or network is healthy. CPU capture check verifies actual rank1 norm shape32x1536, rank0/1QKV32x1024 and matching sample IDs. GEMM checkpoint identity/layout check is in progress; no GPUprobe submission yet.

- No-prefix long control completed its actual business entrypoint;256response-token comparison MAE0.0605199088604FAIL,p950.2237538993,max1.1958322525. All8samplemeans exceed0.02. Actualengine logs confirm enable_prefix_caching=False.1successfulupdate/0skipped,loss-1.09825278e-8,grad0.00337985158;finite. This rejects prefix reuse as a sufficient correction without asserting paired identical trajectories with the previous run. Native terminalobjects are being collected separately.
- One-H800 GEMM probe actual checkpoint/shard/capture boundaryPASS (session23845exit0) and personaldryrunPASS (52803exit0); submittedonce23:24:21HKTasexp-1112-232421-248453,launcher74653. Monitor87080 seesStarting; retainlauncher,no duplicate. Exact6QKVweights+bias copied selectively for CPUlayout only,6.3MB tasktmp;CPUFP32rounded/nativenumeric max0.25,meanMAE1.56648e-6 over16records. This observed reference is not a GPU kernel pass. Probe runtime remains actualstudentBF16shards and32capturedrows.
- Additional ranked D predictions before actual decode instrumentation: H1 batchshapeGEMM first diverges inQKV and disablingBF16reductionremovesit;H2RMSNorm rowshape first diverges after matchingQKV;H3attention/KVreduction firstdiverges inattention aftermatchingQKV/RoPE. Prepare a temporary2GPU actualdecode + samegeneratedtokensprefill observer by reusing nativeLLM.generate and existinglayerhooks; saveoneforward perpredictionposition to preserveall31decode steps despite existingonecallcapture semantics. No fullpromptcaptureexpansion,noformalworkerhooks,and no productionprecisionchange. Preparation only; native samegeneratedtokens replay follows actualoutput if needed.


## Resumed completed GEMM milestone and D frontier correction

- Complete GEMM16-record JSON and business/terminal objects read successfully. All256 against-native/full-shape comparisons exact;1H800, creator/local source/runtime verified; Succeeded/exit0/restart0, launcher74653 and monitor87080exit0. Reject first-layer QKV shape/reduced-precision hypothesis; no production precision change. Selected result/business/terminal evidence copied into task evidence.
- No-prefix-cache actual update/parity FAIL and batch-invariant initial-NCCL failure now reconciled in final workflow test report; terminal objects retained. Plan no longer says these controls are Running.
- Actual eager decode observer uses original8 long prompts,32 fresh response tokens, then full prefill of those same new tokens. Production capture/backend unchanged. Preparation CPU invocation first lacked project PYTHONPATH; corrected before GPU. Actual outer replay boundary then caught returning new-token output to the original-token replay; restored its existing independent original-token replay call. No GPU allocation occurred for either CPU failure. Actual CLI and serialized callbacks remain under verification.
- check-progress: A/Q/B/C PASS,D/E/F TODO. D1-D7 CLEAR: each control tested a recorded prediction and narrowed the cause; unsupported trace mode not attempted. S5 HIT: plan/review had stale no-prefix Running; corrected. Mainline frontier is D actual-decode versus same-token-prefill followed by native replay; E continues to wait for numerical acceptance. TP4gradient and raw reference-gradient failures remain unresolved and are not relabeled PASS.

- Actual decode CPU boundary PASS8samples/256response tokens/256decode snapshots/8prefill snapshots/serialized callbacks, Python3.13.14, command in final test report. Pinned Ruff0.9.9format/checkPASS. PersonalPython3.10dry-runPASS exact2H800/codesign/local/data source; submitted once **exp-1112-233623-543941**, launcher45185, read-only monitor50481. Keep launcheralive; no duplicate. Engine remains eager/TP2/FA4/FP32RoPE/head; existing no-prefix setting retained for isolation, maximum32. Newtokens/native replay pending, formal200unsubmitted.


## Actual decode callback failure and scoped repair

- Original observer exp-1112-233623-543941 is Failed, container exit1/restart0, with no model forward or numerical result. Model initialization completed. The first apply_model crosses ray_executor_v2 -> multiproc_executor.worker_busy_loop -> shm_broadcast.dequeue -> pickle.loads and raises AttributeError: Can't get attribute 'begin_capture' on worker __main__. Callback provenance is the root cause; NIXL warning is unrelated. Worker failure23:38:18HKT and container finish23:38:23 precede launcher cleanup23:38:40.523. Launcher45185exit1, monitors50481/23050exit0.
- Exact-job RCA six primary reads succeeded; two all-operation audit reads timed out15s; no existing DeepScan. Manual-stop actor/reason/request time remain UNKNOWN. Evidence final-actual-decode-terminal-objects.json; raw365-line stderr under tasktmp/final-actual-decode-terminal-watch/.
- Previous CPU preflight exercised cloudpickle only, so its serialization verdict did not cover the actual second plain-pickle worker hop. The minimal independent-process command uv run --no-sync python check_decode_callback_pickle.py --mode original returns exit1 with the identical AttributeError (session94303). Preserve decode-callback-pickle-original-red.log.
- Scoped repair in progress: importable start_forward_capture in existing generation_layer_outputs.py, observer at tests/manual/mopd/inspect_actual_decode.py; no objective/backend/precision change. CPU preflight now exercises cloudpickle then plain pickle in an independent worker before callback execution. GPU rerun waits for GREEN actual CLI and personal dry-run.

- Corrected plain-pickle independent-worker GREEN (4482exit0), and actual CLI cloudpickle->plainpickle boundary GREEN (61915exit0) on Python3.13.14. Same8samples/256response positions/256decode snapshots/8prefill snapshots; both imported callback identities validated in independent processes. Exact commands use PYTHONPATH=/data/ycfeng/sota-engine/RL, actor UV_PROJECT_ENVIRONMENT, OMP/MKL/OPENBLAS4, CUDAcompat LD_LIBRARY_PATH and timeout60s; scripts check_decode_callback_pickle.py --mode fixed and preflight_actual_decode.py under tasktmp. Logs decode-callback-pickle-fixed-green.log and actual-decode-2-cpu-boundary.log. GPU numerical behavior remains unverified.
- Pinned Ruff0.9.9lint/format PASS for generation_layer_outputs.py and inspect_actual_decode.py. Personal dry-run82351exit0. Submitted once via submit_final_actual_decode_2.py as exp-1112-234539-001675,2H800/codesign/current-local source; launcher10891 and monitor29400 alive. New cloud output root final-actual-decode-2 preserves the failed job. No formal200job submitted.


## Actual eager decode observed; attention is the first mismatch

- Corrected observer exp-1112-234539-001675 completed all8samples and235actual response tokens. Two responses stop at17/26tokens dueEOS; six reach32. Same-token decode/prefill MAE0.02390396525, p950.05624575615,p990.171537323,max0.584892273: FAIL unchanged0.02. All actual tokens/masks retained; 4/8sample means exceed0.02. This is a new trajectory set and is not a paired comparison to prior async training trajectories.
- All8samples/twoTP ranks: embedding, first input norm, QKV and pre/postRoPE Q/K are exactly equal at every scored position. First mismatched stage is layer_0_attention on both ranks. Firstresponseprediction attention is exact; incremental decode attention differs sparsely (rank0 per-sample MAE2.88e-7–1.58e-6, max0.0009765625–0.015625), then projection/MLP/later layers diverge. This observation narrows the numerical boundary but does not yet prove a causal repair.
- Auxiliary analysis initially could not use local-only uv path on SSHhost (exit127). Run standalone script via SSH stdin with remote Python3.10/Torch2.5.1 instead. First coverage assertion found one additional unscored forward afterEOS in each17/26token sample. Corrected analysis explicitly retains these unused positions separately and compares every actual235 response-prediction position, no response filtering. Second analysis85972exit0 covers all16sample/rank records. Exact command: ssh shai-ycfeng env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout60s python3 - --root <personal cloud root>, stdin analyze_actual_decode_layers.py; raw errors retained separately. Numerical training unaffected.
- Business finished.json=native_execution_succeeded; platform/replica Succeeded, containerexit0/restart0. H800gpu-h800-0329, Python3.13.14/Torch2.13.0+cu130,personalcreator,codesign,localNFS100.96.128.195verified. Launcher10891/monitor29400exit0. Selected comparison/allstage errors/terminal/business/runtime retainedinevidence/final-actual-decode-2-*.
- Native same-newtokens replay actualCLI/Policy-boundary55771exit0; independent CPU token/mask/weight_version0 checksPASS8/235. Personaldry-run49180exit0. Submittedonce2H800exp-1112-235130-765032via submit_final_actual_decode_native_2.py,launcher41980/monitor35946alive. No formal200submission.

- Verified imported-callback/GPU step committed with sign-off as8b8b6e7a; branch remains local. Business/runtime receipt now encoded as one JSON object.
- Ranked remaining attention hypotheses before the next control: H1 FA4 split-KV reduction varies between one-query decode and full prefill; H2 dense/paged attention or score arithmetic differs despite same selectedQ/K; H3 unobserved promptKV differs before selectedpositions. Every selectedfirstlayerQ/K/V/RoPE row is exact, which narrows but does not exclude H3 for the uncaptured prompt. Source flash_attn.py:587–600/1153–1183 exposes eager heuristic num_splits0; the existing split limit applies only to CUDAgraphs. Whole batchinvariant mode was already rejected; do not replay it or add a new core config/interface.
- Next bounded control uses existing vLLM attention_config.backend=TRITON_ATTN only, with the same8sourceprompts/TP2/no-prefix/eager/FP32head/FP32RoPE/fullvocab and actualdecode followed by samegeneratedtokensprefill. Prediction: if the FA4 attention path dominates the gap, changing only the existing attention selector reduces per-token MAE below0.02 and/or changes first mismatchedstage. A continuingattention-firstFAIL rejects this backend as sufficient; no numericetolerance change. Preserve pairednative replay and originalfixedtoken replay to measure the training-side difference before promotion. CPU actualCLI and personaldry-run precede anyGPUcontrol.


## 2026-11-13 — resume and ordinary native replay

- Recovered HEAD8b8b6e7a and existing task requirements/plan/artifacts. TRITON exp-1112-235613-448362 finished Succeeded00:00:51HKT; launcher40958/monitor88351 completed. Complete8samples/241tokens decode-prefill MAE0.02291390176, signedmean0.00022068496, p950.06023598, p990.10800505,max0.18308735: FAIL unchanged0.02.6/8sample means exceed0.02. Do not promote this existing selector as a sufficient fix. Layer/terminal/business reconciliation continues.
- Native observer exp-1112-235130-765032 Failed on shortEOS sequences65521/65530 padded natively to65536. Embedding/SP hook compared against real input length; physical shard32768 requires padded length65536. Original error retained. CPU hook RED88960exit1 and GREEN85806exit0 verify reading actual embedding input_ids with with_kwargs=True. Padding fix remains uncommitted/unverified on GPU; no complete8sample native result exists from the failed observer. RCA six primary reads successful/two all-operation audits unavailable; initiating manual-stop actor/reason/time UNKNOWN, launch cleanup follows business error.
- Shortest D route uses ordinary replay_logprobs.py, not further observer extensions. ActualCLI/Policy CPU preflight41724exit0 and personaldryrun65687exit0 already covered unchanged8samples/235tokens/version0/TP2PP1CP1/fullvocab. Submitted once locally00:02:48HKT through personal Python RJobBackend as exp-1113-000248-302886,2H800/codesign,launcher56177,monitor9188. Exact args final-actual-decode-native-3-submit-args.json; dump final-actual-decode-2/exp-1112-234539-001675/logs/actual-decode-tokens.jsonl. Keep launcher alive; no formal200run submitted.
- Session-memory recovery now loaded and initialized; handoff state stays in this progress record. Preserve unrelated recipe/evidence changes.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 模型、数据、全词表路径已记录 |
| Q | PASS | 用户已确认工程验收与运行授权 |
| B | PASS | 原student与五教师短流程、runtime已验证 |
| C | PASS | 数值观测与参照机制可执行；原梯度FAIL保留 |
| D | TODO | 长序列update通过，训推一致性仍FAIL |
| E | TODO | 正式200update与保留checkpoint独立replay未完成 |
| F | TODO | 最终归档依赖D/E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 8b8b6e7a callback修复至今｜**当前前沿：** D native同tokens核验、attention诊断、原梯度与TP4异常解释

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | padding observer停止扩展，改用已工作的普通replay |
| D2 细节反复重试 | CLEAR | TRITON是预先记录的单变量attention假设；FAIL不重复 |
| D3 支线问题修补 | CLEAR | 当前工作均核验D接受条件 |
| D4 可并行任务停滞 | CLEAR | native replay执行时并行收TRITON证据、检查gradient |
| D5 超范围镀金 | CLEAR | 无新核心配置或接口 |
| D6 分析空转 | CLEAR | attention首差异与selector结果已形成路线决定 |
| D7 路线违背 | CLEAR | 正式run仍等待D；0.02不改变 |

**文档检查：** S5 plan.md旧native observer Running与实际Failed冲突；已更新。S1–S4/S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：先更新状态文档，再继续。

**文档更新（先执行）：**
1. plan.md当前D：记录observerFailed、TRITONFAIL和普通native新job（可直接执行，已完成）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：普通native replay全部235tokens → 保留总体/样本/尾部误差与完整终态。
2. 并行：TRITON证据与层边界 ‖ 原gradient reference和TP4异常因果核验。
3. E：D通过后普通resume入口200/256/65536 → 原生save/exit/resume继续update。

## 4. 最值得关注的 3 点

1. TRITON全量仍FAIL，不能当修复。
2. 新native job用普通评分入口，不依赖失败observer。
3. 原raw gradientFAIL与TP4grad70.84613尚未解释；正式200update未提交。


## 2026-11-13 — ordinary native replay and next precision control

- exp-1113-000248-302886 completed business replay and platform Succeeded; launcher56177 returned exit0. All235same-weight FA4 response tokens retained. Native minus decode: signed mean0.00062814260, MAE0.02335487433,p950.06115665,p990.13129736,max0.53070211: FAIL0.02. Native minus identical-token prefill: signed mean0.00112446242, MAE0.01920516313,p950.04833212,p990.06405975,max0.17683506: overallPASS;3/8sample meansFAIL. Neither mean is filtered.
- TRITON reconciliation covers all16sample/rank captures; first difference is layer_0_attention everywhere, same selected embedding/QKV/RoPE inputs. Its original256fixedtoken prefill versus recordedasyncFA4generation MAE0.05341011584FAIL. Business completion, H800/runtime and container exit0/restart0 retained; changing the existing selector is insufficient.
- Ranked falsifiable hypotheses before the next control: H1 sparse BF16 attention/norm rounding is amplified through the remaining layers; matched FP16 under the same model weights/tokens/FA4/head/RoPE should lower native/prefill and actualdecode/prefill MAEs below0.02. H2 split-KV reduction is sufficient: setting single split should remove first attention mismatch; current eager config has no such existing selector, so this remains untested. H3 earlier prompt KV is already different outside captured positions; a continuing FP16 FAIL would require focused KV-input investigation.
- Select the existing float16 precision route for one paired control, not a new schema or production patch. First run actual8longpromptdecode+samegeneratedtokenprefill on FP16, and native FP16 replay of the original256fixedtokens in parallel. The originalBF16 weights are still the requested source; precision changes only computation/storage. Keep full151936vocabulary, no filters/scales/tolerance changes. Native optimizer flags must follow fp16/bf16 consistency, FP32 head remains enabled. ActualCPU config/CLI and personaldryrun precede both submissions. No FP16 setting is promoted before actual parity/training/finite evidence.


## 2026-11-13 — parallel precision and TP4 gradient controls

- FP16 actual decode CPU boundary 99779 exit0, including actual CLI and cloudpickle/plain-pickle worker hops; native FP16 preflight 81358 exit0 verifies actual Policy config, native precision/optimizer dtype validation and FP32 logits. Personal dry-runs 17826/77093 exited0. Submitted once: actual decode exp-1113-000959-614638 (launcher11189) and original256token native replay exp-1113-001000-063749 (launcher55372), each2H800/codesign/localNFS. Monitor preparation reuses the existing exact-job monitor.
- TP4 causal hypothesis before execution: the original studentTP4/teacherTP2 arithmetic difference drives the large self-distillation gradient. Changing only the existing teacher TP to4 should reduce loss toward zero and gradient substantially below70.84613; an unchanged large gradient rejects this attribution. Original8longsamples, studentTP4, fullvocabulary, BF16/FP32 head, token normalization, generationTP2 and native optimizer remain unchanged. This control diagnoses D; it does not promote TP4 for formal training. CPUnative/teachergeometry and personaldryrun precede allocation.


## 2026-11-13 — FP16 recovery and matched-TP control

- Recovered the two precision jobs and inspected business artifacts independently of launcher exit. Original native FP16 replay exp-1113-001000-063749 retained all 8 samples / 256 response tokens; business succeeded, platform Succeeded, container exit0/restart0. It is not yet a matched-FP16 parity PASS: the stored generation baseline is BF16. Exact replay/runtime/business/terminal records are in evidence/final-original-native-fp16-*.json.
- FP16 decode exp-1113-000959-614638 failed during Qwen2 rotary factory initialization: `ValueError: FP32 RoPE requires default full-head BF16 RotaryEmbedding`. Native prefill never began. Eight RCA primary queries succeeded, exact RJob/replica UIDs retained; container exit1/restart0. Startup scheduling/preemption messages precede execution and are not evidence of workload preemption. Platform cleanup and launcher Job Stop follow the business exception. Initiating manual stop actor/reason/time are UNKNOWN. Raw collection: /data/ycfeng/tmp/nemo-mopd-20261112/rca-exp-1113-000959-614638/.
- The original CPU callback preflight mocked engine construction and did not exercise the real factory. Existing factory regression extended to FP16: RED session54187 exit1 (FP16 factory exception; BF16 passed). First broad pytest invocation72723 timed out while parent conftest packaged the repository into Ray; no numerical verdict. Scoped actual factory tests92305 exit0: 7 PASS. Existing RoPE source patch now supports default full-head BF16 or FP16, keeps FP32 cache/FlashInfer and rejects FP32, partial heads and scaled RoPE. Exact previous installed snippet was backed up and reversed, then the normal patch applied; no new compatibility path. CPU installed-factory initial PermissionError was a root-owned patch_lock, repaired with scoped chown; rerun84425 exit0, cache [65537,128] FP32, query dtype FP16. GPU arithmetic remains pending.
- Submitted corrected same-variable FP16 decode once locally: exp-1113-002107-537824,2H800/codesign, launcher79624. Same model/prompts/FA4/FP32head/FP32RoPE/unchanged0.02 ceiling; only the already-tested dtype support and output directory differ from failed initialization. No precision promotion before full numerical and training results.
- TP4 matched-teacher preflight83877 failed because the temporary script imported nonexistent nemo_rl.algorithms.on_policy_distillation. Inspected actual opd.py and fixed only that import. Preflight41571 exit0; existing personal dry-run3534 exit0. Submitted locally exp-1113-001611-128499,24H800,launcher37211,ordinary controller, original8x65536 self-distillation and1update. Only teacherTP2→TP4 changes the causal variable. Prediction from preceding entry retained: gradient should decrease substantially if TP mismatch causes70.84613. No TP4 promotion implied.
- Exact-job monitor51636 covers corrected FP16decode andTP4, with45s interval and7200s ceiling. Both launcher lifetimes are preserved. Formal200/256/65536 run remains unsubmitted; D still requires paired precision results, actual objective gradients and anomaly diagnosis.


## 2026-11-13 — head arithmetic and native objective reference hypothesis

- Ranked before the next GPU measurement: H1 mixed-precision native head GEMM (BF16 inputs/weights, FP32 TE output) differs from FP32 torch.matmul reconstruction. Prediction: calling the installed MCore `_linear_forward` on identical captured hidden states and head weight eliminates direct-logit and raw-gradient error. H2 TP layout mismatch drives student gradient (matched-TP run already executing). H3 raw-gradient tail includes non-loss tokens: native scalar loss with actual response/sample masks and observed microbatch token denominator should differ from the raw all-position statistic; this does not erase the original raw FAIL.
- Extend only the existing diagnostic NativeForwardWorker to retain the current FP32-reconstruction result, a same-native-GEMM control, raw full-vocabulary gradients, and gradients through actual ClippedPGLoss using native data masks. Report the denominator explicitly as observed microbatch tokens; this control is not the global-update gradient or a replacement bound. Proposed short control is8freshsamples/4096ceiling/32responsecap/1update,24H800,original student used as selfteacher,TP2CP1/fullvocabulary. No production head change before this measurement. Preflight verifies the actual task's token-level/no-reference-penalty objective and worker imports.


## 2026-11-13 — recovered FP16 three-way results and gradient frontier

- Recovered new exact IDs: ordinary nativeFP16 replay exp-1113-002800-473836 (launcher36813), head/objective-gradient diagnostic exp-1113-002800-492957 (launcher98272). Native replay is now Succeeded with complete8samples/241response tokens, business marker, containerexit0/restart0; corrected decode exp-1113-002107-537824 also Succeeded/containerexit0/restart0, launcher79624 exit0. Head diagnostic is Running with3nodes/24H800 and mounted Python3.13.14/Torch2.13+cu130; TP4 exp-1113-001611-128499 remainsStarting, launcher37211 alive. Exact-job monitor7493 uses watch_final_new_fp16_gradient.py --interval45 --duration7200; originalmonitor51636 is retained.
- Complete paired numerical result (no token/sample filtering): native minus actualdecode MAE0.00382072871157, signedmean0.00075029103,p950.00904369354,p990.01644172668,max0.17280316353; native minus identical-tokenprefill MAE0.00260485053310,p950.00713109970,p990.00889987946,max0.01182937622; prefill minusdecode MAE0.00337638158769,p950.00783824921,p990.01604747772,max0.17231357098. All three unchanged0.02MAE checks PASS;8/8samplemeansPASS each. Original256fixedresponse-token native/prefillFP16 paired MAE0.00290350709alsoPASS. Evidence final-actual-fp16-three-way-analysis.json preserves all samples,group_id,absolutepositions,tokenIDs,all signederrors and the30largest tails.
- Largest tail: sample3975f292-6577-4d89-88f0-4ec81b32a364_g0,responseoffset11,token198(newline),position65515. Native-1.42182541 vs decode-1.59462857 vs prefill-1.42231500. Native/prefill at this point differs0.00048959. Complete16rank/sample layer comparisons first differ at layer_0_attention; at the tail prediction position65514, capturedQKV/RoPE are exact,attentionmaxdelta0.00012207031,layer0max0.001953125,layer13max16.625,finalnormmax0.7890625. This localizes the measured decode/prefill residual; selected positions do not prove the entire prompt KV is equal. No tail is excluded and no tolerance changed. All original long synthetic prompts cover math only; five-domain real training still needs its own evidence.
- Collection commands: python3 collect_final_new_fp16_gradient.py (read-only SSH exact personal cloud roots) and python3 analyze_final_fp16_three_way.py, followed by personalPython inspect_final_new_fp16_terminal.py. Native/vLLM runtime, business and terminalobjects archived in evidence/final-actual-decode-{native-fp16,fp16-2}-*.json. Head/newnative submitargs retained. Initial python command was absent; host read/analysis uses python3, workload probes use pinned uv.
- Found existingruff0.9.9 at /data/ycfeng/tmp/uv-cache/archive-v0/4Z7oFB0RosbDJ1nO/bin/ruff; priorofflineuvtool attempt had not formatted anything. Scopedformat found2files, then formatted only native_forward_worker.py and test_vllm_patches.py;4filelintPASS. RoPE runtime support has7previous actualfactorytestPASS and successfulFP16 GPUdecode; this code step can now be committed separately. Newhead diagnostics remainunverified/uncommitted.
- Next independent D measurement is actualFP16 longtraining using nativeTP2/PP2/CP1,8x65536,1update,24H800 and ordinarycontroller. Match student/vLLM/teacher computation and transported hiddenpayload tofloat16; retain FP32logits/full151936vocab/responsemasks. Initialnativeoptimizerloss_scale default2**32 was confirmed on CPU; initialize its unchangeddynamic scaler at installedtorch.amp.GradScaler default65536 instead, retaining unscale/overflow handling. This is optimizer gradient scaling, not a change to the distillation objective. ActualCPUdtype/config andpersonaldryrun mustPASSbefore launch. Prediction: successfulfiniteupdate/0skipped plus<=0.02sameweightparity; fullnorm/gradient diagnosis andfive-teacherFP16save-resume remain required beforeformalrun.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原模型和数据身份已核验 |
| Q | PASS | 已确认200-update工程合同与GPU授权 |
| B | PASS | 运行环境和全词表路径已验证 |
| C | PASS | 现有数值参照可执行 |
| D | TODO | FP16三方一致性PASS，梯度与实际FP16训练仍待验收 |
| E | TODO | 正式200updates与正式checkpoint replay未执行 |
| F | TODO | 全体验收和归档未完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次checkpoint至本轮FP16完整配对结果｜**当前前沿：** D梯度参照、FP16长训练与五教师save/resume

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用exact-job monitor/普通replay；分析保留验收所需全量误差 |
| D2 细节反复重试 | CLEAR | FP16为预先记录的单变量精度假设，配对结果收敛；失败selector不重试 |
| D3 支线问题修补 | CLEAR | RoPE factory修复直接解除已发生的FP16初始化失败 |
| D4 可并行任务停滞 | CLEAR | native replay、headgradient、TP4同时推进，FP16长训练独立准备 |
| D5 超范围镀金 | CLEAR | 无新核心schema或并行框架 |
| D6 分析空转 | CLEAR | 新241token结果已决定推进真实FP16训练 |
| D7 路线违背 | CLEAR | 保留原五教师/151936/0.02/200/256/65536；正式run仍待D |

**文档检查：** S5 plan.md/test_report仍记新native pending且缺失FP16PASS，与终态矛盾；本轮更新。S1–S4/S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：先同步最新验收事实，再继续。

**文档更新（先执行）：**
1. plan.md/progress.md/test_report：记录完整FP16配对PASS、精确终态、新jobIDs和实际训练未验证（可直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：收head同GEMM/实际maskgradient、TP4matchedteacher → 解释原gradientFAIL和70.84613异常。
2. 并行：原生FP16长训练 ‖ 五教师FP16短训练/save/exit/resume，保留finite/0skipped/参数变化证据。
3. E：D通过后普通resume入口提交200/256/65536，完成post-refit和checkpointreplay。

## 4. 最值得关注的 3 点

1. FP16实际native/decode全量MAE0.00382073PASS，尾部和所有样本已保留。
2. 原headgradientFAIL和TP4norm异常仍未解释；不能算D完成。
3. 正式200updates尚未提交，FP16训练健康与恢复是下一项实际验收。


## 2026-11-13 continuation: gradient failures and current training frontier

- The RoPE repair is signed commit `986ec788`; only patches.py and the corresponding unit test were committed. Seven factory/regression checks and real FP16 decode/replay passed. Diagnostic worker changes remain uncommitted pending GPU verification.
- First gradient probe `exp-1113-002800-492957` Failed with `KeyError: token_mask` in teacher scoring, which does not carry training masks. Launcher98272 exit1; zero verified updates. Move masked objective comparison to the actual prepare_loss_input stage; teacher scoring retains only same-hidden/head GEMM/distribution/raw-gradient checks. CPU reproducer `check_teacher_gradient_stage.py`: RED73733 exit1; GREEN25095 exit0,2validtokens. A 5-token payload with 8-position native logits additionally reproduced a shape error (RED64865 exit1); padding the payload consistently passed (GREEN23878 exit0). These observations are diagnostic correction evidence, not a production-gradient PASS. CP1 remains the supported diagnostic setting.
- Corrected probe `exp-1113-004508-331543`,3x4H800, Failed on 2026-11-13 00:47:09 HKT before the training command: wait_cluster detected repeated Ray NodeManagerAddress values. Source is `tests/manual/mopd/run_on_rjob.py:92`, where distinct physical nodes are required. Native head replica log shows ValueError; later non-head errors and launcher stop are cleanup. Launcher86699 exit1. No numerical result exists. Six primary RCA queries succeeded; RJob/replica all-operation audit queries timed out. Initiating manual-stop actor/reason/request time remain UNKNOWN. Raw evidence: tasktmp/rca-exp-1113-004508-331543/ and head-gradient-reference-2-rca-collector.log.
- Placement hypothesis: 4-GPU replicas can share an 8-GPU physical node; submitting full 8-GPU replicas should produce distinct nodes without changing the guard. Prepared2x8H800 configuration keeps student4/generation4 on one node and teacher4 on the other (4allocationGPUs unused). The native setup.py policy_nodes==1 branch explicitly supports that split. CPU native preflight3207 exit1 because this continuation omitted the established CUDA compatibility LD_LIBRARY_PATH; original error preserved. Rerun12029 restores that existing environment, with no dependency edits.
- Current exact-job monitor observations at approximately00:50HKT: longFP16 exp-1113-003652-303384 Running; TP4matchedteacher exp-1113-001611-128499 Running; five-teacherFP16 exp-1113-004218-029183 Pending. Creator=i-fengyicheng and current-machine source100.96.128.195:/data/ycfeng:/data/ycfeng verified in status receipts. Launcher processes1422146/1051155/1519813 alive, respectively; do not submit duplicates.
- LongFP16 contract:8freshtrajectories,65536ceiling,32responsecap,1update,nativeTP2/PP2/CP1; five-teacherFP16 short contract:32freshtrajectories,8192ceiling,128responsecap,5updates,native save/exit/resume. Student, vLLM and teacher compute precision=float16; full hidden payload=float16; FP32logits and151936fullvocabulary retained. Native dynamic scaler initialized at installed torch.amp.GradScaler init_scale65536, retaining native unscale/overflow handling. This does not rescale the training objective.
- Formal CPU preflight30123 exit0: FORMAL_FP16_200_256_65536_NATIVE_CONFIG_PASS,initial_loss_scale65536,loss_scaleNone. Personal dry-run28872 exit0. Commands: uv run--no-sync python tasktmp/preflight_formal_fp16.py in pinned Megatronactor runtime; personalPython tasktmp/submit_formal_200updates_fp16.py--dry-run. Exact args/configs remain tasktmp/formal-200updates-fp16-{submit-args,resolved-config}.json. No formal submission performed.
- Acceptance remains unchanged: all samples/tails, MAE<=0.02; actual raw/masked objective gradients, norm70.84613 explanation, FP16 long update health, five-teacherFP16 save/resume and post-refit parity must be verified before formal200updates.


## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原student、五教师和数据身份已核验 |
| Q | PASS | 200-update合同与H800执行授权已记录 |
| B | PASS | 全词表和正式FP16配置CPU预检通过 |
| C | PASS | 数值参照和观察入口可执行 |
| D | TODO | FP16配对PASS；梯度probe需完成，长训练Running |
| E | TODO | 五教师FP16save/resume排队；正式200updates未提交 |
| F | TODO | 完整验收归档待D/E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 前次FP16配对checkpoint至本轮实际状态恢复｜**当前前沿：** D梯度/训练健康与独立短save-resume

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | mask/padding修正直接验证既定梯度条件；placement恢复训练入口 |
| D2 细节反复重试 | CLEAR | mask和placement是不同已观测故障；本次明确2x8 placement预测 |
| D3 支线问题修补 | CLEAR | 全部变更服务D的数值与工作流验收 |
| D4 可并行任务停滞 | CLEAR | longFP16、TP4和五教师save/resume均已提交；继续原job |
| D5 超范围镀金 | CLEAR | 无新接口、schema或通用框架 |
| D6 分析空转 | CLEAR | 真实训练已Running；gradientreplacement准备中 |
| D7 路线违背 | CLEAR | 保留原五教师、151936、0.02与200/256/65536；formal等待D |

**文档检查：** S5 plan/test_report原记headRunning，与Failed终态矛盾；已纠正。S1–S4/S6无明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：先同步文档再继续。

**文档更新（先执行）：**
1. plan/progress/test_report：保存两次失败、CPU修正、livejob状态与formalpreflight（已直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：完成2x8梯度probe的CPU预检和GPU实测；解释rawgradientFAIL与TP4norm。
2. 并行：收longFP16实际update/finite/0skipped/parity ‖ 五教师FP16 save/resume/参数变化/状态连续性。
3. E：D通过后提交ordinaryresume入口200/256/65536，验收实际200successfulupdates和checkpointreplay。

## 4. 最值得关注的 3 点

1. FP16三方配对已PASS，原梯度数值问题仍未验收。
2. replacementprobe失败原因是物理节点重复，零verifiedupdates；修正placement后才重提。
3. 三个既有trainingjobs继续运行或排队，正式200updates未提交。


## 2026-11-13 00:53 HKT gradient placement resubmission

- Native2x8placement preflight12029 exit0 and personaldryrun84054 exit0. Actual local personal submission launcher50016 created `exp-1113-005256-166805`; keep the launcher alive. Study allocation remains student4/generation4/teacher4; runtime16H800. No corelauncher change. Exactargs/resolvedconfig and priorfailureRCA objects/index retained in evidence/.
- Command: personalPython tasktmp/submit_final_head_gradient_reference_3.py with companyproxy andPYTHONPATH=/data/ycfeng/steptron; rawlog tasktmp/head-gradient-reference-3-launch.log. Nativegradientresult pending.


## 2026-11-13 01:05 HKT: verified gradient and TP4 controls; FP16 optimizer configuration repair

- Gradient probe `exp-1113-005256-166805` Succeeded. Complete32teacher-shard comparisons: native mixed-precision `_linear_forward` vs directnativehead logits and rawgradient both exact0. OriginalFP32reconstruction rawgradient remains1.476183479098836e-5 FAIL at1e-6, now explained by measurednativeGEMM versusFP32GEMM arithmetic; do not relabel it exactPASS. Full-distribution logprobMAEmax3.23359418e-5,max1.25885010e-4; selected-token4117all-positionlogprobMAE5.94749225e-6,max4.57763672e-5. Complete16actualmask/objective shardchecks use32validresponsetokens and1validsequence each: gradientmax3.29035032e-7,normmax6.86279293e-7,16/16PASSat1e-6. Referencegradientnorm0 inallrecords; relativeerrorundefined. This is observedmicrobatchobjectivegradient, notglobaloptimizerparametergradient. Allrecords retained in evidence/final-head-gradient-reference-3-analysis.json. Signed diagnosticcommit `d6dea5bb`, native_forward_worker.py only; actualCPUteacherstage/paddingrepro, nativeGPUrun andscopedRufflint/formatPASS. No productionlosschange.
- BF16TP4matchedteacher `exp-1113-001611-128499` Succeeded;1successfuloptimizerupdate,0skipped,loss-2.901869411431335e-9,preclipgradnorm0.0027350757736712694. Paired against original `exp-1112-222130-343023`: all8completeinputsequences/responsemasks and256student/generationlogprobs exact. TeacherTP2->TP4 removes originalstudent/teachergapMAE0.04910842158/max3.18075824 toexact0, and originalgradnorm70.84613 collapses to0.00273508. This controlledresult explains the anomalousself-distillationgradient throughteacher/studentforwarddifferences acrossparallelism, withoutobjective/rescalingchanges. Native/vLLMMAEremains0.06852769961FAIL;TP4notpromoted. Exactcompletepairedresponseerrors in evidence/tp4-matched-teacher-paired-analysis.json andalltrainingmetrics in final-tp4-matched-teacher-frontier2-analysis.json. Launcher37211 exit0.
- First actualFP16longrun `exp-1113-003652-303384` Failed before a successfulupdate, atnativeoptimizer clip_grads.py:173: decoupled_grad.dtype mustbeFP32/BF16, butDDPgrad_reduce_in_fp32=false selectsmodelFP16gradbuffers. Launcher34895 exit1. NativebufferselectionCPUrepro check_fp16_gradient_buffer.py RED67800exit1 reproducesexactassert; existinggrad_reduce_in_fp32=true GREEN92623exit0 selectsFP32andpassesrealclippingdtypeguard. GREENusesCPUoverflowbuffer andnormbelowcliplimit, soGPUfusedclippingstillrequiresactualtrainingverification. Noinstalleddependency/coreoptimizeredit. SixprimaryRCAreadssucceeded;2all-operationauditquerytimeouts remainexplicitUNKNOWNmanualactor/reason/requesttime. Evidence/final-long-training-fp16-rca-{objects,index}.json andtasktmp/rca-exp-1113-003652-303384/ preservecoverage.
- Five-teacherFP16run `exp-1113-004218-029183` usedtheidenticalincompatibleconfig; stoppedwithrequired rjob_stop.sh at2026-11-1300:58:49HKT,personalcreatorverified,reasonrecordedintasktmp/five-teacher-fp16-gradient-config-stop.log andlocalstopaudit. Thisisintentionalstopbecauseanativeoptimizerfailurewasreproduced, notanauxiliaryvalidatorverdict. Launcher9758exit0doesnotestablishbusinesssuccess. Initialconfiguration remainsnoverifiedsuccessfulupdates.
- Threecorrectednativepreflights65494exit0 (long/five/formal) andthreepersonaldryruns87938exit0. Preserveinitialdynamicloss_scale65536andnativeunscale/overflowhandling;changeonlyexistingDDPgrad_reduce_in_fp32=true. Configartifactsretained,includingformal-200updates-fp16-fp32grad-{submit-args,resolved-config}.json. Noformalrunsubmitted.
- Newexactjobs: longFP16 `exp-1113-010116-136916`,24H800,nativeTP2/PP2/CP1,8x65536,1update,launcher45655/monitor87633; five-teacherFP16 `exp-1113-010116-581403`,56H800,32x8192,5updates,nativesave/exit/resume,launcher23922/monitor99707. BothlastobservedStarting,creatori-fengyicheng,currentlocalNFS100.96.128.195:/data/ycfeng:/data/ycfeng. Keepbothlaunchersalive;monitorinterval45s/duration10800s. Completeformal200/256/65536stillwaitsactualFP16training/save-resumeverification.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原模型与五领域身份已核验 |
| Q | PASS | 合同和执行授权已记录 |
| B | PASS | correctedFP16原生配置与dryrun通过 |
| C | PASS | 真实maskedgradient参照完成 |
| D | TODO | 梯度/norm对照已测；actualFP16training重跑待验收 |
| E | TODO | corrected五教师save/resume已提交；formal未提交 |
| F | TODO | 最终200updates验收未完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次恢复checkpoint至梯度/TP4控制及原生optimizer配置修正｜**当前前沿：** D actualFP16training与独立短save/resume

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | head诊断完成即提交；复用native配置和monitor |
| D2 细节反复重试 | CLEAR | placement恢复已Succeeded；新clipping错误以nativebufferCPUloop定位 |
| D3 支线问题修补 | CLEAR | 原生optimizer错误直接阻止Dsuccessfulupdate |
| D4 可并行任务停滞 | CLEAR | longFP16和五教师恢复同时重提，两个launcher保持alive |
| D5 超范围镀金 | CLEAR | 无核心接口或schema扩展 |
| D6 分析空转 | CLEAR | gradient/norm控制给出数值结论；FP32grad配置执行已提交 |
| D7 路线违背 | CLEAR | 阈值、原模型和200/256/65536不变；未跳过D提交formal |

**文档检查：** S5 plan/progress旧状态未包含新Failed/intentionalStopped和替代job；本轮已同步。S1–S4/S6无明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：先同步已测事实再继续。

**文档更新（先执行）：**
1. plan/progress/test_report：记录完整head/TP4测量、nativeclippingRCA和correctedjobIDs（已直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：actualFP16longrun → finite loss/grad、successfulupdate、0skipped、sameweightparity。
2. 并行：五教师FP16shortsave/resume → 全教师覆盖、模型/optimizer/scheduler/data-plane恢复和参数变化。
3. E：D通过后formal200/256/65536 → 实际successfulupdates和postcheckpointreplay。

## 4. 最值得关注的 3 点

1. Headmaskedgradient16/16PASS；TP4异常norm有完整tokens匹配的对照解释。
2. ActualFP16optimizer的梯度buffer配置错误已CPU定位；GPU新run尚待结果。
3. 两个correctedtrainingjobs已提交；formal200updates仍未提交。


## 2026-11-13 01:16 HKT: continuation evidence and PP4 memory control

- Completed head terminal checks: `exp-1113-005256-166805` Succeeded; launcher 50016 and monitor 33434 exit0. Both exact owner-bound replica containers exited0, restart0, finished at 2026-11-12T17:01:24Z. Evidence: `evidence/final-head-gradient-reference-3-terminal-objects.json`.
- Corrected long FP16 job `exp-1113-010116-136916` Failed, launcher45655 exit1/monitor87633 exit0. Native traceback reaches `_chunked_distributed_student_teacher_backward` at model_utils.py:2677, allocating 18.55 GiB with18.33 GiB free; allocated41.98/reserved-unallocated15.38 GiB. No successful train step or accepted train dump; TB rollout/train_steps=0. Retained finished/runtime/command files identify the exact job, Python3.13.14/Torch2.13.0+cu130/H800 and full65536 scenario.
- RCA collector uses exact RJob/replica UID and incident17:07–17:10Z. Six primary queries successful, two audit queries time out; initiating manual actor/reason/request time remain UNKNOWN. This is a native PyTorch allocation failure; no auxiliary validator stopped healthy training. Raw RCA remains `tasktmp/rca-exp-1113-010116-136916`; selected objects/index and metrics are in evidence/.
- Ranked memory hypotheses before the next GPU attempt: H1 FP32 DDP buffers leave insufficient capacity for dense full-vocabulary gradient; reduce student per-stage decoder activations by PP2→PP4, retaining8 student GPUs and all other numerical settings. H2 allocator fragmentation contributes (15.38 GiB reserved but unallocated); H3 pipeline retained activations dominate. The PP4 result can reject sufficiency of the selected capacity correction. GPU allocator behavior cannot be reproduced faithfully on this CPU host; retain the actual required65536 scenario as the red-capable loop rather than claim a CPU memory proof.
- New sparse PP2 FP16 recipe passes exact resolved-config equivalence to the prepared formal config: removed11 redundant CLI overrides with zero config differences. Existing `test_shipped_config` PASS:1 test,5 subtests,3 not-applicable skips,435 deselected,12.71s. Raw commands/logs: tasktmp/check_fp16_recipe_equivalence.py,fp16-recipe-equivalence.log,fp16-recipe-shipped-config.log. Formal temporary arguments now consume that equivalent recipe; no formal job submitted.
- PP4 native config preflight passed in the established Megatron actor environment (initial_loss_scale65536, FP32 reduction). First CPU invocation omitted existing cuda-compat LD_LIBRARY_PATH and raised libcuda.so.1 OSError; preserved in long-training-fp16-pp4-preflight.log. Restore the documented CUDA compatibility environment only; no install or core edit. Successful evidence: long-training-fp16-pp4-preflight-2.log. Personal dry-run is in progress.
- Five-teacher FP16 `exp-1113-010116-581403` remains Starting; launcher23922/monitor99707 alive. Exact scheduling records show a seven-node gang waiting for resources. No duplicate submitted; collect business artifacts only after workers start.
- check-progress: A/Q/B/C PASS;D/E/F TODO. D1–D7 CLEAR;S5 stale state corrected in plan/review/report. Next route: actual PP4 long training ‖ existing five-teacher save/resume, then formal200 after D acceptance.

- PP4 native preflight session38973 exit0; personal dry-run22360 exit0. Local personal submission launcher99023 created `exp-1113-011736-657255` once,24H800/codesign; monitor92222 interval45s/duration10800s. Output root: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-long-training-fp16-pp4/exp-1113-011736-657255`. Keep launcher alive. Formal recipe preflight also passed, but D still prevents formal submission.


## 2026-11-13 01:28 HKT: native FP16 overflow finalizer repair

- Inspected native optimizer.py step(): a detected overflow returns `(False, None, None)`. The split worker converted native `grad_norm=None` with `torch.tensor([grad_norm])`, raising `RuntimeError: Could not infer dtype of NoneType` before returning its successful-update flag. Existing TQ result aggregation and controller aggregate_step_metrics already accept None and omit an unavailable norm.
- Tight CPU fault loop modifies the existing first-update lifecycle test to use the native overflow return tuple. RED session65122 exit1:1 failed/1 passed/105 deselected,19.97s; exact failure at worker line2627. Ranked causes before repair: H1 local tensor conversion rejects the standard native overflow result; H2 aggregation rejects optional norms; H3 optimizer returns an unsupported state. Native source and existing aggregation reject H2/H3; the red loop isolates H1.
- Minimal production fix preserves None in the split worker result instead of constructing a tensor from it. The native loss, scaler, optimizer, scheduler, update-success flag and normal finite-norm path retain their existing behavior. Existing tests now cover native overflow returns, success-flag reduction across replicas, unavailable norm omission, and a subsequent successful update on the same worker.
- GREEN session23431 exit0:35 TestFinish checks pass (72 deselected,17.36s) and39 TQ/controller checks pass (10.97s). Continuous next-update verification session28336 exit0:1 affected check passed after adding the follow-up assertions. Scoped Ruff0.9.9 check/format and git diff --check PASS.
- Signed scoped commit `86ad0be771bc0bc9d302a4de6df3e7afd15abb8c`: only megatron_policy_worker.py and the two existing test modules. No push. GPU validation of the ordinary training path remains pending; both exact GPU jobs were still Starting during this fix and will load the locally mounted corrected source when workers start.
- Prepared existing checkpoint state/optimizer/finite readers for `exp-1113-010116-581403`, using unique local output names. Parameter-delta check compares saved step1 to step5 after native resume, avoiding a comparison that mistakes BF16→FP16 source conversion for optimizer change. These read-only checks are prepared, not executed without checkpoints.
- check-progress after verified code step: A/Q/B/C PASS;D/E/F TODO. D1–D7 CLEAR. The fault test covers a reachable native FP16 result that can interrupt D; no unrelated worker path or new interface is added. S5 HEAD/source state is synchronized below. Next: actual PP4 long training ‖ existing five-teacher save/resume, then formal200 only after D acceptance.


## 2026-11-13 01:34 HKT: continuation and actual GPU runtime recovery

- Recovered exact existing launchers99023/23922 and monitors92222/99707; all alive. No duplicate or formal submission. Long PP4 `exp-1113-011736-657255` Running as of17:31:17Z, three active replicas; five-teacher `exp-1113-010116-581403` Running as of17:31:08Z, seven active replicas. Both are personal creator i-fengyicheng, codesign, non-preemptible, local NFS100.96.128.195:/data/ycfeng:/data/ycfeng.
- Read-only collectors `python3 tasktmp/collect_long_fp16_pp4.py` and `collect_fp16_five_teacher_2.py` exit0. PP4 has three runtime receipts and initial rollout trace, no accepted update yet. Five-teacher has all seven runtime receipts and command, no accepted update yet. All ten nodes report8 NVIDIA H800, Python3.13.14 in the mounted nemo-mopd-py313 environment, Torch2.13.0+cu130/CUDA13.0. Selected evidence: `evidence/final-fp16-active-training-runtime.json`. Source boundary: recorded local HEAD86ad0be7 and NFS identity; already-started processes did not emit code-file digests.
- Adapted the existing corrected-long collector and analyzer to the exact PP4 job/root under tasktmp; no new persistent harness. Checkpoint analyzers remain pending until actual checkpoint completion.
- Inspected ordinary native SingleController: `_train_steps` is the iteration count, and worker `optimizer_update_successful` is logged separately. Formal acceptance will sum the latter, retaining every skipped iteration. Never infer200successfulupdates from current_step200. Final horizon/config will be decided from actual diagnostic results, retaining the approved200successful target.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原 student、五教师和数据身份已记录 |
| Q | PASS | 200 次成功更新和执行授权已确认 |
| B | PASS | 全词表与原生 FP16 配置预检通过 |
| C | PASS | 三方 parity 和实际 masked-objective gradient 参照完成 |
| D | TODO | PP4 长上下文和五教师训练实际验收待完成 |
| E | TODO | 短 save/resume 已 Running；正式训练未提交 |
| F | TODO | 完成报告和 archive 待 D/E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次 overflow 修复 checkpoint 至恢复现有作业状态｜**当前前沿：** D 长上下文训练与独立五教师 save/resume

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用 collector，只更新现有 PP4 job 身份 |
| D2 细节反复重试 | CLEAR | PP4 是已记录容量假设；等待原作业实际结果 |
| D3 支线问题修补 | CLEAR | 本轮工作均用于 D 训练和 E 恢复验收 |
| D4 可并行任务停滞 | CLEAR | 两个独立作业同时 Running |
| D5 超范围镀金 | CLEAR | 无额外接口或新机制 |
| D6 分析空转 | CLEAR | 核对实际 GPU runtime；没有重做已 PASS 数值检查 |
| D7 路线违背 | CLEAR | 原模型、全词表和 200/256/65536 合同保留；formal 等待 D |

**文档检查：** S5 plan 状态此前为 Starting，与新 Running 回执不符；已同步。S1–S4/S6 未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan.md 当前状态：更新两项 Running 与实际 runtime 回执（已直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：PP4 实际 65536-token update → 完成标志：有效 tokens、成功更新、finite loss/grad 和完整 parity。
2. 并行：五教师 save/resume → 完成标志：教师覆盖、成功更新、参数变化与 native 状态连续性。
3. E：D 通过后提交正式 run → 完成标志：累计 200 次成功更新及保存/恢复和最终数值验收。

## 4. 最值得关注的 3 点

1. 两个原有作业已 Running，无重复提交。
2. PP4 的容量假设须由实际 full-vocabulary backward 结果检验。
3. 200 次成功更新必须单独计数；原生 iteration 计数不能替代。

## 2026-11-13: cleanup authorization and continuation recovery

- User explicitly authorized Codex cleanup of current-task large, smoke and unused artifacts; requirements/plan now preserve that decision. The MOPD contract remains unfinished.
- Local UV cleanup completed: 94 rebuildable temporary/download/index directories deleted. Available-space delta 1,606,836,224 bytes (1.50 GiB), free-inode delta 29,678. The host already had approximately 113 GiB available before this cleanup; do not attribute that earlier recovery to this task. Installed environments and hardlinked archive-v0 retained. Evidence: evidence/cleanup_2026-11-13_local_cache.json.
- Session63544 completed: 18 archives deleted only after comparing every regular member's size and complete bytes to extracted copies. Logical archive bytes removed 403,136,717; exact extracted copies retained. Evidence: evidence/cleanup_2026-11-13_duplicate_archives.json.
- Session12823 exited1 with effective RED: `uv run --no-sync pytest --noconftest tests/unit/models/megatron/test_megatron_setup.py::TestCreateMegatronConfigGradOverflow -m mcore -q -s --tb=short` yields the actual DDP fatal Inf symptom (1 failed, stop after first failure, 26.05s). This CPU seam uses native bucket checks and scaler and does not need autouse Ray setup. Earlier red-4 exit124 is a fixture startup timeout, not a test verdict. Exact log: tasktmp/native-fp16-overflow-regression-red-5.log.
- Recovered PP4 exp-1113-011736-657255 PASS: 1 successful/0 skipped, 8 sequences, 202 response tokens, 6 sequences length65536 and two65511/65507. Loss1.2871393728630665e-8, grad0.0004988232394680381; independent native scorer/vLLM MAE0.007388088723072912, p950.02813922/p990.05262076/max0.21245265. Three containers exit0/restart0/platformSucceeded. Limitation: independent scorer, not actual training-forward capture; group mapping in the temporary analyzer needs correction.
- Recovered five-teacher exp-1113-010116-581403 FAIL: 0 verified successful updates, no saved checkpoint/resume. Native first traceback is fatal local bucket Inf before data-parallel communication. Production setup.py is not yet edited. Initiating external stop actor/reason/time remain UNKNOWN; later task-fail cleanup is not the initiating reason.
- Cloud inventory is273.6957GiB logical. Preparing explicit obsolete-cache/three-smoke-weight candidates; preserve final-five-teacher-fp16-resume and final-long-training-fp16-pp4 caches, source models, data, runtime/terminal/log/metric/sample evidence. Deletion will use local personal 0GPU Python RJobBackend, not remote SSH mutation.

- Cleanup worker exp-1113-094259-836505 completed all41 listed directories at2026-11-13T01:43:45Z; platformSucceeded and local launcher emitted `MOPD_CLEANUP_TERMINAL ... succeeded`. Logical bytes removed263,072,167,440 (245.005GiB); remaining task cloud logical bytes30,806,354,945 (28.690GiB). This is logical removal, not measured backend physical quota reclaim. Preserved18 small native checkpoint metadata files with exact byte verification, both required FP16 caches, all6 original source models and train.jsonl. Receipt/manifest/job states in evidence/cleanup_2026-11-13_cloud_*.json.
- Also removed7 obsolete downloaded wheels/interrupted download files after verifying no downloader references the path: logical1,486,795,398bytes; actual available-space delta1,044,639,744bytes. Sparse interrupted files explain the difference. Installed environments, UV archive hardlinks and the still-running task Ray service remain intact. Evidence: evidence/cleanup_2026-11-13_local_wheels.json.
- Effective overflowGREEN4/4 passed21.52s. Actual OptimizerConfig is created once and reused; DDP fatal checks are disabled only when its fp16=True. Native unscale/scaler/global MAX nonfinite check remains responsible for skip/backoff; BF16 and FP32 preserve fatal detection. No manual loss-scale lowering or blanket nonfinite suppression. Affected setup regressions are running.
- Terminal replica inspection initially used the Kubernetes Pod listing endpoint and received403; this is an inspection endpoint limitation, not cleanup failure. Correct to the existing custom Replica API as used by other task collectors; no job retry. Initial lint invocation found no ruff in the base environment; use the existing installed tool environment.

## 2026-11-13: cleanup milestone and overflow repair checkpoint

- Cleanup receipt and the owner-bound custom Replica API confirm41 deletions,18 retained metadata files, native platform/replicaSucceeded and containerCompleted/exit0/restart0. The cleanup launcher23841 is terminal. Report:test_report_2026-11-13_artifact_cleanup.md.
- Native original prepared formal policy reproduction passed with Inf then NaN, loss_scale65536→32768 and finite gradient0.25. Effective regression4/4 and affected config14/14 PASS; Ruff0.9.9 check/format and diff whitespace PASS. Report:test_report_2026-11-13_native_fp16_overflow.md.
- Scoped code staging succeeded, but commit repeatedly encountered index.lock. Initial empty lock was apparently8hours old; /proc and lsof found no holder, and one authorized stale-lock deletion was recorded. Subsequent lock reappearance proves unresolved concurrent index access or visibility; stop deleting or blindly retrying locks. Preserve staged files, advance GPU preflight independently, then retry the ordinary commit only after the lock is absent. No index/HEAD rewrite.
- Next diagnostic uses unchanged original student and five teachers, full vocabulary,56H800/codesign, studentTP2/PP4/CP1, nativeFP16/FP32gradientreduction and initial_loss_scale65536. Keep16 iterations for scaler backoff and at least5 successful updates; count optimizer_update_successful separately. Initial and resume subprocesses share the same16-iteration scheduler horizon. All five teacher routes and32 fresh trajectories/8192 ceiling remain diagnostic inputs; formal remains200successful/256/65536. Reuse the retained conversion root via native fully_parallel_load rather than creating another six-model cache.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原 student、五教师和数据身份已核验 |
| Q | PASS | 用户确认合同，并明确授权当前任务清理 |
| B | PASS | 原生 FP16 配置和运行环境已验证 |
| C | PASS | 全词表和真实 masked gradient 参照已测 |
| 清理 | PASS | 云端245.005GiB 删除、保留清单和原生终态已核验 |
| D | TODO | PP4 长上下文已通过，五教师修复后的 GPU 验收待完成 |
| E | TODO | 正式200次成功更新尚未提交 |
| F | TODO | scoped commit 暂受 Git index 访问影响，最终报告待 D/E |

## 2. 偏离检测

**结论：** 已偏离：命中 D2。
**检测窗口：** 上次01:34 checkpoint 至清理完成和 overflow CPU GREEN｜**当前前沿：** D 五教师 native save/resume

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 清理由用户直接授权；已完成即退出清理工作 |
| D2 细节反复重试 | HIT | Git index.lock 连续阻止 commit，重复操作尚未产生提交 |
| D3 支线问题修补 | CLEAR | overflow 直接阻止原生五教师更新 |
| D4 可并行任务停滞 | CLEAR | 清理 worker 与 CPU 回归并行；GPU preflight 已继续 |
| D5 超范围镀金 | CLEAR | 只改 DDP 与原生 scaler 的检查顺序，无接口/schema扩展 |
| D6 分析空转 | CLEAR | 明确 RED/GREEN 与 cleanup 终态已得到 |
| D7 路线违背 | CLEAR | 原模型、全词表和200/256/65536合同保留，正式 run 未提前提交 |

**文档检查：** S5 plan 中旧 Running/待清理状态与终态、CPU修复矛盾，已直接同步；S1–S4/S6无明显缺陷。
**归因：** D2 执行偏离。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. plan/progress/report：同步清理完成、CPU修复通过、Git提交暂未完成（可直接执行，已完成）。

**需停止的偏离行为：**
- D2：停止盲重试或再次删除 index.lock；独立推进 native GPU preflight，确认锁消失后完成 scoped commit。

**接下去的路线：**
1. D：运行新五教师 PP4 save/resume → 至少5次成功更新、全教师覆盖、参数变化和 native 状态连续性。
2. 并行：F scoped commit → 仅提交验证过的两个代码文件与必要任务记录。
3. E：D通过后正式200/256/65536 → 累计200次实际成功更新和最终 checkpoint 验收。

## 4. 最值得关注的 3 点

1. 清理已完成，未删源模型或后续复测所需缓存。
2. DDP/scaler 修复已 CPU 验证，实际五教师 GPU 路径仍须复测。
3. 200次成功更新尚未完成，不能把 iteration 数当作成功更新数。

- Ordinary scoped commit completed after checking the index lock was absent:20173f71bd5f48623e36953f0b778e942e1bfc96, signed-off, only setup.py and test_megatron_setup.py. No push. Stop Git-lock investigation; the code milestone is committed.
- PP4 per-group analysis corrected using metadata.prompt_idx and the exact8-row repeated source input file. Source groups covermath/code/stem/logic/if; overall MAE0.0073880887230729116 is unchanged apart from displayed final decimal. Group MAEs:math0.00594953/code0.00834931/stem0.00843527/logic0.00584120/if0.00724255; all<=0.02. Observed tail max0.21245265 remains reported. This is source-group coverage in a same-weight control, not five different teacher weight coverage. Selected evidence:final-long-training-fp16-pp4-analysis.json. A temporary archive assertion compared differently rounded decimal floats exactly and failed; changing only that assertion to1e-15 absolute comparison confirmed unchanged numerical content. No healthy training was canceled or gated by it.
- First PP4 preflight caught obsolete +initial_loss_scale override after switching to the sparse FP16 recipe where the key already exists. Removed that redundant override, preserving inherited65536. Second preflight passed native MasterConfig, dtype/teacher/save-resume validations,FP32 reduction and DDP/scaler ordering. Updated personal dry-run is pending terminal collection; no bad configuration submitted toGPU.

- Final corrected personal dry-run92348 exit0. Local launcher80676 submittedexactlyonce:exp-1113-095147-620678,56H800/codesign,7nodes. Watcher83030 uses45s interval/10800s duration. Output root remains final-five-teacher-fp16-resume to reuse retained initial conversion caches. Keep launcher80676 alive through queue/run; native five-teacher results are pending.
- check-progress correction applied:Git lock investigation ended after ordinary scoped commit20173f71; cleanup milestone remainsPASS and D is actual five-teacher GPU save/resume. No further cache cleanup or rejected BF16 diagnostics are needed. Formal200/256/65536 remains unsubmitted.

## 2026-11-13 09:54 HKT: durable handoff after completed cleanup

- Cleanup milestone complete. Documentation commit3a33baaa is signed-off and unpushed; source repair20173f71 remains the production code used by the new native workload. Logs remain in persistent evidence despite the repository's *.log ignore rule; no force staging of raw logs.
- exp-1113-095147-620678 isRunning as of01:53:34Z,7active replicas,creatori-fengyicheng,nonpreemptiblecodesign,localNFS100.96.128.195:/data/ycfeng:/data/ycfeng. All7 runtime receipts show8H800 each,Python3.13.14/Torch2.13.0+cu130/CUDA13.0 and preserved platformNCCL values. Selected receipts:final-five-teacher-fp16-resume-pp4-{latest-runtime-status,worker-runtime}.json. No successful update/checkpoint/resume verdict yet.
- Live launcher80676 and monitor83030 must remain alive. Native collector:tasktmp/collect_fp16_five_teacher_pp4.py; current extracted root:tasktmp/final-five-teacher-fp16-resume-pp4-current-inputs. Selected state/optimizer/finite/parameter-delta readers have been adapted to this exact job with pp4 filenames; execute only after corresponding checkpoints exist. Last checkpoint may be step15 or16 depending on native final-save behavior; select the actual final checkpoint rather than require an invented path. All auxiliary validators remain non-blocking on this56GPU run.
- Remaining route:1) observe native overflow/scaler and at least5 successful updates, all five teacher routes, parameter changes and state continuity after save/resume;2) promote the verified PP4 setting to the formal200successful/256fresh/65536run, summing optimizer_update_successful rather than iteration count;3) complete post-update/final-checkpoint parity, finite metric/parameter analysis and final archive. Formal run remains unsubmitted.
- Requirements reread:cleanup authorization does not cancel the original MOPD contract. No completed overall summary.md is written. No unresolved user decision blocks the next native verification.

## 2026-11-13 10:43 HKT: exact PP4 checkpoint failure recovered

- exp-1113-095147-620678 is Failed (finishTime 02:03:18Z), personal owner UID d09dee5f-b749-41a9-b178-f2da02cc686a. Launcher 80676 and watcher 83030 have exited; no live launcher remains for that terminal run. No formal 200-update job has been submitted.
- Retained incident logs identify the first native business traceback at 02:03:02Z: MegatronPolicyWorker.save_checkpoint -> Bridge generate_state_dict -> DistributedOptimizer.sharded_state_dict -> get_parameter_state_dp_reshardable -> _get_main_param_and_optimizer_states -> KeyError('master_param'). The earlier DDP repair allowed native scaled-overflow skip (`grad_norm=None`); first requested checkpoint now exposes lazy Adam state. No complete checkpoint or resume is verified.
- Actual FusedAdam plus native parameter-state checkpoint reader, on CPU with only CUDA allocation redirected, reproduces exactly the missing master_param without offload. Command: actor-environment uv run --no-sync python tasktmp/probe_native_preupdate_checkpoint.py; traceback at distrib_optimizer.py:1233 retained in tasktmp/native-preupdate-checkpoint-red.log. This is an effective native RED, not a mocked exception.
- Ranked hypotheses before correction: H1 no successful Adam step left lazy state empty (initialization should create exact master weights and zero moments without advancing counters); H2 offload removed state (CPU no-offload reproduction rules it out); H3 native initialization callback is incompatible with current TE initialize_state signature (verify the pinned implementation before use).
- Exact-job RCA collector completed Events, incident logs and existing DeepScan reads; all-operation audits timed out and remain failed queries, not empty findings. Job fail policy stopTimestamp 02:03:17Z and launcher Job Stop 02:03:19Z follow the business exception. Initiating external stop actor/reason/request time remain UNKNOWN. Evidence: tasktmp/rca-exp-1113-095147-620678 and tasktmp/five-teacher-pp4-business-traceback.txt.
- Continue at D with a minimal checkpoint-state integration repair; preserve the tested native scaler, source models, full-vocabulary objective, and formal 200/256/65536 contract. Do not resubmit until the new checkpoint RED/GREEN and native preflight pass.

## 2026-11-13 10:49 HKT: pre-update checkpoint repair verified and committed

- Effective checkpoint regression RED reproduces native KeyError(master_param), not a simulated error: 1 failed in21.12s. The NeMo-RL checkpoint entry now initializes empty distributed FusedAdam state with its native method and the installed two-argument API. Existing Adam states and counters remain unchanged; no optimizer step or loss-scale override is added. No third-party checkout was edited.
- GREEN8/8 in21.01s: FP16/BF16 and chained/direct checkpoint cases4/4, existing overflow cases4/4. Original CPU reproduction is GREEN with exact FP32 master[1.25,-0.5] and zero moments. The first GREEN test-fixture assertion assumed remainder storage while not enabling it; corrected the fixture only, preserving production code. Native recipe preflight, personal dry-run and scoped Ruff0.9.9 check/format PASS.
- Signed scoped code commit ac26e5a8 contains setup.py, MegatronPolicyWorker save integration and the native regression test. No push. Report: test_report_2026-11-13_preupdate_checkpoint.md.
- Current codesign node projection at02:48:23Z:123 machines,72 usable GPUs and137 preemptible GPUs; it is not a personal quota or guaranteed H800 placement verdict. A temporary numeric aggregation first rejected string-valued quantities; int conversion corrected that auxiliary read without affecting training.
- Submitted once after GREEN:exp-1113-104851-054395,56H800,unchangedTP2PP4/FP16/FP32gradreduction/fullvocab/five original teachers,16attempt horizon,32fresh/8192diagnostic. Launcher73501 stays alive; watcher66988 monitors the exact job. Cloud root remains final-five-teacher-fp16-resume to reuse retained model caches. Actual successful updates, checkpoint serialization and resume are pending; formal200/256/65536remains unsubmitted.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原模型和数据身份已核验 |
| Q | PASS | 用户确认执行合同 |
| B | PASS | 原生环境及配置预检通过 |
| C | PASS | 全词表参照和三方 parity 已测 |
| 清理 | PASS | 授权清理及保留清单验收完成 |
| D | TODO | 最新 checkpoint 缺陷已 CPU 修复；新 GPU 回归待验收 |
| E | TODO | 正式200成功更新尚未提交 |
| F | TODO | 代码修复已提交，最终 archive 待 D/E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 09:54交接至10:49checkpoint修复提交｜**当前前沿：** D五教师原生保存恢复

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | collector只补齐当前失败所缺的业务证据 |
| D2 细节反复重试 | CLEAR | exact-job traceback和native RED先于修复；无盲目重提 |
| D3 支线问题修补 | CLEAR | master_param缺陷直接阻止D保存恢复 |
| D4 可并行任务停滞 | CLEAR | preflight与CPU回归并行，E依赖真实D验收 |
| D5 超范围镀金 | CLEAR | 一个checkpoint集成函数，无第三方改动或schema扩展 |
| D6 分析空转 | CLEAR | 已取得native RED/GREEN、scoped commit并提交GPU回归 |
| D7 路线违背 | CLEAR | 原五教师、全词表和200/256/65536合同保留 |

**文档检查：** S5 plan旧Running与实际Failed矛盾；已同步Failed、首个业务异常和新回归提交。S1–S4/S6无明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/report：记录旧job终态、CPU验证、新job和活跃launcher（可直接执行，已完成）。

**需停止的偏离行为：** 无

**接下去的路线：**
1. D：核验exp-1113-104851-054395 → 至少5成功更新、全教师覆盖、保存恢复连续性和数值检查。
2. E：D通过后正式200/256/65536 → 计实际成功更新，保持native optimizer/scheduler/data-plane恢复。
3. F：归档实际结果 → 最终报告、scoped commits和English summary。

## 4. 最值得关注的 3 点

1. 原生checkpoint缺陷已有效RED/GREEN验证并提交，GPU验收仍未完成。
2. 新作业只提交一次，保持launcher活跃并观察exact-job。
3. 正式200次成功更新尚未完成，不能以iteration计数替代。

## 2026-11-13 10:54 HKT: formal PP4 preparation while the diagnostic queues

- Current diagnostic exp-1113-104851-054395 has creator i-fengyicheng, UID6696f121-e116-4125-b93b-02b78cfae2a2 and current-machine NFS100.96.128.195:/data/ycfeng:/data/ycfeng. Queue reason at02:50:46Z explicitly says request56H800 vs remaining54; rank2. Keep launcher73501 and watcher66988 alive, inspect the existing queued job; no duplicate submission or unrelated-job stop. Native acceptance remains pending.
- New reusable final recipe: examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml. It uses the already inspected CP1/FP32-head/FP32-RoPE/FA4 path and original student/five teachers, PP4/FP16/FP32gradreduction/fullvocab. It sets explicit native scheduler.lr_decay_iters=200; LR=min_LR=1e-6 and weight_decay=0.01 are constant, with zero warmup.
- Native MasterConfig/preflight and personal launcher dry-run PASS for200attempt preparation,256fresh trajectories,65536ceiling/max_new_tokens,fullvocabandoriginal65536loss_scale. These are CPU preparation results, not200successfulupdates. Since native controller iterations and scheduler ticks advance on overflow skips, formal acceptance sums optimizer_update_successful and checks native optimizer state independently; after a completed native segment, resume only the measured missing successful-update count while preserving the explicit constant scheduler settings. Do not equate max_num_steps=200 with acceptance.
- before-steptron-run loaded for formal preparation. Existing inspected lifecycle metadata and failure/replacement paths are unchanged (design.md and263localchecks in earlier report); approved256group batch, stable prompt_idx/sample_id, weight version, timeout/error references and native data-plane checkpoints remain required. No new profiling or recording framework was added. Formal submission still waits for D's actual checkpoint/resume and health acceptance.
- Existing state/optimizer/finite/delta/parity readers now target exactjob104851 and pp4-2inputs. Run them only against actual existing checkpoints; final native save path must be observed rather than assumed step16. Readers remain auxiliary soft gates on the56GPUjob.

## 2026-11-13 11:02 HKT: recovered diagnostic is running

- Existing job exp-1113-104851-054395 was inspected, not resubmitted. It transitioned from Pending to Starting after queue approval02:57:52Z and is now Running with7active replicas. The7runtime receipts report56H800 on7distinct nodes, Python3.13.14/Torch2.13.0+cu130/CUDA13.0, personal creator i-fengyicheng and current-machine NFS100.96.128.195:/data/ycfeng:/data/ycfeng. Platform NCCL settings remain retained. Evidence: evidence/final-five-teacher-fp16-resume-pp4-2-runtime-status.json. Launcher73501/PID4037301 and watcher66988/PID4045417 remain alive.
- The native initial phase has reserved all five teacher clusters. Actual updates/checkpoint serialization/resume are still pending; no process-startup result is counted as D acceptance. Read-only collection confirmed7runtime records and the native command, with no train/checkpoint records yet.
- Formal output root now reuses final-five-teacher-fp16-resume. Each RJob has its own job-named directory, while the native run_on_rjob.py mechanism uses the common parent's initial-megatron-checkpoints cache. This prevents recreating six initial FP16 conversions after the approved cleanup. The200/256/65536recipe and scheduler remain unchanged. Personal --dry-run PASS; no formal job was created. Exact final arguments: evidence/formal-200updates-fp16-pp4-submit-args.json.
- Native TQPolicy/SingleController success-metric path is present: megatron_policy_worker.py -> tq_policy.py -> single_controller_utils/utils.py -> train/optimizer_update_successful. LMPolicy's different aggregate path is not used by this selected runtime. Formal acceptance will also check native Adam group steps rather than infer updates from scheduler/controller iterations.

## 2026-11-13 11:09 HKT: independent mounted-volume evidence read

- The previously working SSH collector timed out after65s, and a separate read-only `ssh ... id -un` timed out after15s. No evidence result is inferred from those timeouts. Exact-job personal API monitoring still shows Running; all five teacher inference routes now appear in native logs. The training allocation was not stopped or altered.
- The source-reading gap is addressed by a separate0GPU/4CPU/16GB codesign personal RJob, exp-1113-110906-631985, submitted once from this machine. Launcher40536 remains alive. It uses the same required cloud volume and local NFS parent and only copies current-job small metadata/metrics/sample records into tasktmp/final-five-teacher-fp16-resume-pp4-2-current-inputs. It does not control the56GPUallocation. Script: tasktmp/collect_fp16_pp4_mounted.py; submitter: tasktmp/submit_mounted_evidence_reader.py; request: tasktmp/mounted-evidence-reader-request.json.
- The reader is a temporary operational substitute for unavailable SSH reads, not a training gate. Its isolated I/O/JSON errors are logged as unverified; native training proceeds. It polls every30s for at most1hour, exits after the business finished marker, and does not copy large checkpoint tensors. Actual reader execution and current business checkpoint/results remain pending.

## 2026-11-13 11:11 HKT: native early checkpoint completes

- Initial native subprocess in104851 completed its real overflow-skipped `step_1` checkpoint and emitted `MOPD_CHECKPOINT_PHASE resume`; no missing-master exception recurred. The206entry metadata contains model/master/moment/scaler records, and controller/data-plane/dataloader/replacement/recovery records are present. Initial native success flag0, loss0.1536673754 and absent gradient norm agree with overflow skip. Resume initialization remains in progress; D is not complete.
- Mounted reader110906 executes successfully:20smallfiles,32sampleidentities,zero reported collection failures. CPU native-state reader exit0 and initial scorer/vLLM parity analysis exit0/PASS:32samples/4096tokens, MAE0.00138685256≤0.02, all five group means≤0.002116, no mask/version issues. This is scorer/vLLM evidence, not actual training-forward instrumentation.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原模型与数据身份已核验 |
| Q | PASS | 用户已确认执行合同 |
| B | PASS | 原生环境和配置预检通过 |
| C | PASS | 已记录全词表、梯度与三方参照 |
| 清理 | PASS | 授权清理及保留清单已验收 |
| D | TODO | 实际 early save通过；resume与≥5成功更新待验收 |
| E | TODO | 正式200成功更新尚未提交 |
| F | TODO | 最终报告和English archive待D/E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 10:49 checkpoint修复验收到11:11 GPU早期保存｜**当前前沿：** D原生resume与训练健康

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 0GPU reader仅补足SSH读取失败，已取得D所需证据 |
| D2 细节反复重试 | CLEAR | ac26e5a8的native RED/GREEN已获得真实GPU早期保存结果 |
| D3 支线问题修补 | CLEAR | checkpoint保存和证据读取均直接服务D |
| D4 可并行任务停滞 | CLEAR | GPU训练、独立读取及CPU状态/parity分析同时推进；E依赖D |
| D5 超范围镀金 | CLEAR | 未扩展生产接口或config schema |
| D6 分析空转 | CLEAR | 从排队推进到early save和初始parity实测 |
| D7 路线违背 | CLEAR | 保留原五教师、全词表和正式200/256/65536合同 |

**文档检查：** S1–S6均未发现明显缺陷；当前GPU结果已追加到preupdate checkpoint报告。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. D：完成同一104851的native resume → ≥5成功更新、五教师覆盖、finite状态、参数变化和post-update parity。
2. 并行：在训练继续时读取实际checkpoint/state和样本metrics；辅助读取错误不得停止训练。
3. E：D通过后正式200/256/65536 → 实际success flag和native Adam steps共同验收，并native resume补足缺额。
4. F：完成最终指标/一致性报告和signed scoped commits → 所有合同项有实际结果，写English summary。

## 4. 最值得关注的 3 点

1. 当前GPU已越过原来的缺陷位置，首次原生保存返回并进入resume。
2. 初始scorer/vLLM MAE0.001387通过；≥5实际更新和恢复连续性仍待验证。
3. 正式200成功更新尚未执行，不能以controller iteration或batch skipped metric替代。

## 2026-11-13 11:30 HKT: D completed; formal E is the frontier

- Personal exact-job inspection confirms104851 Succeeded03:22:50Z;all7containers exit0/restart0. Readers110906 and112131 also Succeeded/exit0. Native complete marker and finalstep_16 are retained. Existing launchers/monitor ended after workload success; do not submit them again.
-16attempts yield11explained native FP16 overflow skips then5successful updates. Successfulloss0.134770–0.233819;preclipgrad1.979079–3.665760;nativeclip1.0. All8nativeAdam records step5;scaler65536/growth0/hysteresis1 ->64/growth5/hysteresis-9;constant scheduler ticks32->512. Firstoverflow hysteresis explains unchangedstep1scale;no counter restart inferred.
-512unique samples/prompts/all5groups;complete65536tokenparity MAE0.0013328945,p950.006328,p990.012342,max0.168925,maxsampleMAE0.003638. Weightlag0. This is scorer/vLLM;actualtrainingforward not inserted in this diagnostic. Final1591tensorchunks/7,108,614,656elements finite;3196selectedmodelelements changed. Report and exact evidence paths updated;Daccepted.
-before-steptron-run rechecked against unchanged selected lifecycle code and prior263localchecks:sampleID/prompt locator,run/step/weight version,timeout/error records,replacement/group validity and native data-plane recovery remain active. Formalreader errors are soft gates;no new blanket exception handling or hardgate. Approved256batch,65536ceiling,fullvocab and native scheduler200 remain unchanged. Formalrun pending immediate submission.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原模型与数据身份已核验 |
| Q | PASS | 用户确认合同 |
| B | PASS | 原生环境与recipe预检通过 |
| C | PASS | 全词表loss/gradient与三方参照已记录 |
| D | PASS |104851原生保存恢复、5成功更新、finite状态和parity通过 |
| 清理 | PASS | 授权清理与保留清单已验收 |
| E | TODO | 正式200成功更新待提交 |
| F | TODO | 最终验收和Englisharchive依赖E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：**11:11早期保存至11:30终态验收｜**当前前沿：** E正式200更新

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 两个reader取得缺失终态/finite证据后结束 |
| D2 细节反复重试 | CLEAR | 同一GPU回归跑完，无重复提交 |
| D3 支线问题修补 | CLEAR | 读取和nativecheckpoint均是D验收项 |
| D4 可并行任务停滞 | CLEAR | 训练、读取及formal预检同时完成 |
| D5 超范围镀金 | CLEAR | 未增加生产接口或configschema |
| D6 分析空转 | CLEAR | D由TODO转PASS，E准备提交 |
| D7 路线违背 | CLEAR | 保留原5教师、全词表及200/256/65536 |

**文档检查：** S5plan顶部仍写GPUresume待验收，已据最新终态更新；S1–S4/S6无明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1.plan/progress/report：以实际终态、完整parity和nativeAdam计数更新D（可直接执行；已完成）。

**需停止的偏离行为：** 无

**接下去的路线：**
1.E：提交preparedPP4formal →200实际成功更新；native初期保存恢复；必要时按缺额resume。
2.并行：formalmetrics/coverage/parity与GPU监控；辅助错误不停止训练。
3.F：最终checkpoint和trainingforward一致性验收 → scopedcommits、最终报告和Englishsummary。

## 4. 最值得关注的 3 点

1.D真实保存恢复和5成功更新通过；所有replica正常退出。
2.FP16早期11次overflow由native降scale解释；formal必须计实际成功更新。
3.正式200/256/65536尚未提交，下一步立即进入E。

## 2026-11-13 11:30 HKT: E submitted once

- Formaljob exp-1113-112953-201895 created11:29:54HKT through current-machine personal PythonRJobBackend.56H800/codesign,7nodes,TP2/PP4/CP1,FP16,FP32gradreduction,original5teachers,full151936vocab,256fresh/65536ceiling. Native two-phase save/resume entrypoint,constant scheduler200,LR1e-6/WD0.01. Output-root reuses initial conversion caches;newjob owns an independent directory. Launcher session92989 stays alive; do not resubmit.
- D evidence is archived and JSON/diff-whitespace checks pass. Ordinary scopedgitadd failed with existingindex.lock; no lock inspection/removal or history operation performed. Documentation commit is deferred whileE proceeds.

## 2026-11-13 11:48 HKT: Formal workers started; evidence and final checkpoint replay prepared

- E in-progress: existing exp-1113-112953-201895 moved Pending -> Starting at03:44:57Z -> Running by03:46:02Z. Personal exact-job inspection reports7active replicas, creator i-fengyicheng, codesign/non-preemptible, source100.96.128.195:/data/ycfeng:/data/ycfeng. All7 runtime records report8H800 each, Python3.13.14 from /data/ycfeng/envs/nemo-mopd-py313/bin/python, Torch2.13.0+cu130, CUDA13, architecture9.0. PlatformNCCL values are retained. Native initial phase has started; no completed training update is yet observed.
- Independent0GPU reader exp-1113-113211-147826 remains active (launcher74102). It copies selected small runtime/events/native state and reduces atomically published training dumps in place on the mounted personal cloud volume; it does not duplicate large trajectories locally. Currentreceipt: 8files, 0steps, 0errors. Auxiliary failures mark evidence unverified and cannot stop or block healthy56GPUtraining.
- The step reducer was checked against all16 archived short-regression dumps; sample counts and exact per-step MAE/p50/p95/p99/max match independent complete evidence within1e-14. Full-run overall quantiles require joint response-token errors; averaging step quantiles is invalid.
- Prepared finalcheckpoint entrypoint tests/manual/mopd/replay_checkpoint_logprobs.py uses native Policy weights loading, existing non-colocated vLLM and create_weight_synchronizer; it compares fresh generated responses from one selected prompt in each original group after native checkpoint refit. CPU --help/import, Ruff check/format, actual archived prompt/sample selection and2node MasterConfig preparation passed. Evidence: evidence/checkpoint-replay-input-preflight.json. This is CPU preparation only; GPU execution and actual finalcheckpoint remain pending. It measures native scorer/vLLM, not actualtrainingforward.
- Ran `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_fp16_pp4_metrics.py`. Observed0attempts/0successes/0skips before first completed batch. Later acceptance deduplicates successflags by step and latest walltime and cross-checks final nativeAdam steps. Initialsegment's200attempt horizon does not satisfy200successful updates by itself.
- Preserved existing launchers92989/74102 and monitor34971; no duplicate submission, cleanup or Gitlock changes. Next: native early save/resume and firstbatchmetrics ->200actualupdates, measured-deficitresume if needed ->finalnativecheckpoint/refit and evidence reconciliation.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student、五教师和数据身份已核验 |
| Q | PASS | 用户确认200成功更新合同和执行授权 |
| B | PASS | 持久环境、全词表recipe和原生准备已验收 |
| C | PASS | 全词表loss/gradient与三方参照已记录 |
| D | PASS | 五教师5成功更新、save/resume、finite状态和parity通过 |
| 清理 | PASS | 已授权清理完成，不重复清理共享目录 |
| E | TODO | 正式56H800已Running；200成功更新和最终checkpoint待完成 |
| F | TODO | 完整报告、signed scopedcommits和Englisharchive待E验收 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 11:30D验收至2026-11-13 11:49 HKT｜**当前前沿：** E正式训练和最终checkpoint验证

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | reader与replay准备对应E必需证据；开始执行后不再扩大工具范围 |
| D2 细节反复重试 | CLEAR | 保持同一正式RJob及launcher，没有重复提交 |
| D3 支线问题修补 | CLEAR | runtime、successflags和checkpoint均在E验收路径 |
| D4 可并行任务停滞 | CLEAR | 正式训练、独立reader、finalcheckpoint准备同时推进 |
| D5 超范围镀金 | CLEAR | 未扩展生产接口、数据格式或训练范围 |
| D6 分析空转 | CLEAR | 已实际从Pending进入56H800Running和nativeinitialphase |
| D7 路线违背 | CLEAR | 原5教师、151936全词表、200成功/256fresh/65536保留 |

**文档检查：** S1–S4/S6未发现明显缺陷；S5早期表仍写E待提交，而最新记录及真实状态已提交并Running，已更新plan与本checkpoint。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan.md顶部和progress当前状态：更新E已Running、原生initialphase开始；200成功更新仍TODO（可直接执行；已完成）。

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：保持正式allocation ->原生earlysave/resume、200实际成功更新；以successflag与Adamstate双重计数，必要时native恢复补足。
2. 并行：每步loss/grad/coverage/parity与56H800监控；reader错误仅标unverified。
3. E/F：最终nativecheckpoint完整finite/state检查和refit响应一致性 ->完成最终测试报告、scopedcommits与Englishsummary。

## 4. 最值得关注的 3 点

1. 56H800已Running，运行环境和本机source验证通过，不能把startup当作训练完成。
2. 200attempts和scheduler ticks不是200成功更新；nativeoverflow必须单独计数并补足。
3. finalcheckpoint GPU replay尚未执行；保留scorer/vLLM与actualtrainingforward的证据边界。

- E runtime supplement: all7exact replicaobjects owned by formalUID report Running/restart0 and identical imageID `hub.i.basemind.com/stepmind/megatron-step@sha256:d29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177`. Selectedstatusobjects archived. Training dataloader capacity is55000slots/256 ->214batches/epoch; native setup clamps max_num_steps to max_num_epochs*len(dataloader). If measured-deficit resume horizon exceeds214, override `grpo.max_num_epochs=null` in that resume segment to preserve the200success contract. This is a routine execution setting; no active recipe edit or job restart is needed.


## 2026-11-13 12:10 HKT: Formal first-batch failure, initiating cause recovered

- E FAILED/in-progress repair. Exact formal RJob exp-1113-112953-201895/UID0d70a68d-1ac2-44ef-9cab-f65be352490c finished Failed04:03:58Z. Initial256rollouts completed; no recorded optimizer attempt/success, loss, gradnorm, native earlycheckpoint or resume. Launcher92989 exited1; do not reuse its Running snapshot.
- Local personal failure collector session91110 completed exit1 because both all-operation audit queries timed out45s. Objects, live/historical Events, incident logs and existingDeepScan reads succeeded; no existingDeepScan. Query failures are not empty observations. Native head exit1/restart0 and job fail-policy stop are later consequences.
- Locally submitted0GPUcodesign reader exp-1113-120706-340132/launcher52832 read221archivedRayfiles and selected19. Rank3 raylet establishes initiating CPU-memory pressure at12:03:08HKT and kills12workers12:03:38. Nodeused192257081344B >191260262400Bthreshold; total201326592000B. Final186.44/187.50GiB;8student processes18.99–20.74GiB each;objectstore36324B. Owner exitNODE_OUT_OF_MEMORY explains downstreamteacher deaths and shutdownerrors. This does not establish CUDA OOM, kernel OOMKilled, or preemption.
- Native data path inspected: tq_policy.py::train_microbatches_from_meta -> shard_meta_for_dp -> worker_mixin.py::train_microbatch_presharded/_fetch -> codec.pad_batch -> MegatronPolicyWorker.train_microbatch. DP1/TP2/PP4 fetches the256-row training slice on all8studentranks before backend internalmicrobatches; jaggedteacherpayload is naturallypadded to batchmaximum. No unbounded queue or objectstore allocation is evidenced. The first formalbatch is the already-run red reproduction; reducing256/65536 to minimize it would not verify the formal acceptance. The GPU-dependent original command is the decisive feedback loop; static inspection alone cannot prove a memory fix.
- Ranked hypotheses before rerun: H1 configured192GB/node is below normal batchpayload+8rank CPUpeak (increasing only node memory should pass this boundary); H2 a leak will keep growing across updates even with a larger request (check successiveupdate health); H3 an independentGPUallocation error may appear after this CPUboundary (requires its own traceback and diagnosis). Native Raymemorymonitor remains enabled; no threshold/killpolicy workaround.
- H1 test prepared: only memory request192->768GB/node. At256*65536*1536*2bytes, one denselymaterialized FP16hiddenpayload is48GiB/rank;8ranks384GiB before temporary copies/runtime.768GB request provides750GiB total rather than187.5GiB. Original5teachers/full151936/200success/256fresh/65536/FP16/TP2PP4CP1 and native save/resume remain unchanged. Need dryrun then actualrun to accept correction.
- Evidence: evidence/formal-200updates-fp16-pp4-failure-rca.json; rawcollector andRaylogs remain in tasktmp. No additional cleanup or modelchanges.


## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A/Q/B/C/D/清理 | PASS | 已有验收记录保持有效 |
| E | FAILED | 首个正式batch在CPUmemorypressure下失败，修正后重跑待验收 |
| F | TODO | 依赖200成功更新及最终checkpoint |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次11:49checkpoint至12:12HKT｜**当前前沿：** E首批失败修复及正式重跑

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 一个0GPUreader补齐首个fatal证据即结束 |
| D2 细节反复重试 | CLEAR | 本次只进行H1已记录的单变量memory修正 |
| D3 支线问题修补 | CLEAR | CPUmemorypressure直接阻断E |
| D4 可并行任务停滞 | CLEAR | RCA读取和formal资源预检并行，finalcheckpoint尚不存在 |
| D5 超范围镀金 | CLEAR | 无生产schema/接口或训练目标扩展 |
| D6 分析空转 | CLEAR | 已取得原始Raykill记录并实际重新提交 |
| D7 路线违背 | CLEAR | 原始200成功/256fresh/65536/5teachers/fullvocab保留 |

**文档检查：** S5plan/progress/report原Running快照已过期；本次已改为Failed和修复中。S1–S4/S6无明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/report：记录实际Failed、0updates、CPUOOMinitiatingevidence和H1资源重跑（可直接执行；已完成）。

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：保持修正memory的正式launcher -> 首批nativeupdate/earlysave/resume ->200实际成功更新。
2. 并行：独立reader取得metrics/coverage/parity；辅助错误不停止训练。
3. E/F：finalcheckpoint状态/finite/refit与actualforward证据 -> scopedcommits及completionarchive。

## 4. 最值得关注的 3 点

1. 第一正式jobFailed、0成功更新；短诊断PASS不变。
2. RayCPU内存峰值已确认；768GB配置的实际验收尚待训练。
3. 新作业exp-1113-121031-998701/launcher91314已提交一次；不要重复提交。


## 2026-11-13 12:26 HKT: Resumed E monitoring and PP4 observer regression

- E in-progress: existing memory-corrected formal RJob exp-1113-121031-998701 remains Pending; latest 04:24:36Z snapshot reports queue ranking2 and H800 remaining46 for request56. Launcher91314, independent mounted reader launcher81924 and monitor81236 are alive. Reader receipt has0files/0steps/0errors because the formal allocation has not started. No resubmission or repeat cleanup.
- Final actual-training-forward observation preparation uncovered a diagnostic-only PP4 rank-selection defect. Original callback required globalrank0, while full logits exist on the last pipeline stage. Changed recording to TP0/CP0 per DP replica and included globalrank. Filenames now include hostname plusPID to avoid identicalPIDs on different nodes appending to one file. Loss, optimizer and native numerical calls are unchanged; the observer is not loaded by the formaljob.
- Valid CPU RED uses the actual OPD_FULL-enabled callback from HEAD: rank6/TP0/CP0 fails expected1 versus actual0. Valid GREEN after formatting passes5rankcases (rank6,0,7,14,12), plus cross-node equalPID output isolation. Environment is the mounted MegatronactorPython3.13environment through uv run --no-sync, with the existing CUDAcompat library directory. Exact commands/environment and selected results: evidence/pp4-forward-observation-regression.json. Early fixture/environment-error logs are not numerical validation.
- Pinned Ruff0.9.9 check and format --check PASS for native_forward_worker.py and replay_checkpoint_logprobs.py. Final checkpoint replay and actualtrainingforward GPU validation remain pending; CPU observation correctness does not establish trained-model parity.
- Next mainline: keep the already-submitted formalallocation -> firstbatch memory/update acceptance and earlynative save/resume ->200actualsuccessfulupdates ->finalnativecheckpoint/refit/actualforward.


## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | student、五教师和数据身份已有验收 |
| Q | PASS | 200实际成功更新合同已获确认 |
| B | PASS | 持久环境与全词表recipe已有验收 |
| C | PASS | full-loss/gradient参照已有验收 |
| D | PASS | 五教师native save/resume短诊断通过 |
| 清理 | PASS | 授权清理完成，保留后续必需资产 |
| E | TODO | CPU内存修正的正式job排队；200成功更新和最终checkpoint待验收 |
| F | TODO | 完整报告与archive依赖E；已通过的observer CPU修正可提交 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 12:12checkpoint至2026-11-13 12:26HKT｜**当前前沿：** E既有正式job及finalactualforward准备

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | observer修改解决PP4漏记录，回归通过后停止扩展 |
| D2 细节反复重试 | CLEAR | 仅一个memory修正job；未重复提交或再跑D |
| D3 支线问题修补 | CLEAR | observer记录是E actualtrainingforward所需证据 |
| D4 可并行任务停滞 | CLEAR | GPUjob排队时完成CPU准备；最终权重尚不存在 |
| D5 超范围镀金 | CLEAR | 未扩展训练目标或生产接口 |
| D6 分析空转 | CLEAR | 已实际完成rank漏记录RED/GREEN并保持正式allocation |
| D7 路线违背 | CLEAR | 5教师/151936/256fresh/65536/200success保持 |

**文档检查：** S5：plan顶部和旧E表只描述失败job，最新修正jobPending与observer回归需补记；本次已更新。S1–S4/S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/report：补记修正job精确身份、Pending与launcher，以及observerCPU证据边界（可直接执行；已完成）。

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：保持现有launcher ->实际首batch、nativeearlysave/resume ->200成功flags与Adamstate一致。
2. 并行：独立reader观察loss/grad/coverage/parity，辅助失败不控制训练。
3. E/F：最终checkpointfinite/state、nativecheckpoint refit和actualforward ->最终报告和archive。

## 4. 最值得关注的 3 点

1. 768GB修正仍排队；不能把CPU配置检查记为GPU修复PASS。
2. observerCPU漏记录回归已PASS，最终actualtrainingforwardGPU结果仍待执行。
3. 已完成的清理和D诊断不重复；保持91314/81924存活。


## 2026-11-13 12:28 HKT: Preserve verified checkpoint replay preparation

- Completed CPU observer step is signedcommit322c5a5f; no unrelatedfile staged. Preserve nativecheckpoint refit entrypoint tests/manual/mopd/replay_checkpoint_logprobs.py and its existing CPUinput preflight as a separate preparedcode step. The entrypoint loads nativeweights, uses existing create_weight_synchronizer transport, and measures fresh scorer/vLLM response logprobs from one sourceprompt pergroup. Allfive archivedlocators are exercised by CPUselectionpreflight; final formalweights and GPUrefit remain pending. No newtests or broadenedGPUchecks were run.
- Exact personalAPI at04:27:31Z: correctedjob Pending, queue ranking2, H800 remaining55/request56; reader exp-1113-121235-452668 Running. Formalserializedresources are7replicas,8GPU/32CPU/750Gi each, both RDMAdefaults, and localNFS100.96.128.195:/data/ycfeng:/data/ycfeng. Rawfilteredreceipt: /data/ycfeng/tmp/nemo-mopd-20261112/formal-memory-status-20261113-1230.json.
- Currentmetrics read through mountedbaseenvironment uv run --no-sync reports0attempts/0successes/0skips and no loss/gradnorm yet. This is queuedstatus, not anothertrainingfailure. Existing formal andreaderlaunchers remain necessary.


## 2026-11-13 12:40 HKT: Final native training-forward preparation complete on CPU

- E independent preparation completed. The finalcheckpoint replay now collects allfivefresh groupresponses and native scorer outputs before calling Policy.train once on that identicalbatch. MegatronPolicyWorker.train sets model.train() and runs native forward/backward; the existing observer records its prepared next-token logprobs. One diagnostic ClippedPGLossFn optimizer attempt happens after the observedforward and is not counted toward formal200successes. This compares the trainedstudent forward independently of the fullOPD objective; formal full-vocabulary OPD loss/gradient verification remains on its own path. No formaljobinstrumentation, objective or configuration changed.
- Reused test-local workerselection from native_forward_entrypoint.py; preserved OPD_FULL objectivegradient observation. LOGPROB observations reuse the actual prepared native logprobs and add no recomputation. Final report matches exact tokenIDs/lengths, applies next-token offset and originalresponsemask, reports per-sample/overall train-minus-vLLM and train-minus-scorer errors, and marks missing/invalid records UNVERIFIED. Sourcegroup metadata is joined through report.sample_index. The independent finalreplay uses2nodes/16H800, not the healthy56GPUformalallocation.
- New reusableCPUcheck tests/manual/mopd/check_checkpoint_training_forward.py PASS. Synthetic4response-token fixture generationMAE expected0.0175/actual0.01750004291534424, scorerMAE expected0.015/actual0.01500004529953003. Native diagnosticloss expected2.5175/actual2.5174999237060547; each valid-token derivative is-0.25, maskedprompt/padding gradients0, maxgradienterror0. Missingrecords and invalidmasks returnUNVERIFIED. Mountedactorregistration PASS. Python3.13.14/Torch2.13.0+cu130/CUDA13.0. Exactcommands/environment/evidence: evidence/checkpoint-training-forward-cpu-preflight.json. These are syntheticCPUvalues, not H800parity measurements.
- Affected OPD_FULL PP4 rank/file regression, archivedfiveprompt selection/configuration, CLI --help and pinnedRuff0.9.9 check/format allPASS after the extension. Stop extending this tooling; GPUexecution awaits finalformalcheckpoint. Prepared code60dc291b preceded the actualtrainingforward extension, which will receive its own scoped signedcommit.
- Existing correctedformalRJob exp-1113-121031-998701 remains queued. At04:36:38Z: Pending/ranking2/H800 remaining46/request56;0recordedattempts/updates. Readerexp-1113-121235-452668 remains alive with0files/0steps/0errors. Keep launchers91314/81924 andmonitor81236; no duplicate submission or cleanup.


## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始模型与数据身份已验收 |
| Q | PASS | 200成功更新合同与授权已确认 |
| B | PASS | 环境与全词表recipe已验收 |
| C | PASS | full-loss/gradient参照已验收 |
| D | PASS | 五教师native save/resume短诊断已通过 |
| 清理 | PASS | 授权清理完成 |
| E | TODO | 正式job排队；finalactualforward CPU准备完成，GPU结果待执行 |
| F | TODO | 部分代码步骤已签署提交，overallreport和archive仍依赖E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 12:26checkpoint至2026-11-13 12:40HKT｜**当前前沿：** E正式训练与最终checkpoint验证

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 补齐缺失的实际trainingforward观测；CPU准备通过后停止扩展 |
| D2 细节反复重试 | CLEAR | 同一memory修正job持续排队，无重复提交和参数试换 |
| D3 支线问题修补 | CLEAR | 全部修改对应E finalcheckpoint一致性验收 |
| D4 可并行任务停滞 | CLEAR | 排队时完成CPU准备和signedcommits；final权重尚不存在 |
| D5 超范围镀金 | CLEAR | 未扩展核心接口、生产目标和数据格式 |
| D6 分析空转 | CLEAR | 新入口实际完成CPU执行、对齐/梯度检查和观测回归 |
| D7 路线违背 | CLEAR | 正式5teacher/151936/256/65536/200success保留；诊断单独计数 |

**文档检查：** S1–S6均未发现明显缺陷；plan/report已记录新actualforward准备和GPU证据边界。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：保持91314/81924 ->首batch内存和nativeupdate验收 ->earlysave/resume ->200实际成功更新与Adamstate一致。
2. 并行：独立reader记录loss/grad/coverage/parity，auxiliary错误只标unverified。
3. E/F：最终checkpointfinite/state与独立16H800三方replay ->正式测试报告、signedcommits和Englisharchive。

## 4. 最值得关注的 3 点

1. 正式job仍Pending/0成功更新；不能将CPU准备记为训练完成。
2. finalactualtrainingforward入口CPU准备已完成，不再扩展工具。
3. 16GPU诊断与正式200update计数分开；保留全部既有launcher及清理证据。

- Git save iteration: first scoped git add encountered an index.lock; read-only inspection found it already released. HEAD remains60dc291b and no stagedfiles were present. Retry the scopedadd; no lock deletion or unrelatedGitoperation performed.


## 2026-11-13 12:41 HKT: Current handoff state

- HEADa0101fc5, branchmopd-sys. This continuation added signedcommits322c5a5f,60dc291b,a0101fc5. Four affectedmanualPythonfiles and their preparedCPUevidence are committed; unrelatedlayer_output_worker.py/session/andpriorothermodifications remain untouched.
- Correctedformalexp-1113-121031-998701/UID2003e09a-e896-4ecd-ac29-453b3dcaf327: latest04:40:40Z Pending/ranking2/47H800remaining for56 requested. Formal0attempts/0successfulupdates, no loss/gradnorm or checkpoint yet. Launcher91314 andmountedreaderlauncher81924 were polledalive at12:41HKT; monitor81236 continuespublishingexactjob snapshots. Preserve allthree.
- D andauthorizedcleanup remaincomplete; do not repeat either. E remainsunfinished: unchanged5teachers/151936fullvocab/256fresh/65536ceiling,750Gi/node memorycorrectionawaitingactualfirstbatch, thennativeearlysave/resume and200successfulupdates. Countdeduplicatedtrain/optimizer_update_successful andcross-checknativeAdamstep. If extensionhorizonexceeds214batches/epoch, native resumegrpo.max_num_epochs=null withconstantlr_decay_iters200,LR=minLR1e-6,WD0.01.
- Final16H800checkpointreplayCPUpreparationnowcomplete. tests/manual/mopd/replay_checkpoint_logprobs.py includesactualnative trainingforward plusfreshscorer/vLLM comparisons; tests/manual/mopd/check_checkpoint_training_forward.py isthereusableCPUregression. AllcurrentrequiredCPUchecks/Ruff PASS. ActualGPUcheckpointrefit/forwardvalidationhasnotrun. Useactualfinalcheckpointandactualnativeformal-dump/data-identities; independentdiagnosticPGattemptisnotformalOPDprogress.
- ResumeatE: monitorexistingformaljob ->7runtimes/replicas/imageID/resources andfirstbatchhealth/memory ->earlycheckpoint/resume ->200actualsuccesses ->finalcheckpointfinite/state and16GPUthree-wayreplay ->fulltestreportandEnglishsummaryafterrequirementsreread. Noextraobservertoolingneededwhilequeued.


## 2026-11-13: Recover second formal failure

- Corrected formal RJob exp-1113-121031-998701 (UID 2003e09a-e896-4ecd-ac29-453b3dcaf327) is Failed, finishTime 2026-11-13T06:32:26Z. Launcher cleanup stop at 14:32:33 HKT followed the failure; launchers 91314/81924 and monitor 81236 have ended. The mounted reader exp-1113-121235-452668 Succeeded; receipt reports 19 copied files, zero summarized optimizer steps, no reader failures.
- Executed /data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_fp16_pp4_memory_metrics.py. TensorBoard contains zero optimizer attempt records, zero successful updates, and no train loss/grad metrics. This establishes no recorded update, not merely old queue progress.
- Native exception in exact head replica log is CUDA OOM during Megatron backward, actor rank7 on 10.191.9.69: allocation18.55GiB, free18.38GiB, Torch allocated42.97GiB/reserved-unallocated14.64GiB, GPU capacity79.19GiB. This is distinct from the prior Ray CPU-memory kill. Full incident/platform collection and checkpoint absence confirmation remain in-progress; initiating platform stop actor/reason/time remain UNKNOWN until inspected.
- E remains unfinished. Recover exact first-batch token shapes, inspect the existing full-loss allocation path, test an evidence-based memory correction, then run the original five teachers/151936 full vocabulary/256 fresh trajectories/65536 ceiling. Do not repeat completed D or cleanup.


## 2026-11-13: Packed backward allocation diagnosis and CPU repair

- E hypotheses recorded before the code change: full-logit gradient allocation, tensors retained over multiple microbatches, allocator fragmentation. Real wrapper RED regression reproduced three full-sized slice-gradient allocations (111024bytes/37008bytes logits) with already-correct numerical loss/gradient. Replace independent narrow calls with one torch.split to concatenate gradients once; preserve the shorter fused-linear-logprob path.
- GREEN118 CPU checks PASS after formatting (14.92s), including routing, masks, padding, exact reverse-KL loss/gradient and allocation regression. GPU formal backward remains pending. Report: test_report_2026-11-13_packed_loss_memory.md.
- Exact failed formal batch has256 samples/2157723totaltokens/length303–20185, allfivegroups and2086520response-loss tokens; scorer/vLLM MAE0.0013139912592 PASSmean, max0.22152394 reported. This is an unpublished batch and0successfuloptimizerupdates; it is not formal training acceptance. Cloud has no checkpoint.
- Latest input-preparation requirement is enforced before reallocation: 0GPU CPU RJob exp-1113-150603-564709, local personal Python submission, validates every source safetensor for integrity/finite values and reproduces the exact prepared training/probe datasets. Native publisher READY evidence is not available/reused. Launcher56785 must remain alive; training submission waits for readiness. Startup checks are cheap nonblocking auxiliary evidence.


## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student、五教师和输入身份已记录 |
| Q | PASS | 用户已确认200次成功更新合同 |
| B | PASS | 既有输入与环境准备完成；新要求的CPU完整复验由E执行 |
| C | PASS | 全词表 loss/gradient 参照已验收 |
| D | PASS | 原始65536诊断和五教师 native save/resume 已通过 |
| 授权清理 | PASS | 删除回执和保留证据已验收 |
| E | TODO | 第二次正式作业 CUDA OOM；CPU 修复通过，0-GPU 输入复验进行中 |
| F | TODO | 最终200-success证据、GPU replay 和 English archive 尚未完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次12:40HKT checkpoint 至本次恢复｜**当前前沿：** E 正式第一批 backward 与输入 readiness

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | packed-loss RED/118GREEN直接验证正式OOM修复；CPU完整输入复验来自最新用户要求 |
| D2 细节反复重试 | CLEAR | 新CUDA失败与旧CPU失败分开取证；一次确定的梯度分配假设、一项修复 |
| D3 支线问题修补 | CLEAR | 所有新修改服务E训练启动和正式backward |
| D4 可并行任务停滞 | CLEAR | CPU输入复验同时进行RCA/回归/记录；GPU启动依赖CPUready |
| D5 超范围镀金 | CLEAR | 保留原始模型、五教师、全词表、256fresh、65536与200success合同 |
| D6 分析空转 | CLEAR | 实际复现111024/37008bytes额外分配，产生CPU验证的代码修复 |
| D7 路线违背 | CLEAR | 不重复D、清理和已拒绝BF16/TP4路线 |

**文档检查：** S5旧plan与report仍引用已退出launcher和Pending状态；本次actualFailed/0update证据替换当前状态。S1–S4/S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan.md/current status和preupdate report：明确第二次正式CUDA失败、相关launcher已结束、输入CPU复验及新修复待GPU验证（可直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：0GPU完整输入校验PASS，cheap identity检查和本机dry-run完成 → 才能提交正式GPU复验。
2. 并行：平台RCA收尾 ‖ scoped code commit；停止扩展已能验证的loss工具。
3. E：实际正式firstbackward、earlysave/resume、200成功更新 → final finite/state和16H800三方replay。
4. F：完成实际报告、signed commits、requirements reread和English archive。

## 4. 最值得关注的 3 点

1. 正式训练仍0成功更新；CPU118PASS不能替代正式GPUbackward。
2. 新CPU资产复验没有占用GPU，满足最新启动前输入准备规则。
3. 无checkpoint可续跑本次failed作业；新正式作业从原始student启动并保留nativeearlysave/resume。


## 2026-11-13 15:15 HKT: CPU readiness and formal submission preparation

- E full CPU prerequisites PASS: exp-1113-150603-564709 Succeeded at07:10:15Z; launcher56785 exit0. Six original models,2034tensors/10662528000finiteelements, train55000/probe320exactrows; hashes and native identity evidence recorded. No source tensors or data modified and no native READY evidence reused.
- Prepared-input checker now compares directory file sets plus recorded size/mtime. Submission wraps the auxiliary invocation so interpreter/import failures also remain soft. Missing-report native CLI execution printsUNVERIFIED/exit0. Current-identity check and changed-record negative control run in0GPU exp-1113-151346-746393/launcher56642; await actual PASS before allocating training GPUs.
- Exact native formal configuration and personal dry-run PASS. Same original6paths/5teachers/151936/TP2PP4CP1/256fresh/65536;32CPU/768GBpernode/7nodes/56H800. Setmax_num_epochs=null to permit native horizon extension after overflow skips while keeping constant schedulerhorizon200. Initial baseenv preflight lackedMegatron; installedactorenv passes. CachedRuff0.9.9lint/formatpasses; noGPUenvironmentinstall.
- HeadRCAcollectionended: businessOOM06:31:11Z precedesheadexit1/Error06:31:15Z,fail-policy stop06:32:24Z andfinish06:32:26Z. HeadEvents/logs/DeepScanqueriespassed;head2auditqueriesReadTimeout. PreviouslycompletedactorRJobauditsremaincomplete. Initiatingmanualactor/reason/request-timeUNKNOWN. No further platform investigation blocksE.
- Report:test_report_2026-11-13_formal_input_readiness.md. Frontier remains actualformalGPUbackward -> earlynativecheckpoint/resume ->200successfulupdates -> finalstate/finite/refit/actualforward. Do notrepeatD orcleanup.


## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student和五教师身份已记录 |
| Q | PASS | 200实际成功更新合同已确认 |
| B | PASS | 输入准备已通过新CPU完整复验 |
| C | PASS | full-loss和gradient参照已验收 |
| D | PASS | 原始long-context和native短训练已验收 |
| 清理 | PASS | 用户授权清理已完成 |
| E | TODO | CPU修复与输入复验通过，正式GPUbackward待复验 |
| F | TODO | 完整结果、最终checkpoint和archive依赖E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次packed-loss checkpoint至2026-11-13 15:16HKT｜**当前前沿：** E当前输入状态与正式GPUbackward

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 新输入检查满足GPU前置要求，通过后停止扩展 |
| D2 细节反复重试 | CLEAR | 没有重复提交GPU；CPU环境误选已定位并更正 |
| D3 支线问题修补 | CLEAR | 修改均对应E readiness和辅助检查软门 |
| D4 可并行任务停滞 | CLEAR | CPU作业运行时完成配置、提交验证和RCA收尾 |
| D5 超范围镀金 | CLEAR | 保留原始正式训练合同，没有新增核心机制 |
| D6 分析空转 | CLEAR | 输入复验实际PASS，后续正式提交配置已具体化 |
| D7 路线违背 | CLEAR | GPU提交仍等待CPU当前输入检查；没有缩减batch/vocab/length |

**文档检查：** S5旧plan/report把已完成的完整CPU复验记为pending，本轮已更新。S1–S4/S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/report当前状态：完整CPU复验Succeeded/PASS，当前cheap check身份和正式提交准备（可直接执行，已更新）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：cheap check实际PASS -> signed scopedcommit -> 正式原始配置GPU提交 -> first backward与earlysave/resume。
2. 并行：现有mounted reader记录loss/grad、sample/group、parity；辅助错误保持nonblocking。
3. E/F：200成功flags与Adamstate核对 -> 最终state/finite/三方parity -> 完整报告与Englisharchive。

## 4. 最值得关注的 3 点

1. 正式训练仍0成功更新，GPU修复不能由CPU结果推断。
2. 六模型全量扫描和完整数据重建已通过，不再重复昂贵准备。
3. 下一步执行正式训练，保留D和清理结果，不扩展已准备好的最终replay工具。

- 15:16HKT currentcheapcheckbusinessPASS:41files/all6modelroles; deliberately changed recordedsize rejected. ActualgeneratedsubmissionshellcontinuesafterinjectedImportError;FAILCPUreportrejectedpre-submission. Stopaddinginputchecks; committhenformalGPUrun.


## 2026-11-13 15:19 HKT: Corrected formal GPU allocation queued

- Signed-offcodecommit3a1fe3e2 preservesfullCPUreadiness/preparedinputchecker/submissionsoftgate. Packedbackwardrepairb66b957f remainsGPUunverified.
- Formalexp-1113-151718-896703/UID9f0eb021-6bda-44fa-af2f-5469dd08b02b submittedonce locally aspersonal i-fengyicheng. Sevenreplicas/56H800/32CPU/768GBpernode, bothnativeRDMArequests, currentNFS100.96.128.195:/data/ycfeng:/data/ycfeng. Latest07:19:22ZPending/ranking2/H800remaining36for56requested. NoGPUmodelstartup,0formaloptimizerupdates. Launcher73918 mustremainalive.
- Independent0GPUmountedreaderexp-1113-151748-639492/UID406c9397-4ba1-4edf-8e51-4a211a9c5217/launcher37752 isStarting. Reusedexistingcollectorwithexactnewjob/destination; readerhandleslargecloudsampleswithoutcopyingweights. Read-onlymonitor15510 persistsin tasktmp/formal-split-monitor. Auxiliaryerrorsremainnonblocking.
- Cheap-inputRJobexp-1113-151346-746393nowSucceeded. ItsbusinessreportPASS41files/negativeidentitycontrol. Fullvalidationworkerexit0/Completed/restart0/imageIDd29740f4 andplatformSucceededarearchived.
- UnfinishedE/F: actualfirstbackward/loss/grad,memoryandnativeearlysave/resume ->200deduplicatedsuccessflagsandnativeAdamstep -> finalcheckpointfinite/state/refit/actualtrainingforwardandscorer/vLLMparity -> completedreportandEnglisharchive. DandcleanupremainPASS; donotrerunorresubmitwhilequeued.


## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 模型、五教师和数据身份已验收 |
| Q | PASS | 200成功更新合同已确认 |
| B | PASS | 六模型完整CPU校验和当前文件检查通过 |
| C | PASS | 全词表loss/gradient参照已验收 |
| D | PASS | 五教师native诊断、65536和save/resume已通过 |
| 清理 | PASS | 授权清理已完成 |
| E | TODO | 修复后正式job已提交，等待56卡quota |
| F | TODO | 最终训练证据和archive待E完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 15:16CPUreadiness checkpoint至15:21HKT｜**当前前沿：** E已提交正式作业和独立结果读取

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | readiness通过即停止扩展，现有reader/monitor仅换exactjob |
| D2 细节反复重试 | CLEAR | 正式修复只提交一次，没有因quota不足重提 |
| D3 支线问题修补 | CLEAR | 没有新增不影响训练验收的问题修复 |
| D4 可并行任务停滞 | CLEAR | reader/monitor并行运行；最终checkpoint尚不存在 |
| D5 超范围镀金 | CLEAR | 原始模型和正式目标保持 |
| D6 分析空转 | CLEAR | 已实际提交formaljob并保留launcher |
| D7 路线违背 | CLEAR | CPU准备先完成，32+GPU辅助检查为softgate |

**文档检查：** S1–S6均未发现明显缺陷，当前Pending和0正式成功更新已记录。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：保留73918launcher，等待exactjob运行 -> firstbackward/loss/grad、earlysave/resume验收。
2. 并行：37752reader和15510monitor保留业务和平台结果，辅助错误不控制训练。
3. E/F：200实际成功flags与Adamstep -> 最终checkpointfinite/state/refit和actualforward/scorer/vLLM三方检查 -> 最终报告和Englisharchive。

## 4. 最值得关注的 3 点

1. 07:20:52ZformalPending/ranking2/36H800remaining，0实际成功更新。
2. reader已发布0files/0steps/0failures回执；这不是训练通过。
3. 3a1fe3e2已提交输入准备；b66b957f内存修复仍待GPU验证。

- 15:21HKT continuation state: formal73918, reader37752 andmonitor15510 were polledalive. Cheap-inputlauncher56642 hasexit0. Keep existingformaljobexp-1113-151718-896703; exactcloudrunprefix /mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-151718-896703. Resumebyreadinglatestmonitorstatusandreaderreceipt before anyfurtheraction. NocheckpointorformalmetricsexistwhilePending; GPUfix,200successesandfinalreplayremainunfinished.


## 2026-11-13 15:36 HKT: Resume exact queued formal run

- Recovered requirements, plan, latest reports and actual processes. Direct local personal Python RJobClient reads verify exact formal UID9f0eb021-6bda-44fa-af2f-5469dd08b02b and creator i-fengyicheng; Pending/ranking2/33remaining for56H800 requested. Local NFS source is100.96.128.195:/data/ycfeng:/data/ycfeng. Readerexp-1113-151748-639492 isRunning with the same local source.
- Polled73918formal launcher,37752reader and15510monitor alive. Current reader receipt is0files/0steps/0failures; there is no formal optimizer metric or checkpoint. No new GPU submission, code edit, resource change or cleanup.
- Inspected existing native resume_training.py: initial must-save-by then ordinary resume with unchanged scheduler settings. Inspected prepared final replay_checkpoint_logprobs.py: native checkpoint loading, non-colocated refit, fresh vLLM/scorer and actual model.train() comparison; independent diagnostic attempt does not count as formal MOPD progress.
- Next actions: keep the exact queued job and launcher alive; verify all7replicas/runtime when Running, then repaired backward/loss/grad norm/early checkpoint and resume. Complete200deduplicated successful updates with native Adamstep, then final finite/state/refit/three-way parity and English archive. Preserve completed D and cleanup evidence.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student和五教师身份已验收 |
| Q | PASS | 200成功更新合同已确认 |
| B | PASS | 六模型全量CPU准备和当前身份检查通过 |
| C | PASS | 全词表loss与gradient参照已完成 |
| D | PASS | 五教师短训、65536、native save/resume已有GPU证据 |
| 清理 | PASS | 用户授权范围清理已完成 |
| E | TODO | exact formal job等待quota，0正式成功更新 |
| F | TODO | 最终checkpoint与报告等待E产物 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 15:16CPU准备checkpoint至15:36HKT｜**当前前沿：** E原始正式作业与独立读取

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 准备后未扩展工具；现有reader和monitor持续读取exactjob |
| D2 细节反复重试 | CLEAR | b66b957f具备具体gradient allocation假设和CPU结果；GPU只提交一次 |
| D3 支线问题修补 | CLEAR | 未继续已收尾RCA或清理 |
| D4 可并行任务停滞 | CLEAR | reader与monitor并行；最终检查依赖尚不存在的checkpoint |
| D5 超范围镀金 | CLEAR | 未新增验收合同以外能力 |
| D6 分析空转 | CLEAR | 独立状态读取确认实际排队；不重做完成的准备 |
| D7 路线违背 | CLEAR | 原模型、五教师、151936词表、256batch和65536ceiling保持 |

**文档检查：** S1–S6均未发现明显缺陷；实时排队状态已更新，E/F未被标记完成。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：保持73918及原job → 完成标志：真实backward、正常loss/grad、早期native save/resume。
2. 并行：37752reader ‖ 15510monitor → 完成标志：持续记录业务和平台证据，辅助错误不停止训练。
3. E/F：200成功flags与Adamstep → 最终checkpoint finite/state/refit/三方parity → 最终报告与English summary。

## 4. 最值得关注的 3 点

1. 15:36HKT stillPending，remaining33小于请求56，正式GPU修复尚未验证。
2. 原job与所有长驻launcher存活，保留一次提交，不因排队重复分配。
3. 最终replay已有CPU准备，等待真实checkpoint；不扩展工具或重跑D。


## 2026-11-13 16:11 HKT: Reject unsupported resource replacements and preserve formal run

- **completed:** Current CPU input check `exp-1113-155805-262829` published business PASS at08:02:00Z:41 files, all six model roles, deliberately changed identity rejected, no repeated tensor scan. Personal API verifies Succeeded at08:04:53Z, container exit0/restart0, expected d297 imageID and current local NFS source. Launcher85510 exited0.
- **completed:** Recovered the 8×4 H800 candidate's native CPU preflight PASS. Server admission changed required hard anti-affinity to preferred weight10. Direct GET404 confirms `exp-1113-160428-882591` was an unretained dry-run. Current Lark training manual revision3668 requires8GPU per replica for distributed NVIDIA/RDMA jobs; affinity manual revision261 explains the requested semantics but not the mutation. Reject 8×4 and earlier10×2 candidates without GPU submission.
- **completed:** Tested the ordinary4×8 alternative with native `_build_clusters`. It raises ValueError before Ray initialization because five reserved teacher nodes exceed four total nodes. This is a passing negative control; the candidate is rejected. Existing allocation interfaces were not changed.
- **completed:** Removed only the two unverified `task_anti_affinities` lines from `submit_on_rjob.py`; that file is back to HEAD. No formal stop, resubmit, preemptible operation or new GPU allocation. Retained source assets, FP16 caches and completed D/cleanup evidence remain intact.
- **in-progress:** Formal `exp-1113-151718-896703` remains Pending at08:10:51Z, queue rank2, H800 remaining28 against56 requested. Independent reader `exp-1113-151748-639492` is Running. Sessions73918,37752 and15510 remain alive. Formal updates/loss/grad/checkpoints are still absent.
- **completed:** Selected evidence and exact commands are recorded in `test_report_2026-11-13_formal_placement.md` and its referenced evidence JSON files. The local platform handbook now states the verified multi-node GPU geometry and the limits of this admission result.
- The first local query failed on an incorrect `brainpp` import before any API request; the corrected query uses inspected `brainpp.rjob` and the documented return type. Systemd scope creation required interactive authentication, so the CPU negative control used the existing30GiB address-space limit and60s timeout. No host authentication or service settings were changed.
- **pending:** Actual formal GPU backward, memory/loss/grad health and early native save/resume →200 deduplicated success flags plus Adam step →final checkpoint finite/state/refit and actual training forward/scorer/vLLM parity →final report and English summary. The final replay tooling is already prepared and does not need expansion while no formal checkpoint exists.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student 和五教师身份已验收 |
| Q | PASS | 200次实际成功更新合同已确认 |
| B | PASS | 完整 CPU 准备及当前41文件身份检查、平台终态通过 |
| C | PASS | 全词表 loss/gradient 参照检查通过 |
| D | PASS | 五教师短训、65536 context 和 native save/resume 通过 |
| 清理 | PASS | 授权范围已清理并验证 |
| E | BLOCKED | 原56卡正式作业等待 quota，正式更新为0 |
| F | TODO | 最终 checkpoint 验收及 archive 依赖 E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 15:36 HKT checkpoint 至16:11 HKT｜**当前前沿：** E 正式作业资源及执行

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 候选检查用于判断能否启动 E；取得拒绝证据后停止，没有扩展工具 |
| D2 细节反复重试 | CLEAR | admission 读取和 native4×8负控分别判断放置保证与已有节点分配；没有反复提交 GPU 作业 |
| D3 支线问题修补 | CLEAR | 平台规则直接关系到 E 的资源方案；未重跑完成的 D 和清理 |
| D4 可并行任务停滞 | CLEAR | 原 reader/monitor 持续运行；最终验证缺少正式 checkpoint |
| D5 超范围镀金 | CLEAR | 保留原始模型、词表、batch/context 和200成功更新目标；撤回无效修改 |
| D6 分析空转 | CLEAR | 调查已形成明确决定：否决两个较小方案并保留原56卡队列作业 |
| D7 路线违背 | CLEAR | 未提交未满足前提的 GPU 方案，未重试已失败的 preemptible 路线 |

**文档检查：** S5 `plan.md` 的顶部状态未包含16:04准入失败和资源候选，已在继续执行前更新。S1–S4、S6未发现明显缺陷。
**归因：** 无执行偏离；缺失的资源调查记录已补齐。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. `plan.md` 顶部：补齐候选拒绝、当前 CPU 终态、原 job 状态和 E/F 未完成项（已直接执行，未改变验收范围）。
2. `progress.md` 与本轮测试报告：记录命令、admission 结果、明确资源决定和局限（已直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：保留73918和原56卡 job；启动后验证真实 backward、loss/grad 及早期 native save/resume。
2. 并行：保留37752 reader 和15510 monitor；取得业务及平台证据，辅助错误不停止训练。
3. E/F：200成功 flags 与 Adam step 核对 →最终 finite/state/refit/三方 parity →报告与 English summary。

## 4. 最值得关注的 3 点

1. 当前 quota28小于请求56；正式 GPU 修复及200成功更新均未验收。
2. 32卡候选分别被平台8卡规则和 native五教师节点预留前提否决；没有提交替代作业。
3. 当前输入已通过业务和平台验证，后续直接训练，不重复全量扫描或完成的诊断。


## 2026-11-13 17:31 HKT: Diagnose whole-node teacher over-reservation

- **in-progress E:** Original formal job remains Pending at09:27:40Z, rank2, remaining42 H800 against56 requested; original launcher/reader/monitor remain alive, reader0files/0steps/0failures.
- Inspected native `_non_colocated_teacher_node_count`, `_build_clusters`, `reserve_teacher_clusters` and RayVirtualCluster. With no topology constraint, Ray reserves the configured teacher GPU bundles, not exclusive whole physical nodes. The student allocator nevertheless subtracts one whole node for every teacher logical node. This rejects a GPU-sufficient four-node layout before Ray reservation.
- Hypothesis before modification: calculate the reserved physical GPU capacity from the summed teacher GPU requests when `cluster.segment_size` is absent. Keep dedicated-node counting for topology-constrained placement. Expected regression: five independent2GPU teachers plus8GPU student and8GPU generation fit4×8GPU; existing full-node and topology-constrained cases retain their allocation.
- Gain: remove unused whole-node reservations so the requested real five-teacher workflow can use32H800 under the platform8GPUperreplica rule. No public config schema, worker-group interface, loss, model identity, batch, context or scheduler change. GPU placement/memory/numerical acceptance remains pending.

- 17:37HKT CPU Ray diagnosis: clearing inherited OTEL_* removed gRPC metadata abort and registered all four simulated Ray nodes. The60s run then timed out while packaging the entire repository. Inspected installed Ray worker.py: it snapshots RAY_ENABLE_UV_RUN_RUNTIME_ENV when Ray is imported; the probe imported Ray before NeMo-RL disables that feature. Next hypothesis test imports NeMo-RL first, matching the workload environment, and must reach native26-bundle placement rather than package the repository. No formal runtime or host-service changes. Current input business PASS41files/all6roles at09:35:00Z; platform terminal verification pending.

- 17:39HKT actual CPU Ray native teacher-first placement FAIL: all five teacher groups reserve, but a student/generation8-bundle group spans multiple physical simulated nodes. PACK is soft, not a node guarantee. Hypothesis: reserve both full-node student groups before the five2GPU teacher groups; predicted4node/26bundles allocation with one physical node per student group, two physical nodes for teachers, and no shared student/teacher nodes. Test the changed order in the CPU probe before editing native setup or submitting GPU. This changes no model/data/objective/batch/context contract. Current0GPU input job Succeeded09:35:48Z, exit0/restart0/d297imageID/currentpersonalcreator/localNFS verified.


## 2026-11-13T17:42:10+08:00: Packed32 CPU/storage readiness complete

- completedEprerequisite: student-first actualRayCluster PASS26bundles/4simulatednodes, no physicalGPUallocation; exact receipt retained. Reservation-order regression first failed on missing studentclaims, then6affected allocation/order/topology checks passed11.80s. Scopedrufflint/format andgitdiffwhitespacePASS. Broadconfigfixture error retained, not counted asPASS.
- completed: current0GPUinputbusiness andterminalPASS41files/sixroles/negativecontrol, expectedcreator/source/image/exit0/restart0. Nativeconfig200/256/65536/exactidentities/FP16FP32grad/scheduler200 PASS.
- in-progressE: sign scopedsetup/tests/recipe and selectedevidence → recheckexact56GPUjob → stoponlyifPending andverify → singlelocal32GPUsubmission. ActualGPUmemory/backward,200successes andfinalparity stillunverified. Do not rerunD orcleanup.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student、五教师和数据身份已确认 |
| Q | PASS |200成功更新合同已确认 |
| B | PASS | 完整CPU准备和当前41文件身份、终态通过 |
| C | PASS | 全词表loss/gradient参照已完成 |
| D | PASS | 五教师诊断、65536和native save/resume已通过 |
| 清理 | PASS | 授权清理已完成 |
| E | TODO |32GPU实际Ray预留与CPU检查通过，正式GPU证据仍缺 |
| F | TODO | 最终checkpoint和archive依赖E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：**16:11HKT checkpoint至当前CPU准备完成｜**当前前沿：**E正式训练资源和启动

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 真实Ray检查发现资源碎片，必要修复后停止准备扩展 |
| D2 细节反复重试 | CLEAR | OTEL、uvhook和预留顺序各有前置假设与不同结果；未重复GPU提交 |
| D3 支线问题修补 | CLEAR | 配额与资源分配直接阻碍E |
| D4 可并行任务停滞 | CLEAR | 输入检查与Ray/CPU检查并行；最终replay无checkpoint |
| D5 超范围镀金 | CLEAR | 无公共schema/interface或目标变更 |
| D6 分析空转 | CLEAR | 调查形成最小代码修复和通过的真实资源预留结果 |
| D7 路线违背 | CLEAR | 保持模型、词表、batch/context、200successes和GPUsoftgate合同 |

**文档检查：** S5 plan顶部仍写4×8已拒绝，与新修复和RayPASS矛盾；已更新实际状态与依赖。S1–S4、S6未发现明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan顶部和现有placement报告：补齐32GPU修复、真实Ray/CPU结果和替换条件（可直接执行，已更新）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：scoped signed commit → re-read原job →若Pending，停止并确认 →本机单次32GPU提交 →实际backward/loss/grad/earlysave-resume。
2. 并行：现有reader/monitor更新exact新job；GPU辅助检查保持nonblocking。
3. E/F：200实际successflags与Adamstep →最终finite/state/refit/三方parity →完整报告和Englishsummary。

## 4. 最值得关注的 3 点

1.32GPU方案当前只有CPU/storagePASS，正式更新仍0。
2. 新预留顺序消除已复现的资源碎片；整体验收标准保持。
3. 准备完成即训练，不扩大工具或重复已通过的全量扫描。


## 2026-11-13T17:47:07+08:00: Verified32H800 replacement submitted once

- completed: signedscopedcommitd1111060;6CPUregressions/actualRayplacement/nativeconfig/currentinputs/lintPASS.
- completed: exactold56GPUname/UID/creator re-readPending09:42:53Z, stoppedthroughrequiredscript09:42:57Z, verifiedStopped/noReplicas09:43:25Z. Originallauncher73918 andmonitor15510 nowexit0.
- in-progressE: newformalexp-1113-174329-659007/UIDda3fd441-bc51-4d5b-aa27-ad216d7fe445;4×8H800/32CPU/768GB/companyimage/localNFS/bothRDMArequests. Latest20261113T094702Z Pending/rank2/remaining30. Launcher82605mustremainalive; noformalupdates/loss/grad/checkpoint.
- parallel: new0GPUreaderexp-1113-174403-962663/UIDf5341c5d-fc1f-49be-87a6-2650428f1b9f/launcher53011; monitor5931. Exactreader destination tasktmp/formal-200updates-fp16-pp4-packed32-current-inputs. Formalcloudprefix endswithnewjobname. Oldreaderexp-1113-151748-639492 stopinitiated2026-11-13T09:44:11Z; verifyterminalratherthansubmitduplicates.
- pendingE/F: actualGPUplacement/modelstartup/backward/loss/grad/earlynativesave-resume →200deduplicatedsuccessfuloptimizerflags andnativeAdamstep →finitefinalstate/refit/trainingforward/scorer/vLLMparity →Englishcompletionarchive. resume_training.py onlyrunsinitial/resume;200controllersteps donotprove200successes.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student和五教师输入已确认 |
| Q | PASS |200成功更新目标已确认 |
| B | PASS | 完整CPU准备及当前身份、终态通过 |
| C | PASS | 全词表参照已完成 |
| D | PASS | 五教师诊断、长context和native save/resume通过 |
| 清理 | PASS | 授权范围已完成 |
| E | BLOCKED | 新32H800job排队，0正式成功更新 |
| F | TODO | 最终checkpoint与archive依赖E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：**32GPU准备完成至当前submissioncheckpoint｜**当前前沿：**E正式GPU运行

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | CPU准备完成后直接提交，没有扩展工具 |
| D2 细节反复重试 | CLEAR | 单次32GPU替换；排队时保持exactjob |
| D3 支线问题修补 | CLEAR | 没有启动无关修复或重跑D |
| D4 可并行任务停滞 | CLEAR | reader与monitor已并行启动，最终验证等待checkpoint |
| D5 超范围镀金 | CLEAR | 验收范围和训练设定保持 |
| D6 分析空转 | CLEAR | 已完成真实stop/no-duplicate/submission操作 |
| D7 路线违背 | CLEAR | CPU/storage前提通过，个人本机提交，GPU辅助softgate保持 |

**文档检查：** S5 plan仍写替换未执行，已更新新job/UID/launcher和实际Pending状态。S1–S4、S6未发现明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan与现有placement报告：原job替换证据、新job与独立reader/monitor身份、当前0更新和接续路径（可直接执行，已更新）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：保持82605/exactjob排队 →物理replicas/runtime/backward/loss/grad/earlynative save-resume。
2. 并行：53011reader ‖5931monitor；取得业务/平台证据并确认旧reader终态。
3. E/F：200successflags与Adamstep →finalfinite/state/refit/三方parity →报告与Englisharchive。

## 4. 最值得关注的 3 点

1.32GPU正式作业仍Pending；CPUPASS不能替代GPU修复或训练验收。
2. 原56GPUjob已Stopped/noReplicas；没有重复训练申请。
3. Launcher82605必须存活，后续读取新job和新reader路径。

- 17:48HKT final current-state check: personal API queried2026-11-13T09:46:33Z. Newformal exp-1113-174329-659007 remainsPending/rank2/remaining30 for32 requested and0replicas. New0GPUreaderexp-1113-174403-962663 isStarting; oldreaderexp-1113-151748-639492 remainsRJobStopping afterexplicitstop. Launcher82605,reader53011 andmonitor5931 werepolledalive; old73918 and15510exit0. NoformalGPUmetric/checkpoint/acceptance exists. Preserve queuedjob andlaunchers; no duplicate submission, furtherpreparation, completedD rerun orcleanup.

- 17:49HKT queue transition: `exp-1113-174329-659007` approved2026-11-13T09:47:37.481255069Z, monitorStarting09:47:48Z; directpersonalAPI09:48:26Z showsfourPendingreplicas, no assignednode/container yet. Queue-reason annotation stillmentions30remaining butisnowhistorical; actualphase isStarting. ContinuephysicalGPU/runtimeverification; noformalmetrics orcheckpoint.

- Currentfrontierupdate: monitor20261113T095506Z isStarting, fourpendingreplicas. LiveEventsread2026-11-13T09:53:47Z passedexactjob/fourreplicas withnocontinuation; jobhasfourStart-replica events09:47:38–39Z, replicaprojections empty. Physicalscheduling cause remainsunidentified; no containerbusinessfailure observed. Savedstartingstatus/monitor/events undercurrentevidence. CurrentE isTODO(startup) ratherthanBLOCKEDonquota: quotaapproved09:47:37Z. Preservelauncher82605,reader53011,monitor5931; firstinspectactualworker/source/GPU/runtime, thenbackward/loss/grad/earlyresume.200successes/finalparity/archive remainpending.


## 2026-11-13 18:02 HKT: Native physical scheduling blocker identified

- completed E investigation: existing launcher82605, reader53011 and monitor5931 were polled alive. Personal API09:59:36Z confirms exact32H800 job Starting, four unscheduled replicas, `PodScheduled=False/Unschedulable`; no physical training worker, optimizer attempt or checkpoint.
- completed E evidence: exact live Events10:00:52Z report `FailedScheduling`, GPU resource gap and zero assigned pods. All five reads PASS without pagination. This replaces the earlier unknown scheduling cause with observed native scheduler evidence; it does not identify the scheduler's internal resource calculation.
- completed E resource observation: codesign API10:00:51Z returns123/123machines, usableGPUtotal3, no8GPUusablemachine. Preserve approved job/launcher; do not change topology, submit duplicates or repeat full preparation based on a capacity snapshot.
- completed reader lifecycle: new0GPUreader Running09:59:36Z, expectedcreator/currentNFS/exactimage/restart0; old explicitly stoppedreader jobStopped09:55:26Z/replicaStopped09:55:25Z. Stale container-running field retained and not interpreted as current execution. Currentreader0files/0steps/no failures; localTensorBoard0attempts/0successes/no loss or grad.
- in-progress E/F: first actual formalbackward/health/earlynative save-resume →200actualsuccessfulupdates andAdamstate →finalfinite/state/refit/three-wayparity. No overallPASS or Englishcompletionarchive. Exact dated snapshots and commands added to existingplacementreport; no implementation change or completedD/cleanup rerun.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student 与五教师已确认 |
| Q | PASS | 200 次实际成功更新合同已确认 |
| B | PASS | 完整 CPU/storage 准备和当前输入身份已通过 |
| C | PASS | 全词表 loss/gradient 参照已完成 |
| D | PASS | 五教师诊断、65536 和 native save/resume 已通过 |
| 清理 | PASS | 授权清理已完成并验证 |
| E | BLOCKED | quota 已批准，native scheduler 报 GPU resource gap；无正式更新 |
| F | TODO | 最终 checkpoint 和 archive 依赖 E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次 Starting checkpoint 至当前原生调度取证｜**当前前沿：** E 正式 GPU 运行

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用已有精确查询和 reader；未扩展 harness |
| D2 细节反复重试 | CLEAR | 只读查询得到新的 scheduler 原文；没有反复修改或重新提交 |
| D3 支线问题修补 | CLEAR | 调度直接阻碍 E；未修复无关环境或宿主服务 |
| D4 可并行任务停滞 | CLEAR | reader 与 monitor 均存活；最终 replay 缺少正式 checkpoint |
| D5 超范围镀金 | CLEAR | 模型、数据、词表、更新数和误差上限保持 |
| D6 分析空转 | CLEAR | 从未知原因取得 FailedScheduling 和资源快照，决定保留已批准作业 |
| D7 路线违背 | CLEAR | 个人本机读平台，soft gates 保持，无重复提交或准备 |

**文档检查：** S5 plan 顶部旧 reader 终态待确认、调度原因未知，已被09:59–10:00Z证据替代；更新当前阻碍与终态。S1–S4、S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan 顶部与现有 placement 报告：记录 GPU resource gap、新 reader Running、旧 reader Stopped、正式0更新和证据边界（可直接执行，已更新）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：保持82605和已批准32H800job，待四个物理workers到位 → runtime/Ray32 → backward/loss/grad/teacher routing/earlyresume。
2. 并行：53011 reader ‖5931 monitor；按实际阶段收集业务和平台证据。
3. E/F：200 actual success flags与Adamstate → finalfinite/state/refit/三方parity → 完整报告和Englisharchive。

## 4. 最值得关注的 3 点

1. 配额批准不等于物理 GPU 已分配；当前 native scheduler 明确报资源不足。
2. 正式0更新、没有 loss/grad/checkpoint；CPU和D诊断PASS不能替代正式验收。
3. 现有 launcher 和 reader/monitor 保持；等待外部资源，不重新申请或重跑已完成准备。

## 2026-11-14 01:52 HKT — Formal failure recovered; rank3 preemption established

- completed E state recovery: formal job failed13:38:07Z November13; four actual H800 nodes/runtime receipts exist. Latest personal query17:42:14Z confirms terminal replicas. Previous queue/launcher-alive prose is historical and superseded. Launcher82605 exit1, reader53011 exit0, monitor5931 exit0; reader job Succeeded,32files/11summarizedsteps/no reader failures.
- completed E business recovery:11optimizer attempts,0successful flags; last finite loss0.1954945773,256samples/2,242,562response tokens. No formalgradnorm/scaler metric yet. Five-group scorer/vLLM MAE0.00128222–0.00135436<0.02, zero issues/finite logprobs, tailmax~0.63457. These comparisons precede every successful update and do not settle final training/scorer/vLLM parity.
- completed E initiating-event RCA: personal collector exactrank3 replica`exp-1113-174329-659007-eccbc87e`, UID160eb189-67d9-461e-93ce-b808243a4b8f, job UIDda3fd441-bc51-4d5b-aa27-ad216d7fe445. HistoricalEvents7/7rows bind nodegpu-h800-0320 and13:37:14Z `CleanPolicyTriggered`/`Preempted`. Earlier warning13:16:04Z; action at13:37:14Z marked replica preemptible. Requestingpod UNKNOWN. NativeGCS missing-heartbeat death13:37:30Z follows; JobStop13:38:17Z is later cleanup. CUDAallocator warnings are not the demonstrated termination cause.
- completed bounded RCA queries: live/historicalEvents,exactreplica incidentlog(1row),existingDeepScan(0flows) PASS; RJob/replica audits ReadTimeout15s, preserved as query failures. Sherlock existingprivate session read faileduserinfo; no QR issued. Positive preemption event resolves immediate cause without broad node scanning.
- in-progress native step1/step10 inspection: current actor `UV_PROJECT_ENVIRONMENT` invocation with `timeout60s uv run--no-sync python-` returned124/no output before a Python record. Preserve this setup failure; next use `uv run--no-project--no-sync` with the exact mounted interpreter to avoid project-environment lock/discovery, then inspect small native state files. No dependency install orGPUallocation.
- next E: native state/scaler/Adam inspection and targeted fix if needed → currentcheap input checks and one native resume fromstep10 → firstactualsuccessfulupdate/loss/grad →200successfulupdates →final16H800three-waycheckpointverification. Unsavedattempt11 rolls back; keep it in attempt history. No overall completion archive.

## 1. 规划完成情况

**汇总：** PASS 6｜FAILED 1｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student、五教师及数据身份已确认 |
| Q | PASS | 200 次实际成功更新合同已授权 |
| B | PASS | 完整输入和 CPU 环境准备已通过，可复用 |
| C | PASS | 全词表 loss/gradient 参照和观测路径已准备 |
| D | PASS | 五教师诊断、长 context 和 native save/resume 已通过 |
| 清理 | PASS | 授权清理已完成 |
| E | FAILED | 正式作业被抢占；11 次尝试、0 次成功，恢复尚未执行 |
| F | TODO | 最终 checkpoint 验收和 archive 依赖 E |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次调度 checkpoint 至当前失败恢复｜**当前前沿：** E 正式恢复和 gradient health

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 精确失败取证和小字节 state 读取直接服务恢复；没有扩展最终 harness |
| D2 细节反复重试 | CLEAR | rank3 查询取得新 Preempted 原文；未重新提交相同失败作业 |
| D3 支线问题修补 | CLEAR | 节点失联和零更新均直接影响 E |
| D4 可并行任务停滞 | CLEAR | 平台 RCA 与 native state 读取均推进 |
| D5 超范围镀金 | CLEAR | 模型、数据、词表、200 次成功和 0.02 上限保持 |
| D6 分析空转 | CLEAR | 已确定抢占及 scaler 128，转入原生恢复 |
| D7 路线违背 | CLEAR | 本机个人 API、精确云路径和辅助 soft gates 保持 |

**文档检查：** S5 plan/progress 顶部仍写排队和 launcher 存活，被 Failed 终态、11 次尝试和退出记录否定；本轮已修正。S1–S4、S6 未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress 当前状态和 failure report：替换过期调度描述，记录 Preempted、0 成功更新和 native scaler 状态（可直接执行，已更新）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. E：核实 native common scheduler/data recovery，完成最小 loss-scale 指标接线和受影响检查 → 一次原生 step10 恢复。
2. 并行：当前廉价输入身份核对 ‖ 个人平台当前资源/作业查询；不重复全扫描。
3. E/F：首个成功更新及 loss/grad → 200 成功 flags 和 Adam step → 最终 finite/state/refit/三方 parity → English archive。

## 4. 最值得关注的 3 点

1. 直接终止原因已确认为 rank3 抢占；原生 loss scale 已回退至 128。
2. 当前正式成功更新仍为 0，grad health 和最终 parity 仍未验收。
3. 从 step10 恢复原生 optimizer/scaler/data state；保留未提交 attempt11 的历史，不重置缩放或放宽误差。

### 2026-11-14 — Latest checkpoint storage constraint (in-progress, E)

The user explicitly requires only the latest checkpoint and timely removal of old checkpoints because OSS space is insufficient. The prior resume preparation with unrestricted retention and extra five-step saves must not be submitted. Native retention is a union; candidate `keep_top_k=0` plus `ft_keep_latest_k=1` excludes periodic history and retains only the latest checkpoint. Verify this behavior, inventory only current-task checkpoint directories, protect the complete formal step10 recovery state, and delete older checkpoints using a locally submitted personal zero-GPU RJobBackend workload. No extra checkpoint copies are authorized. Formal successful updates remain0/200; no new GPU training request has been submitted.

### 2026-11-14 — Native retention verified; regression collection RCA (E)

Ten direct native retention checks passed with `keep_top_k=0`, `ft_keep_latest_k=1`, covering synchronous and asynchronous publication at periodic and non-periodic steps1,10,11,20,21. Exact current-task inventory found10 obsolete checkpoint directories (149,303,776,903logicalbytes) and one complete latest formal step10 (24,884,408,375bytes,31files,8tensor shards). Six exact parent jobs are terminal and owned by i-fengyicheng. Local0GPU cleanup jobexp-1114-020444-806306 is submitted; completion is still pending.

The original metric test collected no items because tests/unit/conftest.py excludes mcore markers unless selected. After explicit `-m mcore`, its session fixture connected an unrelated existing Ray cluster and aborted with `GRPC_CALL_ERROR_INVALID_METADATA`; this did not exercise the regression. The module now uses existing pure-CPU fixture precedent fromtests/unit/models/value/test_lm_value.py. The next execution reached real begin_train_step but exposed explicit CUDA scalar allocations on a CPU-only host. The shared mocked-worker fixture now redirects scalar constructors toCPU and mocks CUDA synchronization, using the existing CPU buffer regression's exact approach. These changes isolate state-machine tests; they do not change training or suppress native runtime failures. The next run must reach the loss_scale assertion before any production edit.

#### check-progress self-check — retention milestone and test startup fixes

## 1. 规划完成情况

**汇总：** PASS5｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | Source identities recorded. |
| Q | PASS | Execution authorized. |
| B | PASS | Native inputs fully validated previously. |
| C | PASS | Numerical reference checks recorded. |
| D | PASS | Five diagnostic updates and parity passed. |
| E | TODO | Formal0/200; latest-only cleanup and metric repair advance native resume. |
| F | TODO | Final evidence and completion archive depend onE. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** previous failure-recovery checkpoint至今｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Minimal retention and metric checks settle the current user constraint and missing formal health evidence. |
| D2 细节反复重试 | CLEAR | Collection filtering, Ray startup abort, and explicit CUDA scalar allocation were identified from each actual traceback; next test isolates these exact causes. |
| D3 支线问题修补 | CLEAR | Cleanup is explicitly requested; metric loss_scale is needed for formal overflow diagnosis. |
| D4 可并行任务停滞 | CLEAR | Zero-GPU cleanup andCPUregression checks run concurrently. |
| D5 超范围镀金 | CLEAR | Existing native retention and test fixture precedents reused; no new training interface. |
| D6 分析空转 | CLEAR | Inspection produced exact deletion manifest, submitted cleanup, and corrected resume configuration. |
| D7 路线违背 | CLEAR | New storage instruction is recorded and overrides prior prepared arguments. |

**文档检查：** S1–S6均未发现明显缺陷。
**归因：** 无

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：确认清理完成且step10未变；得到有效指标回归失败并完成最小接线，随后运行受影响检查。
2. E：更新nativepreflight与dry-run，提交一次H800恢复并核实首个成功更新、loss_scale、grad_norm和教师覆盖。

## 4. 最值得关注的 3 点

1. Formal training remains0/200 successful updates.
2. Latest-only retention has10directPASS checks; cloud deletion must still reach terminal completion.
3. Test startup failures have not validated the metric fix; the actual loss_scale assertion must execute.

### 2026-11-14 — Latest-only cleanup and loss-scale metrics verified (completed E prerequisites)

- Local personal0GPUcleanupjob`exp-1114-020444-806306`, UID`d647985a-bc4a-4cbf-824a-bfcd26f1a18b`, and its replica bothSucceeded; containerexit0/restart0, expected company image digestd29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177, source100.96.128.195:/data/ycfeng:/data/ycfeng. Deleted10current-task checkpoint directories,149,303,776,903logicalbytes. Exactly one checkpoint remains: formalexp-1113-174329-659007/checkpoints/step_10. All31retainedfiles match inventory sizes/mtime, including8shards. This proves namespace deletion and unchanged retained inputs, not backend physical quota reclamation.
- Actual native scaler state: all8shards atstep1have scale65536/growth0/hysteresis1; atstep10they have scale128/growth0/hysteresis−8. The optimizer contains param_groups but no successfulAdamstep. Common schedulernum_steps2560,constantLR1e−6/WD0.01,lr_decay_steps51200; nativeiteration0. Data-plane/training_info boundary remainscurrent_step10/consumed_samples2560. Reuse this state without resetting the scaler or changing loss normalization.
- Valid metric RED now reaches the actual worker→TQ→step-aggregation assertion and fails`KeyError: loss_scale`. Minimal production fix captures `float(optimizer.get_loss_scale().item())` beforeoptimizer.step, and passes it through existing aggregation totrain/loss_scale. The test mutates native mockscale128→64duringoverflow and verifies reportedscale128,success0,absentgrad_norm. Actual scaler, optimizer and loss arithmetic are unchanged.
- Complete mockedCPUstate-machine suite108PASS/33.21s; Python3.13.14/Torch2.13.0+cu130/pytest9.1.1. Ruff0.9.9checkandformat4filesPASS;gitdiffwhitespacePASS. The global Ray/session fixtures were isolated using existing pureCPUtest precedents; explicitCUDA scalar constructors and synchronization are mocked in this state-machine fabric. Early collection/Ray/CUDAstartupfailures remain evidence and do not count as validRED.
- Formal4node recipe now useskeep_top_k0/ft_keep_latest_k1/save_period200. Additionalfive-step saves are removed; native75minute deadline and finalstep remain required save boundaries. NativeMasterConfig/OptimizerConfig/teacher/topology/model-input/horizon preflightPASS; personaldry-runPASS. Scheduler horizon remains200, original sixmodelpaths/fiveteachers/full151936vocabulary/256freshtrajectories/65536ceiling are unchanged. All41cheappublication/identity checksPASSwithoutrepeating full validation.
- No new GPU training submitted yet; formal0/200 remains unfinished. Preparedexactargs/config are archived inevidence/formal-native-resume-1-{submit-args,resolved-config}-20261114.json.

#### check-progress self-check — cleanup and instrumentation delivered

## 1. 规划完成情况

**汇总：** PASS5｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | Recorded exact source models and data. |
| Q | PASS | User authorized execution. |
| B | PASS | Inputs and runtime previously validated. |
| C | PASS | Numerical checks and instrumentation available. |
| D | PASS | Five diagnostic optimizer updates verified. |
| E | TODO | Cleanup and metrics verified; formal0/200 still requires native resume. |
| F | TODO | Final parity and archive awaitE. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** prior retention checkpoint至今｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Required108checkspassed; stop extending instrumentation and submit resume. |
| D2 细节反复重试 | CLEAR | ActualmetricREDfollowedbyGREEN; no unresolved test retry. |
| D3 支线问题修补 | CLEAR | Only explicit cleanup and formalhealth metric changed. |
| D4 可并行任务停滞 | CLEAR | Cleanup,CPUregression andpreflightadvancedconcurrently. |
| D5 超范围镀金 | CLEAR | No newoptimizer/scaler/lossinterface. |
| D6 分析空转 | CLEAR | Concrete cleanup completion and tested code delivered. |
| D7 路线违背 | CLEAR | Latest-only storage and originalformalcontractpreserved. |

**文档检查：** S1–S6均未发现明显缺陷。
**归因：** 无

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：scoped signedcommit并提交一次H800native恢复；确认实际step10恢复和首个成功optimizer更新。
2. E：持续核实loss、gradnorm、loss_scale、教师覆盖和每次checkpoint发布后仅保留最新。

## 4. 最值得关注的 3 点

1. Formalsuccessfulupdatesremain0/200.
2. Onlystep10remains; oldcheckpointcleanupandnative retentionareverified.
3. InstrumentationisCPUverified;GPUoverflow/gradienthealthstillrequiresactualtraining.

### 2026-11-14 — Formal native resume submitted and runtime verified (in-progress, E)

Signed scoped commit `6674642e` records the loss-scale metric and latest-only retention. Exactly one new training job, `exp-1114-021713-015045`, UID `4637305a-eb69-469e-96a7-9e2002abe37e`, was submitted locally as `i-fengyicheng`; all four replicas are Running, restart0, with the expected company image digest and NFS `100.96.128.195:/data/ycfeng:/data/ycfeng`. Runtime receipts confirm Python3.13.14, Torch2.13.0+cu130, CUDA13.0, and8H800 per worker. All four workers pass41cheap prepared-input checks; head reports `MOPD_RAY_READY {"nodes":4,"gpus":32}`. Native command preserves the final override pointing checkpoint_dir to the original formal job's latest-only recovery directory.

Training launcher session60173 must stay alive. Independent0GPUreader `exp-1114-021833-441442`, launcher59190, observes only this exact job with a two-hour bound and nonblocking error reporting. Its first receipt has5files/0summarized steps/0failures. It reads dataset source-group labels once rather than repeatedly; no full model scan or tensor copy occurs. Exact outputs are under `/data/ycfeng/tmp/nemo-mopd-20261112/formal-native-resume-1-current-inputs-20261114/`.

Latest-only retention is verified, old checkpoints are deleted, and final parity was inspected: `tests/manual/mopd/replay_checkpoint_logprobs.py` uses native weights loading plus the existing in-memory synchronizer and writes no additional checkpoint or HF export. Formal model initialization/resume is still underway; actual scaler restoration, first successful optimizer update, grad norm and final200updates remain unverified. Do not mark this task complete.

Next actions: (1) inspect actual native step10 restore and first optimizer result; (2) reconcile `train/loss_scale`, successful flags, loss, grad norm, and five-group parity from new events; (3) on each native deadline/final publication, verify that only the latest checkpoint remains, then resume until200successful updates; (4) run final16H800three-way parity without checkpoint copies.

#### check-progress — native runtime milestone

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始模型和数据已确定。 |
| Q | PASS | 执行已获授权。 |
| B | PASS | 输入及运行环境准备完成。 |
| C | PASS | 数值检查与指标记录已验证。 |
| D | PASS | 五次诊断更新通过。 |
| E | TODO | 正式恢复已启动，尚无新成功更新。 |
| F | TODO | 最终三方 parity 和归档未完成。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** cleanup/instrumentation checkpoint 至今｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | CPU 检查结束，已提交正式恢复。 |
| D2 细节反复重试 | CLEAR | 当前原生初始化正在推进，无重复 GPU 提交。 |
| D3 支线问题修补 | CLEAR | 工作均推进正式恢复和存储约束。 |
| D4 可并行任务停滞 | CLEAR | 训练与零 GPU evidence reader 同时运行。 |
| D5 超范围镀金 | CLEAR | 没有新增训练机制或额外 checkpoint。 |
| D6 分析空转 | CLEAR | 已有真实 worker、Ray、模型加载证据。 |
| D7 路线违背 | CLEAR | 原始模型、全词表和 200 更新合同保持。 |

**文档检查：** S5：plan.md 仍写“无活跃训练请求”，与新恢复作业 Running 冲突。S1–S4/S6 未见缺陷。
**归因：** 无执行偏离；S5 是状态文档滞后。

## 3. 纠正建议

**分支：** B：先更新状态文档再继续。

**文档更新（先执行）：**
1. plan.md 当前状态与下一步：记录恢复 job、四 worker 验证及当前等待首个 optimizer 结果；明确后续继续 latest-only pruning。（可直接执行，已完成）

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：检查真实恢复及第一个 optimizer 结果；确认 loss_scale、loss、grad norm 和成功标记。
2. E：按原生 deadline 保存并及时删除旧 checkpoint；完成 200 次实际更新后执行最终三方 parity。

## 4. 最值得关注的 3 点

1. 正式成功更新仍为 0/200。
2. 训练 Running；只提交了一份恢复作业，launcher 保持存活。
3. 启动正在复用转换缓存；没有新增 checkpoint 副本。

### 2026-11-14 — Actual native recovery and resumed attempts (E, in progress)

The trace restores the exact retained step10 and synchronizes weight version 10. Controller step11 records used loss scale 128, loss 0.2018780112, optimizer success 0, 256 samples and 2,275,836 valid response tokens. Step12 records scale 64, loss 0.1982083023, success 0, 256 samples and 2,009,960 valid tokens. This verifies actual saved-scale restoration, native backoff and GPU metric wiring. Formal successful updates remain 0/200; finite unscaled grad norm is still pending. No scaler reset, loss rescaling, checkpoint copy or HF export occurred.

Both published batches exercise math/code/stem/logic/if and have zero mask or weight-version issues. Scorer/vLLM MAE is 0.0013568863 at step11 and 0.0013624505 at step12. Corresponding p99 errors are 0.0099423647 and 0.0100831985, with maxima 0.3230872154 and 0.4857079983. These are pre-update scorer/generation measurements. Final actual-training three-way parity is pending.

Native recovery reports 512 unfinished siblings restarted, with zero reused or continued. `group_discarded` also denotes successful COMMITTED ownership cleanup in the legacy Gym path in rollout_manager.py; the event name alone is not a failure verdict. The head selector omitted normal rollout, teacher and compile messages. A bounded head tail confirms continuing rollout and teacher work without an observed new training error in that snapshot.

At 2026-11-13T18:40:45Z, the task checkpoint namespace contains only the original step10: 31 files and 24,884,408,375 bytes. Latest-only retention remains active. Training launcher 60173, reader 59190 and monitor 79639 remain alive. Observer 39268 reads local metrics with a 30-minute bound and controls no jobs. Exact commands and evidence are in test_report_2026-11-14_formal_failure_recovery.md. The first batch has lengths 300–23103; prior D evidence separately covers actual execution at the sequence ceiling.

### 2026-11-14 — Next native segment prepared with more generation replicas (E)

Historical 11-step measurements of this layout show mean exposed generation time 534.17 seconds, policy training 75.69 seconds and total step time 632.07 seconds. The resumed first and second total times are 583.58 and 527.19 seconds. Prepare 7 × 8 H800 for the next segment: 8 student GPUs at TP2/PP4/CP1, 32 generation GPUs at TP2, five 2-GPU teachers and 6 spare GPUs. Keep the original models, full vocabulary, 256 trajectories, 65536 ceiling, constant scheduler horizon 200 and native optimizer/scaler state.

The node-based TQ resolver initially changed storage units from 8 to 14. Actual step10 metadata sets storage_saved=true and contains eight storage entries. The installed AsyncSimpleStorageManager.load_checkpoint rejects saved 8/current 14 before any storage RPC. With current count 8, the native loader passes all eight positional mapping calls in a CPU check with payload RPCs replaced. The initial test used an incorrect class name and failed at import; its traceback is retained separately and is not valid RED evidence. No tensor payloads were read or copied, and the temporary metadata fixture was removed.

Prepared arguments explicitly set data_plane.simple.num_storage_units=8. The corrected native preflight returns seven PASS markers and passes the count assertion; the corrected personal dry-run exits 0. Final resolved differences are only cluster.num_nodes 4→7 and generation resource num_nodes 1→4. No native code, schema, checkpoint format or training arithmetic changes are needed. New GPU training has not been submitted; the current 32-H800 job continues to its native save and terminal boundary.

Prepared scripts are /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_native_resume_2_generation32_20261114.py and submit_formal_native_resume_2_generation32_20261114.py. Arguments, resolved configuration, field differences, saved data-plane structure and the count-contract receipt are retained under evidence/. Preflight/dry-run/count-test logs remain under task temporary storage. Expanded generation throughput, native restore and numerical parity remain unverified until actual execution.

Next actions: (1) verify the first successful optimizer update and finite grad norm; (2) reconcile five-teacher coverage and parity after each batch; (3) verify the native publication and latest-only retention before submitting the prepared next segment; (4) continue to 200 actual successful updates, then execute final training/scorer/vLLM parity without weight copies.

#### check-progress — native recovery and next-segment readiness

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student、五教师和数据已确定。 |
| Q | PASS | 验收及执行已获授权。 |
| B | PASS | 输入、环境和 CPU 前置验证已完成。 |
| C | PASS | 数值检查与 loss-scale 指标已验证。 |
| D | PASS | 五次诊断成功更新及 parity 已完成。 |
| E | TODO | 真实原生恢复已观察；正式成功更新 0/200。 |
| F | TODO | 最终三方 parity 和归档依赖 E。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次 native runtime checkpoint 至今｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用 reader 和 preflight；数量检查直接避免下一段 GPU 恢复失败。 |
| D2 细节反复重试 | CLEAR | 原生 scaler 自行 backoff；CPU import 失败已按实际类名修正，无重复 GPU 提交。 |
| D3 支线问题修补 | CLEAR | 状态、梯度、样本、retention 和恢复配置均服务 E。 |
| D4 可并行任务停滞 | CLEAR | 正式训练与 reader 均运行；下一段 CPU 准备同时推进。 |
| D5 超范围镀金 | CLEAR | 没有新增训练机制或额外权重产物。 |
| D6 分析空转 | CLEAR | 已有真实恢复、两次 attempt 和修正后的具体下一段配置。 |
| D7 路线违背 | CLEAR | 原始验收及 latest-only 限制保持。 |

**文档检查：** S1–S6 均未发现明显缺陷；计划已记录真实状态和下一段准备。
**归因：** 无

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：继续当前原生运行 → 完成标志：首次成功 flag 为 1、grad norm 有限、五教师及样本/parity 证据完整。
2. E：原生 deadline 保存后确认只剩最新 checkpoint，提交已准备的下一段 → 完成标志：原生恢复继续成功训练。
3. E → F：按实际成功数推进至 200，然后完成最终三方 parity、报告和签名提交。

## 4. 最值得关注的 3 点

1. 已观察 scale 128→64 的原生 backoff；正式成功更新仍为 0/200。
2. 前两批的五教师覆盖和 scorer/vLLM 均值 parity 通过；实际训练最终三方 parity 尚未执行。
3. 只有最新 step10；下一段保留 8 个 TQ storage units，尚未提交重复训练。

### 2026-11-14 — Third resumed attempt and current storage inspection (E)

Controller step13 used the native loss scale32, returned loss0.1890251786, optimizer success0,256samples and1,890,880valid response tokens. Formal successful updates remain0/200. All five teacher groups are present, masks/weight versions have zero issues, and scorer/vLLM MAE0.0013389613 passes the0.02 mean criterion. This measures pre-update scorer/generation only. Exact metrics and full parity summaries are in evidence/formal-native-resume-1-step13-*-20261114.json.

Personal platform query at2026-11-13T19:04:47Z confirms four Running replicas/restart0. The independent reader has no recorded collection failures. Read-only exact namespace inventory at19:05:36Z confirms only the complete step10 remains,31files/8tensor shards/24,884,408,375bytes. The same native latest-only retention remains configured. No new checkpoint, copied weight shard or HF export was created. Continue at E; current native deadline must finish publication and terminal cleanup before the prepared next segment writes this checkpoint directory.

Controller step14 is now observed: scale16, loss0.1901396066, success0,256samples/2,103,101response tokens. Scorer/vLLM MAE0.0013298019,p99 0.0101060867,max0.4830121994; all five groups and zero issues. The formal counter remains0/200. D's short-response diagnostic succeeded at scale64, which does not prove stability for these formal long-response batches. Keep observing the restored native scaler and actual unscaled gradients; no manual scaler reset or objective rescaling is authorized or needed at this stage. Step14 evidence is retained beside step13.

#### check-progress — resumed formal attempts and latest-only storage

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student、五教师和数据身份已核实。 |
| Q | PASS | 验收和执行已获授权。 |
| B | PASS | CPU 输入和环境前置检查完成。 |
| C | PASS | 全词表参考和 loss-scale 接线已验证。 |
| D | PASS | 五次短诊断成功更新与 native save/resume 已验证。 |
| E | TODO | 正式 step14 仍 overflow，成功更新为0/200。 |
| F | TODO | 最终训练/scorer/vLLM parity 和归档依赖 E。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 340794c2 native-recovery checkpoint 至今｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用实际 reader、平台查询和小范围 namespace inventory。 |
| D2 细节反复重试 | CLEAR | 没有参数轮换或重复提交；同一作业原生 scaler 从128降到16。 |
| D3 支线问题修补 | CLEAR | 仅核实 E 的更新、parity 和用户要求的存储约束。 |
| D4 可并行任务停滞 | CLEAR | 训练和 CPU reader 同时运行；下一段 CPU 准备已完成。 |
| D5 超范围镀金 | CLEAR | 无新增 checkpoint、HF export 或训练机制。 |
| D6 分析空转 | CLEAR | step13/14 的实际业务结果已留存，后续动作明确。 |
| D7 路线违背 | CLEAR | 正式模型、151936词表、256条样本、65536上限和200成功更新合同保持。 |

**文档检查：** S1–S6 均未发现明显缺陷；plan 已同步 step14，后续按实测成功数补足 controller horizon。
**归因：** 无

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：观察当前原生 scaler 和首个成功更新 → 完成标志：successflag1、有限 unscaled grad norm、五教师有效样本。
2. E：确认 deadline publication 和终态，核实只有最新 checkpoint → 完成标志：完整恢复状态、旧 checkpoint 已删除，随后提交已准备的56H800配置。
3. E → F：按实际成功数推进至200，然后执行最终三方 parity → 完成标志：全部数值/状态验收和报告完成。

## 4. 最值得关注的 3 点

1. 正式成功更新仍为0/200，短诊断不能代替正式长响应梯度验收。
2. 最新 inventory 只有 step10，24,884,408,375bytes；backend物理空间回收仍未证明。
3. 下一段需等待当前原生保存和结束，保留八个 TQ storage units 和 native optimizer/scaler。

### 2026-11-14 — Fifth resumed attempt and bounded FP16 arithmetic hypothesis

Formal controller step15 used loss scale 8, returned loss 0.1932196021, success flag 0, 256 samples and 2,240,865 valid response tokens. Formal successful updates remain 0/200. Current native training continues. Code inspection confirms that `_train_microbatch_body` passes normalization placeholder 1 and `_finish_train_step_body` divides accumulated gradients by the true token count afterwards. The native scaler minimum is 1; no scaler or objective change was made.

Before the CPU probe, the ranked predictions were: (1) raw token sums can overflow a FP16 backward before finish-time normalization, in which case an early count normalization with exact later compensation will stay finite; (2) a FP16 operator can overflow before its result reaches the configured FP32 grad buffer; (3) a specific long sample can trigger a different backward failure even though the loss is finite. The first probe uses native Torch linear backward with 65,536 tokens, a two-token vocabulary, fixed teacher logits [-1,0], initial student weights [1,0], and native loss scale 8. It allocates no GPU or checkpoint.

Observed: FP32 scaled raw gradient ±206084.234375, normalized ±0.3930744827; FP16 raw gradient Inf/-Inf, still non-finite after late normalization. Early division by the configured batch count 256 and later exact compensation gives finite raw gradient ±805 and normalized ±0.39306640625. Maximum absolute difference from FP32 is 8.0764294e-6; relative difference is 2.0546817e-5. The explicit finite/overflow assertions and rtol1e-3/atol1e-4 comparison pass. This establishes the arithmetic failure pattern and supports prediction1; it does not identify the actual CUDA/TE operator or establish a production repair. Do not replace the prepared formal path solely on this toy verdict.

### 2026-11-14 — Sixth resumed attempt and exact latest-only inventory (E)

Controller step16 records loss0.1900885552, usedscale4, success0,256samples and2,145,439valid response tokens. Formal successful updates remain0/200. Allfivegroups are covered; mask/weight-version issues0; scorer/vLLM MAE0.0013085080,p990.0098823595,max0.3649930954. Generation succeeded for256/256 samples; teacher_alias_unique andteacher_model_unique are5. Latest exact personal inventory at2026-11-13T19:29:45Z still contains onlystep10,31files/8tensor shards/24,884,408,375bytes. No additional checkpoint orHFweight export was written. Backendphysical quota reclamation remainsunverified.

Step15 retained tail analysis covers three worst samples: the maximum0.8163185120-nat error is a code response atpredictionposition10568/token264, with small neighboring differences except a few localized positions; two other retained if-sample maxima are0.4316844940 and0.2824678421. This is evidence of localized sampled error tails, not proof of their complete cause or final actual-training parity. The mean criterion passes.

The current native job remainsRunning withfourreplicas/restart0 at19:28:26Z. Keep the launcher60173alive and wait for the native deadline/saveboundary. If native scale1 still overflows, diagnose actual forward/backward before submitting repeated unchanged segments. Production loss/scaler behavior is unchanged. Evidence: formal-native-resume-1-step16-{metrics,parity}-20261114.json; step15-error-tails; retention-current.

#### check-progress — resumed formal gradient frontier after context recovery

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student、五教师和全词表路径已确定。 |
| Q | PASS |200次成功更新及执行获授权；latest-only限制已记录。 |
| B | PASS | 输入和环境完整CPU验证已有证据。 |
| C | PASS | 全词表参考及原生loss-scale指标已验证。 |
| D | PASS | 五次短诊断更新及save/resume完成。 |
| E | TODO | 正式step16仍overflow，成功更新0/200。 |
| F | TODO | 最终三方parity和归档依赖E。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：**576b6535后至今｜**当前前沿：**E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR |复用已有reader、metricreader和inventory，未扩建工具。 |
| D2 细节反复重试 | CLEAR |同一原生作业内scale8→4，未重复提交或调参；scale1仍失败时必须转真实backward诊断。 |
| D3 支线问题修补 | CLEAR |数值和存储检查服务E及用户最新要求。 |
| D4 可并行任务停滞 | CLEAR |训练与零GPUreader运行；下一段CPU准备已完成，共享checkpoint必须顺序写入。 |
| D5 超范围镀金 | CLEAR |没有额外checkpoint、权重export或新增训练机制。 |
| D6 分析空转 | CLEAR |新增step16业务结果和retention证据；CPU算术证据未被当作生产修复。 |
| D7 路线违背 | CLEAR |原始模型、151936词表、256fresh samples、65536上限和200成功更新保持。 |

**文档检查：**S5：plan当前指标停在step15，实际已到step16；其余S1–S4/S6无明显缺陷。
**归因：**无执行偏离；S5是状态滞后。

## 3. 纠正建议

**分支：**B：先更新状态文档再继续

**文档更新（先执行）：**
1. plan.md当前状态：更新step16和19:29:45Z的单checkpointinventory。（可直接执行；已完成）

**需停止的偏离行为：**无

**接下去的路线：**
1. E：观察原生低scale结果并定位持续overflow → 完成标志：成功flag1、有限unscaled grad norm、五教师覆盖。
2. E：当前native publication/terminal后确认latest-only并接续56H800 → 完成标志：真实恢复并持续成功更新。
3. E→F：累计200成功更新，再完成最终training/scorer/vLLMparity → 完成标志：全部正式数值/状态验收及报告完成。

## 4. 最值得关注的 3 点

1. 正式成功更新仍0/200；finite loss不证明gradient正常。
2. 只有最新step10，24.88GB；未创建额外权重。
3. 当前launcher保持存活；下一段不与当前作业同时写共享checkpoint。

### 2026-11-14 — Native step17 published, only latest retained, next56H800 segment submitted (E)

The final pending response completed; step17 includes256validsamples/2,194,353response tokens and maximum total length40,481. Native loss0.1972962320/usedscale2/success0; formal successfulupdates0/200. Scorer/vLLM MAE0.0013301397,p990.0100550652,max1.0676422119,allfivegroups andzero mask/weight-version issues. Healthy generation/teacher work progressed; no new systemic trace was observed.

Native publication event namesstep17. Fourreplicas andRJobexp-1114-021713-015045 areSucceeded at19:42:00Z,allcontainerexit0/restart0; traininglauncher60173/reader59190 exited0. Native retention removedstep10 before the19:42:00Z inventory. Onlystep17remains:31files/8shards/24,884,408,379bytes. Noextra checkpoint orHFweight export was written.

Bounded read-only metadata/state inspection uses tasktmp/inspect_latest_formal_native_state_20261114.py through actoruvrun ontheCPUhost and two exactSSHbyte-rangequeries. It reads551,606bytes of metadata/small native items and0model/Adam tensor chunks. All8savedscalers are1/growth0/hysteresis−15; Adamparam_groups stillhave no successfulstep; scheduler num_steps4352/constantLR1e−6/lr_decay_steps51200;8orderedTQstorage positions0–7; samplerdispatch17/consumed4352. Structure/publication/state counters areverified, not a newfulltensorfinite scan. SourcefullCPUreadiness remains reusable; native restore still requires actual worker evidence.

Preparednextsegment parameters nowsetgrpo.max_num_steps217/save_period217, because savedcontroller17has0successes and200are stillrequired. Scheduler horizon remains200. Native config/dtype/optimizer/model-identity/topology/preflight returned sevenPASSmarkers; personaldry-runexit0. No productionloss/scaler behavior orinterfaces changed.

Submittedexp-1114-034353-134003/UID405b8c82-9752-4a57-9443-b48c1cd32b5e through localpersonalPythonRJobBackend:7×8H800,codesign,32CPU/768GBpernode;student8/generation32/fiveteachers10/spare6. SevenreplicasStarting at19:44:12Z;source100.96.128.195:/data/ycfeng:/data/ycfeng;creatori-fengyicheng;samepinnedcompanyimage. Initialcontainer-config waiting reports missingconfigmap beforestartup; thissnapshot is not a terminalfailure. Traininglauncher46542 staysalive. Existingreader/monitor were reused for this exactjob:0GPUreaderlauncher62396,monitor32590. Step17restore/firstscale1update/actualthroughput remainpending.

Next: verifynewworker readiness/native restore and firstscale1optimizerresult; diagnose realbackward ifscale1stillfails; continue native latest-only segments to200actualsuccesses, then finalthree-wayparity. Evidence: formal-latest-step17-native-state, step17metrics/parity, previousterminal, nextsubmitargs/resolvedconfig. Exactcommands/logs are in the current failure-recovery report.


### 2026-11-14 — Resumed job runtime reconciled after context recovery (E)

Personal exact-job query at 2026-11-13T19:49:08Z confirms exp-1114-034353-134003, UID 405b8c82-9752-4a57-9443-b48c1cd32b5e, is Running with seven Running replicas and restart count 0. Creator is i-fengyicheng, source is 100.96.128.195:/data/ycfeng:/data/ycfeng, and every replica uses the recorded d29740f4 company image. The 19:49:16Z log query confirms all seven prepared-input checks and H800 runtime. Generation engines initialize; student starts NCCL and reuses the existing initial conversion. Exact step17 restore and step18 optimizer outcome are not yet observed. The local reader contains no new TensorBoard scalar, which is an incomplete observation rather than a workload failure. Launcher sessions 46542 and 62396 remain active.

The current segment is the native test of the previously recorded low-scale prediction. If a full batch remains non-finite at scale1, replay the real backward with module/tensor gradient observations, retaining no new checkpoint; do not submit the same unmodified segment again to obtain another overflow verdict. Training health and 200 actual successful updates remain required.

#### check-progress — current runtime and gradient frontier

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student、五教师与全词表路径已确定。 |
| Q | PASS | 验收及执行授权已记录。 |
| B | PASS | CPU 输入、环境和 native preflight 完成。 |
| C | PASS | 全词表参考检查及 loss-scale 指标已验证。 |
| D | PASS | 五次诊断成功更新及 native save/resume 完成。 |
| E | TODO | step17 已保存，当前七 worker Running；正式成功更新仍为 0/200。 |
| F | TODO | 最终训练/scorer/vLLM parity、报告和归档依赖 E。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 43d307b4 至本次真实运行核查｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用已有 query、reader 和指标读取；未扩建已通过工具。 |
| D2 细节反复重试 | CLEAR | 只有一份恢复作业；scale1 是已记录的原生 backoff 检查。若仍失败则进入真实 backward 定位。 |
| D3 支线问题修补 | CLEAR | runtime、更新数和存储证据均直接服务 E。 |
| D4 可并行任务停滞 | CLEAR | 训练与零 GPU reader 同时运行；共享 checkpoint writer 必须顺序接续。 |
| D5 超范围镀金 | CLEAR | 没有额外 checkpoint、权重副本或新的训练机制。 |
| D6 分析空转 | CLEAR | 有新的 terminal、step17 状态和七 worker runtime 证据；下一判断由实际 scale1 结果决定。 |
| D7 路线违背 | CLEAR | 原始模型、151936 全词表、256 样本、65536 上限、200 成功更新与 latest-only 保持。 |

**文档检查：** S5：plan.md 的 Starting 与最新七 worker Running 冲突；已同步。S1–S4/S6 未见明显缺陷。
**归因：** 无执行偏离；S5 是状态记录滞后。

## 3. 纠正建议

**分支：** B：先同步状态文档再继续

**文档更新（先执行）：**
1. plan.md 当前状态：记录 19:49:08Z 的 Running、七 replicas 与恢复尚未观察。（可直接执行，已完成）

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：核对 step17 真正恢复与 step18 的 scale1 结果 → 完成标志：成功 flag1、有限 unscaled grad norm、五教师覆盖；若失败则获得实际 backward 定位证据。
2. E：完成 native boundary 与 latest-only 核查 → 完成标志：新恢复状态已发布、旧 checkpoint 已删除且 writer 终态。
3. E → F：累计 200 次成功更新并完成最终三方 parity → 完成标志：正式数值、状态验收及 English archive 完成。

## 4. 最值得关注的 3 点

1. 正式成功更新仍为 0/200；有限 loss 不能替代梯度验收。
2. step17 是唯一保留 checkpoint，24,884,408,379 bytes；没有额外权重副本。
3. 当前七 worker Running；scale1 的真实结果将决定继续训练或定位 backward。


### 2026-11-14 — Expanded resume failed before update; native scheduler restoration corrected (E)

The 19:51:50Z head tail reveals the exact initialization failure: `OptimizerParamScheduler: class input value 55552 and checkpointvalue 51200 for total number of weight decay iterations do not match`. Personal exact-job inspection at19:52:12Z confirms two Failed/five Stopped replicas; head container exit1 at19:49:14Z and RJob finish19:49:16Z. Launcher46542 exits1; reader62396/monitor32590 exit0. Reader has16smallbusinessfiles,zero steps andzero failures. No step18 or formal optimizer update occurred. Formal successes remain0/200, and scale1 backward is still untested.

Cause: `ConfigContainer._calculate_scheduler_steps` derives WD horizon from train_iters217×batch256 even though explicit lr_decay_iters200 keeps LR horizon51200. The native SchedulerConfig default selects strict saved/configured equality. Setting scheduler.max_steps200 is invalid because native validation requires max_steps>=train_iters. The existing `use_checkpoint_opt_param_scheduler` explicitly loads every saved scheduler value and is the smallest supported correction for preserving the approved native continuation. Production loss/scaler/optimizer arithmetic and checkpoint format are unchanged.

Added tests/manual/mopd/check_native_scheduler_resume.py to call the real Bridge horizon calculation, `_get_scheduler`, MCore `load_state_dict`, and the next scheduler step against recorded native small state. Before the recipe edit, the CPU command exits1 at the same51200/55552 assertion. After the one-line FP16 recipe edit, exit0: saved state matches exactly; counter4352→4608; LR/WD horizons51200; LR1e−6 and groupWD0.01/0 stay constant. Ruff check/format pass. No model or Adam tensor payload is read and no checkpoint is written.

Failure collector uses exact head UIDd13727b8-75b6-4662-9bcf-a32f5540599a/nodegpu-h800-0223 and incident window19:48–19:52Z. Objects,live/historicalEvents,incidentlogs and existingDeepScan queries succeed; noexistingDeepScan flow. All-operationRJob/replicaaudits time out after20seconds and remain query failures. The system critical-task cleanup timestamp follows the observed Python failure. Manualstop actor/reason/requesttime remainUNKNOWN. Selected cause and coverage are retained in evidence/formal-native-resume-2-scheduler-failure-rca-20261114.json. Latest-only inventory at19:55:11Z confirms onlystep17,31files/24,884,408,379bytes.

Prepared replacement uses formal-native-resume-3-generation32 arguments and the same approved resources/native saved state. Native scheduler RED/GREEN is now part of CPU preflight. First appended-check import used `tests.manual` from an absolute temporary script and failed before the new check; seven prior preparation checks passed. That import error is retained separately and is not scheduler RED evidence. The runner now uses the existing absolute-script runpy mechanism. Corrected preflight is in progress; personaldry-run already exits0. GPU submission waits for corrected preflight success.


Corrected absolute-script preflight exits0 with all seven preparation checks plus native scheduler restore/next-step and final scheduler-preflight markers. Personal dry-run exits0. The only new production/config change is the existing native scheduler selection in the FP16 recipe. The replacement is ready for local personal submission; old writer/reader/monitor are terminal.


### 2026-11-14 — Corrected formal resume submitted (E)

Scoped signed commit5d3c3304 restores saved scheduler values via the existing native recipe setting and adds the real CPU scheduler check; a3a7216a retains step17/runtime evidence. An initial git commit attempt collided with a concurrent zero-age git-status index lock; no lock was removed. The next scoped commit completed normally. Unrelated files remain unstaged and preserved.

Replacement exp-1114-035754-642199, UIDf7b3451c-6b0d-4687-9b0d-91e6f0ec5e10, was locally submitted at19:57:55Z as i-fengyicheng. Exact API at19:58:20Z reports Starting with seven replicas/currentNFS100.96.128.195:/data/ycfeng:/data/ycfeng. The initial missingConfigMap waiting snapshot is recorded; it does not establish terminal failure. Same7×8H800/codesign/32CPU/768GBpernode, student8/generation32/teachers10/spare6, originalmodels/full151936/256freshtrajectories/65536ceiling/eightTQunits. Native horizon217/latest-only retention unchanged. Traininglauncher80619, independent0GPUreader4033 andmonitor20311 remainactive. No additional checkpoint, model copy orHFexport is created. Actualrestore/scale1gradienthealth pending.

#### check-progress — native scheduler correction and replacement

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始模型、五教师和全词表路径已核实。 |
| Q | PASS | 验收与执行已授权。 |
| B | PASS | CPU 输入和环境前置证据可复用。 |
| C | PASS | 全词表参考和原生指标已验证。 |
| D | PASS | 五次诊断成功更新及 save/resume 已完成。 |
| E | TODO | 正式成功更新0/200；scheduler恢复已在CPU修复，替换作业Starting。 |
| F | TODO | 最终三方parity和归档依赖E。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** a3a7216a 至替换作业提交｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 新检查直达实际scheduler load，可在CPU防止相同GPU恢复失败。 |
| D2 细节反复重试 | CLEAR | 同一51200/55552断言在CPU复现，现有native设置使完整state恢复；未轮换参数盲重提。 |
| D3 支线问题修补 | CLEAR | scheduler失败实际阻断E；已修复并提交。 |
| D4 可并行任务停滞 | CLEAR | 旧writer终态后接续训练与reader；共享checkpoint顺序写入。 |
| D5 超范围镀金 | CLEAR | 一行recipe配置与直接检查；没有新schema或额外权重。 |
| D6 分析空转 | CLEAR | 实际失败→CPU RED/GREEN→签名提交→替换运行。 |
| D7 路线违背 | CLEAR | 原始训练合同和latest-only限制保持。 |

**文档检查：** S5：plan仍为准备状态，与替换作业Starting冲突；已同步。S1–S4/S6未见明显缺陷。
**归因：** 无执行偏离；S5是状态滞后。

## 3. 纠正建议

**分支：** B：先同步状态文档再继续

**文档更新（先执行）：**
1. plan.md当前状态：记录旧运行失败、CPU修复及新作业Starting。（可直接执行，已完成）

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：验证GPU真实恢复step17/scaler1/scheduler/TQ → 完成标志：恢复trace和完整batch记录。
2. E：观察scale1的grad和success → 完成标志：有限grad、success1；若仍overflow则定位真实backward。
3. E→F：200实际成功更新、latest-only边界与最终三方parity → 完成标志：正式数值/状态验收及English归档。

## 4. 最值得关注的 3 点

1. 原生scheduler mismatch已CPU修复；scale1梯度问题仍未验证。
2. 当前替换作业已提交，正式成功更新仍0/200。
3. 唯一checkpoint仍为step17；禁止额外权重保存或exports。

### 2026-11-14 — Corrected resume actual runtime and latest-only inventory (E)

At2026-11-13T20:02:12Z (November14,2026,04:02:12HKT), the exact personal API confirms exp-1114-035754-642199 is Running with all seven replicas, restart0, creator i-fengyicheng, local NFS100.96.128.195:/data/ycfeng:/data/ycfeng, and identical full d29740f4 imageID. Returned worker logs show H800 and MOPD_PREPARED_INPUTS_CHECK_PASS41 on every replica. vLLM model initialization is underway. Its optional nixl_ep import warning appears during startup; no terminal workload failure is established. Do not spend a new GPU run on this warning without a workload failure. Local TensorBoard has zero new attempts; native restoration and scale1 update remain unverified. Required traininglauncher80619,0GPUreader4033,monitor20311 and CPUobserver35082 are alive.

The20:03:18Z task-scoped read-only namespace inventory confirms only the published step17 checkpoint remains:31files,8positive tensor shards,24,884,408,379bytes. No checkpoint, tensorchunk orHFexport was copied or created. The original inventory script still named deleted step10; a temporary exact-step17 adaptation was used before inspection. That stale script was not run and did not produce a false failure. Reader receipts cover eight small files,zero summarizedsteps,zero failures and nofinished.json at this checkpoint. Physical backend quota reclamation remains unverified.

Commands:source tasktmp/personal_env_20261113.sh; personal Python3.10.6 runs inspect_formal_native_resume_3_20261114.py and read_formal_native_resume_3_logs_20261114.py; actor Python3.13.14 runs read_formal_native_resume_3_metrics_20261114.py. All three exit0. Retention:timeout60s ssh -o BatchMode=yes -o ConnectTimeout=10 shai-ycfeng python3 - < tasktmp/inventory_latest_only_step17_20261114.py; exit0. Exact small status/log/retention evidence is retained under evidence/formal-native-resume-3-*-20261114.json.

#### check-progress — recovered live state and storage constraint

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始模型和五教师身份已有完整记录。 |
| Q | PASS | 200次实际成功更新与执行已授权。 |
| B | PASS | 完整输入准备和运行环境证据可复用。 |
| C | PASS | 全词表参考与原生指标检查完成。 |
| D | PASS | 五次成功诊断更新和原生save/resume完成。 |
| E | TODO | 新作业七replicas Running；正式成功0/200，首个scale1结果待查。 |
| F | TODO | 最终三方比较和归档依赖正式更新完成。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次scheduler修复checkpoint至20:03:18Z｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用已有reader、指标和精确作业检查；没有扩充框架。 |
| D2 细节反复重试 | CLEAR | scheduler修复后核对当前实例；没有重复提交或precision试跑。 |
| D3 支线问题修补 | CLEAR | 只核查原生恢复和用户要求的latest-only保留。 |
| D4 可并行任务停滞 | CLEAR | 当前training、只读reader、monitor并行；只有一个checkpoint writer。 |
| D5 超范围镀金 | CLEAR | 本轮没有生产代码或接口变更。 |
| D6 分析空转 | CLEAR | 最新实测Running与唯一step17已回写，等待实际首批结果。 |
| D7 路线违背 | CLEAR | 原始模型、全词表、256轨迹、65536上限和禁额外checkpoint保持。 |

**文档检查：** S5：plan仍记Starting，与20:02:12Z实际Running冲突；已直接同步。S1–S4/S6均未发现明显缺陷。
**归因：** 无执行偏离；状态记录滞后已纠正。

## 3. 纠正建议

**分支：** B：spec/tickets存在状态记录缺陷，先同步再继续。

**文档更新（先执行）：**
1. plan.md当前状态：记录七replicas Running和20:03:18Z唯一step17库存。（可直接执行，已完成）

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：核查实际step17恢复、native scheduler与首个scale1 optimizer结果 → 完成标志：恢复日志、有限grad和success1。
2. E：若scale1仍overflow，使用真实batch/backward定位第一个非有限操作 → 完成标志：实际CUDA复现、根因修复及原始场景验证。
3. E→F：补足200实际成功更新，维持唯一最新native checkpoint，再完成最终三方parity和归档。

## 4. 最值得关注的 3 点

1. 正式成功更新仍0/200；初始化Running不能算训练验收。
2. 只有step17完整checkpoint，约24.88GB；没有额外权重副本。
3. 当前scale1的实际梯度结果决定下一步，禁止继续盲重提相同overflow段。

### 2026-11-14 — Actual native GPU policy and TQ restoration confirmed (E)

The exact head log query at2026-11-13T20:05:14Z returns797rows, below the1800row request. It confirms loading the step17 distributed policy checkpoint and `successfully loaded checkpoint ... [t0/2,p0/4]`, followed by `Native TQ checkpoint restored and validated:groups=0`. Teacher initialization follows these markers. The previous WD scheduler assertion has not recurred on this actual load. These markers establish native load progression; they do not yet establish a finite scale1 backward or a successful Adam update. Formal successes remain0/200.

Command:personal Python3.10.6 RJobClient.logs_replica('exp-1114-035754-642199-cfcd2084',tail_lines=1800), wrapped in timeout55s after existing personal environment initialization; exit0. Filtered secret-redacted restore markers are retained in evidence/formal-native-resume-3-restore-selected-logs-20261114.json. No weight payload is copied and no new checkpoint is written. Continue existing training and first-update observer; if scale1 overflows, locate the actual backward failure before another unchanged formal submission.

### 2026-11-14 — First successful formal optimizer update and metric-statistic diagnosis (E)

Step18 succeeds at native scale1, optimizer_update_successful1/skipped_step0. Loss0.19038212299346924, pre-clippinggradnorm1.1475327014923096, configuredclip_grad1.0, LR1e-6,256validsamples/2,019,541validresponsetokens. Formal successes are now1/200; no arithmetic/scaler reset was made. Native generation reports256OK/0failed,teacher_samples256,weight_version_lag0. Scorer/vLLM MAE0.0013267311125374486,p990.010008955001831077,max0.2572273015975952,allfivegroups andzero mask/versionissues. Final post-update training/scorer/generation comparison remains pending.

An independent reporting anomaly is observed: token-lengthstddev isNaN andmedian equalsmean. The actual256lengths are finite,235–30622. The native calculate_single_metric explicitly returnsNaN for a singleton's sample standarddeviation; aggregate_rollout_metrics currently averages per-groupmedians/stddevs while already pooling their histograms. This does not imply nonfinite modelgradients. Healthy56GPUtraining continues; this CPU reporting diagnosis cannot gate it.

Before the CPUprobe,the ranked predictions are:(1) averaging per-group nonlinearstatistics causes the anomaly; pooled histogrammedian/stddev will differ from current output;(2) nonfinite inputlengths would make the pooled reference invalid,already ruled out by exactsample lengths;(3) exporterserialization would show correctin-memoryaggregator statistics but wrong TensorBoardvalues,which the nativefunction regression will distinguish. Added a regression in the existing TestAggregateRolloutMetrics seam: real calculate_single_metric→aggregate_rollout_metrics for singleton and unequal-size groups, checking the pooledmedian/stddev,histogram preservation and unchanged mean semantics. Production aggregation is still unchanged at this RED preparation point. The metric fix only changes existing observation summaries; native training arithmetic and checkpoint format remain out of scope for this fix.


### 2026-11-14 — Pooled rollout summary fix and continued formal training (E)

The CPU test setup failures were unrelated to the target assertion: the module's autouse reset_env_calls requested remote mock_env/mock_envs even under --noconftest. Reused the existing neighboring class-level fixture override pattern for the pure scalar aggregation class. No shared Ray service or GPU job was controlled. Effective RED now reaches the real calculate_single_metric→aggregate_rollout_metrics path and fails with median4.333333333333333 instead of2 for[1,2,10]. The second parameter case covers unequal group sizes. The earlier Ray startup errors are retained as setup failures, not RED evidence.

Smallest production fix: reuse the already pooled sibling /histogram and native calculate_single_metric to replace only existing /median and /stddev keys. Existing histogram identity, empty distributions, mean denominator semantics, /std_in_group summaries, min/max and scalar legacy behavior remain covered. CPU GREEN:15cases passed in11.91s. The actual step18 replay checks12 distributions (overall plus five domains, both total and generated tokens): all PASS. Overall total length mean8147.59765625,median7799.5,stddev6180.540175503602; generated length mean7888.83203125,median7559,stddev6108.047592100288. All256 input lengths and2,019,541response tokens are finite; this fixes observation summaries and does not modify loss/gradient/scaler arithmetic. The active training driver is not hot-patched; next native resume loads this code.

Commands: UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker;TMPDIR=tasktmp;PYTHONPATH=repo;uv run --no-sync python -m pytest --noconftest tests/unit/algorithms/test_grpo.py::TestAggregateRolloutMetrics[::test_histogram_statistics_use_pooled_observations] -q --disable-warnings (the suffix is used only for RED);uv run --no-sync python tasktmp/check_pooled_step18_lengths_20261114.py. Python3.13.14/Torch2.13.0+cu130. Logs:pooled-rollout-statistics-valid-red-20261114.log,pooled-rollout-statistics-green-20261114.log,pooled-step18-lengths-green-20261114.log. Exact actual-batch receipt:evidence/formal-native-resume-3-step18-pooled-statistics-20261114.json. Initial ruff invocation found no executable in either actor/base environment; use the previously recorded pinned ruff0.9.9 installation instead of installing worker dependencies.

At20:27:14Z,the exact personal API verifies the existing exp-1114-035754-642199 still Running/allsevenreplicas/restart0/currentlocalsource. Traininglauncher80619 and0GPUreader4033 are alive. At20:29:13Z,headlogs show the nextbatch's native TQ sample retrieval,without a returned Pythonerror. Latest summarizedmetrics still show1formal success atstep18. This is progress withinE,not proof that the nextbatch succeeded. No new checkpoint or weightcopy is created by these checks.

#### check-progress — first success and required metric-statistic repair

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始student和五教师身份已核实。 |
| Q | PASS | 200实际成功更新的验收已授权。 |
| B | PASS | 完整输入准备和CPU环境验证可复用。 |
| C | PASS | 全词表loss/gradient参考和指标入口已验证。 |
| D | PASS | 五次诊断成功更新及native save/resume已验证。 |
| E | TODO | 正式1/200成功；当前七worker Running，CPU统计修复已通过。 |
| F | TODO | 最终三方parity和English归档依赖E。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次native恢复checkpoint至本次实际batch统计GREEN｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用已有聚合测试和batch样本，统计验收后停止扩充。 |
| D2 细节反复重试 | CLEAR | 检查fixture依赖后有效RED，再按根因修复；没有修理共享Ray环境。 |
| D3 支线问题修补 | CLEAR | 关键指标异常属于用户明确要求，CPU修复与GPU训练并行。 |
| D4 可并行任务停滞 | CLEAR | 原训练、只读reader和CPU修复同时推进；最终checkpoint比较有真实权重依赖。 |
| D5 超范围镀金 | CLEAR | 只重算现有统计键，没有新schema、配置或训练缩放。 |
| D6 分析空转 | CLEAR | 真实异常→有效RED→最小修复→15项及真实batchGREEN。 |
| D7 路线违背 | CLEAR | 原始模型、全词表、256/65536/200成功和latest-only约束保持。 |

**文档检查：** S5：plan当前段仍称实际scheduler恢复/首个scale1待查，与step18success1矛盾；已同步。S1–S4/S6未发现明显缺陷。
**归因：** 无执行偏离；S5是状态滞后。

## 3. 纠正建议

**分支：** B：spec/tickets存在状态记录缺陷，先同步再继续。

**文档更新（先执行）：**
1. plan.md当前状态：记录1/200、实际恢复、统计GREEN和下一native边界。（可直接执行，已完成）

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：监测后续成功更新及版本18以后的refit/scorer证据 → 完成标志：真实success flags、有限grad和明确mask/version状态。
2. E：核查下一native publication与step17删除 → 完成标志：唯一最新完整checkpoint和保存的Adam/scaler/scheduler/sampler/TQ状态。
3. E→F：补足200成功更新并运行已有最终三方比较 → 完成标志：200真实成功、MAE≤0.02、最终状态和报告完成。

## 4. 最值得关注的 3 点

1. 只有1次正式成功更新；单批健康不能替代后续稳定性与最终验收。
2. 统计修复已CPU验证；当前训练继续，下一resume才加载修复。
3. 唯一checkpoint仍step17，下一publication后必须及时删除旧checkpoint。


### 2026-11-14 — Second formal optimizer success and sampled post-update refit parity (E)

Step19 succeeds at20:29:59Z:success1/skipped0,loss0.19377940893173218,pre-clippinggrad1.1761640310287476,scale1,LR1e−6,256samples/2,237,967response tokens. Formaltotal2/200;currentsegment2attempts/2successes/0skips. Freshbatchweightversion18 confirms sampled scorer/vLLM parity afterstep18update/refit:MAE0.0013295876747451956,p990.0099334716796875,max0.29484084248542786,allfinite,issues0;fivegroupsstem50/logic61/if48/code45/math52. Exactsmallmetrics/sample summaries are retained in evidence/formal-native-resume-3-step19-{metrics,parity}-20261114.json. Finalactualtrainingforward and200-updateacceptance remainpending.

Pinnedruff0.9.9 at the previously recordedcachepath successfully checksgrpo.py/test_grpo.py. Initial formatcheck requests only thetestfile;scopedformat applied,thenbothcheckandformatcheckPASS. No behavior changed after15regression/12realdistributionGREEN,so unrelated suites are not repeated. gitdiffwhitespacePASS. Training/readerlaunchers remainalive andnoextraweightswerewritten. Next: nativeboundary/latest-onlyinventory andcontinuedformalupdates.


### 2026-11-14 — Third formal success, retained-checkpoint constraint and prepared120-H800 continuation (E)

At this resumed-session inspection (2026-11-13T20:46:25.640858+00:00), the exact personal API reports exp-1114-035754-642199 Running/seven replicas/restart0/current-machine NFS source100.96.128.195:/data/ycfeng:/data/ycfeng. Existing launcher,0GPU mounted reader, platform monitor and continuation wait processes are alive. No new training job has been submitted. Continuation waits on this exact job/UID, then checks native publication/state and the real scheduler restore before submission. The reader/monitor waiters derive the next exact job name from the launcher log; do not launch duplicates.

Formal successes now3/200:step20 has success1/skipped0,loss0.18652436137199402,pre-clippinggradnorm1.1433717012405396,scale1,LR1e−6,256validsamples and2,190,389response tokens. Freshbatchweightversion19 coversstem47/logic63/code44/math51/if51. Sampled scorer/vLLMMAE0.0013174562888005074,p990.010034031867980964,max0.3331570625305176,signedmean−4.813743656178109e−6;allfinite;mask/versionissues0. Finalactualtrainingforward and200-success acceptance remainpending. Exactstep20metrics/parity are retained under evidence/.

Next approved resource shape is15×8H800=120GPU,student8/generation96/teachers10/spare6,TP2/PP4/CP1,eightTQstorageunits,controllerhorizon217,save_period10,keep_top_k0/ft_keep_latest_k1,checkpoint_must_save_byNone. It reuses the same checkpoint directory and native small state. NativeCPUconfig/schedulerpreflight,personal dry-run and41file/sixmodel cheap identity checkPASS; no repeatedfullHFscan,newcheckpoint,weightcopyorHFexport. Preparationreceipt/args/resolvedconfig are retained under evidence/. Generation accounts for474.10/571.23seconds atstep19 and541.64/640.98seconds atstep20; actual speedup from96generationGPUs isUNVERIFIED until the next segment runs. Input preparation completes before GPUsubmission.

One read-only inventory invocation accidentally selected the historical fixed-step10 script and asserted because step10 has already been removed. This is an auxiliaryinspectionfailure,notatrainingfailure; the error is retained in the tooltrace and noGPUjob wascontrolled. Reuse the already preparedstep17inventory/current publication. Latest-only means only the newest complete native recovery checkpoint; oldcheckpoints must be deleted after nativepublication, and finalparity must use native load plus in-memoryrefit. PhysicalOSSquota reclamation remainsUNVERIFIED.

Commands:personal_env_20261113.sh → timeout55s Python3.10.6 inspect_formal_native_resume_3_20261114.py; existingactorPython3.13.14 read_formal_native_resume_3_metrics_20261114.py; localPython3JSONsample/receipt inspection. Exactsuccessflags,notcontrollerstep,determine completion. Next:currentsegmentpublication/retention → actual120GPUresume andlengthsummaryfix → continued200successes → finalnativecheckpointthree-waycomparison.

Current-task namespace inventory at2026-11-13T20:46:26.010328+00:00:onlystep17,31files/8positive tensor shards/24,884,408,379bytes. Historicalstep10 isabsent. Existingfixedstep17inventory exits0;latest-onlystructurePASS. statvfs reports8399279947776freebytes for the sharedmountedfilesystem; this doesnotprovepersonalOSSquota orphysicaltrashreclamation. Selectedsmallreceipt:evidence/formal-native-resume-3-retention-current-20261114.json. No tensorpayload/newcheckpoint/HFexport wascopied.


## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A | PASS | 原始 student、五教师及源任务已核实。 |
| Q | PASS | 200 次实际成功更新的验收已授权。 |
| B | PASS | 完整输入准备证据有效，接续廉价身份检查通过。 |
| C | PASS | 全词表 loss/gradient 参考及指标入口已验证。 |
| D | PASS | 五次诊断成功更新及 native save/resume 已验证。 |
| E | TODO | 正式 3/200；当前 56 H800 正常运行，120 H800 接续已准备。 |
| F | TODO | 最终三方 parity 与 English archive 依赖最终 checkpoint。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次首个成功更新及统计修复 checkpoint 至今｜**当前前沿：** E

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用已有状态、样本及 native loader；没有扩展测试体系。 |
| D2 细节反复重试 | CLEAR | 当前段三次成功；旧库存脚本误用后复用已有当前脚本通过。 |
| D3 支线问题修补 | CLEAR | 统计修复和 latest-only 核查均属于明确要求。 |
| D4 可并行任务停滞 | CLEAR | 健康训练、只读观察及下一段 CPU 准备同时推进；最终比较依赖最终权重。 |
| D5 超范围镀金 | CLEAR | 增加 generation 容量针对实测主耗时，未改模型、数据或训练验收。 |
| D6 分析空转 | CLEAR | 新增 step20 成功及 parity 实测；准备接续减少重复人工重提。 |
| D7 路线违背 | CLEAR | 原始模型、全词表、256/65536/200 实际成功及 latest-only 保持。 |

**文档检查：** S5：plan.md 曾保留 2/200 且未记录已准备的 120 H800 接续，与真实 step20 和后台等待进程矛盾；其余 S1–S4/S6 未发现明显缺陷。
**归因：** 无执行偏离；S5 是状态滞后。

## 3. 纠正建议

**分支：** B：spec/tickets 存在状态记录缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. plan.md 当前状态与 progress/report：同步 3/200、step20 parity、当前唯一 step17 及接续准备；已完成。（可直接执行）

**需停止的偏离行为：** 无

**接下去的路线：**
1. E：当前 native publication 与旧 checkpoint 删除 → 完成标志：publication、原生状态和唯一最新完整 checkpoint 一致。
2. E：120 H800 实际启动与 native 恢复 → 完成标志：personal creator、当前本机 source、15 replicas、H800、恢复后成功更新、统计修复和实测耗时。
3. E→F：补足 200 实际成功并复用最终 native 三方比较 → 完成标志：200 success、MAE≤0.02、有限最终参数、终态与报告；不写额外权重。

## 4. 最值得关注的 3 点

1. 正式成功仅 3/200；准备文件和 Running 状态不能替代完整训练验收。
2. 当前只保留 step17，下一 publication 后必须及时删除旧 checkpoint；共享 statvfs 不能证明个人 OSS quota 回收。
3. 120 H800 接续尚未提交；最终 actual-training parity 和扩容收益仍待实测。
