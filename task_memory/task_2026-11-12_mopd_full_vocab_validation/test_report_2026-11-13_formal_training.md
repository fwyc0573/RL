## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Record actual formal56H800startup and finalcheckpoint replay preparation; training acceptance pending. |

# Formal MOPD training validation

Recorded: 2026-11-13 12:10 HKT. The first formal run FAILED; overall acceptance remains PENDING.

## Configuration and execution

- Training RJob: `exp-1113-112953-201895`; UID `0d70a68d-1ac2-44ef-9cab-f65be352490c`; creator `i-fengyicheng`;7nodes x8H800, codesign, non-preemptible.
- Recipe: `examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml`. Original initial student and math/code/stem/logic/if teachers;151936full vocabulary;256fresh trajectories/attempt;65536sequence ceiling;TP2/PP4/CP1;FP16/native dynamic scaler and FP32gradient reduction.
- Native constant scheduler: LR=minLR1e-6, WD0.01, lr_decay_iters200. Initial200attempt horizon must be extended by native checkpoint resume if successfulupdates are fewer than200.
- Submission: `/data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_200updates_fp16_pp4.py` with exact arguments archived in `evidence/formal-200updates-fp16-pp4-submit-args.json`; resolved config beside it.
- Ordinary workload wrapper: `tests/manual/mopd/run_on_rjob.py`; native two-phase entrypoint: `tests/manual/mopd/resume_training.py`; checkpoint_must_save_by00:00:00:01 for earlyinitial save.
- Durable outputs: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-112953-201895`.

## Observed checks

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Exact RJob origin/resources | personalcreator, currentlocal NFS,7active workers | creator i-fengyicheng; source100.96.128.195:/data/ycfeng:/data/ycfeng;7Running | PASS |
| Worker runtime |56H800; selected persistent environment; preservedNCCL |7receipts,8H800each;Python3.13.14/Torch2.13.0+cu130/CUDA13; executable selected as required; NCCL recorded | PASS |
| Native initialphase | ordinarynative entrypoint begins | MOPD_RAY_READY56GPU; MOPD_CHECKPOINT_PHASE initial | PASS for startup only |
| Reader isolation | mountedpersonalprefix; small evidence; no trainingcontrol |0GPUreader;8files/0steps/0errors in first receipt | PASS for exercised reads |
| Successfulupdates |200successful flags, agreeing nativeAdamstate |0 attempts/0 successes before Failed04:03:58Z | FAIL; repair pending |
| Native earlysave/resume | saved fullstate, loaded, further successfulupdates | not yet observed | PENDING |
| Fullrunhealth/parity/coverage | finite loss/parameters; explainedskips;5groups; MAE<=0.02 | firstbatchpending | PENDING |
| Finalcheckpoint refit | actualcheckpoint loaded, freshresponse parity<=0.02 | CPUimport/--help/Ruff/inputselection/config PASS;GPUnotrun | PENDING |

## Evidence methods and limits

Personal API queries use the mounted local StepMind Python3.10environment, credential files read privately, exact RJobname and boundedrequests. Worker logs contain actualH800 driver570.124.06 and81559MiB/card. Runtime receipts are archived in `evidence/formal-200updates-fp16-pp4-worker-runtimes.json`; currentreplicas in `evidence/formal-200updates-fp16-pp4-started-replicas.json`. All7exact replicaobjects report restartCount0 and imageID`hub.i.basemind.com/stepmind/megatron-step@sha256:d29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177`; archived in `evidence/formal-200updates-fp16-pp4-started-objects.json`.

Metrics command: `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_fp16_pp4_metrics.py`. Outputs retain scalarflags, loss, gradnorm, validtokens and samples; countlatest flagperstep. NativeAdamstate is the independent success-counter check.

Reader: `/data/ycfeng/tmp/nemo-mopd-20261112/collect_formal_fp16_pp4_mounted.py`, submitted by `submit_formal_mounted_evidence_reader.py` as `exp-1113-113211-147826`. Each published native dump is reduced using allmasked responsetokens, grouped by exacttrainingpool indices. Absolute-error distributions and IDs are retained; fullpayloads remain incloud. Overallquantiles will be computed from jointtokenerrors rather than stepquantile averages.

Final replay: `tests/manual/mopd/replay_checkpoint_logprobs.py --config <recipe> --checkpoint <published-step-dir> --dump <native-dump> --identities <train-jsonl> cluster.num_nodes=2`; sourcefullceiling is retained. Five selected originalgroup prompts produce freshresponses after native checkpoint loading and existing weighttransport refit. CPUpreflight used actualarchivedsamplelocators/tokens with metadata-onlyindexfixture; it did not load checkpointtensors or executeCUDA.

Independent scorer/vLLM comparisons are explicitly separate from actualtrainingforward instrumentation. The existing earlierthree-way and maskedobjectivegradient evidence remains retained; formalwhole-run scorerparity alone does not establish bitwise trainingforward identity. No auxiliaryvalidator may terminate healthy56GPUtraining.


## First formal failure and resource correction

The first job failed at2026-11-13T04:03:58Z after256successfulrollouts and before any recorded optimizer attempt or nativecheckpoint. Rank3RayCPU-memory monitor observed192257081344B used above191260262400B threshold (95% of201326592000B total) and killed12workers at04:03:38Z. Eight student processes used18.99–20.74GiB each;objectstore only36324B. Actor owner NODE_OUT_OF_MEMORY explains subsequent shutdownerrors. CUDA OOM and kernelOOMKilled are not observed verdicts.

Collector command (personal auth and company proxy initialized):

```bash
timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python -u -B \
  /data/ycfeng/stepcode-codex-home/.stepcode/codex/skills/stepmind-gpu-worker/scripts/collect_failure_evidence.py \
  exp-1113-112953-201895 --replica exp-1113-112953-201895-cfcd2084 \
  --expected-creator i-fengyicheng \
  --credential-dir /data/ycfeng/tmp/brainctl-auth-i-fengyicheng \
  --start 2026-11-13T04:01:30Z --end 2026-11-13T04:05:30Z \
  --timeout 45 \
  --output /data/ycfeng/tmp/nemo-mopd-20261112/formal-200updates-fp16-pp4-rca
```

Exit1 reflects two audit ReadTimeouts; other queries succeeded. Existing DeepScan count0. Failed audit coverage remains UNKNOWN.

Rayreader: locally submitted /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_failure_ray_reader.py invokes read_formal_failure_ray.py as0GPUcodesign exp-1113-120706-340132.221files inspected,19selected. Selected initiatingexcerpt andexactobjects: evidence/formal-200updates-fp16-pp4-failure-rca.json. Nativepipeline fetches the fullDPslice before internalmicrobatch iteration; normal CPUmaterialization can exceed192GB. Hypothesis-controlled rerun increases only nodeCPU-memory request to768GB; actualpass still pending.


## Resource-corrected formal submission

- RJob `exp-1113-121031-998701`, created2026-11-13T04:10:32Z. Launcher session91314 must remain alive. Node memory request768GB (native SDK768000MiB);56H800,7nodes,32CPU/node, codesign/non-preemptible, current-machine NFS source.
- Dry-run command: `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u -B /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_200updates_fp16_pp4_memory.py --dry-run`. PASS/exit0. Verified RDMA defaults, absolute persistent environment and unchanged native command. Launch uses the same command without `--dry-run`, with authenticated company proxy initialized.
- Exact arguments: `evidence/formal-200updates-fp16-pp4-memory-submit-args.json`; dry-run receipt: `evidence/formal-200updates-fp16-pp4-memory-dry-run.json`.
- Durable outputs: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-121031-998701`.
- Resource correction is not accepted until the actual first batch completes;200successful updates, formalnative resume and finalcheckpoint acceptance remain pending.


## PP4 actual-forward observer preparation

CPU regression PASS only. The old OPD_FULL callback selected globalrank0 and omitted finalpipeline-stage logits at rank6 forTP2/PP4/CP1. The revised callback records TP0/CP0 in eachDP replica, includes globalrank, and writes `forward-<hostname>-<pid>.jsonl`. Native loss/optimizer computation is unchanged. The formaljob does not select this observer.

The valid RED logs expected1 versus actual0 at rank6. The formatted GREEN exits0, passes ranks6/0/7/14/12 with expected recordcounts1/1/0/1/0, and passes separatefiles for twohosts with identicalPID1830. Commands, exact mountedactor environment and logpaths are archived in `evidence/pp4-forward-observation-regression.json`. Pinned Ruff0.9.9 check and format --check pass for the observer and checkpoint-replay entrypoint. The earlier fixture-error runs are excluded. No GPU trainingforward acceptance is claimed by this CPUcheck.
