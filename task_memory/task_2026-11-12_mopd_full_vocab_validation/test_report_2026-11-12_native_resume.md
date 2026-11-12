## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record native five-teacher FP32 save/resume execution; state inspection is in progress. |

# Five-teacher FP32 native checkpoint and resume

Status: native save/resume execution, successful updates, short per-group scorer/vLLM parity PASS. Optimizer/scheduler continuation and every listed model/optimizer tensor finite PASS; retained sampler/data-state confirms the observed empty-buffer boundary. Formal200updates are not satisfied.

Job`exp-1112-191947-030955`,56H800/codesign, creator i-fengyicheng, localNFS100.96.128.195:/data/ycfeng:/data/ycfeng;7replicas Succeeded2026-11-12T11:43:50Z, launcher95310exit0. Original student and all5distinct teachers, full151936vocabulary,TP2/CP4/packing,FP32 heads. Ordinary controllers run in separate subprocesses with the same5-update horizon. Native checkpoint deadline00:00:00:01 savesstep1; resumed stage clears only deadline and loadsstep1, completessteps2–5.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Successful/skipped optimizer updates | 5/0 across both controllers | TB5/0 atsteps1–5 | PASS |
| Accepted samples/tokens | 32/4096perupdate | 160samples/20480tokens total | PASS |
| Loss/gradient norm | Finite | Loss0.10995886–0.16735572; pre-clippinggradnorm1.69602823–2.98565817 | PASS |
| Native progress | Resume1→5 | step/version1→5; consumed32→160; valid4096→20480; dispatch1→5 | PASS |
| Scorer/vLLM mean | ≤0.02every group and step | Overall0.00977465355; IF0.01607742233; allsteps/groups≤0.02 | PASS |
| Error tails | Retain/investigate | p990.08889551,max0.44571996,signedmean−0.00003033 | Observed |
| Native data-plane recovery | Actual load | Restoring nativeTQ checkpoint; restored and validatedgroups0 | PASS for empty recovery boundary |
| Replay/replacement | Actual reload | Restored0replaygroups and32pooled spareprompts fromstep1 | PASS for observed boundary |
| Optimizer/scheduler continuation | Native retained state | BothTP optimizerparamgroupsstep5, scheduler num_steps160/decay_steps160; resumed4updates aftersavedstep1 | PASS |
| Entire model/optimizer finite | All tensors | 1801chunks,1777088000model elements and5331526656optimizer elements;0nonfinite | PASS |

Native clipping maximum1.0 is applied after reporting the norm; LR remains1e-6. No arbitrary gradient scaling. Per-group singleton stddev is definedNaN inmetric_utils, separate from loss/gradient health. This diagnostic disables task answer verification (`verification_skipped=1`); numerical/full-loss checks remain enabled.

Exact command:`PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_five_teacher_fp32_resume_diagnostic.py`. Args:`five-teacher-fp32-resume-diagnostic-submit-args.json`; reusableentrypoint:`tests/manual/mopd/resume_training.py`. Analysis commands use `UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 uv run --no-sync python` withtasktmp/analyze_five_teacher_fp32_resume.py andinspect_resume_state.py. MountedbasePython3.13.14,Torch2.13/CUDA13; personal launcherPython3.10.6.

Selected evidence:`evidence/five-teacher-fp32-resume-{terminal,parity,training-metrics}.json`. Entire artifacts:`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/five-teacher-fp32-resume-diagnostic/exp-1112-191947-030955/`. Native retention removedstep1afterstep5; earlierstep1training_infoandload logs are retained. Initialtarattempt failed on absentstep1andwrong`sampler_state.pt` filename; preservederror, then requested only verifiedstep5files. Independent scorer parity is measured; actual training-forward parity is not observed in this run.


## Complete tensor and retained-state audit

Targeted native metadata/byte-range read recovered both TP optimizerparamgroups withAdamstep5 (not4), betas0.9/0.95,eps1e-8,LR1e-6,WD0.01/0. Native commonstate has scheduler num_steps160,lr_decay_steps160,constantLR,matching5updates×32GBS andtrain_state.consumed_train_samples160. Step0 inBridge train_state is its savediteration convention; controllertraining_info carries actualRLstep5. Together with the recorded step1load and four subsequent successful updates, these final counters establish retained optimizer/scheduler progress. The removedstep1optimizer bytes cannot be compared independently.

Dataloader snapshot has256samples_yielded (includes trained160,prefetchedandreserve prompts); training_info.next_nemo_gym_task_index224. Native load log restores the dataloader,emptyreplay/TQboundary and32reserveprompts. Asyncrollout recovery records remain generating atsave and are handled by native recovery; these are attempted-work state, not extra successful samples. NonemptyTQrecovery is not exercised by thisrun.

Full tensor scan exact command:`python3 -u /data/ycfeng/tmp/nemo-mopd-20261112/scan_resume_checkpoint_finite.py > /data/ycfeng/tmp/nemo-mopd-20261112/resume-finite-scan.log 2>&1`. LocalPython3.10orchestrates2boundedread-onlySSHqueries atatime; remotePython3.12.3/Torch2.5.1+cu124reads exactcheckpointchunk byte ranges,eachprocess15GiBulimit and60stimeout. NativeTorch2.13metadata is decoded locally withPython3.13.14; no remoteenvironmentchangeorcloudwrite.8files/1801chunks/7108614656elements allfinite. Modelcoverage1777088000elements; optimizercoverage5331526656,includingmasterparameters,momentsandpadding.0nonfinitechunks. Evidence:`evidence/five-teacher-fp32-resume-finite.json` and`five-teacher-fp32-resume-optimizer-state.json`. Savedstate is finite; this does not replace finalformalcheckpointrefit/parityvalidation.
