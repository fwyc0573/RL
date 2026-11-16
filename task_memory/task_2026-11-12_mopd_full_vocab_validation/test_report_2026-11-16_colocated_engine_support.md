## Modification History

| Date | Change |
| --- | --- |
| 2026-11-16 | Record source inspection, actual configuration resolution, isolated configuration guard checks and their limits. |

# Role engines and co-location support verification

Environment: `/data/ycfeng/sota-engine/RL`, branch `mopd-sys`, inspected HEAD `a096647167ce98418b8f0332c126407cc6e064b2`, `/usr/bin/python3` 3.10.6, PyYAML 5.4.1. No Python/conda environment installation, GPU allocation, checkpoint save or deployment change.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Actual recipe plus writer overrides | Megatron student, vLLM rollout, separated Megatron teachers | Training TP2/PP4/CP1; rollout TP2/six nodes; teacher TP2/PP1/CP1/two GPUs; hidden-state payload; colocated false | PASS |
| SC student co-location guard | Only Megatron accepted | vLLM/SGLang/TRT-LLM rejected; Megatron accepted at 256 groups/update | PASS |
| SC co-location streaming guard | Whole update required | 128-group chunks rejected against 256-group update | PASS |
| SC teacher co-location guard | Separate teachers required | false rejected with native error; true accepted | PASS |
| Megatron matching layout/implementation | Shared training model | Existing function returned None | PASS |
| Megatron different inference PP | Dedicated inference model | Training PP4/inference PP1 returned dedicated config | PASS |
| Megatron different implementation | Dedicated inference model | Training transformer_engine/inference inference_optimized returned dedicated config | PASS |
| Megatron training CP2 | Inference CP1/dedicated model | Existing function pinned inference CP1 and selected dedicated config | PASS |
| Generic GRPO full-vocabulary guard | Refuse this implementation outside SC | Source grpo.py834 raises ValueError | PASS, source only |
| Generic async GRPO co-location | Only Megatron | Source grpo.py5363 asserts MegatronGeneration | PASS, source only |
| Co-located five-teacher production throughput/parity | Actual GPU workflow needed | No such run performed in this task | UNVERIFIED |

## Reproduction commands

Actual inputs: [writer argv](evidence/formal-step40-fifo-72gpu-live/command.json) and `examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-4n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml`. Follow each `defaults` path recursively, merge nested dictionaries, then apply each dotted command override. The actual resolved fields and source chain are in [co_located_engine_support.md](co_located_engine_support.md).

Independent guard and pure-config checks ran using this command, without imports of Ray, CUDA or production worker initialization:

```sh
python3 - <<'PYCODE'
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

tree = ast.parse(Path('nemo_rl/algorithms/single_controller_utils/config.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_algo_settings')
guard = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "generation_config['colocated']['enabled']")
code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), 'existing_colocation_guard', 'exec')
for backend in ('vllm', 'sglang', 'trtllm', 'megatron'):
    ns = {'generation_config': {'backend': backend, 'colocated': {'enabled': True}}, 'async_config': SimpleNamespace(min_groups_for_streaming_train=256), 'algo_cfg': SimpleNamespace(num_prompts_per_step=256)}
    try:
        exec(code, ns)
    except ValueError:
        assert backend != 'megatron'
    else:
        assert backend == 'megatron'
ns['async_config'].min_groups_for_streaming_train = 128
try:
    exec(code, ns)
except ValueError:
    pass
else:
    raise AssertionError('Expected whole-update guard')
guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'not opd_module.is_non_colocated_teachers_enabled(master_config)')
code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), 'existing_teacher_guard', 'exec')
for enabled in (False, True):
    ns = {'master_config': None, 'opd_module': SimpleNamespace(is_non_colocated_teachers_enabled=lambda _: enabled)}
    try:
        exec(code, ns)
    except ValueError:
        assert not enabled
    else:
        assert enabled
path = Path('nemo_rl/models/generation/megatron/config.py')
funcs = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('merged_inference_megatron_cfg', 'dedicated_inference_megatron_cfg')]
ns = {'cast': cast, 'Optional': Optional, 'Any': Any, 'MCoreGenerationConfig': dict, 'PolicyConfig': dict}
exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(path), 'exec', flags=__future__.annotations.compiler_flag), ns)
mcfg = {'tensor_model_parallel_size': 2, 'pipeline_model_parallel_size': 4, 'expert_model_parallel_size': 1, 'expert_tensor_parallel_size': 1, 'context_parallel_size': 1, 'transformer_impl': 'transformer_engine', 'sequence_parallel': True}
cfg = {'megatron_cfg': mcfg, 'generation': {'mcore_generation_config': {}}}
assert ns['dedicated_inference_megatron_cfg'](cfg) is None
cfg['generation']['mcore_generation_config'] = {'pipeline_model_parallel_size': 1}
assert ns['dedicated_inference_megatron_cfg'](cfg)['pipeline_model_parallel_size'] == 1
cfg['generation']['mcore_generation_config'] = {'transformer_impl': 'inference_optimized'}
assert ns['dedicated_inference_megatron_cfg'](cfg)['transformer_impl'] == 'inference_optimized'
cfg['generation']['mcore_generation_config'] = {}
mcfg['context_parallel_size'] = 2
assert ns['dedicated_inference_megatron_cfg'](cfg)['context_parallel_size'] == 1
print('PASS: existing SC guards and Megatron co-location selection')
PYCODE
```

Source inspection commands included `sed -n '1360,1435p' nemo_rl/algorithms/single_controller_utils/config.py`, `sed -n '2040,2115p' nemo_rl/algorithms/single_controller_utils/config.py`, `sed -n '820,850p' nemo_rl/algorithms/grpo.py`, `sed -n '5350,5380p' nemo_rl/algorithms/grpo.py` and `sed -n '375,660p' nemo_rl/algorithms/distillation.py`. The research report links every support claim to the inspected source.

## Iterations and limits

Plain `python` is absent; the document-write attempt exited127 before writes and was rerun with `python3`. A first isolated teacher-guard extraction searched the wrong function and raised StopIteration; the correction located the actual guard through the module AST. A pure-config namespace incorrectly substituted `object` for typing.Optional and raised TypeError; importing the real typing.Optional corrected the inspection harness. No production code or configuration changed, and no acceptance tolerance was relaxed.

These are isolated configuration behavior and source checks. They bypass full MasterConfig validation, model publication/loading, controller startup, HTTP serving, CUDA/NCCL initialization, rollout and optimizer execution. Checked-in functional scripts show intended coverage and are not recorded as passing GPU runs. Online web calls returned no usable sources; conclusions are bound to this fork's inspected code rather than claims about the newest upstream release.
