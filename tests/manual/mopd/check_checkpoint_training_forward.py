# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check native checkpoint-forward observation and response-token alignment on CPU."""

from pathlib import Path
from unittest.mock import Mock, patch
import json
import math
import os
import tempfile

import torch
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.manual.mopd import native_forward_worker as module
from tests.manual.mopd.replay_checkpoint_logprobs import training_forward_report

from tests.manual.mopd.native_forward_entrypoint import (
    select_observed_workers,
    OBSERVED_WORKER,
    NATIVE_WORKER,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.models.policy import lm_policy


def main() -> None:
    """Check the diagnostic callback, masked loss inputs, and recorded comparisons."""
    worker_class = module.NativeForwardWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.cfg = {"logprob_chunk_size": 128}
    worker._observe_objective_gradient = Mock(
        side_effect=AssertionError("Unexpected OPD gradient probe")
    )
    observations = []
    worker._write_observation = observations.append
    loss = module.ClippedPGLossFn(
        ClippedPGLossConfig(disable_ppo_ratio=True, reference_policy_kl_penalty=0)
    )
    logprobs = torch.tensor([[-1.01, -2.02]])
    data = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "input_lengths": torch.tensor([3]),
        "token_mask": torch.tensor([[0, 1, 1]]),
        "sample_mask": torch.ones(1),
        "prev_logprobs": torch.tensor([[0, -1.02, -2.01]]),
        "generation_logprobs": torch.tensor([[0, -1, -2]]),
    }
    with (
        patch.object(module.MegatronPolicyWorkerImpl, "__init__", return_value=None),
        patch.dict(
            module.os.environ,
            {"MOPD_OBSERVATION_DIR": "/mnt/codesign-exp/ycfeng/diagnostic"},
        ),
        patch.object(
            module.train,
            "prepare_loss_input",
            return_value=({"next_token_logprobs": logprobs}, data),
        ),
        patch.object(module.train.TeacherFullPayloadPostProcessor, "__call__"),
        patch.object(
            module,
            "get_next_token_logprobs_from_logits",
            side_effect=AssertionError("Unexpected extra logprob calculation"),
        ),
        patch.object(torch.distributed, "get_rank", return_value=6),
        patch.object(
            module.parallel_state, "get_tensor_model_parallel_rank", return_value=0
        ),
        patch.object(
            module.parallel_state, "get_context_parallel_rank", return_value=0
        ),
    ):
        worker_class.__init__(worker)
        module.train.prepare_loss_input(torch.zeros(1, 3, 4), data, loss)
    assert len(observations) == 1
    assert observations[0]["next_token_logprobs"] == logprobs.tolist()
    print("LOGPROB_OBSERVER_USES_NATIVE_RESULT_PASS")

    second = BatchedDataDict(
        {
            "input_ids": torch.tensor([[8, 9, 10, 11]]),
            "input_lengths": torch.tensor([4]),
            "token_mask": torch.tensor([[0, 0, 1, 1]]),
            "sample_mask": torch.ones(1),
            "prev_logprobs": torch.tensor([[0, 0, -3.0, -4.0]]),
            "generation_logprobs": torch.tensor([[0, 0, -3.0, -4.0]]),
        }
    )
    batch = BatchedDataDict.from_batches([BatchedDataDict(data), second])
    record = {
        "kind": "training_forward",
        "rank": 6,
        "logits_dtype": "torch.float16",
        "input_ids": [[5, 6, 7, 0], [8, 9, 10, 11]],
        "input_lengths": [3, 4],
        "token_mask": [[0, 1, 1, 0], [0, 0, 1, 1]],
        "next_token_logprobs": [[-1.01, -2.02, 0], [-0.5, -3.01, -4.03]],
    }
    folder = Path(
        tempfile.mkdtemp(prefix="checkpoint-forward-report-", dir=os.environ["TMPDIR"])
    )
    path = folder / "forward-node-1.jsonl"
    path.write_text(json.dumps(record) + "\n")
    result = training_forward_report(folder, batch)
    assert result["verdict"] == "MEASURED" and result["observed_samples"] == 2
    assert result["generation_error"]["tokens"] == 4
    assert math.isclose(result["generation_error"]["mae"], 0.0175, abs_tol=1e-6)
    assert math.isclose(result["scorer_error"]["mae"], 0.015, abs_tol=1e-6)
    print("SHIFTED_TOKEN_ALIGNMENT_AND_PADDING_PASS", json.dumps(result))

    empty = folder / "empty"
    empty.mkdir()
    missing = training_forward_report(empty, batch)
    assert missing["verdict"] == "UNVERIFIED" and missing["missing_samples"] == [0, 1]
    print("MISSING_OBSERVATION_SOFT_VERDICT_PASS")
    record["token_mask"][0] = [0, 0, 1, 0]
    path.write_text(json.dumps(record) + "\n")
    invalid = training_forward_report(folder, batch)
    assert (
        invalid["verdict"] == "UNVERIFIED"
        and "sample_0_response_mask_mismatch" in invalid["issues"]
    )
    print("INVALID_MASK_SOFT_VERDICT_PASS")
    batch["advantages"] = torch.ones_like(batch["token_mask"], dtype=torch.float32)
    probe = torch.tensor([[-1.01, -2.02, 0], [-0.5, -3.01, -4.03]], requires_grad=True)
    objective, metrics = loss(
        data=batch,
        next_token_logprobs=probe,
        global_valid_toks=batch["token_mask"][:, 1:].sum(),
        global_valid_seqs=batch["sample_mask"].sum(),
    )
    assert math.isclose(objective.item(), 2.5175, abs_tol=1e-6)
    gradient = torch.autograd.grad(objective, probe)[0]
    assert torch.equal(gradient, torch.tensor([[-0.25, -0.25, 0], [0, -0.25, -0.25]]))
    print("NATIVE_DIAGNOSTIC_LOSS_INPUT_PASS", objective.item(), gradient.tolist())
    select_observed_workers()
    assert ACTOR_ENVIRONMENT_REGISTRY[OBSERVED_WORKER].endswith(
        NATIVE_WORKER + "/bin/python"
    )
    assert lm_policy.RayWorkerBuilder.__name__ == "NativeForwardWorkerBuilder"
    print("MOUNTED_OBSERVER_REGISTRATION_PASS")
    print("CHECKPOINT_TRAINING_FORWARD_CPU_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
