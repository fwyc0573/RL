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

"""Exercise byte serialization of tensor views used by the simple TQ backend."""

from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("transfer_queue")

from transfer_queue.storage.managers.simple_storage_manager import (  # noqa: E402
    AsyncSimpleStorageManager,
)
from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter  # noqa: E402


@pytest.mark.parametrize("nested", [False, True], ids=["dense", "jagged"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bfloat16])
def test_singleton_zero_stride_tensor_wire_round_trip(
    monkeypatch: pytest.MonkeyPatch, nested: bool, dtype: torch.dtype
) -> None:
    column = torch.tensor([7], dtype=dtype).expand(3)[:1]
    assert column.is_contiguous() and column.stride() == (0,)
    if nested:
        # The metadata encoder sees the same view when a jagged row is unbound.
        column = torch.nested.nested_tensor_from_jagged(
            column, offsets=torch.tensor([0, 1])
        )
        assert column.unbind()[0].stride() == (0,)
    content = np.empty(1, dtype=object)
    content[0] = {"answer": "seven"}
    fields = TensorDict({"input_lengths": column, "content": content}, batch_size=[1])
    received = []

    def put(**kwargs: Any) -> None:
        wire_fields = kwargs["fields"]
        storage_data = {
            name: AsyncSimpleStorageManager._select_by_positions(wire_fields[name], [0])
            for name in wire_fields.keys()
        }
        received.append(MsgpackDecoder().decode(MsgpackEncoder().encode(storage_data)))

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", put)
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._data_operations_started = False
    client.put_samples(["sample-0"], "rollout_data", fields=fields)

    [stored] = received
    actual = stored["input_lengths"][0] if nested else stored["input_lengths"]
    assert actual.dtype == dtype and actual.shape == (1,)
    assert actual.tolist() == [7]
    assert stored["content"][0] == content[0]


def test_ordinary_contiguous_tensor_keeps_zero_copy_buffer() -> None:
    tensor = torch.tensor([3, 4, 5], dtype=torch.int32)
    buffers = MsgpackEncoder().encode(tensor)
    assert len(buffers) == 2
    tensor[1] = 9
    assert MsgpackDecoder().decode(buffers).tolist() == [3, 9, 5]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bfloat16])
def test_jagged_metadata_encoder_preserves_zero_stride_row(dtype: torch.dtype) -> None:
    values = torch.tensor([7], dtype=dtype).expand(3)[:1]
    tensor = torch.nested.nested_tensor_from_jagged(
        values, offsets=torch.tensor([0, 1])
    )
    restored = MsgpackDecoder().decode(MsgpackEncoder().encode(tensor))
    assert restored.is_nested and restored.dtype == dtype
    assert restored.unbind()[0].tolist() == [7]
