# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import set_stop_value_multi_ends

np.random.seed(1)

bs = 64
max_model_len = 1024
stop_seqs_bs = 5
stop_seqs_max_len = 8

# test beam_search=False
topk_ids = paddle.arange(0, bs, dtype="int64")
next_tokens = paddle.full([bs], 0, dtype="int64")
stop_flags = paddle.to_tensor(np.random.randint(0, 2, [bs]), "bool")
seq_lens = paddle.to_tensor(np.random.randint(0, 5, [bs]), "int32")
end_ids = paddle.to_tensor([0, 1, 2, 3, 4, 5], "int64")

# New parameters: min_tokens=0, step_idx large enough so can_stop=True, empty stop_seqs
token_ids_all = paddle.full([bs, max_model_len], -1, dtype="int64")
prompt_lens = paddle.zeros([bs, 1], dtype="int64")
step_idx = paddle.full([bs, 1], 100, dtype="int64")
stop_seqs = paddle.full([bs, stop_seqs_bs, stop_seqs_max_len], -1, dtype="int64")
stop_seqs_len = paddle.zeros([bs, stop_seqs_bs], dtype="int32")
min_tokens = paddle.zeros([bs, 1], dtype="int64")

print("topk_ids\n", topk_ids)
print("next_tokens\n", next_tokens)
print("stop_flags\n", stop_flags)
set_stop_value_multi_ends(
    topk_ids, stop_flags, seq_lens, end_ids, next_tokens,
    token_ids_all, prompt_lens, step_idx, stop_seqs, stop_seqs_len,
    min_tokens, False,
)
print("topk_ids\n", topk_ids)
print("next_tokens\n", next_tokens)
print("stop_flags\n", stop_flags)
print("seq_lens\n", seq_lens)
print("end_ids\n", end_ids)

ref_topk_ids = np.array(
    [
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        9,
        10,
        0,
        12,
        0,
        -1,
        15,
        16,
        0,
        18,
        19,
        20,
        0,
        22,
        23,
        0,
        25,
        26,
        27,
        -1,
        29,
        30,
        31,
        0,
        0,
        0,
        -1,
        -1,
        37,
        38,
        39,
        -1,
        -1,
        0,
        0,
        0,
        0,
        46,
        -1,
        0,
        49,
        50,
        0,
        52,
        53,
        0,
        -1,
        0,
        57,
        -1,
        59,
        60,
        0,
        0,
        63,
    ],
    "int64",
)
ref_next_tokens = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        9,
        10,
        0,
        12,
        0,
        0,
        15,
        16,
        0,
        18,
        19,
        20,
        0,
        22,
        23,
        0,
        25,
        26,
        27,
        0,
        29,
        30,
        31,
        0,
        0,
        0,
        0,
        0,
        37,
        38,
        39,
        0,
        0,
        0,
        0,
        0,
        0,
        46,
        0,
        0,
        49,
        50,
        0,
        52,
        53,
        0,
        0,
        0,
        57,
        0,
        59,
        60,
        0,
        0,
        63,
    ],
    "int64",
)
ref_stop_flags = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
    ],
    "bool",
)
diff_topk_ids = np.sum(np.abs(ref_topk_ids - topk_ids.numpy()))
print("diff_topk_ids\n", diff_topk_ids)
assert diff_topk_ids == 0, "Check failed."
diff_next_tokens = np.sum(np.abs(ref_next_tokens - next_tokens.numpy()))
print("diff_next_tokens\n", diff_next_tokens)
assert diff_next_tokens == 0, "Check failed."
diff_stop_flags = np.sum(np.abs(ref_stop_flags.astype(np.int32) - stop_flags.numpy().astype(np.int32)))
print("diff_stop_flags\n", diff_stop_flags)
assert diff_stop_flags == 0, "Check failed."

# test beam_search=True
topk_ids = paddle.arange(0, bs, dtype="int64")
next_tokens = paddle.full([bs], 0, dtype="int64")
stop_flags = paddle.to_tensor(np.random.randint(0, 2, [bs]), "bool")
seq_lens = paddle.to_tensor(np.random.randint(0, 5, [bs]), "int32")
end_ids = paddle.to_tensor([0, 1, 2, 3, 4, 5], "int64")

token_ids_all = paddle.full([bs, max_model_len], -1, dtype="int64")
prompt_lens = paddle.zeros([bs, 1], dtype="int64")
step_idx = paddle.full([bs, 1], 100, dtype="int64")
stop_seqs = paddle.full([bs, stop_seqs_bs, stop_seqs_max_len], -1, dtype="int64")
stop_seqs_len = paddle.zeros([bs, stop_seqs_bs], dtype="int32")
min_tokens = paddle.zeros([bs, 1], dtype="int64")

print("topk_ids\n", topk_ids)
print("next_tokens\n", next_tokens)
print("stop_flags\n", stop_flags)
set_stop_value_multi_ends(
    topk_ids, stop_flags, seq_lens, end_ids, next_tokens,
    token_ids_all, prompt_lens, step_idx, stop_seqs, stop_seqs_len,
    min_tokens, True,
)
print("topk_ids\n", topk_ids)
print("next_tokens\n", next_tokens)
print("stop_flags\n", stop_flags)
print("seq_lens\n", seq_lens)
print("end_ids\n", end_ids)

ref_topk_ids = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        -1,
        9,
        10,
        0,
        -1,
        13,
        14,
        15,
        0,
        17,
        18,
        19,
        20,
        0,
        22,
        23,
        24,
        25,
        -1,
        -1,
        28,
        29,
        0,
        0,
        -1,
        33,
        34,
        35,
        36,
        37,
        0,
        -1,
        0,
        41,
        -1,
        0,
        44,
        45,
        46,
        0,
        0,
        49,
        0,
        0,
        0,
        53,
        0,
        0,
        0,
        0,
        58,
        -1,
        60,
        61,
        -1,
        63,
    ],
    "int64",
)
ref_next_tokens = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        0,
        9,
        10,
        0,
        0,
        13,
        14,
        15,
        0,
        17,
        18,
        19,
        20,
        0,
        22,
        23,
        24,
        25,
        0,
        0,
        28,
        29,
        0,
        0,
        0,
        33,
        34,
        35,
        36,
        37,
        0,
        0,
        0,
        41,
        0,
        0,
        44,
        45,
        46,
        0,
        0,
        49,
        0,
        0,
        0,
        53,
        0,
        0,
        0,
        0,
        58,
        0,
        60,
        61,
        0,
        63,
    ],
    "int64",
)
ref_stop_flags = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
    ],
    "bool",
)
diff_topk_ids = np.sum(np.abs(ref_topk_ids - topk_ids.numpy()))
print("diff_topk_ids\n", diff_topk_ids)
assert diff_topk_ids == 0, "Check failed."
diff_next_tokens = np.sum(np.abs(ref_next_tokens - next_tokens.numpy()))
print("diff_next_tokens\n", diff_next_tokens)
assert diff_next_tokens == 0, "Check failed."
diff_stop_flags = np.sum(np.abs(ref_stop_flags.astype(np.int32) - stop_flags.numpy().astype(np.int32)))
print("diff_stop_flags\n", diff_stop_flags)
assert diff_stop_flags == 0, "Check failed."


# ==========================================
# Test stop_seqs matching
# ==========================================
def test_stop_seqs():
    print("\n=== test_stop_seqs ===")
    sampled_token_ids = paddle.to_tensor([61502, 2], dtype="int64")
    stop_flags_t = paddle.to_tensor([False, True], dtype="bool")
    seq_lens_t = paddle.to_tensor([1, 0], dtype="int32")
    eos_token_id = paddle.to_tensor([2], dtype="int64")
    next_tokens_t = paddle.to_tensor([61502, 2], dtype="int64")

    token_ids_all_t = paddle.full([2, 32768], -1, dtype="int64")
    token_ids_all_t[0, :10] = paddle.to_tensor([21, 22, 23, 24, 25, 26, 27, 28, 8038, 61502], dtype="int64")
    prompt_lens_t = paddle.zeros([2, 1], dtype="int64")
    step_idx_t = paddle.to_tensor([10, 0], dtype="int64")

    stop_token_ids_t = paddle.full([2, 5, 8], -1, dtype="int64")
    stop_token_ids_t[0, 0, :2] = paddle.to_tensor([8038, 61502], dtype="int64")

    stop_seqs_len_t = paddle.full([2, 5], 0, dtype="int32")
    stop_seqs_len_t[0, 0] = 2
    min_tokens_t = paddle.to_tensor([0, 0], dtype="int64")

    set_stop_value_multi_ends(
        sampled_token_ids, stop_flags_t, seq_lens_t, eos_token_id, next_tokens_t,
        token_ids_all_t, prompt_lens_t, step_idx_t, stop_token_ids_t, stop_seqs_len_t,
        min_tokens_t, False,
    )

    assert bool(stop_flags_t[0]) is True, f"Expected stop_flags[0]=True, got {bool(stop_flags_t[0])}"
    assert int(sampled_token_ids[0]) == 2, f"Expected topk_ids[0]=2 (eos), got {int(sampled_token_ids[0])}"
    assert int(next_tokens_t[0]) == 2, f"Expected next_tokens[0]=2 (eos), got {int(next_tokens_t[0])}"
    print("test_stop_seqs PASSED")


# ==========================================
# Test min_tokens
# ==========================================
def test_min_tokens():
    print("\n=== test_min_tokens ===")
    # Sample 0: step_idx=5 < min_tokens=10, should NOT stop even with EOS token
    # Sample 1: step_idx=50 >= min_tokens=0, should stop with EOS token
    # Sample 2: step_idx=10 >= min_tokens=5, should stop with EOS token
    sampled_token_ids = paddle.to_tensor([2, 2, 2], dtype="int64")
    stop_flags_t = paddle.to_tensor([False, False, False], dtype="bool")
    seq_lens_t = paddle.to_tensor([1, 1, 1], dtype="int32")
    eos_token_id = paddle.to_tensor([2], dtype="int64")
    next_tokens_t = paddle.to_tensor([2, 2, 2], dtype="int64")

    token_ids_all_t = paddle.full([3, 100], -1, dtype="int64")
    prompt_lens_t = paddle.zeros([3, 1], dtype="int64")
    step_idx_t = paddle.to_tensor([5, 50, 10], dtype="int64")

    stop_seqs_t = paddle.full([3, 5, 8], -1, dtype="int64")
    stop_seqs_len_t = paddle.zeros([3, 5], dtype="int32")

    min_tokens_t = paddle.to_tensor([10, 0, 5], dtype="int64")

    set_stop_value_multi_ends(
        sampled_token_ids, stop_flags_t, seq_lens_t, eos_token_id, next_tokens_t,
        token_ids_all_t, prompt_lens_t, step_idx_t, stop_seqs_t, stop_seqs_len_t,
        min_tokens_t, False,
    )

    # Sample 0: step_idx(5) < min_tokens(10), can_stop=False, should NOT stop
    assert bool(stop_flags_t[0]) is False, f"Expected stop_flags[0]=False, got {bool(stop_flags_t[0])}"

    # Sample 1: step_idx(50) >= min_tokens(0), can_stop=True, EOS hit, should stop
    assert bool(stop_flags_t[1]) is True, f"Expected stop_flags[1]=True, got {bool(stop_flags_t[1])}"
    assert int(sampled_token_ids[1]) == 2, f"Expected topk_ids[1]=2 (eos), got {int(sampled_token_ids[1])}"

    # Sample 2: step_idx(10) >= min_tokens(5), can_stop=True, EOS hit, should stop
    assert bool(stop_flags_t[2]) is True, f"Expected stop_flags[2]=True, got {bool(stop_flags_t[2])}"
    assert int(sampled_token_ids[2]) == 2, f"Expected topk_ids[2]=2 (eos), got {int(sampled_token_ids[2])}"

    print("test_min_tokens PASSED")


test_stop_seqs()
test_min_tokens()
print("\nAll tests PASSED!")
