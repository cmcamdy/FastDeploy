// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include "paddle/extension.h"
#include "xpu/plugin.h"

void GetStopFlagsMulti(const paddle::Tensor &topk_ids,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens,
                       const paddle::Tensor &end_ids,
                       const paddle::Tensor &next_tokens,
                       const paddle::Tensor &token_ids_all,
                       const paddle::Tensor &prompt_lens,
                       const paddle::Tensor &step_idx,
                       const paddle::Tensor &stop_seqs,
                       const paddle::Tensor &stop_seqs_len,
                       const paddle::Tensor &min_tokens,
                       const bool beam_search) {
  PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  std::vector<int64_t> shape = topk_ids.shape();
  int64_t bs_now = shape[0];
  int64_t end_length = end_ids.shape()[0];
  int stop_seqs_bs = stop_seqs.shape()[1];
  int stop_seqs_max_len = stop_seqs.shape()[2];
  int64_t max_model_len = token_ids_all.shape()[1];
  int r = fastdeploy::plugin::set_stop_value_multi_ends<int64_t>(
      xpu_ctx->x_context(),
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int64_t *>(topk_ids.data<int64_t>()),
      const_cast<int64_t *>(next_tokens.data<int64_t>()),
      end_ids.data<int64_t>(),
      seq_lens.data<int>(),
      bs_now,
      end_length,
      token_ids_all.data<int64_t>(),
      max_model_len,
      prompt_lens.data<int64_t>(),
      step_idx.data<int64_t>(),
      stop_seqs.data<int64_t>(),
      stop_seqs_len.data<int>(),
      stop_seqs_bs,
      stop_seqs_max_len,
      min_tokens.data<int64_t>(),
      beam_search);
  PD_CHECK(r == 0, "fastdeploy::plugin::set_stop_value_multi_ends failed.");
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids",
             "stop_flags",
             "seq_lens",
             "end_ids",
             "next_tokens",
             "token_ids_all",
             "prompt_lens",
             "step_idx",
             "stop_seqs",
             "stop_seqs_len",
             "min_tokens"})
    .Attrs({"beam_search: bool"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti));
