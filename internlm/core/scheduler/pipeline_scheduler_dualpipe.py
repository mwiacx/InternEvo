#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import queue
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.scheduler import comm
from internlm.core.scheduler.base_scheduler import BaseScheduler
from internlm.core.scheduler.pipeline_scheduler_1f1b import (
    PipelineScheduler,
    pack_return_tensors,
)
from internlm.utils.common import (
    SchedulerHook,
    check_data_is_packed,
    get_current_device,
    move_to_device,
)
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_isp
from internlm.utils.timeout import llm_timeout

logger = get_logger(__file__)


def _get_tensor_or_tensors_shape(
    tensor_or_list: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[torch.Size, List[torch.Size]]:
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.shape
    else:
        return [_t.shape for _t in tensor_or_list]


DEBUG = True
LOG_RANKS = (0,)


def debug_print(msg: str) -> None:
    rank = gpc.get_global_rank()

    if rank not in LOG_RANKS:
        return

    print(f"# rank{rank}: {msg}", flush=True)


class WeightGradStore:
    """
    When using zero bubble pp, WeightGradStore is used to store the args and func for computating weight grad.
    """

    _cache = []
    _weight_grad_queue = queue.Queue()
    _hooks = {}
    pp_mode = None
    optim = None
    temp = []

    @classmethod
    def set_pp_mode(cls, mode):
        cls.pp_mode = mode

    @classmethod
    def set_optim(cls, optim):
        cls.optim = optim

    @classmethod
    def size(cls):
        return cls._weight_grad_queue.qsize()

    @classmethod
    def put(cls, weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args):
        if cls.pp_mode == "ZBH1":
            assert not gpc.is_first_rank(ParallelMode.PIPELINE), "pp rank 0 should not arrive here"
        # Store the weight gradient computation of linear layers.
        cls._cache.append((weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args))

    @classmethod
    def flush(cls):
        if cls.pp_mode == "ZBH1" and gpc.is_first_rank(ParallelMode.PIPELINE):
            return
        # Collect all stored computations during backward as a W for each micro batch.
        cls._weight_grad_queue.put(cls._cache)
        cls._cache = []

    @classmethod
    def pop(cls):
        if cls.pp_mode == "ZBH1" and gpc.is_first_rank(ParallelMode.PIPELINE):
            return
        assert cls._weight_grad_queue.qsize() > 0
        stored_w_grad_computation = cls._weight_grad_queue.get()
        # Run computation for a single W.
        for weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args in stored_w_grad_computation:
            assert weight.requires_grad
            grad_weight, grad_bias = grad_compute_func(input_tensor, grad_output, has_d_bias)

            if is_using_isp():
                isp_grad_hook = args[0]
                module = args[1]
                grad_weight, handle_weight = isp_grad_hook(grad_weight, async_op=True, is_bias=False, module=module)
                handle_weight.wait()
                if grad_bias is not None:
                    grad_bias, handle_bias = isp_grad_hook(grad_bias, async_op=True, is_bias=True, module=module)
                    handle_bias.wait()

            # Gradient Accumulation
            weight.grad = weight.grad.data + grad_weight if weight.grad is not None else grad_weight
            if has_d_bias:
                bias.grad = bias.grad.data + grad_bias if bias.grad is not None else grad_bias

            # overlap hook
            if weight in cls._hooks:
                for hook in cls._hooks[weight]:
                    hook()
                if has_d_bias:
                    for hook in cls._hooks[bias]:
                        hook()

    @classmethod
    def register_hook(cls, param, hooks):
        cls._hooks[param] = hooks


class DualPipelineScheduler(PipelineScheduler):
    """
    Interleaved Pipeline Scheduler.
    """

    def __init__(
        self,
        num_microbatches: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        communication_overlap: bool = False,
        optimizer=None,
    ):
        """A helper schedule class for pipeline parallelism running environment.
        It uses interleaved 1F1B strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            dtype (torch.dtype, optional): The data type of the tensors. Default is torch.float.
            data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
            scheduler_hooks (List[SchedulerHook], optional): List of scheduler hooks. Default is None.
            communication_overlap (bool, optional): Whether to enable communication overlap. Default is False.
        """
        assert num_microbatches % 2 == 0, f"num_microbatches: {num_microbatches} must be an integer multiple of 2"

        super().__init__(
            num_microbatches=num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )

        WeightGradStore.set_pp_mode("ZBV")
        WeightGradStore.set_optim(optimizer)

        # TODO: debug
        communication_overlap = True
        self._zero_tensor = torch.tensor(0, device=get_current_device(), dtype=self.dtype)

        self._num_pipe_stream = 2  # DualPipe
        self._communication_overlap = communication_overlap
        # switch 1f1b loop runner function according to communication overlap
        self._run_1f1b_loop = (
            self._run_1f1b_loop_with_overlap if communication_overlap else self._run_1f1b_loop_without_overlap
        )

        # states
        self._pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        self._pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        self._half_pp_size = self._pp_size // 2
        self._down_stream_rank = self._pp_rank
        self._up_stream_rank = self._pp_size - self._down_stream_rank - 1
        self._lower_pp_rank = min(self._down_stream_rank, self._up_stream_rank)
        self._pipe_stream_id = 0 if self._down_stream_rank < self._half_pp_size else 1
        self._auto_backward_microbatches = self._get_auto_backward_microbatches()

        debug_print(f"{self._auto_backward_microbatches}")

        self._accum_loss = None
        self._accum_moe_loss = None
        self._return_tensors = None
        self._backward_a2a_comm_handler = None
        self._partial_backward_func = None
        self._current_pipe_id = 0
        self._tmp_output_objs = []
        self._input_obj_idx = [0 for _ in range(self._num_pipe_stream)]
        self._input_objs = [[] for _ in range(self._num_pipe_stream)]
        self._output_objs = [[] for _ in range(self._num_pipe_stream)]
        self._output_obj_grads = [[] for _ in range(self._num_pipe_stream)]
        self._moe_losses = [[] for _ in range(self._num_pipe_stream)]
        self._preload_micro_data = [None for _ in range(self.num_microbatches)]

        self._input_obj_shapes = [self.tensor_shape for _ in range(self._num_pipe_stream)]
        self._output_obj_shapes = [None for _ in range(self._num_pipe_stream)]
        self._send_tensor_shape_flags = [self.tensor_shape is None for _ in range(self._num_pipe_stream)]

    @property
    def tensor_shape(self) -> torch.Size:
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, tensor_shape: torch.Size):
        self._tensor_shape = tensor_shape
        self._input_obj_shapes = [self._tensor_shape for _ in range(self._num_pipe_stream)]
        self._send_tensor_shape_flags = [self._tensor_shape is None for _ in range(self._num_pipe_stream)]

    def _clear_state(self) -> None:
        self._accum_loss = None
        self._accum_moe_loss = None
        self._return_tensors = None
        self._backward_a2a_comm_handler = None
        self._partial_backward_func = None
        self._current_pipe_id = 0
        self._tmp_output_objs = []
        self._input_obj_idx = [0 for _ in range(self._num_pipe_stream)]
        self._input_objs = [[] for _ in range(self._num_pipe_stream)]
        self._output_objs = [[] for _ in range(self._num_pipe_stream)]
        self._output_obj_grads = [[] for _ in range(self._num_pipe_stream)]
        self._moe_losses = [[] for _ in range(self._num_pipe_stream)]
        self._preload_micro_data = [None for _ in range(self.num_microbatches)]

        self._input_obj_shapes = [self.tensor_shape for _ in range(self._num_pipe_stream)]
        self._output_obj_shapes = [None for _ in range(self._num_pipe_stream)]
        self._send_tensor_shape_flags = [self.tensor_shape is None for _ in range(self._num_pipe_stream)]

    def load_batch(self, engine, data_iter):
        # Pipeline schedule just puts data in memory,
        batch_data, actual_batch_size = engine.load_batch(data_iter, to_gpu=True)

        # Even if 'use_flash_attn' is False, the data seen when the 'load_batch' is called is still packed,
        # because internlm's current train dataset is packed, even using dummy data.
        # The unpack operation is performed in load_micro_batch().
        if check_data_is_packed(batch_data):
            micro_num = actual_batch_size
        else:
            micro_num = actual_batch_size // gpc.config.data["micro_bsz"]

        self.microbatch_offset = 0
        self.batch_size = actual_batch_size
        self.batch_data, self.batch_label = batch_data
        self.bsz_stride = self.batch_size // micro_num
        # 'num_microbatches' is no longer an initialization parameter,
        # but is determined on the fly by the Scheduler.
        self.num_microbatches = micro_num  # Rampup or variable bsz size.

        for mbs in range(self.num_microbatches):
            micro_batch_data, micro_batch_label = self._load_micro_batch(
                data=self.batch_data,
                label=self.batch_label,
                offset=mbs * self.bsz_stride,
                bsz_stride=self.bsz_stride,
            )

            if self.data_process_func:
                micro_batch_data, micro_batch_label = self.data_process_func(micro_batch_data, micro_batch_label)

            micro_batch_data["label"] = micro_batch_label
            self._preload_micro_data[mbs] = micro_batch_data

        # overwrite microbatch_offset, since model chunks load the same microbatch, and should tract the offset
        num_stream_microbatches = self.num_microbatches // 2
        self.microbatch_offset = [num_stream_microbatches * i for i in range(self._num_pipe_stream)]

    def load_micro_batch(self, pipe_id):
        offset = self.microbatch_offset[pipe_id]
        assert self._preload_micro_data[offset] is not None, "preload micro batch data is None"

        micro_batch_data = self._preload_micro_data[offset]
        self.microbatch_offset[pipe_id] += 1

        result = move_to_device(micro_batch_data)
        return result

    def _is_pipeline_first_stage(self, pipe_id):
        if pipe_id == 0:
            return self._down_stream_rank == 0

        return self._up_stream_rank == 0

    def _is_pipeline_last_stage(self, pipe_id):
        if pipe_id == 0:
            return self._down_stream_rank == self._pp_size - 1

        return self._up_stream_rank == self._pp_size - 1

    def _get_prev_next_rank(self, pipe_id) -> Tuple[int, int]:
        next = +1 if pipe_id == 0 else -1

        prev_local_rank = (self._pp_rank - next) % self._pp_size
        next_local_rank = (self._pp_rank + next) % self._pp_size
        global_ranks = gpc.get_ranks_in_group(ParallelMode.PIPELINE)

        return global_ranks[prev_local_rank], global_ranks[next_local_rank]

    def _get_auto_backward_microbatches(self) -> List[int]:
        # rank0/7: 4(6/7/8/9)+3(10/11/12) rank1/6: 3(7/8/9)+2(10/11)+1(19)
        # rank2/5: 3(7/8/9)+1(10)+1(19) rank3/4: 2(8/9)+2(18/19)
        # rank4/3: 2(8/9) + 2(18/19), rank5/2: 1(0)+1(9) + 3(17/18/19)
        # rank6/1: 2(0/1)+1(9) + 3(17/18/19), rank7/0: 3(0/1/2) + 4(16/17/18/19)
        manual_microbatches = []

        num_stream_microbatches = self.num_microbatches // 2
        down_stream_microbatches = list(range(num_stream_microbatches))
        up_stream_microbatches = [num_stream_microbatches + _i for _i in range(num_stream_microbatches)]

        num = (self._up_stream_rank + 1) // 2
        down_stream_tails = down_stream_microbatches[-num:] if num else []
        manual_microbatches.extend(down_stream_tails)

        num = max(0, (self._down_stream_rank - self._half_pp_size))
        down_stream_heads = down_stream_microbatches[:num]
        manual_microbatches.extend(down_stream_heads)

        num = max(0, (self._up_stream_rank - self._half_pp_size))
        up_stream_heads = up_stream_microbatches[:num]
        manual_microbatches.extend(up_stream_heads)

        num = (self._down_stream_rank + 1) // 2
        up_stream_tails = up_stream_microbatches[-num:] if num else []
        manual_microbatches.extend(up_stream_tails)

        return manual_microbatches

    def _get_current_microbatch_id(self, step_id=None) -> int:
        id = self.microbatch_offset[self._current_pipe_id] - 1
        return id

    def _cut_compute_graph(self, *output_objs: Optional[torch.Tensor]) -> torch.Tensor:
        pipe_id = self._current_pipe_id
        input_obj_idx = self._input_obj_idx[pipe_id]
        # 如果该forward对应的backward属于1f1b phase，截断
        if self._get_current_microbatch_id() not in self._auto_backward_microbatches:

            self._tmp_output_objs.append(output_objs)

            detached_output_obj = []
            for tensor in output_objs:
                if tensor is None:
                    detached_output_obj.append(tensor)
                else:
                    is_requires_grad = tensor.requires_grad if isinstance(tensor, torch.Tensor) else False
                    if isinstance(tensor, torch.Tensor):
                        tensor = tensor.detach()
                    if is_requires_grad:
                        tensor.requires_grad_()
                    detached_output_obj.append(tensor)

            # 记录每一个断点的input和output，以便在backward手动执行计算图
            try:
                self._input_objs[pipe_id][input_obj_idx].append(detached_output_obj)
            except AttributeError:
                item = self._input_objs[pipe_id][input_obj_idx]
                self._input_objs[pipe_id][input_obj_idx] = [item, detached_output_obj]
            except IndexError:
                self._input_objs[pipe_id].append([detached_output_obj])
        else:
            detached_output_obj = output_objs

        return detached_output_obj

    def _save_comm_handler(self, handle) -> None:
        self._backward_a2a_comm_handler = handle

    @torch.no_grad()
    def _inpterrupt_forward_excute_backward(self, handle) -> None:
        if self._backward_a2a_comm_handler is not None:
            self._backward_a2a_comm_handler.wait()

        if self._partial_backward_func is not None:
            self._partial_backward_func()

        handle.wait()

    def _fuse_fwd_bwd(self, engine, fwd_pipe_id, bwd_pipe_id, is_pure_backward):
        """Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.
        """
        backward_cnt = 0

        def _partial_backward():
            nonlocal backward_cnt

            if backward_cnt == 0:
                moe_loss = self._moe_losses[bwd_pipe_id].pop(0)
            else:
                moe_loss = None

            input_obj = self._input_objs[bwd_pipe_id][0].pop()
            output_obj = self._output_objs[bwd_pipe_id][0].pop()
            output_obj_grad = self._output_obj_grads[bwd_pipe_id].pop(0)

            self._current_pipe_id = bwd_pipe_id
            # debug_print(f"do fuse partial backward{backward_cnt}")
            input_obj_grad = self._engine_backward_step(engine, 0, input_obj, output_obj, output_obj_grad, moe_loss)

            WeightGradStore.flush()
            WeightGradStore.pop()
            self._current_pipe_id = fwd_pipe_id

            self._output_obj_grads[bwd_pipe_id].append(input_obj_grad)

            backward_cnt += 1

        if self._is_pipeline_last_stage(bwd_pipe_id) and len(self._output_obj_grads[bwd_pipe_id]) == 0:
            self._output_obj_grads[bwd_pipe_id].append(None)

        self._partial_backward_func = _partial_backward

        # 提交第一个layer的backward a2a
        _partial_backward()

        if not is_pure_backward:
            # 执行 forward 过程，并在forward的两个a2a断点处，插空执行 backward
            self._current_pipe_id = fwd_pipe_id
            output_obj = self._forward_step(engine, fwd_pipe_id)
        else:
            output_obj = None
            while len(self._output_objs[bwd_pipe_id][0]) > 0:
                _partial_backward()

        self._input_objs[bwd_pipe_id].pop(0)
        self._output_objs[bwd_pipe_id].pop(0)
        self._input_obj_idx[bwd_pipe_id] -= 1
        self._partial_backward_func = None

        input_obj_grad = self._output_obj_grads[bwd_pipe_id].pop()

        return output_obj, input_obj_grad

    def _forward_step(self, engine, pipe_id):
        """Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.
        """

        debug_print(
            f"foward_step_before: {self._input_obj_idx[pipe_id]}, {[len(a) if isinstance(a, list) else 1 for a in self._input_objs[pipe_id]]}"
        )

        input_obj = self._input_objs[pipe_id][self._input_obj_idx[pipe_id]]

        if not self._is_pipeline_first_stage(pipe_id):
            assert input_obj is not None, f"{gpc.get_global_rank()} input is None"

        # TODO(chenxun): 需要确保micro batches 是正确的分片
        micro_batch_data = self.load_micro_batch(pipe_id)
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        # TODO: FIXME
        # self._call_hooks("before_forward", data)
        if hasattr(gpc.config.model, "num_experts"):
            output_obj, moe_losses = self._call_engine(engine.model[pipe_id], data)
        else:
            output_obj = self._call_engine(engine.model[pipe_id], data)
        # Convert output_obj to fp32 when last model chunk of last stage
        if self._is_pipeline_last_stage(pipe_id) and isinstance(engine.model[pipe_id], NaiveAMPModel):
            output_obj = engine.model[pipe_id].convert_to_fp32(output_obj)
        # TODO: FIXME
        # self._call_hooks("after_forward", output_obj)

        if self._is_pipeline_last_stage(pipe_id):
            self._call_hooks("post_helper_func", output_obj, label)

            if self._return_tensors is not None:
                self._return_tensors.append((output_obj, label))
            if self._accum_loss is not None:
                # TODO: FIXME
                # self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                # TODO: FIXME
                # self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                self._accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        moe_loss = (
            sum(moe_losses) * gpc.config.loss.moe_loss_coeff  # pylint: disable=E0606
            if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1
            else torch.tensor(0.0, device=get_current_device(), dtype=gpc.config.model.get("dtype"))
        )
        # the moe_loss is computed among the "tensor" group if sequence parallel is enabled, so we need to do allreduce
        if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
            dist.all_reduce(moe_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.TENSOR))
        moe_loss /= self.num_microbatches

        if self._accum_moe_loss is not None:
            self._accum_moe_loss.add_(moe_loss.detach())

        self._moe_losses[pipe_id].append(moe_loss)

        if len(self._tmp_output_objs) > 0:
            self._tmp_output_objs.append(output_obj)
            self._output_objs[pipe_id].append(self._tmp_output_objs)
            self._tmp_output_objs = []
        else:
            self._output_objs[pipe_id].append(output_obj)

        assert output_obj is not None, f"{gpc.get_global_rank()} chunk{pipe_id} output is None"

        if self._output_obj_shapes[pipe_id] is None:
            self._output_obj_shapes[pipe_id] = _get_tensor_or_tensors_shape(output_obj)

        self._input_obj_idx[pipe_id] += 1

        debug_print(
            f"foward_step_after: {self._input_obj_idx[pipe_id]}, {[len(a) if isinstance(a, list) else 1 for a in self._input_objs[pipe_id]]}"
        )

        return output_obj

    def _backward_step(self, engine, pipe_id, step_id):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        # TODO: fix step_id for skip grad reduce.

        debug_print(
            f"input_backward: {pipe_id=}, {[len(a) if isinstance(a, list) else 1 for a in self._input_objs[0]]}, {[len(a) if isinstance(a, list) else 1 for a in self._input_objs[1]]}"
        )
        debug_print(
            f"output_backward: {pipe_id=}, {[len(a) if isinstance(a, list) else 1 for a in self._output_objs[0]]}, {[len(a) if isinstance(a, list) else 1 for a in self._output_objs[1]]}"
        )

        if self._is_pipeline_last_stage(pipe_id) and len(self._output_obj_grads[pipe_id]) == 0:
            self._output_obj_grads[pipe_id].append(None)

        input_obj = self._input_objs[pipe_id].pop(0)
        output_obj = self._output_objs[pipe_id].pop(0)
        self._input_obj_idx[pipe_id] -= 1
        output_obj_grad = self._output_obj_grads[pipe_id].pop(0)
        moe_loss = self._moe_losses[pipe_id].pop(0)

        input_obj_grad = self._engine_backward_step(engine, step_id, input_obj, output_obj, output_obj_grad, moe_loss)

        WeightGradStore.flush()
        return input_obj_grad

    def _engine_backward_step(self, engine, step_id, input_obj, output_obj, output_obj_grad, moe_loss=None):
        """
        Backward step through the passed-in output tensor. If it is the last stage, the
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            step_id (int): The ID of the current step.
            input_obj (Union[torch.Tensor, List[torch.Tensor]]): Input tensor for this stage.
            output_obj (Union[torch.Tensor, List[torch.Tensor]]): Output tensor for this stage.
            output_obj_grad (Union[torch.Tensor, List[torch.Tensor]]): Gradient of output tensor for this stage.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor) and input_obj.requires_grad:
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None and isinstance(in_tensor, torch.Tensor) and in_tensor.requires_grad:
                        in_tensor.retain_grad()

        # Backward pass.
        if not isinstance(output_obj, torch.Tensor):
            output_obj_, output_obj_grad_ = [], []
            for _idx, output in enumerate(output_obj):
                if output is None or isinstance(output, torch.Tensor) is False  or output.requires_grad is False:
                    continue

                output_obj_.append(output)
                output_obj_grad_.append(output_obj_grad[_idx])
            output_obj, output_obj_grad = output_obj_, output_obj_grad_

        # Only the last microbatch does syncing grad.
        # TODO: FIXME
        skip_grad_sync = False

        # TODO: FIXME
        # self._call_hooks("before_backward", output_obj, output_obj_grad)
        if moe_loss is None or torch.equal(moe_loss, self._zero_tensor):
            if output_obj_grad is None:
                engine.backward(output_obj)
            else:
                engine.backward_by_grad(output_obj, output_obj_grad)
        else:
            if output_obj_grad is None:
                engine.backward(output_obj + moe_loss)
            else:
                # scale the latent loss
                moe_loss = moe_loss * engine.optimizer.loss_scale
                # we perform chain rule here by projecting the grad to the direction of
                # [output_obj_grad, 1], Because moe_loss have no relation with subsequent
                # layer, we set it to None (will be ragarded as 1).
                engine.backward_by_grad([output_obj, moe_loss], [output_obj_grad, None])

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    if in_tensor is not None and isinstance(in_tensor, torch.Tensor) and in_tensor.requires_grad:
                        input_obj_grad.append(in_tensor.grad)
                    else:
                        input_obj_grad.append(None)
        # TODO: FIXME
        # self._call_hooks("after_backward", input_obj_grad)

        return input_obj_grad

    def _run_warmup_loop(
        self,
        engine: Engine,
        num_microsteps: int,
        num_warmup_microsteps: int,
        receive_extra_backward: bool = False,
        forward_only: bool = False,
    ) -> None:
        """
        Run the warm-up loop
        """
        micro_step_id = 0

        num_self_pipe = (self._half_pp_size - self._lower_pp_rank - 1) * 2

        # 阶段1：上一半pp stage执行up stream, 下一半pp stage执行down stream.
        pipe_stream_id = 0 if self._lower_pp_rank == self._down_stream_rank else 1
        prev_rank, next_rank = self._get_prev_next_rank(pipe_stream_id)

        self._current_pipe_id = pipe_stream_id

        debug_print(
            f"warmup phase1: {num_self_pipe=}, { self._lower_pp_rank =}, {self._down_stream_rank=}, {self._up_stream_rank=}, {pipe_stream_id =}, {prev_rank=}, {next_rank=}"
        )

        if self._is_pipeline_first_stage(pipe_stream_id):
            self._input_objs[pipe_stream_id].append(None)
        else:
            if self._input_obj_shapes[pipe_stream_id] is None:
                self._input_obj_shapes[pipe_stream_id] = comm.recv_obj_meta()

            debug_print(f"recv forward input {self._input_obj_shapes[pipe_stream_id]} for rank{prev_rank}")

            self._input_objs[pipe_stream_id].append(
                comm.recv_forward(
                    self._input_obj_shapes[pipe_stream_id],
                    prev_rank=prev_rank,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )

        for i in range(num_self_pipe):
            debug_print(f"step {micro_step_id}, do phase1 forward compute {pipe_stream_id}")
            output_obj = self._forward_step(engine, pipe_stream_id)

            if not self._is_pipeline_last_stage(pipe_stream_id):

                if self._send_tensor_shape_flags[pipe_stream_id]:
                    comm.send_obj_meta(output_obj)
                    self._send_tensor_shape_flags[pipe_stream_id] = False  # send only once for each chunk.

            if self._is_pipeline_last_stage(pipe_stream_id):
                output_obj = None

            if self._is_pipeline_first_stage(pipe_stream_id):
                input_shape = None
            else:
                input_shape = self._input_obj_shapes[pipe_stream_id]

            # Normal warm-up communication process, or no need to prepare backward input for the 1F1B stage
            debug_print(
                f"step {micro_step_id}, send_forward_recv_forward, output_obj={getattr(output_obj, 'shape', None)}, {input_shape=}, {prev_rank=}, {next_rank=}"
            )
            input_obj = comm.send_forward_recv_forward(
                output_obj,
                input_shape,
                prev_rank=prev_rank,
                next_rank=next_rank,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )

            micro_step_id += 1
            self._input_objs[pipe_stream_id].append(input_obj)

        # 阶段2：cross pipe, 每个rank交替执行up stream和down stream
        num_cross_pipe = 2 * (self._lower_pp_rank + 1)
        debug_print(f"warmup phase2: {num_cross_pipe=}")
        for i in range(num_cross_pipe):
            next_pipe_stream_id = (pipe_stream_id + 1) % self._num_pipe_stream
            self._current_pipe_id = pipe_stream_id

            prev_rank, next_rank = self._get_prev_next_rank(pipe_stream_id)
            next_prev_rank, _ = self._get_prev_next_rank(next_pipe_stream_id)

            debug_print(f"step {micro_step_id}, do phase2 forward compute, {pipe_stream_id}")
            output_obj = self._forward_step(engine, pipe_stream_id)

            if self._is_pipeline_last_stage(pipe_stream_id):
                output_obj = None
            else:
                if self._send_tensor_shape_flags[pipe_stream_id]:
                    comm.send_obj_meta(output_obj)
                    self._send_tensor_shape_flags[pipe_stream_id] = False

            # Determine if tensor should be received from previous stage.
            if (
                not self._is_pipeline_first_stage(next_pipe_stream_id)
                and self._input_obj_shapes[next_pipe_stream_id] is None
            ):
                self._input_obj_shapes[next_pipe_stream_id] = comm.recv_obj_meta()

            if i == (num_cross_pipe - 1) or self._is_pipeline_first_stage(next_pipe_stream_id):
                input_shape = None
            else:
                input_shape = self._input_obj_shapes[next_pipe_stream_id]

            if i != (num_cross_pipe - 1):
                if i == 0:
                    if self._is_pipeline_first_stage(pipe_stream_id):
                        self._input_objs[pipe_stream_id].append(None)
                    else:
                        _input_shape = self._input_obj_shapes[pipe_stream_id]
                        debug_print(f"step {micro_step_id}, recv_forward, {input_shape=}, {prev_rank=}")
                        input_obj = comm.recv_forward(
                            _input_shape,
                            prev_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        self._input_objs[pipe_stream_id].append(input_obj)

                debug_print(
                    f"step {micro_step_id}, send_forward_recv_forward, output_obj={getattr(output_obj, 'shape', None)}, {input_shape=}, {next_prev_rank=}, {next_rank=}"
                )
                input_obj = comm.send_forward_recv_forward(
                    output_obj,
                    input_shape,
                    prev_rank=next_prev_rank,
                    next_rank=next_rank,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
                self._input_objs[next_pipe_stream_id].append(input_obj)
            else:
                if not self._is_pipeline_last_stage(pipe_stream_id):
                    output_obj_shape = self._output_obj_shapes[pipe_stream_id]

                    debug_print(
                        f"step {micro_step_id}, send_forward_recv_backward, output_obj={getattr(output_obj, 'shape', None)}, {output_obj_shape=}, {next_rank=}"
                    )
                    output_obj_grad = comm.send_forward_recv_backward(
                        output_obj,
                        output_obj_shape,
                        next_rank=next_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    self._output_obj_grads[pipe_stream_id].append(output_obj_grad)

            micro_step_id += 1
            pipe_stream_id = next_pipe_stream_id

        # 阶段3：cross_fwd_bwd, 每个rank交替执行b,w,f.
        num_cross_fwd_bwd = self._half_pp_size - self._lower_pp_rank - 1
        pipe_stream_id = 1 if self._lower_pp_rank == self._down_stream_rank else 0
        prev_rank, next_rank = self._get_prev_next_rank(pipe_stream_id)
        self._current_pipe_id = pipe_stream_id

        for i in range(num_cross_fwd_bwd):
            # 执行一个b, 发送output_grad给下一个rank
            debug_print(f"step {micro_step_id}, do phase3 backward compute, {pipe_stream_id}")
            input_obj_grad = self._backward_step(engine, pipe_stream_id, 0)
            micro_step_id += 1

            # TODO: first_last check
            if not self._is_pipeline_first_stage(pipe_stream_id):
                debug_print(
                    f"step {micro_step_id}, send_backward_recv_forward, {input_obj_grad.shape=}, {self._input_obj_shapes[pipe_stream_id]}, {prev_rank=}"
                )
                input_obj = comm.send_backward_recv_forward(
                    input_obj_grad,
                    self._input_obj_shapes[pipe_stream_id],
                    prev_rank=prev_rank,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
                self._input_objs[pipe_stream_id].append(input_obj)

            # 执行一个w
            WeightGradStore.pop()

            # 执行一个f, 发送output给下一个rank, 并接收 output_obj_grad
            debug_print(f"step {micro_step_id}, do phase3 forward compute, {pipe_stream_id}")
            output_obj = self._forward_step(engine, pipe_stream_id)
            micro_step_id += 1

            # TODO: first_last check
            if not self._is_pipeline_last_stage(pipe_stream_id):
                output_obj_shape = self._output_obj_shapes[pipe_stream_id]

                debug_print(
                    f"step {micro_step_id}, phase3 send_forward_recv_backward, output_obj={getattr(output_obj, 'shape', None)}, {output_obj_shape=}, {next_prev_rank=}, {next_rank=}"
                )
                output_obj_grad = comm.send_forward_recv_backward(
                    output_obj,
                    output_obj_shape,
                    dtype=self.dtype,
                    next_rank=next_rank,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
                self._output_obj_grads[pipe_stream_id].append(output_obj_grad)

    def _run_1f1b_loop_with_overlap(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        num_1f1b_microbatches: int,
        all_warmup_microsteps: bool = False,
    ) -> None:
        """
        Run the 1F1B loop with overlap.

        The 1F1B loop with overlap consists of the following steps:
        1. Perform the forward pass.
        2. Check if the backward input is ready.
        3. Send the forward output and receive the forward input for the next iteration.
        4. Perform the backward pass.
        5. Check if the forward input is ready.
        6. Send the backward output and receive the backward input for the next iteration.

        Args:
            engine (Engine): The engine to run the 1F1B loop.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micropairs.
            all_warmup_microsteps (bool, optional): Whether to run all warm-up microsteps. Default is False.
        """

        debug_print(f"enter 1f1b: {len(self._output_obj_grads[0])=}, {len(self._output_obj_grads[1])=}")

        left_fwd_microbatches = list(num_1f1b_microbatches)

        # phase1: cross-pipe, d_fwd/u_bwd+u_fwd/d_bwd
        num_cross_pipe = max(num_1f1b_microbatches) * 2 + self._lower_pp_rank + 1
        fwd_pipe_stream_id = 0 if self._lower_pp_rank == self._down_stream_rank else 1
        bwd_pipe_stream_id = (fwd_pipe_stream_id + 1) % self._num_pipe_stream

        debug_print(f"1f1b phase1: {num_cross_pipe=}, {fwd_pipe_stream_id=}, {bwd_pipe_stream_id=}")

        for i in range(num_cross_pipe):
            next_fwd_pipe_stream_id, next_bwd_pipe_stream_id = (bwd_pipe_stream_id, fwd_pipe_stream_id)
            debug_print(f"step1f1b {i}, {fwd_pipe_stream_id=}, {bwd_pipe_stream_id=}")

            is_pure_backward = left_fwd_microbatches[fwd_pipe_stream_id] <= 0
            # # Forward pass.
            # if left_fwd_microbatches[fwd_pipe_stream_id] > 0:
            #     output_obj = self._forward_step(engine, fwd_pipe_stream_id)
            #     _sum = torch.sum(output_obj) if output_obj is not None else None
            #     torch.cuda.synchronize()
            #     debug_print(f"step1f1b {i}, 1f1b forward compute done, {fwd_pipe_stream_id}, {_sum}")
            #     left_fwd_microbatches[fwd_pipe_stream_id] -= 1
            # else:
            #     output_obj = None

            # # Backward pass.
            # input_obj_grad = self._backward_step(engine, bwd_pipe_stream_id, 0)
            # WeightGradStore.pop()

            output_obj, input_obj_grad = self._fuse_fwd_bwd(
                engine, fwd_pipe_stream_id, bwd_pipe_stream_id, is_pure_backward
            )

            if not is_pure_backward:
                left_fwd_microbatches[fwd_pipe_stream_id] -= 1

            if self._is_pipeline_last_stage(fwd_pipe_stream_id):
                output_obj = None

            if self._is_pipeline_first_stage(bwd_pipe_stream_id):
                input_obj_grad = None
            else:
                # TODO: FIXME!!!!
                if input_obj_grad is None:
                    debug_print(
                        f"Warning: step1f1b {i}, input_obj_grad is None, create a fake input_obj_grad {fwd_pipe_stream_id=}, {bwd_pipe_stream_id=}"
                    )
                    input_obj_grad = torch.empty(
                        [1, 2048, 4096], dtype=self.dtype, device=get_current_device(), requires_grad=True
                    )
                _sum = torch.sum(input_obj_grad) if input_obj_grad is not None else None
                torch.cuda.synchronize()
                debug_print(
                    f"step1f1b {i}, 1f1b fused fwd_bwd compute done, {fwd_pipe_stream_id}, {bwd_pipe_stream_id}, {_sum}"
                )

            if (
                self._is_pipeline_first_stage(next_fwd_pipe_stream_id)
                or left_fwd_microbatches[next_fwd_pipe_stream_id] <= 0
            ):
                recv_prev = False
            else:
                recv_prev = True

            if self._is_pipeline_last_stage(next_bwd_pipe_stream_id):
                recv_next = False
            else:
                recv_next = True

            _, next_rank = self._get_prev_next_rank(fwd_pipe_stream_id)
            prev_rank, _ = self._get_prev_next_rank(next_fwd_pipe_stream_id)
            input_shape = self._input_obj_shapes[next_fwd_pipe_stream_id] if recv_prev else None
            output_grad_shape = self._output_obj_shapes[next_bwd_pipe_stream_id] if recv_next else None

            # Communicate objs.
            assert output_obj is None or output_obj.dtype == self.dtype
            assert prev_rank == next_rank, f"pp{self._pp_rank}: prev_rank{prev_rank} != next_rank{next_rank}"
            debug_print(
                f"step1f1b {i}, send_forward_backward_recv_forward_backward, output_obj={getattr(output_obj, 'shape', None)}, input_obj_grad={getattr(input_obj_grad, 'shape', None)}, {input_shape=}, {output_grad_shape=}, {prev_rank=}, {next_rank=}"
            )
            input_obj, output_obj_grad = comm.send_forward_backward_recv_forward_backward(
                output_obj,
                input_obj_grad,
                input_shape,
                output_grad_shape,
                prev_rank=prev_rank,
                next_rank=next_rank,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )
            debug_print(f"step1f1b {i}, send_forward_backward_recv_forward_backward done")

            # Put input_obj and output_obj_grad in data structures in the
            # right location.
            if recv_prev:
                self._input_objs[next_fwd_pipe_stream_id].append(input_obj)
            elif left_fwd_microbatches[next_fwd_pipe_stream_id] > 0:
                self._input_objs[next_fwd_pipe_stream_id].append(None)

            if recv_next:
                self._output_obj_grads[next_bwd_pipe_stream_id].append(output_obj_grad)

            fwd_pipe_stream_id, bwd_pipe_stream_id = (next_fwd_pipe_stream_id, next_bwd_pipe_stream_id)

        # receive necessary data for next cooldown loop
        # if all_warmup_microsteps:
        #     if not gpc.is_pipeline_last_stage():
        #         self._output_obj_grads[self._num_chunks - 1].append(
        #             comm.recv_backward(
        #                 self._output_obj_shapes[self._num_chunks - 1],
        #                 dtype=self.dtype,
        #                 scatter_gather_tensors=self.scatter_gather_tensors,
        #             )
        #         )
        #     else:
        #         self._output_obj_grads[self._num_chunks - 1].append(None)
        debug_print(f"1f1b end")

    def _run_1f1b_loop_without_overlap(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        num_1f1b_microbatches: int,
        all_warmup_microsteps: bool = False,
    ) -> None:
        """
        Run the 1F1B loop without overlap.

        The 1F1B loop without overlap consists of the following steps:
        1. Perform the forward pass.
        2. Perform the backward pass.
        3. Send the forward output of this iteration to the next stage, and send the backward output of this iteration
           to the previous stage, and receive the forward and backward inputs for the next iteration.

        Args:
            engine (Engine): The engine to use for computation.
            num_warmup_microsteps (int): The number of warmup microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micro-pairs.
            all_warmup_microsteps (bool, optional): Whether to run all warmup microsteps. Defaults to False.
        """

        debug_print(f"enter 1f1b: {len(self._output_obj_grads[0])=}, {len(self._output_obj_grads[1])=}")

        left_fwd_microbatches = list(num_1f1b_microbatches)

        # phase1: cross-pipe, d_fwd/u_bwd+u_fwd/d_bwd
        num_cross_pipe = max(num_1f1b_microbatches) * 2 + self._lower_pp_rank + 1
        fwd_pipe_stream_id = 0 if self._lower_pp_rank == self._down_stream_rank else 1
        bwd_pipe_stream_id = (fwd_pipe_stream_id + 1) % self._num_pipe_stream

        debug_print(f"1f1b phase1: {num_cross_pipe=}, {fwd_pipe_stream_id=}, {bwd_pipe_stream_id=}")

        for i in range(num_cross_pipe):
            next_fwd_pipe_stream_id, next_bwd_pipe_stream_id = (bwd_pipe_stream_id, fwd_pipe_stream_id)
            debug_print(f"step1f1b {i}, {fwd_pipe_stream_id=}, {bwd_pipe_stream_id=}")
            # Forward pass.
            if left_fwd_microbatches[fwd_pipe_stream_id] > 0:
                output_obj = self._forward_step(engine, fwd_pipe_stream_id)
                _sum = torch.sum(output_obj) if output_obj is not None else None
                torch.cuda.synchronize()
                debug_print(f"step1f1b {i}, 1f1b forward compute done, {fwd_pipe_stream_id}, {_sum}")
                left_fwd_microbatches[fwd_pipe_stream_id] -= 1
            else:
                output_obj = None

            # Backward pass.
            input_obj_grad = self._backward_step(engine, bwd_pipe_stream_id, 0)
            # TODO: FIXME!!!!
            if input_obj_grad is None:
                input_obj_grad = torch.empty(
                    [1, 2048, 4096], dtype=self.dtype, device=get_current_device(), requires_grad=True
                )
            _sum = torch.sum(input_obj_grad) if input_obj_grad is not None else None
            torch.cuda.synchronize()
            debug_print(f"step1f1b {i}, 1f1b backward compute done, {bwd_pipe_stream_id}, {_sum}")

            if self._is_pipeline_last_stage(fwd_pipe_stream_id):
                output_obj = None

            if self._is_pipeline_first_stage(bwd_pipe_stream_id):
                input_obj_grad = None

            if (
                self._is_pipeline_first_stage(next_fwd_pipe_stream_id)
                or left_fwd_microbatches[next_fwd_pipe_stream_id] <= 0
            ):
                recv_prev = False
            else:
                recv_prev = True

            if self._is_pipeline_last_stage(next_bwd_pipe_stream_id):
                recv_next = False
            else:
                recv_next = True

            _, next_rank = self._get_prev_next_rank(fwd_pipe_stream_id)
            prev_rank, _ = self._get_prev_next_rank(next_fwd_pipe_stream_id)
            input_shape = self._input_obj_shapes[next_fwd_pipe_stream_id] if recv_prev else None
            output_grad_shape = self._output_obj_shapes[next_bwd_pipe_stream_id] if recv_next else None

            # Communicate objs.
            assert output_obj is None or output_obj.dtype == self.dtype
            assert prev_rank == next_rank, f"pp{self._pp_rank}: prev_rank{prev_rank} != next_rank{next_rank}"
            debug_print(
                f"step1f1b {i}, send_forward_backward_recv_forward_backward, output_obj={getattr(output_obj, 'shape', None)}, input_obj_grad={getattr(input_obj_grad, 'shape', None)}, {input_shape=}, {output_grad_shape=}, {prev_rank=}, {next_rank=}"
            )
            input_obj, output_obj_grad = comm.send_forward_backward_recv_forward_backward(
                output_obj,
                input_obj_grad,
                input_shape,
                output_grad_shape,
                prev_rank=prev_rank,
                next_rank=next_rank,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )
            debug_print(f"step1f1b {i}, send_forward_backward_recv_forward_backward done")

            # Put input_obj and output_obj_grad in data structures in the
            # right location.
            if recv_prev:
                self._input_objs[next_fwd_pipe_stream_id].append(input_obj)
            if recv_next:
                self._output_obj_grads[next_bwd_pipe_stream_id].append(output_obj_grad)

            fwd_pipe_stream_id, bwd_pipe_stream_id = (next_fwd_pipe_stream_id, next_bwd_pipe_stream_id)

        # receive necessary data for next cooldown loop
        # if all_warmup_microsteps:
        #     if not gpc.is_pipeline_last_stage():
        #         self._output_obj_grads[self._num_chunks - 1].append(
        #             comm.recv_backward(
        #                 self._output_obj_shapes[self._num_chunks - 1],
        #                 dtype=self.dtype,
        #                 scatter_gather_tensors=self.scatter_gather_tensors,
        #             )
        #         )
        #     else:
        #         self._output_obj_grads[self._num_chunks - 1].append(None)
        debug_print(f"1f1b end")

    def _run_cooldown_loop(self, engine: Engine, num_microsteps: int, num_1f1b_micropairs: int) -> None:

        # 分成三个阶段
        mb_idx = 0

        # 阶段1：连续pipe_stream交错的B
        num_cross_pipe = self._lower_pp_rank
        bwd_pipe_id = 0 if self._pp_rank % 2 == 0 else 1
        debug_print(f"cooldown phase1: {num_cross_pipe=}, {bwd_pipe_id=}")

        for _ in range(num_cross_pipe):
            prev_rank, _ = self._get_prev_next_rank(bwd_pipe_id)
            next_bwd_pipe_id = (bwd_pipe_id + 1) % self._num_pipe_stream
            # 执行一个b, 发送output_grad给下一个rank
            debug_print(f"cooldown step {mb_idx}, do phase1 backward compute, {bwd_pipe_id}")
            input_obj_grad = self._backward_step(engine, bwd_pipe_id, 0)

            debug_print(
                f"cooldown step {mb_idx}, send_backward_recv_backward, {input_obj_grad.shape=}, {self._output_obj_shapes[bwd_pipe_id]}, {prev_rank=}"
            )
            output_obj_grads = comm.send_backward_recv_backward(
                input_obj_grad,
                self._output_obj_shapes[bwd_pipe_id],
                prev_rank=prev_rank,
                next_rank=prev_rank,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )
            self._output_obj_grads[bwd_pipe_id].append(output_obj_grads)

            bwd_pipe_id = next_bwd_pipe_id
            mb_idx += 1

        # 阶段2：交错的b,w
        num_cross_bw = self._half_pp_size - self._lower_pp_rank
        bwd_pipe_id = 0 if self._lower_pp_rank == self._down_stream_rank else 1
        prev_rank, _ = self._get_prev_next_rank(bwd_pipe_id)
        debug_print(f"cooldown phase2: {num_cross_pipe=}, {bwd_pipe_id=}")

        for _ in range(num_cross_bw):
            debug_print(f"cooldown step {mb_idx}, do phase2 backward compute, {bwd_pipe_id}")
            input_obj_grad = self._backward_step(engine, bwd_pipe_id, 0)

            if self._is_pipeline_last_stage(bwd_pipe_id):
                output_grad_shape = None
            else:
                output_grad_shape = self._output_obj_shapes[bwd_pipe_id]

            if self._is_pipeline_last_stage(bwd_pipe_id):
                input_obj_grad = None
            else:
                assert input_obj_grad is not None

            debug_print(
                f"cooldown step {mb_idx}, phase2 send_backward_recv_backward, {getattr(input_obj_grad, 'shape', None)}, {output_grad_shape}, {prev_rank=}"
            )
            output_obj_grads = comm.send_backward_recv_backward(
                input_obj_grad,
                self._output_obj_shapes[bwd_pipe_id],
                prev_rank=prev_rank,
                next_rank=prev_rank,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )
            self._output_obj_grads[bwd_pipe_id].append(output_obj_grads)

            # 执行一个w
            WeightGradStore.pop()

            mb_idx += 1

        # 阶段3：flush所有余下的w
        while WeightGradStore.size() > 0:
            WeightGradStore.pop()

        # for k in range(num_1f1b_micropairs, num_microsteps):
        #     chunk_id = self._get_chunk_by_microbatch(k, backward=True)

        #     input_obj_grad = self._backward_step(engine, chunk_id, k)

        #     next_backward_chunk_id = self._get_chunk_by_microbatch(k + 1, backward=True)

        #     if k != (num_microsteps - 1) and not (
        #         gpc.is_pipeline_last_stage(ignore_virtual=True)
        #         and next_backward_chunk_id == (self._num_pipe_stream - 1)
        #     ):
        #         output_shape = self._output_obj_shapes[next_backward_chunk_id]
        #     else:
        #         output_shape = None

        #     self._output_obj_grads[next_backward_chunk_id].append(
        #         comm.send_backward_recv_backward(
        #             input_obj_grad,
        #             output_shape,
        #             dtype=self.dtype,
        #             scatter_gather_tensors=self.scatter_gather_tensors,
        #         )
        #     )

    def _forward_backward_step(self, engine: Engine):
        # Compute number of warmup and remaining microbatches.
        all_warmup_microsteps = False
        num_half_microbatches = self.num_microbatches // 2

        if False:  # TODO(chenxun): num_microbatches=?的时候会导致all warmup
            num_warmup_microbathes = num_half_microbatches
            all_warmup_microsteps = True
        else:
            if self._down_stream_rank == self._lower_pp_rank:
                num_warmup_microbathes = (self._pp_size - self._lower_pp_rank - 1, self._half_pp_size)
            else:
                num_warmup_microbathes = (self._half_pp_size, self._pp_size - self._lower_pp_rank - 1)

        num_1f1b_microbatches = (
            num_half_microbatches - num_warmup_microbathes[0],
            num_half_microbatches - num_warmup_microbathes[1],
        )

        # We usually need to prepare an extra backward data for the 1F1B stage when the WarmUp stage ends,
        # because the 1F1B stage typically performs one forward and backward pass together,
        # except in the following cases:
        # receive_extra_backward = not (
        #     all_warmup_microsteps  # Only warmup microsteps
        #     or gpc.is_pipeline_last_stage(ignore_virtual=True)  # The rank is the last pipeline stage
        # )

        # 1. Warmup
        self._run_warmup_loop(
            engine,
            num_half_microbatches,
            num_warmup_microbathes,
        )

        # 2. 1F1B
        self._run_1f1b_loop(
            engine,
            num_warmup_microbathes,
            num_1f1b_microbatches=num_1f1b_microbatches,
            all_warmup_microsteps=all_warmup_microsteps,
        )

        # 3. Cooldown
        # self._run_cooldown_loop(engine, num_half_microbatches, num_1f1b_micropairs=num_1f1b_microbatches)

    @llm_timeout(func_name="interleaved_forward_backward_step")
    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss, moe_loss), loss and label could be None.
                The loss would be returned only in the last stage. And the moe_loss is accumulated from all stages.
        """
        assert forward_only is False, "forward_only is not supported by DualPipe scheduler yet."

        # gpc.set_virtual_pipeline_parallel_rank(0)

        self.load_batch(engine, data_iter)

        # TODO(chenxun): add check for num_microbatches.
        assert self.num_microbatches % 2 == 0, f"micro_num {self.num_microbatches} should be odd number."
        # assert self.num_microbatches // 2 > ?

        if return_loss and (self._is_pipeline_last_stage(0) or self._is_pipeline_last_stage(1)):
            self._accum_loss = torch.zeros(1, device=get_current_device())
        self._accum_moe_loss = torch.zeros(1, device=get_current_device())

        if return_output_label:
            self._return_tensors = []

        self._forward_backward_step(engine)

        if return_output_label and len(self._return_tensors) > 0:
            output, label = pack_return_tensors(self._return_tensors)
        else:
            output, label = (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(self._accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))
        accum_moe_loss = self._accum_moe_loss

        accum_loss = self._accum_loss
        if accum_loss is not None:
            accum_loss += self._accum_moe_loss

        self._clear_state()

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return output, label, accum_loss, accum_moe_loss
        else:
            return output, label, accum_loss
