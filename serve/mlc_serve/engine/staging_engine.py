"""
An implementation of InferenceEngine that offloads the text generation loop to another worker process.
"""
import logging
import multiprocessing
import queue
from threading import Lock
from collections import defaultdict
from typing import Callable

import structlog

from .base import (
    FinishReason,
    InferenceStepResult,
    Request,
    RequestId,
    SequenceId,
    RequestOutput,
    RequestState,
    ScopedInferenceEngine,
    SequenceOutput,
)
from .engine_common import (
    get_new_request_state,
    update_sequence,
    logprobs_detokenize
)
from .model_module import ModelModule, TokenizerModule
from .staging_engine_worker import (
    AddRequestsCommand,
    CancelRequestCommand,
    StopSequenceCommand,
    ShutdownCommand,
    run_generation_loop_worker,
)

from ..logging_utils import log_every

LOG = structlog.stdlib.get_logger(__name__)


class StagingInferenceEngine(ScopedInferenceEngine):
    """
    An implementation of InferenceEngine that offloads the text generation loop to another worker process,
    Text tokens are generated asynchronously from the invocation of `step`. The generation progress could be one step
    ahead of the invocation of `step`. Tokenization and detokenization is still processed synchronously
    when `step` is called.
    """

    def __init__(
        self,
        tokenizer_module: TokenizerModule,
        model_module_loader: Callable[..., ModelModule],
        model_module_loader_kwargs: dict,
        # maybe find a better way to do this
        json_log_output: bool = False,
        init_timeout: int = 120,
    ):
        self.next_generation_output = None
        self.requests_lock = Lock()
        self.requests = dict[RequestId, RequestState]()

        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template

        self.mp_context = multiprocessing.get_context("spawn")
        self.command_queue = self.mp_context.Queue()
        self.result_queue = self.mp_context.Queue(maxsize=1)
        self.ready_event = self.mp_context.Event()
        self.init_timeout = init_timeout

        self.worker_process = self.mp_context.Process(
            target=run_generation_loop_worker,
            args=(
                model_module_loader,
                model_module_loader_kwargs,
                self.command_queue,
                self.result_queue,
                self.ready_event,
                # Log state
                structlog.contextvars.get_contextvars(),
                json_log_output,
                logging.getLevelName(logging.getLogger().level),
            ),
        )

    def start(self):
        LOG.info("StagingInferenceEngine.start")
        try:
            self.worker_process.start()
            if not self.ready_event.wait(timeout=self.init_timeout):
                raise RuntimeError(
                    "StagingInferenceEngine worker is not ready before timeout."
                )
        except:
            raise RuntimeError(
                f"Failed to start StagingInferenceEngine worker process with timeout {self.init_timeout}."
            )

    def stop(self):
        self.command_queue.put(ShutdownCommand())
        self.worker_process.join()

    def add(self, requests: list[Request]):
        LOG.info("StagingInferenceEngine.add", requests=requests)
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            # wrap the stop sequence with list if necessary
            if req.stopping_criteria.stop_sequences:
                if isinstance(req.stopping_criteria.stop_sequences, str):
                    req.stopping_criteria.stop_sequences = [
                        req.stopping_criteria.stop_sequences
                    ]
                assert isinstance(req.stopping_criteria.stop_sequences, list)

            # If the request violates the tokenization, this returns None, so skip.
            state = get_new_request_state(
                req, self.conversation_template, self.tokenizer
            )
            new_request_states.append(state)

        self.command_queue.put(AddRequestsCommand(request_states=new_request_states))

        with self.requests_lock:
            self.requests.update({s.request_id: s for s in new_request_states})

    def cancel(self, request_id: RequestId):
        LOG.info("StagingInferenceEngine.cancel", request_id=request_id)
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")
        self.command_queue.put(CancelRequestCommand(request_id))

    def stop_sequence(self, sequence_id: SequenceId):
        LOG.info("StagingInferenceEngine.stop_sequence", sequence_id=sequence_id)
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")
        self.command_queue.put(StopSequenceCommand(sequence_id))

    def has_pending_requests(self) -> bool:
        with self.requests_lock:
            return len(self.requests) > 0

    def wait_for_request(self, timeout_seconds=None) -> bool:
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")

        if self.next_generation_output is not None:
            return True

        try:
            self.next_generation_output = self.result_queue.get(timeout=timeout_seconds)
            return True
        except queue.Empty:
            return False

    def step(self) -> InferenceStepResult:
        log_every(
            self,
            1000,
            LOG.debug,
            "StagingInferenceEngine.step",
            _is_ready_to_serve=self._is_ready_to_serve(),
            has_pending_requests=self.has_pending_requests(),
        )

        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")

        if not self.has_pending_requests():
            return InferenceStepResult([])

        if self.next_generation_output is None:
            generation_output = self.result_queue.get()
        else:
            generation_output = self.next_generation_output
            self.next_generation_output = None

        if generation_output.error is not None:
            raise RuntimeError(
                f"Error from GenerationLoopWorker process: {generation_output.error}"
            ) from generation_output.error

        outputs = list[RequestOutput]()

        with self.requests_lock:
            LOG.debug(
                "StagingInferenceEngine.step obtained requests_lock",
                generation_output=generation_output,
            )

            seq_outputs = defaultdict(list)
            prompt_len = {}

            for seq_output in generation_output.sequences:
                request_id = seq_output.id.request_id
                if request_id not in self.requests:
                    LOG.warn(
                        "Unknown or already deleted request %s from GenerationLoopWorkerOutput",
                        request_id,
                    )
                    continue

                state = self.requests[request_id]

                with structlog.contextvars.bound_contextvars(**state.contextvars):
                    if seq_output.error is not None:
                        outputs.append(
                            RequestOutput(
                                request_id,
                                sequences=[],
                                error=seq_output.error,
                                num_prompt_tokens=state.prompt_len,
                            )
                        )
                        del self.requests[request_id]
                        continue

                    gen_seq = state.generation_sequences[seq_output.id.sequence_index]
                    new_token_ids = seq_output.new_tokens

                    if new_token_ids:
                        delta = update_sequence(
                            gen_seq,
                            new_token_ids,
                            state.prompt_token_ids,
                            self.tokenizer,
                            state.stopping_criteria,
                        )
                    else:
                        delta = None

                    if not state.is_prefilled:
                        # Successfully completed a prefill request
                        state.is_prefilled = True

                    finish_reason = seq_output.finish_reason

                    if seq_output.finish_reason is not None:
                        gen_seq.is_finished = True
                    elif gen_seq.is_finished:
                        # update_sequence() has detected a stop word
                        finish_reason = FinishReason.Stop
                        self.stop_sequence(gen_seq.seq_id)

                    output = SequenceOutput(
                        seq_output.id.sequence_index,
                        delta,
                        finish_reason,
                        num_generated_tokens=len(gen_seq.generated_token_ids),
                        logprob_info=logprobs_detokenize(self.tokenizer, seq_output.logprob_info),
                    )

                    seq_outputs[request_id].append(output)
                    prompt_len[request_id] = state.prompt_len

            for request_id, out_seqs in seq_outputs.items():
                outputs.append(
                    RequestOutput(
                        request_id,
                        sequences=out_seqs,
                        num_prompt_tokens=prompt_len[request_id],
                    )
                )
                if self.requests[request_id].is_finished:
                    del self.requests[request_id]

        return InferenceStepResult(outputs=outputs)

    def _is_ready_to_serve(self) -> bool:
        return self.worker_process is not None and self.worker_process.is_alive()
