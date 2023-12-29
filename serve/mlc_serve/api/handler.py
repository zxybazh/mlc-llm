import time
import uuid
import json
import os

from http import HTTPStatus
from typing import Annotated, AsyncIterator, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# TODO(amalyshe): handle random_seed
# from .base import set_global_random_seed
from ..api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    Logprobs,
    LogprobsContent,
    TopLogprobs,
    UsageInfo,
)
from ..engine import (
    DebugOptions,
    Request,
    RequestOutput,
    SamplingParams,
    StoppingCriteria,
)
from ..engine.async_connector import AsyncEngineConnector
from .dependencies import get_async_engine_connector


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )


router = APIRouter()

def _get_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
    sampling_params = SamplingParams(
        # These params came from vllm
        # TODO(amnalyshe): should they be put into mlc-llm batch serving ChatCompletionRequest?
        # best_of=request.best_of,
        # top_k=request.top_k,
        # ignore_eos=request.ignore_eos,
        # use_beam_search=request.use_beam_search,
    )
    if request.presence_penalty is not None:
        sampling_params.presence_penalty = request.presence_penalty
    if request.frequency_penalty is not None:
        sampling_params.frequency_penalty = request.frequency_penalty
    if request.repetition_penalty is not None:
        sampling_params.repetition_penalty = request.repetition_penalty
    if request.temperature is not None:
        sampling_params.temperature = request.temperature
    if request.top_p is not None:
        sampling_params.top_p = request.top_p
    if request.logit_bias is not None:
        sampling_params.logit_bias = request.logit_bias
    if request.logprobs is not None:
        sampling_params.top_logprobs = request.top_logprobs
        sampling_params.logprobs = request.logprobs
    return sampling_params


@router.get("/metrics")
def metrics():
    # See https://prometheus.github.io/client_python/multiprocess/ for why we need this.
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        from prometheus_client import (
            CollectorRegistry,
            generate_latest,
            multiprocess,
        )
        from starlette.responses import Response

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        return Response(content=generate_latest(registry))
    else:
        return {}


@router.post("/v1/chat/completions")
async def request_completion(
    request: ChatCompletionRequest,
    async_engine_connector: Annotated[
        AsyncEngineConnector, Depends(get_async_engine_connector)
    ],
) -> ChatCompletionResponse:
    """
    Creates model response for the given chat conversation.
    """

    def random_uuid() -> str:
        return str(uuid.uuid4().hex)

    request_id = f"cmpl-{random_uuid()}"
    model_name = request.model
    try:
        sampling_params = _get_sampling_params(request)
    except ValueError as e:
        raise ValueError(
            """
            issues with sampling parameters
            """
        )
    # TODO(amalyshe): need to verify quantity, according to OpenAI API:
    # <<Up to 4 sequences where the API will stop generating further tokens.>>
    # The behavior in case of bigger number of stop sequences is unknown - need to shrink
    # or return error
    # To figure out how to handle properly and add verification
    if isinstance(request.stop, list):
        stop_sequences = request.stop
    else:
        stop_sequences = [request.stop]

    ignore_eos = False if request.ignore_eos is None else request.ignore_eos
    text_generation_request = Request(
        request_id=request_id,
        messages=request.messages,
        num_sequences=request.n,
        sampling_params=sampling_params,
        stopping_criteria=StoppingCriteria(max_tokens=request.max_tokens, stop_sequences=stop_sequences),
        debug_options=DebugOptions(ignore_eos=ignore_eos),
    )
    if isinstance(request.messages, str):
        text_generation_request.debug_options.prompt = request.messages

    result_generator = async_engine_connector.generate(text_generation_request)

    if request.stream:
        return StreamingResponse(
            generate_completion_stream(
                request_id, model_name, request.n, result_generator
            ),
            media_type="text/event-stream",
        )
    else:
        return await collect_result_stream(
            request_id, model_name, request.n, result_generator
        )


async def generate_completion_stream(
    request_id: str,
    model_name: str,
    num_sequences: int,
    result_generator: AsyncIterator[RequestOutput],
) -> AsyncIterator[str]:
    created_time = int(time.time())

    def create_stream_response(
        choices: List[ChatCompletionResponseStreamChoice],
    ) -> ChatCompletionStreamResponse:
        return ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
        )

    first_chunk = create_stream_response(
        choices=[
            ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
            for i in range(num_sequences)
        ],
    )
    yield f"data: {json.dumps(first_chunk.dict(exclude_unset=True), ensure_ascii=False)}\n\n"
    async for res in result_generator:
        if res.error:
            raise RuntimeError(f"Error when generating: {res.error}")
        chunk = create_stream_response(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=seq.index,
                    delta=(
                        DeltaMessage(content=seq.delta)
                        if seq.delta is not None
                        else DeltaMessage()
                    ),
                    finish_reason=seq.finish_reason.value
                    if seq.finish_reason is not None
                    else None,
                    logprob_info=seq.logprob_info[0] if seq.logprob_info else None
                )
                for seq in res.sequences
            ]
        )
        yield f"data: {json.dumps(chunk.dict(exclude_unset=True), ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


async def collect_result_stream(
    request_id: str,
    model_name: str,
    num_sequences: int,
    result_generator: AsyncIterator[RequestOutput],
) -> ChatCompletionResponse:
    created_time = int(time.time())
    sequences: List[List[str]] = [[] for _ in range(num_sequences)]
    finish_reasons = [None] * num_sequences
    num_prompt_tokens = 0
    num_generated_tokens = [0 for _ in range(num_sequences)]
    logprob_infos = [[] for _ in range(num_sequences)] # type: ignore
    async for res in result_generator:
        # TODO: verify that the request cancellation happens after this returns
        if res.error:
            raise RuntimeError(f"Error when generating: {res.error}")
        if res.num_prompt_tokens is not None:
            num_prompt_tokens = res.num_prompt_tokens
        for seq in res.sequences:
            if seq.logprob_info:
                logprob_infos[seq.index].append(seq.logprob_info)
            if seq.index >= len(sequences):
                raise RuntimeError(f"Unexpected sequence index: {seq.index}.")
            num_generated_tokens[seq.index] = seq.num_generated_tokens

            if seq.delta:
                sequences[seq.index].append(seq.delta)

            if seq.is_finished:
                assert seq.finish_reason is not None
                finish_reasons[seq.index] = seq.finish_reason.value  # type: ignore
    
    choices = []
    for index, (chunks, finish_reason) in enumerate(zip(sequences, finish_reasons)):
        content = []
        if logprob_infos[index] != []:
            for logprob_info in logprob_infos[index]:
                top_logprobs = [TopLogprobs(
                    token=str(token),
                    logprob=float(logprob),
                    # TODO(vvchernov): implement bytes bases on https://platform.openai.com/docs/api-reference/chat/object
                    bytes=None,
                ) for token, logprob in logprob_info[1]]
                content.append(LogprobsContent(
                    token=str(logprob_info[0][0]),
                    logprob=float(logprob_info[0][1]),
                    # TODO(vvchernov): implement bytes bases on https://platform.openai.com/docs/api-reference/chat/object
                    bytes=None,
                    top_logprobs=top_logprobs,
                ))
        choice = ChatCompletionResponseChoice(
            index=index,
            message=ChatMessage(role="assistant", content="".join(chunks)),
            finish_reason=finish_reason,
            logprobs=Logprobs(content=content),
        )
        choices.append(choice)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=sum(num_generated_tokens),
        total_tokens=num_prompt_tokens + sum(num_generated_tokens),
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    return response
