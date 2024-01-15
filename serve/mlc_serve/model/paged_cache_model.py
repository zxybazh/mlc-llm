from pathlib import Path
from typing import List, Union

import structlog
import tvm

from .base import get_model_artifact_config
from .paged_cache_manager import CacheManager
from .tokenizer import HfTokenizerModule, ConversationTemplate, Tokenizer
from ..engine import MLCServeEngineConfig
from ..engine.model_module import (
    DecodeRequest,
    PrefillRequest,
    TextGenerationResult,
)
from ..engine.model_module import ModelModule

LOG = structlog.stdlib.get_logger(__name__)


class PagedCacheModelTextGenerator:
    def __init__(self, model: Model):
        self.model = model

    def generate(
        self, requests: List[Union[PrefillRequest, DecodeRequest]], kv_cache
    ) -> List[TextGenerationResult]:
        prefill_requests = [r for r in requests if isinstance(r, PrefillRequest)]
        decode_requests = [r for r in requests if isinstance(r, DecodeRequest)]

        out = []
        if prefill_requests:
            out.extend(self.model.generate(prefill_requests, kv_cache))
        if decode_requests:
            out.extend(self.model.generate(decode_requests, kv_cache))

        return out


class PagedCacheModelModule:
    engine_config: MLCServeEngineConfig
    text_generator: PagedCacheModelTextGenerator
    cache_manager: CacheManager
    tokenizer: Tokenizer
    conversation_template: ConversationTemplate

    def __init__(
        self,
        model_artifact_path: Path,
        engine_config: MLCServeEngineConfig,
    ):
        model_artifact_config = get_model_artifact_config(model_artifact_path)

        dev = tvm.device("cuda", 0)

        model = Model(model_artifact_config, dev)

        if model_artifact_config.num_shards > 1:
            model.disco_session.sync_worker_0()

        num_kv_heads = (
            model_artifact_config.num_key_value_heads
            // model_artifact_config.num_shards
        )
        head_size = (
            model_artifact_config.hidden_size
            // model_artifact_config.num_attention_heads
        )

        if engine_config.max_num_batched_tokens > 0:
            LOG.info("Running memory profiling.")
            num_blocks = get_num_cache_blocks(
                model,
                [engine_config.max_input_len] * engine_config.max_num_sequences,
                model_artifact_config.num_hidden_layers,
                num_kv_heads,
                head_size,
            )
        else:
            num_blocks = 500

        num_cache_slots = num_blocks * CacheManager.block_size

        if num_cache_slots <= engine_config.max_num_batched_tokens:
            raise RuntimeError(
                f"max_num_batched_tokens = {engine_config.max_num_batched_tokens} but"
                f" only {num_blocks} cache blocks can be allocated. The number of"
                f" available cache slots is {num_cache_slots}, not enough for"
                f" {engine_config.max_num_batched_tokens} tokens. Try reducing"
                " --max_input_len or --max_num_sequences."
            )

        LOG.info(f"Using {num_blocks} cache blocks.")

        if model.disco_session:
            init_cache_func = model.disco_session.get_global_func(
                "tvm.contrib.vllm.allocate_kv_cache"
            )
        else:
            init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

        cache_blocks = init_cache_func(
            head_size,
            model_artifact_config.num_hidden_layers,
            num_kv_heads,
            CacheManager.block_size,
            num_blocks,
        )

        cache_manager = CacheManager(
            cache_blocks,
            num_blocks,
            model_artifact_config.sliding_window,
        )

        LOG.info("Allocated KV cache blocks.")

        self.engine_config = engine_config
        self.model_artifact_config = model_artifact_config
        self.text_generator = PagedCacheModelTextGenerator(model)
        self.cache_manager = cache_manager

        tokenizer_module = HfTokenizerModule(model_artifact_path)
        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template

    def _check_implements_model_module(self) -> ModelModule:
        return self  # type: ignore
