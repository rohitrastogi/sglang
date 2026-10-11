"""Reasoning accounting consumes only the retained prefix of accepted tokens."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_MATCHED_STR, Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EOS = 2
STOP = 3
THINK_END = [7, 8]


class _Tokenizer:
    eos_token_id = EOS
    additional_stop_token_ids = None

    def encode(self, text, add_special_tokens=False):
        return [STOP]

    def decode(self, ids):
        return "".join("STOP" if token == STOP else "x" for token in ids)


def _make_req(max_new_tokens=1000, **sampling_kwargs):
    params = SamplingParams(max_new_tokens=max_new_tokens, **sampling_kwargs)
    params.normalize(_Tokenizer())
    req = Req(
        rid="reasoning",
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=params,
        eos_token_ids={EOS},
        vocab_size=100,
        require_reasoning=True,
    )
    req.tokenizer = _Tokenizer()
    return req


def _make_processor(overlap=False):
    metrics = Mock(num_generated_tokens=0, forward_ct_decode=0)
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=overlap,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=THINK_END),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics,
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=Mock(),
    )


def _decode(processor, req, tokens, *, speculative=True):
    batch = SimpleNamespace(
        reqs=[req],
        has_grammar=False,
        return_logprob=False,
        spec_algorithm=(
            SpeculativeAlgorithm.EAGLE if speculative else SpeculativeAlgorithm.NONE
        ),
        batch_size=lambda: 1,
    )
    result = GenerationBatchResult(
        next_token_ids=torch.tensor(tokens, dtype=torch.long),
        accept_lens=torch.tensor([len(tokens)]) if speculative else None,
        speculative_num_draft_tokens=len(tokens) if speculative else None,
    )
    # Cache release/checkpointing is unrelated to the accounting under test.
    # Keep normalization, append, finish handling and delimiter matching real.
    with patch.object(
        SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"
    ):
        processor.process_batch_result_decode(batch, result)


class TestReasoningTokensThroughStop(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(enable_metrics=False)
        override.install()
        self.addCleanup(override.restore)

    def test_speculative_cutoffs(self):
        for overlap in (False, True):
            for cutoff in (
                "length",
                "eos",
                "stop_token",
                "stop_string",
                "stop_regex",
                "none",
            ):
                with self.subTest(overlap=overlap, cutoff=cutoff):
                    kwargs = {}
                    if cutoff == "stop_token":
                        kwargs["stop_token_ids"] = {STOP}
                    elif cutoff == "stop_string":
                        kwargs["stop"] = ["STOP"]
                    elif cutoff == "stop_regex":
                        kwargs["stop_regex"] = ["ST[O]P"]
                    req = _make_req(
                        max_new_tokens=24 if cutoff == "length" else 1000, **kwargs
                    )
                    processor = _make_processor(overlap)
                    _decode(processor, req, [10] * 22)
                    middle = EOS if cutoff == "eos" else STOP
                    tokens = [11, middle, 12, 13]
                    _decode(processor, req, tokens)

                    retained = 26 if cutoff == "none" else 24
                    self.assertEqual(len(req.output_ids), 26)
                    self.assertEqual(len(req.output_ids_through_stop), retained)
                    self.assertEqual(req.reasoning_tokens, retained)
                    self.assertEqual(req.finished(), cutoff != "none")
                    self.assertFalse(req._is_reasoning_over)

    def test_overlap_tick_after_finish_does_not_change_request(self):
        req = _make_req(max_new_tokens=2)
        processor = _make_processor(overlap=True)
        _decode(processor, req, [10, 7, 11])
        self.assertTrue(req.finished())
        self.assertEqual(req.reasoning_tokens, 2)
        self.assertEqual(req._think_end_match_len, 1)
        output_ids = list(req.output_ids)
        finish_reason = req.finished_reason
        matcher = req._think_end_matcher

        # This would complete the partial marker if the overshoot tick were
        # incorrectly committed or consumed by reasoning accounting.
        _decode(processor, req, [8, 12])

        self.assertEqual(list(req.output_ids), output_ids)
        self.assertEqual(list(req.output_ids_through_stop), [10, 7])
        self.assertIs(req.finished_reason, finish_reason)
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(req.reasoning_tokens, 2)
        self.assertIs(req._think_end_matcher, matcher)
        self.assertEqual(req._think_end_match_len, 1)
        self.assertFalse(req._is_reasoning_over)

    def test_invalid_token_correction_cuts_off_reasoning(self):
        for invalid_token in (-1, 100):
            with self.subTest(invalid_token=invalid_token):
                req = _make_req()
                _decode(_make_processor(), req, [11, invalid_token, 7, 8])

                self.assertEqual(list(req.output_ids), [11, EOS, 7, 8])
                self.assertEqual(list(req.output_ids_through_stop), [11, EOS])
                self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
                self.assertEqual(req.finished_reason.matched, "NaN happened")
                self.assertEqual(req.reasoning_tokens, 2)
                self.assertEqual(req._think_end_match_len, 0)
                self.assertFalse(req._is_reasoning_over)

    def test_external_abort_keeps_accepted_tokens(self):
        for tokens, reasoning, over in (([11, 12], 4, False), ([8, 11], 3, True)):
            with self.subTest(tokens=tokens):
                req = _make_req()
                processor = _make_processor()
                _decode(processor, req, [10, 7])
                abort = FINISH_ABORT("external abort")
                req.to_finish = abort

                _decode(processor, req, tokens)

                self.assertIs(req.finished_reason, abort)
                self.assertIsNone(req.to_finish)
                self.assertIsNone(req.finished_len)
                self.assertEqual(list(req.output_ids_through_stop), [10, 7] + tokens)
                self.assertEqual(req.reasoning_tokens, reasoning)
                self.assertEqual(req._is_reasoning_over, over)

    def test_think_end_around_cutoff(self):
        cases = [
            # Full marker before EOS: text after thinking is completion only.
            ([7, 8, 11, EOS, 12], 1000, 4, 2, True, 0),
            # Marker ends exactly at the length cutoff.
            ([11, 7, 8, 12], 3, 3, 3, True, 0),
            # Marker straddles the length cutoff: retain its partial match.
            ([11, 7, 8, 12], 2, 2, 2, False, 1),
            # The discarded suffix must neither complete nor start a marker.
            ([11, EOS, 7, 8], 1000, 2, 2, False, 0),
            ([11, EOS, 7], 1000, 2, 2, False, 0),
            # No cutoff: count through the marker, excluding subsequent text.
            ([11, 7, 8, 12], 1000, 4, 3, True, 0),
        ]
        for overlap in (False, True):
            for tokens, cap, retained, reasoning, over, match_len in cases:
                with self.subTest(overlap=overlap, tokens=tokens, cap=cap):
                    req = _make_req(max_new_tokens=cap)
                    _decode(_make_processor(overlap), req, tokens)
                    self.assertEqual(len(req.output_ids_through_stop), retained)
                    self.assertEqual(req.reasoning_tokens, reasoning)
                    self.assertEqual(req._is_reasoning_over, over)
                    self.assertEqual(req._think_end_match_len, match_len)

    def test_think_end_across_batches_and_cutoff(self):
        for overlap in (False, True):
            for cap in (3, 4, 1000):
                with self.subTest(overlap=overlap, cap=cap):
                    req = _make_req(max_new_tokens=cap)
                    processor = _make_processor(overlap)
                    _decode(processor, req, [10, 7])
                    self.assertEqual(req._think_end_match_len, 1)
                    _decode(processor, req, [8, 11, EOS, 12])
                    self.assertEqual(req.reasoning_tokens, 3)
                    self.assertTrue(req._is_reasoning_over)
                    self.assertEqual(req._think_end_match_len, 1)
                    self.assertLessEqual(
                        req.reasoning_tokens, len(req.output_ids_through_stop)
                    )

    def test_partial_match_survives_discarded_closer_across_batches(self):
        req = _make_req(max_new_tokens=3)
        processor = _make_processor()
        _decode(processor, req, [10, 7])
        _decode(processor, req, [7, 8, 11])
        self.assertEqual(list(req.output_ids_through_stop), [10, 7, 7])
        self.assertEqual(req.reasoning_tokens, 3)
        self.assertFalse(req._is_reasoning_over)
        self.assertEqual(req._think_end_match_len, 1)

    def test_prefill_accounts_only_retained_token(self):
        for cap in (0, 1, 1000):
            with self.subTest(cap=cap):
                req = _make_req(max_new_tokens=cap)
                processor = _make_processor()
                batch = SimpleNamespace(
                    reqs=[req],
                    return_logprob=False,
                    return_hidden_states=False,
                    return_hidden_states_mode=CaptureHiddenMode.NULL,
                    spec_info=None,
                    decoding_reqs=None,
                    forward_mode=ForwardMode.EXTEND,
                    prefill_stats=None,
                )
                result = GenerationBatchResult(next_token_ids=torch.tensor([7]))
                with (
                    patch(
                        "sglang.srt.managers.scheduler_components."
                        "batch_result_processor.release_kv_cache"
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components."
                        "batch_result_processor.checkpoint_kv_cache"
                    ),
                ):
                    processor.process_batch_result_prefill(batch, result)
                retained = int(cap > 0)
                self.assertEqual(len(req.output_ids_through_stop), retained)
                self.assertEqual(req.reasoning_tokens, retained)
                self.assertEqual(req._think_end_match_len, retained)

    def test_zero_budget_does_not_start_matcher(self):
        req = _make_req(max_new_tokens=0)
        _decode(_make_processor(), req, [7, 8, 11])
        self.assertEqual(list(req.output_ids_through_stop), [])
        self.assertEqual(req.reasoning_tokens, 0)
        self.assertIsNone(req._think_end_matcher)
        self.assertFalse(req._is_reasoning_over)

    def test_ordinary_decode_preserves_delimiter_and_stop(self):
        req = _make_req()
        processor = _make_processor()
        for token in [10, 7, 8, 11, EOS]:
            _decode(processor, req, [token], speculative=False)
        self.assertEqual(list(req.output_ids_through_stop), [10, 7, 8, 11, EOS])
        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)
        self.assertTrue(req.finished())


if __name__ == "__main__":
    unittest.main()
