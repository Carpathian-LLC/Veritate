# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for the pure hallucination-detection functions: span aggregation
#   (byte offsets + confidence rollup), grounded overlap, divergence scoring,
#   refusal + verdict logic, and the training-provenance matcher over a synthetic
#   BM25 index. No live model, no network.
# tests/mri/test_hallucination.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from inference import hallucination as hl
from inference.agent.tools.retriever import BM25Index

# ------------------------------------------------------------------------------------
# Constants

CTX_CAT = "the cat sat on the mat in the house"

# ------------------------------------------------------------------------------------
# Functions


def _metrics(answer, conf=0.9, surprise=1.0, entropy=2.0):
    return [{"confidence": conf, "surprise_bits": surprise, "entropy_bits": entropy}
            for _ in answer.encode("utf-8")]


def test_segment_word_byte_offsets_exact():
    """Word spans carry byte-exact start/end offsets into the answer."""
    answer = "Hello world."
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    words = spans["paragraphs"][0]["sentences"][0]["words"]
    assert [(w["text"], w["start"], w["end"]) for w in words] == \
        [("Hello", 0, 5), ("world.", 6, 12)]


def test_segment_overall_confidence_is_byte_mean():
    """Overall confidence is the mean of the per-byte confidence stream."""
    answer = "abcd"
    metrics = _metrics(answer)
    metrics[0]["confidence"] = 0.1
    _, overall = hl.segment_spans(answer, metrics)
    assert abs(overall - (0.1 + 0.9 * 3) / 4) < 1e-9


def test_segment_confidence_min_is_weakest_byte():
    """A span's confidence_min catches its single lowest-confidence byte."""
    answer = "one two three"
    metrics = _metrics(answer)
    metrics[5]["confidence"] = 0.05
    spans, _ = hl.segment_spans(answer, metrics)
    assert spans["paragraphs"][0]["sentences"][0]["confidence_min"] == 0.05


def test_segment_paragraphs_split_on_blank_line():
    """A blank line splits the answer into separate paragraphs."""
    answer = "First para.\n\nSecond para."
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    assert len(spans["paragraphs"]) == 2


def test_grounded_label_yes_no_partial():
    """A word is grounded yes/no when its content word is present/absent in context."""
    answer = "cat dog"
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    hl.annotate_grounding(spans, CTX_CAT)
    words = spans["paragraphs"][0]["sentences"][0]["words"]
    assert (words[0]["grounded"], words[1]["grounded"]) == ("yes", "no")


def test_grounded_fraction_counts_content_words():
    """grounded_fraction is the fraction of answer content-words found in context."""
    answer = "cat dog"
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    assert hl.annotate_grounding(spans, CTX_CAT) == 0.5


def test_grounded_fraction_null_without_context():
    """grounded_fraction is None when there is no context."""
    answer = "cat dog"
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    assert hl.annotate_grounding(spans, None) is None


def test_divergence_identical_is_zero():
    """Two identical answers have zero context divergence."""
    assert hl.divergence_score("the cat sat", "the cat sat") == 0.0


def test_divergence_disjoint_is_one():
    """Answers with no shared content words diverge fully."""
    assert hl.divergence_score("alpha beta", "gamma delta") == 1.0


def test_is_refusal_detects_dont_know():
    """A refusal phrase marks the answer as a refusal."""
    assert hl.is_refusal("I don't know the answer.") is True
    assert hl.is_refusal("The capital is Paris.") is False


def test_verdict_refused_takes_priority():
    """A refusal answer yields the refused verdict regardless of grounding."""
    overall = hl.build_overall(0.9, 0.9, 0.0, "I don't know.")
    assert overall["verdict"] == "refused"


def test_verdict_grounded_needs_high_fraction():
    """grounded_fraction at/above the high threshold yields the grounded verdict."""
    overall = hl.build_overall(0.9, 0.9, 0.1, "Paris is the capital.")
    assert overall["verdict"] == "grounded"


def test_verdict_mid_grounding_is_partially_grounded_not_grounded():
    """A mid grounded_fraction (below the high floor) is partially_grounded, not grounded."""
    overall = hl.build_overall(0.9, 0.7, 0.1, "Paris is the capital and it rains gold.")
    assert overall["verdict"] == "partially_grounded"


def test_verdict_partially_grounded_low_divergence():
    """Partial grounding with low divergence is the partially_grounded check-this state."""
    overall = hl.build_overall(0.9, 0.5, 0.2, "half supported text here")
    assert overall["verdict"] == "partially_grounded"


def test_verdict_likely_hallucinated_low_grounding_high_divergence():
    """Low grounding plus high divergence yields likely_hallucinated."""
    overall = hl.build_overall(0.9, 0.1, 0.9, "some confident text")
    assert overall["verdict"] == "likely_hallucinated"


def test_verdict_escalates_partial_grounding_under_high_divergence():
    """The '2087 million' case (gf=0.5, divergence=1.0) escalates to likely_hallucinated,
    not a benign verdict, and lands in the hallucination risk band."""
    overall = hl.build_overall(0.5, 0.5, 1.0, "2087 million")
    assert overall["verdict"] == "likely_hallucinated"
    assert overall["hallucination_risk"] >= 0.6


def test_verdict_low_confidence_when_uncertain():
    """An overall confidence below threshold with no context yields low_confidence."""
    overall = hl.build_overall(0.3, None, None, "some text")
    assert overall["verdict"] == "low_confidence" and overall["uncertain"] is True


def test_verdict_ungrounded_ok_confident_no_context():
    """A confident answer with no context and no refusal is ungrounded_ok."""
    overall = hl.build_overall(0.9, None, None, "some text")
    assert overall["verdict"] == "ungrounded_ok"


def test_overall_nulls_without_context():
    """grounded_fraction and context_divergence are null when there is no context."""
    overall = hl.build_overall(0.9, None, None, "some text")
    assert overall["grounded_fraction"] is None and overall["context_divergence"] is None


def test_derive_confidence_falls_with_surprise():
    """Derived confidence decreases as surprise bits rise."""
    assert hl.derive_confidence(0.0) == 1.0
    assert hl.derive_confidence(8.0) == 0.0
    assert hl.derive_confidence(4.0) == 0.5


def test_grounded_sources_maps_span_to_best_chunk():
    """A grounded sentence points at the retrieved chunk it best overlaps."""
    answer = "cats purr"
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    chunks = [{"text": "dogs bark loudly", "score": 1.0},
              {"text": "cats purr softly", "score": 2.0}]
    hl.annotate_grounding(spans, " ".join(c["text"] for c in chunks))
    src = hl.grounded_sources(spans, chunks)
    assert src and src[0]["source_chunk"] == "cats purr softly"


def test_training_matches_returns_similar_passages():
    """training_matches returns nearest training passages (proxy) for weak spans."""
    chunks = [(0, "the mitochondria is the powerhouse of the cell"),
              (1, "photosynthesis converts sunlight into chemical energy")]
    meta = [{"corpus": "bio_train.bin", "text": chunks[0][1]},
            {"corpus": "bio_train.bin", "text": chunks[1][1]}]
    index = BM25Index(chunks)
    answer = "mitochondria powerhouse cell"
    spans, _ = hl.segment_spans(answer, _metrics(answer, conf=0.2))
    matches = hl.training_matches(spans, index, meta)
    assert matches and matches[0]["passages"][0]["corpus"] == "bio_train.bin"


def test_training_matches_empty_index_is_empty():
    """A missing training index yields no matches, not an error."""
    answer = "anything"
    spans, _ = hl.segment_spans(answer, _metrics(answer))
    assert hl.training_matches(spans, None, []) == []


# deferred-mode assembly: grade client-provided frames, no re-generation

def test_assemble_from_frames_reconstructs_answer_and_metrics():
    """_assemble_from_frames rebuilds the answer + per-byte metrics from client frames."""
    from routes import hallucination_routes as hr
    frames = [
        {"byte": ord("h"), "confidence": 0.9, "surprise_bits": 0.2, "entropy_bits": 0.1},
        {"byte": ord("i"), "confidence": 0.8, "surprise_bits": 0.3, "entropy_bits": 0.2},
    ]
    answer, metrics, source = hr._assemble_from_frames(frames)
    assert answer == "hi"
    assert [m["confidence"] for m in metrics] == [0.9, 0.8]
    assert source == hl.CONFIDENCE_SOURCE_BYTE


def test_assemble_from_frames_derives_confidence_when_absent():
    """A frame with no confidence derives it from surprise and flags the source."""
    from routes import hallucination_routes as hr
    frames = [{"byte": ord("x"), "confidence": None, "surprise_bits": 1.0, "entropy_bits": 0.5}]
    answer, _, source = hr._assemble_from_frames(frames)
    assert answer == "x"
    assert source == hl.CONFIDENCE_SOURCE_DERIVED
