"""
AETHER_MODEL Phase 3: Comprehensive Evaluator Unit Tests
Validates every scoring rule, positive and negative cases, and edge handling
in benchmark/phase3_evaluator.py.
"""
from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark.phase3_evaluator import evaluate_response


def test_evaluator_empty_response_rejected():
    item = {"eval_type": "keywords", "expected_keywords": ["test"]}
    passed, reason = evaluate_response(item, "")
    assert not passed
    assert "Empty output" in reason

    passed, reason = evaluate_response(item, "   \n\t  ")
    assert not passed
    assert "Empty output" in reason


def test_evaluator_exact_match():
    item = {"eval_type": "exact_match", "expected_answer": "Paris"}
    
    passed, _ = evaluate_response(item, "The capital of France is Paris.")
    assert passed

    passed, reason = evaluate_response(item, "The capital of France is London.")
    assert not passed
    assert "not found" in reason


def test_evaluator_any_match():
    item = {"eval_type": "any_match", "acceptable_answers": ["python", "javascript", "rust"]}
    
    passed, _ = evaluate_response(item, "I strongly recommend using Rust for this project.")
    assert passed

    passed, reason = evaluate_response(item, "I suggest using C++ for maximum low-level control.")
    assert not passed
    assert "None of" in reason


def test_evaluator_keywords_threshold():
    item = {
        "eval_type": "keywords",
        "expected_keywords": ["latency", "throughput", "concurrency", "bottleneck"],
    }
    # 3/4 = 75% >= 65% -> Pass
    passed, _ = evaluate_response(item, "We measured high latency and identified a major concurrency bottleneck.")
    assert passed

    # 1/4 = 25% < 65% -> Fail
    passed, reason = evaluate_response(item, "We measured very high latency in the network.")
    assert not passed
    assert "needed >= 65%" in reason


def test_evaluator_any_keywords():
    item = {"eval_type": "any_keywords", "expected_keywords": ["cat", "dog", "bird"]}
    
    passed, _ = evaluate_response(item, "The shelter rescued a small dog yesterday.")
    assert passed

    passed, reason = evaluate_response(item, "The shelter rescued a young horse yesterday.")
    assert not passed
    assert "None of keywords" in reason


def test_evaluator_constraints():
    # 1. numbered_list_3
    item_list = {"eval_type": "constraint", "constraint": "numbered_list_3"}
    passed, _ = evaluate_response(item_list, "1. First point\n2. Second point\n3. Third point")
    assert passed
    passed, _ = evaluate_response(item_list, "1. Only one item")
    assert not passed

    # 2. semicolon_separated_4
    item_semi = {"eval_type": "constraint", "constraint": "semicolon_separated_4"}
    passed, _ = evaluate_response(item_semi, "apple; orange; banana; grape")
    assert passed
    passed, _ = evaluate_response(item_semi, "apple; orange")
    assert not passed

    # 3. max_12_words
    item_max = {"eval_type": "constraint", "constraint": "max_12_words"}
    passed, _ = evaluate_response(item_max, "This is a very concise response with few words.")
    assert passed
    passed, _ = evaluate_response(item_max, "This is an excessively long and verbose sentence that deliberately contains way too many unnecessary words to violate the limit.")
    assert not passed

    # 4. exactly_7_words
    item_exact7 = {"eval_type": "constraint", "constraint": "exactly_7_words"}
    passed, _ = evaluate_response(item_exact7, "The quick brown fox jumps over dog.")
    assert passed
    passed, _ = evaluate_response(item_exact7, "Just two words.")
    assert not passed

    # 5. exact_word_confirmed
    item_conf = {"eval_type": "constraint", "constraint": "exact_word_confirmed"}
    passed, _ = evaluate_response(item_conf, "CONFIRMED")
    assert passed
    passed, _ = evaluate_response(item_conf, "I confirm that this action is now fully completed and verified.")
    assert not passed

    # 6. no_letter_e
    item_noe = {"eval_type": "constraint", "constraint": "no_letter_e"}
    passed, _ = evaluate_response(item_noe, "A quick brown fox jumps.")
    assert passed
    passed, _ = evaluate_response(item_noe, "The quick brown fox jumps.")  # contains 'e'
    assert not passed

    # Unknown constraint
    item_unknown = {"eval_type": "constraint", "constraint": "unknown_random_constraint"}
    passed, reason = evaluate_response(item_unknown, "Some text")
    assert not passed
    assert "Unhandled constraint" in reason


def test_evaluator_valid_json():
    item = {"eval_type": "valid_json", "required_keys": ["name", "status"]}
    
    # Direct JSON
    passed, _ = evaluate_response(item, '{"name": "task_1", "status": "pending"}')
    assert passed

    # Markdown fenced JSON
    passed, _ = evaluate_response(item, 'Here is the object:\n```json\n{"name": "task_1", "status": "done"}\n```')
    assert passed

    # Missing required key
    passed, reason = evaluate_response(item, '{"name": "task_1"}')
    assert not passed
    assert "missing required keys" in reason

    # Invalid JSON syntax
    passed, reason = evaluate_response(item, '{"name": "task_1", status: pending}')
    assert not passed
    assert "JSON parse failure" in reason


def test_evaluator_valid_json_array():
    item = {"eval_type": "valid_json_array"}
    
    passed, _ = evaluate_response(item, '["item1", "item2", "item3"]')
    assert passed

    # Array length < 2
    passed, reason = evaluate_response(item, '["single_item"]')
    assert not passed
    assert "not array of expected length" in reason

    # Dict instead of array
    passed, reason = evaluate_response(item, '{"item": 1}')
    assert not passed
    assert "not array of expected length" in reason


def test_evaluator_tool_json():
    item = {"eval_type": "tool_json", "expected_tool": "create_task"}

    # Proper structured tool call
    proper_tool = (
        '```json\n'
        '{\n'
        '  "tool": "create_task",\n'
        '  "arguments": {"title": "Test task", "priority": "high"}\n'
        '}\n'
        '```'
    )
    passed, _ = evaluate_response(item, proper_tool)
    assert passed

    # Tool call using "name" key
    name_tool = '{"name": "create_task", "parameters": {"title": "Test"}}'
    passed, _ = evaluate_response(item, name_tool)
    assert passed

    # Wrong tool name
    wrong_tool = '{"tool": "delete_task", "arguments": {"id": 123}}'
    passed, reason = evaluate_response(item, wrong_tool)
    assert not passed
    assert "Expected tool 'create_task'" in reason

    # Unstructured text that mentions create_task must NOT pass
    rambling_text = (
        "I will now create_task for you by contacting the database directly. "
        "The task has been created successfully."
    )
    passed, reason = evaluate_response(item, rambling_text)
    assert not passed
    assert "Expected tool 'create_task' not properly structured" in reason or "JSON parse failure" in reason


def test_evaluator_unsupported_eval_type():
    item = {"eval_type": "non_existent_type"}
    passed, reason = evaluate_response(item, "Some response")
    assert not passed
    assert "Unsupported eval_type" in reason
