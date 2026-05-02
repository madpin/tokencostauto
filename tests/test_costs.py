#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
from decimal import Decimal
from tokencostauto.costs import (
    count_message_tokens,
    count_string_tokens,
    calculate_cost_by_tokens,
    calculate_prompt_cost,
    calculate_completion_cost,
)

# 15 tokens
MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

MESSAGES_WITH_NAME = [
    {"role": "user", "content": "Hello", "name": "John"},
    {"role": "assistant", "content": "Hi there!"},
]

# 4 tokens
STRING = "Hello, world!"


# Chat models only, no embeddings (such as ada) since embeddings only does strings, not messages
@pytest.mark.parametrize(
    "model,expected_output",
    [
        ("gpt-3.5-turbo", 15),
        ("gpt-3.5-turbo-0301", 17),
        ("gpt-3.5-turbo-0613", 15),
        ("gpt-3.5-turbo-16k", 15),
        ("gpt-3.5-turbo-16k-0613", 15),
        ("gpt-3.5-turbo-1106", 15),
        ("gpt-3.5-turbo-instruct", 15),
        ("gpt-4", 15),
        ("gpt-4-0314", 15),
        ("gpt-4-0613", 15),
        ("gpt-4-32k", 15),
        ("gpt-4-32k-0314", 15),
        ("gpt-4-1106-preview", 15),
        ("gpt-4-vision-preview", 15),
        ("gpt-4o", 15),
        ("azure/gpt-4o", 15),
        pytest.param("claude-3-opus-latest", 11,
                     marks=pytest.mark.skipif(
                         not os.getenv("ANTHROPIC_API_KEY"),
                         reason="ANTHROPIC_API_KEY environment variable not set"
                     )),
    ],
)
def test_count_message_tokens(model, expected_output):
    print(model)
    assert count_message_tokens(MESSAGES, model) == expected_output


# Chat models only, no embeddings
@pytest.mark.parametrize(
    "model,expected_output",
    [
        ("gpt-3.5-turbo", 17),
        ("gpt-3.5-turbo-0301", 17),
        ("gpt-3.5-turbo-0613", 17),
        ("gpt-3.5-turbo-1106", 17),
        ("gpt-3.5-turbo-instruct", 17),
        ("gpt-3.5-turbo-16k", 17),
        ("gpt-3.5-turbo-16k-0613", 17),
        ("gpt-4", 17),
        ("gpt-4-0314", 17),
        ("gpt-4-0613", 17),
        ("gpt-4-32k", 17),
        ("gpt-4-32k-0314", 17),
        ("gpt-4-1106-preview", 17),
        ("gpt-4-vision-preview", 17),
        ("gpt-4o", 17),
        ("azure/gpt-4o", 17),
        # ("claude-3-opus-latest", 4), # NOTE: Claude only supports messages without extra inputs
    ],
)
def test_count_message_tokens_with_name(model, expected_output):
    """Notice: name 'John' appears"""

    assert count_message_tokens(MESSAGES_WITH_NAME, model) == expected_output


def test_count_message_tokens_empty_input():
    """Empty input should raise a KeyError"""
    with pytest.raises(KeyError):
        count_message_tokens("", "")


def test_count_message_tokens_invalid_model():
    """Invalid model should raise a KeyError"""

    with pytest.raises(KeyError):
        count_message_tokens(MESSAGES, model="invalid_model")


@pytest.mark.parametrize(
    "model,expected_output",
    [
        ("gpt-3.5-turbo", 4),
        ("gpt-3.5-turbo-0301", 4),
        ("gpt-3.5-turbo-0613", 4),
        ("gpt-3.5-turbo-16k", 4),
        ("gpt-3.5-turbo-16k-0613", 4),
        ("gpt-3.5-turbo-1106", 4),
        ("gpt-3.5-turbo-instruct", 4),
        ("gpt-4-0314", 4),
        ("gpt-4", 4),
        ("gpt-4-32k", 4),
        ("gpt-4-32k-0314", 4),
        ("gpt-4-0613", 4),
        ("gpt-4-1106-preview", 4),
        ("gpt-4-vision-preview", 4),
        ("text-embedding-ada-002", 4),
        ("gpt-4o", 4),
        # ("claude-3-opus-latest", 4), # NOTE: Claude only supports messages
    ],
)
def test_count_string_tokens(model, expected_output):
    """Test that the string tokens are counted correctly."""

    # 4 tokens
    assert count_string_tokens(STRING, model=model) == expected_output

    # empty string
    assert count_string_tokens("", model=model) == 0


def test_count_string_invalid_model():
    """Test that the string tokens are counted correctly."""

    assert count_string_tokens(STRING, model="invalid model") == 4


# Costs from https://openai.com/pricing
# https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
@pytest.mark.parametrize(
    "prompt,model",
    [
        (MESSAGES, "gpt-3.5-turbo"),
        (MESSAGES, "gpt-3.5-turbo-0301"),
        (MESSAGES, "gpt-3.5-turbo-0613"),
        (MESSAGES, "gpt-3.5-turbo-16k"),
        (MESSAGES, "gpt-3.5-turbo-16k-0613"),
        (MESSAGES, "gpt-3.5-turbo-1106"),
        (MESSAGES, "gpt-3.5-turbo-instruct"),
        (MESSAGES, "gpt-4"),
        (MESSAGES, "gpt-4-0314"),
        (MESSAGES, "gpt-4-32k"),
        (MESSAGES, "gpt-4-32k-0314"),
        (MESSAGES, "gpt-4-0613"),
        (MESSAGES, "gpt-4-1106-preview"),
        (MESSAGES, "gpt-4-vision-preview"),
        (MESSAGES, "gpt-4o"),
        (MESSAGES, "azure/gpt-4o"),
        pytest.param(MESSAGES, "claude-3-opus-latest",
                     marks=pytest.mark.skipif(
                         not os.getenv("ANTHROPIC_API_KEY"),
                         reason="ANTHROPIC_API_KEY environment variable not set"
                     )),
        (STRING, "text-embedding-ada-002"),
    ],
)
def test_calculate_prompt_cost(prompt, model):
    """Test that the cost calculation is correct."""
    from tokencostauto.constants import TOKEN_COSTS
    from tokencostauto.costs import _normalize_model_for_pricing

    cost = calculate_prompt_cost(prompt, model)

    # Dynamically verify against current TOKEN_COSTS
    norm_model = _normalize_model_for_pricing(model)
    price_per_token = TOKEN_COSTS[norm_model]["input_cost_per_token"]

    # We need the token count to verify
    if isinstance(prompt, str) and "claude-" not in model:
        tokens = count_string_tokens(prompt, model)
    else:
        tokens = count_message_tokens(prompt, model)

    expected = Decimal(str(price_per_token)) * Decimal(tokens)
    assert cost == expected


def test_invalid_prompt_format():
    with pytest.raises(TypeError):
        calculate_prompt_cost(
            {"role": "user", "content": "invalid message type"}, model="gpt-3.5-turbo"
        )


@pytest.mark.parametrize(
    "prompt,model",
    [
        (STRING, "gpt-3.5-turbo"),
        (STRING, "gpt-3.5-turbo-0301"),
        (STRING, "gpt-3.5-turbo-0613"),
        (STRING, "gpt-3.5-turbo-16k"),
        (STRING, "gpt-3.5-turbo-16k-0613"),
        (STRING, "gpt-3.5-turbo-1106"),
        (STRING, "gpt-3.5-turbo-instruct"),
        (STRING, "gpt-4"),
        (STRING, "gpt-4-0314"),
        (STRING, "gpt-4-32k"),
        (STRING, "gpt-4-32k-0314"),
        (STRING, "gpt-4-0613"),
        (STRING, "gpt-4-1106-preview"),
        (STRING, "gpt-4-vision-preview"),
        (STRING, "gpt-4o"),
        (STRING, "azure/gpt-4o"),
        # (STRING, "claude-3-opus-latest", Decimal("0.000096")), # NOTE: Claude only supports messages
        (STRING, "text-embedding-ada-002"),
    ],
)
def test_calculate_completion_cost(prompt, model):
    """Test that the completion cost calculation is correct."""
    from tokencostauto.constants import TOKEN_COSTS
    from tokencostauto.costs import _normalize_model_for_pricing

    cost = calculate_completion_cost(prompt, model)

    norm_model = _normalize_model_for_pricing(model)
    price_per_token = TOKEN_COSTS[norm_model]["output_cost_per_token"]

    if "claude-" in model:
        completion_list = [{"role": "assistant", "content": prompt}]
        tokens = count_message_tokens(completion_list, model) - 13
    else:
        tokens = count_string_tokens(prompt, model)

    expected = Decimal(str(price_per_token)) * Decimal(tokens)
    assert cost == expected


def test_calculate_cost_invalid_model():
    """Invalid model should raise a KeyError"""

    with pytest.raises(KeyError):
        calculate_prompt_cost(STRING, model="invalid_model")


def test_calculate_invalid_input_types():
    """Invalid input type should raise a KeyError"""

    with pytest.raises(KeyError):
        calculate_prompt_cost(STRING, model="invalid_model")

    with pytest.raises(KeyError):
        calculate_completion_cost(STRING, model="invalid_model")

    with pytest.raises(KeyError):
        # Message objects not allowed, must be list of message objects.
        calculate_prompt_cost(MESSAGES[0], model="invalid_model")


@pytest.mark.parametrize(
    "num_tokens,model,token_type",
    [
        (10, "gpt-3.5-turbo", "input"),
        (5, "gpt-4", "output"),
        (10, "ai21.j2-mid-v1", "input"),
    ],
)
def test_calculate_cost_by_tokens(num_tokens, model, token_type):
    """Test that the token cost calculation is correct."""
    from tokencostauto.constants import TOKEN_COSTS
    from tokencostauto.costs import _normalize_model_for_pricing, _get_field_from_token_type

    cost = calculate_cost_by_tokens(num_tokens, model, token_type)

    norm_model = _normalize_model_for_pricing(model)
    field = _get_field_from_token_type(token_type)
    price_per_token = TOKEN_COSTS[norm_model][field]

    expected = Decimal(str(price_per_token)) * Decimal(num_tokens)
    assert cost == expected
