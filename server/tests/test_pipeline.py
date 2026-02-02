"""Tests for the response pipeline."""

import pytest
from modules.config import ConfigEnv
from modules.response.response import ResponsePipeline
from modules.response.llm_client import LLMClient


@pytest.mark.asyncio
async def test_pipeline_initialization():
    """Test that pipeline can be initialized."""
    pipeline = ResponsePipeline(api_key="test_api_key")
    
    assert pipeline is not None
    assert pipeline.llm_client is not None
    assert pipeline.tool_registry is not None


@pytest.mark.asyncio
async def test_llm_client_initialization():
    """Test LLM client initialization."""
    client = LLMClient(api_key="test_api_key")
    
    assert client is not None
    assert client.api_key == "test_api_key"
    assert client.model_name == ConfigEnv.CEREBRAS_MODEL_ID
    assert client.temperature == 0.7
    assert client.model is not None


def test_pipeline_load_test_prompts():
    """Test loading test prompts from file."""
    pipeline = ResponsePipeline(api_key="test_api_key")
    prompts = pipeline.load_test_prompts()
    
    # Should load prompts from test-prompts.json
    assert isinstance(prompts, list)
    # May be empty if file doesn't exist or is empty, that's ok


@pytest.mark.asyncio
async def test_pipeline_process_text_structure(mock_langchain_chat_model, monkeypatch):
    """Test that process_text returns correct structure."""
    # Mock the ChatCerebras to avoid actual API calls
    monkeypatch.setattr("modules.response.llm_client.ChatCerebras", lambda **kwargs: mock_langchain_chat_model)
    
    pipeline = ResponsePipeline(api_key="test_api_key")
    result = await pipeline.process_text("Test prompt")
    
    # Check structure of result
    assert isinstance(result, dict)
    assert "response" in result
    assert "tool_calls" in result
    assert "tool_results" in result
    assert "metadata" in result
