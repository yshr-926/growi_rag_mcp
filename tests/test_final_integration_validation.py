"""
Final integration validation suite (RED phase).

This suite encodes the acceptance criteria for T030 Final Integration Testing:
1) Enforce the 10-second SLA for growi_rag_search responses
2) Verify all user-facing scenarios end to end (retrieve + RAG)
3) Assert resilience for representative error paths and edge cases
4) Exercise the 1,000 page development limit for dataset handling
5) Track memory/resource usage for model loading and query execution

Spec references:
- docs/spec.md Section 11 Performance Requirements
- docs/spec.md Section 7 MCP Tool Specification
- docs/spec.md Section 10 Error Responses
- docs/spec.md Section 4 Data Acquisition and Sync
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

PERFORMANCE_TARGET_MS = 10_000
RESOURCE_BUDGET_MB = 4096
DEVELOPMENT_PAGE_LIMIT = 1000


def _load_final_integration_module():
    """Import the final integration validation module or fail RED with guidance."""
    try:
        from src.validation import final_integration as module  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover - intentional RED failure path
        pytest.fail(
            "Missing final integration validation module 'src.validation.final_integration'. "
            "Implement run_final_integration_suite & supporting types per docs/spec.md Section 11. "
            f"Error: {exc}"
        )
    return module


@pytest.fixture
def final_integration_inputs(tmp_path: Path) -> Dict[str, Any]:
    """Provide shared inputs for the final validation suite."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "growi:\n"
            "  api_url: https://growi.example.com\n"
            "  api_token: dummy-token\n"
            "  page_limit: 1000\n"
            "vector_db:\n"
            "  persist_directory: ./chroma_db\n"
            "  collection_name: growi_documents\n"
            "llm:\n"
            "  model_path: ./models/dev-model\n"
        ),
        encoding="utf-8",
    )

    dataset = [
        {
            "_id": f"page_{idx}",
            "grant": 1,
            "path": f"/public/{idx}",
            "title": f"Public Page {idx}",
            "updatedAt": "2025-01-15T10:30:00Z",
            "revision": {"body": "# Heading\nPublic content\n"},
            "tags": ["integration", "guide"],
        }
        for idx in range(DEVELOPMENT_PAGE_LIMIT)
    ]

    scenario_matrix = [
        {"name": "retrieve-basic", "tool": "growi_retrieve", "params": {"query": "Public Page 1"}},
        {"name": "rag-summary-ja", "tool": "growi_rag_search", "params": {"query": "overview", "lang": "ja"}},
        {"name": "rag-summary-en", "tool": "growi_rag_search", "params": {"query": "overview", "lang": "en"}},
        {"name": "retrieve-edge-filters", "tool": "growi_retrieve", "params": {"query": "grant filter"}},
    ]

    error_matrix = [
        {"code": "GROWI_AUTH_ERROR", "description": "Expired token handled without crash"},
        {"code": "GROWI_RATE_LIMITED", "description": "429 triggers retry with backoff"},
        {"code": "VECTOR_STORE_ERROR", "description": "Vector store failure surfaces recoverable error"},
        {"code": "RESOURCE_EXHAUSTED", "description": "Embedding OOM downgraded to warning"},
        {"code": "TIMEOUT", "description": "End-to-end execution aborts inside 30s timeout"},
    ]

    return {
        "call_kwargs": {
            "config_path": config_path,
            "scenario_matrix": scenario_matrix,
            "error_matrix": error_matrix,
            "dataset_factory": lambda limit=DEVELOPMENT_PAGE_LIMIT: dataset[:limit],
            "performance_target_ms": PERFORMANCE_TARGET_MS,
            "resource_budget_mb": RESOURCE_BUDGET_MB,
        },
        "expected": {
            "scenario_names": [scenario["name"] for scenario in scenario_matrix],
            "error_codes": {entry["code"] for entry in error_matrix},
            "dataset_size": len(dataset),
        },
    }


@pytest.fixture
def final_validation_report(final_integration_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the final validation suite (expected to fail in RED)."""
    module = _load_final_integration_module()
    return module.run_final_integration_suite(**final_integration_inputs["call_kwargs"])


def test_final_validation_reports_performance_budget(final_validation_report: Dict[str, Any]) -> None:
    """
    Test that final validation enforces the 10-second SLA target.

    Given: Complete RAG system with growi_rag_search
    When: Final validation runs with performance monitoring
    Then: All growi_rag_search calls complete within 10 seconds
    """
    performance_metrics = final_validation_report["performance_metrics"]

    # Check that performance tracking exists
    assert "rag_search_latency_ms" in performance_metrics
    assert "retrieve_latency_ms" in performance_metrics

    # Enforce 10-second SLA for RAG search
    rag_latencies = performance_metrics["rag_search_latency_ms"]
    for scenario_name, latency_ms in rag_latencies.items():
        assert latency_ms <= PERFORMANCE_TARGET_MS, (
            f"growi_rag_search for scenario '{scenario_name}' took {latency_ms}ms, "
            f"exceeding {PERFORMANCE_TARGET_MS}ms SLA"
        )


def test_final_validation_covers_all_user_scenarios(
    final_validation_report: Dict[str, Any],
    final_integration_inputs: Dict[str, Any]
) -> None:
    """
    Test that final validation covers all required user scenarios.

    Given: Scenario matrix covering retrieve and RAG search tools
    When: Final validation executes all scenarios
    Then: Each scenario completes successfully with expected result structure
    """
    scenario_results = final_validation_report["scenario_results"]
    expected_scenarios = final_integration_inputs["expected"]["scenario_names"]

    # Check all scenarios were executed
    executed_scenarios = set(scenario_results.keys())
    expected_scenario_set = set(expected_scenarios)
    assert executed_scenarios == expected_scenario_set, (
        f"Missing scenarios: {expected_scenario_set - executed_scenarios}, "
        f"Extra scenarios: {executed_scenarios - expected_scenario_set}"
    )

    # Check each scenario has proper structure
    for scenario_name, result in scenario_results.items():
        assert "status" in result, f"Scenario '{scenario_name}' missing status field"
        assert "response" in result, f"Scenario '{scenario_name}' missing response field"
        assert result["status"] == "success", f"Scenario '{scenario_name}' failed: {result.get('error')}"

        # Validate response structure based on tool type
        response = result["response"]
        if "retrieve" in scenario_name:
            assert "results" in response, f"growi_retrieve scenario '{scenario_name}' missing results"
            assert "total_chunks_found" in response, f"growi_retrieve scenario '{scenario_name}' missing total_chunks_found"
        elif "rag" in scenario_name:
            assert "summary" in response, f"growi_rag_search scenario '{scenario_name}' missing summary"
            assert "related_pages" in response, f"growi_rag_search scenario '{scenario_name}' missing related_pages"


def test_final_validation_tracks_error_resilience(
    final_validation_report: Dict[str, Any],
    final_integration_inputs: Dict[str, Any]
) -> None:
    """
    Test that final validation verifies error handling for known failure modes.

    Given: Error matrix covering authentication, rate limiting, vector store, and timeout errors
    When: Final validation simulates error conditions
    Then: System handles each error gracefully with appropriate error codes
    """
    error_results = final_validation_report["error_resilience"]
    expected_error_codes = final_integration_inputs["expected"]["error_codes"]

    # Check all error scenarios were tested
    tested_error_codes = set(error_results.keys())
    assert tested_error_codes == expected_error_codes, (
        f"Missing error scenarios: {expected_error_codes - tested_error_codes}, "
        f"Extra error scenarios: {tested_error_codes - expected_error_codes}"
    )

    # Check each error scenario was handled properly
    for error_code, result in error_results.items():
        assert "handled_gracefully" in result, f"Error code '{error_code}' missing graceful handling status"
        assert result["handled_gracefully"], f"Error code '{error_code}' not handled gracefully"
        assert "recovery_action" in result, f"Error code '{error_code}' missing recovery action"


def test_final_validation_handles_development_page_limit(
    final_validation_report: Dict[str, Any],
    final_integration_inputs: Dict[str, Any]
) -> None:
    """
    Test that final validation verifies handling of the 1000 page development limit.

    Given: Dataset approaching the 1000 page development limit
    When: Final validation processes the complete dataset
    Then: System processes pages efficiently without exceeding resource constraints
    """
    dataset_metrics = final_validation_report["dataset_metrics"]
    expected_size = final_integration_inputs["expected"]["dataset_size"]

    # Check dataset size handling
    assert "pages_processed" in dataset_metrics
    assert "processing_rate_pages_per_sec" in dataset_metrics
    assert "memory_usage_mb" in dataset_metrics

    processed_pages = dataset_metrics["pages_processed"]
    assert processed_pages == expected_size, (
        f"Expected to process {expected_size} pages, actually processed {processed_pages}"
    )

    # Check processing efficiency
    processing_rate = dataset_metrics["processing_rate_pages_per_sec"]
    assert processing_rate > 0, "Processing rate should be positive"

    # Ensure memory usage is within bounds
    memory_usage = dataset_metrics["memory_usage_mb"]
    assert memory_usage <= RESOURCE_BUDGET_MB, (
        f"Memory usage {memory_usage}MB exceeds budget of {RESOURCE_BUDGET_MB}MB"
    )


def test_final_validation_reports_resource_usage(final_validation_report: Dict[str, Any]) -> None:
    """
    Test that final validation tracks resource usage throughout execution.

    Given: Resource monitoring during final validation
    When: All scenarios and datasets are processed
    Then: Resource usage stays within acceptable limits and is properly reported
    """
    resource_metrics = final_validation_report["resource_metrics"]

    # Check required resource metrics exist
    required_metrics = [
        "peak_memory_mb",
        "model_loading_time_ms",
        "vector_store_size_mb",
        "cpu_usage_percent",
        "total_execution_time_ms"
    ]

    for metric in required_metrics:
        assert metric in resource_metrics, f"Missing required resource metric: {metric}"

    # Check resource constraints
    peak_memory = resource_metrics["peak_memory_mb"]
    assert peak_memory <= RESOURCE_BUDGET_MB, (
        f"Peak memory usage {peak_memory}MB exceeds budget of {RESOURCE_BUDGET_MB}MB"
    )

    cpu_usage = resource_metrics["cpu_usage_percent"]
    assert 0 <= cpu_usage <= 100, f"Invalid CPU usage percentage: {cpu_usage}%"

    total_time = resource_metrics["total_execution_time_ms"]
    assert total_time > 0, "Total execution time should be positive"