from src.orchestration.context_pipeline import ContextPacket


def test_context_packet_prioritizes_relevant_sections():
    packet = ContextPacket(
        task_focus="improve API endpoint auth validation",
        interfaces_context="service interface for auth endpoint validation",
        dependencies_context="numpy pandas matplotlib",
        documentation_context="frontend style guide",
        retrieved_code_context="api endpoint auth middleware and validator",
    )

    output = packet.to_prompt_context(max_chars=700)

    assert "## Interfaces" in output
    assert "## Retrieved Code" in output
    assert output.find("## Interfaces") < output.find("## Documentation")


def test_context_packet_compresses_large_sections_under_budget():
    body = "\n".join(
        [
            "line with low signal",
            "critical_fn(path=/src/api/auth.py) -> validates token signatures and expiry",
            "another low signal line",
            "error_code=AUTH_401 mapping: unauthorized when signature mismatch",
            "filler filler filler",
        ]
    )

    packet = ContextPacket(task_focus="auth signature validation", interfaces_context=body)
    output = packet.to_prompt_context(max_chars=240)

    assert len(output) <= 240
    assert "critical_fn" in output or "AUTH_401" in output
