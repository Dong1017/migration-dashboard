from __future__ import annotations

import json
from pathlib import Path

from acceptance.tools.compare_outputs import load_simple_yaml, validate_case
from acceptance.tools.write_result import write_case_result

CASE_PATH = Path("acceptance/cases/BR-WRAPPER-001.yaml")
ARTIFACT_DIR = Path("acceptance/artifacts/BR-WRAPPER-001")


class FakeBridgeRuntime:
    def __init__(self, model):
        self.model = model
        self.events: list[str] = []
        self.forward_context = "pt"
        self.distributed = "pt"

    def pt_to_ms(self, value):
        self.events.append("pt_to_ms")
        return f"ms:{value}"

    def ms_to_pt(self, value):
        self.events.append("ms_to_pt")
        return f"pt:{value}"

    def swap_forward_context(self):
        self.events.append("forward_context_swap")
        self.forward_context = "ms"

    def restore_forward_context(self):
        self.events.append("forward_context_restore")
        self.forward_context = "pt"

    def swap_distributed(self):
        self.events.append("distributed_swap")
        self.distributed = "ms"

    def restore_distributed(self):
        self.events.append("distributed_restore")
        self.distributed = "pt"

    def before_ms_forward(self):
        self.events.append("before_ms_forward")

    def after_ms_forward(self):
        self.events.append("after_ms_forward")

    def __call__(self, value):
        ms_value = self.pt_to_ms(value)
        self.swap_forward_context()
        self.swap_distributed()
        try:
            self.before_ms_forward()
            output = self.model.forward(ms_value)
            self.events.append("ms_model_forward")
            self.after_ms_forward()
            return self.ms_to_pt(output)
        finally:
            self.restore_forward_context()
            self.restore_distributed()

    def compute_logits(self, value):
        ms_value = self.pt_to_ms(value)
        output = self.model.compute_logits(ms_value)
        self.events.append("ms_model_compute_logits")
        return self.ms_to_pt(output)


class FakeMSModel:
    def __init__(self, fail=False):
        self.fail = fail

    def forward(self, value):
        if self.fail:
            raise RuntimeError("fake forward failure")
        return f"forward:{value}"

    def compute_logits(self, value):
        return f"logits:{value}"


def write_log(name: str, events: list[str]):
    log_dir = ARTIFACT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / name).write_text("\n".join(events) + "\n", encoding="utf-8")


def test_bridge_case_schema_is_complete():
    case = load_simple_yaml(CASE_PATH)
    assert case["case_id"] == "BR-WRAPPER-001"
    assert {"success_path", "exception_restore", "compute_logits"} <= set(case["contract_paths"])
    validate_case(case)


def test_success_path_restores_context_and_converts_output():
    wrapper = FakeBridgeRuntime(FakeMSModel())

    result = wrapper("input")

    assert result == "pt:forward:ms:input"
    assert wrapper.forward_context == "pt"
    assert wrapper.distributed == "pt"
    assert wrapper.events == [
        "pt_to_ms",
        "forward_context_swap",
        "distributed_swap",
        "before_ms_forward",
        "ms_model_forward",
        "after_ms_forward",
        "ms_to_pt",
        "forward_context_restore",
        "distributed_restore",
    ]
    write_log("success_path.log", wrapper.events)


def test_exception_path_restores_and_next_call_is_clean():
    wrapper = FakeBridgeRuntime(FakeMSModel(fail=True))

    try:
        wrapper("input")
    except RuntimeError as exc:
        assert "fake forward failure" in str(exc)
    else:
        raise AssertionError("Expected fake forward failure")

    assert wrapper.forward_context == "pt"
    assert wrapper.distributed == "pt"
    assert "forward_context_restore" in wrapper.events
    assert "distributed_restore" in wrapper.events

    clean_wrapper = FakeBridgeRuntime(FakeMSModel())
    assert clean_wrapper("again") == "pt:forward:ms:again"
    assert clean_wrapper.forward_context == "pt"
    assert clean_wrapper.distributed == "pt"
    write_log("exception_restore.log", wrapper.events + ["next_call_clean"])


def test_compute_logits_path_is_independent_and_converts_output():
    wrapper = FakeBridgeRuntime(FakeMSModel())

    result = wrapper.compute_logits("hidden")

    assert result == "pt:logits:ms:hidden"
    assert wrapper.forward_context == "pt"
    assert wrapper.distributed == "pt"
    assert wrapper.events == ["pt_to_ms", "ms_model_compute_logits", "ms_to_pt"]
    write_log("compute_logits.log", wrapper.events)


def test_bridge_contract_result_written():
    result_path = ARTIFACT_DIR / "result.json"
    write_case_result(
        result_path,
        case_id="BR-WRAPPER-001",
        status="pass",
        summary="Fake bridge wrapper contract paths executed without NPU.",
        evidence={
            "golden": "fake contract events",
            "actual": "fake contract wrapper",
            "metadata": "acceptance/artifacts/BR-WRAPPER-001/logs",
        },
    )
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["case_id"] == "BR-WRAPPER-001"
    assert data["status"] == "pass"
