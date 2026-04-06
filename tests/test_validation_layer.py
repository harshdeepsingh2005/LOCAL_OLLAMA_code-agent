from src.orchestration.context_pipeline import ValidationLayer


def test_validation_layer_accepts_write_style_tasks():
    validator = ValidationLayer()
    errors = validator.validate_task(
        "Write a Python script that reads a JSON file and prints total users."
    )
    assert errors == []


def test_validation_layer_rejects_ambiguous_task_without_action():
    validator = ValidationLayer()
    errors = validator.validate_task("this thing maybe somehow")
    assert any("ambiguous" in err.lower() for err in errors)
