default:
    @just --list

model_name := "tiny-gpt2"
model_id := "sshleifer/tiny-gpt2"
model_dir := ".models" / model_name

# Pinned deps for isolated environments (coremltools native extensions need Python <=3.13)
_python := "3.13"
_isolated_deps := "--with 'coremltools>=8.0' --with numpy --with 'transformers>=4.35,<4.46'"

# Convert tiny-gpt2 to CoreML for integration testing
convert-test-model:
    uv run --isolated --python {{ _python }} \
        --with 'torch>=2.7,<2.8' \
        {{ _isolated_deps }} \
        python scripts/convert_model.py \
        {{ model_id }} {{ model_dir }}

# Run unit tests only (excludes integration)
test *args:
    uv run pytest -m "not integration" {{ args }}

# Run integration tests only (requires `just convert-test-model` first)
# Uses --isolated with Python 3.13 because coremltools native extensions require it
test-integration *args: convert-test-model
    uv run --isolated --python {{ _python }} \
        {{ _isolated_deps }} \
        --with pytest \
        --with llm \
        pytest -m integration {{ args }}

# Run all tests (unit + integration)
test-all: convert-test-model
    uv run --isolated --python {{ _python }} \
        {{ _isolated_deps }} \
        --with pytest \
        --with llm \
        pytest

# Run all quality gates
check:
    prek run --all-files
