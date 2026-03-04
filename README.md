# llm-coreml

An [llm](https://github.com/simonw/llm) plugin that runs CoreML `.mlpackage` LLM models locally on macOS. Point it at a model and a HuggingFace tokenizer, then prompt it like any other llm model.

## Requirements

- macOS (CoreML is Apple-only)
- Python 3.11+

## Installation

```bash
llm install llm-coreml
```

Or for development:

```bash
git clone https://github.com/anentropic/llm-coreml.git
cd llm-coreml
llm install -e .
```

## Quick start

Register a model with a name and a path to the `.mlpackage`. The `--tokenizer` argument is the HuggingFace model name to load the tokenizer from. This should match the model your `.mlpackage` was converted from:

```bash
llm coreml add my-llama /path/to/llama.mlpackage \
    --tokenizer meta-llama/Llama-3.2-1B-Instruct
```

Prompt it:

```bash
llm -m coreml/my-llama "Explain quantum computing in one sentence"
```

Check it shows up in `llm models`:

```bash
llm models | grep coreml
```

## Usage

### Prompting

```bash
# Basic prompt
llm -m coreml/my-llama "What is Rust?"

# With a system prompt
llm -m coreml/my-llama "Hello" -s "You are a pirate"

# Continue a conversation
llm -m coreml/my-llama "What is Rust?"
llm -c "Compare it to Go"
```

### Model options

Pass options with `-o`:

```bash
llm -m coreml/my-llama "Write a haiku" \
    -o temperature 0.7 \
    -o top_p 0.9 \
    -o max_tokens 50
```

### Python API

```python
import llm

model = llm.get_model("coreml/my-llama")
response = model.prompt("What is the capital of France?")
print(response.text())
```

## CLI reference

### `llm coreml add`

```
llm coreml add <name> <path> --tokenizer <hf_id> [--compute-units <units>]
```

Register a CoreML model.

| Argument | Description |
|---|---|
| `name` | Model name, used as `coreml/<name>` |
| `path` | Path to the `.mlpackage` directory (resolved to absolute) |
| `--tokenizer` | HuggingFace tokenizer model ID (required) |
| `--compute-units` | Compute units: `all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_ne` (default: `all`) |

### `llm coreml list`

```
llm coreml list
```

Lists registered models with their paths, tokenizer IDs, and compute units.

### `llm coreml remove`

```
llm coreml remove <name>
```

Removes a registered model. Exits with code 1 if the model doesn't exist.

## Model options reference

| Option | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | int | 200 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature. 0 = greedy (deterministic) |
| `top_p` | float | 1.0 | Top-p nucleus sampling threshold |

## How it works

### Format auto-detection

The plugin reads the CoreML model spec at load time and checks the input names:

- `inputIds` (camelCase) = Apple format, uses float16 causal masks
- `input_ids` (snake_case) = HuggingFace format, uses int32 attention masks

No config file needed.

### Stateful KV-cache

If the model spec declares `stateDescriptions`, the plugin uses stateful inference with KV-cache. Otherwise it falls back to stateless inference, which reprocesses the full sequence each step (slower, but works with older models).

### Tokenization

The plugin uses `transformers.AutoTokenizer` with `apply_chat_template()` to handle chat formatting. The tokenizer is downloaded and cached the first time you use a model.

## Getting CoreML models

You can get `.mlpackage` LLM models by:

- Converting HuggingFace models with [coremltools](https://apple.github.io/coremltools/docs-guides/)
- Using Apple's [ml-explore](https://github.com/ml-explore) tools
- Downloading pre-converted models from HuggingFace (search for "coreml" tagged models)

## Development

```bash
uv sync --dev
```

### Quality gates

```bash
uv run basedpyright      # Type checking (strict)
uv run ruff check        # Linting
uv run ruff format       # Formatting
uv run pytest            # Tests
```

Or all at once:

```bash
prek run --all-files
```

## License

MIT
