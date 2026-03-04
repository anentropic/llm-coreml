"""
Convert a HuggingFace causal LM to CoreML .mlpackage format.

Run via `just convert-test-model` which uses `uv run --isolated` with
pinned dependencies to avoid version conflicts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def convert(model_id: str, output_dir: Path, max_seq_len: int = 512) -> None:
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    # Wrapper maps positional args to keyword args correctly,
    # since GPT2's forward signature has past_key_values before attention_mask.
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )[0]  # logits

    wrapper = Wrapper(model)
    wrapper.eval()

    # Trace with example input
    example = tokenizer("Hello world", return_tensors="pt")
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (input_ids, attention_mask))

    # Convert to CoreML with flexible sequence length.
    # Uses neuralnetwork format because mlprogram segfaults on predict()
    # with GPT-2 models (coremltools #1960, #2166).
    seq_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=input_ids.shape[1])
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, seq_dim), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, seq_dim), dtype=np.int32),
    ]

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="neuralnetwork",
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / f"{Path(model_id).name}.mlpackage"
    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))

    # Save tokenizer alongside model
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved to {output_dir}")

    # Show output names for debugging
    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    print(f"Output names: {output_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="HuggingFace model ID (e.g. sshleifer/tiny-gpt2)")
    parser.add_argument("output_dir", type=Path, help="Output directory for .mlpackage")
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Maximum sequence length (default: 512)"
    )
    args = parser.parse_args()
    convert(args.model_id, args.output_dir, args.max_seq_len)


if __name__ == "__main__":
    main()
