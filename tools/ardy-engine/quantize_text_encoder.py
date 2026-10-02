# quantize_text_encoder.py — llm2vec-base-merged (bf16, ~16GB) を bitsandbytes で 4bit / 8bit へ量子化
#
# 使い方:
#   python quantize_text_encoder.py --in <merged-base> --out <quantized-dir> --bits 4 --verify 3
#
# - 4bit (nf4 + double-quant + bf16 compute) → 約 4.5GB
# - 8bit (load_in_8bit) → 約 8.7GB
# - 保存後は config.json に quantization_config が入るため、server.py --merged-base で自動的に量子化済みとして読み込まれる
# - 検証: 再読み込みしてサンプル文をエンコード、bf16 基準とコサイン類似度 >= 0.98 で PASS
#
# 要 bitsandbytes (engine venv で `pip install bitsandbytes`)

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig, AutoTokenizer

# build_text_encoder.py と同様に ardy のモデルを使用
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel
except ImportError as e:
    print(f"LlamaBiModel import failed: {e}")
    print("Run inside ARDY engine venv (ardy must be editable installed)")
    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description="Quantize LLM2Vec merged base to 4-bit/8-bit")
    ap.add_argument("--in", dest="src", required=True, help="Path to llm2vec-base-merged")
    ap.add_argument("--out", required=True, help="Output directory for quantized model")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits (4 or 8)")
    ap.add_argument("--verify", type=int, default=3, help="Number of verification samples (0 to skip)")
    ap.add_argument("--device", default="auto", help="device_map for loading (auto/cuda/cpu)")
    return ap.parse_args()


def build_bnb_config(bits: int) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        return BitsAndBytesConfig(load_in_8bit=True)


def main():
    args = parse_args()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.exists():
        print(f"Source not found: {src}")
        sys.exit(1)

    if out.exists():
        print(f"Output exists, removing: {out}")
        shutil.rmtree(out)

    print(f"Loading source: {src}")
    print(f"Starting {args.bits}-bit quantization...")

    bnb_config = build_bnb_config(args.bits)

    # Quantized load: converts shard by shard, never holds full bf16 on GPU
    model = LlamaBiModel.from_pretrained(
        str(src),
        quantization_config=bnb_config,
        device_map=args.device,
        dtype=torch.bfloat16,
    )

    print(f"Saving to: {out}")
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)

    # Restore _name_or_path (same as build_text_encoder.py)
    # vendored llm2vec uses this to pick Llama-3-8B-Instruct instruction wrapper
    cfg_path = out / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_name_or_path"] = "meta-llama/Meta-Llama-3-8B-Instruct"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Marker file (same convention as installer)
    (out / "quantize-complete.marker").write_text(f"bits={args.bits}\n")

    # Size report
    src_size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    out_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    ratio = out_size / src_size * 100
    print(f"\n=== SIZE REPORT ===")
    print(f"Source:     {src_size / 1024**3:.2f} GB")
    print(f"Quantized:  {out_size / 1024**3:.2f} GB ({ratio:.1f}%)")
    print(f"Reduction:  {(1 - ratio/100)*100:.1f}%")

    # Verification
    if args.verify > 0:
        print(f"\n=== VERIFICATION ({args.verify} samples) ===")
        samples = [
            "A person waves their right hand.",
            "Walking forward while looking around.",
            "Sadly bowing head with shoulders slumped.",
            "Jumping up with both arms raised.",
            "Sitting down and crossing legs.",
        ][:args.verify]

        # Reload quantized model (auto-detects quantization_config from config.json)
        print("Reloading quantized model...")
        model_q = LlamaBiModel.from_pretrained(str(out), device_map=args.device)
        model_q.eval()

        # bf16 baseline on CPU if RAM allows
        baseline = None
        try:
            print("Loading bf16 baseline on CPU (RAM permitting)...")
            baseline = LlamaBiModel.from_pretrained(str(src), device_map="cpu", dtype=torch.bfloat16)
            baseline.eval()
        except Exception as e:
            print(f"  Baseline skipped (RAM/CPU): {e}")

        passed = True
        for i, text in enumerate(samples):
            with torch.no_grad():
                out_q = model_q.encode([text], batch_size=1, show_progress_bar=False, device=args.device if args.device != "auto" else "cuda")
                out_q = out_q.float()

            if baseline is not None:
                with torch.no_grad():
                    out_b = baseline.encode([text], batch_size=1, show_progress_bar=False, device="cpu")
                    out_b = out_b.float()
                cos = torch.nn.functional.cosine_similarity(out_q, out_b, dim=-1).mean().item()
                ok = cos >= 0.98
                mark = "OK" if ok else "FAIL"
                if not ok:
                    passed = False
                print(f"  Sample {i+1}: cosine={cos:.6f} {mark}")
            else:
                has_nan = torch.isnan(out_q).any().item()
                has_inf = torch.isinf(out_q).any().item()
                ok = not (has_nan or has_inf)
                mark = "OK" if ok else "FAIL"
                if not ok:
                    passed = False
                print(f"  Sample {i+1}: dim={out_q.shape[-1]}, finite={ok} {mark}")

        if passed:
            print("\n=== VERIFICATION PASS ===")
        else:
            print("\n=== VERIFICATION FAIL ===")
            sys.exit(1)

    print(f"\nDone! Quantized model: {out}")
    print("Launch command:")
    print(f"  TEXT_ENCODER_DEVICE=cuda python server.py --merged-base {out}")


if __name__ == "__main__":
    main()