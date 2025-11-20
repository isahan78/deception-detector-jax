#!/usr/bin/env python3
"""Generate synthetic deception task datasets."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deception_detector_jax.config import DatasetConfig
from deception_detector_jax.data.deception_tasks import (
    generate_full_dataset,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic deception task datasets"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["hidden_check", "secret_goal", "concealed_step"],
        help="Which deception task to generate",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for datasets",
    )

    parser.add_argument(
        "--num-train",
        type=int,
        default=10000,
        help="Number of training examples",
    )

    parser.add_argument(
        "--num-val",
        type=int,
        default=1000,
        help="Number of validation examples",
    )

    parser.add_argument(
        "--num-test",
        type=int,
        default=1000,
        help="Number of test examples",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128,
        help="Vocabulary size",
    )

    parser.add_argument(
        "--deception-rate",
        type=float,
        default=0.3,
        help="Fraction of deceptive examples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = DatasetConfig(
        task_name=args.task,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        deception_rate=args.deception_rate,
        seed=args.seed,
    )

    print(f"Generating {args.task} dataset...")
    print(f"  Train: {args.num_train}")
    print(f"  Val: {args.num_val}")
    print(f"  Test: {args.num_test}")
    print(f"  Deception rate: {args.deception_rate}")

    # Generate datasets
    train_data, val_data, test_data = generate_full_dataset(config)

    # Save datasets
    task_dir = output_dir / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    train_path = task_dir / "train.npz"
    val_path = task_dir / "val.npz"
    test_path = task_dir / "test.npz"

    save_dataset(str(train_path), train_data)
    save_dataset(str(val_path), val_data)
    save_dataset(str(test_path), test_data)

    print("\n✓ Dataset generation complete!")
    print(f"  Saved to: {task_dir}")

    # Print sample statistics
    print("\nDataset statistics:")
    print(f"  Input shape: {train_data['input_ids'].shape}")
    print(f"  Target shape: {train_data['target_ids'].shape}")

    if "forbidden" in train_data:
        forbidden_rate = train_data["forbidden"].mean()
        print(f"  Forbidden rate: {forbidden_rate:.2%}")

    if "deceptive" in train_data:
        deceptive_rate = train_data["deceptive"].mean()
        print(f"  Deceptive rate: {deceptive_rate:.2%}")


if __name__ == "__main__":
    main()
