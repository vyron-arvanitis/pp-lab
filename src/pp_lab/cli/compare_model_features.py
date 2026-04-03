from __future__ import annotations

import argparse

CATEGORY_CHOICES = ["feature_importance", "best_models", "graph_variants"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare validation metrics across trained models.")
    parser.add_argument("--category", choices=sorted(CATEGORY_CHOICES), default=None)
    parser.add_argument("--base-path", default="saved_models")
    parser.add_argument("--output-dir", default="compare_model_features")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from pp_lab.reporting import CATEGORIES, compare_models

    categories = [args.category] if args.category else list(CATEGORIES)
    for category in categories:
        compare_models(category=category, base_path=args.base_path, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
