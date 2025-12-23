"""
Evaluation runner for running evals on datasets.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from genai_project.core.logging import get_logger
from genai_project.eval.metrics import EvalResult, evaluate_all

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """A single evaluation case."""

    id: str
    input: str
    expected: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRun:
    """Results of an evaluation run."""

    dataset: str
    total_cases: int
    results: list[dict[str, Any]]
    aggregate_scores: dict[str, float]


class EvalRunner:
    """Runner for evaluating LLM outputs against datasets."""

    def __init__(
        self,
        datasets_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize evaluation runner.

        Args:
            datasets_dir: Directory containing evaluation datasets
        """
        base_dir = Path(__file__).parent
        self.datasets_dir = Path(datasets_dir) if datasets_dir else base_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, name: str) -> list[EvalCase]:
        """
        Load evaluation dataset from JSONL file.

        Expected format:
        {"id": "1", "input": "...", "expected": "...", "metadata": {...}}
        """
        filepath = self.datasets_dir / f"{name}.jsonl"

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        cases = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cases.append(
                        EvalCase(
                            id=data["id"],
                            input=data["input"],
                            expected=data["expected"],
                            metadata=data.get("metadata", {}),
                        )
                    )

        logger.info("Loaded dataset", name=name, cases=len(cases))
        return cases

    async def run_eval(
        self,
        dataset_name: str,
        generate_fn: Callable[[str], Any],
        metrics: list[str] | None = None,
    ) -> EvalRun:
        """
        Run evaluation on a dataset.

        Args:
            dataset_name: Name of the dataset to evaluate
            generate_fn: Async function that takes input and returns prediction
            metrics: List of metrics to compute (default: all)

        Returns:
            EvalRun with results
        """
        cases = self.load_dataset(dataset_name)
        results = []
        all_scores: dict[str, list[float]] = {}

        for case in cases:
            try:
                # Generate prediction
                prediction = await generate_fn(case.input)

                # Evaluate
                eval_results = evaluate_all(prediction, case.expected)

                # Filter metrics if specified
                if metrics:
                    eval_results = {k: v for k, v in eval_results.items() if k in metrics}

                # Collect scores
                scores = {}
                for metric_name, result in eval_results.items():
                    scores[metric_name] = result.score
                    if metric_name not in all_scores:
                        all_scores[metric_name] = []
                    all_scores[metric_name].append(result.score)

                results.append(
                    {
                        "id": case.id,
                        "input": case.input,
                        "expected": case.expected,
                        "prediction": prediction,
                        "scores": scores,
                    }
                )

                logger.debug(
                    "Evaluated case",
                    id=case.id,
                    scores=scores,
                )

            except Exception as e:
                logger.error("Evaluation failed for case", id=case.id, error=str(e))
                results.append(
                    {
                        "id": case.id,
                        "error": str(e),
                    }
                )

        # Calculate aggregate scores
        aggregate_scores = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in all_scores.items()
        }

        logger.info(
            "Evaluation complete",
            dataset=dataset_name,
            total=len(cases),
            aggregate_scores=aggregate_scores,
        )

        return EvalRun(
            dataset=dataset_name,
            total_cases=len(cases),
            results=results,
            aggregate_scores=aggregate_scores,
        )

    def save_results(
        self,
        run: EvalRun,
        output_path: str | Path,
    ) -> None:
        """Save evaluation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "dataset": run.dataset,
                    "total_cases": run.total_cases,
                    "aggregate_scores": run.aggregate_scores,
                    "results": run.results,
                },
                f,
                indent=2,
            )

        logger.info("Saved results", path=str(output_path))
