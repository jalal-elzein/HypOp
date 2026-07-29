"""Shared W&B logging wrapper, implementing the schema in wandb_tracking_spec.md.

This keeps step-counter/phase bookkeeping in one place so call sites in
solver.py only need to report a (phase, duration, quality) triple per phase
and don't have to reimplement the monotonic step counter or the
pre/post_refinement -> summary field mapping themselves.
"""

import os
from pathlib import Path

# `wandb` is imported lazily inside RunLogger, not at module level, so that
# solver.py (which imports this module unconditionally) still works in
# environments where wandb isn't installed and wandb_enabled is False.

# Maps this codebase's `params['mode']` values onto the shared `problem_type`
# config field used across all three codebases (see spec Section 2).
PROBLEM_TYPE_BY_MODE = {
    "sat": "maxsat",
    "maxcut": "maxcut",
    "QUBO_maxcut": "maxcut",
    "maxcut_annea": "maxcut",
}


def compute_instance_id(path):
    """Instance file path -> id relative to BENCHMARK_ROOT (spec Section 5).

    Must be relative to the shared BENCHMARK_ROOT, not to cwd or this
    codebase's own --folder_path, so instance_id lines up with the other two
    codebases' runs on the same underlying file.
    """
    root = os.environ.get("BENCHMARK_ROOT")
    if not root:
        raise RuntimeError(
            "BENCHMARK_ROOT is not set. It must point to the shared benchmark "
            "root directory (identically in effect across all three codebases) "
            "so that instance_id is joinable across methods - see "
            "wandb_tracking_spec.md, Section 5."
        )
    benchmark_root = Path(root).resolve()
    instance_path = Path(path).resolve()
    return str(instance_path.relative_to(benchmark_root)).replace(os.sep, "/")


def quality_from_score(problem_type, score, num_constraints):
    """solver.py objective score -> the schema's normalized (quality, feasible).

    Mirrors the conventions of loss_sat_numpy_boost / loss_maxcut_numpy_boost
    in src/loss.py:
      - loss_sat_numpy_boost sums to the number of UNsatisfied clauses.
      - loss_maxcut_numpy_boost sums to -(number of cut hyperedges).
    """
    if not num_constraints:
        return None, None
    score = float(score)
    if problem_type == "maxsat":
        quality = 1.0 - score / num_constraints
        feasible = score == 0
        return quality, feasible
    if problem_type == "maxcut":
        quality = -score / num_constraints
        return quality, None
    return None, None


class RunLogger:
    """One instance = one (instance, method, condition, seed) W&B run.

    Owns the step counter so it increments monotonically across the
    pre_refinement -> post_refinement boundary (never resets to 0), per
    wandb_tracking_spec.md Section 3.
    """

    def __init__(
        self,
        *,
        problem_type,
        instance_id,
        instance_size,
        difficulty_param,
        seed,
        condition="default",
        pretrain_scale="n/a",
        time_budget_s=None,
        project=None,
        method="hypop",
    ):
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set. Refusing to fall back to offline "
                "mode, since offline runs are easy to lose track of across "
                "codebases - see wandb_tracking_spec.md, Section 1."
            )
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "hypergraph-csp-eval"),
            config={
                "method": method,
                "problem_type": problem_type,
                "instance_id": instance_id,
                "instance_size": instance_size,
                "difficulty_param": difficulty_param,
                "seed": seed,
                "condition": condition,
                "pretrain_scale": pretrain_scale,
                "time_budget_s": time_budget_s,
            },
        )
        self._step = 0
        self._elapsed_s = 0.0

    def log_phase_end(self, phase, phase_duration_s, quality, feasible=None):
        """Log the single end-of-phase point required for baselines (spec Section 8)."""
        self._elapsed_s += phase_duration_s
        record = {"phase": phase, "wall_time_s": self._elapsed_s, "quality": quality}
        if feasible is not None:
            record["feasible"] = feasible
        self._wandb.log(record, step=self._step)
        self._step += 1

        if phase == "pre_refinement":
            self.run.summary["t_pre_refinement_total_s"] = phase_duration_s
            self.run.summary["final_quality_raw"] = quality
        elif phase == "post_refinement":
            self.run.summary["t_post_refinement_total_s"] = phase_duration_s
            self.run.summary["final_quality_refined"] = quality

    def finish(self):
        self._wandb.finish()
