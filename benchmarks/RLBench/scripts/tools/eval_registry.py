#!/usr/bin/env python3
"""Inspect and resolve the centralized RLBench checkpoint-task registry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


# argparse rejects both members of these BooleanOptional-style groups even
# though registry layers intentionally use a last-writer-wins override model.
# Normalize them after defaults/checkpoint/task arguments have been merged.
MUTUALLY_EXCLUSIVE_FLAG_GROUPS = (
    frozenset(("--collision-checking", "--no-collision-checking")),
    frozenset(("--add-gripper-cloud", "--no-add-gripper-cloud")),
    frozenset(("--gripper-lock-after-close", "--no-gripper-lock-after-close")),
)


def normalize_mutually_exclusive_flags(args: list[str]) -> list[str]:
    """Keep only the last occurrence from each known flag group."""
    last_index: dict[frozenset[str], int] = {}
    group_by_flag = {
        flag: group for group in MUTUALLY_EXCLUSIVE_FLAG_GROUPS for flag in group
    }
    for index, item in enumerate(args):
        group = group_by_flag.get(item)
        if group is not None:
            last_index[group] = index
    return [
        item
        for index, item in enumerate(args)
        if (group := group_by_flag.get(item)) is None or last_index[group] == index
    ]


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("version") != 1:
        raise ValueError(f"Unsupported registry version in {path}: {registry.get('version')!r}")
    checkpoints = registry.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError(f"Registry has no checkpoints: {path}")
    return registry


def checkpoint_candidates(value: str) -> set[Path]:
    path = Path(os.path.expandvars(value)).expanduser().resolve()
    candidates = {path}
    if path.name == "pretrained_model":
        candidates.add(path.parent)
    elif path.name == "checkpoints":
        pass
    else:
        candidates.add(path / "pretrained_model")
    return candidates


def resolve_checkpoint(registry: dict[str, Any], value: str) -> tuple[str, dict[str, Any]]:
    checkpoints = registry["checkpoints"]
    if value in checkpoints:
        return value, checkpoints[value]
    requested = checkpoint_candidates(value)
    matches: list[tuple[str, dict[str, Any]]] = []
    for checkpoint_id, config in checkpoints.items():
        registered = checkpoint_candidates(str(config["policy_path"]))
        if requested & registered:
            matches.append((checkpoint_id, config))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        aliases = ", ".join(item[0] for item in matches)
        raise ValueError(
            f"Path {value!r} matches multiple parameter profiles: {aliases}. "
            "Select one by checkpoint alias."
        )
    available = ", ".join(sorted(checkpoints))
    raise ValueError(f"Unknown checkpoint/profile {value!r}. Available: {available}")


def supported_tasks(checkpoint: dict[str, Any]) -> list[str]:
    tasks = checkpoint.get("tasks", {})
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Checkpoint has no registered tasks")
    return list(tasks)


def training_setting(checkpoint_id: str, checkpoint: dict[str, Any]) -> str:
    value = str(checkpoint.get("training_setting", "")).strip()
    if not value:
        raise ValueError(f"{checkpoint_id}: training_setting must be a non-empty string")
    if any(character.isspace() for character in value) or "__" in value:
        raise ValueError(
            f"{checkpoint_id}: training_setting must not contain whitespace or '__': {value!r}"
        )
    return value


def checkpoint_step(checkpoint_id: str, checkpoint: dict[str, Any]) -> str:
    policy_path = Path(os.path.expandvars(checkpoint["policy_path"])).expanduser()
    step = policy_path.parent.name if policy_path.name == "pretrained_model" else policy_path.name
    if not step.isdigit():
        raise ValueError(
            f"{checkpoint_id}: cannot derive numeric checkpoint step from {policy_path}"
        )
    return step.zfill(6)


def resolve_task(
    registry: dict[str, Any], checkpoint: dict[str, Any], task: str
) -> tuple[dict[str, str], list[str], str]:
    tasks = checkpoint.get("tasks", {})
    if task not in tasks:
        available = ", ".join(tasks)
        raise ValueError(f"Task {task!r} is not registered for this checkpoint. Available: {available}")
    task_config = tasks[task] or {}
    if "env" in task_config:
        raise ValueError(
            f"Task {task!r} defines env values. Per-task differences must use CLI args "
            "because one multi-task evaluator process shares its environment."
        )
    env: dict[str, str] = {}
    env.update({str(k): str(v) for k, v in registry.get("defaults", {}).get("env", {}).items()})
    env.update({str(k): str(v) for k, v in checkpoint.get("env", {}).items()})
    args: list[str] = []
    args.extend(str(item) for item in registry.get("defaults", {}).get("args", []))
    args.extend(str(item) for item in checkpoint.get("args", []))
    args.extend(str(item) for item in task_config.get("args", []))
    args = normalize_mutually_exclusive_flags(args)
    return env, args, str(task_config.get("notes", ""))


def emit_null(items: list[str]) -> None:
    for item in items:
        sys.stdout.buffer.write(item.encode("utf-8") + b"\0")


def validate_registry(registry: dict[str, Any], check_paths: bool) -> None:
    for checkpoint_id, checkpoint in registry["checkpoints"].items():
        if not isinstance(checkpoint.get("policy_path"), str):
            raise ValueError(f"{checkpoint_id}: policy_path must be a string")
        training_setting(checkpoint_id, checkpoint)
        checkpoint_step(checkpoint_id, checkpoint)
        tasks = supported_tasks(checkpoint)
        for task in tasks:
            resolve_task(registry, checkpoint, task)
        if check_paths:
            model_path = Path(os.path.expandvars(checkpoint["policy_path"])).expanduser()
            required = model_path / "model.safetensors"
            if not required.is_file():
                raise FileNotFoundError(f"{checkpoint_id}: missing {required}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List checkpoint aliases and supported tasks")

    validate = subparsers.add_parser("validate", help="Validate registry structure")
    validate.add_argument("--check-paths", action="store_true")

    for command in (
        "checkpoint-id",
        "training-setting",
        "checkpoint-step",
        "policy",
        "tasks",
        "env",
        "args",
        "show",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--checkpoint", required=True)
        if command in ("args", "show"):
            child.add_argument("--task", required=True)
        if command in ("env", "args"):
            child.add_argument("--format", choices=("lines", "null", "json", "shell"), default="lines")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry_path = args.registry.expanduser().resolve()
    registry = load_registry(registry_path)

    if args.command == "validate":
        validate_registry(registry, args.check_paths)
        print(f"registry_ok={registry_path}")
        print(f"checkpoints={len(registry['checkpoints'])}")
        return

    if args.command == "list":
        for checkpoint_id, checkpoint in registry["checkpoints"].items():
            task_names = ",".join(supported_tasks(checkpoint))
            description = checkpoint.get("description", "")
            print(f"{checkpoint_id}\t{task_names}\t{description}")
        return

    checkpoint_id, checkpoint = resolve_checkpoint(registry, args.checkpoint)
    if args.command == "checkpoint-id":
        print(checkpoint_id)
        return
    if args.command == "training-setting":
        print(training_setting(checkpoint_id, checkpoint))
        return
    if args.command == "checkpoint-step":
        print(checkpoint_step(checkpoint_id, checkpoint))
        return
    if args.command == "policy":
        print(Path(os.path.expandvars(checkpoint["policy_path"])).expanduser().resolve())
        return
    if args.command == "tasks":
        print("\n".join(supported_tasks(checkpoint)))
        return

    task = args.task if hasattr(args, "task") else supported_tasks(checkpoint)[0]
    env, resolved_args, notes = resolve_task(registry, checkpoint, task)
    if args.command == "show":
        document = {
            "checkpoint": checkpoint_id,
            "training_setting": training_setting(checkpoint_id, checkpoint),
            "checkpoint_step": checkpoint_step(checkpoint_id, checkpoint),
            "description": checkpoint.get("description", ""),
            "policy_path": str(Path(os.path.expandvars(checkpoint["policy_path"])).expanduser().resolve()),
            "task": task,
            "env": env,
            "args": resolved_args,
            "notes": notes,
        }
        print(json.dumps(document, indent=2, ensure_ascii=False))
        return

    if args.command == "env":
        items = [f"{key}={value}" for key, value in env.items()]
    else:
        items = resolved_args
    if args.format == "null":
        emit_null(items)
    elif args.format == "json":
        print(json.dumps(items, ensure_ascii=False))
    elif args.format == "shell":
        print(" ".join(shlex.quote(item) for item in items))
    else:
        print("\n".join(items))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"registry error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
