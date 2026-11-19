"""Export runs to Docent and read them back.

Every generator and both eval suites take ``export_docent=true``. The run then uploads one Docent
agent run per item, prints the collection link, and saves it as ``docent_collection_url`` in its
output metadata. Rerunning reuses the collection (its id is saved as ``docent_collection_id`` in the
output directory) and skips items already uploaded.

Inputs can come from Docent too: anywhere a training run or eval takes a data path, a collection
link or ``docent://<collection_id>`` works (see ``load_items``).

Docent only knows the user, assistant, system and tool roles, so each message keeps its real role
(``query``, ``observer``, ...) in ``metadata["dataset_role"]``. ``download`` reverses this:

    python -m introspect.docent collection=<id or link> output=data/sft/from_docent.json kind=sft

Needs the optional extra (``uv sync --extra docent``) and ``DOCENT_API_KEY``. The key is read from
the environment, never from a config, so it never lands in output files or wandb.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import chz

if TYPE_CHECKING:
    from docent import Docent
    from docent.data_models import AgentRun

COLLECTION_ID_FILE = "docent_collection_id"
_ASSISTANT_ROLES = {"assistant", "observer"}


def check_api_key() -> None:
    """Fail before any work starts if an export would fail at the end."""
    if not os.environ.get("DOCENT_API_KEY"):
        raise ValueError("export_docent=true needs the DOCENT_API_KEY environment variable")
    try:
        import docent  # noqa: F401
    except ImportError as error:
        raise ValueError("export_docent=true needs the docent extra: uv sync --extra docent") from error


@chz.chz
class DocentOptions:
    """Mixed into every generator and eval config."""

    export_docent: bool = False  # upload results to Docent and save the collection link
    docent_collection: str | None = None  # collection name; None names it after the run

    @chz.validate
    def _check_docent(self) -> None:
        if self.export_docent:
            check_api_key()


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def agent_run(
    name: str, messages: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> AgentRun:
    """One Docent run holding one conversation in this repo's ``{"role", "content"}`` format."""
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import AssistantMessage, SystemMessage, UserMessage

    converted = []
    for message in messages:
        role = message["role"]
        content = _text(message["content"])
        if message.get("tool_calls"):  # keep tool calls visible without modeling them
            content += "\n\n[TOOL CALLS]\n" + _text(message["tool_calls"])
        tagged = {"dataset_role": role}
        if role in _ASSISTANT_ROLES:
            converted.append(AssistantMessage(content=content, metadata=tagged))
        elif role == "system":
            converted.append(SystemMessage(content=content, metadata=tagged))
        else:  # user, query, tool, ...
            converted.append(UserMessage(content=content, metadata=tagged))
    json_metadata = json.loads(json.dumps(dict(metadata), ensure_ascii=False, default=str))
    return AgentRun(
        name=name,
        transcripts=[Transcript(name="conversation", messages=converted, metadata=json_metadata)],
        metadata=json_metadata,
    )


def collection_url(client: Docent, collection_id: str) -> str:
    return f"{client._frontend_url}/dashboard/{collection_id}"


def export(
    runs: Sequence[AgentRun], *, output_dir: str | Path, name: str, metadata: Mapping[str, Any]
) -> str:
    """Upload ``runs`` to the run's collection (created on first use) and return its link."""
    from docent import Docent

    client = Docent()
    state = Path(output_dir) / COLLECTION_ID_FILE
    if state.exists():
        collection_id = state.read_text().strip()
    else:
        collection_id = client.create_collection(
            name=name, metadata=json.loads(json.dumps(dict(metadata), default=str))
        )
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(collection_id)
    result = client.execute_dql(
        collection_id, "SELECT agent_runs.name FROM agent_runs", ensure_latest_results=True
    )
    uploaded = {row[0] for row in result["rows"]}
    new = [run for run in runs if run.name not in uploaded]
    if new:
        client.add_agent_runs(collection_id, list(new), wait=True)
    url = collection_url(client, collection_id)
    print(f"Docent: {len(new)} new runs ({len(runs) - len(new)} already there) -> {url}")
    return url


# --- Reading back ----------------------------------------------------------------------------


def is_docent_source(path: str) -> bool:
    """Whether an input path names a Docent collection instead of a file."""
    return path.startswith("docent://") or bool(re.search(r"docent[^/]*/dashboard/", path))


def collection_id_from(collection: str) -> str:
    """Accept ``docent://<id>``, a bare collection id, or any Docent link that contains one."""
    match = re.search(r"/dashboard/([^/?#]+)", collection)
    if match:
        return match.group(1)
    return collection.strip().removeprefix("docent://")


@functools.lru_cache(maxsize=8)  # one download per collection per process
def download(collection: str) -> list[dict[str, Any]]:
    """Every run as ``{"name", "metadata", "messages"}`` with the original roles restored."""
    from docent import Docent

    client = Docent()
    collection_id = collection_id_from(collection)
    run_ids = client.list_agent_run_ids(collection_id)
    with ThreadPoolExecutor(16) as pool:
        runs = list(pool.map(lambda run_id: client.get_agent_run(collection_id, run_id), run_ids))
    rows = []
    for run in runs:
        if run is None or not run.transcripts:
            continue
        messages = [
            {"role": (message.metadata or {}).get("dataset_role", message.role), "content": message.text}
            for message in run.transcripts[0].messages
        ]
        rows.append({"name": run.name, "metadata": run.metadata, "messages": messages})
    return sorted(rows, key=lambda row: row["name"] or "")


def to_sft_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Runs from ``generate_sft`` back into train_sft.py items (hidden system prompts dropped)."""
    return [
        {"messages": [m for m in row["messages"] if m["role"] != "system"]}
        for row in rows
        if row["messages"] and row["messages"][-1]["role"] == "observer"
    ]


def to_rl_items(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    """Runs from ``generate_rl`` back into train_rl.py items for one split."""
    items = []
    for row in rows:
        metadata = row["metadata"]
        if metadata.get("split") != split:
            continue
        *context, query = row["messages"]
        items.append(
            {
                "item_id": metadata["item_id"],
                "prompt_group": metadata.get("prompt_group"),
                "context_messages": context,
                "query": query["content"],
                "answer": metadata["answer"],
                "metadata": metadata.get("item_metadata", {}),
            }
        )
    return items


def to_eval_items(rows: Sequence[Mapping[str, Any]], task: str) -> list[dict[str, Any]]:
    """Eval questions for one task, from a collection written by an eval run with export_docent.

    Each run stores its original question item in ``metadata["eval_item"]``, so tool calls and
    every field come back exactly; the summary runs carry none and are skipped.
    """
    rows = [row for row in rows if row["metadata"].get("task") == task and "eval_item" in row["metadata"]]
    return [row["metadata"]["eval_item"] for row in sorted(rows, key=lambda r: r["metadata"]["eval_index"])]


def load_items(
    source: str,
    kind: Literal["sft", "rl", "eval"],
    *,
    split: str = "train",
    task: str = "",
    snapshot_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Items from a Docent collection in the format a file of that kind would hold.

    With ``snapshot_dir``, the items are also saved there, so a training run records exactly
    what it read.
    """
    rows = download(source)
    if kind == "sft":
        items = to_sft_items(rows)
    elif kind == "rl":
        items = to_rl_items(rows, split)
    else:
        items = to_eval_items(rows, task)
    if not items:
        raise ValueError(f"Docent collection {source} has no {kind} items ({split or task})")
    if snapshot_dir is not None:
        name = f"docent_{collection_id_from(source)}_{kind}_{split if kind == 'rl' else task or 'items'}.json"
        path = Path(snapshot_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": {"source": source}, "items": items}, ensure_ascii=False))
    return items


@chz.chz
class DownloadConfig:
    collection: str  # collection id or Docent link
    output: str  # a file, or for kind=rl a directory
    # runs: every run as-is; sft: a train_sft.py file; rl: train.json + held_out.json
    kind: Literal["runs", "sft", "rl"] = "runs"


def main(cfg: DownloadConfig) -> None:
    rows = download(cfg.collection)
    output = Path(cfg.output)
    source = {"source": "docent", "docent_collection_id": collection_id_from(cfg.collection)}
    if cfg.kind == "rl":
        output.mkdir(parents=True, exist_ok=True)
        for split in ("train", "held_out"):
            items = to_rl_items(rows, split)
            (output / f"{split}.json").write_text(json.dumps({"metadata": source, "items": items}))
            print(f"wrote {len(items)} {split} items to {output / f'{split}.json'}")
        return
    items = to_sft_items(rows) if cfg.kind == "sft" else rows
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metadata": source, "items": items}, ensure_ascii=False))
    print(f"wrote {len(items)} items to {output}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
