#!/usr/bin/env python3
"""Select, download and verify completed scale diagnostics without model dependencies."""

# Keep the local control plane compatible with pre-3.11 system Python.
# ruff: noqa: UP017

import hashlib
import json
import logging
import os
import re
import runpy
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import suppress
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

LOGGER = logging.getLogger("probe-download")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "diagnostics/representation-scales/"
LOCAL_ROOT = Path("artifacts/diagnostics/representation-scales")
ARTIFACTS = (
    "probe.log", "report.json", "summary.md", "samples.jsonl", "probes.npz",
    "validation_predictions.npz", "status.json",
)
HASHED_ARTIFACTS = ("samples.jsonl", "probes.npz", "validation_predictions.npz")
MAX_JSON_BYTES = 4 * 1024**2
MAX_STATUS_BYTES = 64 * 1024
PAGE_SIZE = 200
MAX_WORKERS = 4
WORKER_MEMORY_BYTES = 256 * 1024**2
COMMAND_TIMEOUT_SECONDS = 120


def safe_run_id(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value) is None
        or "--" in value
    ):
        raise ValueError("PROBE_RUN_ID must be a model run ID, not a path or option")
    return value


def worker_count(project_root):
    """Reserve 75% of available memory and bound both processes and queued work."""
    resources = runpy.run_path(str(project_root / "src/stock_forecasting/runtime_resources.py"))
    cpus = resources["detect_visible_cpu_count"]()
    try:
        memory = resources["detect_available_memory"]()
        available, source = memory.available_bytes, memory.source
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        # macOS does not expose SC_AVPHYS_PAGES; use reclaimable VM page counts.
        result = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5
        )
        page_size = re.search(r"page size of (\d+) bytes", result.stdout)
        counts = re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)\.", result.stdout)
        if page_size is None or not counts:
            raise RuntimeError("Unable to determine available macOS memory") from None
        available = int(page_size.group(1)) * sum(map(int, counts))
        source = "macos_reclaimable_pages"
    requested = int(os.environ.get("RUNPOD_PROBE_DOWNLOAD_WORKERS", MAX_WORKERS))
    if not 1 <= requested <= MAX_WORKERS:
        raise ValueError("RUNPOD_PROBE_DOWNLOAD_WORKERS must be between 1 and 4")
    memory_limit = available // 4 // WORKER_MEMORY_BYTES
    if memory_limit < 1:
        raise MemoryError("Insufficient memory headroom for one bounded S3 download worker")
    count = min(requested, MAX_WORKERS, cpus, memory_limit)
    LOGGER.info(
        "Download I/O: workers=%d max_pending=%d available_memory=%d (%s) "
        "estimated_memory_per_worker=%d",
        count, count, available, source, WORKER_MEMORY_BYTES,
    )
    if count == 1:
        LOGGER.warning(
            "Using one I/O worker: requested=%d CPU_limit=%d memory_limit=%d",
            requested, cpus, memory_limit,
        )
    return count


def parallel_results(executor, function, items, workers):
    """Keep at most one batch of futures and propagate every worker failure."""
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, workers))
        if not batch:
            return
        futures = [executor.submit(function, item) for item in batch]
        try:
            for future in futures:
                yield future.result()
        finally:
            for future in futures:
                future.cancel()
            # Finish active writers before their temporary directory can be cleaned up.
            wait(futures)


class S3Reader:
    """Use the existing credential-isolated wrapper for read-only S3 requests."""

    def __init__(self, project_root, volume_id):
        if re.fullmatch(r"[A-Za-z0-9_-]+", volume_id or "") is None:
            raise ValueError("The project .env has no valid network volume ID")
        self.wrapper = project_root / "scripts/runpod_s3_project.sh"
        self.volume_id = volume_id

    def command(self, arguments, output):
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                ["bash", str(self.wrapper), *arguments], stdout=output, stderr=errors,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=COMMAND_TIMEOUT_SECONDS)
            except BaseException:
                # Terminate the wrapper and its child if it has not reached exec yet.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            if code:
                # Do not echo transport output that could contain credential values.
                raise RuntimeError(f"RunPod S3 read failed (exit {code}); retry the command")

    def json_command(self, arguments, limit=MAX_JSON_BYTES):
        with tempfile.TemporaryFile() as output:
            self.command(arguments, output)
            output.seek(0)
            raw = output.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Remote diagnostic metadata exceeds the bounded read limit")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Remote diagnostic metadata must be a JSON object")
        return value, hashlib.sha256(raw).hexdigest()

    def read_json(self, key, limit=MAX_JSON_BYTES):
        return self.json_command(
            ["s3", "cp", f"s3://{self.volume_id}/{key}", "-", "--only-show-errors"],
            limit,
        )

    def status_keys(self, prefix):
        token = None
        while True:
            arguments = [
                "s3api", "list-objects-v2", "--bucket", self.volume_id,
                "--prefix", prefix, "--max-keys", str(PAGE_SIZE), "--no-paginate",
                "--output", "json",
            ]
            if token:
                arguments.extend(("--continuation-token", token))
            page, _ = self.json_command(arguments)
            entries = page.get("Contents", [])
            if not isinstance(entries, list) or len(entries) > PAGE_SIZE:
                raise ValueError("Invalid or oversized diagnostic listing page")
            for entry in entries:
                key = entry.get("Key") if isinstance(entry, dict) else None
                if not isinstance(key, str) or not key.startswith(prefix):
                    raise ValueError("Diagnostic listing escaped the requested prefix")
                if key.endswith("/status.json"):
                    yield key
            if page.get("IsTruncated") is False:
                return
            next_token = page.get("NextContinuationToken")
            if (
                page.get("IsTruncated") is not True or not isinstance(next_token, str)
                or not next_token or next_token == token
            ):
                raise ValueError("Invalid diagnostic listing continuation")
            token = next_token

    def download(self, key, destination):
        with destination.open("xb") as output:
            self.command(
                ["s3", "cp", f"s3://{self.volume_id}/{key}", "-", "--only-show-errors"],
                output,
            )


def candidate(reader, key):
    parts = key.split("/")
    if (
        len(parts) != 6 or "/".join(parts[:2]) + "/" != REMOTE_ROOT
        or re.fullmatch(r"checkpoint-[0-9]{6,}", parts[3]) is None
        or re.fullmatch(r"probe-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", parts[4]) is None
        or parts[5] != "status.json"
    ):
        raise ValueError("Invalid diagnostic result identity: " + key)
    run_id = safe_run_id(parts[2])
    expected_checkpoint = f"/runpod-volume/savedModel/{run_id}/{parts[3]}"
    status, status_hash = reader.read_json(key, MAX_STATUS_BYTES)
    state = status.get("state")
    if state in {"running", "failed"}:
        return None
    if state != "complete" or status.get("checkpoint") != expected_checkpoint:
        raise ValueError("Invalid completed diagnostic status: " + key)
    prefix = key[:-len("status.json")]
    report, report_hash = reader.read_json(prefix + "report.json")
    if (
        report.get("schema_version") != "1.0"
        or report.get("kind") != "historical-scale-representation-probe"
        or not isinstance(report.get("checkpoint"), dict)
        or report["checkpoint"].get("path") != expected_checkpoint
    ):
        raise ValueError("Diagnostic report identity does not match its completed status: " + key)
    stamp = report.get("created_at")
    if not isinstance(stamp, str):
        raise ValueError("Diagnostic report completion timestamp is missing: " + key)
    completed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if completed.utcoffset() is None:
        raise ValueError("Diagnostic report timestamp must include a timezone: " + key)
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(HASHED_ARTIFACTS):
        raise ValueError("Diagnostic report has an invalid artifact manifest: " + key)
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in hashes.values()):
        raise ValueError("Diagnostic report has an invalid SHA-256: " + key)
    return {
        "order": (completed.astimezone(timezone.utc), run_id, parts[3], parts[4]),
        "run_id": run_id, "relative": Path(*parts[2:5]), "prefix": prefix,
        "hashes": dict(hashes, **{"report.json": report_hash, "status.json": status_hash}),
    }


def select_result(reader, run_id, executor, workers):
    prefix = REMOTE_ROOT + (safe_run_id(run_id) + "/" if run_id is not None else "")
    newest = None
    for result in parallel_results(
        executor, lambda key: candidate(reader, key), reader.status_keys(prefix), workers
    ):
        if result is not None and (newest is None or result["order"] > newest["order"]):
            newest = result
    if newest is None:
        raise FileNotFoundError(
            "No completed scale diagnostic found" + (" for " + run_id if run_id else "")
        )
    return newest


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_local_path(path):
    if path.resolve() != path or (path.exists() and not path.is_dir()):
        raise ValueError("Diagnostic download directory is not a canonical directory: " + str(path))


def download_result(reader, result, project_root, executor, workers):
    destination = project_root / LOCAL_ROOT / result["relative"]
    safe_local_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only validated files. Temporary state is private to this invocation.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        staging = Path(temporary)

        def transfer(name):
            target = staging / name
            reader.download(result["prefix"] + name, target)
            if target.stat().st_size == 0:
                raise ValueError("Downloaded diagnostic artifact is empty: " + name)
            expected = result["hashes"].get(name)
            if expected is not None and sha256_file(target) != expected:
                raise ValueError("Downloaded diagnostic SHA-256 mismatch: " + name)

        for _ in parallel_results(executor, transfer, ARTIFACTS, workers):
            pass
        safe_local_path(destination)
        destination.mkdir(exist_ok=True)
        for name in ARTIFACTS:
            target = destination / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("Unsafe existing diagnostic artifact: " + str(target))
        # State is published last; unlisted local files are never removed.
        for name in ARTIFACTS:
            (staging / name).replace(destination / name)
    return destination


def main():
    if len(sys.argv) > 2:
        raise ValueError("Expected only an optional PROBE_RUN_ID")
    run_id = safe_run_id(sys.argv[1]) if len(sys.argv) == 2 else None
    workers = worker_count(PROJECT_ROOT)
    reader = S3Reader(PROJECT_ROOT, os.environ.get("RUNPOD_NETWORK_VOLUME_ID"))
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="probe-download")
    try:
        result = select_result(reader, run_id, executor, workers)
        LOGGER.info(
            "Selected completed diagnostic: %s (%s)", result["relative"], result["order"][0]
        )
        destination = download_result(reader, result, PROJECT_ROOT, executor, workers)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    print(f"Downloaded scale diagnostic: run_id={result['run_id']} state=complete")
    print("Result directory: " + str(destination))
    print("Summary: " + str(destination / "summary.md"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except (OSError, ValueError, RuntimeError, MemoryError, subprocess.SubprocessError) as error:
        LOGGER.error("Diagnostic download failed: %s", error)
        sys.exit(1)
