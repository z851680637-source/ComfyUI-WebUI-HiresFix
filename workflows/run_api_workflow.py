"""Run a ComfyUI API-format workflow (Save (API Format)) and wait for the result.

Usage:
    $env:PYTHONNOUSERSITE='1'
    python workflows/run_api_workflow.py --server http://127.0.0.1:8188 --workflow workflows/webui_hires_fix_api.json

The API-format JSON is a mapping {node_id: {class_type, inputs}}; this script wraps it
in {"prompt": ...} as POST /prompt requires.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def _get(server: str, path: str):
    with urllib.request.urlopen(server + path, timeout=60) as r:
        return json.loads(r.read().decode())


def _post(server: str, path: str, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--client-id", default="webui-hires-fix-runner")
    args = ap.parse_args()

    server = args.server.rstrip("/")
    with open(args.workflow, "r", encoding="utf-8") as f:
        prompt = json.load(f)
    if "prompt" in prompt and isinstance(prompt["prompt"], dict):
        prompt = prompt["prompt"]

    result = _post(server, "/prompt", {"prompt": prompt, "client_id": args.client_id})
    if result.get("node_errors"):
        print("Node errors:", json.dumps(result["node_errors"], indent=2))
        return 2
    prompt_id = result["prompt_id"]
    print("prompt_id:", prompt_id)

    start = time.time()
    while time.time() - start < args.timeout:
        history = _get(server, "/history/" + prompt_id)
        item = history.get(prompt_id)
        if item is not None:
            status = item.get("status", {})
            print("status:", status.get("status_str"))
            outputs = item.get("outputs", {})
            for node_id, out in outputs.items():
                for image in out.get("images", []):
                    print(
                        "image:",
                        image.get("filename"),
                        "subfolder=" + str(image.get("subfolder", "")),
                        "type=" + str(image.get("type")),
                        image.get("metadata", {}).get("width"),
                        "x",
                        image.get("metadata", {}).get("height"),
                    )
            for message in status.get("messages", []):
                if isinstance(message, list) and len(message) == 2 and message[0] in ("execution_error", "execution_interrupted"):
                    print("message:", json.dumps(message))
            return 0 if status.get("status_str") == "success" else 3
        time.sleep(1.5)
    print("timeout waiting for", prompt_id)
    return 4


if __name__ == "__main__":
    sys.exit(main())
