# ple-harness

An agent harness for **Physical-Engineering's Last Exam** (PLE), a
benchmark of 85 engineering tasks (29 public dev tasks, 56 hidden test
tasks) across structural, civil, electrical, and mechanical
engineering. Each task runs in an isolated container and asks for a
project deliverable — a calculation package, design review, drawing,
or technical memo — graded from 0 to 1.

The harness is a WebSocket server. For each task the evaluator opens
one connection, sends one `task` frame (instruction, workdir, limits,
MCP servers), and the harness drives a model through the work using
four request types — `exec`, `read_file`, `write_file`, `http` —
answered by the evaluator inside the task container. The harness
reports token usage after every model call and ends the task with a
single `done` frame whose final message goes to the graders.

Declared model: `google-cloud/claude-opus-5-5` (see
`submission.yaml`). In evaluation, model calls go through the
organizers' proxy, which serves only the declared model.

## Layout

```
submission.yaml   name, model_api, install and serve commands
serve.py          WebSocket server: bind, frame I/O, session handling,
                  watchdog and cancel handling; also hosts the
                  validating serialiser — the only outbound frame path
harness.py        the agent loop (Brief -> Execute -> Finalize);
                  socket-free: run_harness(task, emit, workdir, config)
model_client.py   OpenAI-compatible model client (streaming, retries,
                  usage ledger)
tools.py          tool schemas and dispatch for the model-facing tools
requirements.txt  pinned dependencies for the install step
LICENSE           Apache-2.0
THIRD-PARTY.md    third-party dependency notices (name, version,
                  licence), kept current as dependencies change
FIX-NOTES.md      notes on an integration fix made during development
stub_proxy/       local stub model server for development (below)
tests/            smoke tests (below)
check_secrets.sh  secret scan over tracked files and git history
run_kit_check.sh  runs the kit check against this entry via the stub,
                  plus the advisory secret scan
```

There is no `artifacts/` directory yet: it is generated later by
the kit's eval runner (see Notes), not present in this tree.

The install step in `submission.yaml` creates a venv at
`/opt/harness/venv` (`uv venv --python 3.12`) and installs
`requirements.txt` into it; `serve` runs `serve.py` with that
interpreter. Only `/opt/harness` carries into the serve container.

## Running the checks locally

Development uses a stub model server, so no real model access (and no
cost) is involved. It speaks the same OpenAI-compatible
chat-completions API as the kit's model proxy and returns canned
replies with usage counts; it can also simulate a slow streamed reply
and a stream that drops mid-reply. See the header of
`stub_proxy/server.py` for its modes.

```bash
# one-time: create the development venv
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python websockets pytest

# the kit check, with the model stages served by the stub proxy
./run_kit_check.sh                 # add --cancel-test for the cancel stage

# the smoke tests
.venv/bin/python -m pytest tests/test_integration.py -v
```

`run_kit_check.sh` starts the stub on port 4100, exports
`MODEL_PROXY_URL` / `MODEL_PROXY_KEY` for it, runs
`uv run python -m kit check` from the kit clone
(`../stage0/ple-repo`), and prints a per-stage PASS/FAIL summary; the
full output is kept in `kit-check-last.log`. It also runs
`check_secrets.sh` — a scan of the tracked files and the git history
for credential-shaped strings — as an advisory stage: the result is
printed as PASS/FAIL but does not change the kit verdict. The build, listen, and
smoke stages run containers, so they need Docker; without it the run
stops at the build stage and reports the stages before it.

The check prints `ready to submit` only when every stage passes and
the entry sits at the root of its own git repository with everything
committed — a green run in this working folder is not by itself a
readiness verdict.

To check against real models instead of the stub, run the kit's own
proxy (`kit/model_proxy/start.sh` in the kit clone, which needs a
Google Cloud project with Vertex AI) and point `MODEL_PROXY_URL` /
`MODEL_PROXY_KEY` at it, or pass the same variables through
`--env-file` to `python -m kit check`.

## Notes

- `artifacts/` does not exist in this tree yet. It is generated
  later, only by `uv run python eval/run.py` from the kit, against
  the declared model. It is never edited by hand, and the harness
  never reads it at build or run time.
- No keys belong in this repository or in `submission.yaml`; model
  credentials reach the harness through the environment the evaluator
  sets (`MODEL_BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME`).
