# Answer-from-context (RAG) action

## what it is

The "Answer from context (RAG)" action of the Training tab's train flow. Lets a user pick a base checkpoint, build a context-anchored question/answer corpus, and continue-train (SFT) that base on it so the model answers from supplied context. Markup is static in `index.html` (`#ragTrainPanel`); logic is in `veritate_mri/web/rag_train.js` + `rag_train.css`. Backend: `veritate_mri/routes/rag_routes.py`.

## how it works

- The action picker modal (`#trainFlowModal`) lists RAG as a card with `data-flow="rag"`. Selecting it calls `flowPick("rag")` (`index.js`), which shows `#ragTrainPanel` via `_trShowRagPanel(true)` and hides the trainer picker, plugin form, run row, and the synth/export panels.
- The panel markup (`#ragTrainPanel`) lives inline in the train-a-model panel body in `index.html`, next to `#synthPanel`. It holds base-model select, new-name field, facts/steps inputs, a single Train button, a status line, and a log tail. There is no separate Build button — training auto-builds the corpus when missing.
- Button logic runs in `rag_train.js`: `wire()` attaches the Train handler once and starts the status poll. Base-model list comes from `GET /pytorch-models` via `load_models()`.
- Train: `train()` posts `{source, name, steps, n_facts}` to `POST /rag/train`. Tolerant start: the frontend auto-picks the first base model if none selected and defaults the name to `<source>_rag`; the backend defaults steps/facts and, if the `rag_ui` corpus is missing, prepends a build step (build then train run as one chained job via `_run_steps`). Response `auto_built` flags the chained case.
- The corpus build (`build_grounded.py`) generates its facts/Q&A through the configured Teacher Model via `veritate_core.plugin.get_teacher_client()` — not a hardcoded model. RAG is therefore a teacher-required flow (`TEACHER_REQUIRED_FLOWS`); with no teacher set, the `#teacherGate` banner shows and the Train button is disabled. The builder exits with a clear message if no teacher is configured.
- Status: `poll()` hits `GET /rag/status` every 1500 ms while a job runs and renders job state + the log tail (`.rag_run.log`, last 8 KB). Both buttons disable while a job runs.
- Stop: `#ragStop` shows while a job runs (`set_buttons`); `stop_job()` (`rag_train.js`) confirms via the shared `window.confirmDialog`, then posts `POST /rag/stop`, which terminates the subprocess and sets status `stopped`.
- Job model: one job at a time (module-level lock in `rag_routes.py`). Build and train share the same log file and status. Mirrors `veritate_mri/training/trainer_runner.py`.

## dependencies

- `#ragTrainPanel` markup in `index.html` and `_trShowRagPanel` in `index.js` (visibility toggle per flow).
- `GET /pytorch-models` for the base-model list.
- `experiments/v2/rag/build_grounded.py` (accepts `--n_facts`, `--stem`).
- `experiments/v2/rag/sft_grounded.py` (accepts `--source`, `--name`, `--corpus`, `--steps`).
- CSS variables `--line`, `--text`, `--soft`, `--accent`, `--warm` from `index.css`.

## pitfalls

- Only one job runs at a time; starting a second while one is in flight returns HTTP 409.
- Status reads a tail of `.rag_run.log`. Both build and train overwrite that file on start, so the log reflects only the current job.
- The corpus stem is fixed to `rag_ui`. Training reads that stem, so build must complete before train can find the corpus bins.
- No PID-file recovery: a server restart mid-job drops the in-memory status (the subprocess keeps running, but the panel reports idle).
- `rag_train.js` wires and polls on load regardless of which flow is active; the panel is hidden until the RAG action is picked.
