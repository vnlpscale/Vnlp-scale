# Contributing

## Development setup

```bash
git clone https://github.com/vnlscale/Vnlp-scale.git
cd Vnlp-scale
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Change requirements

- Add tests for behavioral changes and regression fixes.
- Preserve bounded-memory behavior; do not replace chunk reads with full-tensor loads in runtime paths.
- Reject malformed data explicitly rather than silently repairing it.
- Document benchmark hardware, software versions, model revision, and exact command.
- Do not report throughput estimates as measurements.
- Keep optional backends importable only when their dependencies are installed.

## Pull requests

A pull request should contain one coherent change, a rationale, tests, and documentation where public behavior changes. Store-format changes require a version decision and a migration plan.

## Reporting results

Compression results must include bits per parameter, reconstruction metric, downstream evaluation, model identifier, model revision, and preset/configuration. Runtime results must include cold and warm runs and distinguish storage, RAM-cache, and GPU-cache bandwidth.
