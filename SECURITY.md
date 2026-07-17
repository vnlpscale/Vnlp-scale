# Security policy

## Supported versions

Only the latest released minor version receives security fixes during the alpha period.

## Threat model

Model repositories and local checkpoints are treated as untrusted input.

Vnlp-scale:

- accepts safetensors weights and JSON metadata;
- does not load pickle checkpoints;
- rejects local shard paths that escape the checkpoint directory;
- validates store structure and byte ranges;
- can verify SHA-256 for every stored payload;
- uses an exclusive writer lock and atomic manifest replacement.

The project does not sandbox tokenizer plugins, custom Python model code, GPU drivers, or third-party dependencies. `trust_remote_code` is not enabled by the CLI.

## Reporting a vulnerability

Do not open a public issue for an exploitable vulnerability. Use GitHub private vulnerability reporting for the repository. Include the affected version, reproduction steps, impact, and any proposed remediation.

## Operational guidance

- Pin a Hugging Face revision or commit for reproducible recording.
- Run `vnlp-scale verify` after copying a store between machines.
- Keep stores read-only for inference users.
- Do not use `--force-unlock` unless no writer process is active.
- Treat tokenizer files as executable-adjacent input and source them from trusted repositories.
