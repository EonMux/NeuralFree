# Security Policy

## Supported versions

NeuralFree has no releases yet. Security fixes are made on the `main` branch only.

| Version | Supported |
|---|---|
| `main` | Yes |
| Anything else (forks, older commits) | No |

## Reporting a vulnerability

Report vulnerabilities privately, through GitHub's private vulnerability reporting. Do not open a public issue, pull request or discussion about it.

1. Open the [Security tab of github.com/EonMux/NeuralFree](https://github.com/EonMux/NeuralFree/security) and choose **Report a vulnerability**, or go directly to [the new advisory form](https://github.com/EonMux/NeuralFree/security/advisories/new).
2. Describe:
   - the affected component (for example the GGUF parser, the tokenizer or Jinja engine, a GPU backend, or `TCPBackend`) and the commit you tested;
   - how to reproduce it: the input file or code, the command, and the platform (JDK, operating system, GPU and driver);
   - the impact you observed or expect.
3. Do not include real model weights or private data; a minimal crafted file or a script that generates it is enough.

The maintainers discuss the report with you in the private advisory, and publish the advisory, crediting you unless you prefer otherwise, once a fix is available on `main`. Please keep the details private until then.

If private reporting is unavailable, contact the maintainers through [github.com/EonMux](https://github.com/EonMux) without describing the vulnerability, and ask for a private channel.

## Scope

In scope:
- **Untrusted model files.** GGUF models, LoRA adapters and draft models, including their metadata and embedded chat templates, are untrusted input. Memory corruption, out-of-bounds reads or writes on the device, code execution, file access, or unbounded memory or time consumption caused by a crafted file are vulnerabilities.
- **Hugging Face checkpoint inputs.** The whole checkpoint is untrusted input when an HF model directory, a lone `.safetensors` file, or a PEFT adapter directory is opened:
  - the model directory itself, and every path it can make the loader touch: names read from file contents (index `weight_map` entries, references inside the configs, Mistral `params.json` fields) must be relative and resolve lexically inside the model directory, while directory entries and user-supplied paths may be symlinks and are followed;
  - `config.json` and the other JSON files: `*.safetensors.index.json`, `generation_config.json`, `processor_config.json`, `chat_template.json`, `modules.json`, `quantization_config` sections and the rest;
  - the tokenizer and template files: `tokenizer.json`, `tokenizer.model`, `*.tiktoken`, `tekken.json`, `rwkv_vocab_v*.txt`, `tokenizer.jsonl`, `chat_template.jinja`, `additional_chat_templates/*.jinja` and the Jinja templates they declare;
  - side files: Bitsandbytes `quant_state` JSON, `hadamard_packing.json`, and every other file a config references;
  - PEFT adapter directories (`adapter_config.json`, `adapter_model.safetensors`, their tokenizer and template files), and the base-directory references inside them;
  - the materialization cache directory: it is untrusted as a write target and as an input source. The loader's protection — a per-process directory created afresh, files opened `CREATE_NEW` with an exclusive directory lock, deleted on close, previous runs' stale directories cleaned up — is part of the exploit surface; ways to make the loader consume or overwrite another account's files through it are vulnerabilities.
  - a checkpoint inspection must never execute code: pickle files, repository modeling files and other executable formats are refused by design, and a bypass of that refusal is a vulnerability. The loader's size caps are part of the defence and their bypass is in scope, because their purpose is to keep a crafted checkpoint from running the engine out of memory or time: JSON documents bounded in nesting depth, value count and string size; safetensors headers, tokenizer files and configs bounded in byte size; a bounded `quant_state` JSON; bounded counts of files and tensors per checkpoint; bounded model dimensions (vocabulary, layers, experts, heads, widths); and per-match step budgets on the tokenizer regexes, which throw when exhausted.
- **Serialized state.** The same applies to model and checkpoint files read by `ModelSerializer` and `ModelCheckpoint`, and to safetensors files read and written by the training state dict (`SafetensorsStateDict`).
- **Shared resources.** Ways for another local user to read or alter the files and caches NeuralFree creates, such as the `SharedMemoryBackend` segments, the kernel caches and the HF tensor cache under `~/.cache/neuralfree`, or the per-process materialization sidecar directories.

Out of scope:
- **The documented properties of `TCPBackend`.** Its protocol is neither authenticated nor encrypted, by design, and is meant for trusted private networks only ([TRAINING.md](TRAINING.md#distributed-training)). Flaws beyond that, such as a crash of a rank caused by malformed traffic, are in scope.
- **Vulnerabilities in dependencies** (JCuda, JOCL, LWJGL, Log4j and others) or in GPU drivers, unless NeuralFree uses them in a way that makes them exploitable. Report those to their maintainers.
- **The limitations listed in [GAPS.md](GAPS.md)**, which are known and documented.
