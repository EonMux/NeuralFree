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
- **Serialized state.** The same applies to model and checkpoint files read by `ModelSerializer` and `ModelCheckpoint`.
- **Shared resources.** Ways for another local user to read or alter the files and caches NeuralFree creates, such as the `SharedMemoryBackend` segments or the kernel caches under `~/.cache/neuralfree`.

Out of scope:
- **The documented properties of `TCPBackend`.** Its protocol is neither authenticated nor encrypted, by design, and is meant for trusted private networks only ([TRAINING.md](TRAINING.md#distributed-training)). Flaws beyond that, such as a crash of a rank caused by malformed traffic, are in scope.
- **Vulnerabilities in dependencies** (JCuda, JOCL, LWJGL, Log4j and others) or in GPU drivers, unless NeuralFree uses them in a way that makes them exploitable. Report those to their maintainers.
- **The limitations listed in [GAPS.md](GAPS.md)**, which are known and documented.
