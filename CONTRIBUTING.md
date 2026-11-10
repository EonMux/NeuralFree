# Contributing to NeuralFree

Thank you for helping improve NeuralFree. This page covers the build, the code style, the tests, the pull request process and the licence terms that apply to contributions. For a security vulnerability, do not open an issue or a pull request: follow [SECURITY.md](SECURITY.md).

## Before you start

- **Bugs.** Open an issue with the NeuralFree commit, the JDK version, the operating system, the GPU and driver, the command or code you ran, and the full output. For the GGUF engine, also give the model file's name and quantization, and the `runLLMDemo` flags.
- **Larger changes.** Open an issue describing the change before you write it, so that the approach can be agreed first. This applies in particular to new public API, new dependencies and changes to the kernel libraries.
- **Known limitations.** The guides ([USAGE.md](USAGE.md), [LLM-INFERENCE.md](LLM-INFERENCE.md)) document what the code refuses or does differently from llama.cpp and PyTorch; check them before reporting one of those as a bug.

## Build

You need JDK 11 or newer; the Gradle wrapper downloads Gradle itself. The GPU toolchains (`nvcc`, `glslc` with `spirv-dis`, `clang`) are optional, and the build warns about each one that is missing. [QUICK-START.md](QUICK-START.md) describes the setup, and [USAGE.md](USAGE.md#build) every task and build property.

```bash
./gradlew assemble          # compile and build the GPU kernels the installed tools allow
./gradlew test              # unit tests
./gradlew build             # both
```

The sources compile with `--release 11`, so they must not use language features or APIs newer than Java 11.

## Code style

Follow the style of the surrounding code:
- **Formatting.** Four-space indentation, braces on every block, and lines of about 120 characters at most. Files are UTF-8 with LF line endings; `.gitattributes` enforces LF except for `gradlew.bat`.
- **Documentation.** Public classes and methods have Javadoc that states what the code does now: its contract, units, ownership of device memory, threading and the exceptions it throws. Comments explain why, not what. Do not add `@author` tags, change logs or history to comments or docs.
- **Ported code.** Code that follows llama.cpp, ggml or another project cites the upstream file and function it follows. Code copied or translated from another project must be under a licence compatible with the NeuralFree License, keep its copyright notice in a file header, and be added to [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) with its licence text.
- **Locale-independent text.** `String.format`, `toLowerCase` and `toUpperCase` take `Locale.ROOT` wherever the text is parsed, compared or written to a file.
- **Logging.** Log through Log4j 2 (`LoggingUtil.getLogger(...)`); only the demos print to standard output.
- **Device memory.** The garbage collector does not free GPU memory. Code that creates tensors, buffers or backends closes them, or documents who does.
- **Errors.** Validate arguments at public entry points and fail with a message that names the bad value. Do not swallow exceptions; an operation the code cannot perform throws instead of returning a wrong result.

### GPU kernels

The kernels live in two libraries, `llm` (the GGUF engine) and `train` (the training framework), under `src/main/resources` ([USAGE.md](USAGE.md#kernel-libraries)):
- each OpenCL C kernel in `kernels/<lib>` also compiles as CUDA through `kernels/common/dialect.h`, and has a GLSL twin in `shaders/<lib>` with its rows in `shaders/<lib>/variants/*.tsv`;
- a new kernel source file is listed in `kernels/<lib>/manifest.txt`;
- after adding, removing or renaming a kernel, or changing its arguments, run `./gradlew updateKernelInventory` and commit the regenerated `kernels/<lib>/inventory.txt`;
- `train` kernels must keep the numerical contract of `tensor/gpu/package-info.java`: the CPU reference's formula, ascending-order loops and fixed-order reductions.

Check kernel changes with `./gradlew inventoryCheck verifyGpuKernels`. Adding `-Pneuralfree.cuda.archs=75 -Pneuralfree.cuda.ptxOnly=true` makes the CUDA part a quick PTX-only compile check.

### Generated HF tables

`src/main/resources/hf` holds the data half of llama.cpp's HF-to-GGUF converter as checked-in resources: the architecture rows (`archs.tsv`, `dynamic_archs.tsv`), the tensor-name patterns, the GGUF writer keys, the hyperparameter aliases, the pre-tokenizer hashes, the tensor filters, and byte copies of the converter's bundled chat templates and built-in vocabularies. They are generated, not edited by hand:

```bash
./gradlew updateHfTables -PllamaCpp=<dir>   # <dir> is a llama.cpp checkout
```

- `updateHfTables` runs `gradle/hfgen/hfgen.py` on the checkout with an isolated Python (`python3 -I`, standard library only; llama.cpp is never imported), parses the converter sources, stages the tables, validates them, and replaces `src/main/resources/hf` with the result (the previous directory is moved aside and restored on failure). It needs git (to record the checkout's commit) and Python 3.9+, and warns when the checkout has local changes under the paths the tables are generated from. `-Pneuralfree.python` and `-Pneuralfree.git` point at the interpreters.
- Do not edit the generated files. Every one of them starts with a `# ... do not edit` header naming the source commit and the row count. After a llama.cpp update, regenerate and commit the whole directory (including `manifest.tsv`, generated last).

`verifyHfTables` (Groovy, in `buildSrc`; `processResources` depends on it) checks the checked-in directory on every build without Python:
- every table's header line — file name, source commit, recorded row count — and every row's column count and values against the table's declared format;
- the file set itself (missing, unexpected or empty);
- the byte copies (`templates/*.jinja`, `vocab/*.gguf`), and every other file's size and sha256, against `src/main/resources/hf/manifest.tsv`.

A build fails here when a stale or hand-edited table is checked in; regenerate with `updateHfTables` instead of editing the table.

## Tests

- Tests use JUnit 4 and live under `src/test/java`, in the package of the class they test.
- A bug fix comes with a test that fails without the fix. New behaviour comes with tests of its contract, including the error cases.
- A test that needs a GPU API skips itself through a JUnit assumption when the API is unavailable, for example with `GpuTestUtil.assumeOpenCLAvailable()`, `assumeCudaAvailable()` or `assumeVulkanAvailable()`. It must never pass silently without running.
- A device op is tested against its CPU reference within the tolerance class of its op ([USAGE.md](USAGE.md#the-training-framework-on-a-gpu-backend)).
- Tests do not depend on model files, network access or wall-clock timing, and close the device resources they create.

Run one class with `./gradlew test --tests 'com.eonmux.neuralfree.tensor.ShapeTest'` (any fully qualified class name or pattern).

### Microbenchmarks

Host hot paths have [JMH](https://github.com/openjdk/jmh) microbenchmarks under `src/jmh/java`, each over a synthetic, fixed-seed fixture so nothing is downloaded: BPE encode of a 4 KB text over learned merges (`BpeEncodeBenchmark`), the JSON-object grammar mask over a 151,936-token vocabulary (`GrammarMaskBenchmark`), the sampler chains over a whole logits row (`SamplerChainBenchmark`) and over a device top-k cone (`SamplerConeBenchmark`), and the remote-compute dispatch frame (`RpcDispatchFramingBenchmark`). `GrammarMaskBenchmark` runs over a model's own vocabulary when its `vocabFile` parameter names a GGUF file. `./gradlew jmh` runs them; `-Pjmh.args` passes JMH's own options and a benchmark filter:

```bash
./gradlew jmh -Pjmh.args='-f 1 -wi 3 -i 5 GrammarMask'
./gradlew jmh -Pjmh.args='-f 1 -wi 3 -i 5 -p vocabFile=/models/Qwen3-0.6B-Q8_0.gguf GrammarMask'
./gradlew jmh -Pjmh.args='-p chain=default,greedy SamplerChain'
```

JMH is a benchmark-only dependency: neither it nor the benchmarks are in the jar or the distribution. Compare runs on a quiet machine, alternating the builds.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one change. Separate refactoring from behaviour changes.
3. Update the documentation the change affects: the Javadoc, and the root guides (`README.md`, `QUICK-START.md`, `USAGE.md`, `TRAINING.md`, `LLM-INFERENCE.md`) when a flag, property, API or limitation changes.
4. Make sure `./gradlew build` passes. For kernel changes, also run `./gradlew inventoryCheck verifyGpuKernels` with the toolchains you have, and say in the pull request which backends you tested on.
5. Write commit messages as `<type>: <summary>`, with `type` one of `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore` or `ci`, and a body that explains why when the summary does not.
6. Describe in the pull request what changed, why, and how you tested it.

A maintainer reviews every pull request. Changes may be requested before it is merged.

## Licensing of contributions

NeuralFree is licensed under the [NeuralFree License](LICENSE): the Apache License, Version 2.0, with an Additional Commercial Use Condition. EONMUX LTD also offers NeuralFree under commercial licences ([COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).

By submitting a contribution (a pull request, patch, or other material intended for inclusion in NeuralFree), you agree that:
1. **Project licence.** Your contribution is licensed under the NeuralFree License, as Section 5 of the Apache License provides, including the patent licence of its Section 3.
2. **Commercial licensing.** You grant EONMUX LTD a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable licence to use, reproduce, modify, prepare derivative works of, publicly display, publicly perform, sublicense and distribute your contribution and derivative works of it, including under the commercial licences of NeuralFree that EONMUX LTD grants and under future versions of the NeuralFree License.
3. **Your rights.** You have the right to submit the contribution under these terms: it is your original work, or you have the permission of its copyright holder (for example your employer). Material from third parties is identified as such, with its licence.
4. **Ownership.** You keep the copyright in your contribution; these grants do not transfer it.

The maintainers may require a signed Contributor License Agreement (CLA) that records these terms before a contribution is merged. When one is required, the maintainers ask for it in the pull request.

## Code of conduct

Be respectful and constructive in issues, pull requests and reviews. Harassment and personal attacks are not tolerated, and the maintainers may remove such content and block those who post it.
