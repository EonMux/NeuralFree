# Third-party notices

NeuralFree is Copyright (c) 2017-2026 EONMUX LTD and Amir Farhad Eslampanah and is licensed under the
[NeuralFree License](LICENSE). Parts of it are derived from, or are copies of, the third-party works listed below.
Those parts remain under their original licences, whose terms and copyright notices are reproduced here, and the
Additional Commercial Use Condition of the NeuralFree License does not restrict any right those licences grant
(LICENSE, paragraph A.5).

Paths are relative to the repository root. Source files that contain derived code also cite the upstream file and
lines they follow.

## llama.cpp / ggml

- **Upstream:** https://github.com/ggml-org/llama.cpp
- **Licence:** MIT License
- **Copyright:** Copyright (c) 2023-2026 The ggml authors

Derived material:

- **Quantization tables, copied value for value** from `ggml/src/ggml-common.h` (`ksigns_iq2xs`, `iq2xxs_grid`,
  `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid`, `kvalues_iq4nl`, `kvalues_mxfp4`) and the
  packed IQ1_S grid of `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`:
  - `src/main/java/com/eonmux/neuralfree/gguf/quant/QuantTables.java`
  - `src/main/resources/kernels/llm/common/iq_tables.h`
  - `src/main/resources/shaders/llm/common/iq_tables.glsl`
  - `src/main/resources/kernels/llm/common/quant_blocks.h`
  - `src/main/resources/shaders/llm/common/quant_blocks.glsl`
  - `src/main/resources/shaders/llm/common/kv_types.glsl`
- **Scalar helpers translated statement for statement** from `ggml-quants.c` (`get_scale_min_k4`,
  `best_index_int8`, the `quantize_row_*_ref` quantizers), `ggml-impl.h` (`ggml_e8m0_to_fp32_half`,
  `ggml_compute_fp32_to_bf16`) and `ggml-cpu/ops.cpp` (`rope_yarn_ramp`, `rope_yarn`):
  - `src/main/resources/kernels/llm/common/quant_blocks.h`
  - `src/main/resources/shaders/llm/common/quant_blocks.glsl`
  - `src/main/resources/kernels/llm/common/kv_types.h`
  - `src/main/resources/shaders/llm/common/kv_quant.glsl`
  - `src/main/resources/kernels/llm/common/rope_common.h`
  - `src/main/resources/shaders/llm/common/rope_common.glsl`
  - `src/main/java/com/eonmux/neuralfree/gguf/ops/rope/RopeMath.java`
- **The GGUF inference engine**, which ports llama.cpp code and data: the quantized block decoders, the tokenizers
  (pre-tokenizer regular expressions, Unicode splitting, BPE, SentencePiece, WordPiece, Unigram, RWKV and PLaMo-2
  sessions), chat message handling and the ChatML template of `common/chat.cpp`, the Jinja engine's behaviour of
  `common/jinja`, the GGUF key and tensor names of `src/llama-arch.cpp`, the architecture tables of
  `src/llama-model.cpp`, and the model graphs and operators of `src/models` and the ggml CPU, CUDA and Vulkan
  backends:
  - `src/main/java/com/eonmux/neuralfree/gguf/`
  - `src/main/resources/kernels/llm/`
  - `src/main/resources/shaders/llm/`
- **The Vulkan driver priority table** of `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`driver_priorities`):
  - `src/main/java/com/eonmux/neuralfree/acceleration/vulkan/VkDeviceRanking.java`
- **Reference transcriptions in the tests** (for example `quantize_row_q8_0_ref`, `quantize_row_tq1_0_ref`,
  `ggml_ssm_scan` and the gated delta net):
  - `src/test/java/com/eonmux/neuralfree/gguf/`

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## YaRN reference implementation

- **Upstream:** https://github.com/jquesnelle/yarn (`LlamaYaRNScaledRotaryEmbedding.py`)
- **Licence:** MIT License
- **Copyright:** Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng

ggml's `rope_yarn` is based on this implementation and carries its notice; NeuralFree's port of `rope_yarn` is in:

- `src/main/java/com/eonmux/neuralfree/gguf/ops/rope/RopeMath.java`
- `src/main/resources/kernels/llm/common/rope_common.h`
- `src/main/resources/shaders/llm/common/rope_common.glsl`

```text
MIT License

Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## JCuda samples

- **Upstream:** https://github.com/jcuda/jcuda-samples (`JCudaDeviceQueryV8.java`)
- **Licence:** MIT License
- **Copyright:** Copyright (c) 2008-2016 Marco Hutter - http://www.jcuda.org

The CUDA device attribute names, descriptions and list, and the zero-terminated string decoding, are derived from
the JCudaDeviceQuery sample:

- `src/main/java/com/eonmux/neuralfree/acceleration/cuda/CUDA_Utils.java`

```text
The MIT License (MIT)

Copyright (c) 2008-2016 Marco Hutter - http://www.jcuda.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## JOCL samples

- **Upstream:** https://github.com/gpu/JOCLSamples (`JOCLDeviceQuery.java`, Copyright 2009-2019 Marco Hutter)
- **Licence:** MIT License
- **Copyright:** Copyright (c) 2019 Marco Hutter

The OpenCL device and platform info helpers (`getInt`, `getInts`, `getLong`, `getLongs`, `getString`, `getSize`,
`getSizes`) are derived from the JOCLDeviceQuery sample:

- `src/main/java/com/eonmux/neuralfree/acceleration/ocl/CLDevice.java`

```text
MIT License

Copyright (c) 2019 Marco Hutter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Jinja2

- **Upstream:** https://github.com/pallets/jinja (version 3.1.6, `src/jinja2/filters.py`)
- **Licence:** BSD 3-Clause License
- **Copyright:** Copyright 2007 Pallets

The `indent` filter is a translation of Jinja2's `do_indent`:

- `src/main/java/com/eonmux/neuralfree/gguf/jinja/JinjaFilters.java`

```text
Copyright 2007 Pallets

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## gpt-oss chat template

- **Upstream:** https://github.com/openai/gpt-oss; the template is distributed with llama.cpp as
  `models/templates/openai-gpt-oss-120b.jinja`
- **Licence:** Apache License, Version 2.0
- **Copyright:** OpenAI's gpt-oss project; the template carries no separate copyright line

The fallback harmony prompt follows the template's structure and reproduces its fixed system-header strings:

- `src/main/java/com/eonmux/neuralfree/gguf/chat/HarmonyFallback.java`

The Apache License, Version 2.0 is reproduced in Part B of [LICENSE](LICENSE). The upstream project has no NOTICE
file.

## Gradle Wrapper

- **Upstream:** https://github.com/gradle/gradle (Gradle 8.14.4)
- **Licence:** Apache License, Version 2.0
- **Copyright:** Copyright 2015 the original author or authors (`gradlew.bat`); Copyright © 2015-2021 the original
  authors (`gradlew`)

Distributed unmodified, with their own licence headers:

- `gradlew`
- `gradlew.bat`
- `gradle/wrapper/gradle-wrapper.jar`

The Apache License, Version 2.0 is reproduced in Part B of [LICENSE](LICENSE). The upstream project has no NOTICE
file.
