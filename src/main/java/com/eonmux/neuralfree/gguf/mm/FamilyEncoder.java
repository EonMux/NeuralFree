package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;

/**
 * One bound audio tower executor the CPU audio encoder dispatches to: every audio family but the whisper and
 * qwen3a graphs, whose executors are built into {@link CpuAudioEncoder}. One call encodes one mel chunk (or one
 * raw-waveform chunk) to text-model embedding rows, one row per output token. Package-private executor support.
 */
interface FamilyEncoder {

    /** Embedding rows of {@code chunk} ({@code [nMel][nLen]}, mel-major): {@code [nTokens][outputWidth]}. */
    float[][] encode(MelChunk chunk);
}
