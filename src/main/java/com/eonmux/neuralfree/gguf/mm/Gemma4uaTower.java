package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The Gemma 4 unified audio embedder (llama.cpp tools/mtmd/models/gemma4ua.cpp, projector gemma4ua.cpp): raw
 * 16 kHz waveform frames of 640 samples run through an RMS norm and one input projection into the text hidden
 * size — no encoder blocks and no spectrogram. The mel chunk holds the frames, mel-major (mtmd-audio.cpp
 * {@code mtmd_audio_preprocessor_gemma4ua}). Package-private executor support.
 */
final class Gemma4uaTower implements FamilyEncoder {

    private final ClipAudioModel model;
    private final ClipHparams hp;
    private final CpuMatVec matVec;
    private final CpuTensorRef projW;

    private Gemma4uaTower(ClipAudioModel model, CpuMatVec matVec, CpuTensorRef projW) {
        this.model = model;
        this.hp = model.hparams();
        this.matVec = matVec;
        this.projW = projW;
    }

    /** Shape-checks the projection the graph reads. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        TensorRef w = tc.require("mm.a.input_projection.weight");
        shape(tc, w, hp.audio().nMelBins(), hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static Gemma4uaTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                              Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new Gemma4uaTower(model, matVec, refs.apply("mm.a.input_projection.weight"));
    }

    /** RMS norm over the 640 frame samples, then the projection: one token row per frame. */
    @Override
    public float[][] encode(MelChunk chunk) {
        int frame = hp.audio().nMelBins();
        if (chunk.nMel() != frame) {
            throw new IllegalArgumentException("the gemma4ua embedder needs " + frame + "-sample frames, the chunk"
                    + " holds " + chunk.nMel());
        }
        int n = chunk.nLen();
        float[] rows = new float[n * frame];
        for (int t = 0; t < n; t++) {
            for (int f = 0; f < frame; f++) {
                rows[t * frame + f] = chunk.data()[f * n + t];
            }
        }
        CpuOps.rmsNorm(rows, 0, rows, 0, n, frame, null, hp.eps());
        float[] out = new float[n * model.outputWidth()];
        matVec.matmul(projW, rows, 0, out, 0, n);
        return VisionRows.tokenRows(out, model.outputWidth());
    }
}
