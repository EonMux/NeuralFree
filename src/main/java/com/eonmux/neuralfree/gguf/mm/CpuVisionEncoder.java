package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;

import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;

/**
 * CPU executor of one vision encoder and its projector (the counterpart of the llama.cpp clip graphs,
 * tools/mtmd/models/*.cpp, over {@link ClipVisionModel}'s tensors). One call encodes one preprocessed image tile
 * to text-model embedding rows, one row per output token, ready for {@code ForwardBatch.Builder#addEmbedding}.
 * The encoder and projector graphs are the shared family graphs of {@link VisionFamilies}, run here by
 * {@link CpuVisionOps}: the patch embedding is one matmul over the {@link ClipPatches} matrix (plus the second
 * kernel of the qwen and GLM-4V encoders), the tower is the shared {@link CpuVitTower} (its attention shards its
 * (head, query row) pairs over the CPU worker pool, matmuls run in the tensors' stored types, so quantized mmproj
 * weights run as they do for the text model), the row reorders run in place and the convolutional encoders
 * ({@link Yasa2Graph}, {@link MobileNetV5Graph}) run their convolutions and pools over host maps. Not
 * thread-safe; a closed encoder refuses calls.
 */
public final class CpuVisionEncoder implements VisionEncoder {

    private final ClipVisionModel model;
    private final CpuVisionOps ops;
    private final CpuProjectionDeltas deltas;
    private LoraAttachment loras;
    /** The open one-caller LoRA session ({@link #openLoraSession}), or null. */
    private LoraAttachment sessionLoras;
    private boolean closed;

    private CpuVisionEncoder(ClipVisionModel model, CpuVisionOps ops, CpuProjectionDeltas deltas) {
        this.model = model;
        this.ops = ops;
        this.deltas = deltas;
    }

    /**
     * Binds an executor over {@code model} with the weights of {@code floats} / {@code refs}; matmuls run
     * through {@code matVec} in the tensors' stored types. The encoder owns the LoRA deltas its products are served
     * to ({@link #projectionDeltas}).
     *
     * @throws IllegalArgumentException when a qwen-rope encoder's (qwen2vl, qwen3vl, paddleocr, dots, glm4v,
     *                                  youtuvl) head width is not a multiple of 4 (the vision rope)
     */
    public static CpuVisionEncoder bind(ClipVisionModel model, CpuMatVec matVec,
                                        Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        return new CpuVisionEncoder(model, CpuVisionOps.bind(model, matVec, floats, refs, deltas), deltas);
    }

    /**
     * The LoRA deltas of this encoder's tower and connector products: an attachment over them
     * ({@link #attachLoras}) applies the adapters' vision pairs to every encoded row.
     */
    public CpuProjectionDeltas projectionDeltas() {
        return deltas;
    }

    @Override
    public LoraAttachment attachLoras(List<LoraAttachment.Binding> bindings) {
        Objects.requireNonNull(bindings, "bindings");
        if (loras != null) {
            throw new IllegalStateException("the vision encoder already runs LoRA adapters (" + loras + ")");
        }
        if (!LoraAttachment.adapts(bindings, LoraAdapter.Side.VISION)) {
            return null;
        }
        loras = LoraAttachment.cpu(bindings, deltas, LoraAdapter.Side.VISION);
        return loras;
    }

    @Override
    public int openLoraSession(List<LoraAttachment.Binding> bindings) {
        Objects.requireNonNull(bindings, "bindings");
        ensureOpen();
        if (sessionLoras != null) {
            throw new IllegalStateException("the vision encoder already runs a LoRA session (" + sessionLoras
                    + "); it serves one caller's session at a time");
        }
        sessionLoras = LoraAttachment.session(bindings, deltas, LoraAdapter.Side.VISION);
        return LORA_SESSION_SLOT;
    }

    @Override
    public void closeLoraSession(int slot) {
        if (sessionLoras == null) {
            throw new IllegalStateException("the vision encoder runs no LoRA session to close");
        }
        if (slot != LORA_SESSION_SLOT) {
            throw new IllegalStateException("the vision encoder's LoRA session runs in adapter slot "
                    + LORA_SESSION_SLOT + ", not " + slot);
        }
        sessionLoras.close();
        sessionLoras = null;
    }

    /**
     * Encodes one image tile to embedding rows.
     *
     * @return {@code [nTokens][outputDim]} rows (fresh arrays owned by the caller)
     * @throws IllegalArgumentException when the image size does not fit the encoder grid (named)
     * @throws IllegalStateException    when the encoder is closed
     */
    @Override
    public float[][] encode(ClipImage image) {
        ensureOpen();
        return VisionFamilies.encode(ops, image);
    }

    /**
     * Encodes one anyres image of the Granite Vision 4.1 encoder: the overview tile, then the {@code gridX x
     * gridY} tiles of the resized image in raster order, each an {@code image_size} square; the rows are the
     * overview's tokens, then the tile grid cropped to the original aspect ratio with a newline row after every
     * row ({@code build_anyres_assembly}).
     *
     * @throws IllegalArgumentException when the encoder is not granite4_vision, the tile count is not
     *                                  {@code gridX * gridY}, or a size is not positive (named)
     * @throws IllegalStateException    when the encoder is closed
     */
    @Override
    public float[][] encodeAnyres(ClipImage overview, List<ClipImage> tiles, int gridX, int gridY, int origWidth,
                                  int origHeight) {
        ensureOpen();
        Objects.requireNonNull(overview, "overview");
        Objects.requireNonNull(tiles, "tiles");
        return VisionFamilies.encodeAnyres(ops, overview, tiles, gridX, gridY, origWidth, origHeight);
    }

    @Override
    public ClipVisionModel model() {
        return model;
    }

    @Override
    public String device() {
        return CPU_DEVICE;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        if (sessionLoras != null) {
            try {
                sessionLoras.close();
            } catch (RuntimeException e) {
                failure = e;
            }
            sessionLoras = null;
        }
        if (loras != null) {
            try {
                loras.close();
            } catch (RuntimeException e) {
                if (failure != null) {
                    failure.addSuppressed(e);
                } else {
                    failure = e;
                }
            }
            loras = null;
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "CpuVisionEncoder{%s, out=%d}", model.hparams().projectorType(),
                model.outputDim());
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the vision encoder is closed");
        }
    }
}
