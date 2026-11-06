package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * An executor of one vision encoder and its projector: {@link CpuVisionEncoder} on the host or
 * {@link GpuVisionEncoder} on a GPU backend, both running the same clip graphs, or {@link DataParallelVisionEncoder}
 * over one whole-weight executor per rank of a {@link MmEncoderTpMode#DATA} rank group. {@link MmprojModel#openVisionEncoder}
 * picks one per the device and rank-group choices of {@link MmprojOptions}.
 */
public interface VisionEncoder extends AutoCloseable {

    /** Name of the CPU device as {@link #device()} reports it. */
    String CPU_DEVICE = "cpu";

    /** The adapter slot one {@link #openLoraSession} registers its pairs in. */
    int LORA_SESSION_SLOT = 1;

    /**
     * Encodes one preprocessed image to embedding rows.
     *
     * @return {@code [nTokens][projectionDim]} rows (fresh arrays owned by the caller)
     * @throws IllegalArgumentException when the image size does not fit the encoder grid
     * @throws IllegalStateException    when the encoder is closed
     */
    float[][] encode(ClipImage image);

    /**
     * Encodes a batch of preprocessed images, one result per image in input order. A data-parallel encoder
     * ({@link DataParallelVisionEncoder}) spreads the batch over its ranks; every other encoder runs the images one
     * after another, so the rows are exactly those of {@link #encode} per image.
     *
     * @param images the preprocessed images, possibly empty
     * @return one {@code [nTokens][projectionDim]} result per image, in input order (fresh arrays owned by the
     *         caller)
     * @throws IllegalArgumentException when an image does not fit the encoder grid
     * @throws IllegalStateException    when the encoder is closed
     */
    default List<float[][]> encodeBatch(List<ClipImage> images) {
        Objects.requireNonNull(images, "images");
        List<float[][]> out = new ArrayList<>(images.size());
        for (ClipImage image : images) {
            out.add(encode(image));
        }
        return out;
    }

    /**
     * Encodes one anyres image of a family that assembles its tiles in the graph (Granite Vision 4.1,
     * {@code build_anyres_assembly}): the overview tile, then the {@code gridX x gridY} tiles of the resized image
     * in raster order, each an {@code image_size} square, and the original image size the tile grid is cropped to.
     * Families without the assembly refuse the call.
     *
     * @param origWidth  width of the original image in pixels, before the anyres resize
     * @param origHeight height of the original image in pixels
     * @return {@code [nTokens][projectionDim]} rows (fresh arrays owned by the caller)
     * @throws IllegalArgumentException when the family has no anyres assembly, or the tiles do not fit it (named)
     * @throws IllegalStateException    when the encoder is closed
     */
    default float[][] encodeAnyres(ClipImage overview, List<ClipImage> tiles, int gridX, int gridY, int origWidth,
                                   int origHeight) {
        throw new IllegalArgumentException("anyres tile assembly is a granite4_vision graph; this encoder is "
                + model().hparams().projectorType());
    }

    /**
     * Attaches the vision-encoder and connector pairs of {@code bindings} ({@link LoraAdapter.Side#VISION}) to this
     * encoder's products: each pair's delta applies to every row the encoder computes, as a startup adapter's does
     * in the text forward pass. The attachment belongs to the encoder, which closes it with itself.
     *
     * @return the attachment, or null when no binding adapts a vision tensor
     * @throws IllegalStateException         when a pair targets a tensor no product of this encoder serves, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving product cannot apply deltas (a conv kernel read as an
     *                                       im2col matrix of another shape)
     */
    LoraAttachment attachLoras(List<LoraAttachment.Binding> bindings);

    /**
     * Opens a LoRA session over this encoder: the vision pairs of {@code bindings} apply to the rows the session's
     * caller encodes, in place of the startup adapters {@link #attachLoras} attached; an empty list opens the session
     * bare, its rows running no adapters. The caller closes the session ({@link #closeLoraSession}) when its rows
     * are done. The encoder is single-caller, so the session's rows are the caller's alone: no other encode runs
     * while the session is open.
     *
     * @return the adapter slot the session's rows run ({@link #LORA_SESSION_SLOT})
     * @throws IllegalStateException         when a session is open already, or a pair targets a tensor no product of
     *                                       this encoder serves or with a different served shape
     * @throws UnsupportedOperationException when the serving product cannot apply deltas
     */
    int openLoraSession(List<LoraAttachment.Binding> bindings);

    /**
     * Closes the LoRA session {@link #openLoraSession} opened: the session's deltas go and the encoder's rows return
     * to the startup adapters, or to none.
     *
     * @throws IllegalStateException when no session is open, or {@code slot} is not the open session's slot
     */
    void closeLoraSession(int slot);

    /** The geometry this executor runs. */
    ClipVisionModel model();

    /** Where the encoder runs: {@value #CPU_DEVICE}, the GPU device name, or the rank devices joined with '+' for a
     * data-parallel encoder. */
    String device();

    /** Ends use of the encoder and releases what it owns; further calls fail. Idempotent. */
    @Override
    void close();
}
