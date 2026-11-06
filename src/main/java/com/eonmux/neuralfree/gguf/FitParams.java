package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The parameters a fit chose for one device load, with the arguments that reproduce it: the context
 * ({@code -c}, llama.cpp's fitted context), the layer placement ({@code -ngl} for a resolved hybrid split, else
 * {@code -ngl all} for the streaming ladder), the expert overflow ({@code --cpu-moe}, or the per-layer
 * {@code -ot} entries llama.cpp's {@code tensor_buft_overrides} take), and {@code -nkvo} when the KV caches stay
 * in host RAM. Every argument is one the demos accept verbatim, so the line of {@link #arguments()} re-runs the
 * model with the fitted parameters (the streaming placements re-derive the same split from the same free device
 * memory, as llama.cpp's {@code llama_params_fit} output does; common/fit.cpp).
 *
 * <p>{@link #describe()} prints the memory estimate behind the choice ({@code -fitp}): the usable device bytes
 * the fit targeted, the device and host weights, the streaming slots, the context plan, the compute buffers.
 * Immutable.</p>
 */
public final class FitParams {

    /** How the fit places the model. */
    public enum Mode {
        /** Every weight resident on the device. */
        RESIDENT,
        /** The dense tensors of every trunk layer are resident; every expert stack stays in host RAM. */
        MOE_OVERFLOW,
        /** As MOE_OVERFLOW, with the expert stacks of the leading layers back on the device. */
        MOE_UPGRADED,
        /** The first trunk weight layers stream through the device; the rest keep their dense tensors there. */
        PARTIAL,
        /** Every trunk layer streams through the device; only model-level tensors stay resident. */
        STREAMED,
        /** A hybrid CPU+GPU layer split: the last {@code -ngl} minus one trunk layers compute on the device. */
        HYBRID
    }

    private static final double MIB = 1024.0 * 1024.0;

    private final Mode mode;
    private final int nCtx;
    private final String arguments;
    private final long usableBytes;
    private final long deviceWeightBytes;
    private final long hostWeightBytes;
    private final long slotBytes;
    private final long contextBytes;
    private final long deviceContextBytes;
    private final long computeBytes;
    private final int streamedLayers;
    private final int gpuLayers;

    FitParams(Mode mode, int nCtx, String arguments, long usableBytes, long deviceWeightBytes, long hostWeightBytes,
              long slotBytes, long contextBytes, long deviceContextBytes, long computeBytes, int streamedLayers,
              int gpuLayers) {
        this.mode = mode;
        this.nCtx = nCtx;
        this.arguments = arguments;
        this.usableBytes = usableBytes;
        this.deviceWeightBytes = deviceWeightBytes;
        this.hostWeightBytes = hostWeightBytes;
        this.slotBytes = slotBytes;
        this.contextBytes = contextBytes;
        this.deviceContextBytes = deviceContextBytes;
        this.computeBytes = computeBytes;
        this.streamedLayers = streamedLayers;
        this.gpuLayers = gpuLayers;
    }

    /** How the fit places the model. */
    public Mode mode() {
        return mode;
    }

    /** The fitted context ({@code -c}). */
    public int nCtx() {
        return nCtx;
    }

    /** The arguments that reproduce the fitted load, one space-separated line the demos accept verbatim. */
    public String arguments() {
        return arguments;
    }

    /** The device bytes the fit targeted: the free VRAM less the load's reserves, the compute buffers and the
     * fit margin. */
    public long usableBytes() {
        return usableBytes;
    }

    /** Device bytes of the weights the fit keeps resident there. */
    public long deviceWeightBytes() {
        return deviceWeightBytes;
    }

    /** Host bytes of the weights the fit keeps in RAM: the streamed sets and the host layers of a split. */
    public long hostWeightBytes() {
        return hostWeightBytes;
    }

    /** Device bytes of the streaming slots the fit reserved for the streamed sets, 0 without streaming. */
    public long slotBytes() {
        return slotBytes;
    }

    /** Bytes of the whole context plan at the fitted context ({@code MemoryPlanner}). */
    public long contextBytes() {
        return contextBytes;
    }

    /** Bytes of the context plan that stay on the device: the whole plan, less the host layers' caches of a
     * split ({@code HybridFit#contextBytes}). */
    public long deviceContextBytes() {
        return deviceContextBytes;
    }

    /** Device bytes of the compute buffers the load allocates before the placement: the model's scratch slots
     * and the kernel placeholders. */
    public long computeBytes() {
        return computeBytes;
    }

    /** Trunk weight layers the fit streams through the device, 0 without streaming. */
    public int streamedLayers() {
        return streamedLayers;
    }

    /** Trunk weight layers whose weights the fit keeps on the device; a split keeps its last layers there. */
    public int gpuLayers() {
        return gpuLayers;
    }

    /** The memory estimate of the fit, one {@code label <MiB>} line per non-zero figure, the arguments last. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        line(sb, "usable device memory", usableBytes, "");
        line(sb, "device weights", deviceWeightBytes, "");
        line(sb, "streaming slots", slotBytes, streamedLayers > 0
                ? streamedLayers + " layer(s) stream through the device" : "");
        line(sb, "host weights", hostWeightBytes, "");
        line(sb, "context plan", contextBytes, "context " + nCtx
                + (deviceContextBytes < contextBytes
                ? String.format(Locale.ROOT, ", device caches %.1f MiB", deviceContextBytes / MIB) : ""));
        line(sb, "compute buffers", computeBytes, "scratch and placeholders");
        return sb.toString().trim();
    }

    private static void line(StringBuilder sb, String label, long bytes, String note) {
        if (bytes == 0) {
            return;
        }
        sb.append(String.format(Locale.ROOT, "  %-20s %10.1f MiB", label, bytes / MIB));
        if (!note.isEmpty()) {
            sb.append("  (").append(note).append(')');
        }
        sb.append('\n');
    }

    @Override
    public String toString() {
        return "FitParams{" + mode + ", context " + nCtx + ", arguments [" + arguments + "]}";
    }

    // ------------------------------------------------------------------ translation

    /**
     * The fitted parameters of a load's placement: the mode, the streaming figures of a streaming fit and the
     * arguments line. {@code contextBytes} is the whole context plan at {@code placement.nCtx};
     * {@code deviceContextBytes} the caches that stay on the device; {@code computeBytes} the scratch and
     * placeholder bytes the load allocated before the placement; {@code deviceWeightBytes} the planned device
     * bytes of the tensors the placement puts on the device and {@code totalWeightBytes} those of every weight,
     * from which the host figure follows for a resident or split placement.
     */
    static FitParams of(GpuLoadPlacement.Placement placement, long usableBytes, long contextBytes,
                        long deviceContextBytes, long computeBytes, long totalWeightBytes, long deviceWeightBytes,
                        int weightLayers) {
        Mode mode;
        long device = deviceWeightBytes;
        long host = totalWeightBytes - deviceWeightBytes;
        long slots = 0L;
        int streamed = 0;
        int gpuLayers = weightLayers;
        if (placement.split != null) {
            mode = Mode.HYBRID;
            gpuLayers = placement.split.nGpuLayers();
        } else if (placement.fit != null) {
            PlacementFitter.Fit fit = placement.fit;
            mode = Mode.valueOf(fit.mode().name());
            device = fit.deviceWeightBytes();
            host = fit.hostWeightBytes();
            slots = fit.slotBytes();
            streamed = fit.streamedLayers();
            gpuLayers = fit.gpuLayers();
        } else {
            mode = Mode.RESIDENT;
        }
        return new FitParams(mode, placement.nCtx, argumentsOf(placement), usableBytes, device, host, slots,
                contextBytes, deviceContextBytes, computeBytes, streamed, gpuLayers);
    }

    /**
     * The arguments that reproduce a placement: the fitted context, then the placement
     * ({@link #placementArgs}): {@code -ngl <n>} for a hybrid split with {@code -nkvo} when its KV caches stay
     * in host RAM; {@code -ngl all} for a resident or streaming placement, with the expert overflow of the MoE
     * modes as {@code --cpu-moe} or the per-layer {@code -ot} entries.
     */
    static String argumentsOf(GpuLoadPlacement.Placement placement) {
        List<String> args = new ArrayList<>();
        args.add("-c " + placement.nCtx);
        args.add(placementArgs(placement));
        return String.join(" ", args);
    }

    /** The placement arguments of a placement: the {@code -ngl} form, the expert overflow and {@code -nkvo}. */
    private static String placementArgs(GpuLoadPlacement.Placement placement) {
        if (placement.split != null) {
            LayerOffload split = placement.split;
            return "-ngl " + (split.nGpuLayers() + 1) + (split.hostKv() ? " -nkvo" : "");
        }
        if (placement.fit == null || placement.fit.mode() == PlacementFitter.Mode.RESIDENT
                || placement.fit.mode() == PlacementFitter.Mode.STREAMED
                || placement.fit.mode() == PlacementFitter.Mode.PARTIAL) {
            return "-ngl all";
        }
        PlacementFitter.Fit fit = placement.fit;
        if (fit.mode() == PlacementFitter.Mode.MOE_OVERFLOW && fit.fullLayers() == 0) {
            return "-ngl all --cpu-moe";
        }
        String overrides = expertOverrides(fit.fullLayers(), fit.weightLayers());
        return overrides.isEmpty() ? "-ngl all" : "-ngl all " + overrides;
    }

    /**
     * The {@code -ot} entries that keep the expert stacks of the layers {@code [fromLayers, weightLayers)} in
     * host RAM, the per-block expert patterns the {@code --n-cpu-moe} setting derives
     * ({@link LayerOffload#EXPERT_TENSORS_REGEX}); single-quoted, so a shell keeps the pattern's backslashes.
     */
    static String expertOverrides(int fromLayers, int weightLayers) {
        if (fromLayers >= weightLayers) {
            return "";
        }
        StringBuilder entries = new StringBuilder();
        for (int il = fromLayers; il < weightLayers; il++) {
            if (entries.length() > 0) {
                entries.append(',');
            }
            entries.append("blk\\.").append(il).append(LayerOffload.EXPERT_TENSORS_REGEX).append("=cpu");
        }
        return "-ot '" + entries + "'";
    }
}
