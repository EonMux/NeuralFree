package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * Immutable VRAM allocation budget computed from the model's memory plan, compute buffers, weights and the
 * device's free memory.
 *
 * <p>Accounts for the sequence memory ({@link Builder#memoryPlan(MemoryPlan)}: KV cache with its per-dtype row
 * sizes, recurrent state and its snapshot shadows, indexer / MSA / compressed side caches, cross-attention K/V and
 * {@code kv_pos} buffers), compute buffers, persistent weights and system VRAM overhead before computing the streaming
 * budget and maximum streaming depth. When the depth reaches the layer count every streamed layer stays resident and no
 * RAM prefetch tier is needed.</p>
 *
 * <p>Free VRAM comes from {@link #queryFreeVram(ComputeBackend)}: the driver's own figure through
 * {@link ComputeBackend#getFreeVramBytes()} where it reflects every process (CUDA {@code cuMemGetInfo}, Vulkan
 * {@code VK_EXT_memory_budget}), {@code nvidia-smi} for other APIs on NVIDIA devices, and otherwise the backend's
 * estimate (total minus this process's allocations) with the larger fixed reserve.</p>
 */
public final class VRAMBudget {

    private static final Logger LOGGER = LogManager.getLogger(VRAMBudget.class);

    /** Fixed VRAM reserve for driver, display compositor, runtime, kernel code, and buffer metadata/alignment
     *  overhead. Used when free VRAM is unknown or only estimated from this process's allocations. */
    private static final long FIXED_RESERVE_BYTES = 512L * 1024 * 1024;

    /** Smaller reserve applied when free VRAM was measured across all processes (buffer alignment, driver
     *  metadata per buffer, and GPU page table overhead). */
    private static final long MEASURED_RESERVE_BYTES = 384L * 1024 * 1024;

    /** PCI vendor id of NVIDIA. */
    private static final int NVIDIA_VENDOR_ID = 0x10DE;

    /** Seconds to wait for {@code nvidia-smi}. */
    private static final long NVIDIA_SMI_TIMEOUT_SECONDS = 5;

    /** Where a free-VRAM figure came from. */
    public enum FreeVramSource {
        /** The backend's driver query, which accounts for every process. */
        DRIVER,
        /** {@code nvidia-smi}, which accounts for every process. */
        NVIDIA_SMI,
        /** Total VRAM minus this process's allocations; other processes are not accounted for. */
        ESTIMATE,
        /** Nothing is known; the budget uses the total VRAM minus the fixed reserve. */
        UNKNOWN;

        /** True when the figure accounts for memory used by other processes. */
        public boolean measured() {
            return this == DRIVER || this == NVIDIA_SMI;
        }
    }

    /** Free VRAM in bytes (-1 when unknown) and its source. Immutable. */
    public static final class FreeVram {
        /** Free VRAM unknown. */
        public static final FreeVram UNKNOWN = new FreeVram(-1, FreeVramSource.UNKNOWN);

        private final long bytes;
        private final FreeVramSource source;

        private FreeVram(long bytes, FreeVramSource source) {
            this.bytes = bytes;
            this.source = source;
        }

        /** A known figure; a non-positive byte count yields {@link #UNKNOWN}. */
        public static FreeVram of(long bytes, FreeVramSource source) {
            Objects.requireNonNull(source, "source");
            if (bytes <= 0 || source == FreeVramSource.UNKNOWN) {
                return UNKNOWN;
            }
            return new FreeVram(bytes, source);
        }

        /** Free bytes, or -1 when unknown. */
        public long bytes() { return bytes; }
        public FreeVramSource source() { return source; }

        @Override
        public String toString() {
            return source == FreeVramSource.UNKNOWN ? "FreeVram{unknown}" : "FreeVram{" + bytes + " B, " + source + '}';
        }
    }

    private final long totalVram;
    private final long kvCacheBytes;
    private final long crossKvBytes;
    private final long recurrentStateBytes;
    private final long stateSnapshotBytes;
    private final long sideCacheBytes;
    private final long positionBytes;
    private final long computeBufferBytes;
    private final long persistentWeightBytes;
    private final long miscBytes;
    private final long fixedReserveBytes;
    private final long streamingBudget;
    private final long perLayerStreamBytes;
    private final int maxStreamingDepth;
    private final int layerCount;
    private final int ramPrefetchWindow;
    private final FreeVramSource freeVramSource;

    private VRAMBudget(Builder b) {
        this.totalVram = b.totalVram;
        this.kvCacheBytes = b.kvCacheBytes;
        this.crossKvBytes = b.crossKvBytes;
        this.recurrentStateBytes = b.recurrentStateBytes;
        this.stateSnapshotBytes = b.stateSnapshotBytes;
        this.sideCacheBytes = b.sideCacheBytes;
        this.positionBytes = b.positionBytes;
        this.computeBufferBytes = b.computeBufferBytes;
        this.miscBytes = b.miscBytes;
        this.layerCount = b.layerCount;
        this.perLayerStreamBytes = b.perLayerStreamBytes;
        this.freeVramSource = b.freeVram.source();

        long usableVram = usableVram(totalVram, b.freeVram, b.fixedReserveBytes);
        this.fixedReserveBytes = totalVram - usableVram;

        // Cap persistent weight bytes at what VRAM can actually hold after other allocations.
        // For MoE models, non-streamed expert weights can total 60+ GB, far exceeding VRAM.
        // Always reserve space for at least 2 streaming buffer sets (minimum for double-buffering).
        long otherReserved = memoryBytes() + computeBufferBytes + miscBytes;
        int minStreamDepth = Math.min(2, layerCount);
        long minStreamReserve = (perLayerStreamBytes > 0) ? perLayerStreamBytes * minStreamDepth : 0;
        long maxPersistent = Math.max(0, usableVram - otherReserved - minStreamReserve);
        this.persistentWeightBytes = Math.min(b.persistentWeightBytes, maxPersistent);

        long reserved = this.persistentWeightBytes + otherReserved;
        long budget = usableVram - reserved;
        this.streamingBudget = Math.max(0, budget);

        if (perLayerStreamBytes > 0 && streamingBudget > 0) {
            this.maxStreamingDepth = (int) Math.min(streamingBudget / perLayerStreamBytes, layerCount);
        } else if (perLayerStreamBytes > 0) {
            // The streaming budget is zero but minStreamReserve was kept free for double-buffering.
            this.maxStreamingDepth = minStreamDepth;
        } else {
            this.maxStreamingDepth = Math.min(4, layerCount);
        }

        // RAM prefetch window: layers beyond VRAM depth that should be pre-loaded into RAM.
        // When all layers fit in VRAM, no RAM tier is needed.
        if (maxStreamingDepth >= layerCount) {
            this.ramPrefetchWindow = 0;
        } else {
            this.ramPrefetchWindow = Math.min(20, layerCount - maxStreamingDepth);
        }
    }

    public long totalVram() { return totalVram; }
    /** Self-attention KV cache bytes (all layers, per-dtype row sizes when built from a memory plan). */
    public long kvCacheBytes() { return kvCacheBytes; }
    /** T5 cross-attention K/V bytes. */
    public long crossKvBytes() { return crossKvBytes; }
    /** Recurrent state bytes. */
    public long recurrentStateBytes() { return recurrentStateBytes; }
    /** Recurrent-state snapshot shadow bytes ({@link MemoryPlan#snapshotBytes()}). */
    public long stateSnapshotBytes() { return stateSnapshotBytes; }
    /** DSA indexer, MSA index-key and DSV4 compressed cache bytes. */
    public long sideCacheBytes() { return sideCacheBytes; }
    /** {@code kv_pos} buffer bytes. */
    public long positionBytes() { return positionBytes; }
    /** All sequence-memory bytes: the sum of the six categories above. */
    public long memoryBytes() {
        return kvCacheBytes + crossKvBytes + recurrentStateBytes + stateSnapshotBytes + sideCacheBytes + positionBytes;
    }
    public long computeBufferBytes() { return computeBufferBytes; }
    public long persistentWeightBytes() { return persistentWeightBytes; }
    public long miscBytes() { return miscBytes; }
    /** VRAM held back from the budget: other processes (when measured) plus the reserve. */
    public long fixedReserveBytes() { return fixedReserveBytes; }
    public long streamingBudget() { return streamingBudget; }
    public long perLayerStreamBytes() { return perLayerStreamBytes; }
    public int maxStreamingDepth() { return maxStreamingDepth; }
    public int layerCount() { return layerCount; }
    public int ramPrefetchWindow() { return ramPrefetchWindow; }
    public FreeVramSource freeVramSource() { return freeVramSource; }

    /** Human-readable breakdown for logging. */
    public String formatBreakdown() {
        long mb = 1024L * 1024;
        StringBuilder sb = new StringBuilder();
        sb.append("=== VRAM Budget ===\n");
        sb.append(format("Total VRAM:            %6d MB\n", totalVram / mb));
        sb.append(format("  KV Cache:            -%5d MB\n", kvCacheBytes / mb));
        if (crossKvBytes > 0) {
            sb.append(format("  Cross-Attention KV:  -%5d MB\n", crossKvBytes / mb));
        }
        if (recurrentStateBytes > 0) {
            sb.append(format("  Recurrent State:     -%5d MB\n", recurrentStateBytes / mb));
        }
        if (stateSnapshotBytes > 0) {
            sb.append(format("  State Snapshots:     -%5d MB\n", stateSnapshotBytes / mb));
        }
        if (sideCacheBytes > 0) {
            sb.append(format("  Indexer/Side Caches: -%5d MB\n", sideCacheBytes / mb));
        }
        if (positionBytes > 0) {
            sb.append(format("  KV Positions:        -%5d MB  (%d bytes)\n", positionBytes / mb, positionBytes));
        }
        sb.append(format("  Compute Buffers:     -%5d MB\n", computeBufferBytes / mb));
        sb.append(format("  Persistent Weights:  -%5d MB\n", persistentWeightBytes / mb));
        sb.append(format("  Misc:                -%5d MB\n", miscBytes / mb));
        sb.append(format("  System/Reserve:      -%5d MB  (%s)\n", fixedReserveBytes / mb, describe(freeVramSource)));
        sb.append("  ---\n");
        sb.append(format("  Streaming Budget:    %5d MB  (per-layer: %d MB)\n",
                streamingBudget / mb, perLayerStreamBytes > 0 ? perLayerStreamBytes / mb : 0));

        if (maxStreamingDepth >= layerCount) {
            sb.append(format("  VRAM Depth:          %d/%d layers (all resident)\n",
                    maxStreamingDepth, layerCount));
            sb.append(format("  RAM Prefetch Window: %d layers (all in VRAM, no RAM tier needed)\n", ramPrefetchWindow));
            sb.append(format("  Disk (mmap):         %d layers\n", 0));
        } else {
            int diskLayers = layerCount - maxStreamingDepth - ramPrefetchWindow;
            sb.append(format("  VRAM Depth:          %d/%d layers (ring buffer)\n", maxStreamingDepth, layerCount));
            sb.append(format("  RAM Prefetch Window: %d layers (loaded from disk ahead of need)\n", ramPrefetchWindow));
            sb.append(format("  Disk (mmap):         %d layers (OS page cache)\n", Math.max(0, diskLayers)));
        }
        return sb.toString();
    }

    private static String format(String fmt, Object... args) {
        return String.format(Locale.ROOT, fmt, args);
    }

    private static String describe(FreeVramSource source) {
        switch (source) {
            case DRIVER:
                return "free VRAM measured by the driver";
            case NVIDIA_SMI:
                return "free VRAM measured via nvidia-smi";
            case ESTIMATE:
                return "free VRAM estimated from this process's allocations";
            default:
                return "static estimate";
        }
    }

    /**
     * VRAM usable by this process: measured free memory minus the small reserve, or the estimated free memory (or
     * the total when nothing is known) minus the fixed reserve. Never negative.
     */
    private static long usableVram(long totalVram, FreeVram free, long fixedReserveBytes) {
        long usable;
        if (free.bytes() > 0 && free.bytes() < totalVram && free.source().measured()) {
            usable = free.bytes() - MEASURED_RESERVE_BYTES;
        } else if (free.bytes() > 0 && free.bytes() < totalVram && free.source() == FreeVramSource.ESTIMATE) {
            usable = free.bytes() - fixedReserveBytes;
        } else {
            usable = totalVram - fixedReserveBytes;
        }
        return Math.max(0, usable);
    }

    /**
     * Free VRAM of the backend's device, preferring figures that account for every process: the driver's
     * ({@link ComputeBackend#getFreeVramBytes()} on CUDA, and on Vulkan with {@link DeviceFeature#MEMORY_BUDGET}),
     * then {@code nvidia-smi} on NVIDIA devices (matched by PCI bus id or UUID), then the backend's estimate.
     * Call it before this process allocates model buffers so the figure is the true baseline.
     */
    public static FreeVram queryFreeVram(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend");
        DeviceCapabilities caps = backend.capabilities();
        ComputeBackend.BackendType type = backend.backendType();
        boolean driverMeasured = type == ComputeBackend.BackendType.CUDA
                || (type == ComputeBackend.BackendType.VULKAN && caps.hasFeature(DeviceFeature.MEMORY_BUDGET));
        long estimate = -1;
        try {
            long free = backend.getFreeVramBytes();
            if (driverMeasured && free > 0) {
                return FreeVram.of(free, FreeVramSource.DRIVER);
            }
            estimate = free;
        } catch (RuntimeException e) {
            LOGGER.warn("{} backend could not report free VRAM on {}: {}", type, caps.getDeviceName(), e.getMessage());
        }
        if (isNvidia(caps)) {
            long smi = nvidiaSmiFreeBytes(caps.getIdentity());
            if (smi > 0) {
                return FreeVram.of(smi, FreeVramSource.NVIDIA_SMI);
            }
        }
        return FreeVram.of(estimate, FreeVramSource.ESTIMATE);
    }

    /**
     * Bytes left for the sequence memory plan ({@code MemoryPlanner} / {@code ContextFitter} budget): usable VRAM
     * minus the weights still to be allocated, the compute buffers and other allocations. Never negative.
     *
     * @param residentWeightBytes weights that will be resident in VRAM and are not yet allocated
     * @param computeBytes        compute and scratch buffers
     * @param miscBytes           every other allocation not yet made (streaming buffer sets, sampling buffers)
     */
    public static long memoryBudgetBytes(ComputeBackend backend, long residentWeightBytes, long computeBytes,
                                         long miscBytes) {
        Objects.requireNonNull(backend, "backend");
        return memoryBudgetBytes(backend.capabilities().getTotalVramBytes(), queryFreeVram(backend),
                residentWeightBytes, computeBytes, miscBytes);
    }

    /** {@link #memoryBudgetBytes(ComputeBackend, long, long, long)} for an explicit device size and free figure. */
    public static long memoryBudgetBytes(long totalVram, FreeVram free, long residentWeightBytes, long computeBytes,
                                         long miscBytes) {
        Objects.requireNonNull(free, "free");
        if (totalVram <= 0 || residentWeightBytes < 0 || computeBytes < 0 || miscBytes < 0) {
            throw new IllegalArgumentException("VRAM sizes must be non-negative with a positive total: total="
                    + totalVram + " weights=" + residentWeightBytes + " compute=" + computeBytes + " misc=" + miscBytes);
        }
        long usable = usableVram(totalVram, free, FIXED_RESERVE_BYTES);
        return Math.max(0, usable - residentWeightBytes - computeBytes - miscBytes);
    }

    private static boolean isNvidia(DeviceCapabilities caps) {
        DeviceIdentity id = caps.getIdentity();
        if (id != null && id.vendorId() == NVIDIA_VENDOR_ID) {
            return true;
        }
        String name = caps.getDeviceName();
        return name != null && name.toLowerCase(Locale.ROOT).contains("nvidia");
    }

    /**
     * Free VRAM of the {@code nvidia-smi} GPU that is {@code id} (same PCI bus id, else same UUID); the only GPU
     * when the identity carries neither; -1 when no GPU matches or the query fails.
     */
    private static long nvidiaSmiFreeBytes(DeviceIdentity id) {
        List<String[]> rows = nvidiaSmi("pci.bus_id,uuid,memory.free");
        boolean comparable = id != null && (id.pciBusId() != null || id.uuid() != null);
        if (!comparable) {
            return rows.size() == 1 && rows.get(0).length >= 3 ? parseMib(rows.get(0)[2]) : -1;
        }
        for (String[] row : rows) {
            if (row.length < 3) {
                continue;
            }
            DeviceIdentity gpu;
            try {
                String uuid = row[1].startsWith("GPU-") ? row[1].substring(4) : null;
                gpu = new DeviceIdentity("", DeviceIdentity.UNKNOWN_VENDOR, row[0], uuid);
            } catch (IllegalArgumentException e) {
                LOGGER.debug("nvidia-smi row {} has no usable identity: {}", String.join(",", row), e.getMessage());
                continue;
            }
            if (id.samePhysicalDevice(gpu)) {
                return parseMib(row[2]);
            }
        }
        return -1;
    }

    private static long parseMib(String value) {
        try {
            return Long.parseLong(value.trim()) * 1024L * 1024L;
        } catch (NumberFormatException e) {
            LOGGER.debug("nvidia-smi reported a non-numeric free memory value '{}'", value);
            return -1;
        }
    }

    /**
     * Runs {@code nvidia-smi --query-gpu=<fields>} and splits each output line on commas; empty when the tool is
     * missing, times out, exits with an error or prints nothing. The output is a few short lines per GPU, which
     * fit the pipe buffer, so it is read after the process ends.
     */
    private static List<String[]> nvidiaSmi(String fields) {
        Process proc;
        try {
            proc = new ProcessBuilder("nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits")
                    .redirectError(ProcessBuilder.Redirect.DISCARD)
                    .start();
        } catch (IOException e) {
            LOGGER.debug("nvidia-smi is not available: {}", e.getMessage());
            return Collections.emptyList();
        }
        try {
            if (!proc.waitFor(NVIDIA_SMI_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                proc.destroyForcibly();
                LOGGER.debug("nvidia-smi did not finish within {} s", NVIDIA_SMI_TIMEOUT_SECONDS);
                return Collections.emptyList();
            }
            if (proc.exitValue() != 0) {
                LOGGER.debug("nvidia-smi --query-gpu={} exited with {}", fields, proc.exitValue());
                return Collections.emptyList();
            }
            List<String[]> rows = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().isEmpty()) {
                        continue;
                    }
                    String[] cells = line.split(",");
                    for (int i = 0; i < cells.length; i++) {
                        cells[i] = cells[i].trim();
                    }
                    rows.add(cells);
                }
            }
            return rows;
        } catch (IOException e) {
            LOGGER.debug("reading nvidia-smi output failed: {}", e.getMessage());
            return Collections.emptyList();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            proc.destroyForcibly();
            return Collections.emptyList();
        }
    }

    public static final class Builder {
        private long totalVram;
        private long kvCacheBytes;
        private long crossKvBytes;
        private long recurrentStateBytes;
        private long stateSnapshotBytes;
        private long sideCacheBytes;
        private long positionBytes;
        private long computeBufferBytes;
        private long persistentWeightBytes;
        private long miscBytes;
        private long perLayerStreamBytes;
        private int layerCount;
        private long fixedReserveBytes = FIXED_RESERVE_BYTES;
        private FreeVram freeVram = FreeVram.UNKNOWN;

        public Builder totalVram(long bytes) { this.totalVram = bytes; return this; }
        /** Self-attention KV bytes; {@link #memoryPlan(MemoryPlan)} sets them from a plan. */
        public Builder kvCacheBytes(long bytes) { this.kvCacheBytes = bytes; return this; }
        public Builder computeBufferBytes(long bytes) { this.computeBufferBytes = bytes; return this; }
        public Builder persistentWeightBytes(long bytes) { this.persistentWeightBytes = bytes; return this; }
        public Builder miscBytes(long bytes) { this.miscBytes = bytes; return this; }
        public Builder perLayerStreamBytes(long bytes) { this.perLayerStreamBytes = bytes; return this; }
        public Builder layerCount(int count) { this.layerCount = count; return this; }
        public Builder fixedReserveBytes(long bytes) { this.fixedReserveBytes = bytes; return this; }

        /**
         * Takes every sequence-memory category from a memory plan: self-attention KV, cross-attention K/V,
         * recurrent state and its snapshot shadows, side caches and {@code kv_pos} buffers, each with its exact
         * per-dtype size.
         */
        public Builder memoryPlan(MemoryPlan plan) {
            Objects.requireNonNull(plan, "plan");
            long cross = 0;
            for (LayerMemoryPlan l : plan.layers()) {
                if (l.crossKv() != null) {
                    cross += l.crossKv().totalBytes();
                }
            }
            this.kvCacheBytes = plan.kvBytes() - cross;
            this.crossKvBytes = cross;
            this.recurrentStateBytes = plan.stateBytes();
            this.stateSnapshotBytes = plan.snapshotBytes();
            this.sideCacheBytes = plan.sideCacheBytes();
            this.positionBytes = plan.positionBytes();
            return this;
        }

        /** Free VRAM and its source ({@link #queryFreeVram(ComputeBackend)}), measured before this process allocates. */
        public Builder freeVram(FreeVram free) {
            this.freeVram = Objects.requireNonNull(free, "free");
            return this;
        }

        public VRAMBudget build() {
            return new VRAMBudget(this);
        }
    }
}
