package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One model tensor resident on the device, as the GPU forward path consumes it.
 *
 * <p>Shape follows ggml: {@code ne[0]} is the contiguous input width ({@link #cols()}), the product of the higher
 * dimensions is the row count ({@link #rows()}; for a 3-D expert tensor {@code ne[1] * ne[2]}). Rows are stored in
 * the GGUF block layout of {@link #type()}, {@link #rowBytes()} bytes each.</p>
 *
 * <p><b>Row shards.</b> A tensor larger than one device allocation or one storage binding is split into row ranges
 * held by separate buffers: shard {@code i} holds rows {@code [shardRowStart(i), shardRowStart(i) + shardRows(i))}
 * starting at byte 0 of {@code shard(i)}. Shards are contiguous, cover every row and, for tensors of rank 3 or more,
 * start and end on {@code ne[1]} (expert) boundaries.</p>
 *
 * <p><b>Transforms.</b> {@link #hadamard()} is the PRISM input transform {@code H*(D*x)} applied to the matmul input
 * of a folded weight (llama-graph.cpp:1598-1626); {@link #inverseHadamard()} is the transform {@code D*(H*z)}
 * applied to rows looked up from a latent embedding table (llama-graph.cpp:2683-2690). {@link #scale()} is
 * the per-tensor output scale {@code w_s} ({@code <tensor>.scale}, llama-graph.cpp:1643-1645), 1 when
 * absent. {@link #inputTransform()} is the full input transform chain HADAMARD &rarr; ACT_QUANT &rarr; PERMUTE
 * ({@link InputTransform}) of the matmul input (vLLM's compressed-tensors input transform before the quantization
 * method and the act-order gather); its HADAMARD part is what {@link #hadamard()} returns, and {@link
 * #inputPerm()} is the read-only device int buffer of the {@code <base>.input_perm} tensor the PERMUTE gather
 * runs over.</p>
 *
 * <p>Immutable. The shard buffers belong to whoever uploaded the tensor (the weight uploader); a {@code GpuWeight}
 * never closes them.</p>
 */
public final class GpuWeight {

    private final String name;
    private final GGMLType type;
    private final long[] ne;
    private final long rows;
    private final long cols;
    private final long rowBytes;
    private final GpuBuffer[] shards;
    private final long[] shardRowStart;
    private final long[] shardRows;
    private final HadamardSpec hadamard;
    private final HadamardSpec inverseHadamard;
    private final float scale;
    private final InputTransform inputTransform;
    private final GpuBuffer inputPerm;

    private GpuWeight(String name, GGMLType type, long[] ne, long rows, long cols, long rowBytes, GpuBuffer[] shards,
                      long[] shardRowStart, long[] shardRows, HadamardSpec hadamard, HadamardSpec inverseHadamard,
                      float scale, InputTransform inputTransform, GpuBuffer inputPerm) {
        this.name = name;
        this.type = type;
        this.ne = ne;
        this.rows = rows;
        this.cols = cols;
        this.rowBytes = rowBytes;
        this.shards = shards;
        this.shardRowStart = shardRowStart;
        this.shardRows = shardRows;
        this.hadamard = hadamard;
        this.inverseHadamard = inverseHadamard;
        this.scale = scale;
        this.inputTransform = inputTransform;
        this.inputPerm = inputPerm;
    }

    /**
     * A tensor held by one buffer.
     *
     * @param name   tensor name as stored in the GGUF file
     * @param type   element type of the stored rows
     * @param ne     ggml shape, {@code ne[0]} first (copied)
     * @param buffer device buffer holding all rows from byte 0
     * @throws IllegalArgumentException when the type has no fixed block layout, the shape is empty or negative, the
     *                                  row width is not a whole number of blocks, or the buffer is too small
     */
    public static GpuWeight of(String name, GGMLType type, long[] ne, GpuBuffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        long[] shape = checkShape(name, type, ne);
        return sharded(name, type, shape, Collections.singletonList(buffer), new long[]{rowsOf(shape)});
    }

    /**
     * A tensor split into row shards.
     *
     * @param name      tensor name as stored in the GGUF file
     * @param type      element type of the stored rows
     * @param ne        ggml shape, {@code ne[0]} first (copied)
     * @param buffers   one buffer per shard, in row order
     * @param shardRows rows held by each shard; positive, summing to {@link #rows()}
     * @throws IllegalArgumentException when the shape or type is invalid, the shard lists disagree, a shard is empty
     *                                  or too small for its rows, the shards do not cover every row, or a shard of an
     *                                  expert tensor splits an expert
     */
    public static GpuWeight sharded(String name, GGMLType type, long[] ne, List<GpuBuffer> buffers, long[] shardRows) {
        long[] shape = checkShape(name, type, ne);
        Objects.requireNonNull(buffers, "buffers");
        Objects.requireNonNull(shardRows, "shardRows");
        if (buffers.isEmpty() || buffers.size() != shardRows.length) {
            throw new IllegalArgumentException("tensor " + name + ": " + buffers.size() + " shard buffers for "
                    + shardRows.length + " shard row counts (both must be equal and non-zero)");
        }
        long rows = rowsOf(shape);
        long cols = shape[0];
        long rowBytes = type.getBytesForElements(cols);
        long expertRows = shape.length >= 3 ? shape[1] : 1;
        int n = shardRows.length;
        GpuBuffer[] bufs = new GpuBuffer[n];
        long[] starts = new long[n];
        long[] counts = shardRows.clone();
        long next = 0;
        for (int i = 0; i < n; i++) {
            GpuBuffer b = buffers.get(i);
            if (b == null) {
                throw new IllegalArgumentException("tensor " + name + ": shard " + i + " has no buffer");
            }
            if (counts[i] <= 0) {
                throw new IllegalArgumentException("tensor " + name + ": shard " + i + " holds " + counts[i] + " rows");
            }
            if (expertRows > 1 && (next % expertRows != 0 || counts[i] % expertRows != 0)) {
                throw new IllegalArgumentException("tensor " + name + ": shard " + i + " rows [" + next + ", "
                        + (next + counts[i]) + ") split an expert of " + expertRows + " rows");
            }
            long need = counts[i] * rowBytes;
            if (b.sizeInBytes() < need) {
                throw new IllegalArgumentException("tensor " + name + ": shard " + i + " buffer holds " + b.sizeInBytes()
                        + " bytes, its " + counts[i] + " rows of " + type + " need " + need);
            }
            bufs[i] = b;
            starts[i] = next;
            next += counts[i];
        }
        if (next != rows) {
            throw new IllegalArgumentException("tensor " + name + ": shards hold " + next + " rows, the tensor has " + rows);
        }
        return new GpuWeight(name, type, shape, rows, cols, rowBytes, bufs, starts, counts, null, null, 1.0f, null,
                null);
    }

    private static long[] checkShape(String name, GGMLType type, long[] ne) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("GPU weight needs a tensor name");
        }
        Objects.requireNonNull(type, "type");
        if (type == GGMLType.UNKNOWN || type.getTypeSize() <= 0 || type.getBlockSize() <= 0) {
            throw new IllegalArgumentException("tensor " + name + ": type " + type + " has no fixed block layout");
        }
        if (ne == null || ne.length == 0) {
            throw new IllegalArgumentException("tensor " + name + " has no dimensions");
        }
        long[] shape = ne.clone();
        for (long d : shape) {
            if (d <= 0) {
                throw new IllegalArgumentException("tensor " + name + " has a non-positive dimension: "
                        + Arrays.toString(shape));
            }
        }
        if (shape[0] % type.getBlockSize() != 0) {
            throw new IllegalArgumentException("tensor " + name + ": row width " + shape[0]
                    + " is not a multiple of the " + type + " block size " + type.getBlockSize());
        }
        return shape;
    }

    private static long rowsOf(long[] ne) {
        long r = 1;
        for (int i = 1; i < ne.length; i++) {
            r = Math.multiplyExact(r, ne[i]);
        }
        return r;
    }

    /**
     * This tensor with the PRISM input transform applied to its matmul input.
     *
     * @throws IllegalArgumentException when the transform does not fit the row width, or its block exceeds what
     *                                  the {@value HadamardSpec#KERNEL} kernel supports
     */
    public GpuWeight withHadamard(HadamardSpec forward) {
        Objects.requireNonNull(forward, "forward");
        checkTransform(forward, "input");
        return new GpuWeight(name, type, ne, rows, cols, rowBytes, shards, shardRowStart, shardRows, forward,
                inverseHadamard, scale, null, null);
    }

    /**
     * This tensor with the inverse transform applied to rows looked up from it (latent embedding table).
     *
     * @throws IllegalArgumentException when the transform does not fit the row width, or its block exceeds what
     *                                  the {@value HadamardSpec#KERNEL} kernel supports
     */
    public GpuWeight withInverseHadamard(HadamardSpec inverse) {
        Objects.requireNonNull(inverse, "inverse");
        checkTransform(inverse, "lookup");
        return new GpuWeight(name, type, ne, rows, cols, rowBytes, shards, shardRowStart, shardRows, hadamard,
                inverse, scale, inputTransform, inputPerm);
    }

    /**
     * This tensor with the full input transform chain of its matmul input; the tensor is left unchanged for
     * {@link InputTransform#IDENTITY}.
     *
     * @throws IllegalArgumentException when the transform does not fit the row width, its PERMUTE part lacks a
     *                                  device buffer, its block exceeds what the {@value HadamardSpec#KERNEL}
     *                                  kernel supports, or it would sit beside a {@link #withHadamard(HadamardSpec)
     *                                  directly attached} rotation
     */
    public GpuWeight withInputTransform(InputTransform t) {
        return withInputTransform(t, null);
    }

    /**
     * {@link #withInputTransform(InputTransform)} with the device int buffer of the {@code <base>.input_perm}
     * tensor the PERMUTE gather reads; the buffer belongs to whoever uploaded it.
     *
     * @throws IllegalArgumentException when {@code t} has a PERMUTE part and {@code perm} is null, or when a
     *                                  PERMUTE part is absent and {@code perm} is given
     */
    public GpuWeight withInputTransform(InputTransform t, GpuBuffer perm) {
        if (t == null || t.isIdentity()) {
            return this;
        }
        if (hadamard != null && t.hadamard() != null) {
            throw new IllegalArgumentException("tensor " + name + " already had a PRISM Hadamard transform attached;"
                    + " a " + t + " would rotate twice");
        }
        if (cols > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + name + ": row width " + cols
                    + " is too wide for an input transform");
        }
        t.requireWidth((int) cols);
        if (t.hadamard() != null) {
            checkTransform(t.hadamard(), "input");
        }
        if (t.hasPerm()) {
            if (perm == null) {
                throw new IllegalArgumentException("tensor " + name + ": its " + t + " needs the device buffer of"
                        + " the tensor " + permNameOf(name));
            }
            if ((long) cols * Integer.BYTES > perm.sizeInBytes()) {
                throw new IllegalArgumentException("tensor " + name + ": the input permutation needs "
                        + cols + " ints, its buffer holds " + perm.sizeInBytes() / Integer.BYTES);
            }
        } else if (perm != null) {
            throw new IllegalArgumentException("tensor " + name + " was given an input permutation buffer without"
                    + " a PERMUTE part");
        }
        return new GpuWeight(name, type, ne, rows, cols, rowBytes, shards, shardRowStart, shardRows, hadamard,
                inverseHadamard, scale, t, perm);
    }

    private void checkTransform(HadamardSpec spec, String role) {
        if (cols > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + name + ": row width " + cols + " is too wide for a Hadamard "
                    + role + " transform");
        }
        spec.checkWidth((int) cols);
        if (!spec.kernelSupported()) {
            throw new IllegalArgumentException("tensor " + name + ": PRISM Hadamard block " + spec.block()
                    + " exceeds the GPU limit of " + HadamardSpec.MAX_KERNEL_BLOCK + " (" + HadamardSpec.KERNEL
                    + "); this model's " + role + " transform has no GPU path");
        }
    }

    /**
     * This tensor with the per-tensor output scale {@code w_s}.
     *
     * @throws IllegalArgumentException for a non-finite scale
     */
    public GpuWeight withScale(float s) {
        if (!Float.isFinite(s)) {
            throw new IllegalArgumentException("tensor " + name + ": output scale " + s + " is not finite");
        }
        return new GpuWeight(name, type, ne, rows, cols, rowBytes, shards, shardRowStart, shardRows, hadamard,
                inverseHadamard, s, inputTransform, inputPerm);
    }

    public String name() { return name; }
    public GGMLType type() { return type; }
    /** ggml shape, {@code ne[0]} first (copy). */
    public long[] ne() { return ne.clone(); }
    public int rank() { return ne.length; }
    /** Output rows: the product of {@code ne[1..]}. */
    public long rows() { return rows; }
    /** Input width {@code ne[0]}. */
    public long cols() { return cols; }
    /** Bytes of one stored row. */
    public long rowBytes() { return rowBytes; }
    /** Bytes of the whole tensor. */
    public long sizeInBytes() { return rows * rowBytes; }
    public int shardCount() { return shards.length; }
    public GpuBuffer shard(int i) { return shards[checkShard(i)]; }
    /** First row held by shard {@code i}. */
    public long shardRowStart(int i) { return shardRowStart[checkShard(i)]; }
    /** Rows held by shard {@code i}. */
    public long shardRows(int i) { return shardRows[checkShard(i)]; }
    /** True when one buffer holds every row. */
    public boolean isSingleShard() { return shards.length == 1; }

    /**
     * Shard holding a row.
     *
     * @throws IndexOutOfBoundsException when the row is outside the tensor
     */
    public int shardOfRow(long row) {
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("tensor " + name + " has no row " + row + " (" + rows + " rows)");
        }
        int lo = 0;
        int hi = shards.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (shardRowStart[mid] <= row) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /** PRISM transform of the matmul input, or null. */
    public HadamardSpec hadamard() { return hadamard != null ? hadamard : transformHadamard(); }
    /** Transform of rows looked up from this table, or null. */
    public HadamardSpec inverseHadamard() { return inverseHadamard; }
    /** Per-tensor output scale; 1 when the tensor has none. */
    public float scale() { return scale; }
    /** True when outputs are multiplied by {@link #scale()}. */
    public boolean hasScale() { return scale != 1.0f; }

    /**
     * The full input transform chain of the matmul input (HADAMARD, ACT_QUANT, PERMUTE), or null: the transform the
     * uploader attaches over {@link #withInputTransform(InputTransform, GpuBuffer)}. Its HADAMARD part is what
     * {@link #hadamard()} returns.
     */
    public InputTransform inputTransform() { return inputTransform; }

    /** True when the matmul input takes a non-identity input transform. */
    public boolean hasInputTransform() { return inputTransform != null; }

    /** Device int buffer of the {@code <base>.input_perm} tensor, or null; belongs to its uploader. */
    public GpuBuffer inputPerm() { return inputPerm; }

    private HadamardSpec transformHadamard() { return inputTransform != null ? inputTransform.hadamard() : null; }

    /** The {@code .input_perm} sibling tensor name of a projection weight. */
    static String permNameOf(String weightName) {
        return (weightName.endsWith(".weight")
                ? weightName.substring(0, weightName.length() - ".weight".length()) : weightName)
                + ".input_perm";
    }

    private int checkShard(int i) {
        if (i < 0 || i >= shards.length) {
            throw new IndexOutOfBoundsException("tensor " + name + " has no shard " + i + " (" + shards.length
                    + " shards)");
        }
        return i;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("GpuWeight{").append(name).append(", ").append(type)
                .append(", ne=").append(Arrays.toString(ne));
        if (shards.length > 1) {
            sb.append(", shards=").append(Arrays.toString(shardRows));
        }
        if (hadamard() != null) {
            sb.append(", hadamard=").append(hadamard());
        }
        if (inverseHadamard != null) {
            sb.append(", inverseHadamard=").append(inverseHadamard);
        }
        if (inputTransform != null) {
            sb.append(", ").append(inputTransform);
        }
        if (hasScale()) {
            sb.append(", scale=").append(scale);
        }
        return sb.append('}').toString();
    }
}
