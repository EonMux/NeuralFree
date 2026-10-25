package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * The bytes of one tensor of a Hugging Face checkpoint view, served row by row: a tensor stored in a checkpoint file,
 * or a producer that derives one (a permutation, a split, a stack, a repack into a ggml block type) from other row
 * sources. All tensor data of the view flows through this interface.
 *
 * <p>Contract:</p>
 * <ul>
 *   <li>{@link #shape()} is in PyTorch order (outermost first); the GGUF {@code ne} is its reverse. A row is one
 *       innermost run of elements: {@link #rows()} is the product of every dimension but the innermost (1 for a
 *       rank-0 or rank-1 tensor) and {@link #rowBytes()} is the bytes of one row in {@link #format()}, which for a
 *       ggml block type is {@code innermost / blockSize * typeSize}.</li>
 *   <li>{@link #readRows} writes exactly {@code count * rowBytes()} bytes of rows {@code [first, first + count)}.</li>
 *   <li>Implementations are immutable, stateless between calls and thread-safe; a producer allocates its scratch per
 *       call.</li>
 *   <li>{@link #rowLocal()} is true when reading rows {@code [r0, r1)} reads only {@code O(r1 - r0)} rows of each
 *       input. Producers that need the whole input for any row (K-major unpacking, transposes of the last two
 *       dimensions, splits along the inner dimension of a transposed source) are not row-local, and readers
 *       materialize them whole.</li>
 *   <li>A tensor stored unchanged in a checkpoint file has a {@link #directVirtualOffset()} in the view's virtual
 *       address space, so its bytes are mapped in place instead of produced.</li>
 *   <li>Quantized checkpoints attach the weight's sibling values: the output multiplier ({@link #outputScale()},
 *       written as {@code <base>.scale}), the activation dequant multiplier ({@link #inputScale()},
 *       {@code <base>.input_scale}), the activation zero point ({@link #inputZeroPoint()},
 *       {@code <base>.input_zero_point}), the act-order input permutation ({@link #inputPerm()},
 *       {@code <base>.input_perm}) and the activation scheme ({@link #actQuant()}). Where a weight has no sibling
 *       slot, the planner folds the permutation or the output scale into the weight
 *       ({@link #foldInputPerm()}, {@link #foldOutputScale()}).</li>
 * </ul>
 */
public interface RowSource {

    /** What this tensor is and where it comes from, for messages. */
    String describe();

    /** The element encoding of the bytes {@link #readRows} writes. */
    ElementFormat format();

    /** The shape, outermost dimension first (a copy). */
    long[] shape();

    /** The number of rows: the product of every dimension but the innermost; 1 for rank 0 and 1. */
    long rows();

    /** Bytes of one row. */
    long rowBytes();

    /** True when reading a range of rows reads a proportional range of every input. */
    boolean rowLocal();

    /**
     * Writes rows {@code [first, first + count)}, {@code count * rowBytes()} bytes, at {@code dst}'s position and
     * advances the position past them.
     *
     * @throws IllegalArgumentException when the rows lie outside the tensor or {@code dst} has fewer bytes remaining
     * @throws IOException              when the underlying data cannot be read or is invalid
     */
    void readRows(long first, int count, ByteBuffer dst) throws IOException;

    /** The virtual offset of the bytes stored unchanged in a checkpoint file, or -1 when they are produced. */
    default long directVirtualOffset() {
        return -1;
    }

    /** The output multiplier: 1 value, or 1 per expert of a stacked tensor; null when there is none. */
    default float[] outputScale() {
        return null;
    }

    /** The activation dequant multiplier: 1 value, or 1 per expert; null when there is none. */
    default float[] inputScale() {
        return null;
    }

    /** The activation zero point: 1 value, or 1 per expert; null when there is none. */
    default int[] inputZeroPoint() {
        return null;
    }

    /** The act-order permutation of the input dimension (length {@code K}); null when there is none. */
    default int[] inputPerm() {
        return null;
    }

    /** The checkpoint's activation quantization scheme for this weight; null for weight-only. */
    default ActQuantScheme actQuant() {
        return null;
    }

    /**
     * This tensor with {@link #inputPerm()} undone in the weight itself (the input dimension restored to its original
     * order), and no input permutation.
     *
     * @throws UnsupportedOperationException when there is no input permutation
     */
    default RowSource foldInputPerm() {
        throw new UnsupportedOperationException(describe() + " has no input permutation to fold");
    }

    /**
     * This tensor with {@link #outputScale()} multiplied into the weight itself, and no output scale.
     *
     * @throws UnsupportedOperationException when there is no output scale
     */
    default RowSource foldOutputScale() {
        throw new UnsupportedOperationException(describe() + " has no output scale to fold");
    }
}
