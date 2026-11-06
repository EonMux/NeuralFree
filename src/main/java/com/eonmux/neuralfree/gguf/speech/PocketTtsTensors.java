package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * Shape and naming checks shared by the pockettts generator graphs. Tensor shapes are read in ggml dimension order
 * ({@code ne[0]} first), the order the clip generator file stores them in.
 */
final class PocketTtsTensors {

    private PocketTtsTensors() {
    }

    /** A positive tensor dimension as an {@code int}. */
    static int dim(TensorRef ref, int i) {
        long v = ref.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("pockettts: tensor " + ref.name() + " dimension " + i + " is " + v);
        }
        return (int) v;
    }

    /** Requires {@code ref} to have {@code ne} as its leading dimensions and 1 in every further one. */
    static void shape(TensorRef ref, long... ne) {
        boolean ok = ref.rank() >= ne.length;
        for (int i = 0; i < ne.length && ok; i++) {
            ok = ref.ne(i) == ne[i];
        }
        for (int i = ne.length; i < ref.rank() && ok; i++) {
            ok = ref.ne(i) == 1;
        }
        if (!ok) {
            throw new GgufFormatException("pockettts: tensor " + ref.name() + " has shape " + ref + ", expected "
                    + java.util.Arrays.toString(ne) + " (ggml order)");
        }
    }

    /** Requires the bound matrix {@code ref} to be {@code rows} x {@code cols}. */
    static void matrix(CpuTensorRef ref, long cols, long rows) {
        if (ref.cols() != cols || ref.rows() != rows) {
            throw new GgufFormatException("pockettts: tensor " + ref + " is " + ref.rows() + " x " + ref.cols()
                    + ", expected " + rows + " x " + cols);
        }
    }

    /** Requires a vector of {@code n} floats. */
    static void vector(float[] v, int n, String name) {
        if (v == null || v.length != n) {
            throw new GgufFormatException("pockettts: " + name + " must hold " + n + " floats, got "
                    + (v == null ? "none" : v.length));
        }
    }

    /** Kernel width of a convolution weight {@code [K, IC, OC]} (ggml order). */
    static int convKernel(TensorRef ref) {
        return dim(ref, 0);
    }

    /** Input channels of a convolution weight {@code [K, IC, OC]}; 1 for a weight without a channel axis. */
    static int convIn(TensorRef ref) {
        return ref.rank() < 3 ? 1 : dim(ref, 1);
    }

    /** Output channels of a convolution weight {@code [K, IC, OC]}; the rows for a weight without a channel axis. */
    static int convOut(TensorRef ref) {
        return ref.rank() < 3 ? dim(ref, 1) : dim(ref, 2);
    }

    /** One plus the highest {@code N} among the tensors named {@code prefix + N + ".*"}; 0 when there are none. */
    static int countIndexed(TensorCatalog tc, String prefix) {
        int highest = -1;
        for (String name : tc.names()) {
            if (!name.startsWith(prefix)) {
                continue;
            }
            int end = name.indexOf('.', prefix.length());
            if (end < 0) {
                continue;
            }
            int value = 0;
            boolean digits = true;
            for (int i = prefix.length(); i < end; i++) {
                char c = name.charAt(i);
                if (c < '0' || c > '9' || value > (Integer.MAX_VALUE - 9) / 10) {
                    digits = false;
                    break;
                }
                value = value * 10 + (c - '0');
            }
            if (digits) {
                highest = Math.max(highest, value);
            }
        }
        return highest + 1;
    }
}
