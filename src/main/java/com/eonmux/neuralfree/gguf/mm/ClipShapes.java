package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The shape checks of the clip loader (tools/mtmd/clip.cpp), shared by {@link ClipVisionModel},
 * {@link ClipAudioModel}, {@link ClipLayer} and {@link ClipProjectorTensors}: a mismatch fails naming the tensor,
 * its shape and the expected one. Static methods only.
 */
final class ClipShapes {

    private ClipShapes() {
    }

    /** The llama.cpp loader's exact shape check: listed dims match, every later dim is 1. */
    static void shape(TensorCatalog tc, TensorRef r, long... ne) {
        boolean ok = true;
        for (int i = ne.length; i < r.rank() && ok; i++) {
            ok = r.ne(i) == 1;
        }
        for (int i = 0; i < ne.length && ok; i++) {
            ok = r.ne(i) == ne[i];
        }
        if (!ok) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + r.name()
                    + " has shape " + r + ", the loader expects " + dims(ne));
        }
    }

    /** {@link #shape} when the tensor is present. */
    static void checkVector(TensorCatalog tc, TensorRef r, int n) {
        if (r != null) {
            shape(tc, r, n);
        }
    }

    /** A weight/bias pair that is absent together or present together with {@code n} values each. */
    static void checkPair(TensorCatalog tc, String base, TensorRef w, TensorRef b, int n) {
        if ((w == null) != (b == null)) {
            throw new MissingTensorException(tc.archName(), -1, (w == null ? base + ".weight" : base + ".bias"));
        }
        if (w != null) {
            shape(tc, w, n);
            shape(tc, b, n);
        }
    }

    /** A weight that may stand alone, and a bias that needs the weight; both {@code n} values. */
    static void checkNorm(TensorCatalog tc, String base, TensorRef w, TensorRef b, int n) {
        if (w == null && b != null) {
            throw new MissingTensorException(tc.archName(), -1, base + ".weight");
        }
        checkVector(tc, w, n);
        checkVector(tc, b, n);
    }

    /** A tensor holding exactly {@code n} elements in any layout (a bias the graph reshapes). */
    static void elements(TensorCatalog tc, TensorRef r, long n) {
        long have = 1;
        for (int i = 0; i < r.rank(); i++) {
            have *= r.ne(i);
        }
        if (have != n) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + r.name()
                    + " has shape " + r + ", the loader expects " + n + " elements");
        }
    }

    static String dims(long... ne) {
        StringBuilder want = new StringBuilder("{");
        for (int i = 0; i < ne.length; i++) {
            want.append(i == 0 ? "" : ", ").append(ne[i]);
        }
        return want.append('}').toString();
    }
}
