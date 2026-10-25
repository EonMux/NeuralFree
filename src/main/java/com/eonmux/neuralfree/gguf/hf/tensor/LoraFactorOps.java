package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * The tensor transforms applied to a LoRA factor pair {@code W = B @ A} ({@link LoraFactorSource}), following
 * llama.cpp's {@code LoraTorchTensor} (LC/convert_lora_to_gguf.py:40-270):
 * <ul>
 *   <li>moves of the output rows of each matrix, and output-side scaling, act on {@code B};</li>
 *   <li>moves and scaling of the innermost (input) columns act on {@code A};</li>
 *   <li>moves of a batch (expert) dimension act on both, or on {@code B} alone when {@code A} is shared;</li>
 *   <li>stacking stacks both factors (one shared {@code A} when every part has the same {@code A});</li>
 *   <li>concatenation keeps a common {@code A} (rows) or {@code B} (columns); different factors concatenate exactly
 *       as a block-diagonal pair whose rank is the sum of the parts' ranks;</li>
 *   <li>{@code transposeLast2} swaps the roles, {@code A' = B^T} and {@code B' = A^T};</li>
 *   <li>{@code convert} converts both factors.</li>
 * </ul>
 * Every other transform throws {@link #unsupported}.
 */
final class LoraFactorOps {

    private LoraFactorOps() {
    }

    /** The error for a transform a low-rank factorization cannot express. */
    static UnsupportedOperationException unsupported(RowSource src, String op) {
        return new UnsupportedOperationException("LoRA delta of " + src.describe() + " needs " + op
                + ", which a low-rank factorization cannot express");
    }

    /** {@code f} applied to {@code B}: a move or scaling of the output rows of each matrix. */
    static RowSource onB(LoraFactorSource l, UnaryOperator<RowSource> f) {
        return pair(l, l.a(), f.apply(l.b()));
    }

    /** {@code f} applied to {@code A}: a move or scaling of the innermost columns. */
    static RowSource onA(LoraFactorSource l, UnaryOperator<RowSource> f) {
        return pair(l, f.apply(l.a()), l.b());
    }

    /** A move of dim 0: the output rows of a 2-D delta, the batch entries of a stacked one. */
    static RowSource onDim0(LoraFactorSource l, UnaryOperator<RowSource> f) {
        if (l.shape().length == 2) {
            return onB(l, f);
        }
        RowSource b = f.apply(l.b());
        RowSource a = l.sharesA() ? l.a() : f.apply(l.a());
        return pair(l, a, b);
    }

    /** {@code W.narrow(d, from, count)}. */
    static RowSource sliceDim(LoraFactorSource l, int d, long from, long count) {
        int n = l.shape().length;
        if (d == n - 1) {
            return onA(l, a -> RowTransforms.sliceDim(a, -1, from, count));
        }
        if (d == n - 2) {
            return onB(l, b -> RowTransforms.sliceDim(b, d, from, count));
        }
        RowSource b = RowTransforms.sliceDim(l.b(), d, from, count);
        RowSource a = l.a().shape()[d] == 1 ? l.a() : RowTransforms.sliceDim(l.a(), d, from, count);
        return pair(l, a, b);
    }

    /** {@code torch.cat(parts, dim=0)}. */
    static RowSource concatRows(List<RowSource> parts) {
        List<LoraFactorSource> ls = factors(parts, "concatRows");
        LoraFactorSource l0 = ls.get(0);
        if (sameA(ls) && (l0.shape().length == 2 || l0.sharesA())) {
            List<RowSource> bs = new ArrayList<>();
            ls.forEach(l -> bs.add(l.b()));
            return pair(l0, l0.a(), RowTransforms.concatRows(f32IfMixed(bs)));
        }
        if (l0.shape().length > 2) {
            List<RowSource> as = new ArrayList<>();
            List<RowSource> bs = new ArrayList<>();
            for (LoraFactorSource l : ls) {
                if (l.sharesA() || l.rank() != l0.rank()) {
                    throw unsupported(l, "a batch concatenation of factors with different shared A or ranks");
                }
                as.add(l.a());
                bs.add(l.b());
            }
            return pair(l0, RowTransforms.concatRows(f32IfMixed(as)), RowTransforms.concatRows(f32IfMixed(bs)));
        }
        int total = totalRank(ls);
        List<RowSource> as = new ArrayList<>();
        List<RowSource> bs = new ArrayList<>();
        int before = 0;
        for (LoraFactorSource l : ls) {
            long out = l.b().shape()[0];
            bs.add(ColumnTransforms.concatCols(paddedCols(l.b(), out, before, total)));
            as.add(ElementTransforms.convert(l.a(), GGMLType.F32));
            before += l.rank();
        }
        return pair(l0, RowTransforms.concatRows(as), RowTransforms.concatRows(bs));
    }

    /** {@code torch.cat(parts, dim=-1)}. */
    static RowSource concatCols(List<RowSource> parts) {
        List<LoraFactorSource> ls = factors(parts, "concatCols");
        LoraFactorSource l0 = ls.get(0);
        boolean sameB = true;
        for (LoraFactorSource l : ls) {
            sameB &= l.b() == l0.b();
        }
        if (sameB) {
            List<RowSource> as = new ArrayList<>();
            ls.forEach(l -> as.add(l.a()));
            return pair(l0, ColumnTransforms.concatCols(f32IfMixed(as)), l0.b());
        }
        if (l0.shape().length > 2) {
            throw unsupported(l0, "a column concatenation of batched factors with different B");
        }
        long totalIn = 0;
        for (LoraFactorSource l : ls) {
            totalIn += l.a().shape()[1];
        }
        List<RowSource> as = new ArrayList<>();
        List<RowSource> bs = new ArrayList<>();
        long before = 0;
        for (LoraFactorSource l : ls) {
            long in = l.a().shape()[1];
            as.add(ColumnTransforms.concatCols(paddedCols(l.a(), l.rank(), before, totalIn)));
            bs.add(ElementTransforms.convert(l.b(), GGMLType.F32));
            before += in;
        }
        return pair(l0, RowTransforms.concatRows(as), ColumnTransforms.concatCols(bs));
    }

    /** {@code torch.stack(parts, dim=0)}. */
    static RowSource stack(List<RowSource> parts) {
        List<LoraFactorSource> ls = factors(parts, "stack");
        LoraFactorSource l0 = ls.get(0);
        List<RowSource> bs = new ArrayList<>();
        ls.forEach(l -> bs.add(l.b()));
        RowSource b = ExpertTransforms.stack(f32IfMixed(bs));
        if (sameA(ls)) {
            long[] as = l0.a().shape();
            long[] shared = new long[as.length + 1];
            shared[0] = 1;
            System.arraycopy(as, 0, shared, 1, as.length);
            return pair(l0, ElementTransforms.reshape(l0.a(), shared), b);
        }
        List<RowSource> as = new ArrayList<>();
        for (LoraFactorSource l : ls) {
            if (l.sharesA() && l.a().shape().length > 2) {
                throw unsupported(l, "a stack of factors with different shared A");
            }
            as.add(l.a());
        }
        return pair(l0, ExpertTransforms.stack(f32IfMixed(as)), b);
    }

    /** {@code W.transpose(-1, -2)}: {@code A' = B^T}, {@code B' = A^T}. */
    static RowSource transposeLast2(LoraFactorSource l) {
        if (l.shape().length > 2 && l.sharesA()) {
            throw unsupported(l, "a transpose of batched factors sharing one A");
        }
        return pair(l, ExpertTransforms.transposeLast2(l.b()), ExpertTransforms.transposeLast2(l.a()));
    }

    /** Both factors converted to {@code target}. */
    static RowSource convert(LoraFactorSource l, GGMLType target) {
        return pair(l, ElementTransforms.convert(l.a(), target), ElementTransforms.convert(l.b(), target));
    }

    /**
     * {@code W.reshape(shape)} with the innermost dimension unchanged: {@code B} takes the new outer shape and a shared
     * {@code A} gets size-1 batch dimensions.
     */
    static RowSource reshape(LoraFactorSource l, long[] shape) {
        long[] cur = l.shape();
        if (shape.length < 2 || shape[shape.length - 1] != cur[cur.length - 1]) {
            throw unsupported(l, "a reshape to " + Arrays.toString(shape) + " that changes its input dimension");
        }
        int n = shape.length;
        long[] bShape = Arrays.copyOf(shape, n);
        bShape[n - 1] = l.rank();
        long[] as = l.a().shape();
        long[] aShape = new long[n];
        Arrays.fill(aShape, 1L);
        aShape[n - 2] = l.rank();
        aShape[n - 1] = as[as.length - 1];
        if (!l.sharesA()) {
            System.arraycopy(shape, 0, aShape, 0, n - 2);
            if (!Arrays.equals(Arrays.copyOf(aShape, n - 2), Arrays.copyOf(as, as.length - 2))) {
                throw unsupported(l, "a reshape of its batch dimensions to " + Arrays.toString(shape));
            }
        }
        return pair(l, ElementTransforms.reshape(l.a(), aShape), ElementTransforms.reshape(l.b(), bShape));
    }

    private static RowSource pair(LoraFactorSource l, RowSource a, RowSource b) {
        return new LoraFactorSource(l.describe(), a, b);
    }

    private static List<LoraFactorSource> factors(List<RowSource> parts, String op) {
        List<LoraFactorSource> ls = new ArrayList<>();
        for (RowSource p : parts) {
            if (!(p instanceof LoraFactorSource)) {
                throw new IllegalArgumentException(op + " mixes the LoRA delta " + parts.get(0).describe()
                        + " with the dense tensor " + p.describe());
            }
            ls.add((LoraFactorSource) p);
        }
        return ls;
    }

    private static boolean sameA(List<LoraFactorSource> ls) {
        RowSource a0 = ls.get(0).a();
        for (LoraFactorSource l : ls) {
            if (l.a() != a0) {
                return false;
            }
        }
        return true;
    }

    private static int totalRank(List<LoraFactorSource> ls) {
        int total = 0;
        for (LoraFactorSource l : ls) {
            total = Math.addExact(total, l.rank());
        }
        return total;
    }

    /** {@code [0 (rows x before) | m | 0 (rows x (total - before - cols))]} in F32. */
    private static List<RowSource> paddedCols(RowSource m, long rows, long before, long total) {
        long cols = m.shape()[m.shape().length - 1];
        List<RowSource> out = new ArrayList<>();
        if (before > 0) {
            out.add(RopeTensors.constant(new long[]{rows, before}, 0f));
        }
        out.add(ElementTransforms.convert(m, GGMLType.F32));
        long after = total - before - cols;
        if (after > 0) {
            out.add(RopeTensors.constant(new long[]{rows, after}, 0f));
        }
        return out;
    }

    /** The parts unchanged when their formats agree, else all converted to F32. */
    private static List<RowSource> f32IfMixed(List<RowSource> parts) {
        ElementFormat f0 = parts.get(0).format();
        boolean same = true;
        for (RowSource p : parts) {
            same &= p.format().equals(f0);
        }
        if (same) {
            return parts;
        }
        List<RowSource> out = new ArrayList<>();
        parts.forEach(p -> out.add(ElementTransforms.convert(p, GGMLType.F32)));
        return out;
    }
}
