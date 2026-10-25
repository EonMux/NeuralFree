package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * The delta {@code W = B @ A} of a PEFT LoRA module kept as its two factors, a port of llama.cpp's
 * {@code LoraTorchTensor} (LC/convert_lora_to_gguf.py:40-270): {@code A} is {@code [..., r, in]} and {@code B} is
 * {@code [..., out, r]}, with equal ranks and leading (batch) dimensions that are equal, or all 1 in {@code A} to share
 * one {@code A} across the batch. The delta's shape is {@code (*B.shape[:-1], A.shape[-1])}.
 *
 * <p>Catalogue transforms given a factor pair dispatch to {@link LoraFactorOps}, which applies them to the factors.
 * The planner stores the factors as {@code <name>.lora_a} and {@code <name>.lora_b}; reading the pair's rows yields the
 * dense delta in F32 and reads all of {@code A} per call, so it is not row-local.</p>
 */
public final class LoraFactorSource implements RowSource {

    private final String name;
    private final RowSource a;
    private final RowSource b;
    private final long[] shape;

    /**
     * @param name what the delta is, for messages (the HF module name)
     * @throws IllegalArgumentException when the factor shapes do not compose
     */
    public LoraFactorSource(String name, RowSource a, RowSource b) {
        this.name = Objects.requireNonNull(name, "name");
        this.a = Objects.requireNonNull(a, "a");
        this.b = Objects.requireNonNull(b, "b");
        long[] as = a.shape();
        long[] bs = b.shape();
        int n = as.length;
        if (n < 2 || bs.length != n || as[n - 2] != bs[n - 1]) {
            throw new IllegalArgumentException(name + ": LoRA factors A " + Arrays.toString(as) + " and B "
                    + Arrays.toString(bs) + " do not compose as B @ A");
        }
        boolean shared = true;
        boolean equal = true;
        for (int i = 0; i < n - 2; i++) {
            shared &= as[i] == 1;
            equal &= as[i] == bs[i];
        }
        if (!shared && !equal) {
            throw new IllegalArgumentException(name + ": batch dimensions of A " + Arrays.toString(as)
                    + " neither match B " + Arrays.toString(bs) + " nor are all 1");
        }
        if (as[n - 2] > Integer.MAX_VALUE || as[n - 1] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + ": LoRA factor A " + Arrays.toString(as) + " is too large");
        }
        this.shape = Arrays.copyOf(bs, n);
        this.shape[n - 1] = as[n - 1];
    }

    /** The factor {@code A}, {@code [..., r, in]}. */
    public RowSource a() {
        return a;
    }

    /** The factor {@code B}, {@code [..., out, r]}. */
    public RowSource b() {
        return b;
    }

    /** The rank {@code r}. */
    public int rank() {
        long[] as = a.shape();
        return (int) as[as.length - 2];
    }

    /** True when one {@code A} is shared by every batch entry (its batch dimensions are all 1). */
    boolean sharesA() {
        long[] as = a.shape();
        for (int i = 0; i < as.length - 2; i++) {
            if (as[i] != 1) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String describe() {
        return name;
    }

    @Override
    public ElementFormat format() {
        return ElementFormat.of(GGMLType.F32);
    }

    @Override
    public long[] shape() {
        return shape.clone();
    }

    @Override
    public long rows() {
        long r = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            r = Math.multiplyExact(r, shape[i]);
        }
        return r;
    }

    @Override
    public long rowBytes() {
        return 4L * shape[shape.length - 1];
    }

    @Override
    public boolean rowLocal() {
        return false;
    }

    @Override
    public void readRows(long first, int count, ByteBuffer dst) throws IOException {
        long rows = rows();
        if (first < 0 || count < 0 || first > rows - count) {
            throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside the " + rows
                    + " rows of the LoRA delta " + name);
        }
        int in = (int) shape[shape.length - 1];
        int r = rank();
        if ((long) count * rowBytes() > dst.remaining()) {
            throw new IllegalArgumentException(count + " rows of the LoRA delta " + name + " need "
                    + (long) count * rowBytes() + " bytes; the buffer has " + dst.remaining());
        }
        RowSource af = ElementTransforms.dequant(a, GGMLType.F32);
        RowSource bf = ElementTransforms.dequant(b, GGMLType.F32);
        float[] av = readAll(af);
        long out = shape[shape.length - 2];
        int aMatrices = (int) (af.rows() / r);
        int chunk = Math.max(1, RowTransforms.chunkRows(rowBytes()));
        ByteBuffer le = dst.slice().order(ByteOrder.LITTLE_ENDIAN);
        for (int done = 0; done < count; ) {
            long r0 = first + done;
            long batch = r0 / out;
            int c = (int) Math.min(Math.min(chunk, count - done), (batch + 1) * out - r0);
            float[] bv = new float[c * r];
            RowTransforms.readRows(bf, r0, c).asFloatBuffer().get(bv);
            float[] v = new float[c * in];
            float[] am = aMatrices == 1 ? av
                    : Arrays.copyOfRange(av, (int) (batch * r * in), (int) ((batch + 1) * r * in));
            multiplyAdd(bv, am, c, r, in, 1f, v);
            for (float x : v) {
                le.putFloat(x);
            }
            done += c;
        }
        dst.position(dst.position() + (int) ((long) count * rowBytes()));
    }

    @Override
    public String toString() {
        return "LoraFactorSource{" + name + ", A " + Arrays.toString(a.shape()) + ", B " + Arrays.toString(b.shape())
                + "}";
    }

    /** All values of an F32 source, row-major. */
    static float[] readAll(RowSource f32) throws IOException {
        long n = Math.multiplyExact(f32.rows(), f32.rowBytes()) / 4;
        if (n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(f32.describe() + " is too large to read whole");
        }
        float[] v = new float[(int) n];
        if (n > 0) {
            RowTransforms.readRows(f32, 0, Math.toIntExact(f32.rows())).asFloatBuffer().get(v);
        }
        return v;
    }

    /**
     * {@code v[count, in] += scale * bv[count, r] @ av[r, in]}, accumulating each element in f32 in rank order.
     */
    static void multiplyAdd(float[] bv, float[] av, int count, int r, int in, float scale, float[] v) {
        float[] acc = new float[in];
        for (int i = 0; i < count; i++) {
            Arrays.fill(acc, 0f);
            for (int k = 0; k < r; k++) {
                float bk = bv[i * r + k];
                int ao = k * in;
                for (int j = 0; j < in; j++) {
                    acc[j] += bk * av[ao + j];
                }
            }
            int vo = i * in;
            for (int j = 0; j < in; j++) {
                v[vo + j] += scale * acc[j];
            }
        }
    }
}
