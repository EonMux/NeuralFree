package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Device data movement against the CPU path, bit for bit: contiguous copies, strided copies (including rank 7),
 * copies into views, zeros, and every device cast, on each GPU API.
 */
@RunWith(Parameterized.class)
public class GpuCoreOpsParityTest {

    private static final float[] EDGE = {
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 2.5f, -2.5f, 123.456f, 65504.0f, 65519.0f, 65520.0f, 70000.0f,
            -1e10f, 1e-8f, 5.96e-8f, 3e-5f, 6.1e-5f, Float.MIN_NORMAL, Float.MIN_VALUE, -Float.MIN_VALUE,
            Float.MAX_VALUE, -Float.MAX_VALUE, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN,
            Float.intBitsToFloat(0xffc00000), Float.intBitsToFloat(0x7f800001), 2.1e9f, -2.2e9f, 2147483520.0f,
            -2147483648.0f, 1.0f + Math.scalb(1.0f, -11), Float.intBitsToFloat(0x3f808000),
            Float.intBitsToFloat(0x3f818000), Math.scalb(1.0f, -25), 3 * Math.scalb(1.0f, -25)
    };

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuCoreOpsParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
        gpu = session.backend();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        try {
            for (Tensor t : owned) {
                t.close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private static float[] values(int n, long seed) {
        float[] random = TrainParity.uniform(seed, n, -3e4f, 3e4f);
        for (int i = 0; i < Math.min(n, EDGE.length); i++) {
            random[i] = EDGE[i];
        }
        return random;
    }

    private Tensor floats(ComputeBackend b, float[] data, int... dims) {
        return keep(Tensor.fromArray(b, data).reshape(Shape.of(dims)));
    }

    private Tensor raw(ComputeBackend b, byte[] bytes, DType dtype, int... dims) {
        TensorStorage st = TensorStorage.fromHost(b, ByteBuffer.wrap(bytes), dtype);
        return keep(Tensor.fromStorage(b, Shape.of(dims), dtype, st));
    }

    private static byte[] bytes(Tensor t) {
        ByteBuffer bb = t.toByteBuffer();
        byte[] out = new byte[bb.remaining()];
        bb.get(out);
        return out;
    }

    private void assertSameBytes(String what, Tensor ref, Tensor dev) {
        assertEquals(what + ": shape", ref.shape(), dev.shape());
        assertEquals(what + ": dtype", ref.dtype(), dev.dtype());
        TrainParity.assertBitIdentical(what, bytes(ref), bytes(dev));
    }

    // ------------------------------------------------------------------ copies

    @Test
    public void contiguousCopyOfTranspose() {
        float[] data = values(37 * 53, 1);
        Tensor ref = keep(floats(host, data, 37, 53).transpose().contiguous());
        Tensor dev = keep(GpuCoreOps.contiguous(floats(gpu, data, 37, 53).transpose()));
        assertNotNull(dev);
        assertTrue(dev.isContiguous());
        assertSameBytes("transpose copy", ref, dev);
    }

    @Test
    public void contiguousCopyOfInt32Slice() {
        int[] data = new int[41 * 9];
        for (int i = 0; i < data.length; i++) {
            data[i] = i * 7919 - 100_000;
        }
        Tensor ref = keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(41, 9)).slice(1, 2, 7).contiguous());
        Tensor dev = keep(keep(Tensor.fromArray(gpu, data)).reshape(Shape.of(41, 9)).slice(1, 2, 7).contiguous());
        assertSameBytes("int32 slice copy", ref, dev);
    }

    /** 8-byte elements move as their two 32-bit words; the bit patterns include NaN payloads as FLOAT64. */
    @Test
    public void contiguousCopyOfEightByteViews() {
        byte[] b = longBytes(35, 6);
        for (DType dtype : new DType[]{DType.INT64, DType.FLOAT64}) {
            Tensor ref = keep(raw(host, b, dtype, 7, 5).transpose().slice(0, 1, 4).contiguous());
            Tensor dev = keep(GpuCoreOps.contiguous(raw(gpu, b, dtype, 7, 5).transpose().slice(0, 1, 4)));
            assertNotNull(dtype + " device copy", dev);
            assertTrue(dev.isContiguous());
            assertSameBytes(dtype + " strided copy", ref, dev);
            assertSameBytes(dtype + " expand copy",
                    keep(raw(host, b, dtype, 35, 1).expand(Shape.of(2, 35, 3)).contiguous()),
                    keep(raw(gpu, b, dtype, 35, 1).expand(Shape.of(2, 35, 3)).contiguous()));
        }
    }

    /** Native-order bytes of {@code n} longs covering sign, high-word and all-ones patterns. */
    private static byte[] longBytes(int n, long seed) {
        long[] edge = {0L, 1L, -1L, 1L << 33, (1L << 32) + 5, Long.MAX_VALUE, Long.MIN_VALUE, 0x7ff8000000000001L,
                0xfff0000000000000L};
        Random random = new Random(seed);
        ByteBuffer bb = ByteBuffer.allocate(n * 8).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            bb.putLong(i * 8, i < edge.length ? edge[i] : random.nextLong());
        }
        return bb.array();
    }

    @Test
    public void denseOffsetViewCopiesWithTheFastKernel() {
        float[] data = values(1000, 2);
        Tensor ref = keep(floats(host, data, 1000).slice(0, 333, 999).contiguous());
        Tensor dev = keep(floats(gpu, data, 1000).slice(0, 333, 999).contiguous());
        assertSameBytes("dense offset copy", ref, dev);
    }

    @Test
    public void stridedCopyOfExpand() {
        float[] data = values(17, 3);
        Tensor ref = keep(floats(host, data, 17, 1).expand(Shape.of(3, 17, 29)).contiguous());
        Tensor dev = keep(floats(gpu, data, 17, 1).expand(Shape.of(3, 17, 29)).contiguous());
        assertSameBytes("expand copy", ref, dev);
    }

    @Test
    public void stridedCopyOfRankSevenPermutation() {
        int[] dims = {2, 3, 2, 3, 2, 3, 5};
        float[] data = values(2 * 3 * 2 * 3 * 2 * 3 * 5, 4);
        int[] perm = {6, 0, 5, 1, 4, 2, 3};
        Tensor ref = keep(floats(host, data, dims).transpose(perm).contiguous());
        Tensor dev = keep(floats(gpu, data, dims).transpose(perm).contiguous());
        assertSameBytes("rank 7 copy", ref, dev);
    }

    @Test
    public void readbackOfStridedViews() {
        float[] data = values(64 * 48, 5);
        Tensor refView = floats(host, data, 64, 48).transpose().slice(0, 5, 40).slice(1, 3, 61);
        Tensor devView = floats(gpu, data, 64, 48).transpose().slice(0, 5, 40).slice(1, 3, 61);
        TrainParity.assertBitIdentical("strided readback", refView.toFloatArray(), devView.toFloatArray());
    }

    @Test
    public void copyIntoViewBroadcastsTheSource() {
        float[] base = values(12 * 20, 6);
        float[] row = values(20, 7);
        Tensor refDst = floats(host, base, 12, 20);
        Tensor devDst = floats(gpu, base, 12, 20);
        Tensor devSrc = floats(gpu, row, 20);
        GpuCoreOps.copyInto(devSrc, devDst.slice(0, 3, 9));
        float[] expected = refDst.toFloatArray();
        for (int i = 3; i < 9; i++) {
            System.arraycopy(row, 0, expected, i * 20, 20);
        }
        TrainParity.assertBitIdentical("copyInto", expected, devDst.toFloatArray());
    }

    @Test
    public void copyIntoTransposedViewOfTheSameBuffer() {
        float[] base = values(16 * 16, 8);
        Tensor devDst = floats(gpu, base, 16, 16);
        GpuCoreOps.copyInto(devDst, devDst.transpose());
        float[] expected = new float[256];
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                expected[j * 16 + i] = base[i * 16 + j];
            }
        }
        TrainParity.assertBitIdentical("copyInto aliasing", expected, devDst.toFloatArray());
    }

    @Test
    public void zerosOnTheDevice() {
        for (DType d : new DType[]{DType.FLOAT32, DType.INT32, DType.BOOL}) {
            Tensor dev = keep(GpuCoreOps.zeros(gpu, Shape.of(7, 13), d));
            assertNotNull(dev);
            assertSameBytes("zeros " + d, keep(Tensor.zeros(host, Shape.of(7, 13), d)), dev);
        }
        assertNull(GpuCoreOps.zeros(gpu, Shape.of(3), DType.FLOAT64));
    }

    // ------------------------------------------------------------------ casts

    private void castBoth(String what, Tensor refSrc, Tensor devSrc, DType target) {
        Tensor dev = keep(GpuCoreOps.cast(devSrc, target));
        assertNotNull(what + ": device cast", dev);
        Tensor ref = keep(refSrc.to(target));
        assertSameBytes(what, ref, dev);
        assertSameBytes(what + " via Tensor.to", ref, keep(devSrc.to(target)));
    }

    @Test
    public void castsFromFloat32() {
        for (int n : new int[]{1, 2, 3, 5, 1027, EDGE.length}) {
            float[] data = values(n, 10 + n);
            for (DType target : new DType[]{DType.INT32, DType.BOOL, DType.FLOAT16, DType.BFLOAT16}) {
                castBoth("F32->" + target + " n=" + n, floats(host, data, n), floats(gpu, data, n), target);
            }
        }
    }

    @Test
    public void castsFromFloat32Views() {
        float[] data = values(30 * 7, 11);
        for (DType target : new DType[]{DType.INT32, DType.BOOL, DType.FLOAT16, DType.BFLOAT16}) {
            castBoth("F32 slice->" + target, floats(host, data, 30 * 7).slice(0, 3, 200),
                    floats(gpu, data, 30 * 7).slice(0, 3, 200), target);
            castBoth("F32 transpose->" + target, floats(host, data, 30, 7).transpose(),
                    floats(gpu, data, 30, 7).transpose(), target);
        }
    }

    @Test
    public void castInt32ToFloat32() {
        int[] data = {0, 1, -1, 16777217, -16777217, Integer.MAX_VALUE, Integer.MIN_VALUE, 123456789, 7};
        castBoth("I32->F32", keep(Tensor.fromArray(host, data)), keep(Tensor.fromArray(gpu, data)), DType.FLOAT32);
    }

    @Test
    public void castsToFloat32FromSubWordTypes() {
        int n = 1031;
        byte[] bools = new byte[n];
        ByteBuffer halves = ByteBuffer.allocate(2 * n).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            bools[i] = (byte) (i % 3 == 0 ? 0 : (i % 3 == 1 ? 1 : 0x7f));
            halves.putShort(i * 2, (short) (i * 64 + i));
        }
        castBoth("BOOL->F32", raw(host, bools, DType.BOOL, n), raw(gpu, bools, DType.BOOL, n), DType.FLOAT32);
        byte[] h = halves.array();
        castBoth("F16->F32", raw(host, h, DType.FLOAT16, n), raw(gpu, h, DType.FLOAT16, n), DType.FLOAT32);
        castBoth("BF16->F32", raw(host, h, DType.BFLOAT16, n), raw(gpu, h, DType.BFLOAT16, n), DType.FLOAT32);
        castBoth("F16 odd offset->F32", raw(host, h, DType.FLOAT16, n).slice(0, 3, 1000),
                raw(gpu, h, DType.FLOAT16, n).slice(0, 3, 1000), DType.FLOAT32);
        castBoth("BOOL offset->F32", raw(host, bools, DType.BOOL, n).slice(0, 5, 1030),
                raw(gpu, bools, DType.BOOL, n).slice(0, 5, 1030), DType.FLOAT32);
    }

    @Test
    public void castInt64ToInt32() {
        long[] values = {0L, 1L, -1L, 1L << 33, (1L << 32) + 5, Long.MAX_VALUE, Long.MIN_VALUE, -123456789012L, 42L};
        ByteBuffer bb = ByteBuffer.allocate(values.length * 8).order(ByteOrder.nativeOrder());
        for (int i = 0; i < values.length; i++) {
            bb.putLong(i * 8, values[i]);
        }
        byte[] b = bb.array();
        castBoth("I64->I32", raw(host, b, DType.INT64, values.length), raw(gpu, b, DType.INT64, values.length),
                DType.INT32);
        castBoth("I64 offset->I32", raw(host, b, DType.INT64, values.length).slice(0, 3, 9),
                raw(gpu, b, DType.INT64, values.length).slice(0, 3, 9), DType.INT32);
        castBoth("I64 transpose->I32", raw(host, b, DType.INT64, 3, 3).transpose(),
                raw(gpu, b, DType.INT64, 3, 3).transpose(), DType.INT32);
    }

    @Test
    public void castsRoundTripThroughHalfTypes() {
        float[] data = values(513, 12);
        for (DType half : new DType[]{DType.FLOAT16, DType.BFLOAT16, DType.BOOL}) {
            Tensor refHalf = keep(floats(host, data, 513).to(half));
            Tensor devHalf = keep(floats(gpu, data, 513).to(half));
            castBoth(half + " round trip", refHalf, devHalf, DType.FLOAT32);
        }
    }

    @Test
    public void unsupportedPairsReturnNull() {
        Tensor f = floats(gpu, values(4, 13), 4);
        assertNull(GpuCoreOps.cast(f, DType.FLOAT64));
        assertNull(GpuCoreOps.cast(f, DType.INT64));
        Tensor i = keep(f.to(DType.INT32));
        assertNull(GpuCoreOps.cast(i, DType.BOOL));
    }
}
