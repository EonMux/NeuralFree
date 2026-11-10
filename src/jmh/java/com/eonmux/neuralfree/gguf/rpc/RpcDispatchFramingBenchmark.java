package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * The wire cost of one remote kernel dispatch ({@link RpcWire}): the client's {@code OP_ENQUEUE} member as
 * {@code RemoteComputeQueue} writes it (kernel id, work sizes, six buffer arguments and five scalars, the shape of a
 * decode matvec) and the server's read of it back into arguments. A decode step of a 0.6B model sends about 420.
 * {@code serverEnqueue} adds what {@code RpcComputeServer}'s enqueue does around that read: it copies the
 * connection's whole buffer table into an id-to-buffer and a buffer-to-device map for every dispatch, so its cost
 * grows with {@code connectionBuffers}, the buffers the connection holds (a model's weights and activations).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class RpcDispatchFramingBenchmark {

    private static final long[] GLOBAL = {1024, 1, 1};
    private static final long[] LOCAL = {128, 1, 1};

    /** Buffers the connection holds; a 0.6B model's weights alone are about 310 tensors. */
    @Param({"16", "512"})
    public int connectionBuffers;

    private Object[] args;
    private Map<Long, Owned> table;
    private byte[] frame;
    private Map<Long, GpuBuffer> buffers;

    @Setup
    public void setUp() {
        buffers = new HashMap<>();
        args = new Object[11];
        for (int i = 0; i < 6; i++) {
            RemoteBuffer b = new RemoteBuffer(null, 100 + i, 1 << 20);
            buffers.put(b.id(), b);
            args[i] = b;
        }
        args[6] = 1024;
        args[7] = 3072;
        args[8] = 0;
        args[9] = 1.0e-6f;
        args[10] = 7L;
        table = new LinkedHashMap<>();
        for (GpuBuffer b : buffers.values()) {
            table.put(((RemoteBuffer) b).id(), new Owned(b, 0));
        }
        for (int i = table.size(); i < connectionBuffers; i++) {
            table.put(1000L + i, new Owned(new RemoteBuffer(null, 1000L + i, 1 << 20), 0));
        }
        frame = encode();
        if (decode().length != args.length) {
            throw new IllegalStateException("the dispatch frame does not round-trip");
        }
    }

    @Benchmark
    public byte[] encode() {
        RpcWire.Writer w = new RpcWire.Writer(RpcWire.OP_ENQUEUE).i64(42).longArray(GLOBAL).longArray(LOCAL);
        RpcWire.writeArgs(w, args);
        return w.toBytes();
    }

    /** A buffer and the device it lives on, the server's per-connection table entry. */
    static final class Owned {
        final GpuBuffer value;
        final int device;

        Owned(GpuBuffer value, int device) {
            this.value = value;
            this.device = device;
        }
    }

    /** The server's read of one dispatch, its per-dispatch copy of the connection's buffer table included. */
    @Benchmark
    public Object[] serverEnqueue() {
        RpcWire.Reader r = new RpcWire.Reader(frame);
        r.opcode();
        r.i64();
        r.longArray();
        r.longArray();
        Map<GpuBuffer, Integer> deviceOfBuffer = new LinkedHashMap<>();
        Map<Long, GpuBuffer> scope = new LinkedHashMap<>();
        for (Map.Entry<Long, Owned> e : table.entrySet()) {
            scope.put(e.getKey(), e.getValue().value);
            deviceOfBuffer.put(e.getValue().value, e.getValue().device);
        }
        Object[] read = RpcWire.readArgs(r, scope);
        for (Object arg : read) {
            if (arg instanceof GpuBuffer && deviceOfBuffer.get(arg) == null) {
                throw new IllegalStateException("an argument buffer is not in the table");
            }
        }
        return read;
    }

    @Benchmark
    public Object[] decode() {
        RpcWire.Reader r = new RpcWire.Reader(frame);
        r.opcode();
        r.i64();
        r.longArray();
        r.longArray();
        return RpcWire.readArgs(r, buffers);
    }
}
