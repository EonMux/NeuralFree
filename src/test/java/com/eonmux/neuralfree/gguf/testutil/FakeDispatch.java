package com.eonmux.neuralfree.gguf.testutil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Immutable record of one kernel dispatch of a {@link FakeComputeBackend}. Array arguments are returned as copies. */
public final class FakeDispatch {
    private final int sequence;
    private final String queue;
    private final String kernelName;
    private final long[] global;
    private final long[] local;
    private final Object[] args;
    private final boolean preBound;
    private final boolean returnedEvent;
    private final int waitCount;

    FakeDispatch(int sequence, String queue, String kernelName, long[] global, long[] local, Object[] args,
                 boolean preBound, boolean returnedEvent, int waitCount) {
        this.sequence = sequence;
        this.queue = queue;
        this.kernelName = kernelName;
        this.global = global;
        this.local = local;
        this.args = args;
        this.preBound = preBound;
        this.returnedEvent = returnedEvent;
        this.waitCount = waitCount;
    }

    public int sequence() {
        return sequence;
    }

    public String queue() {
        return queue;
    }

    public String kernelName() {
        return kernelName;
    }

    public long[] global() {
        return global.clone();
    }

    /** The local size passed by the caller, or null when the caller left it to the backend. */
    public long[] local() {
        return local == null ? null : local.clone();
    }

    public boolean isPreBound() {
        return preBound;
    }

    public boolean returnedEvent() {
        return returnedEvent;
    }

    public int waitCount() {
        return waitCount;
    }

    public int argCount() {
        return args.length;
    }

    /** The snapshot of argument {@code i}; arrays are copied. */
    public Object arg(int i) {
        return copyIfArray(args[i]);
    }

    public List<Object> args() {
        List<Object> out = new ArrayList<>(args.length);
        for (Object a : args) {
            out.add(copyIfArray(a));
        }
        return Collections.unmodifiableList(out);
    }

    /** Argument {@code i} as a 32-bit integer: Integer, Boolean (1/0) or a one-element int[]. */
    public int intArg(int i) {
        Object a = args[i];
        if (a instanceof Integer) {
            return (Integer) a;
        }
        if (a instanceof Boolean) {
            return (Boolean) a ? 1 : 0;
        }
        if (a instanceof int[] && ((int[]) a).length == 1) {
            return ((int[]) a)[0];
        }
        throw new IllegalStateException("arg " + i + " of " + kernelName + " is not an int: " + describe(a));
    }

    /** Argument {@code i} as a float: Float or a one-element float[]. */
    public float floatArg(int i) {
        Object a = args[i];
        if (a instanceof Float) {
            return (Float) a;
        }
        if (a instanceof float[] && ((float[]) a).length == 1) {
            return ((float[]) a)[0];
        }
        throw new IllegalStateException("arg " + i + " of " + kernelName + " is not a float: " + describe(a));
    }

    public FakeGpuBuffer bufferArg(int i) {
        Object a = args[i];
        if (a instanceof FakeGpuBuffer) {
            return (FakeGpuBuffer) a;
        }
        throw new IllegalStateException("arg " + i + " of " + kernelName + " is not a buffer: " + describe(a));
    }

    /** Buffer arguments in signature order, i.e. Vulkan binding order. */
    public List<FakeGpuBuffer> bufferArgs() {
        List<FakeGpuBuffer> out = new ArrayList<>();
        for (Object a : args) {
            if (a instanceof FakeGpuBuffer) {
                out.add((FakeGpuBuffer) a);
            }
        }
        return Collections.unmodifiableList(out);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('#').append(sequence).append(' ').append(queue).append(" DISPATCH ").append(kernelName)
                .append(" g=").append(Arrays.toString(global))
                .append(" l=").append(local == null ? "auto" : Arrays.toString(local))
                .append(" args=[");
        for (int i = 0; i < args.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(describe(args[i]));
        }
        sb.append(']');
        if (preBound) {
            sb.append(" prebound");
        }
        if (waitCount > 0) {
            sb.append(" waits=").append(waitCount);
        }
        return sb.toString();
    }

    private static Object copyIfArray(Object a) {
        if (a instanceof int[]) {
            return ((int[]) a).clone();
        }
        if (a instanceof float[]) {
            return ((float[]) a).clone();
        }
        if (a instanceof byte[]) {
            return ((byte[]) a).clone();
        }
        if (a instanceof short[]) {
            return ((short[]) a).clone();
        }
        if (a instanceof long[]) {
            return ((long[]) a).clone();
        }
        if (a instanceof double[]) {
            return ((double[]) a).clone();
        }
        return a;
    }

    private static String describe(Object a) {
        if (a instanceof int[]) {
            return Arrays.toString((int[]) a);
        }
        if (a instanceof float[]) {
            return Arrays.toString((float[]) a);
        }
        if (a instanceof byte[]) {
            return Arrays.toString((byte[]) a);
        }
        if (a instanceof short[]) {
            return Arrays.toString((short[]) a);
        }
        if (a instanceof long[]) {
            return Arrays.toString((long[]) a);
        }
        if (a instanceof double[]) {
            return Arrays.toString((double[]) a);
        }
        return String.valueOf(a);
    }
}
