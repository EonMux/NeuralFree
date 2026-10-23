package com.eonmux.neuralfree.acceleration.backend;

import java.util.Arrays;

/**
 * Immutable build-time argument signature of one compiled kernel.
 *
 * <p>The signature lists the buffer arguments (in buffer-argument order, with
 * their read-only flags) and the scalar arguments (in scalar-argument order,
 * each a 4-byte scalar). For Vulkan it is recorded from the unoptimised SPIR-V
 * module ({@link #parse}), so it still describes a binding that the optimiser
 * stripped and such a binding can never misalign the remaining arguments. For
 * CUDA it comes from the kernel's parameter list in the kernel inventory
 * ({@link #parseInventory}), because CUDA modules carry no parameter metadata.</p>
 *
 * <p>Scalar kinds: {@code 'i'} int32, {@code 'u'} uint32, {@code 'f'} float32. An int32 or uint32
 * scalar takes an {@link KernelArgs.ArgKind#I32} argument, a float32 scalar an
 * {@link KernelArgs.ArgKind#F32} one.</p>
 *
 * <p>A signature may also carry the parameter layout: whether each parameter, in declaration
 * order, is a buffer or a scalar ({@link #hasLayout}). CUDA passes arguments positionally, so its
 * inventory signatures carry it and a buffer given where the kernel declares a scalar (or the
 * reverse) is rejected even when the counts and kinds still match. Vulkan binds buffers and
 * push constants separately, so its signatures need no layout.</p>
 *
 * <p>The argument checks ({@link #validateArgs}, {@link #checkScalar}, {@link #checkCounts}) are the
 * single argument contract of every backend that validates against a signature: Vulkan against the
 * shader index, CUDA against the kernel inventory.</p>
 */
public final class KernelSignature {

    /** Bytes per scalar argument. */
    public static final int SCALAR_BYTES = 4;

    /** Layout entry of a buffer parameter; a scalar parameter's entry is its kind. */
    public static final char LAYOUT_BUFFER = 'B';

    private final boolean[] bufferReadOnly;
    private final char[] pushKinds;
    /** Buffer-or-scalar flag per parameter in declaration order, or null when the layout is not recorded. */
    private final boolean[] layoutIsBuffer;

    /**
     * A signature without a parameter layout.
     *
     * @param bufferReadOnly read-only flag per buffer argument, in buffer-argument order
     * @param pushKinds      kind per scalar argument ({@code 'i'}, {@code 'u'} or {@code 'f'}), in scalar order
     * @throws IllegalArgumentException when an array is null or a kind is unknown
     */
    public KernelSignature(boolean[] bufferReadOnly, char[] pushKinds) {
        this(bufferReadOnly, pushKinds, null);
    }

    private KernelSignature(boolean[] bufferReadOnly, char[] pushKinds, boolean[] layoutIsBuffer) {
        if (bufferReadOnly == null || pushKinds == null) {
            throw new IllegalArgumentException("signature arrays must not be null");
        }
        for (int i = 0; i < pushKinds.length; i++) {
            char k = pushKinds[i];
            if (k != 'i' && k != 'u' && k != 'f') {
                throw new IllegalArgumentException("unknown scalar kind '" + k + "' at scalar " + i
                        + "; expected i, u or f");
            }
        }
        this.bufferReadOnly = bufferReadOnly.clone();
        this.pushKinds = pushKinds.clone();
        this.layoutIsBuffer = layoutIsBuffer == null ? null : layoutIsBuffer.clone();
    }

    /**
     * Parses the three per-kernel entries of {@code shaders/llm/index.properties}.
     *
     * @param buffers number of buffer arguments, a decimal integer
     * @param ro      one character per buffer: {@code '1'} read-only, {@code '0'} written
     *                (empty when there are no buffers)
     * @param push    comma-separated scalar kinds, for example {@code "i,f,i"} (empty when
     *                there are no scalars)
     * @throws IllegalArgumentException when an entry is malformed or the counts disagree
     */
    public static KernelSignature parse(String buffers, String ro, String push) {
        if (buffers == null || ro == null || push == null) {
            throw new IllegalArgumentException("signature entries must not be null (buffers="
                    + buffers + ", ro=" + ro + ", push=" + push + ")");
        }
        int bufferCount = parseCount(buffers, "buffer");
        String roTrimmed = ro.trim();
        if (roTrimmed.length() != bufferCount) {
            throw new IllegalArgumentException("read-only flags '" + ro + "' have " + roTrimmed.length()
                    + " entries but there are " + bufferCount + " buffers");
        }
        boolean[] readOnly = new boolean[bufferCount];
        for (int i = 0; i < bufferCount; i++) {
            char c = roTrimmed.charAt(i);
            if (c != '0' && c != '1') {
                throw new IllegalArgumentException("read-only flag '" + c + "' of buffer " + i
                        + " must be 0 or 1");
            }
            readOnly[i] = c == '1';
        }
        return new KernelSignature(readOnly, parseKinds(push));
    }

    /**
     * Parses the count and kind columns of one line of the kernel inventory
     * ({@code kernels/llm/inventory.txt}: {@code name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds}),
     * without a parameter layout. The inventory records no buffer access, so every buffer of the result
     * is reported as written ({@link #isBufferReadOnly} false), the conservative assumption.
     *
     * @param buffers number of buffer arguments, a decimal integer
     * @param scalars number of scalar arguments, a decimal integer
     * @param kinds   comma-separated scalar kinds, for example {@code "i,f,i"} (empty when there are no
     *                scalars)
     * @throws IllegalArgumentException when a column is malformed or the scalar count disagrees with the kinds
     */
    public static KernelSignature parseInventory(String buffers, String scalars, String kinds) {
        if (buffers == null || scalars == null || kinds == null) {
            throw new IllegalArgumentException("inventory columns must not be null (buffers=" + buffers
                    + ", scalars=" + scalars + ", kinds=" + kinds + ")");
        }
        int bufferCount = parseCount(buffers, "buffer");
        int scalarCount = parseCount(scalars, "scalar");
        char[] parsed = parseKinds(kinds);
        if (parsed.length != scalarCount) {
            throw new IllegalArgumentException("scalar count " + scalarCount + " disagrees with the "
                    + parsed.length + " scalar kinds '" + kinds + "'");
        }
        return new KernelSignature(new boolean[bufferCount], parsed);
    }

    /**
     * Parses one line of the kernel inventory including its layout column
     * ({@code name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds<TAB>layout}). The layout has one
     * character per parameter in declaration order: {@link #LAYOUT_BUFFER} for a buffer, otherwise the
     * scalar's kind; its buffers and scalars must agree with the count and kind columns. Buffers are
     * reported as written, as in {@link #parseInventory(String, String, String)}.
     *
     * @param layout the layout column, for example {@code "iifBB"} (empty for a kernel without parameters)
     * @throws IllegalArgumentException when a column is malformed or the columns disagree
     */
    public static KernelSignature parseInventory(String buffers, String scalars, String kinds, String layout) {
        KernelSignature counted = parseInventory(buffers, scalars, kinds);
        if (layout == null) {
            throw new IllegalArgumentException("inventory layout column must not be null");
        }
        String l = layout.trim();
        boolean[] isBuffer = new boolean[l.length()];
        StringBuilder scalarKinds = new StringBuilder();
        int bufferCount = 0;
        for (int i = 0; i < l.length(); i++) {
            char c = l.charAt(i);
            if (c == LAYOUT_BUFFER) {
                isBuffer[i] = true;
                bufferCount++;
            } else if (c == 'i' || c == 'u' || c == 'f') {
                scalarKinds.append(c);
            } else {
                throw new IllegalArgumentException("layout '" + layout + "' has '" + c + "' at parameter " + i
                        + "; expected " + LAYOUT_BUFFER + ", i, u or f");
            }
        }
        if (bufferCount != counted.bufferCount() || !scalarKinds.toString().equals(new String(counted.pushKinds))) {
            throw new IllegalArgumentException("layout '" + layout + "' has " + bufferCount + " buffers and scalars '"
                    + scalarKinds + "', the count and kind columns " + counted.bufferCount() + " buffers and scalars '"
                    + new String(counted.pushKinds) + "'");
        }
        return new KernelSignature(counted.bufferReadOnly, counted.pushKinds, isBuffer);
    }

    /** Whether the signature records which parameters, in declaration order, are buffers. */
    public boolean hasLayout() {
        return layoutIsBuffer != null;
    }

    public int bufferCount() {
        return bufferReadOnly.length;
    }

    /** Copy of the read-only flags, in buffer-argument order. */
    public boolean[] bufferReadOnly() {
        return bufferReadOnly.clone();
    }

    public boolean isBufferReadOnly(int buffer) {
        return bufferReadOnly[buffer];
    }

    public int scalarCount() {
        return pushKinds.length;
    }

    /** Size of the push-constant block in bytes. */
    public int pushBytes() {
        return pushKinds.length * SCALAR_BYTES;
    }

    /** Copy of the scalar kinds, in scalar-argument order. */
    public char[] pushKinds() {
        return pushKinds.clone();
    }

    public char pushKind(int scalar) {
        return pushKinds[scalar];
    }

    /** Total number of Java arguments the kernel takes. */
    public int argCount() {
        return bufferReadOnly.length + pushKinds.length;
    }

    /**
     * Validates Java arguments, in declared order, against this signature: the
     * number of buffers, the number of scalars, and each scalar's kind
     * ({@code 'i'}/{@code 'u'} need an I32 argument, {@code 'f'} an F32 one).
     *
     * @throws IllegalArgumentException naming the kernel and the first mismatch
     */
    public void validateArgs(String kernelName, Object[] args) {
        if (args == null) {
            throw new IllegalArgumentException(kernelName + ": argument array is null");
        }
        int buffers = 0;
        int scalars = 0;
        for (int i = 0; i < args.length; i++) {
            KernelArgs.ArgKind kind;
            try {
                kind = KernelArgs.kindOf(args[i]);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(kernelName + ": arg " + i + ": " + e.getMessage(), e);
            }
            checkPosition(kernelName, i, kind, KernelArgs.describe(args[i]));
            if (kind == KernelArgs.ArgKind.BUFFER) {
                buffers++;
                continue;
            }
            checkScalar(kernelName, i, scalars, kind, KernelArgs.describe(args[i]));
            scalars++;
        }
        checkCounts(kernelName, buffers, scalars);
    }

    /**
     * Checks that the scalar argument at Java index {@code argIndex}, the {@code scalar}-th scalar of
     * the call, exists in this signature and has the kind it declares.
     *
     * @param kind     the argument's kind ({@link KernelArgs#kindOf}); never {@code BUFFER}
     * @param describe the argument's description for the message ({@link KernelArgs#describe})
     * @throws IllegalArgumentException naming the kernel, the argument and the expected kind
     */
    public void checkScalar(String kernelName, int argIndex, int scalar, KernelArgs.ArgKind kind, String describe) {
        if (scalar >= pushKinds.length) {
            throw new IllegalArgumentException(kernelName + ": arg " + argIndex + " (" + describe
                    + ") exceeds the " + pushKinds.length + " scalar arguments of the signature");
        }
        if (kind != scalarArgKind(scalar)) {
            throw new IllegalArgumentException(kernelName + ": arg " + argIndex + " (" + describe
                    + ") does not match scalar " + scalar + " (" + scalarTypeName(scalar) + ")");
        }
    }

    /**
     * Checks that the argument at Java index {@code argIndex} is a buffer exactly where the parameter
     * layout declares one. Does nothing when the signature has no layout ({@link #hasLayout}); an index
     * past the layout is left to {@link #checkCounts}.
     *
     * @param kind     the argument's kind ({@link KernelArgs#kindOf})
     * @param describe the argument's description for the message ({@link KernelArgs#describe})
     * @throws IllegalArgumentException naming the kernel, the argument and the declared parameter
     */
    public void checkPosition(String kernelName, int argIndex, KernelArgs.ArgKind kind, String describe) {
        if (layoutIsBuffer == null || argIndex >= layoutIsBuffer.length) {
            return;
        }
        boolean buffer = kind == KernelArgs.ArgKind.BUFFER;
        if (buffer != layoutIsBuffer[argIndex]) {
            throw new IllegalArgumentException(kernelName + ": arg " + argIndex + " (" + describe + ") is "
                    + (buffer ? "a buffer" : "a scalar") + " but parameter " + argIndex + " is declared as "
                    + (layoutIsBuffer[argIndex] ? "a buffer" : "a scalar"));
        }
    }

    /**
     * Checks the totals of a complete argument list.
     *
     * @throws IllegalArgumentException naming the kernel when either count differs from the signature
     */
    public void checkCounts(String kernelName, int buffers, int scalars) {
        if (buffers != bufferReadOnly.length || scalars != pushKinds.length) {
            throw new IllegalArgumentException(kernelName + ": got " + buffers + " buffers and " + scalars
                    + " scalars, signature expects " + bufferReadOnly.length + " buffers and "
                    + pushKinds.length + " scalars");
        }
    }

    /** The argument kind scalar {@code scalar} takes: F32 for {@code 'f'}, I32 for {@code 'i'} and {@code 'u'}. */
    public KernelArgs.ArgKind scalarArgKind(int scalar) {
        return pushKinds[scalar] == 'f' ? KernelArgs.ArgKind.F32 : KernelArgs.ArgKind.I32;
    }

    /** {@code int32}, {@code uint32} or {@code float32}: the declared type of scalar {@code scalar}. */
    public String scalarTypeName(int scalar) {
        char k = pushKinds[scalar];
        return k == 'f' ? "float32" : k == 'u' ? "uint32" : "int32";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KernelSignature)) {
            return false;
        }
        KernelSignature that = (KernelSignature) o;
        return Arrays.equals(bufferReadOnly, that.bufferReadOnly) && Arrays.equals(pushKinds, that.pushKinds)
                && Arrays.equals(layoutIsBuffer, that.layoutIsBuffer);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * Arrays.hashCode(bufferReadOnly) + Arrays.hashCode(pushKinds)) + Arrays.hashCode(layoutIsBuffer);
    }

    @Override
    public String toString() {
        StringBuilder ro = new StringBuilder();
        for (boolean b : bufferReadOnly) {
            ro.append(b ? '1' : '0');
        }
        StringBuilder push = new StringBuilder();
        for (int i = 0; i < pushKinds.length; i++) {
            if (i > 0) {
                push.append(',');
            }
            push.append(pushKinds[i]);
        }
        StringBuilder layout = new StringBuilder();
        if (layoutIsBuffer != null) {
            int scalar = 0;
            for (boolean b : layoutIsBuffer) {
                layout.append(b ? LAYOUT_BUFFER : pushKinds[scalar++]);
            }
        }
        return "KernelSignature{buffers=" + bufferReadOnly.length + ", ro=" + ro + ", push=" + push
                + (layoutIsBuffer != null ? ", layout=" + layout : "") + "}";
    }

    private static int parseCount(String value, String what) {
        int n;
        try {
            n = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(what + " count '" + value + "' is not an integer", e);
        }
        if (n < 0) {
            throw new IllegalArgumentException(what + " count must not be negative: " + n);
        }
        return n;
    }

    /** Comma-separated single-character kinds; the constructor validates each character. */
    private static char[] parseKinds(String kinds) {
        String trimmed = kinds.trim();
        if (trimmed.isEmpty()) {
            return new char[0];
        }
        String[] parts = trimmed.split(",", -1);
        char[] out = new char[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String p = parts[i].trim();
            if (p.length() != 1) {
                throw new IllegalArgumentException("scalar kind '" + parts[i] + "' at scalar " + i
                        + " in '" + kinds + "' must be one of i, u, f");
            }
            out[i] = p.charAt(0);
        }
        return out;
    }
}
