package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.kernels.FaSmem;

import java.util.Collections;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The specialisation constants of one SPIRV-Cross-generated MSL kernel, by {@code SpecId}.
 *
 * <p>SPIRV-Cross emits a constant in one of two forms. When the constant is only read at run time
 * it is an MSL function constant ({@code constant uint x_tmp [[function_constant(0)]]}), which the
 * backend supplies when it builds the pipeline. When it takes part in a constant expression (a
 * threadgroup array size, a constant comparison) it is instead the preprocessor macro
 * {@code SPIRV_CROSS_CONSTANT_ID_<id>}, which the backend supplies when it compiles the library,
 * so such a library is compiled once per constant set.</p>
 *
 * <p>The kernel libraries use ids 0-2 for the dispatch's local size ({@code uint}) and id 3 for
 * {@code NF_FA_SMEM_FLOATS} ({@code int}); {@link #values(long[], int)} maps a dispatch's
 * threadgroup and the device's flash-attention shared-memory size onto them. Any other id or type
 * is refused by name, because the backend cannot bind it.</p>
 */
final class SpecConstants {

    /** Number of local-size constants: ids 0, 1 and 2. */
    static final int LOCAL_SIZE_IDS = 3;
    /** SpecId of {@code NF_FA_SMEM_FLOATS}. */
    static final int FA_SMEM_SPEC_ID = FaSmem.VULKAN_SPEC_ID;

    private static final Pattern FUNCTION_CONSTANT = Pattern.compile(
            "\\bconstant\\s+(\\w+)\\s+\\w+\\s+\\[\\[function_constant\\((\\d+)\\)\\]\\]");
    private static final Pattern MACRO_CONSTANT = Pattern.compile("SPIRV_CROSS_CONSTANT_ID_(\\d+)");

    private final Map<Integer, Character> kinds;
    private final Map<Integer, Character> functionConstants;
    private final Map<Integer, Character> macroConstants;

    private SpecConstants(Map<Integer, Character> kinds, Map<Integer, Character> functionConstants,
                          Map<Integer, Character> macroConstants) {
        this.kinds = Collections.unmodifiableMap(kinds);
        this.functionConstants = Collections.unmodifiableMap(functionConstants);
        this.macroConstants = Collections.unmodifiableMap(macroConstants);
    }

    /**
     * Parses the specialisation constants of one MSL translation unit.
     *
     * @param msl        MSL source of one kernel
     * @param kernelName kernel the source defines, for error messages
     * @throws IllegalArgumentException when a constant has an id the backend cannot supply or a
     *                                  type that disagrees with that id
     */
    static SpecConstants parse(String msl, String kernelName) {
        if (msl == null) {
            throw new IllegalArgumentException("MSL of " + kernelName + " must not be null");
        }
        SortedMap<Integer, Character> kinds = new TreeMap<>();
        SortedMap<Integer, Character> function = new TreeMap<>();
        SortedMap<Integer, Character> macros = new TreeMap<>();
        Matcher m = FUNCTION_CONSTANT.matcher(msl);
        while (m.find()) {
            char type = kindOf(m.group(1), kernelName);
            int id = checkedId(m.group(2), kernelName);
            char expected = expectedKind(id);
            if (type != expected) {
                throw new IllegalArgumentException("kernel " + kernelName + " declares function constant "
                        + id + " as " + m.group(1) + "; expected " + nameOf(expected));
            }
            put(kinds, function, id, type, kernelName);
        }
        m = MACRO_CONSTANT.matcher(msl);
        while (m.find()) {
            int id = checkedId(m.group(1), kernelName);
            put(kinds, macros, id, expectedKind(id), kernelName);
        }
        return new SpecConstants(kinds, function, macros);
    }

    /** True when the kernel has no specialisation constants and one pipeline for every size. */
    boolean isEmpty() {
        return kinds.isEmpty();
    }

    /** True when some constant is a library-compile macro and needs one library per constant set. */
    boolean hasMacros() {
        return !macroConstants.isEmpty();
    }

    /** The declared constants by SpecId, with their MSL type ({@code 'u'}, {@code 'i'} or {@code 'f'}). */
    SortedMap<Integer, Character> kinds() {
        return new TreeMap<>(kinds);
    }

    /**
     * The values of this kernel's constants for one dispatch.
     *
     * @param local3       local size padded to three dimensions
     * @param faSmemFloats {@code NF_FA_SMEM_FLOATS} of the device
     * @return the value of each declared constant, by SpecId
     * @throws IllegalArgumentException when {@code local3} is malformed
     */
    SortedMap<Integer, Long> values(long[] local3, int faSmemFloats) {
        if (local3 == null || local3.length != 3) {
            throw new IllegalArgumentException("local size must have three dimensions, got "
                    + (local3 == null ? "null" : String.valueOf(local3.length)));
        }
        if (faSmemFloats < 1) {
            throw new IllegalArgumentException("NF_FA_SMEM_FLOATS must be positive, got " + faSmemFloats);
        }
        SortedMap<Integer, Long> values = new TreeMap<>();
        for (Map.Entry<Integer, Character> e : kinds.entrySet()) {
            int id = e.getKey();
            values.put(id, id < LOCAL_SIZE_IDS ? local3[id] : (long) faSmemFloats);
        }
        return values;
    }

    /** The values of the function-constant style ids only, by SpecId. */
    SortedMap<Integer, Long> functionValues(SortedMap<Integer, Long> values) {
        SortedMap<Integer, Long> out = new TreeMap<>();
        for (Map.Entry<Integer, Long> e : values.entrySet()) {
            if (functionConstants.containsKey(e.getKey())) {
                out.put(e.getKey(), e.getValue());
            }
        }
        return out;
    }

    /**
     * The compiler options ({@code -DNAME=VALUE}) that supply this kernel's macro-style constants.
     *
     * @return a space-separated options string, or "" when the kernel has none
     */
    String macroOptions(SortedMap<Integer, Long> values) {
        StringBuilder options = new StringBuilder();
        for (Integer id : macroConstants.keySet()) {
            Long value = values.get(id);
            if (value == null) {
                throw new IllegalArgumentException("no value for function constant " + id);
            }
            if (options.length() > 0) {
                options.append(' ');
            }
            options.append("-DSPIRV_CROSS_CONSTANT_ID_").append(id).append('=').append(value);
        }
        return options.toString();
    }

    private static void put(Map<Integer, Character> kinds, Map<Integer, Character> style, int id, char type,
                            String kernelName) {
        Character previous = kinds.put(id, type);
        if (previous != null && previous != type) {
            throw new IllegalArgumentException("kernel " + kernelName + " declares constant " + id
                    + " twice with different types");
        }
        style.put(id, type);
    }

    private static int checkedId(String text, String kernelName) {
        int id = Integer.parseInt(text);
        if (id < 0 || id > FA_SMEM_SPEC_ID) {
            throw new IllegalArgumentException("kernel " + kernelName + " has specialisation constant "
                    + id + "; only ids 0-2 (local size) and " + FA_SMEM_SPEC_ID
                    + " (NF_FA_SMEM_FLOATS) can be supplied");
        }
        return id;
    }

    private static char expectedKind(int id) {
        return id < LOCAL_SIZE_IDS ? 'u' : 'i';
    }

    private static char kindOf(String type, String kernelName) {
        if ("uint".equals(type)) {
            return 'u';
        }
        if ("int".equals(type)) {
            return 'i';
        }
        if ("float".equals(type)) {
            return 'f';
        }
        throw new IllegalArgumentException("kernel " + kernelName + " has a function constant of type "
                + type + "; only uint, int and float are supported");
    }

    private static String nameOf(char kind) {
        return kind == 'u' ? "uint" : (kind == 'i' ? "int" : "float");
    }
}
