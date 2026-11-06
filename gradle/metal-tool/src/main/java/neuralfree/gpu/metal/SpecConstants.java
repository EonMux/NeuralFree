package neuralfree.gpu.metal;

import java.util.Collections;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The specialisation constants of a generated MSL kernel, by {@code SpecId}.
 *
 * <p>SPIRV-Cross emits a constant either as an MSL function constant
 * ({@code [[function_constant(SpecId)]]}) or, when it takes part in a constant expression, as the
 * {@code SPIRV_CROSS_CONSTANT_ID_<id>} compile macro; the Metal backend supplies both at run time.
 * The kernel libraries use ids 0-2 for the dispatch's local size ({@code uint}) and id 3 for
 * {@code NF_FA_SMEM_FLOATS} ({@code int}). Any other id or type is refused by name, because the
 * backend would not know what to bind it to.</p>
 */
final class SpecConstants {

    /** Number of local-size constants: ids 0, 1 and 2. */
    static final int LOCAL_SIZE_IDS = 3;
    /** SpecId of {@code NF_FA_SMEM_FLOATS} (the backend supplies it from the device's shared memory). */
    static final int FA_SMEM_SPEC_ID = 3;

    private static final Pattern DECLARATION = Pattern.compile(
            "\\bconstant\\s+(\\w+)\\s+\\w+\\s+\\[\\[function_constant\\((\\d+)\\)\\]\\]");
    private static final Pattern MACRO = Pattern.compile("SPIRV_CROSS_CONSTANT_ID_(\\d+)");

    private final Map<Integer, Character> kinds;

    private SpecConstants(Map<Integer, Character> kinds) {
        this.kinds = Collections.unmodifiableMap(kinds);
    }

    /**
     * Parses the specialisation constants of one MSL translation unit.
     *
     * @param msl        the generated MSL text
     * @param kernelName kernel the text defines, for error messages
     * @throws IllegalArgumentException when a constant has an id the backend cannot supply or a
     *                                  type that disagrees with that id
     */
    static SpecConstants parse(String msl, String kernelName) {
        if (msl == null) {
            throw new IllegalArgumentException("MSL of " + kernelName + " must not be null");
        }
        SortedMap<Integer, Character> kinds = new TreeMap<>();
        Matcher m = DECLARATION.matcher(msl);
        while (m.find()) {
            char type = kindOf(m.group(1), kernelName);
            int id = checkedId(m.group(2), kernelName);
            char expected = id < LOCAL_SIZE_IDS ? 'u' : 'i';
            if (type != expected) {
                throw new IllegalArgumentException("kernel " + kernelName + " declares function constant "
                        + id + " as " + m.group(1) + "; expected " + nameOf(expected));
            }
            Character previous = kinds.put(id, type);
            if (previous != null && previous != type) {
                throw new IllegalArgumentException("kernel " + kernelName + " declares function constant "
                        + id + " twice with different types");
            }
        }
        m = MACRO.matcher(msl);
        while (m.find()) {
            int id = checkedId(m.group(1), kernelName);
            Character previous = kinds.put(id, id < LOCAL_SIZE_IDS ? 'u' : 'i');
            if (previous != null && previous != (id < LOCAL_SIZE_IDS ? 'u' : 'i')) {
                throw new IllegalArgumentException("kernel " + kernelName + " declares constant " + id
                        + " twice with different types");
            }
        }
        return new SpecConstants(kinds);
    }

    /** True when the kernel has no specialisation constants. */
    boolean isEmpty() {
        return kinds.isEmpty();
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
