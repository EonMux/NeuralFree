package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The GGUF file holds tensors its architecture never loads. llama.cpp refuses such a file once every tensor has
 * been created ({@code done_getting_tensors}: "wrong number of tensors", src/llama-model-loader.cpp:1400-1410),
 * because the tensors belong to a feature or model the graph would silently drop. The message names the
 * architecture, the tensor counts and the unused tensors.
 */
public class UnusedTensorException extends GgufFormatException {

    private static final long serialVersionUID = 1L;

    /** Unused tensors listed in the message; the rest are counted. */
    private static final int MESSAGE_LIMIT = 16;

    private final String arch;
    private final int fileTensors;
    private final List<String> unused;

    /**
     * @param arch        {@code general.architecture} of the file
     * @param fileTensors number of tensors in the file
     * @param unused      stored names of the tensors the architecture does not load, at least one
     * @throws IllegalArgumentException when {@code unused} is empty or larger than the file
     */
    public UnusedTensorException(String arch, int fileTensors, List<String> unused) {
        super(message(arch, fileTensors, requireEntries(unused, fileTensors)));
        this.arch = arch;
        this.fileTensors = fileTensors;
        this.unused = Collections.unmodifiableList(new ArrayList<>(unused));
    }

    private static List<String> requireEntries(List<String> unused, int fileTensors) {
        Objects.requireNonNull(unused, "unused");
        if (unused.isEmpty() || unused.size() > fileTensors) {
            throw new IllegalArgumentException("an unused-tensor report needs between 1 and " + fileTensors
                    + " entries, got " + unused.size());
        }
        return unused;
    }

    private static String message(String arch, int fileTensors, List<String> unused) {
        StringBuilder sb = new StringBuilder("architecture '").append(arch).append("': the GGUF file has ")
                .append(fileTensors).append(" tensors but the architecture loads ").append(fileTensors - unused.size())
                .append(" of them; ").append(unused.size()).append(unused.size() == 1 ? " tensor is" : " tensors are")
                .append(" never used: ");
        int shown = Math.min(unused.size(), MESSAGE_LIMIT);
        for (int i = 0; i < shown; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append('\'').append(unused.get(i)).append('\'');
        }
        if (unused.size() > shown) {
            sb.append(" and ").append(unused.size() - shown).append(" more");
        }
        return sb.toString();
    }

    public String arch() {
        return arch;
    }

    /** Number of tensors in the file. */
    public int fileTensors() {
        return fileTensors;
    }

    /** Stored names of the unused tensors, in file order (unmodifiable). */
    public List<String> unused() {
        return unused;
    }
}
