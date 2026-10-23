package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A tensor the model needs is not in the GGUF file. The message names the architecture, layer and tensor; a
 * load-time check that finds several missing tensors reports all of them in one exception ({@link #missing()}).
 */
public class MissingTensorException extends GgufFormatException {

    private static final long serialVersionUID = 1L;

    /** Missing entries listed in the message of a multi-tensor report; the rest are counted. */
    private static final int MESSAGE_LIMIT = 16;

    private final String arch;
    private final int layer;
    private final String tensor;
    private final List<String> missing;

    /**
     * @param arch   {@code general.architecture} of the file
     * @param layer  layer index, or -1 for a model-level tensor
     * @param tensor full tensor name that was looked up
     */
    public MissingTensorException(String arch, int layer, String tensor) {
        super(message(arch, layer, tensor));
        this.arch = arch;
        this.layer = layer;
        this.tensor = tensor;
        this.missing = Collections.singletonList(tensor);
    }

    /**
     * Report of every unmet tensor requirement of a file.
     *
     * @param arch    {@code general.architecture} of the file
     * @param missing descriptions of the unmet requirements (a tensor name, or alternatives joined by {@code or}),
     *                at least one
     * @throws IllegalArgumentException when {@code missing} is empty
     */
    public MissingTensorException(String arch, List<String> missing) {
        super(message(arch, requireEntries(missing)));
        this.arch = arch;
        this.layer = -1;
        this.tensor = missing.get(0);
        this.missing = Collections.unmodifiableList(new ArrayList<>(missing));
    }

    private static List<String> requireEntries(List<String> missing) {
        Objects.requireNonNull(missing, "missing");
        if (missing.isEmpty()) {
            throw new IllegalArgumentException("a missing-tensor report needs at least one entry");
        }
        return missing;
    }

    private static String message(String arch, int layer, String tensor) {
        String where = layer >= 0 ? " (layer " + layer + ")" : "";
        return "architecture '" + arch + "': required tensor '" + tensor + "'" + where + " is missing from the GGUF file";
    }

    private static String message(String arch, List<String> missing) {
        if (missing.size() == 1) {
            return message(arch, -1, missing.get(0));
        }
        StringBuilder sb = new StringBuilder("architecture '").append(arch).append("': ").append(missing.size())
                .append(" required tensors are missing from the GGUF file: ");
        int shown = Math.min(missing.size(), MESSAGE_LIMIT);
        for (int i = 0; i < shown; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append('\'').append(missing.get(i)).append('\'');
        }
        if (missing.size() > shown) {
            sb.append(" and ").append(missing.size() - shown).append(" more");
        }
        return sb.toString();
    }

    public String arch() {
        return arch;
    }

    /** Layer index, or -1 for a model-level tensor or a multi-tensor report. */
    public int layer() {
        return layer;
    }

    /** The missing tensor, or the first entry of a multi-tensor report. */
    public String tensor() {
        return tensor;
    }

    /** Every missing entry (one for a single lookup; unmodifiable). */
    public List<String> missing() {
        return missing;
    }
}
