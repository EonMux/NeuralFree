package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Reads the torchao tensor subclasses a safetensors checkpoint stores in its {@code __metadata__} entries, the form
 * torchao's {@code flatten_tensor_state_dict} writes: a {@code tensor_names} JSON list of the checkpoint's tensors,
 * and for each tensor a JSON descriptor {@code {"_type": ..., "_data": {...}, "_tensor_data_names": [...]}}. A
 * subclass's data tensors are stored as ordinary safetensors under
 * {@code <module>._<weight>_<data name>}.
 *
 * <p>Only the tensors the metadata lists are read, so unrelated metadata ({@code format}, tokenizer hints) is left
 * alone. Every descriptor is checked: the type is a torchao subclass, the data names are unique strings, and each
 * names a tensor the checkpoint stores. A plain tensor descriptor is verified to name a stored tensor and otherwise
 * skipped. A subclass NeuralFree does not decode is still read, so that the decoder refuses it by name with its own
 * message.</p>
 */
public final class TorchaoSafetensors {

    /** The checkpoint's ordinary tensors, by name, as the metadata reader resolves a subclass's data tensors. */
    public interface Tensors {

        /** The tensor's entry, or null when the checkpoint stores no such tensor. */
        StTensor tensor(String name);

        /** A row source over the tensor's stored bytes. */
        RowSource source(String name);
    }

    /** The metadata key that lists the checkpoint's tensors. */
    public static final String TENSOR_NAMES = "tensor_names";
    /** The descriptor field naming the subclass. */
    public static final String TYPE = "_type";
    /** The descriptor field holding the subclass attributes. */
    public static final String DATA = "_data";
    /** The descriptor field listing the subclass's data tensors. */
    public static final String DATA_NAMES = "_tensor_data_names";
    /** The descriptor type of a plain tensor ({@code torch.Tensor.__name__}). */
    public static final String PLAIN = "Tensor";
    /** The descriptor type of a plain tensor as older docs spell it. */
    public static final String PLAIN_QUALIFIED = "torch.Tensor";

    private TorchaoSafetensors() {
    }

    /**
     * The torchao tensors of a checkpoint, in the order its metadata lists them.
     *
     * @param metadata the {@code __metadata__} entries of every checkpoint file, in file order
     * @param tensors  the checkpoint's ordinary tensors
     * @param where    the checkpoint, for error messages
     * @throws IOException when a descriptor is not valid JSON, names an unknown type, misses data, or names a tensor
     *                     the checkpoint does not store
     */
    public static List<TorchaoWeight> read(List<Map<String, String>> metadata, Tensors tensors, String where)
            throws IOException {
        Objects.requireNonNull(metadata, "metadata");
        Objects.requireNonNull(tensors, "tensors");
        Objects.requireNonNull(where, "where");
        Map<String, String> descriptors = descriptors(metadata, where);
        List<TorchaoWeight> out = new ArrayList<>();
        for (Map.Entry<String, String> e : descriptors.entrySet()) {
            String name = e.getKey();
            Object parsed = parse(e.getValue(), name, where);
            if (!(parsed instanceof Map)) {
                throw error(where, name, "its metadata descriptor is not a JSON object");
            }
            Map<?, ?> descriptor = (Map<?, ?>) parsed;
            Object type = descriptor.get(TYPE);
            if (!(type instanceof String)) {
                throw error(where, name, "its metadata descriptor has no string '" + TYPE + "' field");
            }
            String subclass = (String) type;
            if (PLAIN.equals(subclass) || PLAIN_QUALIFIED.equals(subclass)) {
                if (tensors.tensor(name) == null) {
                    throw error(where, name, "its metadata calls it a plain tensor, but the checkpoint stores no"
                            + " tensor of that name");
                }
                continue;
            }
            int dot = subclass.lastIndexOf('.');
            String simple = dot < 0 ? subclass : subclass.substring(dot + 1);
            if (!TorchaoPickle.TENSOR_SIMPLE_NAMES.contains(simple)) {
                throw error(where, name, "its metadata names the tensor class '" + subclass + "', which is not a"
                        + " torchao tensor subclass NeuralFree decodes");
            }
            out.add(weight(name, simple, descriptor, tensors, where));
        }
        return Collections.unmodifiableList(out);
    }

    /** The descriptor JSON of every tensor the files' {@code tensor_names} lists, checked for conflicts. */
    private static Map<String, String> descriptors(List<Map<String, String>> metadata, String where)
            throws IOException {
        Set<String> order = new LinkedHashSet<>();
        for (Map<String, String> entries : metadata) {
            String listed = entries.get(TENSOR_NAMES);
            if (listed == null) {
                continue;
            }
            Object names = parseMetadata(listed, TENSOR_NAMES, where);
            if (!(names instanceof List)) {
                throw new IOException("Invalid torchao metadata in " + where + ": '" + TENSOR_NAMES + "' is not a"
                        + " JSON list of tensor names");
            }
            for (Object n : (List<?>) names) {
                if (!(n instanceof String) || ((String) n).isEmpty()) {
                    throw new IOException("Invalid torchao metadata in " + where + ": '" + TENSOR_NAMES + "' holds "
                            + n + " instead of a tensor name");
                }
                order.add((String) n);
            }
        }
        Map<String, String> out = new LinkedHashMap<>();
        for (String name : order) {
            String descriptor = null;
            for (Map<String, String> entries : metadata) {
                String candidate = entries.get(name);
                if (candidate == null) {
                    continue;
                }
                if (descriptor != null && !descriptor.equals(candidate)) {
                    throw new IOException("Invalid torchao metadata in " + where + ": tensor '" + name
                            + "' has two different descriptors across the checkpoint files");
                }
                descriptor = candidate;
            }
            if (descriptor == null) {
                throw new IOException("Invalid torchao metadata in " + where + ": tensor '" + name + "' is listed"
                        + " in '" + TENSOR_NAMES + "' but has no descriptor");
            }
            out.put(name, descriptor);
        }
        return out;
    }

    /** One torchao subclass: its attributes from {@code _data} and its flattened data tensors. */
    private static TorchaoWeight weight(String name, String subclass, Map<?, ?> descriptor, Tensors tensors,
                                        String where) throws IOException {
        Object data = descriptor.get(DATA);
        if (!(data instanceof Map)) {
            throw error(where, name, "its metadata descriptor has no '" + DATA + "' object");
        }
        Object declared = descriptor.get(DATA_NAMES);
        if (!(declared instanceof List)) {
            throw error(where, name, "its metadata descriptor has no '" + DATA_NAMES + "' list");
        }
        int dot = name.lastIndexOf('.');
        if (dot <= 0 || dot == name.length() - 1) {
            throw error(where, name, "its name does not split into a module and a weight name");
        }
        String prefix = name.substring(0, dot) + "._" + name.substring(dot + 1) + "_";
        Map<String, TorchaoWeight.Inner> inner = new LinkedHashMap<>();
        for (Object d : (List<?>) declared) {
            if (!(d instanceof String) || ((String) d).isEmpty()) {
                throw error(where, name, "its '" + DATA_NAMES + "' holds " + d + " instead of a data name");
            }
            String innerName = (String) d;
            if (inner.containsKey(innerName)) {
                throw error(where, name, "its '" + DATA_NAMES + "' lists '" + innerName + "' twice");
            }
            String raw = prefix + innerName;
            StTensor tensor = tensors.tensor(raw);
            if (tensor == null) {
                throw error(where, name, "its '" + innerName + "' data tensor is stored as '" + raw
                        + "', which the checkpoint does not hold");
            }
            inner.put(innerName, new TorchaoWeight.Inner(tensor.dtype(), tensor.shape(), tensors.source(raw),
                    raw));
        }
        Map<String, Object> attributes = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) data).entrySet()) {
            if (!(e.getKey() instanceof String)) {
                throw error(where, name, "its '" + DATA + "' object has a non-string key");
            }
            if (inner.containsKey(e.getKey())) {
                throw error(where, name, "its '" + DATA + "' object repeats the data tensor '" + e.getKey() + "'");
            }
            attributes.put((String) e.getKey(), e.getValue());
        }
        return new TorchaoWeight(name, subclass, null, attributes, inner);
    }

    private static Object parse(String json, String name, String where) throws IOException {
        try {
            return Json.parse(json, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw error(where, name, "its metadata is not valid JSON: " + e.getMessage());
        }
    }

    private static Object parseMetadata(String json, String field, String where) throws IOException {
        try {
            return Json.parse(json, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw new IOException("Invalid torchao metadata in " + where + ": '" + field + "' is not valid JSON: "
                    + e.getMessage(), e);
        }
    }

    private static IOException error(String where, String name, String message) {
        return new IOException("Invalid torchao tensor '" + name + "' in " + where + ": " + message);
    }
}
