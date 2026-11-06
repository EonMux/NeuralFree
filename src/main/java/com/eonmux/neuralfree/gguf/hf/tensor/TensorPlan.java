package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The GGUF tensors planned for a Hugging Face checkpoint, in plan order, with the metadata tensor hooks derived, the
 * activation schemes of quantized weights and the checkpoint tensors left out. Built by {@link TensorPlanBuilder};
 * immutable.
 */
public final class TensorPlan implements HfPlanView {

    private final List<PlannedTensor> tensors;
    private final Map<String, PlannedTensor> byName;
    private final Map<String, ActQuantScheme> actQuant;
    private final List<KvEntry> kv;
    private final Map<String, String> dropped;

    TensorPlan(List<PlannedTensor> tensors, Map<String, ActQuantScheme> actQuant, List<KvEntry> kv,
               Map<String, String> dropped) {
        this.tensors = Collections.unmodifiableList(new ArrayList<>(tensors));
        Map<String, PlannedTensor> m = new LinkedHashMap<>();
        for (PlannedTensor t : this.tensors) {
            if (m.put(t.ggufName(), t) != null) {
                throw new IllegalArgumentException("tensor '" + t.ggufName() + "' is planned twice");
            }
        }
        this.byName = Collections.unmodifiableMap(m);
        this.actQuant = Collections.unmodifiableMap(new LinkedHashMap<>(actQuant));
        this.kv = Collections.unmodifiableList(new ArrayList<>(kv));
        this.dropped = Collections.unmodifiableMap(new LinkedHashMap<>(dropped));
    }

    /** The planned tensors, in plan order. */
    public List<PlannedTensor> tensors() {
        return tensors;
    }

    /**
     * The planned tensor {@code ggufName}.
     *
     * @throws IllegalArgumentException when the name is not planned
     */
    public PlannedTensor tensor(String ggufName) {
        PlannedTensor t = byName.get(ggufName);
        if (t == null) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' is not planned");
        }
        return t;
    }

    /** GGUF weight name to the activation quantization scheme the checkpoint applies before it, in plan order. */
    public Map<String, ActQuantScheme> actQuantEntries() {
        return actQuant;
    }

    /** The metadata keys tensor hooks recorded, in record order. */
    public List<KvEntry> kv() {
        return kv;
    }

    /** Checkpoint tensor name to the reason it is left out, in record order. */
    public Map<String, String> dropped() {
        return dropped;
    }

    /**
     * The plan of the two planning passes {@code first} and {@code second}: the tensors, activation entries, metadata
     * keys and dropped names of both, the first pass's first. A tensor name, activation entry or metadata key both
     * passes hold is refused: the passes' tensor hooks plan disjoint name spaces, so an overlap is a broken pairing.
     *
     * @throws IllegalArgumentException when a planned tensor name, an activation-quantization entry or a metadata key
     *                                  is recorded by both passes
     */
    public static TensorPlan merge(TensorPlan first, TensorPlan second) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        List<PlannedTensor> tensors = new ArrayList<>(first.tensors);
        Set<String> tensorNames = new HashSet<>();
        for (PlannedTensor t : tensors) {
            tensorNames.add(t.ggufName());
        }
        Map<String, ActQuantScheme> actQuant = new LinkedHashMap<>(first.actQuant);
        for (Map.Entry<String, ActQuantScheme> e : second.actQuant.entrySet()) {
            if (actQuant.putIfAbsent(e.getKey(), e.getValue()) != null) {
                throw new IllegalArgumentException("the activation quantization of '" + e.getKey()
                        + "' is recorded by both passes");
            }
        }
        List<KvEntry> kv = new ArrayList<>(first.kv);
        Set<String> kvKeys = new HashSet<>();
        for (KvEntry e : kv) {
            kvKeys.add(e.key());
        }
        for (KvEntry e : second.kv) {
            if (!kvKeys.add(e.key())) {
                throw new IllegalArgumentException("the metadata key '" + e.key() + "' is recorded by both passes");
            }
            kv.add(e);
        }
        for (PlannedTensor t : second.tensors) {
            if (!tensorNames.add(t.ggufName())) {
                throw new IllegalArgumentException("tensor '" + t.ggufName() + "' is planned by both passes");
            }
            tensors.add(t);
        }
        Map<String, String> dropped = new LinkedHashMap<>(first.dropped);
        for (Map.Entry<String, String> e : second.dropped.entrySet()) {
            dropped.putIfAbsent(e.getKey(), e.getValue());
        }
        return new TensorPlan(tensors, actQuant, kv, dropped);
    }

    @Override
    public List<String> names() {
        return Collections.unmodifiableList(new ArrayList<>(byName.keySet()));
    }

    @Override
    public boolean has(String ggufName) {
        return byName.containsKey(ggufName);
    }

    @Override
    public long[] ne(String ggufName) {
        return tensor(ggufName).ne();
    }

    @Override
    public GGMLType type(String ggufName) {
        return tensor(ggufName).type();
    }

    @Override
    public float[] readF32(String ggufName, int maxElements) throws IOException {
        PlannedTensor t = checkedSize(ggufName, maxElements);
        RowSource s = t.source();
        float[] out = new float[(int) t.elements()];
        long[] shape = s.shape();
        int inner = (int) (shape.length == 0 ? 1 : shape[shape.length - 1]);
        ByteBuffer b = RowTransforms.readRows(s, 0, Math.toIntExact(s.rows()));
        for (int r = 0; r < s.rows(); r++) {
            DtypePolicy.decode(s.format(), b, (int) (r * s.rowBytes()), out, r * inner, inner);
        }
        return out;
    }

    @Override
    public long[] readI64(String ggufName, int maxElements) throws IOException {
        PlannedTensor t = checkedSize(ggufName, maxElements);
        GGMLType type = t.type();
        int width;
        switch (type) {
            case I8:
                width = 1;
                break;
            case I16:
                width = 2;
                break;
            case I32:
                width = 4;
                break;
            case I64:
                width = 8;
                break;
            default:
                throw new IllegalArgumentException("tensor '" + ggufName + "' is " + type + ", not an integer tensor");
        }
        RowSource s = t.source();
        ByteBuffer b = RowTransforms.readRows(s, 0, Math.toIntExact(s.rows()));
        long[] out = new long[(int) t.elements()];
        for (int i = 0; i < out.length; i++) {
            out[i] = width == 1 ? b.get() : width == 2 ? b.getShort() : width == 4 ? b.getInt() : b.getLong();
        }
        return out;
    }

    private PlannedTensor checkedSize(String ggufName, int maxElements) {
        if (maxElements < 0 || maxElements > MAX_READ_ELEMENTS) {
            throw new IllegalArgumentException("maxElements must be in [0, " + MAX_READ_ELEMENTS + "], got "
                    + maxElements);
        }
        PlannedTensor t = tensor(ggufName);
        if (t.elements() > maxElements) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' has " + t.elements()
                    + " elements, more than the " + maxElements + " the caller reads");
        }
        return t;
    }

    @Override
    public String toString() {
        return "TensorPlan{" + tensors.size() + " tensors, " + kv.size() + " keys, " + dropped.size() + " dropped}";
    }

    /**
     * One metadata key recorded by a tensor hook. The value follows the GGUF reader's model: 8-, 16- and 32-bit
     * integers as Integer (UINT32 as its bit pattern), 64-bit as Long, FLOAT32 as Float, FLOAT64 as Double, BOOL as
     * Boolean, STRING as String, and an ARRAY as an unmodifiable List of those. Immutable.
     */
    public static final class KvEntry {
        private final String key;
        private final GGUFValueType type;
        private final GGUFValueType elementType;
        private final Object value;

        KvEntry(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            this.key = Objects.requireNonNull(key, "key");
            this.type = Objects.requireNonNull(type, "type");
            this.elementType = elementType;
            this.value = Objects.requireNonNull(value, "value");
        }

        public String key() {
            return key;
        }

        public GGUFValueType type() {
            return type;
        }

        /** The element type of an ARRAY, null otherwise. */
        public GGUFValueType elementType() {
            return elementType;
        }

        public Object value() {
            return value;
        }

        @Override
        public String toString() {
            return key + " (" + type + (elementType == null ? "" : " of " + elementType) + ") = " + value;
        }
    }
}
