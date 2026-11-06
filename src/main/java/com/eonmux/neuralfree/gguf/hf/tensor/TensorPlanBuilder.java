package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Collects the decisions of the tensor hooks into a {@link TensorPlan}. Each added tensor is validated (a unique
 * name, rank 1 to 4, rows of whole blocks of the planned type) and given its GGUF type by the {@link DtypePolicy} and
 * the architecture's {@link HfForceType}, converted when that type differs from the source's.
 *
 * <p>Sibling values of quantized sources become sibling tensors on projections the runtime gives them a slot
 * ({@link TensorClaims#scaledBase}): {@code <base>.scale} (F32), {@code <base>.input_scale} (F32),
 * {@code <base>.input_zero_point} (I32), {@code <base>.input_perm} (I32) and {@code <base>.input_pre_scale} (F32).
 * Elsewhere the output scale and the permutation are folded into the weight, and a static activation scheme is
 * dropped (one warning per plan). An activation scheme kept for a weight is recorded in
 * {@link TensorPlan#actQuantEntries()}.</p>
 *
 * <p>A {@link LoraFactorSource} is planned as its two factors {@code <name>.lora_a} and {@code <name>.lora_b}, with
 * {@code A} transposed for {@code token_embd.weight} (LC/convert_lora_to_gguf.py:521-529). Not thread-safe; one
 * builder serves one planning pass.</p>
 */
public final class TensorPlanBuilder implements TensorSink {

    private static final Logger LOG = LogManager.getLogger(TensorPlanBuilder.class);

    private final DtypePolicy policy;
    private final HfForceType forceType;
    private final List<PlannedTensor> tensors = new ArrayList<>();
    private final Set<String> names = new HashSet<>();
    private final Map<String, ActQuantScheme> actQuant = new LinkedHashMap<>();
    private final List<TensorPlan.KvEntry> kv = new ArrayList<>();
    private final Set<String> kvKeys = new HashSet<>();
    private final Map<String, String> dropped = new LinkedHashMap<>();
    private final Set<String> consumed = new HashSet<>();
    private final List<String> droppedSchemes = new ArrayList<>();

    public TensorPlanBuilder(DtypePolicy policy, HfForceType forceType) {
        this.policy = Objects.requireNonNull(policy, "policy");
        this.forceType = Objects.requireNonNull(forceType, "forceType");
    }

    @Override
    public void add(String ggufName, RowSource src) {
        Objects.requireNonNull(ggufName, "ggufName");
        Objects.requireNonNull(src, "src");
        if (src instanceof LoraFactorSource) {
            add(ggufName, (LoraFactorSource) src);
            return;
        }
        requireNew(ggufName);
        String base = TensorClaims.scaledBase(ggufName, src.format().ggmlType());
        RowSource s = src;
        List<Sibling> siblings = new ArrayList<>();
        if (s.inputPerm() != null) {
            if (base != null && s.shape().length == 2) {
                siblings.add(new Sibling(base + "." + TensorClaims.INPUT_PERM, permSource(s)));
            } else {
                s = ExpertTransforms.Siblings.withoutPerm(s);
            }
        }
        activation(ggufName, base, s, siblings);
        preScale(ggufName, base, s, siblings);
        float[] os = s.outputScale();
        if (os != null) {
            requirePerTensorOrExpert(ggufName, s, os.length, "output scale");
            if (base != null) {
                siblings.add(new Sibling(base + "." + TensorClaims.SCALE, RopeTensors.fromFloats(new long[]{os.length},
                        os)));
            } else {
                s = ExpertTransforms.Siblings.withoutScale(s);
            }
        }
        plan(ggufName, s);
        for (Sibling sib : siblings) {
            add(sib.name, sib.source);
        }
    }

    /**
     * Plans the factors of a LoRA delta as {@code <ggufName>.lora_a} ({@code ne {in, r}}) and
     * {@code <ggufName>.lora_b} ({@code ne {r, out}}).
     */
    public void add(String ggufName, LoraFactorSource delta) {
        RowSource a = delta.a();
        if (ggufName.contains("token_embd.weight")) {
            a = ExpertTransforms.transposeLast2(a);
        }
        add(ggufName + ".lora_a", a);
        add(ggufName + ".lora_b", delta.b());
    }

    @Override
    public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(type, "type");
        if (kvKeys.contains(key)) {
            throw new IllegalArgumentException("metadata key '" + key + "' is already recorded");
        }
        Object v;
        if (type == GGUFValueType.ARRAY) {
            if (elementType == null || elementType == GGUFValueType.ARRAY) {
                throw new IllegalArgumentException("metadata key '" + key + "': an array needs a scalar element type,"
                        + " got " + elementType);
            }
            v = normalizeArray(key, elementType, value);
        } else {
            if (elementType != null) {
                throw new IllegalArgumentException("metadata key '" + key + "': element type " + elementType
                        + " given for the scalar type " + type);
            }
            v = scalar(key, type, value);
        }
        kvKeys.add(key);
        kv.add(new TensorPlan.KvEntry(key, type, elementType, v));
    }

    @Override
    public void drop(String hfName, String reason) {
        Objects.requireNonNull(hfName, "hfName");
        dropped.putIfAbsent(hfName, Objects.requireNonNull(reason, "reason"));
    }

    @Override
    public boolean isConsumed(String hfName) {
        return consumed.contains(hfName);
    }

    @Override
    public void consume(String hfName) {
        consumed.add(Objects.requireNonNull(hfName, "hfName"));
    }

    /** The plan of every tensor added so far. */
    public TensorPlan build() {
        if (!droppedSchemes.isEmpty()) {
            LOG.warn("static activation quantization dropped for {} weights without an activation-scale slot"
                    + " (first: {}); they load weight-only", droppedSchemes.size(), droppedSchemes.get(0));
        }
        return new TensorPlan(tensors, actQuant, kv, dropped);
    }

    // ------------------------------------------------------------------ tensors

    private void requireNew(String ggufName) {
        if (ggufName.isEmpty()) {
            throw new IllegalArgumentException("a planned tensor needs a name");
        }
        if (names.contains(ggufName)) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' is already planned");
        }
    }

    /** Types, validates and records {@code src} as {@code ggufName}. */
    private void plan(String ggufName, RowSource src) {
        requireNew(ggufName);
        long[] shape = src.shape();
        int nDims = Math.max(1, shape.length);
        if (shape.length > GGUFTensorInfo.MAX_DIMS) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' from " + src.describe() + " has "
                    + shape.length + " dimensions; GGUF allows at most " + GGUFTensorInfo.MAX_DIMS);
        }
        GGMLType forced = forceType.force(ggufName, nDims, src.format(), policy.outType());
        RowSource typed = policy.apply(ggufName, src, forced);
        GGMLType type = typed.format().ggmlType();
        long[] ne = PlannedTensor.neOf(typed.shape());
        if (ne[0] % type.getBlockSize() != 0) {
            throw new IllegalArgumentException("tensor '" + ggufName + "': ne[0] = " + ne[0]
                    + " is not a multiple of the " + type + " block size " + type.getBlockSize());
        }
        long bytes = Math.multiplyExact(typed.rows(), typed.rowBytes());
        tensors.add(new PlannedTensor(ggufName, typed, type, ne, bytes));
        names.add(ggufName);
    }

    /** Records the activation scheme of {@code s} and queues its activation siblings. */
    private void activation(String ggufName, String base, RowSource s, List<Sibling> siblings) {
        ActQuantScheme scheme = s.actQuant();
        float[] scale = s.inputScale();
        int[] zp = s.inputZeroPoint();
        if (scheme == null) {
            if (scale != null || zp != null) {
                throw new IllegalArgumentException("tensor '" + ggufName + "' from " + s.describe()
                        + " has activation scales without an activation quantization scheme");
            }
            return;
        }
        boolean needsScale = scheme.isStatic() || scheme.needsGlobalScale();
        if (needsScale && (scale == null || (scheme.needsZeroPoint() && zp == null))) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' from " + s.describe() + ": activation"
                    + " scheme " + scheme + " needs " + (scale == null ? "input_scale" : "input_zero_point"));
        }
        if (base == null) {
            if (needsScale) {
                droppedSchemes.add(ggufName);
                return;
            }
            actQuant.put(ggufName, scheme);
            return;
        }
        if (scale != null) {
            requirePerTensorOrExpert(ggufName, s, scale.length, "activation scale");
            siblings.add(new Sibling(base + "." + TensorClaims.INPUT_SCALE,
                    RopeTensors.fromFloats(new long[]{scale.length}, scale)));
        }
        if (zp != null) {
            requirePerTensorOrExpert(ggufName, s, zp.length, "activation zero point");
            siblings.add(new Sibling(base + "." + TensorClaims.INPUT_ZERO_POINT,
                    ElementTransforms.keepInteger(RopeTensors.fromInts(new long[]{zp.length}, zp))));
        }
        actQuant.put(ggufName, scheme);
    }

    /** Queues the {@code <base>.input_pre_scale} sibling of {@code s} when it carries one. */
    private static void preScale(String ggufName, String base, RowSource s, List<Sibling> siblings) {
        float[] pre = s.inputPreScale();
        if (pre == null) {
            return;
        }
        long[] shape = s.shape();
        int rank = Math.max(1, shape.length);
        long k = shape[rank - 1];
        long experts = shape.length >= 3 ? shape[0] : 0;
        long want = experts > 0 ? Math.multiplyExact(experts, k) : k;
        if (pre.length != k && pre.length != want) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' from " + s.describe() + ": "
                    + pre.length + " activation pre-scale values for " + k + " columns"
                    + (experts > 0 ? " or " + want + " (one per expert)" : ""));
        }
        if (base == null) {
            throw new IllegalArgumentException("tensor '" + ggufName + "' from " + s.describe() + " has an"
                    + " activation pre-scale, but the runtime gives no weight outside a blk.N.<projection> slot an"
                    + " input_pre_scale sibling");
        }
        long[] preShape = experts > 0 && pre.length == want ? new long[]{experts, k} : new long[]{k};
        siblings.add(new Sibling(base + "." + TensorClaims.INPUT_PRE_SCALE,
                RopeTensors.fromFloats(preShape, pre)));
    }

    /**
     * The {@code <base>.input_perm} sibling of {@code s}.
     *
     * @throws IllegalArgumentException when the permutation does not cover the input dimension or is not a
     *                                  permutation of its indices
     */
    private static RowSource permSource(RowSource s) {
        int[] perm = s.inputPerm();
        long[] shape = s.shape();
        if (perm.length != shape[shape.length - 1]) {
            throw new IllegalArgumentException(s.describe() + ": input permutation of " + perm.length
                    + " entries for rows of " + shape[shape.length - 1]);
        }
        boolean[] seen = new boolean[perm.length];
        for (int p : perm) {
            if (p < 0 || p >= perm.length || seen[p]) {
                throw new IllegalArgumentException(s.describe() + ": input_perm is not a permutation of 0.."
                        + (perm.length - 1));
            }
            seen[p] = true;
        }
        return ElementTransforms.keepInteger(RopeTensors.fromInts(new long[]{perm.length}, perm));
    }

    /**
     * Requires {@code n} sibling values of {@code s} to be one per tensor, or one per expert of a stacked tensor
     * (rank 3 or more, {@code n == shape[0]}).
     */
    private static void requirePerTensorOrExpert(String ggufName, RowSource s, int n, String what) {
        long[] shape = s.shape();
        if (n == 1 || (shape.length >= 3 && n == shape[0])) {
            return;
        }
        throw new IllegalArgumentException("tensor '" + ggufName + "' from " + s.describe() + ": " + n + " " + what
                + " values for the shape " + Arrays.toString(shape) + "; expected 1, or 1 per expert");
    }

    /** A sibling tensor queued behind its weight. */
    private static final class Sibling {
        final String name;
        final RowSource source;

        Sibling(String name, RowSource source) {
            this.name = name;
            this.source = source;
        }
    }

    // ------------------------------------------------------------------ metadata values

    private static Object normalizeArray(String key, GGUFValueType elementType, Object value) {
        List<Object> out = new ArrayList<>();
        if (value instanceof List) {
            for (Object o : (List<?>) value) {
                out.add(scalar(key, elementType, o));
            }
        } else if (value != null && value.getClass().isArray()) {
            int n = Array.getLength(value);
            for (int i = 0; i < n; i++) {
                out.add(scalar(key, elementType, Array.get(value, i)));
            }
        } else {
            throw new IllegalArgumentException("metadata key '" + key + "': an array value must be a List or an"
                    + " array, got " + (value == null ? "null" : value.getClass().getSimpleName()));
        }
        return Collections.unmodifiableList(out);
    }

    private static Object scalar(String key, GGUFValueType type, Object value) {
        if (value == null) {
            throw new IllegalArgumentException("metadata key '" + key + "' has a null " + type + " value");
        }
        switch (type) {
            case STRING:
                if (!(value instanceof String)) {
                    throw mismatch(key, type, value);
                }
                return value;
            case BOOL:
                if (!(value instanceof Boolean)) {
                    throw mismatch(key, type, value);
                }
                return value;
            case FLOAT32:
                if (!(value instanceof Number)) {
                    throw mismatch(key, type, value);
                }
                return ((Number) value).floatValue();
            case FLOAT64:
                if (!(value instanceof Number)) {
                    throw mismatch(key, type, value);
                }
                return ((Number) value).doubleValue();
            default:
                return integer(key, type, value);
        }
    }

    private static Object integer(String key, GGUFValueType type, Object value) {
        if (!(value instanceof Long || value instanceof Integer || value instanceof Short || value instanceof Byte)) {
            throw mismatch(key, type, value);
        }
        long v = ((Number) value).longValue();
        long min;
        long max;
        switch (type) {
            case UINT8:
                min = 0;
                max = 0xFF;
                break;
            case INT8:
                min = Byte.MIN_VALUE;
                max = Byte.MAX_VALUE;
                break;
            case UINT16:
                min = 0;
                max = 0xFFFF;
                break;
            case INT16:
                min = Short.MIN_VALUE;
                max = Short.MAX_VALUE;
                break;
            case UINT32:
                min = 0;
                max = 0xFFFF_FFFFL;
                break;
            case INT32:
                min = Integer.MIN_VALUE;
                max = Integer.MAX_VALUE;
                break;
            case UINT64:
            case INT64:
                return v;
            default:
                throw new IllegalArgumentException("metadata key '" + key + "': unhandled type " + type);
        }
        if (v < min || v > max) {
            throw new IllegalArgumentException("metadata key '" + key + "': " + v + " does not fit " + type);
        }
        return (int) v;
    }

    private static IllegalArgumentException mismatch(String key, GGUFValueType type, Object value) {
        return new IllegalArgumentException("metadata key '" + key + "': a " + value.getClass().getSimpleName()
                + " value does not fit " + type);
    }
}
