package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecValidator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The stored tensors of a model that only its host heads read: the forward backends produce the final hidden rows
 * ({@code ForwardBackend.hiddenRows}) and the heads run on the host over the mapped file ({@code CpuWeights}), so no
 * device code reads these tensors and a GPU load leaves them off the device and out of its VRAM fit. Derived from the
 * {@link ModelSpec}:
 * <ul>
 *   <li>the pooling and classifier heads ({@link PoolingHead}, {@link ClassifierHead}): the weights and biases of
 *       {@code cls}, {@code cls.norm}, {@code cls.output}, {@code dense_2} and {@code dense_3}, and the embedding
 *       output with its bias ({@link HeadSpec#embeddingOutput()}, qwen35's {@code cls.output});</li>
 *   <li>the decision heads, when the model has one ({@link DecisionScorer#present}):
 *     <ul>
 *       <li>every {@code decision.*} tensor (vllm-decision2 and clef);</li>
 *       <li>{@code token_types} when the embedding stage does not add it (the heads add one row per question type);</li>
 *       <li>the decision blocks of a modern-bert or lfm2 decision model, {@code blk.N} for
 *           {@code firstHeadLayer <= N < n_layer}, which the forward body ({@link DecisionBody}) excludes;</li>
 *       <li>clef's decoder stack {@code dec.blk.N} when no forward layer runs that stack ({@link ClefDecoder}).</li>
 *     </ul></li>
 * </ul>
 * The predicate takes stored tensor names, as the file holds them. Immutable.
 */
public final class HostHeadTensors implements Predicate<String> {

    /** Model-level bases of the pooling and classifier heads, each read with its weight and its bias. */
    private static final List<String> HEAD_BASES = Collections.unmodifiableList(Arrays.asList(
            TensorNames.CLS, TensorNames.CLS_NORM, TensorNames.CLS_OUT, TensorNames.DENSE_2_OUT,
            TensorNames.DENSE_3_OUT));

    /** Prefix of the decision tensors of vllm-decision2 and clef. */
    static final String DECISION_PREFIX = "decision.";

    private static final HostHeadTensors NONE = new HostHeadTensors(Collections.<String>emptySet(),
            Collections.<String>emptyList(), 0, 0);

    private final Set<String> names;
    private final List<String> prefixes;
    private final int blockFirst;
    private final int blockEnd;

    private HostHeadTensors(Set<String> names, List<String> prefixes, int blockFirst, int blockEnd) {
        this.names = names;
        this.prefixes = prefixes;
        this.blockFirst = blockFirst;
        this.blockEnd = blockEnd;
    }

    /** The host-only head tensors of {@code spec}. */
    public static HostHeadTensors of(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        TensorCatalog tc = spec.tensors();
        HeadSpec head = spec.head();
        Set<String> names = new LinkedHashSet<>();
        for (String base : HEAD_BASES) {
            addStored(names, tc, TensorNames.weight(base));
            addStored(names, tc, TensorNames.bias(base));
        }
        if (head != null) {
            addRef(names, head.embeddingOutput());
            addRef(names, head.embeddingOutputBias());
        }
        List<String> prefixes = new ArrayList<>(2);
        int first = 0;
        int end = 0;
        if (DecisionScorer.present(spec)) {
            prefixes.add(DECISION_PREFIX);
            if (spec.embedding() == null || spec.embedding().tokenTypes() == null) {
                addStored(names, tc, TensorNames.weight(TensorNames.TOKEN_TYPES));
            }
            if (DecisionBody.applies(spec)) {
                first = ModernBertDecisionHead.firstHeadLayer(spec.hparams());
                end = spec.hparams().nLayer();
            }
            if (spec.arch() == Arch.CLEF && !runsStack(spec, TensorNames.STACK_DEC)) {
                prefixes.add(TensorNames.STACK_DEC + ".");
            }
        }
        if (names.isEmpty() && prefixes.isEmpty() && end <= first) {
            return NONE;
        }
        return new HostHeadTensors(Collections.unmodifiableSet(names), Collections.unmodifiableList(prefixes),
                first, end);
    }

    /** True when tensor {@code storedName} is read only by the host heads. */
    @Override
    public boolean test(String storedName) {
        Objects.requireNonNull(storedName, "storedName");
        if (names.contains(storedName)) {
            return true;
        }
        for (String p : prefixes) {
            if (storedName.startsWith(p)) {
                return true;
            }
        }
        return blockEnd > blockFirst && ModelSpecValidator.isTrunkTensor(storedName, blockEnd)
                && !ModelSpecValidator.isTrunkTensor(storedName, blockFirst);
    }

    /** True when the model has no host-only head tensor. */
    public boolean isEmpty() {
        return this == NONE;
    }

    /** The exact stored names of the host-only head tensors (the prefixed and block-ranged ones aside). */
    public Set<String> names() {
        return names;
    }

    private static boolean runsStack(ModelSpec spec, String stack) {
        for (List<LayerSpec> layers : Arrays.asList(spec.layers(), spec.nextnLayers(), spec.encoderLayers())) {
            for (LayerSpec l : layers) {
                if (stack.equals(l.stack())) {
                    return true;
                }
            }
        }
        return false;
    }

    private static void addStored(Set<String> names, TensorCatalog tc, String name) {
        addRef(names, tc.get(name));
    }

    private static void addRef(Set<String> names, TensorRef ref) {
        if (ref != null) {
            names.add(ref.name());
        }
    }

    @Override
    public String toString() {
        return "HostHeadTensors{" + names + (prefixes.isEmpty() ? "" : ", prefixes=" + prefixes)
                + (blockEnd > blockFirst ? ", blocks=[" + blockFirst + ", " + blockEnd + ")" : "") + '}';
    }
}
