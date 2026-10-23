package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * One entry of an architecture's tensor list, transcribed from llama.cpp {@code load_arch_tensors}
 * ({@code create_tensor} with or without {@code TENSOR_NOT_REQUIRED}).
 *
 * <p>An entry names one or more alternatives (groups of names). A single-alternative entry is the common case.
 * Several alternatives express fused/unfused forms such as {@code attn_qkv} versus
 * {@code attn_q}+{@code attn_k}+{@code attn_v}; by default every alternative is created (each name is loaded with
 * {@code TENSOR_NOT_REQUIRED} and the entry is satisfied when one alternative is complete). A {@link #probed()} entry
 * transcribes a loader that probes instead: it creates the leading tensor of each alternative with
 * {@code TENSOR_NOT_REQUIRED} and loads only the first alternative whose leading tensor exists, or the last one when
 * none does ({@code create_tensor_qkv}, {@code create_tensor_gate_up_exps}, llama-model.cpp:3846-3898). Layer-scoped
 * names contain one {@code %d} for the layer index and apply to the layers selected by a {@link LayerCondition}
 * (trunk layers by default).
 *
 * <p>Refinements express the conditional {@code create_tensor} calls of llama.cpp:</p>
 * <ul>
 *   <li>{@link #onlyWhen(ModelCondition)}: a model-level entry applies only when the hparams select it
 *       (for example {@code if (n_embd_per_layer > 0)});</li>
 *   <li>{@link #ifPresent(String)} / {@link #ifAbsent(String)}: the entry applies only when a trigger tensor exists,
 *       or does not exist (at the same layer for layer entries), for tensors created inside
 *       {@code if (other_tensor)} / {@code if (!other_tensor)} blocks;</li>
 *   <li>{@link #overLayers(LayerSpan)}: a layer entry scans layer slots beyond {@code nLayerAll}, for blocks
 *       numbered past the trunk (decoder stacks, drafter heads). Its {@link LayerCondition} is then
 *       evaluated for slots without per-layer hparams and must not read per-layer getters there;</li>
 *   <li>{@link #withTrunkFlags()}: a model-level tensor created with the trunk flags of llama.cpp's
 *       {@code nextn_flags} (llama-model.cpp:3901-3920), which an MTP-only file need not hold. Layer entries are
 *       trunk tensors on their trunk slots without marking;</li>
 *   <li>{@link #creatingAllWhenSkipped()}: an entry of the {@code create_tensor_qkv} /
 *       {@code create_tensor_gate_up_exps} helpers, whose {@code TENSOR_SKIP} branch creates every name of every
 *       alternative without probing (llama-model.cpp:3847-3853, 3868-3880).</li>
 * </ul>
 * Immutable.
 */
public final class TensorRequirement {

    /** Whether the names are model-level or per layer. */
    public enum Scope { MODEL, LAYER }

    /** Selects the layers a layer-scoped requirement applies to. */
    @FunctionalInterface
    public interface LayerCondition {
        boolean test(ModelHparams hp, int il);
    }

    /** Selects whether a model-scoped requirement applies. */
    @FunctionalInterface
    public interface ModelCondition {
        boolean test(ModelHparams hp);
    }

    /** Number of layer slots {@code [0, count)} a layer-scoped requirement scans. */
    @FunctionalInterface
    public interface LayerSpan {
        int count(ModelHparams hp);
    }

    /** Every model. */
    public static final ModelCondition ALWAYS = hp -> true;
    /** The layers that have hparams, {@code [0, nLayerAll)}. */
    public static final LayerSpan ALL_LAYERS = ModelHparams::nLayerAll;

    /** Trunk layers {@code [0, nLayer)}, excluding NextN/MTP blocks. */
    public static final LayerCondition TRUNK = (hp, il) -> il < hp.nLayer();
    /** NextN/MTP blocks {@code [nLayer, nLayerAll)}. */
    public static final LayerCondition NEXTN = (hp, il) -> il >= hp.nLayer();
    /** Trunk attention layers (not recurrent). */
    public static final LayerCondition TRUNK_ATTENTION = (hp, il) -> il < hp.nLayer() && !hp.isRecurrent(il);
    /** Trunk recurrent layers. */
    public static final LayerCondition TRUNK_RECURRENT = (hp, il) -> il < hp.nLayer() && hp.isRecurrent(il);

    private static final String LAYER_PREFIX = "blk.%d.";

    private final Scope scope;
    private final boolean required;
    private final List<List<String>> alternatives;
    private final LayerCondition when;
    private final ModelCondition modelWhen;
    private final String trigger;
    private final boolean triggerAbsent;
    private final LayerSpan span;
    private final boolean probed;
    private final boolean allWhenSkipped;
    private final boolean trunkFlags;

    private TensorRequirement(Scope scope, boolean required, List<List<String>> alternatives, LayerCondition when,
                              ModelCondition modelWhen, String trigger, boolean triggerAbsent, LayerSpan span,
                              boolean probed, boolean allWhenSkipped, boolean trunkFlags) {
        this.scope = scope;
        this.required = required;
        this.alternatives = alternatives;
        this.when = when;
        this.modelWhen = modelWhen;
        this.trigger = trigger;
        this.triggerAbsent = triggerAbsent;
        this.span = span;
        this.probed = probed;
        this.allWhenSkipped = allWhenSkipped;
        this.trunkFlags = trunkFlags;
    }

    /** Required model-level tensor, e.g. {@code token_embd.weight}. */
    public static TensorRequirement model(String name) {
        return of(Scope.MODEL, true, null, new String[][] {{name}});
    }

    /** Optional model-level tensor. */
    public static TensorRequirement modelOptional(String name) {
        return of(Scope.MODEL, false, null, new String[][] {{name}});
    }

    /** Required {@code blk.%d.<suffix>} tensor on every trunk layer. */
    public static TensorRequirement layer(String suffix) {
        return of(Scope.LAYER, true, TRUNK, new String[][] {{LAYER_PREFIX + suffix}});
    }

    /** Required {@code blk.%d.<suffix>} tensor on the layers selected by {@code when}. */
    public static TensorRequirement layer(String suffix, LayerCondition when) {
        return of(Scope.LAYER, true, when, new String[][] {{LAYER_PREFIX + suffix}});
    }

    /** Optional {@code blk.%d.<suffix>} tensor on trunk layers. */
    public static TensorRequirement layerOptional(String suffix) {
        return of(Scope.LAYER, false, TRUNK, new String[][] {{LAYER_PREFIX + suffix}});
    }

    /**
     * Required layer entry satisfied by any one group of {@code blk.%d.<suffix>} names, e.g.
     * {@code {{"attn_qkv.weight"}, {"attn_q.weight", "attn_k.weight", "attn_v.weight"}}}.
     */
    public static TensorRequirement layerAnyOf(LayerCondition when, String[]... suffixGroups) {
        return of(Scope.LAYER, true, when, prefixed(LAYER_PREFIX, suffixGroups));
    }

    /**
     * {@code <prefix><suffix>} name groups, e.g. {@code blk.%d.} with {@code {"attn_qkv.weight"}}; the result is a
     * fresh array.
     */
    public static String[][] prefixed(String prefix, String[]... suffixGroups) {
        Objects.requireNonNull(prefix, "prefix");
        String[][] groups = new String[suffixGroups.length][];
        for (int g = 0; g < suffixGroups.length; g++) {
            groups[g] = new String[suffixGroups[g].length];
            for (int i = 0; i < suffixGroups[g].length; i++) {
                groups[g][i] = prefix + suffixGroups[g][i];
            }
        }
        return groups;
    }

    /**
     * General form with full name formats (layer formats contain one {@code %d}), for prefixes other than
     * {@code blk.} such as {@code enc.blk.%d.} and {@code dec.blk.%d.}.
     */
    public static TensorRequirement of(Scope scope, boolean required, LayerCondition when, String[][] nameGroups) {
        Objects.requireNonNull(scope, "scope");
        if (nameGroups == null || nameGroups.length == 0) {
            throw new IllegalArgumentException("a tensor requirement needs at least one name group");
        }
        List<List<String>> groups = new ArrayList<>(nameGroups.length);
        for (String[] g : nameGroups) {
            if (g == null || g.length == 0) {
                throw new IllegalArgumentException("empty tensor name group");
            }
            for (String name : g) {
                boolean layered = name.contains("%d");
                if (layered != (scope == Scope.LAYER)) {
                    throw new IllegalArgumentException("tensor name '" + name + "' does not match scope " + scope);
                }
            }
            groups.add(Collections.unmodifiableList(new ArrayList<>(Arrays.asList(g))));
        }
        LayerCondition cond = scope == Scope.LAYER ? (when != null ? when : TRUNK) : null;
        return new TensorRequirement(scope, required, Collections.unmodifiableList(groups), cond, ALWAYS, null,
                false, ALL_LAYERS, false, false, false);
    }

    /** Required model-level tensor when {@code when} holds, e.g. {@code if (n_embd_per_layer > 0)}. */
    public static TensorRequirement modelWhen(ModelCondition when, String name) {
        return model(name).onlyWhen(when);
    }

    /**
     * This model-level entry, applying only when {@code when} holds as well.
     *
     * @throws IllegalStateException when this entry is layer-scoped (use a {@link LayerCondition})
     */
    public TensorRequirement onlyWhen(ModelCondition when) {
        Objects.requireNonNull(when, "when");
        if (scope != Scope.MODEL) {
            throw new IllegalStateException("onlyWhen applies to model-level entries; " + this + " is per layer");
        }
        final ModelCondition prev = modelWhen;
        ModelCondition both = prev == ALWAYS ? when : hp -> prev.test(hp) && when.test(hp);
        return new TensorRequirement(scope, required, alternatives, this.when, both, trigger, triggerAbsent, span,
                probed, allWhenSkipped, trunkFlags);
    }

    /**
     * This entry, applying only where the tensor {@code triggerFormat} exists: a model-level name for a
     * model-level entry, a layer format with one {@code %d} (resolved at the same layer) for a layer entry.
     *
     * @throws IllegalArgumentException when the trigger's form does not match the scope
     * @throws IllegalStateException    when the entry already has a trigger
     */
    public TensorRequirement ifPresent(String triggerFormat) {
        return withTrigger(triggerFormat, false);
    }

    /**
     * This entry, applying only where the tensor {@code triggerFormat} does not exist (the {@code else} branch of
     * an {@code if (other_tensor)} block); forms as for {@link #ifPresent}.
     *
     * @throws IllegalArgumentException when the trigger's form does not match the scope
     * @throws IllegalStateException    when the entry already has a trigger
     */
    public TensorRequirement ifAbsent(String triggerFormat) {
        return withTrigger(triggerFormat, true);
    }

    private TensorRequirement withTrigger(String triggerFormat, boolean absent) {
        Objects.requireNonNull(triggerFormat, "triggerFormat");
        if (triggerFormat.contains("%d") != (scope == Scope.LAYER)) {
            throw new IllegalArgumentException("trigger '" + triggerFormat + "' does not match scope " + scope);
        }
        if (trigger != null) {
            throw new IllegalStateException(this + " already has the trigger " + trigger);
        }
        return new TensorRequirement(scope, required, alternatives, when, modelWhen, triggerFormat, absent, span,
                probed, allWhenSkipped, trunkFlags);
    }

    /**
     * This entry with probed alternatives: only the first alternative whose leading tensor exists is created, or the
     * last alternative when no leading tensor does (see the class comment).
     *
     * @throws IllegalStateException when the entry has a single alternative
     */
    public TensorRequirement probed() {
        if (alternatives.size() < 2) {
            throw new IllegalStateException("probing needs several alternatives; " + this + " has one");
        }
        return new TensorRequirement(scope, required, alternatives, when, modelWhen, trigger, triggerAbsent, span,
                true, allWhenSkipped, trunkFlags);
    }

    /**
     * This entry, marked as created by the {@code create_tensor_qkv} / {@code create_tensor_gate_up_exps} helpers:
     * on blocks loaded with {@code TENSOR_SKIP} every name of every alternative is created, whatever the probe and
     * the trigger say ({@link #createdWhenSkipped}).
     */
    public TensorRequirement creatingAllWhenSkipped() {
        return new TensorRequirement(scope, required, alternatives, when, modelWhen, trigger, triggerAbsent, span,
                probed, true, trunkFlags);
    }

    /**
     * This model-level entry, created with the trunk flags of {@code nextn_flags} (llama-model.cpp:3901-3920): it
     * is not required in an MTP-only file.
     *
     * @throws IllegalStateException when this entry is layer-scoped (its trunk slots are trunk tensors already)
     */
    public TensorRequirement withTrunkFlags() {
        if (scope != Scope.MODEL) {
            throw new IllegalStateException("withTrunkFlags applies to model-level entries; " + this + " is per layer");
        }
        return new TensorRequirement(scope, required, alternatives, when, modelWhen, trigger, triggerAbsent, span,
                probed, allWhenSkipped, true);
    }

    /**
     * This layer entry, scanning layer slots {@code [0, span.count(hp))} instead of {@code [0, nLayerAll)}.
     *
     * @throws IllegalStateException when this entry is model-level
     */
    public TensorRequirement overLayers(LayerSpan layers) {
        Objects.requireNonNull(layers, "layers");
        if (scope != Scope.LAYER) {
            throw new IllegalStateException("overLayers applies to layer entries; " + this + " is model-level");
        }
        return new TensorRequirement(scope, required, alternatives, when, modelWhen, trigger, triggerAbsent, layers,
                probed, allWhenSkipped, trunkFlags);
    }

    public Scope scope() {
        return scope;
    }

    public boolean required() {
        return required;
    }

    /** Name formats per alternative; layer formats contain {@code %d}. */
    public List<List<String>> alternatives() {
        return alternatives;
    }

    /** True when only the probed alternative is created ({@link #probed()}). */
    public boolean isProbed() {
        return probed;
    }

    /**
     * True for a {@code create_tensor_qkv} / {@code create_tensor_gate_up_exps} entry
     * ({@link #creatingAllWhenSkipped}).
     */
    public boolean createsAllWhenSkipped() {
        return allWhenSkipped;
    }

    /** True for a model-level entry created with the {@code nextn_flags} trunk flags ({@link #withTrunkFlags}). */
    public boolean hasTrunkFlags() {
        return trunkFlags;
    }

    /**
     * True when this layer-scoped entry applies to layer slot {@code il} by its hparams condition and span;
     * false for model-scoped entries. A trigger, when set, is checked separately ({@link #triggerHolds}).
     */
    public boolean appliesTo(ModelHparams hp, int il) {
        return scope == Scope.LAYER && il >= 0 && il < span.count(hp) && when.test(hp, il);
    }

    /** True when this model-scoped entry applies to the model by its hparams; false for layer entries. */
    public boolean appliesToModel(ModelHparams hp) {
        return scope == Scope.MODEL && modelWhen.test(hp);
    }

    /** The trigger tensor format set by {@link #ifPresent} or {@link #ifAbsent}, or null. */
    public String trigger() {
        return trigger;
    }

    /** True when the trigger selects the entry by the trigger tensor's absence ({@link #ifAbsent}). */
    public boolean triggerAbsent() {
        return triggerAbsent;
    }

    /**
     * True when the entry's trigger, if any, selects it at layer slot {@code il} ({@code il < 0} for a model-level
     * entry).
     *
     * @param present tells whether a tensor name exists in the file
     */
    public boolean triggerHolds(int il, Predicate<String> present) {
        if (trigger == null) {
            return true;
        }
        return present.test(il < 0 ? trigger : layerName(trigger, il)) != triggerAbsent;
    }

    /**
     * The concrete names the loader creates for this entry at layer slot {@code il} ({@code il < 0} for a
     * model-level entry) when the entry applies there by its conditions: none when the trigger does not select it,
     * the probed alternative of a {@link #probed()} entry, otherwise every alternative.
     *
     * @param present tells whether a tensor name exists in the file
     */
    public List<String> created(int il, Predicate<String> present) {
        if (!triggerHolds(il, present)) {
            return Collections.emptyList();
        }
        if (probed) {
            return names(alternatives.get(selected(il, present)), il);
        }
        return allNames(il);
    }

    /**
     * As {@link #created}, on a block loaded with {@code TENSOR_SKIP}: an entry of the qkv / gate-up helpers then
     * creates every name of every alternative ({@link #creatingAllWhenSkipped}).
     */
    public List<String> createdWhenSkipped(int il, Predicate<String> present) {
        return allWhenSkipped ? allNames(il) : created(il, present);
    }

    /** Number of layer slots a layer entry scans for this model; 0 for model-level entries. */
    public int layerSlots(ModelHparams hp) {
        return scope == Scope.LAYER ? span.count(hp) : 0;
    }

    /** Concrete tensor name of a layer format. */
    public static String layerName(String format, int il) {
        return String.format(Locale.ROOT, format, il);
    }

    /**
     * Descriptions of the unmet parts of a required entry; empty when satisfied or optional.
     *
     * @param present tells whether a tensor name exists in the file
     */
    public List<String> missing(ModelHparams hp, Predicate<String> present) {
        return missing(hp, present, false);
    }

    /**
     * Descriptions of the unmet parts of a required entry; empty when satisfied or optional. An entry with
     * alternatives is met when one alternative is complete, a {@link #probed()} entry only when its probed
     * alternative is.
     *
     * @param present      tells whether a tensor name exists in the file
     * @param trunkOptional true for an MTP-only file ({@code nextn_flags} trunk flags {@code TENSOR_NOT_REQUIRED}):
     *                     trunk slots {@code [0, nLayer)} and {@link #withTrunkFlags()} entries are not required
     */
    public List<String> missing(ModelHparams hp, Predicate<String> present, boolean trunkOptional) {
        if (!required) {
            return Collections.emptyList();
        }
        if (scope == Scope.MODEL) {
            if (!modelWhen.test(hp) || !triggerHolds(-1, present) || (trunkOptional && trunkFlags)) {
                return Collections.emptyList();
            }
            if (satisfied(-1, present)) {
                return Collections.emptyList();
            }
            return Collections.singletonList(describe(-1, present) + withTrigger(-1));
        }
        List<String> out = new ArrayList<>();
        int slots = span.count(hp);
        for (int il = trunkOptional ? Math.max(0, hp.nLayer()) : 0; il < slots; il++) {
            if (when.test(hp, il) && triggerHolds(il, present) && !satisfied(il, present)) {
                out.add(describe(il, present) + withTrigger(il));
            }
        }
        return out;
    }

    private String withTrigger(int il) {
        if (trigger == null) {
            return "";
        }
        String name = il < 0 ? trigger : layerName(trigger, il);
        return triggerAbsent ? " (required without " + name + ")" : " (required with " + name + ")";
    }

    /** Index of the alternative a probing loader creates: the first whose leading tensor exists, else the last. */
    private int selected(int il, Predicate<String> present) {
        int last = alternatives.size() - 1;
        for (int g = 0; g < last; g++) {
            String lead = alternatives.get(g).get(0);
            if (present.test(il < 0 ? lead : layerName(lead, il))) {
                return g;
            }
        }
        return last;
    }

    private boolean satisfied(int il, Predicate<String> present) {
        if (probed) {
            return complete(alternatives.get(selected(il, present)), il, present);
        }
        for (List<String> group : alternatives) {
            if (complete(group, il, present)) {
                return true;
            }
        }
        return false;
    }

    private static boolean complete(List<String> group, int il, Predicate<String> present) {
        for (String f : group) {
            if (!present.test(il < 0 ? f : layerName(f, il))) {
                return false;
            }
        }
        return true;
    }

    private List<String> allNames(int il) {
        List<String> out = new ArrayList<>();
        for (List<String> group : alternatives) {
            out.addAll(names(group, il));
        }
        return out;
    }

    private static List<String> names(List<String> group, int il) {
        List<String> out = new ArrayList<>(group.size());
        for (String f : group) {
            out.add(il < 0 ? f : layerName(f, il));
        }
        return out;
    }

    /** The alternatives an unmet entry needs: the probed one for a {@link #probed()} entry, otherwise all. */
    private String describe(int il, Predicate<String> present) {
        List<List<String>> needed = probed
                ? Collections.singletonList(alternatives.get(selected(il, present))) : alternatives;
        StringBuilder sb = new StringBuilder();
        for (int g = 0; g < needed.size(); g++) {
            if (g > 0) {
                sb.append(" or ");
            }
            List<String> group = needed.get(g);
            for (int i = 0; i < group.size(); i++) {
                if (i > 0) {
                    sb.append('+');
                }
                sb.append(il < 0 ? group.get(i) : layerName(group.get(i), il));
            }
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return (required ? "required " : "optional ") + scope + " " + alternatives
                + (probed ? " probed" : "")
                + (trigger == null ? "" : " if " + trigger + (triggerAbsent ? " absent" : " present"))
                + (trunkFlags ? " (trunk flags)" : "");
    }
}
