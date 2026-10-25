package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Helpers the Hugging Face hook packs share: name surgery, required hyperparameters, the vocabulary writers' input,
 * and the per-layer expert collection most mixture-of-experts converter classes implement in {@code modify_tensors}.
 */
public final class HfHookSupport {

    private HfHookSupport() {
    }

    /**
     * {@code name} with each of {@code prefixes} removed in turn when the name starts with it, as successive
     * {@code str.removeprefix} calls.
     */
    public static String stripPrefixes(String name, String... prefixes) {
        String out = Objects.requireNonNull(name, "name");
        for (String p : prefixes) {
            if (!p.isEmpty() && out.startsWith(p)) {
                out = out.substring(p.length());
            }
        }
        return out;
    }

    /**
     * The block number of {@code hfName} ({@link HfLoadContext#bidOf}), required.
     *
     * @throws GgufFormatException when the name has no block number
     */
    public static int layerOf(String hfName) {
        int bid = HfLoadContext.bidOf(hfName);
        if (bid < 0) {
            throw new GgufFormatException("tensor '" + hfName + "' has no layer number");
        }
        return bid;
    }

    /**
     * {@code name} with its suffix {@code oldTail} replaced by {@code newTail}.
     *
     * @throws IllegalArgumentException when the name does not end with {@code oldTail}
     */
    public static String renameTail(String name, String oldTail, String newTail) {
        Objects.requireNonNull(newTail, "newTail");
        if (!name.endsWith(oldTail)) {
            throw new IllegalArgumentException("tensor '" + name + "' does not end with '" + oldTail + "'");
        }
        return name.substring(0, name.length() - oldTail.length()) + newTail;
    }

    /**
     * The first present of {@code aliases} as an {@code int}, required ({@code self.find_hparam(aliases)} without
     * {@code optional}).
     *
     * @throws GgufFormatException naming the configuration and the aliases when none has a value
     */
    public static int requireInt(HfConfig cfg, String... aliases) {
        return cfg.requireInt(aliases);
    }

    /**
     * True for the norm weights the Gemma-style classes shift by one ({@code name.endswith("norm.weight")},
     * LC/conversion/gemma.py:67, :119; nemotron.py:401; minimax.py:202).
     */
    public static boolean isNorm(String hfName) {
        return hfName.endsWith("norm.weight");
    }

    /**
     * The vocabulary writers' input for {@code ctx}: its configuration and tokenizer files, the exact pre-tokenizer
     * specs, the GGUF architecture, and the row count of the index tensor that becomes {@code token_embd.weight} when
     * there is one. Every hook pack builds its writer input this way, adding its class's fallback chat template with
     * {@link HfVocabWriters.Input#withFallbackChatTemplate} where the class has one.
     *
     * @throws GgufFormatException when the token embedding has no rows or more than an {@code int} holds
     */
    public static HfVocabWriters.Input vocabInput(HfLoadContext ctx) {
        return vocabInput(ctx, ctx.tokenizerFiles(), ctx.config());
    }

    /**
     * As {@link #vocabInput(HfLoadContext)} over other tokenizer files and writer configuration (a draft that writes
     * its target's tokenizer).
     *
     * @throws GgufFormatException when the token embedding has no rows or more than an {@code int} holds
     */
    public static HfVocabWriters.Input vocabInput(HfLoadContext ctx, HfTokenizerFiles files, HfConfig vocabConfig) {
        HfVocabWriters.Input in = HfVocabWriters.Input.of(vocabConfig, files, PreTokenizerCanonicalizer::toSpec)
                .withGgufArch(ctx.arch().ggufName());
        OptionalInt rows = embeddingRows(ctx);
        return rows.isPresent() ? in.withEmbeddingRows(rows.getAsInt()) : in;
    }

    /**
     * The rows of the index tensor that maps to {@code token_embd.weight}, when the index has one.
     *
     * @throws GgufFormatException when that tensor has no rows or more than an {@code int} holds
     */
    public static OptionalInt embeddingRows(HfLoadContext ctx) {
        String target = TensorNames.weight(TensorNames.TOKEN_EMBD);
        HfTensorNameMap names = ctx.names();
        HfTensorIndex index = ctx.index();
        for (String name : index.names()) {
            Optional<HfTensorNameMap.Mapped> m = names.map(name, ".weight", ".bias");
            if (m.isPresent() && target.equals(m.get().ggufName())) {
                return OptionalInt.of(embeddingRows(name, index.get(name)));
            }
        }
        return OptionalInt.empty();
    }

    /**
     * The outer dimension of the token embedding {@code src} named {@code name}.
     *
     * @throws GgufFormatException when it has no rows or more than an {@code int} holds
     */
    public static int embeddingRows(String name, RowSource src) {
        long[] shape = src.shape();
        if (shape.length == 0 || shape[0] <= 0 || shape[0] > Integer.MAX_VALUE) {
            throw new GgufFormatException("token embedding '" + name + "' has the unusable shape "
                    + Arrays.toString(shape));
        }
        return (int) shape[0];
    }

    /**
     * {@code data_torch + 1}: the stored-minus-one norm weights of the Gemma-style classes restored to their
     * multiplier (LC/conversion/gemma.py:67, :119; nemotron.py:401), computed in the precision the converter adds it:
     * F16 and F32 norms in their own format, every other dtype in F32 and rounded only into the planned type
     * ({@code prepare_tensors} hands its hooks those as F32, base.py:1229-1231; see {@link ElementTransforms}).
     */
    public static RowSource normPlusOne(RowSource src) {
        return ElementTransforms.addScalar(Objects.requireNonNull(src, "src"), 1f);
    }

    /**
     * The checkpoint name a merged expert tensor is mapped under, from the group prefix (ending in {@code .experts}),
     * the part and the tail ({@code .weight}, or empty).
     */
    @FunctionalInterface
    public interface MergedName {
        /**
         * {@code <prefix>.<part><tail>}, the {@code model.layers.<b>.mlp.experts.<part>.weight} most converter classes
         * merge into (for example LC/conversion/deepseek.py:206, glm.py:229).
         */
        MergedName SAME_PREFIX = (prefix, part, tail) -> prefix + "." + part + tail;

        String of(String prefix, String part, String tail);
    }

    /**
     * Stacks per-expert checkpoint tensors {@code <prefix>.experts.<x>.<part><tail>} into one tensor per part and
     * layer, as the converter classes do with their {@code _experts} buffers (for example
     * LC/conversion/llama.py:268-295): the stacked tensors of a layer are planned, part by part in the configured
     * order, when the layer's last member (in index order) is offered, under the GGUF name of the merged checkpoint
     * name. The first time a member of a layer is offered, every member must be in the index, and all members except
     * the last are consumed.
     *
     * <p>One collector serves one load: hooks keep it in {@link HfLoadContext#state}. Not thread-safe; planning is
     * sequential.</p>
     */
    public static final class ExpertCollector {

        private static final Pattern MEMBER = Pattern.compile("^(.*\\.experts)\\.(\\d{1,9})\\.([^.]+)(\\..*)?$");

        private final int nExperts;
        private final List<String> parts;
        private final Function<List<RowSource>, RowSource> stacker;
        private final MergedName mergedName;
        private final Map<String, Group> groups = new LinkedHashMap<>();
        private HfTensorIndex positionsOf;
        private Map<String, Integer> positions;

        /**
         * A collector of {@code nExperts} experts with the given parts, stacked with {@link ExpertTransforms#stack}.
         *
         * @param mergedName the merged checkpoint name whose GGUF name the stacked tensor gets
         * @throws GgufFormatException      when {@code nExperts} is outside {@code [1, HfConfig.MAX_EXPERTS]}
         * @throws IllegalArgumentException when {@code parts} is empty or holds an empty, dotted or repeated part
         */
        public ExpertCollector(int nExperts, List<String> parts, MergedName mergedName) {
            this(nExperts, parts, ExpertTransforms::stack, mergedName);
        }

        /**
         * A collector of {@code nExperts} experts with the given parts and a custom stacking step, for classes that
         * reshape the stacked tensor before it is planned.
         *
         * @param stacker    stacks the experts' sources in expert order into one source with a new outer dimension
         *                   ({@code ExpertTransforms::stack})
         * @param mergedName the merged checkpoint name whose GGUF name the stacked tensor gets
         * @throws GgufFormatException      when {@code nExperts} is outside {@code [1, HfConfig.MAX_EXPERTS]}
         * @throws IllegalArgumentException when {@code parts} is empty or holds an empty, dotted or repeated part
         */
        public ExpertCollector(int nExperts, List<String> parts, Function<List<RowSource>, RowSource> stacker,
                               MergedName mergedName) {
            if (nExperts < 1 || nExperts > HfConfig.MAX_EXPERTS) {
                throw new GgufFormatException("expert count " + nExperts + " is outside [1, " + HfConfig.MAX_EXPERTS
                        + "]");
            }
            Objects.requireNonNull(parts, "parts");
            if (parts.isEmpty() || new LinkedHashSet<>(parts).size() != parts.size()) {
                throw new IllegalArgumentException("expert parts must be non-empty and distinct: " + parts);
            }
            for (String p : parts) {
                if (p == null || p.isEmpty() || p.indexOf('.') >= 0) {
                    throw new IllegalArgumentException("invalid expert part '" + p + "'");
                }
            }
            this.nExperts = nExperts;
            this.parts = Collections.unmodifiableList(new ArrayList<>(parts));
            this.stacker = Objects.requireNonNull(stacker, "stacker");
            this.mergedName = Objects.requireNonNull(mergedName, "mergedName");
        }

        /**
         * Takes {@code hfName} when it is an expert member of a configured part: consumes or stacks as described
         * above and returns true. Returns false for any other tensor, which the hook plans itself.
         *
         * @throws GgufFormatException when the expert index is out of range or a member of the layer is missing
         */
        public boolean offer(HfLoadContext ctx, String hfName, TensorSink out) {
            Matcher m = MEMBER.matcher(hfName);
            if (!m.matches() || !parts.contains(m.group(3))) {
                return false;
            }
            int xid = Integer.parseInt(m.group(2));
            if (xid >= nExperts) {
                throw new GgufFormatException("tensor '" + hfName + "': expert " + xid + " is outside [0, " + nExperts
                        + ")");
            }
            String prefix = m.group(1);
            String tail = m.group(4) == null ? "" : m.group(4);
            String key = prefix + "\u0000" + tail;
            Group g = groups.get(key);
            if (g == null) {
                g = open(ctx, prefix, tail, out);
                groups.put(key, g);
            }
            if (hfName.equals(g.last)) {
                emit(ctx, g, out);
            }
            return true;
        }

        /**
         * Checks that every layer whose members were offered was planned.
         *
         * @throws GgufFormatException naming the first incomplete layer
         */
        public void requireComplete() {
            for (Group g : groups.values()) {
                if (!g.done) {
                    throw new GgufFormatException("experts of '" + g.prefix + "' were not all planned (missing '"
                            + g.last + "')");
                }
            }
        }

        private Group open(HfLoadContext ctx, String prefix, String tail, TensorSink out) {
            HfTensorIndex index = ctx.index();
            Map<String, Integer> pos = positions(index);
            String last = null;
            List<String> members = new ArrayList<>(nExperts * parts.size());
            for (String part : parts) {
                for (int x = 0; x < nExperts; x++) {
                    String name = member(prefix, x, part, tail);
                    Integer p = pos.get(name);
                    if (p == null) {
                        throw new GgufFormatException("experts of '" + prefix + "': tensor '" + name
                                + "' is missing (expected " + nExperts + " experts of " + parts + ")");
                    }
                    members.add(name);
                    if (last == null || p > pos.get(last)) {
                        last = name;
                    }
                }
            }
            for (String name : members) {
                if (!name.equals(last) && !out.isConsumed(name)) {
                    out.consume(name);
                }
            }
            return new Group(prefix, tail, last);
        }

        private void emit(HfLoadContext ctx, Group g, TensorSink out) {
            for (String part : parts) {
                List<RowSource> sources = new ArrayList<>(nExperts);
                for (int x = 0; x < nExperts; x++) {
                    sources.add(ctx.index().get(member(g.prefix, x, part, g.tail)));
                }
                String merged = mergedName.of(g.prefix, part, g.tail);
                out.add(ctx.names().mapOrThrow(merged), stacker.apply(Collections.unmodifiableList(sources)));
            }
            g.done = true;
        }

        private Map<String, Integer> positions(HfTensorIndex index) {
            if (positionsOf != index) {
                Map<String, Integer> m = new HashMap<>();
                List<String> names = index.names();
                for (int i = 0; i < names.size(); i++) {
                    m.put(names.get(i), i);
                }
                positions = m;
                positionsOf = index;
            }
            return positions;
        }

        private static String member(String prefix, int x, String part, String tail) {
            return prefix + "." + x + "." + part + tail;
        }

        private static final class Group {
            final String prefix;
            final String tail;
            final String last;
            boolean done;

            Group(String prefix, String tail, String last) {
                this.prefix = prefix;
                this.tail = tail;
                this.last = last;
            }
        }
    }
}
