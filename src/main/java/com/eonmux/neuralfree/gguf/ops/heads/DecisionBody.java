package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * The forward body of a modern-bert or lfm2 decision model ({@code n_layer_decision > 0}): llama.cpp runs the trunk
 * layers before the decision blocks and the output norm (modern-bert.cpp:159-162, lfm2.cpp:317-320), and the last
 * {@code n_layer_decision} blocks form the host head ({@link ModernBertDecisionHead}, {@link Lfm2DecisionHead}). The
 * body keeps the model's embedding, which carries no {@code token_types} rows on these graphs: {@code build_inp_embd}
 * never adds them, the decision head adds one per question type (modern-bert.cpp:259-261). Shared by the GPU and CPU
 * orchestrators.
 */
public final class DecisionBody {

    private DecisionBody() {
    }

    /** True for modern-bert and lfm2 decision models, whose last {@code n_layer_decision} blocks form the head. */
    public static boolean applies(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        Arch arch = spec.arch();
        return spec.hparams().nLayerDecision() > 0 && (arch == Arch.MODERN_BERT || arch == Arch.LFM2);
    }

    /**
     * The spec a forward backend runs: the decision body for a decision model ({@link #applies}), else
     * {@code spec} itself.
     *
     * @throws IllegalStateException when the decision body does not fit a linear trunk without NextN or encoder
     *                               layers
     */
    public static ModelSpec forwardSpec(ModelSpec spec) {
        if (!applies(spec)) {
            return spec;
        }
        int body = ModernBertDecisionHead.firstHeadLayer(spec.hparams());
        if (spec.schedule().kind() != LayerSchedule.Kind.LINEAR || !spec.nextnLayers().isEmpty()
                || !spec.encoderLayers().isEmpty() || body < 1 || body > spec.nLayer()) {
            throw new IllegalStateException(spec.arch().ggufName() + ": a decision body of " + body
                    + " layers does not fit " + spec);
        }
        ModelSpec.Builder b = ModelSpec.builder(spec.hparams(), spec.tensors())
                .schedule(LayerSchedule.linear(body))
                .embedding(spec.embedding())
                .head(spec.head())
                .memoryKind(spec.memoryKind());
        for (int il = 0; il < body; il++) {
            b.layer(spec.layer(il));
        }
        return b.build();
    }
}
