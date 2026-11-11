package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.mm.ClipDualEncoder;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * The pooling, classification, reranking and scoring entry points of one {@link EngineService}: the pooled and
 * per-token embeddings, the sequence and token classifiers, the rerank inputs and scores, the decision and support
 * kinds, and the next-token logits. A CLIP dual encoder serves the text and image embeddings itself; every other
 * model runs the forward on a replica, over the batch engine when its memory is paged. The service
 * ({@link EngineService}) routes each call to its least-loaded replica and owns the dual encoder and the media
 * projector the calls read.
 */
final class EnginePooling {

    private final EngineService owner;

    EnginePooling(EngineService owner) {
        this.owner = Objects.requireNonNull(owner, "owner");
    }

    /** The pooled embedding of one input ({@code EngineModel.encode}) under {@code lora}. */
    float[] embed(LoraSelection lora, int[] tokens) throws IOException {
        ClipDualEncoder dual = owner.dual();
        if (dual != null) {
            owner.requireLoaded();
            requireNoDualAdapters(lora);
            return dual.embedText(tokens);
        }
        return (float[]) owner.poolingReplica().pooling(tokens, PoolingOutput.POOLED, lora);
    }

    /** The pooled embeddings of several inputs under {@code lora}, spread over the open replicas. */
    float[][] embedAll(LoraSelection lora, int[][] inputs) throws IOException {
        ClipDualEncoder dual = owner.dual();
        if (dual != null) {
            owner.requireLoaded();
            requireNoDualAdapters(lora);
            float[][] out = new float[inputs.length][];
            for (int i = 0; i < out.length; i++) {
                out[i] = dual.embedText(inputs[i]);
            }
            return out;
        }
        return cast(poolAcross(inputs, PoolingOutput.POOLED, lora, true));
    }

    /** The embedding of one image through the CLIP dual encoder's vision tower. */
    float[] embedImage(LoraSelection lora, String reference, String field) {
        owner.requireLoaded();
        ClipDualEncoder dual = owner.dual();
        if (dual == null) {
            throw new UnsupportedOperationException(field + " holds an image, but " + owner.architecture()
                    + " embeds text only; image embeddings need a CLIP dual-encoder model (a two-tower clip GGUF,"
                    + " or a Hugging Face CLIPModel or SiglipModel checkpoint)");
        }
        requireNoDualAdapters(lora);
        byte[] bytes = owner.media().imageBytes(reference, field);
        try {
            return dual.embedImage(bytes);
        } catch (com.eonmux.neuralfree.gguf.image.ImageDecodeException e) {
            throw new IllegalArgumentException(field + ": " + e.getMessage(), e);
        }
    }

    /** Refuses, naming {@code lora}, a selection of adapters on the CLIP dual encoder, which runs none. */
    private void requireNoDualAdapters(LoraSelection lora) {
        if (lora != null) {
            throw new UnsupportedOperationException("lora: " + owner.architecture() + " is a CLIP dual encoder,"
                    + " which runs no LoRA adapters (" + lora + ")");
        }
    }

    /** True when the model is a CLIP dual encoder: it serves the pooling routes only, and takes image inputs. */
    boolean dualEncoder() {
        return owner.dual() != null;
    }

    /** The bi-encoder score of a cosine similarity, through the dual encoder's own activation when it has one. */
    double biEncoderScore(double cosine) {
        ClipDualEncoder dual = owner.dual();
        return dual == null ? cosine : dual.score(cosine);
    }

    /** The pooled embedding of one media prompt. */
    float[] embedMedia(LoraSelection lora, MediaPrompt prompt) throws IOException {
        owner.requireLoaded();
        if (owner.dual() != null) {
            throw new UnsupportedOperationException(owner.dualRefusal());
        }
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().encodeMedia(prompt));
    }

    /** The RANK classifier's relevance score of one media prompt. */
    float rerankScoreMedia(LoraSelection lora, MediaPrompt prompt) throws IOException {
        return rerankScoreMedia(lora, prompt, true);
    }

    /** {@link #rerankScoreMedia} with the classifier's final probability activation applied or skipped. */
    float rerankScoreMedia(LoraSelection lora, MediaPrompt prompt, boolean activation) throws IOException {
        owner.requireLoaded();
        if (owner.dual() != null) {
            throw new UnsupportedOperationException(owner.dualRefusal());
        }
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().classifyOutputMedia(prompt, activation).rankScore());
    }

    /** Whether the model's embeddings are per token. */
    boolean perTokenEmbeddings() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return EncodeGate.embedsPerToken(owner.primary().model().spec());
    }

    /** The per-token embeddings of one input under {@code lora}, for NONE pooling. */
    float[][] embedTokens(LoraSelection lora, int[] tokens) throws IOException {
        return (float[][]) owner.poolingReplica().pooling(tokens, PoolingOutput.TOKEN_EMBEDDINGS, lora);
    }

    /** The per-token embeddings of several inputs, spread over the open replicas. */
    float[][][] embedTokensAll(LoraSelection lora, int[][] inputs) throws IOException {
        Object[] rows = poolAcross(inputs, PoolingOutput.TOKEN_EMBEDDINGS, lora, true);
        float[][][] out = new float[rows.length][][];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float[][]) rows[i];
        }
        return out;
    }

    /** Whether the model classifies per token. */
    boolean tokenClassifier() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return EncodeGate.hasTokenClassifier(owner.primary().model().spec());
    }

    /** Whether the model pools a vector or row before classifying. */
    boolean sequenceClassifier() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return EncodeGate.hasSequenceClassifier(owner.primary().model().spec());
    }

    /** Whether the model answers {@code token_classify}. */
    boolean canTokenClassify() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return EncodeGate.canTokenClassify(owner.primary().model().spec());
    }

    /** Whether the model stores a separate per-token score projection ({@code cls.sparse}). */
    boolean tokenScoreProjection() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return EncodeGate.hasTokenScoreProjection(owner.primary().model().spec());
    }

    /** Whether the model stores a separate per-token embedding projection ({@code cls.colbert}). */
    boolean tokenEmbeddingProjection() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return EncodeGate.hasTokenEmbeddingProjection(owner.primary().model().spec());
    }

    /** The process-reward default step filter ({@code classifier.step_tag_id}), or -1. */
    int defaultStepTagId() {
        owner.requireLoaded();
        return owner.primary().model().spec().hparams().classifierStepTagId();
    }

    /** The leading per-token boundary token id ({@code classifier.bos_token}), or -1. */
    int tokenBoundaryBos() {
        owner.requireLoaded();
        return owner.primary().model().spec().hparams().classifierBosToken();
    }

    /** The trailing per-token boundary token id ({@code classifier.eos_token}), or -1. */
    int tokenBoundaryEos() {
        owner.requireLoaded();
        return owner.primary().model().spec().hparams().classifierEosToken();
    }

    /**
     * The kind of sparse pooling over the per-token classifier scores ({@code EncodeGate.sparsePooling}), the
     * vLLM SPLADE reduction, or null when the model pools none of it.
     */
    String sparsePooling() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return null;
        }
        return com.eonmux.neuralfree.gguf.ops.heads.EncodeGate.sparsePooling(owner.primary().model().spec());
    }

    /**
     * The decision head's type ({@code <arch>.decision.type}), or the architecture's own head family when the
     * model carries no metadata key; null when the model has no decision readout: the families of
     * {@link DecisionFamily} serve the decision routes.
     */
    String decisionType() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return null;
        }
        com.eonmux.neuralfree.gguf.spec.ModelSpec spec = owner.primary().model().spec();
        String type = spec.hparams().decisionType();
        if (type != null) {
            return type;
        }
        return switch (spec.arch()) {
            case VLLM_DECISION2 -> "vllm-decision2";
            case CLEF -> "clef";
            case MODERN_BERT -> spec.hparams().nLayerDecision() > 0 ? "laya" : null;
            case LFM2 -> spec.hparams().nLayerDecision() > 0 ? "lfm2-d1" : null;
            default -> null;
        };
    }

    /** The marker head's question-and-options token budget ({@code decision.max_head_tokens}), 0 when absent. */
    int decisionMaxHeadTokens() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return 0;
        }
        return owner.primary().model().spec().hparams().decisionMaxHeadTokens();
    }

    /** The decision head's per-type temperatures ({@code decision.temperature.<name>}), unmodifiable. */
    Map<String, Float> decisionTemperatures() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return Map.of();
        }
        return owner.primary().model().spec().hparams().decisionTemperatures();
    }

    /** The decision head's score offsets by level count ({@code decision.score_bias.<n>}), unmodifiable. */
    Map<Integer, float[]> decisionScoreBias() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return Map.of();
        }
        return owner.primary().model().spec().hparams().decisionScoreBias();
    }

    /** The decision head's option labels ({@code decision.labels}), unmodifiable. */
    List<String> decisionLabels() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return List.of();
        }
        return List.of(owner.primary().model().spec().hparams().decisionLabels());
    }

    /** The classifier output rows of the head ({@code <arch>.n_embd_out}), the embedding size when unset. */
    int nEmbdOut() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return 0;
        }
        return owner.primary().model().spec().hparams().nEmbdOut();
    }

    /** The per-token classifier scores of one input under {@code lora}. */
    float[][] tokenClassify(LoraSelection lora, int[] tokens) throws IOException {
        return (float[][]) owner.poolingReplica().pooling(tokens, PoolingOutput.TOKEN_SCORES, lora);
    }

    /** The per-token classifier scores of several inputs, spread over the open replicas. */
    float[][][] tokenClassifyAll(LoraSelection lora, int[][] inputs) throws IOException {
        Object[] rows = poolAcross(inputs, PoolingOutput.TOKEN_SCORES, lora, true);
        float[][][] out = new float[rows.length][][];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float[][]) rows[i];
        }
        return out;
    }

    /** The rerank input tokens of one query-document pair. */
    int[] rerankInput(String query, String document) {
        owner.requireLoaded();
        return ClassifierHead.rerankInput(owner.primary().model().tokenizer(), query, document);
    }

    /** The rerank input of an already tokenized query and document. */
    int[] rerankInput(int[] query, int[] document) {
        owner.requireLoaded();
        return ClassifierHead.rerankInput(owner.primary().model().tokenizer(), query, document);
    }

    /** The rerank input of a query, document and task instruction. */
    int[] rerankInput(String query, String document, String instruction) {
        owner.requireLoaded();
        return ClassifierHead.rerankInput(owner.primary().model().tokenizer(), query, document, instruction);
    }

    /** The relevance score of one rerank input under {@code lora}. */
    float rerankScore(LoraSelection lora, int[] tokens) throws IOException {
        return rerankScore(lora, tokens, true);
    }

    /** {@link #rerankScore} with the classifier's final probability activation applied or skipped. */
    float rerankScore(LoraSelection lora, int[] tokens, boolean activation) throws IOException {
        return rerankScores(lora, new int[][]{tokens}, activation)[0];
    }

    /** The relevance scores of several rerank inputs over one replica. */
    float[] rerankScores(LoraSelection lora, int[][] inputs) throws IOException {
        return rerankScores(lora, inputs, true);
    }

    /** {@link #rerankScores} with the final probability activation applied or skipped per input. */
    float[] rerankScores(LoraSelection lora, int[][] inputs, boolean activation) throws IOException {
        return castScores(poolAcross(inputs, PoolingOutput.RERANK_SCORE, lora, activation));
    }

    /** The model's pooling kind. */
    com.eonmux.neuralfree.gguf.arch.PoolingKind poolingKind() {
        owner.requireLoaded();
        ClipDualEncoder dual = owner.dual();
        if (dual != null) {
            return dual.pooling();
        }
        return owner.primary().model().spec().head().pooling();
    }

    /** Whether the model carries a decision head scoring per token. */
    boolean decisionHead() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return false;
        }
        return com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer.present(owner.primary().model().spec());
    }

    /** What the model is for. */
    com.eonmux.neuralfree.gguf.arch.SupportClass supportClass() {
        owner.requireLoaded();
        if (owner.dual() != null) {
            return com.eonmux.neuralfree.gguf.arch.SupportClass.EMBED;
        }
        return owner.primary().model().spec().supportClass();
    }

    /** The classifier output of one input under {@code lora}. */
    ClassifierOutput classifyOutput(LoraSelection lora, int[] tokens) throws IOException {
        return classifyOutput(lora, tokens, true);
    }

    /** {@link #classifyOutput} with the final probability activation applied or skipped. */
    ClassifierOutput classifyOutput(LoraSelection lora, int[] tokens, boolean activation) throws IOException {
        return classifyOutputsAll(lora, new int[][]{tokens}, activation)[0];
    }

    /** The classifier outputs of several inputs over one replica. */
    ClassifierOutput[] classifyOutputsAll(LoraSelection lora, int[][] inputs) throws IOException {
        return classifyOutputsAll(lora, inputs, true);
    }

    /** {@link #classifyOutputsAll} with the final probability activation applied or skipped per input. */
    ClassifierOutput[] classifyOutputsAll(LoraSelection lora, int[][] inputs, boolean activation)
            throws IOException {
        Object[] out = poolAcross(inputs, PoolingOutput.CLASSIFIER_OUTPUT, lora, activation);
        ClassifierOutput[] outputs = new ClassifierOutput[out.length];
        for (int i = 0; i < out.length; i++) {
            outputs[i] = (ClassifierOutput) out[i];
        }
        return outputs;
    }

    /** The classifier scores of one input under {@code lora}. */
    float[] classifyScores(LoraSelection lora, int[] tokens) throws IOException {
        return (float[]) owner.poolingReplica().pooling(tokens, PoolingOutput.SCORES, lora);
    }

    /**
     * The decision scores of one ordered prompt under {@code lora}: output row {@code i} holds the score of the
     * {@code i}-th option span of {@code order} (the vllm-decision2 and clef heads' forward).
     */
    float[] decisionScores(LoraSelection lora, int[] tokens, int[] order) throws IOException {
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().classify(tokens, order));
    }

    /** The classifier scores of several inputs, spread over the open replicas. */
    float[][] classifyScoresAll(LoraSelection lora, int[][] inputs) throws IOException {
        return cast(poolAcross(inputs, PoolingOutput.SCORES, lora, true));
    }

    /**
     * The classifier output over a media prompt under {@code lora}: the sequence classifier's scores over the pooled
     * row of the prompt's text and media rows.
     */
    ClassifierOutput classifyOutputMedia(LoraSelection lora, MediaPrompt prompt, boolean activation)
            throws IOException {
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().classifyOutputMedia(prompt, activation));
    }

    /** The next-token logits of one prompt under {@code lora}. */
    float[] nextLogits(LoraSelection lora, int[] tokens) throws IOException {
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().nextLogits(tokens));
    }

    /** The next-token logits after the last row of a media prompt under {@code lora}. */
    float[] nextLogits(LoraSelection lora, MediaPrompt prompt) throws IOException {
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().nextLogits(prompt));
    }

    /** The decision scores of one media prompt under {@code lora}, read by the model's decision head. */
    float[] decisionScores(LoraSelection lora, MediaPrompt prompt, int[] order) throws IOException {
        EngineReplica replica = owner.poolingReplica();
        return replica.onPump(lora, () -> replica.model().classifyMedia(prompt, order));
    }

    /**
     * The answers of a multi-input pooling call in input order, its inputs spread over the open replicas (vLLM's
     * data-parallel engines each take a share of an embedding batch): the inputs split into contiguous parts of about
     * equal token counts, one part per replica of {@link #poolingReplicas}, each part queued on its replica as one
     * {@link EngineReplica#poolingAll} call. The calling thread runs the first part and virtual threads the others;
     * a failed part fails the call once every part has finished.
     */
    private Object[] poolAcross(int[][] inputs, PoolingOutput output, LoraSelection lora, boolean activation)
            throws IOException {
        List<EngineReplica> targets = poolingReplicas(inputs.length);
        if (targets.size() == 1) {
            return targets.get(0).poolingAll(inputs, output, lora, activation);
        }
        int[] bounds = PoolingSplit.bounds(inputs, targets.size());
        Object[] out = new Object[inputs.length];
        List<CompletableFuture<Void>> parts = new ArrayList<>(targets.size() - 1);
        for (int p = 1; p < targets.size(); p++) {
            EngineReplica replica = targets.get(p);
            int from = bounds[p];
            int to = bounds[p + 1];
            CompletableFuture<Void> part = new CompletableFuture<>();
            Thread.ofVirtual().name("pooling-part-" + p).start(() -> {
                try {
                    poolPart(replica, inputs, from, to, output, lora, activation, out);
                    part.complete(null);
                } catch (Throwable t) {
                    part.completeExceptionally(t);
                }
            });
            parts.add(part);
        }
        Throwable failure = null;
        try {
            poolPart(targets.get(0), inputs, bounds[0], bounds[1], output, lora, activation, out);
        } catch (IOException | RuntimeException e) {
            failure = e;
        }
        for (CompletableFuture<Void> part : parts) {
            failure = PoolingSplit.await(part, failure);
        }
        PoolingSplit.rethrow(failure);
        return out;
    }

    /** Runs inputs {@code from .. to} on {@code replica}, writing their answers into {@code out} at their indices. */
    private static void poolPart(EngineReplica replica, int[][] inputs, int from, int to, PoolingOutput output,
                                 LoraSelection lora, boolean activation, Object[] out) throws IOException {
        Object[] answers = replica.poolingAll(Arrays.copyOfRange(inputs, from, to), output, lora, activation);
        System.arraycopy(answers, 0, out, from, answers.length);
    }

    /**
     * The replicas one multi-input pooling call spreads over, at most {@code limit}: the one the service's
     * {@link EngineService#poolingReplica} picks first, then the scheduler's other replicas that hold their model
     * open, the least loaded first.
     */
    private List<EngineReplica> poolingReplicas(int limit) throws IOException {
        EngineReplica first = owner.poolingReplica();
        DpScheduler scheduler = owner.scheduler();
        if (limit <= 1 || scheduler == null) {
            return List.of(first);
        }
        List<EngineReplica> others = new ArrayList<>();
        for (EngineReplica r : scheduler.routed()) {
            if (r != first && r.open()) {
                others.add(r);
            }
        }
        others.sort(Comparator.comparingInt(EngineReplica::load));
        List<EngineReplica> out = new ArrayList<>();
        out.add(first);
        for (int i = 0; i < others.size() && out.size() < limit; i++) {
            out.add(others.get(i));
        }
        return out;
    }

    /** The batch's answers as {@code float[]} rows ({@link EngineReplica#poolingAll} keeps them as objects). */
    private static float[][] cast(Object[] rows) {
        float[][] out = new float[rows.length][];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float[]) rows[i];
        }
        return out;
    }

    /** The batch's answers as {@code float} scores ({@link EngineReplica#poolingAll} keeps them boxed). */
    private static float[] castScores(Object[] rows) {
        float[] out = new float[rows.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (Float) rows[i];
        }
        return out;
    }
}
