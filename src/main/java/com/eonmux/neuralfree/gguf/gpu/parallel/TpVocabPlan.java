package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * The vocabulary half of a tensor-parallel split (vLLM's {@code VocabParallelEmbedding} and {@code ParallelLMHead}):
 * the token embedding table and the LM head are sharded over the ranks on their vocabulary dimension. Rank {@code r}
 * holds rows {@code [r * rows / ranks, (r + 1) * rows / ranks)} of each table, so the model device never holds the
 * whole vocabulary. The embedding lookup of a step runs on every rank (each rank looks up only the tokens that fall
 * in its slice and zeroes the others) and an all-reduce sums the row vectors into the model device; the LM head runs
 * on every rank over the same hidden rows and the rank logits are gathered into the model device's full logits row.
 *
 * <p>The plan requires both tables to divide into whole rows per rank and refuses a PRISM input transform on either
 * weight (the transform is built for the full vocabulary). An LM head tied to the token table shards the same tensor
 * once; an output bias is sharded with the head. The plan is immutable.</p>
 */
public final class TpVocabPlan {

    private final int ranks;
    private final String tokenTensor;
    private final String headTensor;
    private final String headBias;
    private final long tokenRows;
    private final long headRows;
    private final long tokenRowBytes;
    private final long headRowBytes;

    private TpVocabPlan(int ranks, String tokenTensor, String headTensor, String headBias, long tokenRows,
                        long headRows, long tokenRowBytes, long headRowBytes) {
        this.ranks = ranks;
        this.tokenTensor = tokenTensor;
        this.headTensor = headTensor;
        this.headBias = headBias;
        this.tokenRows = tokenRows;
        this.headRows = headRows;
        this.tokenRowBytes = tokenRowBytes;
        this.headRowBytes = headRowBytes;
    }

    /** Number of ranks the vocabulary divides over. */
    public int ranks() {
        return ranks;
    }

    /** Full name of the token embedding table. */
    public String tokenTensor() {
        return tokenTensor;
    }

    /** Full name of the LM-head weight ({@code output.weight}, or the token table when tied). */
    public String headTensor() {
        return headTensor;
    }

    /** Full name of the LM-head bias, or null when the head has none. */
    public String headBias() {
        return headBias;
    }

    /** Rows of the token embedding table. */
    public long tokenRows() {
        return tokenRows;
    }

    /** Rows of the LM-head weight. */
    public long headRows() {
        return headRows;
    }

    /** First token-table row rank {@code rank} holds. */
    public long tokenStart(int rank) {
        return rank * tokenRows / ranks;
    }

    /** Token-table rows rank {@code rank} holds. */
    public long tokenCount(int rank) {
        return tokenStart(rank + 1) - tokenStart(rank);
    }

    /** First LM-head row rank {@code rank} holds. */
    public long headStart(int rank) {
        return rank * headRows / ranks;
    }

    /** LM-head rows rank {@code rank} holds. */
    public long headCount(int rank) {
        return headStart(rank + 1) - headStart(rank);
    }

    /** Device bytes of rank {@code rank}'s token-table shard. */
    public long tokenBytes(int rank) {
        return tokenCount(rank) * tokenRowBytes;
    }

    /** Device bytes of rank {@code rank}'s LM-head shard, or 0 when the head is the token table (tied). */
    public long headBytes(int rank) {
        return tied() ? 0 : headCount(rank) * headRowBytes;
    }

    /** Device bytes of rank {@code rank}'s head-bias shard, or 0 without a bias. */
    public long headBiasBytes(int rank) {
        return headBias == null ? 0 : headCount(rank) * Float.BYTES;
    }

    /** True when the LM head reads the token table itself. */
    public boolean tied() {
        return headTensor.equals(tokenTensor);
    }

    /** True when a tensor uploads as per-rank vocabulary shards. */
    public boolean splits(String tensorName) {
        return tokenTensor.equals(tensorName) || headTensor.equals(tensorName)
                || (headBias != null && headBias.equals(tensorName));
    }

    /**
     * Plans the vocabulary split of {@code spec} over {@code ranks} ranks.
     *
     * @throws UnsupportedOperationException when the model has no token table or no LM head, a table does not divide
     *                                       into the ranks, or a PRISM transform is built for the full vocabulary
     * @throws IllegalArgumentException      for fewer than two ranks or null arguments
     */
    public static TpVocabPlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader) {
        if (ranks < 2) {
            throw new IllegalArgumentException("a tensor split needs at least two ranks, got " + ranks);
        }
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(uploader, "uploader");
        String arch = spec.arch().ggufName();
        EmbeddingSpec embedding = spec.embedding();
        HeadSpec head = spec.head();
        if (embedding == null || embedding.tokenEmbd() == null) {
            throw new UnsupportedOperationException(arch + ": the model has no token table to split over the ranks");
        }
        if (head == null || head.output() == null) {
            throw new UnsupportedOperationException(arch + ": the model has no LM head to split over the ranks");
        }
        GpuWeightUploader.TensorPlan token = plan(uploader, embedding.tokenEmbd(), arch, "token table");
        GpuWeightUploader.TensorPlan output = plan(uploader, head.output(), arch, "LM head");
        if (token.ne().length < 2 || output.ne().length < 2) {
            throw new UnsupportedOperationException(arch + ": the token table and the LM head must be matrices to"
                    + " split their vocabulary rows");
        }
        long tokenRows = token.rows();
        long headRows = output.rows();
        requireDivides(arch, token, tokenRows, ranks, "token table");
        requireDivides(arch, output, headRows, ranks, "LM head");
        transform(uploader, token, arch);
        transform(uploader, output, arch);
        String bias = null;
        if (head.outputBias() != null) {
            GpuWeightUploader.TensorPlan p = plan(uploader, head.outputBias(), arch, "LM-head bias");
            if (p.needsDecode() || p.deviceType() != com.eonmux.neuralfree.gguf.GGMLType.F32) {
                throw new UnsupportedOperationException(arch + ": LM-head bias " + p.name() + " is "
                        + p.deviceType() + "; a split head bias needs F32");
            }
            if (p.info().numElements() != headRows) {
                throw new UnsupportedOperationException(arch + ": LM-head bias " + p.name() + " holds "
                        + p.info().numElements() + " values, the head has " + headRows + " rows");
            }
            bias = p.name();
        }
        return new TpVocabPlan(ranks, token.name(), output.name(), bias, tokenRows, headRows,
                token.deviceRowBytes(), output.deviceRowBytes());
    }

    private static GpuWeightUploader.TensorPlan plan(GpuWeightUploader uploader, TensorRef ref, String arch,
                                                     String what) {
        GpuWeightUploader.TensorPlan p = uploader.plan(ref.name());
        if (p == null) {
            throw new UnsupportedOperationException(arch + ": the model has no tensor " + ref.name() + " (its "
                    + what + ")");
        }
        return p;
    }

    private static void requireDivides(String arch, GpuWeightUploader.TensorPlan p, long rows, int ranks,
                                       String what) {
        if (rows % ranks != 0) {
            throw new UnsupportedOperationException(arch + ": " + what + " " + p.name() + " has " + rows
                    + " rows, which does not divide into " + ranks + " rank(s)");
        }
    }

    private static void transform(GpuWeightUploader uploader, GpuWeightUploader.TensorPlan p, String arch) {
        InputTransformLoader.Result transforms = uploader.transforms();
        if (transforms == null || transforms.isEmpty()) {
            return;
        }
        if (transforms.forward(p.name()) != null && !transforms.forward(p.name()).isIdentity()) {
            throw new UnsupportedOperationException(arch + ": the PRISM input transform of " + p.name()
                    + " is built for the full vocabulary");
        }
        HadamardSpec hadamard = transforms.hadamard(p.name());
        if (hadamard != null) {
            throw new UnsupportedOperationException(arch + ": the Hadamard rotation of " + p.name()
                    + " is built for the full vocabulary");
        }
    }

    @Override
    public String toString() {
        return "TpVocabPlan{" + tokenRows + " token rows / " + headRows + " head rows over " + ranks + " ranks"
                + (tied() ? ", tied" : "") + (headBias != null ? ", bias" : "") + '}';
    }
}
