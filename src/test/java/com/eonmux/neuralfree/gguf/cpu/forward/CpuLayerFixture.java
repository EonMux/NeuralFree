package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.AttentionResolver;
import com.eonmux.neuralfree.gguf.spec.resolve.EmbeddingResolver;
import com.eonmux.neuralfree.gguf.spec.resolve.FfnResolver;
import com.eonmux.neuralfree.gguf.spec.resolve.HeadResolver;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;
import com.eonmux.neuralfree.gguf.spec.resolve.ResidualResolver;
import com.eonmux.neuralfree.gguf.spec.resolve.RopeResolver;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A one-layer llama for the layer-executor tests: a real KV memory, the registry's real blocks and an FFN wrapped in a
 * counter that records the rows of every run.
 */
final class CpuLayerFixture {

    static final int UBATCH = 4;
    static final int N_CTX = 16;

    final Path path;
    final GGUFFileParser parser;
    final CpuWeights weights;
    final HeapModelMemory memory;
    final CpuForwardContext ctx;
    final CpuLayerExecutor executor;
    final int width;
    final AtomicInteger ffnRuns = new AtomicInteger();
    /** The row count of the FFN's latest run. */
    final AtomicInteger ffnRows = new AtomicInteger();

    CpuLayerFixture() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nLayer(1).build();
        this.path = SyntheticModels.llama(config).writeTempFile();
        this.parser = new GGUFFileParser(path.toString());
        TensorCatalog tc = TensorCatalog.of(parser);
        ModelHparams hp = HparamsLoader.load(GgufKv.of(parser));
        NormResolver norms = new NormResolver(hp, tc);
        RopeResolver rope = new RopeResolver(hp, tc, hp.nCtxTrain());
        AttentionResolver attention = new AttentionResolver(hp, tc, rope, norms);
        FfnResolver.Result ffn = new FfnResolver(hp, tc).resolve(0);
        ResidualSpec res = new ResidualResolver(hp, tc).resolve(0);
        LayerSpec layer = LayerSpec.builder(0, attention.attentionKind(0), ffn.kind())
                .norm(norms.resolve(0))
                .residual(res)
                .attention(attention.resolve(0))
                .rope(rope.forLayer(0))
                .ffn(ffn.ffn())
                .moe(ffn.moe())
                .build();
        ModelSpec spec = ModelSpec.builder(hp, tc)
                .layer(layer)
                .schedule(LayerSchedule.linear(1))
                .embedding(new EmbeddingResolver(hp, tc).resolve())
                .head(new HeadResolver(hp, tc).resolve())
                .memoryKind(MemoryKind.KV)
                .build();
        this.weights = CpuWeights.open(parser);
        MemoryPlan plan = MemoryPlanner.planForContext(spec,
                MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).typeK(KvDType.F32).typeV(KvDType.F32).build(),
                N_CTX);
        this.memory = HeapModelMemory.allocate(plan, spec, null);
        this.ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH))
                .memory(memory).build();
        this.executor = new CpuLayerExecutor(registry(ffn.kind()), ctx, layer);
        this.width = spec.hparams().nEmbd();
    }

    /** The registry's real blocks, its FFN factory of the layer wrapped in a counter. */
    private CpuBlockRegistry registry(FfnKind kind) {
        CpuFfnFactory counted = new CpuFfnFactory() {
            @Override public FfnKind kind() { return kind; }

            @Override public CpuFfnBlock create(LayerSpec layer) {
                CpuFfnBlock real = CpuFactoryCatalog.ffns().stream().filter(f -> f.kind() == kind).findFirst()
                        .orElseThrow(() -> new IllegalStateException("no FFN factory for " + kind))
                        .create(layer);
                return new CpuFfnBlock() {
                    @Override public void bind(CpuForwardContext c, LayerSpec l) { real.bind(c, l); }

                    @Override public void forward(int n, float[] in, float[] out) {
                        ffnRuns.incrementAndGet();
                        ffnRows.set(n);
                        real.forward(n, in, out);
                    }

                    @Override public boolean rowLocal() { return real.rowLocal(); }

                    @Override public void close() { real.close(); }
                };
            }
        };
        return new CpuBlockRegistry(CpuFactoryCatalog.mixers(), Collections.singletonList(counted),
                CpuFactoryCatalog.extras(null));
    }

    /** A whole forward of {@code n} rows at {@code pos0} over the stream {@code x}, updated in place. */
    void forward(int pos0, int n, float[] x) {
        begin(pos0, n);
        executor.forward(pos0, n, x);
    }

    /** The attention part only of {@code n} rows at {@code pos0}. */
    void forwardKV(int pos0, int n, float[] x) {
        begin(pos0, n);
        executor.forwardKV(pos0, n, x);
    }

    /** {@code n} rows at {@code pos0} whose output is read at the step-local {@code rows} only. */
    void forwardRows(int pos0, int n, float[] x, int[] rows) {
        begin(pos0, n);
        executor.forwardRows(pos0, n, x, rows);
    }

    /** Opens the memory step and fills the positions the mixers read ({@code CpuMixerRig} orders it). */
    private void begin(int pos0, int n) {
        memory.beginStep(pos0, n);
        int[] pos = ctx.scratch().ints(CpuScratch.POS);
        for (int t = 0; t < n; t++) {
            pos[t] = pos0 + t;
        }
    }

    void close() throws IOException {
        try {
            executor.close();
            memory.close();
            weights.close();
        } finally {
            try {
                parser.close();
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }
}
