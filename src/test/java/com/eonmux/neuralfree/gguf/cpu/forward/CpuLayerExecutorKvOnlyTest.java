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
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
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
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuLayerExecutor#forwardKV} over a real one-layer llama with a KV memory and the registry's real blocks:
 * the attention part only — the FFN never runs — and the cache rows it writes are the ones a whole {@link
 * CpuLayerExecutor#forward} writes, so the step that follows attends over the same rows and produces the same
 * stream.
 */
public class CpuLayerExecutorKvOnlyTest {

    private static final int ROWS = 3;
    private static final int UBATCH = 4;
    private static final int N_CTX = 16;

    /**
     * The single-token step after the prompt: whether the prompt ran whole ({@code forward}) or as its cache writes
     * only ({@code forwardKV}), the next token reads the same attention and the streams agree.
     */
    @Test
    public void forwardKVWritesTheCachesAForwardWrites() throws IOException {
        Fixture f = new Fixture();
        try {
            float[] prompt = stream(ROWS, f.width);
            float[] token = stream(1, f.width);

            // the whole layer over the prompt, then the next token over the prompt's output
            float[] x = prompt.clone();
            f.forward(0, ROWS, x);
            float[] afterPrompt = x.clone();
            System.arraycopy(token, 0, x, 0, f.width);
            f.forward(ROWS, 1, x);
            float[] wholeNext = Arrays.copyOf(x, f.width);

            // the same prompt as its cache writes only, then the same next token
            f.memory.reset();
            f.ffnRuns.set(0);
            f.forwardKV(0, ROWS, prompt.clone());
            assertEquals("forwardKV runs no FFN", 0, f.ffnRuns.get());
            float[] y = new float[prompt.length];
            System.arraycopy(afterPrompt, (ROWS - 1) * f.width, y, 0, f.width);
            System.arraycopy(token, 0, y, 0, f.width);
            f.forward(ROWS, 1, y);
            float[] kvNext = Arrays.copyOf(y, f.width);
            assertEquals("the following whole step runs its FFN", 1, f.ffnRuns.get());

            assertArrayEquals(wholeNext, kvNext, 1e-9f);
            boolean nonZero = false;
            for (float v : wholeNext) {
                nonZero |= v != 0f;
            }
            assertTrue("the streams are not all zero", nonZero);
        } finally {
            f.close();
        }
    }

    /**
     * The prompt's cache rows after its attention part alone are bit-for-bit the rows a whole {@code forward}
     * writes: the KV-only step runs the same K/V projections, norms, RoPE and store, and nothing else.
     */
    @Test
    public void forwardKVWritesTheSameCacheBits() throws IOException {
        Fixture f = new Fixture();
        try {
            float[] prompt = stream(ROWS, f.width);
            f.forward(0, ROWS, prompt.clone());
            int[] whole = cacheBits(f);

            f.memory.reset();
            f.forwardKV(0, ROWS, prompt.clone());
            int[] kvOnly = cacheBits(f);

            assertArrayEquals(whole, kvOnly);
            boolean nonZero = false;
            for (int bits : whole) {
                nonZero |= bits != 0;
            }
            assertTrue("the cache rows are not all zero", nonZero);
        } finally {
            f.close();
        }
    }

    /** The raw bits of every stored K and V head over the first {@link #ROWS} positions. */
    private static int[] cacheBits(Fixture f) {
        KvAccessor cache = f.memory.kv(0);
        int heads = cache.nHeadKv();
        int headK = cache.headDimK();
        int headV = cache.headDimV();
        int[] out = new int[ROWS * heads * (headK + headV)];
        float[] k = new float[headK];
        float[] v = new float[headV];
        int at = 0;
        for (int row = 0; row < ROWS; row++) {
            for (int h = 0; h < heads; h++) {
                cache.readK(row, h, k, 0);
                for (float x : k) {
                    out[at++] = Float.floatToRawIntBits(x);
                }
                cache.readV(row, h, v, 0);
                for (float x : v) {
                    out[at++] = Float.floatToRawIntBits(x);
                }
            }
        }
        return out;
    }

    private static float[] stream(int n, int width) {
        Random rnd = new Random(42L * n);
        float[] x = new float[n * width];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    /** A one-layer llama with a real KV memory, the registry's real blocks and a counting FFN. */
    private static final class Fixture {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final HeapModelMemory memory;
        final CpuForwardContext ctx;
        final CpuLayerExecutor executor;
        final int width;
        final AtomicInteger ffnRuns = new AtomicInteger();

        Fixture() throws IOException {
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
                            real.forward(n, in, out);
                        }

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
}
