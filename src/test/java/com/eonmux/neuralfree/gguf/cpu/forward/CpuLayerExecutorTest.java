package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link CpuLayerExecutor} over a one-layer synthetic llama (unit norm weights) with recording mixer, FFN and extra
 * blocks: the executor's hook order, each residual kind, residual consumption by an extra, the discarded layer
 * output, the block output scale, the decode / prefill choice, and the wiring checks. Also the feature ids of
 * {@link BuiltInCpuExtras}.
 */
public class CpuLayerExecutorTest {

    private static final int ROWS = 3;
    private static final int UBATCH = 4;
    private static final float DELTA = 1e-6f;
    private static final String RECORDER = "recorder";

    // ------------------------------------------------------------------ recording blocks

    /** Mixer: {@code out = 2 * in + 1}. */
    private static final class FakeMixer implements CpuMixer {
        private final List<String> log;
        private int width;

        FakeMixer(List<String> log) {
            this.log = log;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            width = layer.residual().streamWidth() / layer.residual().hcMult();
        }

        @Override
        public void decode(int pos, float[] in, float[] out) {
            log.add("decode");
            apply(1, in, out);
        }

        @Override
        public void prefill(int pos0, int n, float[] in, float[] out) {
            log.add("prefill");
            apply(n, in, out);
        }

        private void apply(int n, float[] in, float[] out) {
            for (int i = 0; i < n * width; i++) {
                out[i] = 2f * in[i] + 1f;
            }
        }
    }

    /** FFN: {@code out = 0.5 * in - 1}. */
    private static final class FakeFfn implements CpuFfnBlock {
        private final List<String> log;
        private int width;

        FakeFfn(List<String> log) {
            this.log = log;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            width = layer.residual().streamWidth() / layer.residual().hcMult();
        }

        @Override
        public void forward(int n, float[] in, float[] out) {
            log.add("ffn");
            for (int i = 0; i < n * width; i++) {
                out[i] = 0.5f * in[i] - 1f;
            }
        }
    }

    /** Records every hook; optionally consumes the mixer residual without touching the stream. */
    private static final class RecordingExtra implements CpuLayerExtra {
        private final List<String> log;
        private final boolean consumeMixer;

        RecordingExtra(List<String> log, boolean consumeMixer) {
            this.log = log;
            this.consumeMixer = consumeMixer;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            log.add("bind");
        }

        @Override
        public void beforeBlock(int pos0, int n, float[] x) {
            log.add("beforeBlock");
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            log.add("afterMixer");
            return consumeMixer;
        }

        @Override
        public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
            log.add("afterFfn");
            return false;
        }

        @Override
        public void afterBlock(int pos0, int n, float[] x) {
            log.add("afterBlock");
        }

        @Override
        public void close() {
            log.add("close");
        }
    }

    private static CpuBlockRegistry registry(List<String> log, boolean consumeMixer) {
        CpuMixerFactory mixers = new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.ATTENTION;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                return new FakeMixer(log);
            }
        };
        CpuFfnFactory ffns = new CpuFfnFactory() {
            @Override
            public FfnKind kind() {
                return FfnKind.DENSE;
            }

            @Override
            public CpuFfnBlock create(LayerSpec layer) {
                return new FakeFfn(log);
            }
        };
        CpuLayerExtraFactory extras = new CpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return RECORDER;
            }

            @Override
            public CpuLayerExtra create(LayerSpec layer) {
                return new RecordingExtra(log, consumeMixer);
            }
        };
        return new CpuBlockRegistry(Collections.singletonList(mixers), Collections.singletonList(ffns),
                Collections.singletonList(extras));
    }

    // ------------------------------------------------------------------ model fixture

    /** A one-layer llama file, its weights and a memory-less context whose layer uses a chosen residual. */
    private static final class Fixture implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final ModelSpec spec;
        final CpuForwardContext ctx;

        Fixture(ResidualSpec residual, String... extras) throws IOException {
            SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nLayer(1).build();
            this.path = SyntheticModels.llama(config).writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            TensorCatalog tc = TensorCatalog.of(parser);
            ModelHparams hp = HparamsLoader.load(GgufKv.of(parser));
            NormResolver norms = new NormResolver(hp, tc);
            RopeResolver rope = new RopeResolver(hp, tc, hp.nCtxTrain());
            AttentionResolver attention = new AttentionResolver(hp, tc, rope, norms);
            FfnResolver.Result ffn = new FfnResolver(hp, tc).resolve(0);
            ResidualSpec res = residual != null ? residual : new ResidualResolver(hp, tc).resolve(0);
            LayerSpec.Builder layer = LayerSpec.builder(0, attention.attentionKind(0), ffn.kind())
                    .norm(norms.resolve(0))
                    .residual(res)
                    .attention(attention.resolve(0))
                    .rope(rope.forLayer(0))
                    .ffn(ffn.ffn())
                    .moe(ffn.moe());
            for (String id : extras) {
                layer.extra(id);
            }
            this.spec = ModelSpec.builder(hp, tc)
                    .layer(layer.build())
                    .schedule(LayerSchedule.linear(1))
                    .embedding(new EmbeddingResolver(hp, tc).resolve())
                    .head(new HeadResolver(hp, tc).resolve())
                    .memoryKind(MemoryKind.NONE)
                    .build();
            this.weights = CpuWeights.open(parser);
            this.ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH)).build();
        }

        int nEmbd() {
            return spec.hparams().nEmbd();
        }

        NormSpec norm() {
            return spec.layer(0).norm();
        }

        @Override
        public void close() throws IOException {
            try {
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

    private static Fixture llama(ResidualSpec residual, String... extras) throws IOException {
        return new Fixture(residual, extras);
    }

    /** The default llama residual of a one-layer fixture with {@code kind} and the given settings. */
    private static ResidualSpec residual(Fixture f, ResidualKind kind, float scale, boolean mixerResidual,
                                         boolean discard, float outScale) {
        ResidualSpec.Builder b = ResidualSpec.builder(kind, f.nEmbd())
                .mixerResidual(mixerResidual)
                .discardLayerOutput(discard)
                .blockOutputScale(outScale);
        if (kind == ResidualKind.SCALED || kind == ResidualKind.NORMED_INPUT_SCALED) {
            b.residualScale(scale);
        }
        return b.build();
    }

    // ------------------------------------------------------------------ expected math

    private static float[] stream(int n, int width, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[n * width];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 4f - 2f;
        }
        return x;
    }

    private static float[] rms(NormSpec norm, float[] x, int n, int width) {
        assertEquals(NormKind.RMS, norm.kind());
        float[] y = new float[n * width];
        for (int r = 0; r < n; r++) {
            NormMath.rowNorm(NormKind.RMS, norm.eps(), x, r * width, y, r * width, width, null, 0, null, 0);
        }
        return y;
    }

    private static float[] mixerOf(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = 2f * in[i] + 1f;
        }
        return out;
    }

    private static float[] ffnOf(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = 0.5f * in[i] - 1f;
        }
        return out;
    }

    private static float[] plus(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    private static float[] plusScaled(float[] a, float s, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + s * b[i];
        }
        return out;
    }

    /** Runs the fixture's layer over {@code x} (copied into the scratch stream) and returns the stream rows. */
    private static float[] run(Fixture f, CpuBlockRegistry registry, float[] x, int n) {
        float[] stream = f.ctx.scratch().slot(CpuScratch.X);
        System.arraycopy(x, 0, stream, 0, x.length);
        try (CpuLayerExecutor e = new CpuLayerExecutor(registry, f.ctx, f.spec.layer(0))) {
            e.forward(7, n, stream);
        }
        return Arrays.copyOf(stream, x.length);
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void standardResidualFollowsThePreNormOrder() throws IOException {
        List<String> log = new ArrayList<>();
        try (Fixture f = llama(null, RECORDER)) {
            int w = f.nEmbd();
            float[] x0 = stream(ROWS, w, 1);
            float[] x1 = plus(x0, mixerOf(rms(f.norm(), x0, ROWS, w)));
            float[] x2 = plus(x1, ffnOf(rms(f.norm(), x1, ROWS, w)));

            float[] got = run(f, registry(log, false), x0, ROWS);

            assertArrayEquals(x2, got, DELTA);
            assertEquals(Arrays.asList("bind", "beforeBlock", "prefill", "afterMixer", "ffn", "afterFfn",
                    "afterBlock", "close"), log);
        }
    }

    @Test
    public void scaledResidualScalesBothBranches() throws IOException {
        float s = 0.25f;
        try (Fixture base = llama(null)) {
            ResidualSpec res = residual(base, ResidualKind.SCALED, s, true, false, 1f);
            try (Fixture f = llama(res)) {
                int w = f.nEmbd();
                float[] x0 = stream(ROWS, w, 2);
                float[] x1 = plusScaled(x0, s, mixerOf(rms(f.norm(), x0, ROWS, w)));
                float[] x2 = plusScaled(x1, s, ffnOf(rms(f.norm(), x1, ROWS, w)));

                assertArrayEquals(x2, run(f, registry(new ArrayList<>(), false), x0, ROWS), DELTA);
            }
        }
    }

    @Test
    public void normedInputScaledResidualAddsTheScaledNormedInput() throws IOException {
        float s = 0.5f;
        try (Fixture base = llama(null)) {
            ResidualSpec res = residual(base, ResidualKind.NORMED_INPUT_SCALED, s, true, false, 1f);
            try (Fixture f = llama(res)) {
                int w = f.nEmbd();
                float[] x0 = stream(ROWS, w, 3);
                float[] n1 = rms(f.norm(), x0, ROWS, w);
                float[] x1 = plusScaled(mixerOf(n1), s, n1);
                float[] n2 = rms(f.norm(), x1, ROWS, w);
                float[] x2 = plusScaled(ffnOf(n2), s, n2);

                assertArrayEquals(x2, run(f, registry(new ArrayList<>(), false), x0, ROWS), DELTA);
            }
        }
    }

    @Test
    public void parallelResidualFeedsBothBranchesFromTheLayerInput() throws IOException {
        try (Fixture base = llama(null)) {
            ResidualSpec res = residual(base, ResidualKind.PARALLEL, 1f, true, false, 1f);
            try (Fixture f = llama(res)) {
                int w = f.nEmbd();
                float[] x0 = stream(ROWS, w, 4);
                float[] normed = rms(f.norm(), x0, ROWS, w);
                float[] expected = plus(x0, plus(ffnOf(normed), mixerOf(normed)));

                assertArrayEquals(expected, run(f, registry(new ArrayList<>(), false), x0, ROWS), DELTA);
            }
        }
    }

    @Test
    public void extraThatConsumesTheMixerResidualSkipsTheAdd() throws IOException {
        List<String> log = new ArrayList<>();
        try (Fixture f = llama(null, RECORDER)) {
            int w = f.nEmbd();
            float[] x0 = stream(ROWS, w, 5);
            float[] expected = plus(x0, ffnOf(rms(f.norm(), x0, ROWS, w)));

            assertArrayEquals(expected, run(f, registry(log, true), x0, ROWS), DELTA);
            assertTrue(log.contains("afterMixer"));
        }
    }

    @Test
    public void mixerWithoutResidualLeavesTheStreamForTheFfn() throws IOException {
        try (Fixture base = llama(null)) {
            ResidualSpec res = residual(base, ResidualKind.STANDARD, 1f, false, false, 1f);
            try (Fixture f = llama(res)) {
                int w = f.nEmbd();
                float[] x0 = stream(ROWS, w, 6);
                float[] expected = plus(x0, ffnOf(rms(f.norm(), x0, ROWS, w)));

                assertArrayEquals(expected, run(f, registry(new ArrayList<>(), false), x0, ROWS), DELTA);
            }
        }
    }

    @Test
    public void discardedLayerOutputRestoresTheInputAfterRunningTheBlocks() throws IOException {
        List<String> log = new ArrayList<>();
        try (Fixture base = llama(null)) {
            ResidualSpec res = residual(base, ResidualKind.STANDARD, 1f, true, true, 1f);
            try (Fixture f = llama(res)) {
                float[] x0 = stream(ROWS, f.nEmbd(), 7);

                assertArrayEquals(x0, run(f, registry(log, false), x0, ROWS), 0f);
                assertEquals(Arrays.asList("prefill", "ffn"), log);
            }
        }
    }

    @Test
    public void blockOutputScaleMultipliesTheLayerOutput() throws IOException {
        try (Fixture base = llama(null)) {
            ResidualSpec res = residual(base, ResidualKind.STANDARD, 1f, true, false, 0.5f);
            try (Fixture f = llama(res)) {
                int w = f.nEmbd();
                float[] x0 = stream(ROWS, w, 8);
                float[] x1 = plus(x0, mixerOf(rms(f.norm(), x0, ROWS, w)));
                float[] x2 = plus(x1, ffnOf(rms(f.norm(), x1, ROWS, w)));
                for (int i = 0; i < x2.length; i++) {
                    x2[i] *= 0.5f;
                }

                assertArrayEquals(x2, run(f, registry(new ArrayList<>(), false), x0, ROWS), DELTA);
            }
        }
    }

    @Test
    public void singleTokenWithoutMemoryRunsThePrefillPath() throws IOException {
        List<String> log = new ArrayList<>();
        try (Fixture f = llama(null)) {
            run(f, registry(log, false), stream(1, f.nEmbd(), 9), 1);
            assertEquals(Arrays.asList("prefill", "ffn"), log);
        }
    }

    @Test
    public void hyperConnectionResidualWithoutItsExtraIsRejected() throws IOException {
        try (Fixture base = llama(null)) {
            ResidualSpec res = ResidualSpec.builder(ResidualKind.HYPER_CONNECTIONS, base.nEmbd())
                    .hyperConnections(2, true)
                    .build();
            try (Fixture f = llama(res)) {
                try {
                    new CpuLayerExecutor(registry(new ArrayList<>(), false), f.ctx, f.spec.layer(0)).close();
                    fail("a hyper-connection layer without the hc extra must be rejected");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage().contains(LayerSpec.Extras.HC));
                }
            }
        }
    }

    @Test
    public void stepLargerThanTheUbatchIsRejected() throws IOException {
        try (Fixture f = llama(null)) {
            float[] x = new float[(UBATCH + 1) * f.nEmbd()];
            try (CpuLayerExecutor e = new CpuLayerExecutor(registry(new ArrayList<>(), false), f.ctx,
                    f.spec.layer(0))) {
                try {
                    e.forward(0, UBATCH + 1, x);
                    fail("a step above the ubatch must be rejected");
                } catch (IllegalArgumentException expected) {
                    assertTrue(expected.getMessage().contains("outside [1, " + UBATCH + "]"));
                }
            }
        }
    }

    @Test
    public void builtInExtrasServeTheirFeatureIds() {
        Set<String> ids = new HashSet<>();
        for (CpuLayerExtraFactory factory : BuiltInCpuExtras.cpuExtras()) {
            assertTrue("duplicate id " + factory.featureId(), ids.add(factory.featureId()));
        }
        assertEquals(new HashSet<>(Arrays.asList(LayerSpec.Extras.EMBD_SKIP, LayerSpec.Extras.RESID_ALPHA_BETA,
                LayerSpec.Extras.LOOP_NORM)), ids);
    }
}
