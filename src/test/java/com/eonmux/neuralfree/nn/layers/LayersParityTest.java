package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.CheckpointFunction;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.ConvGradFunctions;
import com.eonmux.neuralfree.autograd.GradFunctions;
import com.eonmux.neuralfree.autograd.IndexGradFunctions;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The nn layers on the train dispatchers. {@link CpuPath} checks the layers' CPU branches on the host reference
 * backend against independent definitions (the transposed convolution against its scatter definition) and the
 * seeded random streams. {@link GpuParity} runs each layer on every available GPU API against
 * the same layer on the host backend, under the tolerance classes of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}: convolutions in the reduction class against a
 * double reference, max pooling, embedding and the dropout mask bit-identical, average and adaptive average pooling
 * (PyTorch windows) in the division class and the dropout output in the single-rounding class. {@link Recording} checks
 * the autograd recording of the layers on the host backend, and {@link GpuParity} repeats its gradient cases on every
 * GPU API: each forward records its facade's gradient function, a backward from the output sum reaches the parameters
 * and leaf inputs with exact (or reduction-class) gradients, and the graph is back at its size before the step.
 */
@RunWith(Enclosed.class)
public class LayersParityTest {

    private static final long WEIGHT_SEED = 1234L;

    // ------------------------------------------------------------------ CPU branches

    public static class CpuPath {
        private final Fixture fx = new Fixture();
        private final ComputeBackend host = TrainParity.hostBackend();

        @After
        public void tearDown() {
            fx.close();
        }

        @Test
        public void conv2dMatchesTheDirectDefinition() {
            ConvCase c = ConvCase.conv2d();
            Conv2d layer = fx.module(c.conv2dLayer());
            layer.initialize(host, new Random(WEIGHT_SEED));
            float[] bias = c.bias();
            fx.write(layer.bias().tensor(), bias);
            float[] x = c.input();
            Tensor out = fx.keep(layer.forward(fx.upload(host, x, c.b, c.c, c.h, c.w)));
            Ref ref = c.directRef(x, layer.weight().tensor().toFloatArray(), bias);
            assertEquals(Shape.of(c.b, c.oc, ref.oh, ref.ow), out.shape());
            ref.check("CPU conv2d", out.toFloatArray());
        }

        @Test
        public void conv1dMatchesTheDirectDefinition() {
            ConvCase c = ConvCase.conv1d();
            Conv1d layer = fx.module(c.conv1dLayer());
            layer.initialize(host, new Random(WEIGHT_SEED));
            float[] bias = c.bias();
            fx.write(layer.bias().tensor(), bias);
            float[] x = c.input();
            Tensor out = fx.keep(layer.forward(fx.upload(host, x, c.b, c.c, c.w)));
            Ref ref = c.directRef(x, layer.weight().tensor().toFloatArray(), bias);
            assertEquals(Shape.of(c.b, c.oc, ref.ow), out.shape());
            ref.check("CPU conv1d", out.toFloatArray());
        }

        @Test
        public void convTranspose2dMatchesTheScatterDefinition() {
            ConvCase c = ConvCase.convTranspose2d();
            ConvTranspose2d layer = fx.module(c.convTranspose2dLayer());
            layer.initialize(host, new Random(WEIGHT_SEED));
            float[] bias = c.bias();
            fx.write(layer.bias().tensor(), bias);
            float[] x = c.input();
            Tensor out = fx.keep(layer.forward(fx.upload(host, x, c.b, c.c, c.h, c.w)));
            Ref ref = c.scatterRef(x, layer.weight().tensor().toFloatArray(), bias);
            assertEquals(Shape.of(c.b, c.oc, ref.oh, ref.ow), out.shape());
            ref.check("CPU convTranspose2d", out.toFloatArray());
        }

        @Test
        public void convolutionWeightsAreTheSeededRandnStreamTimesStd() {
            Random seeds = new Random(WEIGHT_SEED);
            long seed = seeds.nextLong();
            Conv2d layer = fx.module(new Conv2d(4, 6, new int[]{3, 2}, new int[]{1, 1}, new int[]{0, 0},
                    new int[]{1, 1}, 2, true));
            layer.initialize(host, new Random(WEIGHT_SEED));
            Shape shape = Shape.of(6, 2, 3, 2);
            float std = (float) Math.sqrt(2.0 / (2 * 3 * 2));
            float[] randn = fx.keep(Tensor.randn(host, shape, seed)).toFloatArray();
            float[] expected = new float[randn.length];
            for (int i = 0; i < randn.length; i++) {
                expected[i] = randn[i] * std;
            }
            assertEquals(shape, layer.weight().tensor().shape());
            assertTrue(layer.weight().tensor().requiresGrad());
            TrainParity.assertBitIdentical("Conv2d weights", expected, layer.weight().tensor().toFloatArray());
            TrainParity.assertBitIdentical("Conv2d bias", new float[6], layer.bias().tensor().toFloatArray());
        }

        @Test
        public void dropoutDrawsOneKeepDecisionPerElementFromItsRandom() {
            Dropout layer = fx.module(new Dropout(0.3f));
            layer.setSeed(77L);
            float[] x = TrainParity.uniform(5, 40, -4f, 4f);
            Tensor out = fx.keep(layer.forward(fx.upload(host, x, 5, 8)));
            Random r = new Random(77L);
            float keep = 1.0f - 0.3f;
            float scale = 1.0f / keep;
            float[] mask = new float[x.length];
            float[] expected = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                boolean kept = r.nextFloat() < keep;
                mask[i] = kept ? scale : 0.0f;
                expected[i] = kept ? x[i] * scale : 0.0f;
            }
            assertEquals(Shape.of(5, 8), out.shape());
            TrainParity.assertBitIdentical("dropout mask", mask, layer.getLastMask().toFloatArray());
            TrainParity.assertBitIdentical("dropout output", expected, out.toFloatArray());
        }

        @Test
        public void dropoutInEvaluationModeReturnsItsInput() {
            Dropout layer = fx.module(new Dropout(0.5f));
            layer.eval();
            Tensor in = fx.upload(host, new float[]{1f, 2f, 3f}, 3);
            assertSame(in, layer.forward(in));
        }

        /** Flatten merges only its dimension range and keeps every other dimension, a batch of one included. */
        @Test
        public void flattenKeepsTheDimensionsOutsideItsRange() {
            float[] x = TrainParity.uniform(6, 2 * 3 * 4 * 5, -1f, 1f);
            Tensor batchOfOne = fx.upload(host, Arrays.copyOf(x, 60), 1, 3, 4, 5);
            Tensor batch = fx.upload(host, x, 2, 3, 4, 5);
            assertEquals(Shape.of(1, 60), fx.keep(fx.module(new Flatten()).forward(batchOfOne)).shape());
            assertEquals(Shape.of(2, 3, 20), fx.keep(fx.module(new Flatten(2)).forward(batch)).shape());
            assertEquals(Shape.of(2, 12, 5), fx.keep(fx.module(new Flatten(1, 2)).forward(batch)).shape());
            assertEquals(Shape.of(2, 3, 20), fx.keep(fx.module(new Flatten(-2, -1)).forward(batch)).shape());
            assertEquals(Shape.of(120), fx.keep(fx.module(new Flatten(0)).forward(batch)).shape());
            Tensor out = fx.keep(fx.module(new Flatten(1, 2)).forward(batch));
            TrainParity.assertBitIdentical("Flatten values", x, out.toFloatArray());
            assertThrows(IllegalArgumentException.class, () -> new Flatten(2, 1).forward(batch));
        }

        @Test
        public void embeddingNarrowsInt64AndRejectsOutOfRangeIndices() {
            Embedding layer = fx.module(new Embedding(5, 3));
            layer.initialize(host, new Random(WEIGHT_SEED));
            float[] table = layer.weight().tensor().toFloatArray();
            long[] wide = {(1L << 32) + 4, 0, 2};
            Tensor out = fx.keep(layer.forward(fx.int64(host, wide)));
            assertEquals(Shape.of(3, 3), out.shape());
            TrainParity.assertBitIdentical("CPU embedding", rows(table, 3, new int[]{4, 0, 2}), out.toFloatArray());
            Tensor bad = fx.keep(Tensor.fromArray(host, new int[]{1, 5}));
            assertThrows(IndexOutOfBoundsException.class, () -> layer.forward(bad));
        }

        /** Both loadStateDict overloads write into Linear's existing parameters, which its fields keep using. */
        @Test
        public void linearLoadStateDictKeepsItsParameters() {
            Linear source = fx.module(new Linear(3, 2));
            source.initialize(host, new Random(WEIGHT_SEED));
            for (boolean strict : new boolean[]{false, true}) {
                Linear target = fx.module(new Linear(3, 2));
                target.initialize(host, new Random(WEIGHT_SEED + 1));
                Parameter weight = target.weight();
                Parameter bias = target.bias();
                if (strict) {
                    target.loadStateDict(source.stateDict(), true);
                } else {
                    target.loadStateDict(source.stateDict());
                }
                assertSame(weight, target.weight());
                assertSame(bias, target.bias());
                TrainParity.assertBitIdentical("loaded weight", source.weight().tensor().toFloatArray(),
                        target.weight().tensor().toFloatArray());
                Tensor in = fx.upload(host, new float[]{0.5f, -1f, 2f}, 1, 3);
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    Tensor expected = fx.keep(source.forward(in));
                    Tensor actual = fx.keep(target.forward(in));
                    TrainParity.assertBitIdentical("forward after load", expected.toFloatArray(),
                            actual.toFloatArray());
                }
            }
        }
    }

    // ------------------------------------------------------------------ recording on the CPU reference path

    public static class Recording {
        private final Fixture fx = new Fixture();
        private final ComputeBackend host = TrainParity.hostBackend();

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
        }

        @After
        public void tearDown() {
            fx.close();
        }

        @Test
        public void linearDeliversExactWeightAndBiasGradients() {
            GradientCases.linear(fx, host, "CPU ");
        }

        @Test
        public void conv2dDeliversParameterAndInputGradients() {
            GradientCases.conv2d(fx, host, "CPU ");
        }

        @Test
        public void convolutionVariantsRecordTheirGradientFunctions() {
            GradientCases.convolutionVariants(fx, host, "CPU ");
        }

        @Test
        public void poolingRoutesGradientsThroughItsWindows() {
            GradientCases.pooling(fx, host, "CPU ");
        }

        @Test
        public void embeddingPaddingRowGetsNoGradient() {
            GradientCases.embedding(fx, host, "CPU ");
        }

        @Test
        public void dropoutGradientIsItsMask() {
            GradientCases.dropout(fx, host, "CPU ");
        }

        @Test
        public void flattenRecordsAReshape() {
            Tensor input = fx.leaf(host, TrainParity.uniform(3, 24, -1f, 1f), 2, 3, 4);
            int before = ComputationGraph.current().size();
            Tensor out = fx.keep(fx.module(new Flatten()).forward(input));
            assertEquals(Shape.of(2, 12), out.shape());
            assertTrue("Flatten records ReshapeGrad", out.gradFn() instanceof GradFunctions.ReshapeGrad);
            GradientCases.backwardOfSum(out, before, "Flatten");
            TrainParity.assertBitIdentical("Flatten input gradient", GradientCases.filled(24, 1f),
                    fx.takeGrad(input));
        }

        @Test
        public void forwardsUnderNoGradRecordNothing() {
            Linear linear = fx.module(new Linear(4, 2));
            linear.initialize(host, new Random(WEIGHT_SEED));
            Conv2d conv = fx.module(new Conv2d(2, 3, new int[]{3, 3}));
            conv.initialize(host, new Random(WEIGHT_SEED));
            Embedding embedding = fx.module(new Embedding(4, 3));
            embedding.initialize(host, new Random(WEIGHT_SEED));
            Dropout dropout = fx.module(new Dropout(0.5f));
            Tensor input = fx.leaf(host, TrainParity.uniform(4, 12, -1f, 1f), 3, 4);
            int before = ComputationGraph.current().size();
            Tensor[] outs;
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                outs = new Tensor[]{
                        fx.keep(linear.forward(input)),
                        fx.keep(conv.forward(fx.upload(host, new float[2 * 2 * 4 * 5], 2, 2, 4, 5))),
                        fx.keep(embedding.forward(fx.keep(Tensor.fromArray(host, new int[]{1, 0})))),
                        fx.keep(dropout.forward(input))
                };
            }
            for (Tensor out : outs) {
                assertFalse("output under noGrad requires grad", out.requiresGrad());
                assertNull("output under noGrad has a gradient function", out.gradFn());
            }
            assertEquals("nodes recorded under noGrad", before, ComputationGraph.current().size());
        }

        /**
         * Dropout inside a checkpointed segment: the recompute replays the forward pass's mask
         * ({@link CheckpointFunction#replayable}), so the input gradient is that mask, and the module's random stream
         * advanced only once (the next forward draws the second mask of the seed).
         */
        @Test
        public void checkpointedDropoutReplaysItsMaskInTheRecompute() {
            Dropout layer = fx.module(new Dropout(0.5f));
            layer.setSeed(33L);
            Tensor input = fx.leaf(host, TrainParity.uniform(24, 4 * 5, -2f, 2f), 4, 5);
            int before = ComputationGraph.current().size();
            Tensor out = fx.keep(CheckpointFunction.checkpoint(layer::forward, input));
            float[] mask = layer.getLastMask().toFloatArray();
            GradientCases.backwardOfSum(out, before, "checkpointed Dropout");
            TrainParity.assertBitIdentical("checkpointed Dropout gradient", mask, fx.takeGrad(input));

            Random stream = new Random(33L);
            for (int i = 0; i < 20; i++) {
                stream.nextFloat();
            }
            float[] next = new float[20];
            for (int i = 0; i < next.length; i++) {
                next[i] = stream.nextFloat() < 0.5f ? 2f : 0f;
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                fx.keep(layer.forward(fx.upload(host, new float[20], 4, 5)));
            }
            TrainParity.assertBitIdentical("mask drawn after the recompute", next,
                    layer.getLastMask().toFloatArray());
        }
    }

    // ------------------------------------------------------------------ GPU against the CPU branch

    @RunWith(Parameterized.class)
    public static class GpuParity {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private final Fixture fx = new Fixture();
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;

        public GpuParity(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                fx.close();
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void conv2dOnATransposedInput() {
            ConvCase c = ConvCase.conv2d();
            Conv2d layer = fx.module(c.conv2dLayer());
            layer.initialize(gpu, new Random(WEIGHT_SEED));
            float[] bias = c.bias();
            fx.write(layer.bias().tensor(), bias);
            float[] x = c.input();
            Tensor input = fx.keep(fx.upload(gpu, transposeHw(x, c.b * c.c, c.h, c.w), c.b, c.c, c.w, c.h)
                    .transpose(0, 1, 3, 2));
            Tensor out = fx.keep(layer.forward(input));
            Ref ref = c.directRef(x, hostWeights(c.conv2dLayer()), bias);
            assertEquals(Shape.of(c.b, c.oc, ref.oh, ref.ow), out.shape());
            ref.check(api + " conv2d", out.toFloatArray());
        }

        @Test
        public void conv1dMatchesTheDirectDefinition() {
            ConvCase c = ConvCase.conv1d();
            Conv1d layer = fx.module(c.conv1dLayer());
            layer.initialize(gpu, new Random(WEIGHT_SEED));
            float[] bias = c.bias();
            fx.write(layer.bias().tensor(), bias);
            float[] x = c.input();
            Tensor out = fx.keep(layer.forward(fx.upload(gpu, x, c.b, c.c, c.w)));
            Ref ref = c.directRef(x, hostWeights(c.conv1dLayer()), bias);
            assertEquals(Shape.of(c.b, c.oc, ref.ow), out.shape());
            ref.check(api + " conv1d", out.toFloatArray());
        }

        @Test
        public void convTranspose2dMatchesTheScatterDefinition() {
            ConvCase c = ConvCase.convTranspose2d();
            ConvTranspose2d layer = fx.module(c.convTranspose2dLayer());
            layer.initialize(gpu, new Random(WEIGHT_SEED));
            float[] bias = c.bias();
            fx.write(layer.bias().tensor(), bias);
            float[] x = c.input();
            Tensor out = fx.keep(layer.forward(fx.upload(gpu, x, c.b, c.c, c.h, c.w)));
            Ref ref = c.scatterRef(x, hostWeights(c.convTranspose2dLayer()), bias);
            assertEquals(Shape.of(c.b, c.oc, ref.oh, ref.ow), out.shape());
            ref.check(api + " convTranspose2d", out.toFloatArray());
        }

        @Test
        public void convolutionsWithoutBiasMatchTheDirectDefinition() {
            ConvCase c = ConvCase.conv2d();
            Conv2d layer = fx.module(new Conv2d(c.c, c.oc, c.k, c.s, c.p, c.d, c.groups, false));
            layer.initialize(gpu, new Random(WEIGHT_SEED));
            float[] x = c.input();
            Tensor out = fx.keep(layer.forward(fx.upload(gpu, x, c.b, c.c, c.h, c.w)));
            Ref ref = c.directRef(x, layer.weight().tensor().toFloatArray(), new float[c.oc]);
            assertEquals(Shape.of(c.b, c.oc, ref.oh, ref.ow), out.shape());
            ref.check(api + " conv2d without bias", out.toFloatArray());

            ConvCase t = ConvCase.convTranspose2d();
            ConvTranspose2d transposed = fx.module(new ConvTranspose2d(t.c, t.oc, t.k, t.s, t.p, t.op, t.d,
                    t.groups, false));
            transposed.initialize(gpu, new Random(WEIGHT_SEED));
            float[] tx = t.input();
            Tensor tout = fx.keep(transposed.forward(fx.upload(gpu, tx, t.b, t.c, t.h, t.w)));
            Ref tref = t.scatterRef(tx, transposed.weight().tensor().toFloatArray(), new float[t.oc]);
            assertEquals(Shape.of(t.b, t.oc, tref.oh, tref.ow), tout.shape());
            tref.check(api + " convTranspose2d without bias", tout.toFloatArray());
        }

        /**
         * Floor and ceil mode over a 10 x 9 input (PyTorch output sizes): ceil mode adds an output row, while the
         * extra column it would add starts at column 9, inside the right padding, and is dropped.
         */
        @Test
        public void maxPool2dIsBitIdentical() {
            float[] x = TrainParity.uniform(61, 2 * 3 * 10 * 9, -10f, 10f);
            x[7] = Float.NaN;
            x[50] = Float.POSITIVE_INFINITY;
            for (boolean ceil : new boolean[]{false, true}) {
                MaxPool2d layer = new MaxPool2d(new int[]{3, 2}, new int[]{2, 2}, new int[]{1, 1},
                        new int[]{2, 1}, ceil);
                Tensor ref = fx.keep(layer.forward(fx.upload(host, x, 2, 3, 10, 9)));
                Tensor out = fx.keep(layer.forward(fx.upload(gpu, x, 2, 3, 10, 9)));
                assertEquals(ceil ? Shape.of(2, 3, 5, 5) : Shape.of(2, 3, 4, 5), ref.shape());
                assertEquals(ref.shape(), out.shape());
                TrainParity.assertBitIdentical(api + " maxPool2d ceil=" + ceil, ref.toFloatArray(),
                        out.toFloatArray());
            }
        }

        /**
         * PyTorch's average-pool windows over a 5 x 7 input with a 2 x 2 window, stride 2 and padding 1: ceil mode
         * drops the last row and column window (it would start inside the right padding), so both modes give 3 x 4,
         * and count_include_pad divides the edge windows by their size clipped to the padded extent.
         */
        @Test
        public void avgPool2dIsInTheDivisionClass() {
            float[] x = TrainParity.uniform(71, 2 * 3 * 5 * 7, -10f, 10f);
            for (boolean ceil : new boolean[]{false, true}) {
                for (boolean includePad : new boolean[]{false, true}) {
                    AvgPool2d layer = new AvgPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{1, 1}, ceil,
                            includePad);
                    Tensor ref = fx.keep(layer.forward(fx.upload(host, x, 2, 3, 5, 7)));
                    Tensor out = fx.keep(layer.forward(fx.upload(gpu, x, 2, 3, 5, 7)));
                    assertEquals(Shape.of(2, 3, 3, 4), ref.shape());
                    assertEquals(ref.shape(), out.shape());
                    TrainParity.assertDivSqrt(api + " avgPool2d ceil=" + ceil + " includePad=" + includePad,
                            ref.toFloatArray(), out.toFloatArray());
                }
            }
        }

        /**
         * PyTorch's adaptive windows: 10 x 7 into 3 x 4 (neighbouring windows overlap), 4 x 9 ({@code O > L} in the
         * width) and the identity size.
         */
        @Test
        public void adaptiveAvgPool2dIsInTheDivisionClass() {
            float[] x = TrainParity.uniform(81, 2 * 3 * 10 * 7, -10f, 10f);
            for (int[] size : new int[][]{{3, 4}, {4, 9}, {10, 7}}) {
                AdaptiveAvgPool2d layer = new AdaptiveAvgPool2d(size);
                Tensor ref = fx.keep(layer.forward(fx.upload(host, x, 2, 3, 10, 7)));
                Tensor out = fx.keep(layer.forward(fx.upload(gpu, x, 2, 3, 10, 7)));
                assertEquals(ref.shape(), out.shape());
                TrainParity.assertDivSqrt(api + " adaptiveAvgPool2d " + size[0] + "x" + size[1],
                        ref.toFloatArray(), out.toFloatArray());
            }
        }

        @Test
        public void embeddingIsBitIdenticalForInt32AndInt64Indices() {
            Embedding onGpu = fx.module(new Embedding(7, 5, 2));
            onGpu.initialize(gpu, new Random(WEIGHT_SEED));
            Embedding onHost = fx.module(new Embedding(7, 5, 2));
            onHost.initialize(host, new Random(WEIGHT_SEED));
            int[] idx = {6, 0, 2, 2, 5, 1};
            Tensor ref = fx.keep(onHost.forward(fx.keep(fx.keep(Tensor.fromArray(host, idx))
                    .reshape(Shape.of(2, 3)))));
            Tensor out32 = fx.keep(onGpu.forward(fx.keep(fx.keep(Tensor.fromArray(gpu, idx)).reshape(Shape.of(2, 3)))));
            assertEquals(Shape.of(2, 3, 5), out32.shape());
            TrainParity.assertBitIdentical(api + " embedding INT32", ref.toFloatArray(), out32.toFloatArray());
            long[] wide = new long[idx.length];
            for (int i = 0; i < idx.length; i++) {
                wide[i] = idx[i] + ((long) (i - 3) << 32);
            }
            Tensor out64 = fx.keep(onGpu.forward(fx.int64(gpu, wide)));
            assertEquals(Shape.of(6, 5), out64.shape());
            TrainParity.assertBitIdentical(api + " embedding INT64", ref.toFloatArray(), out64.toFloatArray());
        }

        @Test
        public void embeddingReportsAnOutOfRangeIndexAtTheNextSynchronization() {
            Embedding layer = fx.module(new Embedding(4, 3));
            layer.initialize(gpu, new Random(WEIGHT_SEED));
            fx.keep(layer.forward(fx.keep(Tensor.fromArray(gpu, new int[]{0, 4, 1}))));
            assertThrows(IndexOutOfBoundsException.class, () -> session.gpu().checkErrors());
        }

        @Test
        public void embeddingCopiesIndicesFromAnotherBackendToTheTableBackend() {
            Embedding layer = fx.module(new Embedding(4, 3));
            layer.initialize(gpu, new Random(WEIGHT_SEED));
            float[] table = layer.weight().tensor().toFloatArray();
            Tensor out = fx.keep(layer.forward(fx.keep(fx.keep(Tensor.fromArray(host, new int[]{3, 1, 0, 3}))
                    .reshape(Shape.of(2, 2)))));
            assertSame(gpu, out.backend());
            assertEquals(Shape.of(2, 2, 3), out.shape());
            TrainParity.assertBitIdentical(api + " embedding host indices", rows(table, 3, new int[]{3, 1, 0, 3}),
                    out.toFloatArray());
        }

        @Test
        public void layerGradientsReachParametersAndInputsOnTheDevice() {
            GradientCases.linear(fx, gpu, api + " ");
            GradientCases.conv2d(fx, gpu, api + " ");
            GradientCases.convolutionVariants(fx, gpu, api + " ");
            GradientCases.pooling(fx, gpu, api + " ");
            GradientCases.embedding(fx, gpu, api + " ");
            GradientCases.dropout(fx, gpu, api + " ");
        }

        @Test
        public void dropoutMaskIsSharedAcrossBackendsAndAppliedOnTheDevice() {
            Dropout onGpu = fx.module(new Dropout(0.4f));
            Dropout onHost = fx.module(new Dropout(0.4f));
            onGpu.setSeed(99L);
            onHost.setSeed(99L);
            float[] stored = TrainParity.uniform(91, 6 * 11, -5f, 5f);
            stored[3] = Float.NaN;
            stored[17] = Float.POSITIVE_INFINITY;
            float[] x = transposeHw(stored, 1, 11, 6);
            Tensor input = fx.keep(fx.upload(gpu, stored, 11, 6).transpose(1, 0));
            for (int pass = 0; pass < 2; pass++) {
                Tensor out = fx.keep(onGpu.forward(input));
                Tensor expected = fx.keep(onHost.forward(fx.upload(host, x, 6, 11)));
                assertEquals(Shape.of(6, 11), out.shape());
                TrainParity.assertBitIdentical(api + " dropout mask pass " + pass,
                        onHost.getLastMask().toFloatArray(), onGpu.getLastMask().toFloatArray());
                TrainParity.assertSingleRounding(api + " dropout output pass " + pass, expected.toFloatArray(),
                        out.toFloatArray());
            }
        }

        /** The weights a layer built like {@code layer} receives from {@link #WEIGHT_SEED} on the host backend. */
        private float[] hostWeights(Module layer) {
            fx.module(layer);
            if (layer instanceof Conv2d) {
                ((Conv2d) layer).initialize(host, new Random(WEIGHT_SEED));
                return ((Conv2d) layer).weight().tensor().toFloatArray();
            }
            if (layer instanceof Conv1d) {
                ((Conv1d) layer).initialize(host, new Random(WEIGHT_SEED));
                return ((Conv1d) layer).weight().tensor().toFloatArray();
            }
            ((ConvTranspose2d) layer).initialize(host, new Random(WEIGHT_SEED));
            return ((ConvTranspose2d) layer).weight().tensor().toFloatArray();
        }
    }

    // ------------------------------------------------------------------ gradient cases

    /**
     * Gradient cases shared by {@link Recording} (host) and {@link GpuParity} (each GPU API). Each runs one forward in
     * grad mode, checks the gradient function its output carries, back-propagates the output sum (an upstream gradient
     * of ones) and checks every delivered gradient: exactly where the sums are exact (small integers and powers of
     * two), in the reduction class against a double reference otherwise.
     */
    static final class GradientCases {

        private GradientCases() {
        }

        /** {@code sum(out).backward()}, then checks the graph is back at {@code sizeBefore} nodes. */
        static void backwardOfSum(Tensor out, int sizeBefore, String what) {
            try (Tensor loss = TensorOps.sum(out)) {
                loss.backward();
            }
            assertEquals(what + ": nodes left after backward", sizeBefore, ComputationGraph.current().size());
        }

        /** {@code out = x W^T + b} on a {@code [3, 4]} input: {@code gW[o, :]} is the column sums, {@code gb = 3}. */
        static void linear(Fixture fx, ComputeBackend b, String what) {
            Linear layer = fx.module(new Linear(4, 2));
            layer.initialize(b, new Random(WEIGHT_SEED));
            float[] x = {1f, 2f, 3f, 4f, 0.5f, -1f, 2f, 0f, -2f, 1f, 0.25f, 3f};
            int before = ComputationGraph.current().size();
            Tensor out = fx.keep(layer.forward(fx.upload(b, x, 3, 4)));
            assertTrue(what + "Linear records AddGrad last", out.gradFn() instanceof GradFunctions.AddGrad);
            backwardOfSum(out, before, what + "Linear");
            float[] columnSums = {-0.5f, 2f, 5.25f, 7f};
            TrainParity.assertBitIdentical(what + "Linear weight gradient", concat(columnSums, columnSums),
                    fx.takeGrad(layer.weight().tensor()));
            TrainParity.assertBitIdentical(what + "Linear bias gradient", filled(2, 3f),
                    fx.takeGrad(layer.bias().tensor()));
        }

        /**
         * A padded 3x3 convolution on an integer-valued leaf input: {@code gb[oc] = B*OH*OW}, {@code gw} sums input
         * taps (exact integers) and {@code gx} sums the weights of the taps that read each input element.
         */
        static void conv2d(Fixture fx, ComputeBackend b, String what) {
            int n = 2, c = 2, h = 4, w = 5, oc = 3;
            Conv2d layer = fx.module(new Conv2d(c, oc, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}));
            layer.initialize(b, new Random(WEIGHT_SEED));
            float[] x = new float[n * c * h * w];
            for (int i = 0; i < x.length; i++) {
                x[i] = (i % 7) - 3;
            }
            Tensor input = fx.leaf(b, x, n, c, h, w);
            int before = ComputationGraph.current().size();
            Tensor out = fx.keep(layer.forward(input));
            assertEquals(Shape.of(n, oc, h, w), out.shape());
            assertTrue(what + "Conv2d records Conv2dGrad", out.gradFn() instanceof ConvGradFunctions.Conv2dGrad);
            float[] weights = layer.weight().tensor().toFloatArray();
            backwardOfSum(out, before, what + "Conv2d");

            Ref gw = new Ref(oc * c * 9, 0, 0);
            Ref gx = new Ref(x.length, 0, 0);
            for (int bi = 0; bi < n; bi++) {
                for (int o = 0; o < oc; o++) {
                    for (int oh = 0; oh < h; oh++) {
                        for (int ow = 0; ow < w; ow++) {
                            for (int ic = 0; ic < c; ic++) {
                                for (int kh = 0; kh < 3; kh++) {
                                    for (int kw = 0; kw < 3; kw++) {
                                        int ih = oh + kh - 1;
                                        int iw = ow + kw - 1;
                                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                            int xi = ((bi * c + ic) * h + ih) * w + iw;
                                            int wi = ((o * c + ic) * 3 + kh) * 3 + kw;
                                            gw.add(wi, x[xi]);
                                            gx.add(xi, weights[wi]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            TrainParity.assertBitIdentical(what + "Conv2d bias gradient", filled(oc, n * h * w),
                    fx.takeGrad(layer.bias().tensor()));
            gw.check(what + "Conv2d weight gradient", fx.takeGrad(layer.weight().tensor()));
            gx.check(what + "Conv2d input gradient", fx.takeGrad(input));
        }

        /** Conv1d and ConvTranspose2d record their gradient functions and deliver every gradient. */
        static void convolutionVariants(Fixture fx, ComputeBackend b, String what) {
            Conv1d conv1d = fx.module(new Conv1d(2, 4, 3, 2, 1));
            conv1d.initialize(b, new Random(WEIGHT_SEED));
            Tensor x1 = fx.leaf(b, TrainParity.uniform(21, 2 * 2 * 9, -1f, 1f), 2, 2, 9);
            int before = ComputationGraph.current().size();
            Tensor y1 = fx.keep(conv1d.forward(x1));
            assertEquals(Shape.of(2, 4, 5), y1.shape());
            assertTrue(what + "Conv1d records Conv1dGrad", y1.gradFn() instanceof ConvGradFunctions.Conv1dGrad);
            backwardOfSum(y1, before, what + "Conv1d");
            TrainParity.assertBitIdentical(what + "Conv1d bias gradient", filled(4, 2 * 5),
                    fx.takeGrad(conv1d.bias().tensor()));
            fx.takeGrad(conv1d.weight().tensor());
            fx.takeGrad(x1);

            ConvTranspose2d up = fx.module(new ConvTranspose2d(2, 3, new int[]{2, 2}, new int[]{2, 2},
                    new int[]{0, 0}));
            up.initialize(b, new Random(WEIGHT_SEED));
            Tensor x2 = fx.leaf(b, TrainParity.uniform(22, 2 * 3 * 3, -1f, 1f), 1, 2, 3, 3);
            Tensor y2 = fx.keep(up.forward(x2));
            assertEquals(Shape.of(1, 3, 6, 6), y2.shape());
            assertTrue(what + "ConvTranspose2d records ConvTranspose2dGrad",
                    y2.gradFn() instanceof ConvGradFunctions.ConvTranspose2dGrad);
            backwardOfSum(y2, before, what + "ConvTranspose2d");
            TrainParity.assertBitIdentical(what + "ConvTranspose2d bias gradient", filled(3, 36f),
                    fx.takeGrad(up.bias().tensor()));
            fx.takeGrad(up.weight().tensor());
            fx.takeGrad(x2);
        }

        /**
         * 2x2 pooling of a strictly increasing input: max pooling routes each window's gradient to its last tap,
         * average pooling gives every element 1/4, adaptive pooling to 1x1 gives every element 1/16. Adaptive pooling
         * of 4 x 4 to 3 x 3 has PyTorch's overlapping 2 x 2 windows (rows {0, 1}, {1, 2}, {2, 3}, columns likewise),
         * so element {@code (r, c)} receives {@code m[r]*m[c]/4} with window multiplicities {@code m = {1, 2, 2, 1}}.
         */
        static void pooling(Fixture fx, ComputeBackend b, String what) {
            float[] x = new float[2 * 4 * 4];
            float[] maxGrad = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                x[i] = i * 0.5f - 7f;
                int row = (i / 4) % 4;
                int col = i % 4;
                maxGrad[i] = row % 2 == 1 && col % 2 == 1 ? 1f : 0f;
            }
            Tensor input = fx.leaf(b, x, 1, 2, 4, 4);
            int before = ComputationGraph.current().size();

            Tensor max = fx.keep(fx.module(new MaxPool2d(new int[]{2, 2})).forward(input));
            assertTrue(what + "MaxPool2d records MaxPool2dGrad",
                    max.gradFn() instanceof ConvGradFunctions.MaxPool2dGrad);
            backwardOfSum(max, before, what + "MaxPool2d");
            TrainParity.assertBitIdentical(what + "MaxPool2d input gradient", maxGrad, fx.takeGrad(input));

            Tensor avg = fx.keep(fx.module(new AvgPool2d(new int[]{2, 2})).forward(input));
            assertTrue(what + "AvgPool2d records AvgPool2dGrad",
                    avg.gradFn() instanceof ConvGradFunctions.AvgPool2dGrad);
            backwardOfSum(avg, before, what + "AvgPool2d");
            TrainParity.assertBitIdentical(what + "AvgPool2d input gradient", filled(x.length, 0.25f),
                    fx.takeGrad(input));

            Tensor adaptive = fx.keep(fx.module(new AdaptiveAvgPool2d(1)).forward(input));
            assertTrue(what + "AdaptiveAvgPool2d records AdaptiveAvgPool2dGrad",
                    adaptive.gradFn() instanceof ConvGradFunctions.AdaptiveAvgPool2dGrad);
            backwardOfSum(adaptive, before, what + "AdaptiveAvgPool2d");
            TrainParity.assertBitIdentical(what + "AdaptiveAvgPool2d input gradient", filled(x.length, 0.0625f),
                    fx.takeGrad(input));

            Tensor overlapping = fx.keep(fx.module(new AdaptiveAvgPool2d(3)).forward(input));
            assertEquals(Shape.of(1, 2, 3, 3), overlapping.shape());
            backwardOfSum(overlapping, before, what + "AdaptiveAvgPool2d 4x4 to 3x3");
            int[] multiplicity = {1, 2, 2, 1};
            float[] overlapGrad = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                overlapGrad[i] = multiplicity[(i / 4) % 4] * multiplicity[i % 4] * 0.25f;
            }
            TrainParity.assertBitIdentical(what + "AdaptiveAvgPool2d overlapping-window input gradient", overlapGrad,
                    fx.takeGrad(input));
        }

        /** Indices {0, 2, 2, 1, 0} with padding row 2: row 0 gets 2, row 1 gets 1, rows 2 and 3 get nothing. */
        static void embedding(Fixture fx, ComputeBackend b, String what) {
            Embedding layer = fx.module(new Embedding(4, 3, 2));
            layer.initialize(b, new Random(WEIGHT_SEED));
            Tensor indices = fx.keep(Tensor.fromArray(b, new int[]{0, 2, 2, 1, 0}));
            int before = ComputationGraph.current().size();
            Tensor out = fx.keep(layer.forward(indices));
            assertTrue(what + "Embedding records IndexSelectRowsGrad",
                    out.gradFn() instanceof IndexGradFunctions.IndexSelectRowsGrad);
            backwardOfSum(out, before, what + "Embedding");
            TrainParity.assertBitIdentical(what + "Embedding table gradient",
                    new float[]{2f, 2f, 2f, 1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f},
                    fx.takeGrad(layer.weight().tensor()));
        }

        /** Under an upstream gradient of ones the dropout input gradient is its mask, which outlives the backward. */
        static void dropout(Fixture fx, ComputeBackend b, String what) {
            Dropout layer = fx.module(new Dropout(0.5f));
            layer.setSeed(21L);
            Tensor input = fx.leaf(b, TrainParity.uniform(23, 6 * 7, -2f, 2f), 6, 7);
            int before = ComputationGraph.current().size();
            Tensor out = fx.keep(layer.forward(input));
            assertTrue(what + "Dropout records MaskScaleGrad",
                    out.gradFn() instanceof IndexGradFunctions.MaskScaleGrad);
            assertFalse(what + "Dropout mask requires grad", layer.getLastMask().requiresGrad());
            float[] mask = layer.getLastMask().toFloatArray();
            backwardOfSum(out, before, what + "Dropout");
            TrainParity.assertBitIdentical(what + "Dropout input gradient", mask, fx.takeGrad(input));
            TrainParity.assertBitIdentical(what + "Dropout mask after backward", mask,
                    layer.getLastMask().toFloatArray());
        }

        static float[] filled(int n, float value) {
            float[] out = new float[n];
            Arrays.fill(out, value);
            return out;
        }

        static float[] concat(float[] a, float[] b) {
            float[] out = new float[a.length + b.length];
            System.arraycopy(a, 0, out, 0, a.length);
            System.arraycopy(b, 0, out, a.length, b.length);
            return out;
        }
    }

    // ------------------------------------------------------------------ cases and references

    /** One convolution configuration (grouped, strided, padded and dilated) with its seeded input and bias. */
    static final class ConvCase {
        final int b, c, h, w, oc, groups;
        final int[] k, s, p, op, d;

        private ConvCase(int b, int c, int h, int w, int oc, int groups, int[] k, int[] s, int[] p, int[] op,
                         int[] d) {
            this.b = b;
            this.c = c;
            this.h = h;
            this.w = w;
            this.oc = oc;
            this.groups = groups;
            this.k = k;
            this.s = s;
            this.p = p;
            this.op = op;
            this.d = d;
        }

        static ConvCase conv2d() {
            return new ConvCase(2, 6, 11, 9, 4, 2, new int[]{3, 2}, new int[]{2, 1}, new int[]{1, 1}, null,
                    new int[]{1, 2});
        }

        /** A 1-D convolution as the 2-D case with height 1 (kernel 3, stride 2, padding 1, dilation 2). */
        static ConvCase conv1d() {
            return new ConvCase(3, 4, 1, 17, 6, 2, new int[]{1, 3}, new int[]{1, 2}, new int[]{0, 1}, null,
                    new int[]{1, 2});
        }

        static ConvCase convTranspose2d() {
            return new ConvCase(2, 4, 5, 6, 6, 2, new int[]{3, 4}, new int[]{2, 3}, new int[]{1, 2},
                    new int[]{1, 2}, new int[]{1, 2});
        }

        Conv2d conv2dLayer() {
            return new Conv2d(c, oc, k, s, p, d, groups, true);
        }

        Conv1d conv1dLayer() {
            return new Conv1d(c, oc, k[1], s[1], p[1], d[1], groups, true);
        }

        ConvTranspose2d convTranspose2dLayer() {
            return new ConvTranspose2d(c, oc, k, s, p, op, d, groups, true);
        }

        float[] input() {
            return TrainParity.uniform(11, b * c * h * w, -3f, 3f);
        }

        float[] bias() {
            return TrainParity.uniform(13, oc, -2f, 2f);
        }

        /** Direct convolution in double ({@code Conv2d} definition; weights {@code [OC, C/groups, KH, KW]}). */
        Ref directRef(float[] x, float[] wt, float[] bias) {
            int oh = (h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
            int ow = (w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
            Ref ref = new Ref(b * oc * oh * ow, oh, ow);
            int cg = c / groups, ocg = oc / groups;
            for (int n = 0; n < b; n++) {
                for (int o = 0; o < oc; o++) {
                    for (int y = 0; y < oh; y++) {
                        for (int z = 0; z < ow; z++) {
                            int i = ((n * oc + o) * oh + y) * ow + z;
                            ref.add(i, bias[o]);
                            for (int ic = 0; ic < cg; ic++) {
                                int cin = (o / ocg) * cg + ic;
                                for (int kh = 0; kh < k[0]; kh++) {
                                    for (int kw = 0; kw < k[1]; kw++) {
                                        int ih = y * s[0] + kh * d[0] - p[0];
                                        int iw = z * s[1] + kw * d[1] - p[1];
                                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                            ref.add(i, (double) x[((n * c + cin) * h + ih) * w + iw]
                                                    * wt[((o * cg + ic) * k[0] + kh) * k[1] + kw]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return ref;
        }

        /**
         * Transposed convolution in double by its scatter definition (each input element adds {@code x * w} to
         * output {@code (ih*SH - PH + kh*DH, iw*SW - PW + kw*DW)}; weights {@code [C, OC/groups, KH, KW]}),
         * independent of the gather form the layer and the kernel use.
         */
        Ref scatterRef(float[] x, float[] wt, float[] bias) {
            int oh = (h - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + op[0] + 1;
            int ow = (w - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + op[1] + 1;
            Ref ref = new Ref(b * oc * oh * ow, oh, ow);
            for (int i = 0; i < b * oc * oh * ow; i++) {
                ref.add(i, bias[(i / (oh * ow)) % oc]);
            }
            int cg = c / groups, ocg = oc / groups;
            for (int n = 0; n < b; n++) {
                for (int cin = 0; cin < c; cin++) {
                    int g = cin / cg;
                    for (int ih = 0; ih < h; ih++) {
                        for (int iw = 0; iw < w; iw++) {
                            double v = x[((n * c + cin) * h + ih) * w + iw];
                            for (int o = 0; o < ocg; o++) {
                                for (int kh = 0; kh < k[0]; kh++) {
                                    for (int kw = 0; kw < k[1]; kw++) {
                                        int y = ih * s[0] - p[0] + kh * d[0];
                                        int z = iw * s[1] - p[1] + kw * d[1];
                                        if (y >= 0 && y < oh && z >= 0 && z < ow) {
                                            ref.add(((n * oc + g * ocg + o) * oh + y) * ow + z,
                                                    v * wt[((cin * ocg + o) * k[0] + kh) * k[1] + kw]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return ref;
        }
    }

    /** Double-precision sums with their absolute-term sums and term counts, checked in the reduction class. */
    static final class Ref {
        final int oh;
        final int ow;
        final double[] value;
        final double[] sumAbs;
        final int[] terms;

        Ref(int n, int oh, int ow) {
            this.oh = oh;
            this.ow = ow;
            this.value = new double[n];
            this.sumAbs = new double[n];
            this.terms = new int[n];
        }

        void add(int i, double term) {
            value[i] += term;
            sumAbs[i] += Math.abs(term);
            terms[i]++;
        }

        void check(String what, float[] actual) {
            assertEquals(what + ": length", value.length, actual.length);
            for (int i = 0; i < value.length; i++) {
                TrainParity.assertReduction(what + " element " + i, value[i], actual[i], Math.max(1, terms[i]),
                        sumAbs[i]);
            }
        }
    }

    /** Rows {@code idx} of a row-major table with {@code dim} columns, concatenated. */
    static float[] rows(float[] table, int dim, int[] idx) {
        float[] out = new float[idx.length * dim];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(table, idx[i] * dim, out, i * dim, dim);
        }
        return out;
    }

    /** {@code planes} row-major {@code [rows, cols]} matrices transposed to {@code [cols, rows]}. */
    static float[] transposeHw(float[] x, int planes, int rows, int cols) {
        float[] out = new float[x.length];
        for (int pl = 0; pl < planes; pl++) {
            for (int r = 0; r < rows; r++) {
                for (int col = 0; col < cols; col++) {
                    out[(pl * cols + col) * rows + r] = x[(pl * rows + r) * cols + col];
                }
            }
        }
        return out;
    }

    /**
     * Tensors and modules a test creates, closed after it. The fixture also opens a {@link ComputationGraph#step()} on
     * the test thread when it is created, so the nodes a test records and does not back-propagate are discarded when
     * it closes.
     */
    static final class Fixture implements AutoCloseable {
        private final List<AutoCloseable> owned = new ArrayList<>();
        private final ComputationGraph.StepScope step = ComputationGraph.current().step();

        <T extends AutoCloseable> T keep(T resource) {
            owned.add(resource);
            return resource;
        }

        <M extends Module> M module(M m) {
            return keep(m);
        }

        Tensor upload(ComputeBackend backend, float[] data, int... dims) {
            return keep(Tensor.fromStorage(backend, Shape.of(dims), DType.FLOAT32,
                    TensorStorage.fromHostFloat(backend, data)));
        }

        Tensor int64(ComputeBackend backend, long[] values) {
            ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
            for (long v : values) {
                bytes.putLong(v);
            }
            bytes.flip();
            return keep(Tensor.fromStorage(backend, Shape.vector(values.length), DType.INT64,
                    TensorStorage.fromHost(backend, bytes, DType.INT64)));
        }

        /**
         * Overwrites a parameter tensor's elements (a dense FLOAT32 tensor at storage offset 0), under
         * {@link ComputationGraph#noGrad()} like every parameter write.
         */
        void write(Tensor target, float[] values) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                GpuEvent done = target.storage().copyFromHostFloat(values, target.backend().transferQueue());
                if (done != null) {
                    done.close();
                }
            }
        }

        /** A new leaf (requires grad, no gradient function) holding {@code data} with shape {@code dims}. */
        Tensor leaf(ComputeBackend backend, float[] data, int... dims) {
            return keep(upload(backend, data, dims).withRequiresGrad(true));
        }

        /** Takes {@code t}'s gradient out of its slot (the fixture closes it) and returns its values. */
        float[] takeGrad(Tensor t) {
            Tensor g = t.grad();
            assertNotNull("no gradient was delivered", g);
            t.setGrad(null);
            keep(g);
            assertEquals("gradient shape", t.shape(), g.shape());
            return g.toFloatArray();
        }

        @Override
        public void close() {
            RuntimeException failure = null;
            try {
                step.close();
            } catch (RuntimeException e) {
                failure = new IllegalStateException("closing the test's graph step failed", e);
            }
            for (int i = owned.size() - 1; i >= 0; i--) {
                try {
                    owned.get(i).close();
                } catch (Exception e) {
                    if (failure == null) {
                        failure = new IllegalStateException("closing a test resource failed", e);
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            owned.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }
}
