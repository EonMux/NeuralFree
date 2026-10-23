package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Module#loadStateDict}, {@link Module#to(ComputeBackend)} and {@link Module#to(DType)} update parameters in
 * place, so modules that keep their parameters in fields (Conv2d) and optimizers holding the parameter list stay
 * valid, and buffers (BatchNorm running statistics and batch counter) follow the module. Parameter tensors stay
 * leaves with their trainable flag through every one of these paths, also in grad mode. {@link Module#stateDict()}
 * holds parameters and persistent buffers, and {@link Module#loadStateDict(Map, boolean)} restores both, refusing a
 * bad entry before it writes anything and, in strict mode, reporting missing and unexpected keys; both carry the
 * values of every dtype. {@link Module#to(DType)} accepts only floating-point dtypes and leaves integer state and
 * dtype-kept buffers (the BatchNorm batch counter) at their dtype. Runs on two host reference backends.
 */
public class ModuleStateTest {
    private ComputeBackend first;
    private ComputeBackend second;

    @Before
    public void setUp() {
        first = TrainParity.hostBackend();
        second = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(first);
        TrainGpu.release(second);
        first.close();
        second.close();
    }

    @Test
    public void loadStateDictWritesIntoTheExistingParameters() {
        try (Conv2d source = conv(first, 1L); Conv2d target = conv(first, 2L);
             Tensor input = input(first)) {
            Parameter weightBefore = target.getParameter("weight");
            Parameter biasBefore = target.getParameter("bias");

            target.loadStateDict(source.stateDict());

            assertSame(weightBefore, target.getParameter("weight"));
            assertSame(biasBefore, target.getParameter("bias"));
            assertArrayEquals(source.getParameter("weight").tensor().toFloatArray(),
                    target.getParameter("weight").tensor().toFloatArray(), 0f);
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor expected = source.forward(input); Tensor actual = target.forward(input)) {
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 0f);
            }
        }
    }

    @Test
    public void loadStateDictRejectsAWrongLength() {
        try (Conv2d conv = conv(first, 3L)) {
            Map<String, Object> state = new HashMap<>();
            state.put("bias", new float[]{1f, 2f});
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> conv.loadStateDict(state));
            assertTrue(e.getMessage(), e.getMessage().contains("bias"));
        }
    }

    @Test
    public void stateDictHoldsParametersThenPersistentBuffersPerModule() {
        try (Holder holder = new Holder(batchNorm(first, 2), new StatefulCache(first), conv(first, 20L))) {
            assertEquals(Arrays.asList("head.weight", "head.bias", "head.running_mean", "head.running_var",
                    "head.num_batches_tracked", "slot.scale", "slot.state", "tail.weight", "tail.bias"),
                    new ArrayList<>(holder.stateDict().keySet()));
            assertTrue("a non-persistent buffer is still a buffer",
                    holder.namedBuffers().containsKey("slot.cache"));
        }
    }

    @Test
    public void batchNormRoundTripRestoresRunningStatistics() {
        try (BatchNorm1d source = batchNorm(first, 3); BatchNorm1d target = batchNorm(first, 3)) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor in = matrix(first, new float[]{1f, 2f, 3f, 3f, 6f, 9f}, 2, 3);
                 Tensor out = source.forward(in)) {
                assertEquals(2, out.shape().dim(0));
            }
            Tensor targetMean = target.runningMean();

            target.loadStateDict(source.stateDict(), true);

            assertSame("a FLOAT32 buffer is loaded in place", targetMean, target.runningMean());
            assertArrayEquals(new float[]{0.2f, 0.4f, 0.6f}, target.runningMean().toFloatArray(), 1e-7f);
            assertArrayEquals(source.runningMean().toFloatArray(), target.runningMean().toFloatArray(), 0f);
            assertArrayEquals(source.runningVar().toFloatArray(), target.runningVar().toFloatArray(), 0f);
            assertEquals("the batch counter is restored", 1, target.numBatchesTrained());

            source.eval();
            target.eval();
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor in = matrix(first, new float[]{0.5f, -1f, 2f, 4f, 0f, -3f}, 2, 3);
                 Tensor expected = source.forward(in); Tensor actual = target.forward(in)) {
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 0f);
            }
        }
    }

    @Test
    public void loadStateDictConvertsBuffersOfAnotherDType() {
        try (BatchNorm1d bn = batchNorm(first, 2)) {
            bn.to(DType.FLOAT16);
            Map<String, Object> state = new HashMap<>();
            state.put("running_var", new float[]{0.5f, 2f});
            state.put("num_batches_tracked", new float[]{3f});

            bn.loadStateDict(state);

            assertEquals(DType.FLOAT16, bn.runningVar().dtype());
            try (Tensor var = bn.runningVar().to(DType.FLOAT32)) {
                assertArrayEquals(new float[]{0.5f, 2f}, var.toFloatArray(), 0f);
            }
            assertSame(bn.runningVar(), bn.namedBuffers().get("running_var"));
            assertEquals(DType.FLOAT32, bn.namedBuffers().get("num_batches_tracked").dtype());
            assertEquals(3, bn.numBatchesTrained());
        }
    }

    @Test
    public void strictLoadReportsMissingUnexpectedAndMistypedKeysAndChangesNothing() {
        try (BatchNorm1d bn = batchNorm(first, 2)) {
            Map<String, Object> state = new HashMap<>();
            state.put("weight", new float[]{3f, 4f});
            state.put("bias", "not a float array");
            state.put("running_mean", new float[]{5f, 6f});
            state.put("num_batches_tracked", new float[]{1f});
            state.put("running_std", new float[]{1f, 1f});

            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> bn.loadStateDict(state, true));

            String message = e.getMessage();
            assertTrue(message, message.contains("missing keys [running_var]"));
            assertTrue(message, message.contains("unexpected keys [running_std]"));
            assertTrue(message, message.contains("keys without a float[] value [bias]"));
            assertArrayEquals(new float[]{1f, 1f}, bn.weight().tensor().toFloatArray(), 0f);
            assertArrayEquals(new float[]{0f, 0f}, bn.runningMean().toFloatArray(), 0f);
            assertEquals(0, bn.numBatchesTrained());
        }
    }

    @Test
    public void nonStrictLoadIgnoresMissingUnexpectedAndMistypedKeys() {
        try (BatchNorm1d bn = batchNorm(first, 2)) {
            Map<String, Object> state = new HashMap<>();
            state.put("bias", "not a float array");
            state.put("running_mean", new float[]{5f, 6f});
            state.put("num_batches_tracked", new float[]{1f});
            state.put("running_std", new float[]{1f, 1f});

            bn.loadStateDict(state);

            assertArrayEquals(new float[]{5f, 6f}, bn.runningMean().toFloatArray(), 0f);
            assertEquals(1, bn.numBatchesTrained());
            assertArrayEquals(new float[]{1f, 1f}, bn.runningVar().toFloatArray(), 0f);
            assertArrayEquals(new float[]{0f, 0f}, bn.bias().tensor().toFloatArray(), 0f);
        }
    }

    @Test
    public void aWrongBufferLengthIsRefusedBeforeAnythingIsWritten() {
        try (BatchNorm1d bn = batchNorm(first, 2)) {
            Map<String, Object> state = new HashMap<>();
            state.put("weight", new float[]{3f, 4f});
            state.put("running_var", new float[]{1f, 2f, 3f});

            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> bn.loadStateDict(state));

            assertTrue(e.getMessage(), e.getMessage().contains("running_var"));
            assertArrayEquals(new float[]{1f, 1f}, bn.weight().tensor().toFloatArray(), 0f);
        }
    }

    @Test
    public void strictLoadAcceptsTheStateWithoutNonPersistentBuffers() {
        try (StatefulCache source = new StatefulCache(first); StatefulCache target = new StatefulCache(first)) {
            Map<String, Object> state = source.stateDict();
            assertEquals(Arrays.asList("scale", "state"), new ArrayList<>(state.keySet()));
            state.put("state", new float[]{7f, 8f});

            target.loadStateDict(state, true);

            assertArrayEquals(new float[]{7f, 8f}, target.buffer("state").toFloatArray(), 0f);
            assertArrayEquals(StatefulCache.CACHE, target.buffer("cache").toFloatArray(), 0f);
            Map<String, Object> withCache = source.stateDict();
            withCache.put("cache", new float[]{0f, 0f, 0f});
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> target.loadStateDict(withCache, true));
            assertTrue(e.getMessage(), e.getMessage().contains("unexpected keys [cache]"));
        }
    }

    @Test
    public void aParameterAndABufferCannotShareAStateDictKey() {
        try (StatefulCache module = new StatefulCache(first);
             Tensor extraBuffer = Tensor.zeros(first, Shape.vector(1));
             Parameter extraParameter = new Parameter(Tensor.zeros(first, Shape.vector(2)), "state")) {
            IllegalArgumentException buffer = assertThrows(IllegalArgumentException.class,
                    () -> module.registerBuffer("scale", extraBuffer));
            assertTrue(buffer.getMessage(), buffer.getMessage().contains("'scale'"));
            IllegalArgumentException parameter = assertThrows(IllegalArgumentException.class,
                    () -> module.registerParameter("state", extraParameter));
            assertTrue(parameter.getMessage(), parameter.getMessage().contains("'state'"));
            assertEquals(Arrays.asList("scale", "state"), new ArrayList<>(module.stateDict().keySet()));
        }
    }

    @Test
    public void targetCopyThroughTheStateDictCopiesParametersAndBuffers() {
        try (Holder source = new Holder(batchNorm(first, 2), conv(first, 21L), conv(first, 22L));
             Holder target = new Holder(batchNorm(first, 2), conv(first, 23L), conv(first, 24L))) {
            Map<String, Object> state = source.stateDict();
            state.put("head.running_mean", new float[]{0.25f, -0.75f});
            source.loadStateDict(state, true);

            target.loadStateDict(source.stateDict());

            assertEquals(source.stateDict().keySet(), target.stateDict().keySet());
            for (Map.Entry<String, Object> entry : source.stateDict().entrySet()) {
                assertArrayEquals(entry.getKey(), (float[]) entry.getValue(),
                        (float[]) target.stateDict().get(entry.getKey()), 0f);
            }
        }
    }

    @Test
    public void toBackendMovesParametersInPlaceAndBuffersAlong() {
        try (BatchNorm1d bn = new BatchNorm1d(3)) {
            bn.initialize(first);
            Parameter weight = bn.weight();
            Tensor oldMean = bn.runningMean();

            bn.to(second);

            assertSame(weight, bn.weight());
            assertSame(second, weight.backend());
            assertSame(second, bn.runningMean().backend());
            assertSame(second, bn.runningVar().backend());
            assertSame(second, bn.namedBuffers().get("num_batches_tracked").backend());
            assertThrows(IllegalStateException.class, oldMean::toFloatArray);
            assertEquals(3, bn.namedBuffers().size());
            assertSame(bn.runningMean(), bn.namedBuffers().get("running_mean"));

            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor in = matrix(second, new float[]{1f, 2f, 3f, 3f, 6f, 9f}, 2, 3);
                 Tensor out = bn.forward(in)) {
                assertSame(second, out.backend());
            }
            float[] mean = bn.runningMean().toFloatArray();
            assertEquals(0.9f * 0f + 0.1f * 2f, mean[0], 0f);
            assertEquals(0.9f * 0f + 0.1f * 4f, mean[1], 0f);
            assertEquals(0.9f * 0f + 0.1f * 6f, mean[2], 0f);
            assertEquals(1, bn.numBatchesTrained());
        }
    }

    @Test
    public void toDTypeConvertsParametersInPlaceAndFloatingBuffers() {
        try (BatchNorm1d bn = new BatchNorm1d(2)) {
            bn.initialize(first);
            Parameter bias = bn.bias();

            assertTrue(ComputationGraph.isGradEnabled());
            bn.to(DType.FLOAT16);

            assertSame(bias, bn.bias());
            assertEquals(DType.FLOAT16, bias.dtype());
            assertTrue(bias.tensor().requiresGrad());
            for (Parameter param : bn.parameters()) {
                assertTrue(param.requiresGrad());
                assertTrue(param.tensor().requiresGrad());
                assertNull(param.tensor().gradFn());
            }
            assertEquals(DType.FLOAT16, bn.runningVar().dtype());
            assertTrue(ComputationGraph.isGradEnabled());

            bn.to(DType.FLOAT32);
            assertArrayEquals(new float[]{1f, 1f}, bn.runningVar().toFloatArray(), 0f);
            assertArrayEquals(new float[]{1f, 1f}, bn.weight().tensor().toFloatArray(), 0f);
        }
    }

    @Test
    public void stateDictRoundTripsModulesOfEveryFloatingDType() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16, DType.FLOAT64}) {
            try (Holder source = new Holder(batchNorm(first, 2), conv(first, 30L), conv(first, 31L));
                 Holder target = new Holder(batchNorm(first, 2), conv(first, 32L), conv(first, 33L))) {
                float[] original = source.getParameter("slot.weight").tensor().toFloatArray();
                source.to(dtype);
                target.to(dtype);

                Map<String, Object> state = source.stateDict();
                assertArrayEquals(dtype.toString(), original, (float[]) state.get("slot.weight"),
                        dtype == DType.FLOAT64 ? 0f : 1e-2f);
                target.loadStateDict(state, true);

                assertEquals(dtype, target.getParameter("slot.weight").dtype());
                assertEquals(dtype, target.namedBuffers().get("head.running_var").dtype());
                Map<String, Object> loaded = target.stateDict();
                for (Map.Entry<String, Object> entry : state.entrySet()) {
                    assertArrayEquals(dtype + " " + entry.getKey(), (float[]) entry.getValue(),
                            (float[]) loaded.get(entry.getKey()), 0f);
                }
            }
        }
    }

    @Test
    public void integerBuffersRoundTripThroughTheStateDictAndKeepTheirDType() {
        try (MixedState module = new MixedState(first)) {
            Map<String, Object> state = module.stateDict();
            assertArrayEquals(new float[]{3f, -7f}, (float[]) state.get("ids"), 0f);
            assertArrayEquals(new float[]{5f, (float) (1L << 40)}, (float[]) state.get("steps"), 0f);
            assertArrayEquals(new float[]{-5f, 4f}, (float[]) state.get("codes"), 0f);

            state.put("ids", new float[]{11f, -2.75f});
            state.put("steps", new float[]{6f, 7.5f});
            state.put("codes", new float[]{-5.9f, 100f});
            module.loadStateDict(state, true);

            assertEquals(DType.INT32, module.buffer("ids").dtype());
            assertArrayEquals(new int[]{11, -2}, module.buffer("ids").toIntArray());
            assertEquals(DType.INT64, module.buffer("steps").dtype());
            assertArrayEquals(new int[]{6, 7}, module.buffer("steps").toIntArray());
            assertEquals(DType.INT8, module.buffer("codes").dtype());
            ByteBuffer codes = module.buffer("codes").toByteBuffer();
            assertEquals(-5, codes.get(0));
            assertEquals(100, codes.get(1));

            module.to(DType.FLOAT16);
            assertEquals(DType.FLOAT16, module.getParameter("scale").dtype());
            assertEquals(DType.INT32, module.buffer("ids").dtype());
            assertEquals(DType.INT64, module.buffer("steps").dtype());
            assertEquals(DType.INT8, module.buffer("codes").dtype());
        }
    }

    @Test
    public void aDTypeKeptBufferStaysExactThroughConversions() {
        try (MixedState module = new MixedState(first)) {
            Map<String, Object> state = new HashMap<>();
            state.put("count", new float[]{4097f});
            module.loadStateDict(state);

            module.to(DType.FLOAT16);
            assertEquals(DType.FLOAT32, module.buffer("count").dtype());
            module.to(DType.BFLOAT16);
            module.to(DType.FLOAT32);

            assertEquals(DType.FLOAT32, module.getParameter("scale").dtype());
            assertArrayEquals(new float[]{4097f}, module.buffer("count").toFloatArray(), 0f);
        }
    }

    @Test
    public void batchNormCounterKeepsItsDTypeAndCountThroughConversions() {
        try (BatchNorm1d bn = batchNorm(first, 2)) {
            Map<String, Object> state = new HashMap<>();
            state.put("num_batches_tracked", new float[]{4097f});
            bn.loadStateDict(state);

            bn.to(DType.FLOAT16);
            assertEquals(DType.FLOAT32, bn.namedBuffers().get("num_batches_tracked").dtype());
            bn.to(DType.BFLOAT16);
            bn.to(DType.FLOAT32);

            assertEquals(4097, bn.numBatchesTrained());
        }
    }

    @Test
    public void toConvertsBetweenReducedPrecisionDTypesThroughFloat32() {
        try (BatchNorm1d bn = batchNorm(first, 2)) {
            Parameter weight = bn.weight();
            bn.to(DType.FLOAT16);
            bn.to(DType.BFLOAT16);

            assertSame(weight, bn.weight());
            assertEquals(DType.BFLOAT16, weight.dtype());
            assertTrue(weight.tensor().requiresGrad());
            assertNull(weight.tensor().gradFn());
            assertEquals(DType.BFLOAT16, bn.runningVar().dtype());

            bn.to(DType.FLOAT64);
            bn.to(DType.FLOAT16);
            bn.to(DType.FLOAT32);
            assertArrayEquals(new float[]{1f, 1f}, weight.tensor().toFloatArray(), 0f);
            assertArrayEquals(new float[]{1f, 1f}, bn.runningVar().toFloatArray(), 0f);
        }
    }

    @Test
    public void toRefusesANonFloatingDTypeAndChangesNothing() {
        try (MixedState module = new MixedState(first)) {
            Parameter scale = module.getParameter("scale");
            Tensor owned = scale.tensor();
            Tensor count = module.buffer("count");
            for (DType dtype : new DType[]{DType.INT32, DType.INT64, DType.BOOL}) {
                IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> module.to(dtype));
                assertTrue(e.getMessage(), e.getMessage().contains(dtype.toString()));
            }
            assertSame(owned, scale.tensor());
            assertEquals(DType.FLOAT32, scale.dtype());
            assertTrue(scale.tensor().requiresGrad());
            assertSame(count, module.buffer("count"));
        }
    }

    @Test
    public void freezeSurvivesMovesConversionsAndLoads() {
        try (Conv2d source = conv(first, 4L); Conv2d target = conv(first, 5L)) {
            target.setRequiresGrad(false);

            target.to(second);
            assertAllFrozen(target);

            target.to(DType.FLOAT16);
            assertAllFrozen(target);
            target.to(DType.FLOAT32);
            assertAllFrozen(target);

            target.loadStateDict(source.stateDict());
            assertAllFrozen(target);
            assertArrayEquals(source.getParameter("weight").tensor().toFloatArray(),
                    target.getParameter("weight").tensor().toFloatArray(), 0f);

            try (Tensor values = Tensor.fromArray(second, source.getParameter("bias").tensor().toFloatArray())) {
                target.getParameter("bias").copyFrom(values);
            }
            assertAllFrozen(target);

            target.setRequiresGrad(true);
            for (Parameter param : target.parameters()) {
                assertTrue(param.requiresGrad());
                assertTrue(param.tensor().requiresGrad());
                assertNull(param.tensor().gradFn());
            }
        }
    }

    @Test
    public void loadStateDictInGradModeKeepsLeaves() {
        try (Conv2d source = conv(first, 6L); Conv2d target = conv(first, 7L)) {
            assertTrue(ComputationGraph.isGradEnabled());
            target.to(DType.FLOAT16);
            target.loadStateDict(source.stateDict());
            for (Parameter param : target.parameters()) {
                assertEquals(DType.FLOAT16, param.dtype());
                assertTrue(param.tensor().requiresGrad());
                assertNull(param.tensor().gradFn());
            }
            assertTrue(ComputationGraph.isGradEnabled());
        }
    }

    @Test
    public void closeReleasesBuffers() {
        BatchNorm1d bn = new BatchNorm1d(2);
        bn.initialize(first);
        Tensor mean = bn.runningMean();
        bn.close();
        assertThrows(IllegalStateException.class, mean::toFloatArray);
    }

    @Test
    public void conversionsAreRefusedWhileSubstitutedAndChangeNothing() {
        try (Conv2d conv = conv(first, 8L); Tensor bias = Tensor.fromArray(first, new float[3])) {
            Parameter weight = conv.getParameter("weight");
            Tensor ownedWeight = weight.tensor();
            Map<Parameter, Tensor> values = new HashMap<>();
            values.put(conv.getParameter("bias"), bias);
            try (Parameter.Substitution scope = conv.substitute(values)) {
                assertThrows(IllegalStateException.class, () -> conv.to(second));
                assertThrows(IllegalStateException.class, () -> conv.to(DType.FLOAT16));
                assertSame(ownedWeight, weight.tensor());
                assertSame(first, weight.backend());
                assertEquals(DType.FLOAT32, weight.dtype());
            }
            conv.to(second);
            assertSame(second, weight.backend());
            assertSame(second, conv.getParameter("bias").backend());
        }
    }

    @Test
    public void replaceModuleKeepsTheSlotAndHandsTheReplacedChildBack() {
        Conv2d replaced = conv(first, 10L);
        try {
            replaceInHolder(replaced);
            assertEquals("the replaced child is not closed with its former parent", 2,
                    replaced.parameters().size());
        } finally {
            replaced.close();
        }
    }

    private void replaceInHolder(Conv2d replaced) {
        try (Holder holder = new Holder(conv(first, 11L), replaced, conv(first, 12L));
             Conv2d replacement = conv(first, 13L)) {
            assertSame(replaced, holder.replaceModule("slot", replacement));
            assertEquals(Arrays.asList("head.weight", "head.bias", "slot.weight", "slot.bias", "tail.weight",
                    "tail.bias"), new ArrayList<>(holder.namedParameters().keySet()));
            assertEquals(new ArrayList<>(holder.namedParameters().keySet()),
                    new ArrayList<>(holder.stateDict().keySet()));
            assertSame(replacement.getParameter("weight"), holder.parameters().get(2));
            assertFalse(containsIdentity(holder.parameters(), replaced.getParameter("weight")));

            holder.eval();
            assertFalse(replacement.isTraining());
            assertTrue("the replaced child is no longer reached", replaced.isTraining());
            holder.to(second);
            assertSame(second, replacement.getParameter("weight").backend());
            assertSame(first, replaced.getParameter("weight").backend());

            Map<String, Object> state = new HashMap<>();
            float[] bias = {1f, 2f, 3f};
            state.put("slot.bias", bias);
            holder.loadStateDict(state);
            assertArrayEquals(bias, replacement.getParameter("bias").tensor().toFloatArray(), 0f);

            assertSame(replacement, holder.replaceModule("slot", null));
            assertNull(holder.getParameter("slot.weight"));
            assertNull(holder.replaceModule("extra", replacement));
            assertEquals(Arrays.asList("head", "tail", "extra"), new ArrayList<>(holder.namedChildren().keySet()));
            assertSame(replacement, holder.replaceModule("extra", null));
        }
    }

    @Test
    public void replaceModuleRefusesACycle() {
        try (Holder holder = new Holder(conv(first, 14L), conv(first, 15L), conv(first, 16L))) {
            assertThrows(IllegalArgumentException.class, () -> holder.replaceModule("self", holder));
            Holder outer = new Holder(holder, conv(first, 17L), conv(first, 18L));
            try {
                assertThrows(IllegalArgumentException.class, () -> holder.replaceModule("outer", outer));
            } finally {
                outer.replaceModule("head", null);
                outer.close();
            }
        }
    }

    private static boolean containsIdentity(List<Parameter> params, Parameter p) {
        for (Parameter candidate : params) {
            if (candidate == p) {
                return true;
            }
        }
        return false;
    }

    /** Three children in registration order: head, slot, tail. */
    private static final class Holder extends Module {
        Holder(Module head, Module slot, Module tail) {
            registerModule("head", head);
            registerModule("slot", slot);
            registerModule("tail", tail);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /**
     * One parameter {@code scale}, the persistent buffer {@code state} and the non-persistent buffer {@code cache}
     * (holding {@link #CACHE}).
     */
    private static final class StatefulCache extends Module {
        static final float[] CACHE = {0.5f, 1.5f, 2.5f};

        StatefulCache(ComputeBackend backend) {
            registerParameter("scale", new Parameter(Tensor.fromArray(backend, new float[]{2f}), "scale"));
            registerBuffer("state", Tensor.zeros(backend, Shape.vector(2)));
            registerBuffer("cache", Tensor.fromArray(backend, CACHE.clone()), false);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /**
     * One parameter {@code scale}, the dtype-kept FLOAT32 scalar buffer {@code count} and the integer buffers
     * {@code ids} (INT32 {@code [3, -7]}), {@code steps} (INT64 {@code [5, 2^40]}) and {@code codes} (INT8
     * {@code [-5, 4]}).
     */
    private static final class MixedState extends Module {
        MixedState(ComputeBackend backend) {
            registerParameter("scale", new Parameter(Tensor.fromArray(backend, new float[]{2f}), "scale"));
            registerBuffer("count", Tensor.zeros(backend, Shape.scalar()), true, true);
            registerBuffer("ids", Tensor.fromArray(backend, new int[]{3, -7}));
            registerBuffer("steps", int64Vector(backend, 5L, 1L << 40));
            registerBuffer("codes", Tensor.fromArray(backend, new byte[]{-5, 4}));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    private static Tensor int64Vector(ComputeBackend backend, long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (int i = 0; i < values.length; i++) {
            bytes.putLong(i * Long.BYTES, values[i]);
        }
        return Tensor.fromStorage(backend, Shape.vector(values.length), DType.INT64,
                TensorStorage.fromHost(backend, bytes, DType.INT64));
    }

    private static BatchNorm1d batchNorm(ComputeBackend backend, int features) {
        BatchNorm1d bn = new BatchNorm1d(features);
        bn.initialize(backend);
        return bn;
    }

    private static void assertAllFrozen(Module module) {
        for (Parameter param : module.parameters()) {
            assertFalse(param.requiresGrad());
            assertFalse(param.tensor().requiresGrad());
            assertNull(param.tensor().gradFn());
            assertNull(param.grad());
        }
    }

    private static Conv2d conv(ComputeBackend backend, long seed) {
        Conv2d conv = new Conv2d(2, 3, new int[]{2, 2});
        conv.initialize(backend, new Random(seed));
        return conv;
    }

    private static Tensor input(ComputeBackend backend) {
        float[] data = new float[2 * 3 * 3];
        for (int i = 0; i < data.length; i++) {
            data[i] = (i % 7) * 0.25f - 0.5f;
        }
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(Shape.of(1, 2, 3, 3));
        }
    }

    private static Tensor matrix(ComputeBackend backend, float[] data, int rows, int cols) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(Shape.matrix(rows, cols));
        }
    }
}
