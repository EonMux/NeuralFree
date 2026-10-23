package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.ModuleList;
import com.eonmux.neuralfree.nn.ModuleDict;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class ModuleOptimizerConvergenceTest {
    
    private DataStoringMockBackend backend;
    
    @Before
    public void setUp() {
        backend = new DataStoringMockBackend();
    }
    
    static class NestedModuleWithParams extends Module {
        private Parameter weight;
        
        NestedModuleWithParams(ComputeBackend backend, int inFeatures, int outFeatures, Module child) {
            this.weight = new Parameter(Tensor.zeros(backend, Shape.matrix(outFeatures, inFeatures)), "weight");
            this.registerParameter("weight", this.weight);
            if (child != null) {
                this.registerModule("child", child);
            }
        }
        
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }
    
    static class LeafModuleWithParams extends Module {
        private Parameter weight;
        private Parameter bias;
        
        LeafModuleWithParams(ComputeBackend backend, int features) {
            this.weight = new Parameter(Tensor.zeros(backend, Shape.vector(features)), "weight");
            this.bias = new Parameter(Tensor.zeros(backend, Shape.vector(features)), "bias");
            this.registerParameter("weight", this.weight);
            this.registerParameter("bias", this.bias);
        }
        
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }
    
    @Test
    public void testRecursiveParameterCollection() {
        LeafModuleWithParams leaf = new LeafModuleWithParams(backend, 10);
        NestedModuleWithParams middle = new NestedModuleWithParams(backend, 20, 10, leaf);
        NestedModuleWithParams root = new NestedModuleWithParams(backend, 30, 20, middle);
        
        List<Parameter> params = root.parameters();
        
        assertEquals("Should collect all 4 parameters recursively (root.weight + middle.weight + leaf.weight + leaf.bias)", 4, params.size());
        
        Set<String> paramNames = new HashSet<>();
        Map<String, Parameter> namedParams = root.namedParameters();
        for (Map.Entry<String, Parameter> entry : namedParams.entrySet()) {
            paramNames.add(entry.getKey());
        }
        
        assertTrue(namedParams.containsKey("weight"));
        assertTrue(namedParams.containsKey("child.weight"));
        assertTrue(namedParams.containsKey("child.child.weight"));
        assertTrue(namedParams.containsKey("child.child.bias"));
        assertEquals(4, namedParams.size());
        
        root.close();
    }
    
    @Test
    public void testDeeplyNestedParameters() {
        LeafModuleWithParams leaf = new LeafModuleWithParams(backend, 5);
        NestedModuleWithParams level1 = new NestedModuleWithParams(backend, 10, 5, leaf);
        NestedModuleWithParams level2 = new NestedModuleWithParams(backend, 15, 10, level1);
        NestedModuleWithParams level3 = new NestedModuleWithParams(backend, 20, 15, level2);
        NestedModuleWithParams root = new NestedModuleWithParams(backend, 25, 20, level3);
        
        List<Parameter> params = root.parameters();
        assertEquals("Deeply nested should collect all 6 parameters (5 nested weights + 1 leaf bias)", 6, params.size());
        
        Map<String, Parameter> namedParams = root.namedParameters();
        assertTrue(namedParams.containsKey("weight"));
        assertTrue(namedParams.containsKey("child.weight"));
        assertTrue(namedParams.containsKey("child.child.weight"));
        assertTrue(namedParams.containsKey("child.child.child.weight"));
        assertTrue(namedParams.containsKey("child.child.child.child.weight"));
        assertTrue(namedParams.containsKey("child.child.child.child.bias"));
        
        root.close();
    }
    
    @Test
    public void testStateDictRoundTrip() {
        LeafModuleWithParams module = new LeafModuleWithParams(backend, 10);
        
        float[] weightData = new float[10];
        float[] biasData = new float[10];
        for (int i = 0; i < 10; i++) {
            weightData[i] = i * 0.1f;
            biasData[i] = i * 0.01f;
        }
        
        Map<String, Object> customState = new LinkedHashMap<>();
        customState.put("weight", weightData);
        customState.put("bias", biasData);
        
        module.loadStateDict(customState);
        
        Map<String, Object> savedState = module.stateDict();
        assertNotNull(savedState);
        assertTrue(savedState.containsKey("weight"));
        assertTrue(savedState.containsKey("bias"));
        
        LeafModuleWithParams module2 = new LeafModuleWithParams(backend, 10);
        module2.loadStateDict(savedState);
        
        float[] loadedWeight = module2.parameters().get(0).tensor().toFloatArray();
        float[] loadedBias = module2.parameters().get(1).tensor().toFloatArray();
        
        assertArrayEquals("Weight should match after round-trip", weightData, loadedWeight, 1e-6f);
        assertArrayEquals("Bias should match after round-trip", biasData, loadedBias, 1e-6f);
        
        module.close();
        module2.close();
    }
    
    @Test
    public void testNestedStateDictRoundTrip() {
        LeafModuleWithParams leaf = new LeafModuleWithParams(backend, 5);
        NestedModuleWithParams root = new NestedModuleWithParams(backend, 10, 5, leaf);
        
        Map<String, Parameter> namedParams = root.namedParameters();
        assertEquals(3, namedParams.size());
        
        Map<String, Object> stateDict = root.stateDict();
        assertNotNull(stateDict);
        assertEquals(3, stateDict.size());
        assertTrue("State dict should contain weight key", stateDict.containsKey("weight"));
        assertTrue("State dict should contain child.weight key", stateDict.containsKey("child.weight"));
        assertTrue("State dict should contain child.bias key", stateDict.containsKey("child.bias"));
        
        root.close();
    }
    
    @Test
    public void testTrainEvalModePropagationDeepNested() {
        LeafModuleWithParams leaf = new LeafModuleWithParams(backend, 5);
        NestedModuleWithParams level1 = new NestedModuleWithParams(backend, 10, 5, leaf);
        NestedModuleWithParams level2 = new NestedModuleWithParams(backend, 15, 10, level1);
        NestedModuleWithParams root = new NestedModuleWithParams(backend, 20, 15, level2);
        
        assertTrue("Root should be in training mode", root.isTraining());
        assertTrue("Deep leaf should be in training mode", leaf.isTraining());
        
        root.eval();
        assertFalse("Root should be in eval mode", root.isTraining());
        assertFalse("Level 1 should be in eval mode", level1.isTraining());
        assertFalse("Level 2 should be in eval mode", level2.isTraining());
        assertFalse("Deep leaf should be in eval mode", leaf.isTraining());
        
        root.train();
        assertTrue("Root should be back in training mode", root.isTraining());
        assertTrue("Deep leaf should be back in training mode", leaf.isTraining());
        
        root.close();
    }
    
    @Test
    public void testSGDConvergesBelowThreshold1000Steps() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {10.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.1f);
        
        for (int i = 0; i < 1000; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            sgd.step();
            sgd.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        float finalLoss = finalX * finalX;
        assertTrue("SGD should minimize x^2 below 1e-4 after 1000 steps, got loss: " + finalLoss + ", x: " + finalX, 
                   finalLoss < 1e-4f);
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testAdamConvergesBelowThreshold500Steps() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {10.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        Adam adam = new Adam(params, 0.5f);
        
        for (int i = 0; i < 500; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            adam.step();
            adam.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        float finalLoss = finalX * finalX;
        assertTrue("Adam should minimize x^2 below 1e-4 after 500 steps, got loss: " + finalLoss + ", x: " + finalX, 
                   finalLoss < 1e-4f);
        
        adam.close();
        param.close();
    }
    
    @Test
    public void testAdamWConvergesBelowThreshold500Steps() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {10.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        AdamW adamw = new AdamW(params, 0.5f, 0.9f, 0.999f, 1e-8f, 0.0f);
        
        for (int i = 0; i < 500; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            adamw.step();
            adamw.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        float finalLoss = finalX * finalX;
        assertTrue("AdamW should minimize x^2 below 1e-4 after 500 steps, got loss: " + finalLoss + ", x: " + finalX, 
                   finalLoss < 1e-4f);
        
        adamw.close();
        param.close();
    }
    
    @Test
    public void testRMSpropConvergesBelowThreshold500Steps() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {10.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        RMSprop rmsprop = new RMSprop(params, 0.1f);
        
        for (int i = 0; i < 500; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            rmsprop.step();
            rmsprop.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        float finalLoss = finalX * finalX;
        assertTrue("RMSprop should minimize x^2 below 5e-3 after 500 steps, got loss: " + finalLoss + ", x: " + finalX, 
                   finalLoss < 5e-3f);
        
        rmsprop.close();
        param.close();
    }
    
    @Test
    public void testCosineAnnealingLRMatchesAnalyticalFormula() {
        Optimizer optimizer = createMockOptimizer(1.0f);
        int tMax = 100;
        float etaMin = 0.0f;
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, tMax, etaMin);
        
        for (int epoch = 0; epoch <= tMax; epoch++) {
            float actualLR = scheduler.getLR();
            float progress = (float) epoch / tMax;
            float expectedLR = etaMin + (1.0f - etaMin) * (1.0f + (float) Math.cos(Math.PI * progress)) / 2.0f;
            
            assertEquals("LR at epoch " + epoch + " should match analytical formula", 
                         expectedLR, actualLR, 1e-6f);
            
            if (epoch < tMax) {
                scheduler.stepEpoch();
            }
        }
    }
    
    @Test
    public void testCosineAnnealingLRKeyPoints() {
        Optimizer optimizer = createMockOptimizer(1.0f);
        int tMax = 100;
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, tMax, 0.0f);
        
        assertEquals("At epoch 0, LR should be initial", 1.0f, scheduler.getLR(), 1e-6f);
        
        for (int i = 0; i < tMax / 2; i++) {
            scheduler.stepEpoch();
        }
        float midLR = scheduler.getLR();
        assertEquals("At epoch T_max/2, LR should be ~0.5 (cos(pi/2)=0)", 0.5f, midLR, 0.01f);
        
        for (int i = tMax / 2; i < tMax; i++) {
            scheduler.stepEpoch();
        }
        float endLR = scheduler.getLR();
        assertEquals("At epoch T_max, LR should be eta_min", 0.0f, endLR, 1e-6f);
    }
    
    @Test
    public void testCosineAnnealingLRWithNonZeroEtaMin() {
        Optimizer optimizer = createMockOptimizer(1.0f);
        float etaMin = 0.01f;
        int tMax = 50;
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, tMax, etaMin);
        
        assertEquals("At epoch 0, LR should be initial", 1.0f, scheduler.getLR(), 1e-6f);
        
        for (int i = 0; i < tMax; i++) {
            scheduler.stepEpoch();
        }
        
        float endLR = scheduler.getLR();
        assertEquals("At epoch T_max, LR should be eta_min", etaMin, endLR, 1e-6f);
    }
    
    @Test
    public void testOptimizerWithModuleParameters() {
        NestedModuleWithParams module = new NestedModuleWithParams(backend, 10, 5, 
            new LeafModuleWithParams(backend, 5));
        
        List<Parameter> params = module.parameters();
        assertEquals("Should have 3 params (nested weight + leaf weight + leaf bias)", 3, params.size());
        
        SGD sgd = new SGD(params, 0.01f);
        
        for (Parameter p : params) {
            Tensor grad = Tensor.ones(backend, p.shape());
            p.setGrad(grad);
        }
        
        sgd.step();
        
        assertEquals("Step count should be 1", 1, sgd.stepCount());
        
        sgd.zeroGrad();
        for (Parameter p : params) {
            assertNull("Gradient should be null after zeroGrad", p.grad());
        }
        
        sgd.close();
        module.close();
    }
    
    @Test
    public void testModuleZeroGradRecursive() {
        LeafModuleWithParams leaf = new LeafModuleWithParams(backend, 5);
        NestedModuleWithParams root = new NestedModuleWithParams(backend, 10, 5, leaf);
        
        for (Parameter p : root.parameters()) {
            Tensor grad = Tensor.ones(backend, p.shape());
            p.setGrad(grad);
            assertNotNull(p.grad());
        }
        
        root.zeroGrad();
        
        for (Parameter p : root.parameters()) {
            assertNull("All gradients should be cleared by zeroGrad", p.grad());
        }
        
        root.close();
    }
    
    @Test
    public void testConvergenceOnMultiDimQuadratic() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f, 3.0f, -4.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        Adam adam = new Adam(params, 0.5f);
        
        for (int i = 0; i < 500; i++) {
            float[] x = param.tensor().toFloatArray();
            float[] grad = new float[x.length];
            for (int j = 0; j < x.length; j++) {
                grad[j] = 2.0f * x[j];
            }
            
            Tensor gradTensor = Tensor.fromArray(backend, grad);
            param.setGrad(gradTensor);
            
            adam.step();
            adam.zeroGrad();
        }
        
        float[] finalX = param.tensor().toFloatArray();
        float finalLoss = 0.0f;
        for (float v : finalX) {
            finalLoss += v * v;
        }
        
        assertTrue("Adam should minimize ||x||^2 below 1e-4 for multi-dim, got: " + finalLoss, 
                   finalLoss < 1e-4f);
        
        adam.close();
        param.close();
    }
    
    @Test
    public void testSequentialParameterCollection() {
        Sequential seq = new Sequential();
        seq.add(new LeafModuleWithParams(backend, 10));
        seq.add(new LeafModuleWithParams(backend, 10));
        seq.add(new LeafModuleWithParams(backend, 10));
        
        List<Parameter> params = seq.parameters();
        assertEquals("Sequential with 3 leaf modules should have 6 params (2 each)", 6, params.size());
        
        Map<String, Parameter> named = seq.namedParameters();
        assertTrue(named.containsKey("0.weight"));
        assertTrue(named.containsKey("0.bias"));
        assertTrue(named.containsKey("1.weight"));
        assertTrue(named.containsKey("1.bias"));
        assertTrue(named.containsKey("2.weight"));
        assertTrue(named.containsKey("2.bias"));
        
        seq.close();
    }
    
    @Test
    public void testModuleListParameterCollection() {
        ModuleList list = new ModuleList();
        list.add(new LeafModuleWithParams(backend, 5));
        list.add(new LeafModuleWithParams(backend, 5));
        
        List<Parameter> params = list.parameters();
        assertEquals("ModuleList with 2 leaf modules should have 4 params", 4, params.size());
        
        list.close();
    }
    
    @Test
    public void testModuleDictParameterCollection() {
        ModuleDict dict = new ModuleDict();
        dict.put("a", new LeafModuleWithParams(backend, 5));
        dict.put("b", new LeafModuleWithParams(backend, 5));
        
        List<Parameter> params = dict.parameters();
        assertEquals("ModuleDict with 2 leaf modules should have 4 params", 4, params.size());
        
        Map<String, Parameter> named = dict.namedParameters();
        assertTrue(named.containsKey("a.weight"));
        assertTrue(named.containsKey("a.bias"));
        assertTrue(named.containsKey("b.weight"));
        assertTrue(named.containsKey("b.bias"));
        
        dict.close();
    }
    
    private Optimizer createMockOptimizer(float lr) {
        List<Parameter> params = new ArrayList<>();
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("lr", lr);
        
        return new Optimizer(params, defaults) {
            @Override
            public void step() {
                incrementStep();
            }
        };
    }
    
    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class DataStoringMockBackend implements ComputeBackend {
        private final AtomicLong bufferIdCounter = new AtomicLong(1);
        private final Map<Long, byte[]> bufferData = new HashMap<>();
        private final DataStoringMockQueue computeQueue = new DataStoringMockQueue(bufferData);
        private final DataStoringMockQueue transferQueue = new DataStoringMockQueue(bufferData);
        private volatile boolean closed = false;

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("DataStoringMockDevice")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(8)
                    .maxClockFrequencyMhz(1000)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
                    .build();
        }

        @Override
        public ComputeQueue computeQueue() {
            return computeQueue;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transferQueue;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            long id = bufferIdCounter.getAndIncrement();
            bufferData.put(id, new byte[(int) sizeInBytes]);
            return new MockGpuBuffer(id, sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return new MockKernelHandle(kernelName);
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }

        @Override
        public void close() {
            closed = true;
            bufferData.clear();
        }
    }

    private static class DataStoringMockQueue implements ComputeQueue {
        private final Map<Long, byte[]> bufferData;
        private volatile boolean closed = false;

        DataStoringMockQueue(Map<Long, byte[]> bufferData) {
            this.bufferData = bufferData;
        }

        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                System.arraycopy(srcData, 0, data, (int) dstOffset, (int) sizeInBytes);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                ByteBuffer buffer = ByteBuffer.wrap(data, (int) dstOffset, (int) sizeInBytes).order(ByteOrder.nativeOrder());
                for (float f : srcData) {
                    buffer.putFloat(f);
                }
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                System.arraycopy(data, (int) srcOffset, dstData, 0, (int) sizeInBytes);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                ByteBuffer buffer = ByteBuffer.wrap(data, (int) srcOffset, (int) sizeInBytes).order(ByteOrder.nativeOrder());
                for (int i = 0; i < dstData.length; i++) {
                    dstData[i] = buffer.getFloat();
                }
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final long bufferId;
        private final long sizeInBytes;
        private volatile boolean closed = false;

        MockGpuBuffer(long bufferId, long sizeInBytes) {
            this.bufferId = bufferId;
            this.sizeInBytes = sizeInBytes;
        }

        @Override
        public long sizeInBytes() {
            return sizeInBytes;
        }

        @Override
        public long bufferId() {
            return bufferId;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() throws InterruptedException {
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;
        private volatile boolean closed = false;

        MockKernelHandle(String kernelName) {
            this.kernelName = kernelName;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
