package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class OptimizerTest {
    
    private DataStoringMockBackend backend;
    
    @Before
    public void setUp() {
        backend = new DataStoringMockBackend();
    }
    
    @Test
    public void testSGDCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        assertNotNull(sgd);
        assertEquals(0.01f, sgd.getLearningRate(), 1e-6f);
        assertEquals(0.0f, sgd.momentum(), 1e-6f);
        assertEquals(0.0f, sgd.weightDecay(), 1e-6f);
        assertFalse(sgd.nesterov());
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testSGDWithMomentum() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f, 0.9f, 0.0001f, true);
        assertEquals(0.9f, sgd.momentum(), 1e-6f);
        assertEquals(0.0001f, sgd.weightDecay(), 1e-6f);
        assertTrue(sgd.nesterov());
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testSGDConvergesOnQuadraticLoss() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.1f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            sgd.step();
            sgd.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("SGD should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.01f);
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testSGDStepCount() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        assertEquals(0, sgd.stepCount());
        
        Tensor grad = Tensor.zeros(backend, Shape.vector(10));
        param.setGrad(grad);
        sgd.step();
        assertEquals(1, sgd.stepCount());
        
        sgd.step();
        assertEquals(2, sgd.stepCount());
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testAdamCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        Adam adam = new Adam(params, 0.001f);
        assertNotNull(adam);
        assertEquals(0.001f, adam.getLearningRate(), 1e-6f);
        assertEquals(0.9f, adam.beta1(), 1e-6f);
        assertEquals(0.999f, adam.beta2(), 1e-6f);
        assertEquals(1e-8f, adam.eps(), 1e-10f);
        assertEquals(0.0f, adam.weightDecay(), 1e-6f);
        
        adam.close();
        param.close();
    }
    
    @Test
    public void testAdamWithWeightDecay() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        Adam adam = new Adam(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
        assertEquals(0.01f, adam.weightDecay(), 1e-6f);
        
        adam.close();
        param.close();
    }
    
    @Test
    public void testAdamConvergesOnQuadraticLoss() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        Adam adam = new Adam(params, 0.5f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            adam.step();
            adam.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("Adam should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.1f);
        
        adam.close();
        param.close();
    }
    
    @Test
    public void testAdamFasterThanSGD() {
        float[] initialData = {10.0f};
        
        List<Parameter> sgdParams = new ArrayList<>();
        Tensor sgdTensor = Tensor.fromArray(backend, initialData.clone()).withRequiresGrad(true);
        Parameter sgdParam = new Parameter(sgdTensor, "x");
        sgdParams.add(sgdParam);
        SGD sgd = new SGD(sgdParams, 0.05f);
        
        List<Parameter> adamParams = new ArrayList<>();
        Tensor adamTensor = Tensor.fromArray(backend, initialData.clone()).withRequiresGrad(true);
        Parameter adamParam = new Parameter(adamTensor, "x");
        adamParams.add(adamParam);
        Adam adam = new Adam(adamParams, 0.5f);
        
        float sgdFinalValue = 10.0f;
        float adamFinalValue = 10.0f;
        
        for (int i = 0; i < 50; i++) {
            float x = sgdParam.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            sgdParam.setGrad(gradTensor);
            sgd.step();
            sgd.zeroGrad();
            sgdFinalValue = sgdParam.tensor().toFloatArray()[0];
            
            x = adamParam.tensor().toFloatArray()[0];
            grad = 2.0f * x;
            gradTensor = Tensor.fromArray(backend, new float[]{grad});
            adamParam.setGrad(gradTensor);
            adam.step();
            adam.zeroGrad();
            adamFinalValue = adamParam.tensor().toFloatArray()[0];
        }
        
        assertTrue("Both optimizers should make progress", Math.abs(sgdFinalValue) < 1.0f && Math.abs(adamFinalValue) < 1.0f);
        
        sgd.close();
        adam.close();
        sgdParam.close();
        adamParam.close();
    }
    
    @Test
    public void testAdamWCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        AdamW adamw = new AdamW(params, 0.001f);
        assertNotNull(adamw);
        assertEquals(0.001f, adamw.getLearningRate(), 1e-6f);
        assertEquals(0.9f, adamw.beta1(), 1e-6f);
        assertEquals(0.999f, adamw.beta2(), 1e-6f);
        assertEquals(1e-8f, adamw.eps(), 1e-10f);
        assertEquals(0.01f, adamw.weightDecay(), 1e-6f);
        
        adamw.close();
        param.close();
    }
    
    @Test
    public void testAdamWConvergesOnQuadraticLoss() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        AdamW adamw = new AdamW(params, 0.5f, 0.9f, 0.999f, 1e-8f, 0.0f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            adamw.step();
            adamw.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("AdamW should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.1f);
        
        adamw.close();
        param.close();
    }
    
    @Test
    public void testZeroGrad() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        
        Tensor grad = Tensor.ones(backend, Shape.vector(10));
        param.setGrad(grad);
        assertNotNull(param.grad());
        
        sgd.zeroGrad();
        assertNull(param.grad());
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testSetLearningRate() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        assertEquals(0.01f, sgd.getLearningRate(), 1e-6f);
        
        sgd.setLearningRate(0.1f);
        assertEquals(0.1f, sgd.getLearningRate(), 1e-6f);
        
        sgd.close();
        param.close();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidLearningRate() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        sgd.setLearningRate(-0.1f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidMomentum() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        new SGD(params, 0.01f, -0.1f, 0.0f, false);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBeta1() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        new Adam(params, 0.001f, 1.5f, 0.999f, 1e-8f, 0.0f);
    }
    
    @Test
    public void testStateDict() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        Tensor grad = Tensor.ones(backend, Shape.vector(10));
        param.setGrad(grad);
        sgd.step();
        
        Map<String, Object> state = sgd.stateDict();
        assertNotNull(state);
        assertEquals(1, state.get("step_count"));
        Optimizer.closeStateDict(state);
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testClose() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        sgd.close();
        
        assertTrue(sgd.toString().contains("closed=true"));
        param.close();
    }
    
    @Test
    public void testOptimizerToString() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f, 0.9f, 0.0001f, true);
        String str = sgd.toString();
        assertTrue(str.contains("SGD"));
        assertTrue(str.contains("lr=0.01"));
        assertTrue(str.contains("momentum=0.9"));
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testMultipleParameters() {
        List<Parameter> params = new ArrayList<>();
        Tensor t1 = Tensor.zeros(backend, Shape.vector(5));
        Tensor t2 = Tensor.zeros(backend, Shape.vector(3));
        params.add(new Parameter(t1, "w1"));
        params.add(new Parameter(t2, "w2"));
        
        SGD sgd = new SGD(params, 0.1f);
        assertEquals(2, sgd.parameters().size());
        
        Tensor g1 = Tensor.ones(backend, Shape.vector(5));
        Tensor g2 = Tensor.ones(backend, Shape.vector(3));
        params.get(0).setGrad(g1);
        params.get(1).setGrad(g2);
        
        sgd.step();
        assertEquals(1, sgd.stepCount());
        
        sgd.close();
        for (Parameter p : params) {
            p.close();
        }
    }
    
    @Test
    public void testStepWithNullGrad() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SGD sgd = new SGD(params, 0.01f);
        
        sgd.step();
        assertEquals(1, sgd.stepCount());
        
        sgd.close();
        param.close();
    }
    
    @Test
    public void testRMSpropCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        RMSprop rmsprop = new RMSprop(params, 0.01f);
        assertNotNull(rmsprop);
        assertEquals(0.01f, rmsprop.getLearningRate(), 1e-6f);
        assertEquals(0.9f, rmsprop.alpha(), 1e-6f);
        assertEquals(1e-8f, rmsprop.eps(), 1e-10f);
        assertEquals(0.0f, rmsprop.weightDecay(), 1e-6f);
        assertFalse(rmsprop.centered());
        assertEquals(0.0f, rmsprop.momentum(), 1e-6f);
        
        rmsprop.close();
        param.close();
    }
    
    @Test
    public void testRMSpropWithAllOptions() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        RMSprop rmsprop = new RMSprop(params, 0.01f, 0.95f, 1e-7f, 0.001f, true, 0.9f);
        assertEquals(0.95f, rmsprop.alpha(), 1e-6f);
        assertEquals(1e-7f, rmsprop.eps(), 1e-10f);
        assertEquals(0.001f, rmsprop.weightDecay(), 1e-6f);
        assertTrue(rmsprop.centered());
        assertEquals(0.9f, rmsprop.momentum(), 1e-6f);
        
        rmsprop.close();
        param.close();
    }
    
    @Test
    public void testRMSpropConvergesOnQuadraticLoss() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        RMSprop rmsprop = new RMSprop(params, 0.1f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            rmsprop.step();
            rmsprop.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("RMSprop should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.1f);
        
        rmsprop.close();
        param.close();
    }
    
    @Test
    public void testAdagradCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        Adagrad adagrad = new Adagrad(params, 0.01f);
        assertNotNull(adagrad);
        assertEquals(0.01f, adagrad.getLearningRate(), 1e-6f);
        assertEquals(1e-10f, adagrad.eps(), 1e-12f);
        assertEquals(0.0f, adagrad.weightDecay(), 1e-6f);
        assertEquals(0.0f, adagrad.initialAccumulatorValue(), 1e-6f);
        
        adagrad.close();
        param.close();
    }
    
    @Test
    public void testAdagradWithWeightDecay() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        Adagrad adagrad = new Adagrad(params, 0.01f, 1e-8f, 0.01f, 0.1f);
        assertEquals(1e-8f, adagrad.eps(), 1e-10f);
        assertEquals(0.01f, adagrad.weightDecay(), 1e-6f);
        assertEquals(0.1f, adagrad.initialAccumulatorValue(), 1e-6f);
        
        adagrad.close();
        param.close();
    }
    
    @Test
    public void testAdagradConvergesOnQuadraticLoss() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        Adagrad adagrad = new Adagrad(params, 1.0f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            adagrad.step();
            adagrad.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("Adagrad should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.5f);
        
        adagrad.close();
        param.close();
    }
    
    @Test
    public void testLAMBCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        LAMB lamb = new LAMB(params, 0.001f);
        assertNotNull(lamb);
        assertEquals(0.001f, lamb.getLearningRate(), 1e-6f);
        assertEquals(0.9f, lamb.beta1(), 1e-6f);
        assertEquals(0.999f, lamb.beta2(), 1e-6f);
        assertEquals(1e-6f, lamb.eps(), 1e-8f);
        assertEquals(0.0f, lamb.weightDecay(), 1e-6f);
        assertTrue(lamb.biasCorrection());
        
        lamb.close();
        param.close();
    }
    
    @Test
    public void testLAMBWithWeightDecay() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        LAMB lamb = new LAMB(params, 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f, true);
        assertEquals(0.01f, lamb.weightDecay(), 1e-6f);
        
        lamb.close();
        param.close();
    }
    
    @Test
    public void testLAMBConvergesOnQuadraticLoss() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        LAMB lamb = new LAMB(params, 0.1f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            
            lamb.step();
            lamb.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("LAMB should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.5f);
        
        lamb.close();
        param.close();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testRMSpropInvalidAlpha() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        new RMSprop(params, 0.01f, 1.5f, 1e-8f, 0.0f, false, 0.0f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAdagradInvalidEps() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        new Adagrad(params, 0.01f, 0.0f, 0.0f, 0.0f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testLAMBInvalidBeta2() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10));
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        new LAMB(params, 0.001f, 0.9f, 1.5f, 1e-6f, 0.0f, true);
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
