package com.eonmux.neuralfree.plugin;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainerConfig;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import static org.junit.Assert.*;

public class PluginSystemTest {

    private PluginTestMockBackend backend;

    @Before
    public void setUp() {
        backend = new PluginTestMockBackend();
    }

    @Test
    public void testCustomModuleForward() {
        CustomSquareModule customModule = new CustomSquareModule();

        float[] inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor input = Tensor.fromArray(backend, inputData);

        Tensor output = customModule.forward(input);

        assertNotNull("Output should not be null", output);
        float[] outputData = output.toFloatArray();
        for (int i = 0; i < inputData.length; i++) {
            float expected = inputData[i] * inputData[i];
            assertEquals("Element " + i + " should be squared", expected, outputData[i], 1e-6f);
        }
    }

    @Test
    public void testCustomModuleInSequential() {
        PassthroughModule passthrough = new PassthroughModule();
        CustomSquareModule squareModule = new CustomSquareModule();
        
        Sequential sequential = new Sequential(passthrough, squareModule, passthrough);
        
        float[] inputData = {2.0f, 3.0f, 4.0f};
        Tensor input = Tensor.fromArray(backend, inputData);
        
        Tensor output = sequential.forward(input);
        
        assertNotNull("Output should not be null", output);
        float[] outputData = output.toFloatArray();
        assertEquals("First element should be squared", 4.0f, outputData[0], 1e-6f);
        assertEquals("Second element should be squared", 9.0f, outputData[1], 1e-6f);
        assertEquals("Third element should be squared", 16.0f, outputData[2], 1e-6f);
    }

    @Test
    public void testCustomOptimizerCreation() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(10)).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        CustomScheduleOptimizer optimizer = new CustomScheduleOptimizer(params, 0.1f);
        
        assertNotNull("Optimizer should be created", optimizer);
        assertEquals("Initial LR should be 0.1", 0.1f, optimizer.getLearningRate(), 1e-6f);
        assertEquals("Initial step count should be 0", 0, optimizer.stepCount());
        
        optimizer.close();
        param.close();
    }

    @Test
    public void testCustomOptimizerSchedule() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {5.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        CustomScheduleOptimizer optimizer = new CustomScheduleOptimizer(params, 0.5f);
        
        float[] lrSchedule = new float[10];
        for (int i = 0; i < 10; i++) {
            Tensor grad = Tensor.fromArray(backend, new float[]{2.0f * param.tensor().toFloatArray()[0]});
            param.setGrad(grad);
            optimizer.step();
            lrSchedule[i] = optimizer.currentLearningRate;
            optimizer.zeroGrad();
        }
        
        assertTrue("LR should decrease over time", lrSchedule[0] > lrSchedule[9]);
        assertTrue("LR should be scaled by 1/(1+step*decay)", lrSchedule[1] < lrSchedule[0]);
        
        optimizer.close();
        param.close();
    }

    @Test
    public void testCustomOptimizerConverges() {
        List<Parameter> params = new ArrayList<>();
        float[] initialData = {10.0f};
        Tensor tensor = Tensor.fromArray(backend, initialData).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "x");
        params.add(param);
        
        CustomScheduleOptimizer optimizer = new CustomScheduleOptimizer(params, 1.0f);
        
        for (int i = 0; i < 100; i++) {
            float x = param.tensor().toFloatArray()[0];
            float grad = 2.0f * x;
            Tensor gradTensor = Tensor.fromArray(backend, new float[]{grad});
            param.setGrad(gradTensor);
            optimizer.step();
            optimizer.zeroGrad();
        }
        
        float finalX = param.tensor().toFloatArray()[0];
        assertTrue("Custom optimizer should converge to near 0, got: " + finalX, Math.abs(finalX) < 0.5f);
        
        optimizer.close();
        param.close();
    }

    @Test
    public void testCustomLossFunction() {
        CustomMAELoss loss = new CustomMAELoss();
        
        float[] predData = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targetData = {1.5f, 2.0f, 2.5f, 5.0f};
        
        Tensor prediction = Tensor.fromArray(backend, predData);
        Tensor target = Tensor.fromArray(backend, targetData);
        
        Tensor lossTensor = loss.compute(prediction, target);
        
        assertNotNull("Loss tensor should not be null", lossTensor);
        
        float lossValue = lossTensor.toFloatArray()[0];
        float expectedMAE = (Math.abs(1.0f - 1.5f) + Math.abs(2.0f - 2.0f) + 
                            Math.abs(3.0f - 2.5f) + Math.abs(4.0f - 5.0f)) / 4.0f;
        assertEquals("MAE should be computed correctly", expectedMAE, lossValue, 1e-6f);
    }

    @Test
    public void testCustomLossFunctionName() {
        CustomMAELoss loss = new CustomMAELoss();
        assertEquals("Loss name should be CustomMAELoss", "CustomMAELoss", loss.name());
    }

    @Test
    public void testCustomLossInTrainer() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(2)).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "weight");
        params.add(param);
        
        SimpleLinearModule model = new SimpleLinearModule(params);
        
        CustomScheduleOptimizer optimizer = new CustomScheduleOptimizer(params, 0.1f);
        CustomMAELoss lossFn = new CustomMAELoss();
        
        Tensor[] inputs = new Tensor[4];
        Tensor[] targets = new Tensor[4];
        for (int i = 0; i < 4; i++) {
            inputs[i] = Tensor.fromArray(backend, new float[]{(float) i});
            targets[i] = Tensor.fromArray(backend, new float[]{(float) i * 2});
        }
        
        TensorDataset dataset = new TensorDataset(inputs, targets);
        DataLoader<TensorDataset.Sample> trainLoader = new DataLoader<>(dataset, 2, false);
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .config(TrainerConfig.builder().verbose(false).build())
            .build();
        
        assertNotNull("Trainer should be created", trainer);
        assertEquals("Trainer should use custom loss", "CustomMAELoss", trainer.lossFunction().name());
        assertTrue("Trainer should use custom optimizer", trainer.optimizer() instanceof CustomScheduleOptimizer);
        
        trainer.close();
    }

    @Test
    public void testAllCustomComponentsWorkTogether() {
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.ones(backend, Shape.vector(3)).withRequiresGrad(true);
        Parameter param = new Parameter(tensor, "scale");
        params.add(param);
        
        CustomModuleWithParams model = new CustomModuleWithParams(params);
        
        CustomScheduleOptimizer optimizer = new CustomScheduleOptimizer(params, 0.1f);
        CustomMAELoss lossFn = new CustomMAELoss();
        
        Tensor[] inputs = new Tensor[2];
        Tensor[] targets = new Tensor[2];
        for (int i = 0; i < 2; i++) {
            inputs[i] = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});
            targets[i] = Tensor.fromArray(backend, new float[]{0.0f, 0.0f, 0.0f});
        }
        
        TensorDataset dataset = new TensorDataset(inputs, targets);
        DataLoader<TensorDataset.Sample> trainLoader = new DataLoader<>(dataset, 2, false);
        
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(lossFn)
            .trainLoader(trainLoader)
            .config(TrainerConfig.builder().verbose(false).build())
            .build();
        
        assertNotNull("Trainer should be created with all custom components", trainer);
        assertTrue("Optimizer should be custom", trainer.optimizer() instanceof CustomScheduleOptimizer);
        assertEquals("Loss should be custom", "CustomMAELoss", trainer.lossFunction().name());
        
        trainer.close();
    }

    @Test
    public void testNoFrameworkModificationRequired() {
        CustomSquareModule customModule = new CustomSquareModule();
        
        List<Parameter> params = new ArrayList<>();
        Tensor tensor = Tensor.zeros(backend, Shape.vector(3)).withRequiresGrad(true);
        params.add(new Parameter(tensor, "w"));
        
        CustomScheduleOptimizer customOptimizer = new CustomScheduleOptimizer(params, 0.1f);
        
        CustomMAELoss customLoss = new CustomMAELoss();
        
        Sequential sequentialModel = new Sequential(customModule);
        
        assertNotNull("Custom module should work", customModule);
        assertNotNull("Custom optimizer should work", customOptimizer);
        assertNotNull("Custom loss should work", customLoss);
        assertNotNull("Custom module should work in Sequential", sequentialModel);
        
        float[] inputData = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, inputData);
        Tensor output = sequentialModel.forward(input);
        assertNotNull("Sequential with custom module should produce output", output);
        
        customOptimizer.close();
    }

    /** A user-defined module that squares its input by composing the public tensor ops. */
    private static class CustomSquareModule extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.mul(input, input);
        }
    }

    /** A user-defined module that squares its input and scales it by a trainable parameter. */
    private static class CustomModuleWithParams extends Module {
        private final Parameter scale;

        CustomModuleWithParams(List<Parameter> params) {
            this.scale = registerParameter("scale", params.get(0));
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor squared = TensorOps.mul(input, input);
            return TensorOps.mul(squared, scale.tensor());
        }
    }

    private static class PassthroughModule extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    private static class SimpleLinearModule extends Module {
        private final Parameter weight;

        SimpleLinearModule(List<Parameter> params) {
            this.weight = registerParameter("weight", params.get(0));
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.mul(input, weight.tensor());
        }
    }

    private static class CustomScheduleOptimizer extends Optimizer {
        private static final float DECAY_RATE = 0.01f;
        private float currentLearningRate;

        CustomScheduleOptimizer(List<Parameter> params, float lr) {
            super(params, createDefaults(lr));
            this.currentLearningRate = lr;
        }

        private static Map<String, Object> createDefaults(float lr) {
            Map<String, Object> defaults = new HashMap<>();
            defaults.put("lr", lr);
            return defaults;
        }

        @Override
        public void step() {
            ensureOpen();
            
            currentLearningRate = getLearningRate() / (1.0f + stepCount * DECAY_RATE);
            
            for (Parameter param : params) {
                Tensor grad = param.grad();
                if (grad == null) {
                    continue;
                }
                
                float[] paramData = param.tensor().toFloatArray();
                float[] gradData = grad.toFloatArray();
                
                for (int i = 0; i < paramData.length; i++) {
                    paramData[i] -= currentLearningRate * gradData[i];
                }
                
                updateParameter(param, paramData);
            }
            
            incrementStep();
        }
        
        private void updateParameter(Parameter param, float[] data) {
            if (params.isEmpty()) return;
            ComputeBackend backend = params.get(0).backend();
            if (backend == null) return;
            
            ComputeQueue queue = backend.transferQueue();
            com.eonmux.neuralfree.tensor.TensorStorage storage = param.tensor().storage();
            if (storage != null && storage.gpuBuffer() != null) {
                queue.enqueueWriteBuffer(storage.gpuBuffer(), 0, data.length * 4L, data);
                queue.finish();
            }
        }
    }

    private static class CustomMAELoss implements LossFunction {

        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            Tensor diff = TensorOps.sub(prediction, target);
            
            float[] diffData = diff.toFloatArray();
            for (int i = 0; i < diffData.length; i++) {
                diffData[i] = Math.abs(diffData[i]);
            }
            
            Tensor absDiff = Tensor.fromArray(diff.backend(), diffData).reshape(diff.shape());
            
            float sum = 0.0f;
            for (float v : diffData) {
                sum += v;
            }
            float mean = sum / diffData.length;
            
            return Tensor.scalar(diff.backend(), mean);
        }

        @Override
        public String name() {
            return "CustomMAELoss";
        }

        @Override
        public Reduction reduction() {
            return Reduction.MEAN;
        }
    }

    private static class PluginTestMockBackend implements ComputeBackend {
        private final Map<Long, byte[]> bufferData = new HashMap<>();
        private final AtomicLong bufferIdCounter = new AtomicLong(1);
        private final PluginTestMockQueue computeQueue = new PluginTestMockQueue(bufferData);
        private final PluginTestMockQueue transferQueue = new PluginTestMockQueue(bufferData);
        private volatile boolean closed = false;

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("PluginTestMockDevice")
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
            return new PluginTestMockBuffer(id, sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return new PluginTestMockKernelHandle(kernelName);
        }

        /** Declines kernel artifacts so tensor ops take their CPU path. */
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

    private static class PluginTestMockQueue implements ComputeQueue {
        private final Map<Long, byte[]> bufferData;
        private volatile boolean closed = false;

        PluginTestMockQueue(Map<Long, byte[]> bufferData) {
            this.bufferData = bufferData;
        }

        @Override
        public QueueType getQueueType() {
            return QueueType.COMPUTE;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                System.arraycopy(srcData, 0, data, (int) dstOffset, (int) sizeInBytes);
            }
            return new PluginTestMockEvent();
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
            return new PluginTestMockEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                System.arraycopy(data, (int) srcOffset, dstData, 0, (int) sizeInBytes);
            }
            return new PluginTestMockEvent();
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
            return new PluginTestMockEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new PluginTestMockEvent();
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

    private static class PluginTestMockBuffer implements GpuBuffer {
        private final long bufferId;
        private final long sizeInBytes;
        private volatile boolean closed = false;

        PluginTestMockBuffer(long bufferId, long sizeInBytes) {
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

    private static class PluginTestMockEvent implements GpuEvent {
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

    private static class PluginTestMockKernelHandle implements KernelHandle {
        private final String kernelName;
        private volatile boolean closed = false;

        PluginTestMockKernelHandle(String kernelName) {
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

    private static class AtomicLong {
        private long value;

        AtomicLong(long initialValue) {
            this.value = initialValue;
        }

        synchronized long getAndIncrement() {
            return value++;
        }
    }
}
