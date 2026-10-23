package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class DatasetLoadersTest {

    private DataStoringMockBackend backend;
    private Path tempDir;

    @Before
    public void setUp() throws IOException {
        backend = new DataStoringMockBackend();
        tempDir = Files.createTempDirectory("dataset_test");
    }

    @After
    public void tearDown() throws IOException {
        deleteRecursively(tempDir.toFile());
    }

    private void deleteRecursively(File file) {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        file.delete();
    }

    // ==================== ImageFolderDataset Tests ====================

    @Test
    public void testImageFolderClassNamesMapping() throws Exception {
        Path class1Dir = tempDir.resolve("cat");
        Path class2Dir = tempDir.resolve("dog");
        Files.createDirectories(class1Dir);
        Files.createDirectories(class2Dir);

        createTestImage(class1Dir.resolve("cat1.jpg"), 10, 10, 0xFF0000FF);
        createTestImage(class1Dir.resolve("cat2.jpg"), 10, 10, 0xFF0000FF);
        createTestImage(class2Dir.resolve("dog1.jpg"), 10, 10, 0xFF00FF00);

        ImageFolderDataset dataset = ImageFolderDataset.builder()
            .rootPath(tempDir)
            .backend(backend)
            .targetSize(10, 10)
            .build();

        List<String> classNames = dataset.classNames();
        assertEquals(2, classNames.size());
        assertTrue(classNames.contains("cat"));
        assertTrue(classNames.contains("dog"));

        Map<String, Integer> classToIndex = dataset.classToIndex();
        assertEquals(2, classToIndex.size());
        assertTrue(classToIndex.containsKey("cat"));
        assertTrue(classToIndex.containsKey("dog"));
        
        assertEquals(3, dataset.size());
        assertEquals(2, dataset.numClasses());
    }

    @Test
    public void testImageFolderSampleRetrieval() throws Exception {
        Path classDir = tempDir.resolve("class_a");
        Files.createDirectories(classDir);
        createTestImage(classDir.resolve("img1.jpg"), 8, 8, 0xFF808080);

        ImageFolderDataset dataset = ImageFolderDataset.builder()
            .rootPath(tempDir)
            .backend(backend)
            .targetSize(8, 8)
            .grayscale(true)
            .build();

        ImageFolderDataset.Sample sample = dataset.get(0);
        assertNotNull(sample.image());
        assertEquals("class_a", sample.labelName());
        assertEquals(3, sample.image().shape().rank());
    }

    @Test
    public void testImageFolderBuilder() throws Exception {
        Path classDir = tempDir.resolve("test");
        Files.createDirectories(classDir);
        createTestImage(classDir.resolve("img.jpg"), 16, 16, 0xFFFFFFFF);

        ImageFolderDataset dataset = ImageFolderDataset.builder()
            .rootPath(tempDir)
            .backend(backend)
            .targetSize(16, 16)
            .imagenetNormalization()
            .build();

        assertEquals(16, dataset.targetWidth());
        assertEquals(16, dataset.targetHeight());
    }

    // ==================== CSVDataset Tests ====================

    @Test
    public void testCSVHeaderParsing() throws Exception {
        Path csvFile = tempDir.resolve("data.csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile.toFile()))) {
            writer.write("feature1,feature2,label\n");
            writer.write("1.0,2.0,0\n");
            writer.write("3.0,4.0,1\n");
            writer.write("5.0,6.0,0\n");
        }

        CSVDataset dataset = CSVDataset.builder()
            .csvPath(csvFile)
            .backend(backend)
            .hasHeader(true)
            .build();

        List<String> columnNames = dataset.columnNames();
        assertEquals(3, columnNames.size());
        assertEquals("feature1", columnNames.get(0));
        assertEquals("feature2", columnNames.get(1));
        assertEquals("label", columnNames.get(2));
    }

    @Test
    public void testCSVNumericColumns() throws Exception {
        Path csvFile = tempDir.resolve("numeric.csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile.toFile()))) {
            writer.write("1.5,2.5,10\n");
            writer.write("3.5,4.5,20\n");
        }

        CSVDataset dataset = CSVDataset.builder()
            .csvPath(csvFile)
            .backend(backend)
            .hasHeader(false)
            .featureColumns(0, 1)
            .targetColumns(2)
            .build();

        assertEquals(2, dataset.size());
        assertEquals(2, dataset.numFeatures());
        assertEquals(1, dataset.numTargets());

        CSVDataset.Sample sample1 = dataset.get(0);
        assertNotNull(sample1.features());
        assertNotNull(sample1.targets());
        assertEquals(1, sample1.features().shape().rank());
    }

    @Test
    public void testCSVCategoricalColumns() throws Exception {
        Path csvFile = tempDir.resolve("categorical.csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile.toFile()))) {
            writer.write("color,size,label\n");
            writer.write("red,small,0\n");
            writer.write("blue,large,1\n");
            writer.write("red,medium,0\n");
        }

        CSVDataset dataset = CSVDataset.builder()
            .csvPath(csvFile)
            .backend(backend)
            .hasHeader(true)
            .featureColumns(0, 1)
            .targetColumns(2)
            .build();

        assertEquals(3, dataset.size());

        CSVDataset.Sample sample1 = dataset.get(0);
        assertNotNull(sample1.features());
    }

    @Test
    public void testCSVNormalizeFeatures() throws Exception {
        Path csvFile = tempDir.resolve("normalize.csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile.toFile()))) {
            writer.write("0.0,10.0\n");
            writer.write("1.0,20.0\n");
            writer.write("2.0,30.0\n");
        }

        CSVDataset dataset = CSVDataset.builder()
            .csvPath(csvFile)
            .backend(backend)
            .hasHeader(false)
            .featureColumns(0)
            .targetColumns(1)
            .normalizeFeatures(true)
            .build();

        float[] mean = dataset.featureMean();
        float[] std = dataset.featureStd();
        assertNotNull(mean);
        assertNotNull(std);
        assertEquals(1.0f, mean[0], 0.01f);
        assertTrue(std[0] > 0);
    }

    @Test
    public void testCSVOneHotEncodeTargets() throws Exception {
        Path csvFile = tempDir.resolve("onehot.csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile.toFile()))) {
            writer.write("1.0,0\n");
            writer.write("2.0,1\n");
            writer.write("3.0,2\n");
        }

        CSVDataset dataset = CSVDataset.builder()
            .csvPath(csvFile)
            .backend(backend)
            .hasHeader(false)
            .featureColumns(0)
            .targetColumns(1)
            .oneHotEncodeTargets(true)
            .build();

        assertEquals(3, dataset.numClasses());

        CSVDataset.Sample sample = dataset.get(0);
        assertEquals(3, sample.targets().shape().dim(0));
    }

    // ==================== TextFileDataset Tests ====================

    @Test
    public void testTextFileLineByLine() throws Exception {
        Path textFile = tempDir.resolve("text.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(textFile.toFile()))) {
            writer.write("Hello world\n");
            writer.write("This is a test\n");
            writer.write("NeuralFree dataset\n");
        }

        TextFileDataset dataset = TextFileDataset.builder()
            .filePath(textFile)
            .backend(backend)
            .whitespaceTokenizer()
            .build();

        assertEquals(3, dataset.size());

        TextFileDataset.Sample sample0 = dataset.get(0);
        assertEquals("Hello world", sample0.text());
        assertNotNull(sample0.tokens());
    }

    @Test
    public void testTextFileVocabulary() throws Exception {
        Path textFile = tempDir.resolve("vocab.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(textFile.toFile()))) {
            writer.write("cat dog bird\n");
            writer.write("dog cat fish\n");
        }

        TextFileDataset dataset = TextFileDataset.builder()
            .filePath(textFile)
            .backend(backend)
            .whitespaceTokenizer()
            .build();

        assertTrue(dataset.vocabularySize() >= 4);
        assertTrue(dataset.vocabulary().containsKey("cat"));
        assertTrue(dataset.vocabulary().containsKey("dog"));
        assertTrue(dataset.vocabulary().containsKey("bird"));
        assertTrue(dataset.vocabulary().containsKey("fish"));
    }

    @Test
    public void testTextFileMaxSequenceLength() throws Exception {
        Path textFile = tempDir.resolve("maxlen.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(textFile.toFile()))) {
            writer.write("one two three four five six seven eight\n");
            writer.write("a b c\n");
        }

        TextFileDataset dataset = TextFileDataset.builder()
            .filePath(textFile)
            .backend(backend)
            .whitespaceTokenizer()
            .maxSequenceLength(4)
            .paddingEnabled(true)
            .build();

        TextFileDataset.Sample sample0 = dataset.get(0);
        assertEquals(4, sample0.length());

        TextFileDataset.Sample sample1 = dataset.get(1);
        assertEquals(4, sample1.length());
    }

    @Test
    public void testTextFileCharacterTokenizer() throws Exception {
        Path textFile = tempDir.resolve("chars.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(textFile.toFile()))) {
            writer.write("ABC\n");
        }

        TextFileDataset dataset = TextFileDataset.builder()
            .filePath(textFile)
            .backend(backend)
            .characterTokenizer()
            .build();

        TextFileDataset.Sample sample = dataset.get(0);
        assertEquals(3, sample.tokenIds().length);
    }

    @Test
    public void testTextFileAttentionMask() throws Exception {
        Path textFile = tempDir.resolve("mask.txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(textFile.toFile()))) {
            writer.write("short\n");
        }

        TextFileDataset dataset = TextFileDataset.builder()
            .filePath(textFile)
            .backend(backend)
            .whitespaceTokenizer()
            .maxSequenceLength(5)
            .paddingEnabled(true)
            .build();

        TextFileDataset.Sample sample = dataset.get(0);
        float[] mask = sample.attentionMask();
        assertEquals(5, mask.length);
        assertEquals(1.0f, mask[0], 0.001f);
    }

    // ==================== Mock Backend Implementation ====================

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
        private final long id;
        private final long size;
        private volatile boolean closed = false;

        MockGpuBuffer(long id, long size) {
            this.id = id;
            this.size = size;
        }

        @Override
        public long sizeInBytes() {
            return size;
        }

        @Override
        public long bufferId() {
            return id;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        @Override
        public void awaitCompletion() {
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String name;

        MockKernelHandle(String name) {
            this.name = name;
        }

        @Override
        public String kernelName() {
            return name;
        }

        @Override
        public void close() {
        }
    }

    private void createTestImage(Path path, int width, int height, int rgb) throws IOException {
        java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(
            width, height, java.awt.image.BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, rgb);
            }
        }
        javax.imageio.ImageIO.write(image, "jpg", path.toFile());
    }
}
