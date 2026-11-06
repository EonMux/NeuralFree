package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * Vocabulary-parallel parity on the OpenCL backend: the token-table lookup and the LM head of a synthetic llama run
 * once over the whole vocabulary and once as a two-rank {@link TpVocab} split whose peer runs on a second backend of
 * the same device. The embedding rows (summed over the ranks) and the gathered logits match the whole-vocabulary
 * stages. The GPU cases are skipped when no OpenCL device is available.
 */
public class TpVocabTest {

    private static final int UBATCH = 3;
    private static final int RANKS = 2;
    private static final float TOL = 1e-4f;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend main;
    private OclComputeBackend peer;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private GpuPlaceholders placeholders;
    private GGUFFileParser parser;
    private WeightStore store;
    private ModelSpec spec;
    private GpuWeightUploader referenceUploader;
    private GpuScratch referenceScratch;
    private GpuOps referenceOps;
    private GpuForwardContext referenceContext;
    private GpuEmbeddingStage referenceEmbedding;
    private GpuLogitsHead referenceHead;
    private GpuWeightUploader splitUploader;
    private GpuScratch splitScratch;
    private GpuOps splitOps;
    private GpuForwardContext splitContext;
    private GpuEmbeddingStage splitEmbedding;
    private GpuLogitsHead splitHead;
    private TensorParallelGroup group;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        MemoryConfig mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);

        referenceScratch = GpuScratch.forModel(main, spec, UBATCH);
        referenceOps = new GpuOps(queue, kernels, referenceScratch, placeholders);
        referenceUploader = uploader(main, queue);
        referenceUploader.uploadResident(t -> false);
        referenceContext = GpuForwardContext.builder(kernels, referenceOps, referenceUploader, spec).build();
        referenceEmbedding = new GpuEmbeddingStage(referenceContext);
        referenceHead = new GpuLogitsHead(referenceContext, Collections.emptyList());

        TpVocabPlan vocab = TpVocabPlan.of(spec, RANKS, referenceUploader);
        FfnRowSplitPlan ffn = FfnRowSplitPlan.of(spec, RANKS, referenceUploader);
        splitScratch = GpuScratch.forModel(main, spec, UBATCH);
        splitOps = new GpuOps(queue, kernels, splitScratch, placeholders);
        splitUploader = uploader(main, queue);
        splitUploader.uploadResident(t -> vocab.splits(t.name()) || ffn.splits(t.name()));
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, ffn, null, vocab, null, null, 0);
        group.upload(store, splitUploader);
        splitContext = GpuForwardContext.builder(kernels, splitOps, splitUploader, spec).build();
        splitEmbedding = new GpuEmbeddingStage(splitContext, group.vocabEmbedding());
        splitHead = new GpuLogitsHead(splitContext, Collections.emptyList(), group.vocabHead());
    }

    private GpuWeightUploader uploader(OclComputeBackend backend, ComputeQueue q) {
        return new GpuWeightUploader(backend, q, parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
    }

    @After
    public void close() throws IOException {
        try {
            if (splitHead != null) {
                splitHead.close();
            }
            if (referenceHead != null) {
                referenceHead.close();
            }
            if (group != null) {
                group.close();
            }
            splitUploader.close();
            splitOps.close();
            splitScratch.close();
            referenceUploader.close();
            referenceOps.close();
            referenceScratch.close();
            store.close();
            placeholders.close();
            main.close();
            peer.close();
            parser.close();
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            Files.deleteIfExists(file);
        }
    }

    private float[] read(GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, (long) floats * Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }

    @Test
    public void theEmbeddingLookupSumsToTheWholeVocabulary() {
        int vocab = (int) referenceUploader.plan(spec.embedding().tokenEmbd().name()).rows();
        int[] tokens = {1, vocab - 1, vocab / 2};
        referenceEmbedding.embed(tokens, 0, UBATCH, null, 0);
        splitEmbedding.embed(tokens, 0, UBATCH, null, 0);
        ScratchSlot refX = referenceScratch.slot(ScratchSlots.X);
        ScratchSlot splitX = splitScratch.slot(ScratchSlots.X);
        int floats = (int) Math.min(refX.floats(), splitX.floats());
        assertArrayEquals("the summed vocabulary-parallel lookup matches the whole token table",
                read(refX.buffer(), floats), read(splitX.buffer(), floats), TOL);
    }

    @Test
    public void theGatheredLogitsMatchTheWholeHead() {
        int width = (int) (referenceScratch.slot(ScratchSlots.X).floats() / UBATCH);
        float[] hidden = new float[UBATCH * width];
        for (int i = 0; i < hidden.length; i++) {
            hidden[i] = (float) Math.sin(i * 0.21);
        }
        write(referenceScratch.slot(ScratchSlots.X), hidden);
        write(splitScratch.slot(ScratchSlots.X), hidden);
        ScratchSlot refLogits = referenceHead.logits(UBATCH, referenceScratch.slot(ScratchSlots.X));
        ScratchSlot splitLogits = splitHead.logits(UBATCH, splitScratch.slot(ScratchSlots.X));
        assertEquals("the vocabulary split keeps the full logits width",
                referenceHead.vocab(), splitHead.vocab());
        int floats = (int) Math.min(refLogits.floats(), splitLogits.floats());
        assertArrayEquals("the gathered vocabulary-parallel logits match the whole head",
                read(refLogits.buffer(), floats), read(splitLogits.buffer(), floats), TOL);
        assertArrayEquals("the vocabulary-parallel head selects the same greedy ids",
                readIds(referenceHead.argmax(UBATCH)), readIds(splitHead.argmax(UBATCH)));
    }

    @Test
    public void thePlanRefusesAVocabularyThatDoesNotDivide() {
        assertThrows(UnsupportedOperationException.class, () -> TpVocabPlan.of(spec, 3, referenceUploader));
    }

    private void write(ScratchSlot slot, float[] values) {
        try (GpuEvent e = queue.enqueueWriteBuffer(slot.buffer(), (long) slot.offset() * Float.BYTES,
                (long) values.length * Float.BYTES, values)) {
            queue.finish();
        }
    }

    private int[] readIds(ScratchSlot slot) {
        ScratchSlot argmax = slot;
        byte[] bytes = new byte[UBATCH * Integer.BYTES];
        try (GpuEvent e = queue.enqueueReadBuffer(argmax.buffer(), (long) argmax.offset() * Float.BYTES,
                bytes.length, bytes)) {
            queue.finish();
        }
        int[] ids = new int[UBATCH];
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < ids.length; i++) {
            ids[i] = bb.getInt(i * Integer.BYTES);
        }
        return ids;
    }
}
