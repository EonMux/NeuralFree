package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.Qwen4ExpSidecars;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;


/**
 * A qwen4exp MTP draft file without its own {@code token_embd} or LM head takes the target's
 * ({@code llama_model_share_mtp_io}); a file that carries them keeps them.
 */
public class NextnDraftModelSharedIoTest {

    /** The block tensors every sidecar holds: the NextN fusion and its gated HC head. */
    private static final String[] BLOCK = {
            "blk.1.nextn.eh_proj.weight", "blk.1.nextn.enorm.weight", "blk.1.nextn.hnorm.weight",
            "blk.1.nextn.hc_head_norm.weight", "blk.1.nextn.hc_head_down.weight", "blk.1.nextn.hc_head_up.weight",
    };
    private static final String TARGET_EMBD = "token_embd.weight";
    private static final String TARGET_HEAD = "output.weight";

    /** A head over recorded tensor tables; only what {@code NextnDraftModel}'s constructor reads is real. */
    private static final class FakeHead extends MtpHead {
        private final Map<Source, Set<String>> tensors = new HashMap<>();
        /** Every (source, name) the head read the shape of, "source name". */
        final List<String> shapeQueries = new ArrayList<>();

        FakeHead(ModelSpec target, Set<String> draftTensors, Set<String> targetTensors) {
            super(target);
            tensors.put(Source.DRAFT, new HashSet<>(draftTensors));
            tensors.put(Source.TARGET, new HashSet<>(targetTensors));
        }

        @Override
        public int ubatch() {
            return 8;
        }

        @Override
        public Rows rows(String name, int width) {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean has(Source s, String name) {
            return tensors.get(s).contains(name);
        }

        @Override
        public long[] shape(Source s, String name) {
            assertTrue(has(s, name));
            shapeQueries.add(s + " " + name);
            return new long[] {64, 64};
        }

        @Override
        public float[] vector(Source s, String name) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int[] integers(Source s, String name) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void write(Rows dst, float[] src, int nRows) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void copyRows(Rows src, int srcRow, Rows dst, int dstRow, int n) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void read(Rows src, int nRows, float[] dst) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void readRow(Rows src, int row, float[] dst) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void embed(Source s, String table, int[] tokens, int n, Rows out) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void norm(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void concat(Rows a, Rows b, Rows out, int n) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void project(Source s, String weight, Rows in, Rows out, int n) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void add(Rows a, Rows b, Rows out, int n) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void scale(Rows x, float s, int n) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void normStreams(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n,
                                int streams) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void concatStreams(Rows a, Rows b, Rows out, int n, int streams) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void projectStreams(Source s, String weight, Rows in, Rows out, int n, int streams) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Collapse collapse(Source s, HcSubBlock site) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void logits(Source s, String weight, Rows in, int row, float[] dst) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Memory ownMemory(ModelSpec spec, MemoryPlan plan) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Memory targetCaches(ModelSpec spec, MemoryPlan plan) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Stack stack(ModelSpec spec, Source s, Memory memory, List<LayerSpec> layers) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Blocks blocks(ModelSpec spec, Source s, Memory memory, LayerSpec layer) {
            throw new UnsupportedOperationException();
        }
    }

    private static ModelSpec draft(ModelSpec target, List<com.eonmux.neuralfree.gguf.GGUFTensorInfo> tensors) {
        return Qwen4ExpSidecars.buildSidecar(target, 4, 0, tensors);
    }

    @Test
    public void aSharedSidecarTakesTheTargetEmbeddingAndHead() {
        ModelSpec target = Qwen4ExpSidecars.buildTarget();
        FakeHead head = new FakeHead(target, new HashSet<>(Arrays.asList(BLOCK)),
                new HashSet<>(Arrays.asList(TARGET_EMBD, TARGET_HEAD)));
        new NextnDraftModel(head, draft(target, Qwen4ExpSidecars.sharedSidecarTensors()), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.DEFAULT);
        assertTrue(head.shapeQueries.toString(),
                head.shapeQueries.contains(MtpHead.Source.TARGET + " " + TARGET_HEAD));
    }

    @Test
    public void aSidecarWithItsOwnIoKeepsIt() {
        ModelSpec target = Qwen4ExpSidecars.buildTarget();
        Set<String> draftTensors = new HashSet<>(Arrays.asList(BLOCK));
        draftTensors.add(TARGET_EMBD);
        draftTensors.add(TARGET_HEAD);
        FakeHead head = new FakeHead(target, draftTensors,
                new HashSet<>(Arrays.asList(TARGET_EMBD, TARGET_HEAD)));
        new NextnDraftModel(head, draft(target, Qwen4ExpSidecars.fullSidecarTensors()), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.DEFAULT);
        assertTrue(head.shapeQueries.toString(),
                head.shapeQueries.contains(MtpHead.Source.DRAFT + " " + TARGET_HEAD));
    }

    @Test
    public void theTargetsOwnBlocksReadItsIo() {
        ModelSpec target = Qwen4ExpSidecars.buildTarget();
        Set<String> all = new HashSet<>(Arrays.asList(BLOCK));
        all.add(TARGET_EMBD);
        all.add(TARGET_HEAD);
        FakeHead head = new FakeHead(target, all, all);
        new NextnDraftModel(head, target, MtpHead.Source.TARGET, DraftModelLoader.Options.DEFAULT);
        assertTrue(head.shapeQueries.toString(),
                head.shapeQueries.contains(MtpHead.Source.TARGET + " " + TARGET_HEAD));
    }

    @Test
    public void theDraftIsRefusedWhenTheTargetCannotShare() {
        ModelSpec target = Qwen4ExpSidecars.buildTarget();
        FakeHead head = new FakeHead(target, new HashSet<>(Arrays.asList(BLOCK)),
                new HashSet<>(Arrays.asList(TARGET_HEAD)));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new NextnDraftModel(head, draft(target, Qwen4ExpSidecars.sharedSidecarTensors()),
                        MtpHead.Source.DRAFT, DraftModelLoader.Options.DEFAULT));
        assertTrue(e.getMessage(), e.getMessage().contains("token embeddings"));
    }
}
