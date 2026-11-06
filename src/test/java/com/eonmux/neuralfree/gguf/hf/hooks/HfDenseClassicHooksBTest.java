package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfDenseClassicHooksB}'s EXAONE 4.5 MTP renames (exaone.py:219-270): the shared {@code mtp.fc}/norm tensors
 * fill every appended NextN block and the {@code mtp.layers.N} tensors land in their appended trunk slot.
 */
public class HfDenseClassicHooksBTest {

    /** The MTP head of exaone.py:225-230; the rest of {@link HfQwenHooksTest#BASE_CONFIG} stays as given. */
    private static final String MTP_CONFIG = "'num_nextn_predict_layers': 2";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HfLoadContext context(String extraConfig) throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.EXAONE4,
                "Exaone4_5_ForConditionalGeneration", extraConfig, Collections.emptyMap(), true);
        return ctx.withBlockCount(HfDenseClassicHooksB.exaone45Blocks(ctx.config()));
    }

    @Test
    public void theSharedMtpTensorFillsEveryAppendedBlock() throws IOException {
        HfLoadContext ctx = context(MTP_CONFIG);
        assertEquals(4, ctx.blockCount());
        RecordingSink sink = new RecordingSink();
        HfDenseClassicHooksB.exaone45Tensors(ctx, "mtp.fc.weight", seq(8, 16), sink);
        assertTrue(sink.tensors.keySet().containsAll(
                java.util.Arrays.asList("blk.2.nextn.eh_proj.weight", "blk.3.nextn.eh_proj.weight")));
        assertEquals(2, sink.tensors.size());

        sink = new RecordingSink();
        HfDenseClassicHooksB.exaone45Tensors(ctx, "mtp.norm.weight", seq(8), sink);
        assertTrue(sink.tensors.keySet().containsAll(
                java.util.Arrays.asList("blk.2.nextn.shared_head_norm.weight",
                        "blk.3.nextn.shared_head_norm.weight")));
    }

    @Test
    public void aLayerTensorLandsInItsAppendedSlot() throws IOException {
        HfLoadContext ctx = context(MTP_CONFIG);
        RecordingSink sink = new RecordingSink();
        HfDenseClassicHooksB.exaone45Tensors(ctx, "mtp.layers.1.self_attn.q_proj.weight", seq(8, 8), sink);
        assertEquals(Collections.singleton("blk.3.attn_q.weight"), sink.tensors.keySet());
    }

    @Test
    public void aTrunkTensorGoesThroughTheDefaultMapping() throws IOException {
        HfLoadContext ctx = context(MTP_CONFIG);
        RecordingSink sink = new RecordingSink();
        HfDenseClassicHooksB.exaone45Tensors(ctx, "model.layers.0.self_attn.q_proj.weight", seq(8, 8), sink);
        assertEquals(Collections.singleton("blk.0.attn_q.weight"), sink.tensors.keySet());
    }

    @Test
    public void sharedLayersReplicateIntoEveryAppendedSlot() throws IOException {
        HfLoadContext ctx = context(MTP_CONFIG + ", 'mtp_share_layers': true");
        RecordingSink sink = new RecordingSink();
        HfDenseClassicHooksB.exaone45Tensors(ctx, "mtp.layers.0.self_attn.q_proj.weight", seq(8, 8), sink);
        assertEquals(new ArrayList<>(java.util.Arrays.asList("blk.2.attn_q.weight", "blk.3.attn_q.weight")),
                new ArrayList<>(sink.tensors.keySet()));
    }

    @Test
    public void theMetadataWritesTheNextnCount() throws IOException {
        HfLoadContext ctx = context(MTP_CONFIG);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.EXAONE4.ggufName());
        HfDenseClassicHooksB.exaone45Meta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(2, ((Number) m.get(String.format(GgufKeys.NEXTN_PREDICT_LAYERS,
                Arch.EXAONE4.ggufName()))).intValue());
    }
}
