package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory.DraftRoute;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture;
import org.junit.Test;

import java.util.List;
import java.util.function.Predicate;

import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.ARCH;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.N_EMBD;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.N_VOCAB;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.TRUNK;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link LLMModelFactory#draftRoute}: the speculative type llama.cpp's {@code --spec-type} chooses. Automatic
 * detection drafts with the NextN blocks of an MTP draft file of the target's model and runs another model of the
 * family that keeps its NextN blocks as a separate draft model; {@code draft-mtp} refuses such a file and
 * {@code draft-simple} always runs a separate draft model. The device tensors of each route are the ones its drafter
 * places.
 */
public class DraftRouteTest {

    private static final String PATH = "draft.gguf";

    /** A draft file: its metadata and tensor catalog. */
    private static final class File {
        final GgufKv kv;
        final TensorCatalog tc;

        File(int nEmbd, int trunk, boolean mtpOnly) {
            this.kv = GgufKv.ofMap(NemotronNextnFixture.metadata(nEmbd, trunk, N_VOCAB));
            List<GGUFTensorInfo> full = NemotronNextnFixture.tensors(nEmbd, trunk, N_VOCAB);
            this.tc = TensorCatalog.of(ARCH, mtpOnly ? NemotronNextnFixture.mtpOnly(full, trunk) : full);
        }

        File(GgufKv kv, TensorCatalog tc) {
            this.kv = kv;
            this.tc = tc;
        }

        DraftRoute route(DraftType type) {
            return LLMModelFactory.draftRoute(Arch.NEMOTRON_H_MOE, kv, tc, target(), type, PATH);
        }
    }

    private static ModelSpec target() {
        return ModelSpecBuilder.build(GgufKv.ofMap(NemotronNextnFixture.metadata()),
                TensorCatalog.of(ARCH, NemotronNextnFixture.tensors()), MemoryConfig.defaults());
    }

    private static File ownFull() {
        return new File(N_EMBD, TRUNK, false);
    }

    private static File widerSibling() {
        return new File(2 * N_EMBD, TRUNK, false);
    }

    private static File deeperSibling() {
        return new File(N_EMBD, 2 * TRUNK, false);
    }

    private static File noNextn() {
        return new File(GgufKv.ofMap(NemotronNextnFixture.metadata()), TensorCatalog.of(ARCH,
                NemotronNextnFixture.withoutNextn(NemotronNextnFixture.tensors(), TRUNK)));
    }

    private static GGUFTensorInfo t(String name) {
        return new GGUFTensorInfo(name, new long[] {1}, GGMLType.F32, 0L);
    }

    @Test
    public void theTargetsOwnNextnFileDraftsWithItsNextnBlocksUnlessASeparateModelIsAskedFor() {
        assertEquals(DraftRoute.MTP_FILE, ownFull().route(DraftType.AUTO));
        assertEquals(DraftRoute.MTP_FILE, ownFull().route(DraftType.MTP));
        assertEquals(DraftRoute.MODEL, ownFull().route(DraftType.MODEL));
        File sidecar = new File(N_EMBD, TRUNK, true);
        assertEquals(DraftRoute.MTP_FILE, sidecar.route(DraftType.AUTO));
        assertEquals(DraftRoute.MTP_FILE, sidecar.route(DraftType.MTP));
    }

    @Test
    public void anotherModelOfTheFamilyWithNextnBlocksDraftsAsASeparateModel() {
        for (File sibling : new File[] {widerSibling(), deeperSibling()}) {
            assertEquals(DraftRoute.MODEL, sibling.route(DraftType.AUTO));
            assertEquals(DraftRoute.MODEL, sibling.route(DraftType.MODEL));
        }
    }

    @Test
    public void draftMtpRefusesAFileThatIsNotAnMtpDraftOfTheTarget() {
        IllegalArgumentException wider = assertThrows(IllegalArgumentException.class,
                () -> widerSibling().route(DraftType.MTP));
        assertTrue(wider.getMessage(), wider.getMessage().contains("n_embd " + 2 * N_EMBD));
        assertTrue(wider.getMessage(), wider.getMessage().contains("draft-simple"));
        IllegalArgumentException deeper = assertThrows(IllegalArgumentException.class,
                () -> deeperSibling().route(DraftType.MTP));
        assertTrue(deeper.getMessage(), deeper.getMessage().contains(2 * TRUNK + " trunk layers"));
        IllegalArgumentException plain = assertThrows(IllegalArgumentException.class,
                () -> noNextn().route(DraftType.MTP));
        assertTrue(plain.getMessage(), plain.getMessage().contains("no NextN block"));
    }

    @Test
    public void aFileWithoutNextnBlocksIsASeparateDraftModel() {
        assertEquals(DraftRoute.MODEL, noNextn().route(DraftType.AUTO));
        assertEquals(DraftRoute.MODEL, noNextn().route(DraftType.MODEL));
    }

    @Test
    public void mtpBlocksOfAnotherModelHaveNoModelToFallBackTo() {
        File foreignSidecar = new File(2 * N_EMBD, TRUNK, true);
        for (DraftType type : DraftType.values()) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> foreignSidecar.route(type));
            assertTrue(e.getMessage(), e.getMessage().contains("only"));
        }
        IllegalArgumentException asModel = assertThrows(IllegalArgumentException.class,
                () -> new File(N_EMBD, TRUNK, true).route(DraftType.MODEL));
        assertTrue(asModel.getMessage(), asModel.getMessage().contains("draft-mtp"));
    }

    @Test
    public void drafterArchitecturesDraftOnlyAsThemselves() {
        File any = ownFull();
        ModelSpec target = target();
        assertEquals(DraftRoute.DFLASH,
                LLMModelFactory.draftRoute(Arch.DFLASH, any.kv, any.tc, target, DraftType.AUTO, PATH));
        assertEquals(DraftRoute.HEAD,
                LLMModelFactory.draftRoute(Arch.EAGLE3, any.kv, any.tc, target, DraftType.AUTO, PATH));
        assertEquals(DraftRoute.HEAD,
                LLMModelFactory.draftRoute(Arch.GEMMA4_ASSISTANT, any.kv, any.tc, target, DraftType.AUTO, PATH));
        for (DraftType forced : new DraftType[] {DraftType.MTP, DraftType.MODEL}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> LLMModelFactory.draftRoute(Arch.EAGLE3, any.kv, any.tc, target, forced, PATH));
            assertTrue(e.getMessage(), e.getMessage().contains("auto"));
        }
    }

    @Test
    public void eachRoutePlacesTheTensorsItsDrafterReads() {
        File own = ownFull();
        MemoryConfig draft = MemoryConfig.defaults().forDraftContext();
        String trunk = "blk.0.ssm_in.weight";
        String nextn = "blk." + TRUNK + ".nextn.eh_proj.weight";
        String embd = "token_embd.weight";

        Predicate<GGUFTensorInfo> mtp = LLMModelFactory.deviceTensors(DraftRoute.MTP_FILE, own.kv, own.tc, draft);
        assertFalse(mtp.test(t(trunk)));
        assertTrue(mtp.test(t(nextn)));
        assertTrue(mtp.test(t(embd)));

        Predicate<GGUFTensorInfo> model = LLMModelFactory.deviceTensors(DraftRoute.MODEL, own.kv, own.tc, draft);
        assertTrue(model.test(t(trunk)));
        assertFalse(model.test(t(nextn)));
        assertFalse(model.test(t("blk." + TRUNK + ".attn_q.weight")));
        assertTrue(model.test(t(embd)));

        assertFalse(LLMModelFactory.deviceTensors(DraftRoute.DFLASH, own.kv, own.tc, draft).test(t(embd)));
        assertTrue(LLMModelFactory.deviceTensors(DraftRoute.HEAD, own.kv, own.tc, draft).test(t(trunk)));
    }
}
