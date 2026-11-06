package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.cpu.head.CpuLogitsHead;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The certified exact-argmax head over the synthetic decoders, CPU path: a greedy loop whose committed
 * tokens come from the certified lane equals the target-only greedy loop (exact logits rows, host
 * argmax) token for token — through the rescore, and, under a cap of one candidate, through the
 * full-head fallback — and so does the batched verify ({@code forwardRowsGreedy}). Heads with a softcap,
 * an ineligible stored type or the lane off keep the exact readout and refuse the certified call by name.
 */
public class CertifiedArgmaxParityTest {

    private static final int[] PROMPT = {4, 17, 30, 43, 56, 69};
    private static final int TOKENS = 8;

    private final List<Path> models = new ArrayList<>();
    private String originalMode;
    private String originalLegacyMode;
    private String originalKmax;
    private String originalLegacyKmax;

    @Before
    public void rememberProperties() {
        originalMode = System.getProperty(CertifiedArgmaxMode.PROPERTY);
        originalLegacyMode = System.getProperty(CertifiedArgmaxMode.LEGACY_PROPERTY);
        originalKmax = System.getProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        originalLegacyKmax = System.getProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY);
    }

    @After
    public void restoreProperties() throws IOException {
        restore(CertifiedArgmaxMode.PROPERTY, originalMode);
        restore(CertifiedArgmaxMode.LEGACY_PROPERTY, originalLegacyMode);
        restore(CertifiedArgmaxTuning.KMAX_PROPERTY, originalKmax);
        restore(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY, originalLegacyKmax);
        for (Path model : models) {
            Files.deleteIfExists(model);
        }
    }

    private static void restore(String property, String value) {
        if (value == null) {
            System.clearProperty(property);
        } else {
            System.setProperty(property, value);
        }
    }

    private void setMode(String value) {
        if (value == null) {
            System.clearProperty(CertifiedArgmaxMode.PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxMode.PROPERTY, value);
        }
    }

    private void setKmax(String value) {
        if (value == null) {
            System.clearProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxTuning.KMAX_PROPERTY, value);
        }
    }

    private GGUFModel open(SyntheticModels.Config config) throws IOException {
        Path path = SyntheticModels.llama(config).writeTempFile();
        models.add(path);
        return new GGUFModel(path.toString(), config());
    }

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static SyntheticModels.Config typed(GGMLType type) {
        return SyntheticModels.Config.llama().toBuilder().weightType(type).build();
    }

    /** The target-only greedy loop: exact logits rows, host argmax per step. */
    private static int[] targetLoop(GGUFModel model) {
        return loop(model, false);
    }

    /** The device greedy loop: {@code decodeGreedyOnDevice} per step. */
    private static int[] deviceLoop(GGUFModel model) {
        return loop(model, true);
    }

    private static int[] loop(GGUFModel model, boolean onDevice) {
        int[] tokens = new int[TOKENS];
        ForwardBackend backend = model.forwardBackend();
        int token = ForwardBackend.argmax(backend.prefill(PROMPT, 0));
        for (int i = 0; i < TOKENS; i++) {
            int pos = PROMPT.length + i;
            tokens[i] = token;
            token = onDevice ? backend.decodeGreedyOnDevice(token, pos)
                    : ForwardBackend.argmax(backend.decode(token, pos));
        }
        return tokens;
    }

    /** Greedy ids of one batched verify call: the certified lane, or the host argmax over the rows. */
    private static int[] verify(GGUFModel model, boolean onDevice) {
        ForwardBackend backend = model.forwardBackend();
        return onDevice ? backend.forwardRowsGreedy(PROMPT, 0, null)
                : idsOf(backend.forwardRows(PROMPT, 0, null));
    }

    private static int[] idsOf(float[][] rows) {
        int[] ids = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            ids[i] = ForwardBackend.argmax(rows[i]);
        }
        return ids;
    }

    /**
     * The certified F16 loop commits exactly the target loop's tokens — through the rescore and, with
     * candidates capped at one, through the full-head fallback — and the batched verify agrees.
     */
    @Test
    public void certifiedF16LoopMatchesTheTargetLoop() throws IOException {
        setMode("int8");
        int[] target;
        int[] verifyHost;
        int[] verifyDevice;
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertTrue("an eligible plain F16 head certifies", head.certifiesArgmax());
            target = targetLoop(model);
            verifyHost = verify(model, false);
        }
        int[] certified;
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            certified = deviceLoop(model);
            verifyDevice = verify(model, true);
        }
        setKmax("1");
        int[] fallback;
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            fallback = deviceLoop(model);
        }
        assertArrayEquals(target, certified);
        assertArrayEquals(target, fallback);
        assertArrayEquals(verifyHost, verifyDevice);
    }

    /** The certified BF16 loop commits exactly the target loop's tokens. */
    @Test
    public void certifiedBf16LoopMatchesTheTargetLoop() throws IOException {
        setMode("int8");
        int[] target;
        int[] verifyHost;
        try (GGUFModel model = open(typed(GGMLType.BF16))) {
            target = targetLoop(model);
            verifyHost = verify(model, false);
        }
        int[] certified;
        int[] verifyDevice;
        try (GGUFModel model = open(typed(GGMLType.BF16))) {
            certified = deviceLoop(model);
            verifyDevice = verify(model, true);
        }
        assertArrayEquals(target, certified);
        assertArrayEquals(verifyHost, verifyDevice);
    }

    /** An F32 head is ineligible by type: the exact readout stays, the certified call refuses by name. */
    @Test
    public void anF32HeadKeepsTheExactReadout() throws IOException {
        setMode("int8");
        try (GGUFModel model = open(typed(GGMLType.F32))) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertFalse("an F32 head certifies nothing", head.certifiesArgmax());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> head.argmaxCertified(1));
            assertTrue(e.getMessage(), e.getMessage().contains("does not certify argmax"));
            int[] target = targetLoop(model);
            assertArrayEquals(target, deviceLoop(model));
        }
    }

    /** A softcap head refuses by name (gemma2 writes a final logit softcap). */
    @Test
    public void aSoftcapHeadDoesNotCertify() throws IOException {
        setMode("int8");
        Path path = SyntheticModels.gemma2().writeTempFile();
        models.add(path);
        try (GGUFModel model = new GGUFModel(path.toString(), config())) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertFalse("a softcap head certifies nothing", head.certifiesArgmax());
        }
    }

    /** With the mode off no head certifies, and the greedy loop is the exact readout. */
    @Test
    public void theLaneOffKeepsTheExactReadout() throws IOException {
        setMode("off");
        int[] target;
        int[] verifyHost;
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertFalse(head.certifiesArgmax());
            target = targetLoop(model);
            verifyHost = verify(model, false);
        }
        int[] device;
        int[] verifyDevice;
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            device = deviceLoop(model);
            verifyDevice = verify(model, true);
        }
        assertArrayEquals(target, device);
        assertArrayEquals(verifyHost, verifyDevice);
    }

    /** An unknown mode value is refused by name at head construction. */
    @Test
    public void anUnknownModeIsRefusedByName() throws IOException {
        setMode("int8");
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            setMode("maybe");
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> new CpuLogitsHead(model.forwardContext(), Collections.<CpuHeadPreprocessor>emptyList()));
            assertTrue(e.getMessage(), e.getMessage().contains(CertifiedArgmaxMode.PROPERTY));
            assertTrue(e.getMessage(), e.getMessage().contains("maybe"));
        }
    }

    /**
     * The default mode is {@code auto}: no CPU head type measured the screen faster, so the plain unset
     * run keeps the exact readout, while {@code on} (the draft-lane value included) forces it.
     */
    @Test
    public void theDefaultModeKeepsTheCpuHeadExactUntilForcedOn() throws IOException {
        setMode(null);
        int[] target;
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertFalse("no CPU type measured the screen faster, so auto keeps the exact readout",
                    head.certifiesArgmax());
            target = targetLoop(model);
            assertArrayEquals(target, deviceLoop(model));
        }
        setMode("on");
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertTrue("on forces the lane on the capable F16 head", head.certifiesArgmax());
            assertArrayEquals(target, deviceLoop(model));
        }
        System.clearProperty(CertifiedArgmaxMode.PROPERTY);
        System.setProperty(CertifiedArgmaxMode.LEGACY_PROPERTY, "int8");
        try (GGUFModel model = open(typed(GGMLType.F16))) {
            CpuLogitsHead head = new CpuLogitsHead(model.forwardContext(),
                    Collections.<CpuHeadPreprocessor>emptyList());
            assertTrue("the draft-lane property aliases to on", head.certifiesArgmax());
        }
    }

    /** An unknown value through the draft-lane property is refused by that property's name. */
    @Test
    public void anUnknownValueThroughTheLegacyPropertyIsRefusedByItsName() throws IOException {
        System.clearProperty(CertifiedArgmaxMode.PROPERTY);
        System.setProperty(CertifiedArgmaxMode.LEGACY_PROPERTY, "maybe");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> open(typed(GGMLType.F16)).close());
        assertTrue(e.getMessage(), e.getMessage().contains(CertifiedArgmaxMode.LEGACY_PROPERTY));
        assertTrue(e.getMessage(), e.getMessage().contains("maybe"));
    }

    /** A non-positive candidate cap is refused by name. */
    @Test
    public void aNonPositiveCapIsRefusedByName() {
        setMode("int8");
        setKmax("0");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, CertifiedArgmaxTuning::kmax);
        assertTrue(e.getMessage(), e.getMessage().contains(CertifiedArgmaxTuning.KMAX_PROPERTY));
    }
}
