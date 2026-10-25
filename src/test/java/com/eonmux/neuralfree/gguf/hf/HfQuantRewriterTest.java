package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.HadamardPlan;
import com.eonmux.neuralfree.gguf.hf.quant.HfQuantRewriter;
import com.eonmux.neuralfree.gguf.hf.quant.QuantModuleIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfQuantRewriter} and {@link QuantModuleIndex}: module grouping of every auxiliary-suffix form, the
 * unconsumed-tensor error, the KV-cache scales, the activation-quantization rules and the online Hadamard plan.
 */
public class HfQuantRewriterTest {

    private static final String L0 = "model.layers.0.";
    private static final QuantArgs GPTQ = QuantArgs.builder(QuantFormat.INT_GPTQ).bits(4).groupSize(32).build();
    private static final QuantArgs HQQ = QuantArgs.builder(QuantFormat.INT_HQQ).bits(4).groupSize(64).build();
    private static final QuantArgs FP8 = QuantArgs.builder(QuantFormat.FP8).build();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A tensor of a dtype and shape whose data must not be read. */
    private static final class Shaped implements RowSource {
        private final StDType dtype;
        private final long[] shape;

        Shaped(StDType dtype, long... shape) {
            this.dtype = dtype;
            this.shape = shape.clone();
        }

        @Override
        public String describe() {
            return "shaped " + dtype + Arrays.toString(shape);
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(dtype);
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        @Override
        public long rows() {
            long r = 1;
            for (int i = 0; i < shape.length - 1; i++) {
                r *= shape[i];
            }
            return r;
        }

        @Override
        public long rowBytes() {
            return format().rowBytes(shape.length == 0 ? 1 : shape[shape.length - 1]);
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            throw new IOException(describe() + " is not read by this test");
        }
    }

    /** An index of the given names; each value is a {@link RowSource} or, for a name alone, an F32 [4] tensor. */
    private static HfTensorIndex index(Object... namesAndSources) {
        Map<String, RowSource> m = new LinkedHashMap<>();
        for (int i = 0; i < namesAndSources.length; i++) {
            String name = (String) namesAndSources[i];
            if (i + 1 < namesAndSources.length && namesAndSources[i + 1] instanceof RowSource) {
                m.put(name, (RowSource) namesAndSources[++i]);
            } else {
                m.put(name, new Shaped(StDType.F32, 4));
            }
        }
        return HfTensorIndex.of(m);
    }

    private static Map<String, List<String>> modules(HfTensorIndex in, Function<String, QuantArgs> argsFor,
                                                     boolean dsv4) {
        return QuantModuleIndex.of(in, argsFor, dsv4).modules();
    }

    private static Map<String, List<String>> expect(Object... moduleAndNames) {
        Map<String, List<String>> m = new LinkedHashMap<>();
        for (int i = 0; i < moduleAndNames.length; i += 2) {
            m.put((String) moduleAndNames[i], Arrays.asList((String[]) moduleAndNames[i + 1]));
        }
        return m;
    }

    private HfConfig config(String quantizationConfig) {
        String json = "{'architectures':['LlamaForCausalLM'],'hidden_size':64,'num_hidden_layers':2,"
                + "'num_attention_heads':4,'num_key_value_heads':2"
                + (quantizationConfig == null ? "" : ",'quantization_config':" + quantizationConfig) + "}";
        return HfConfigLoader.fromJson(tmp.getRoot().toPath(), json.replace('\'', '"'));
    }

    private static HfQuantRewriter.Result rewrite(HfConfig cfg, HfTensorIndex in, ActivationQuant mode)
            throws IOException {
        HfLoadContext ctx = HfLoadContext.builder(cfg, TestOnlyLlamaEntry.entry(), Arch.LLAMA).index(in).build();
        return HfQuantRewriter.rewrite(HfQuantConfig.of(cfg), ctx, in, ModelSourceOptions.defaults(), mode);
    }

    private static final String FP8_CONFIG = "{'quant_method':'fp8','activation_scheme':'dynamic'}";

    // ------------------------------------------------------------------ module grouping

    @Test
    public void dottedSuffixesGroupIntoModules() {
        HfTensorIndex in = index(
                "a.q_proj.qweight", "a.q_proj.qzeros", "a.q_proj.scales", "a.q_proj.g_idx",
                "a.k_proj.weight", "a.k_proj.weight.absmax", "a.k_proj.weight.quant_map",
                "a.k_proj.weight.nested_absmax", "a.k_proj.weight.quant_state.bitsandbytes__nf4",
                "a.v_proj.weight_packed", "a.v_proj.weight_scale", "a.v_proj.weight_scale_2",
                "a.v_proj.weight_global_scale",
                "a.o_proj.weight", "a.o_proj.weight_scale_inv",
                "a.norm.weight");
        QuantModuleIndex mods = QuantModuleIndex.of(in, m -> GPTQ, false);
        assertEquals(expect(
                "a.q_proj", new String[]{"a.q_proj.qweight", "a.q_proj.qzeros", "a.q_proj.scales", "a.q_proj.g_idx"},
                "a.k_proj", new String[]{"a.k_proj.weight.absmax", "a.k_proj.weight.quant_map",
                        "a.k_proj.weight.nested_absmax", "a.k_proj.weight.quant_state.bitsandbytes__nf4"},
                "a.v_proj", new String[]{"a.v_proj.weight_packed", "a.v_proj.weight_scale",
                        "a.v_proj.weight_scale_2", "a.v_proj.weight_global_scale"},
                "a.o_proj", new String[]{"a.o_proj.weight_scale_inv"}), mods.modules());
        assertFalse("weights are not auxiliary", mods.auxiliary().contains("a.k_proj.weight"));
        assertFalse(mods.auxiliary().contains("a.norm.weight"));
        assertEquals(13, mods.auxiliary().size());
    }

    @Test
    public void underscoreSuffixesNeedAQuantizedParameter() {
        HfTensorIndex in = index("m.experts.gate_up_proj_blocks", "m.experts.gate_up_proj_scales",
                "m.experts.gate_up_proj_bias");
        assertEquals(expect("m.experts.gate_up_proj", new String[]{"m.experts.gate_up_proj_blocks",
                "m.experts.gate_up_proj_scales"}), modules(in, m -> m.endsWith("gate_up_proj") ? GPTQ : null, false));
        assertEquals("an unquantized parameter's underscore names are its own tensors", Collections.emptyMap(),
                modules(in, m -> null, false));
    }

    @Test
    public void hqqMetadataSuffixesBelongOnlyToHqqModules() {
        HfTensorIndex in = index("h.q_proj.W_q", "h.q_proj.scale", "h.q_proj.zero", "h.q_proj.meta",
                "h.q_proj.shape", "h.q_proj.nbits", "h.q_proj.axis", "h.q_proj.bias",
                "n.norm.scale", "g.q_proj.qweight", "g.q_proj.zero");
        Map<String, List<String>> m = modules(in, name -> name.startsWith("h.") ? HQQ
                : name.startsWith("g.") ? GPTQ : null, false);
        assertEquals(expect(
                "h.q_proj", new String[]{"h.q_proj.W_q", "h.q_proj.scale", "h.q_proj.zero", "h.q_proj.meta",
                        "h.q_proj.shape", "h.q_proj.nbits", "h.q_proj.axis"},
                "g.q_proj", new String[]{"g.q_proj.qweight"}), m);
    }

    @Test
    public void aFusedScaleInvMakesItsParameterAModule() {
        HfTensorIndex in = index("e.experts.gate_up_proj", new Shaped(StDType.F8_E4M3, 2, 8, 4),
                "e.experts.gate_up_proj_scale_inv", "e.experts.down_proj_scale_inv");
        assertEquals("a scale_inv without its parameter is not grouped",
                expect("e.experts.gate_up_proj", new String[]{"e.experts.gate_up_proj_scale_inv"}),
                modules(in, m -> FP8, false));
        assertEquals(Collections.emptyMap(), modules(in, m -> null, false));
    }

    @Test
    public void deepseekV4ScaleNeedsAnF8Weight() {
        String m = "layers.0.attn.wq_a";
        HfTensorIndex f8 = index(m + ".weight", new Shaped(StDType.F8_E4M3, 256, 128), m + ".scale");
        assertEquals(expect(m, new String[]{m + ".scale"}), modules(f8, n -> FP8, true));
        assertEquals("only DeepSeek V4 reads .scale as a weight scale", Collections.emptyMap(),
                modules(f8, n -> FP8, false));
        HfTensorIndex bf16 = index(m + ".weight", new Shaped(StDType.BF16, 256, 128), m + ".scale");
        assertEquals(Collections.emptyMap(), modules(bf16, n -> FP8, true));
    }

    // ------------------------------------------------------------------ the rewrite

    @Test
    public void anUnconsumedAuxiliaryTensorIsAnError() {
        HfConfig cfg = config(FP8_CONFIG);
        HfTensorIndex in = index(L0 + "self_attn.q_proj.weight", new Shaped(StDType.BF16, 64, 64),
                L0 + "self_attn.q_proj.weight_zero_point", new Shaped(StDType.I8, 64));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> rewrite(cfg, in, ActivationQuant.WEIGHT_ONLY));
        assertEquals("unconsumed quantization tensor " + L0 + "self_attn.q_proj.weight_zero_point (fp8)",
                e.getMessage());
    }

    @Test
    public void kvCacheScalesAreTakenOutForAQuantizedMethod() throws IOException {
        HfConfig cfg = config(FP8_CONFIG);
        HfTensorIndex in = index(L0 + "self_attn.k_proj.weight", new Shaped(StDType.BF16, 32, 64),
                L0 + "self_attn.k_proj.output_scale", RopeTensors.fromFloats(new long[]{}, new float[]{0.5f}),
                "model.layers.1.self_attn.kv_scale", RopeTensors.fromFloats(new long[]{1}, new float[]{0.25f}));
        HfQuantRewriter.Result r = rewrite(cfg, in, ActivationQuant.CHECKPOINT);
        List<String> names = new ArrayList<>();
        for (KvScaleCollector.Scale s : r.kvScales()) {
            names.add(s.ggufName());
        }
        assertEquals(Arrays.asList("blk.0.attn_k.kv_scale", "blk.1.attn_k.kv_scale", "blk.1.attn_v.kv_scale"),
                names);
        assertEquals("a KV projection's output scale is no output activation", Collections.singletonList(
                L0 + "self_attn.k_proj.weight"), r.index().names());
    }

    @Test
    public void anUnquantizedCheckpointIsReturnedAsIs() throws IOException {
        HfConfig cfg = config(null);
        HfTensorIndex in = index(L0 + "self_attn.q_proj.weight", new Shaped(StDType.BF16, 64, 64));
        HfQuantRewriter.Result r = rewrite(cfg, in, ActivationQuant.CHECKPOINT);
        assertSame(in, r.index());
        assertTrue(r.moduleArgs().isEmpty());
        assertTrue(r.kvScales().isEmpty());
        assertFalse(r.hadamard().isPresent());
    }

    @Test
    public void checkpointModeRefusesAnUnemulatedActivationScheme() {
        HfConfig cfg = config("{'quant_method':'fp8','activation_scheme':'dynamic','weight_block_size':[64,64]}");
        HfTensorIndex in = index(L0 + "self_attn.q_proj.weight", new Shaped(StDType.F8_E4M3, 64, 64),
                L0 + "self_attn.q_proj.weight_scale_inv", new Shaped(StDType.F32, 1, 1));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> rewrite(cfg, in, ActivationQuant.CHECKPOINT));
        assertEquals("dynamic FP8 activations in groups of 64 is not emulated; use --act-quant weight-only",
                e.getMessage());
    }

    @Test
    public void checkpointModeRefusesAnOutputActivationScheme() {
        HfConfig cfg = config("{'quant_method':'compressed-tensors','format':'float-quantized',"
                + "'config_groups':{'g':{'targets':['re:.*q_proj$'],"
                + "'weights':{'num_bits':8,'type':'float','strategy':'tensor','symmetric':true,'dynamic':false},"
                + "'output_activations':{'num_bits':8,'type':'float','strategy':'tensor','dynamic':false}}}}");
        HfTensorIndex in = index(L0 + "self_attn.q_proj.weight", new Shaped(StDType.F8_E4M3, 64, 64),
                L0 + "self_attn.q_proj.weight_scale", new Shaped(StDType.F32, 1));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> rewrite(cfg, in, ActivationQuant.CHECKPOINT));
        assertTrue(e.getMessage(), e.getMessage().startsWith(
                "output activation quantization (compressed-tensors output_activations"));
        assertTrue(e.getMessage(), e.getMessage().endsWith("is not emulated; use --act-quant weight-only"));
    }

    @Test
    public void anOutputScaleIsRefusedUnderCheckpointAndDroppedUnderWeightOnly() throws IOException {
        HfConfig cfg = config(FP8_CONFIG);
        HfTensorIndex in = index(L0 + "self_attn.o_proj.weight", new Shaped(StDType.BF16, 64, 64),
                L0 + "self_attn.o_proj.output_scale", RopeTensors.fromFloats(new long[]{}, new float[]{2f}));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> rewrite(cfg, in, ActivationQuant.CHECKPOINT));
        assertEquals("output activation quantization (fp8 output_scale) is not emulated; use --act-quant"
                + " weight-only", e.getMessage());
        HfQuantRewriter.Result r = rewrite(cfg, in, ActivationQuant.WEIGHT_ONLY);
        assertEquals(Collections.singletonList(L0 + "self_attn.o_proj.weight"), r.index().names());
        assertTrue(r.kvScales().isEmpty());
        assertEquals(Collections.singletonList(L0 + "self_attn.o_proj"), new ArrayList<>(r.moduleArgs().keySet()));
        assertEquals(QuantFormat.FP8, r.moduleArgs().get(L0 + "self_attn.o_proj").format());
    }

    @Test
    public void anOnlineHadamardTransformOfBlock2048BecomesThePlan() throws IOException {
        HfConfig cfg = config("{'quant_method':'compressed-tensors','transform_config':{'config_groups':{"
                + "'u':{'type':'hadamard','apply':[{'targets':['Linear'],'location':'input'}]}}}}");
        String down = L0 + "mlp.down_proj";
        HfTensorIndex in = index(down + ".weight", new Shaped(StDType.BF16, 4, 2048),
                down + ".u_input.weight", new Shaped(StDType.BF16, 2048, 2048));
        HfQuantRewriter.Result r = rewrite(cfg, in, ActivationQuant.WEIGHT_ONLY);
        HadamardPlan p = r.hadamard().orElseThrow(AssertionError::new);
        assertEquals(2048, p.blockSize());
        assertEquals("identity", p.signMode());
        assertEquals(Collections.singletonList(down + ".weight"), p.weightNames());
        assertEquals("the transform matrix is consumed", Collections.singletonList(down + ".weight"),
                r.index().names());
    }
}
