package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs.Strategy;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfQuantConfig}: the source order, every method's rules and ignore list, the method overrides, every
 * ModelOpt {@code quant_algo}, the online and rejected methods, and the named rejections.
 */
public class HfQuantConfigTest {

    private static final String Q = "model.layers.0.self_attn.q_proj";
    private static final String DOWN = "model.layers.0.mlp.down_proj";
    private static final String EXPERT = "model.layers.3.mlp.experts.7.w1";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HfQuantConfig quant(String members) throws IOException {
        return HfQuantConfig.of(config(members));
    }

    private HfConfig config(String members) {
        return HfConfigLoader.fromJson(tmp.getRoot().toPath(), HfConfigLoaderTest.json(
                "{'architectures':['A']" + (members.isEmpty() ? "" : "," + members) + "}"));
    }

    private HfQuantConfig qc(String quantizationConfig) throws IOException {
        return quant("'quantization_config':" + quantizationConfig);
    }

    private QuantArgs args(HfQuantConfig q, String module) {
        return q.argsFor(module).orElseThrow(() -> new AssertionError(module + " is not quantized by " + q));
    }

    private void assertRejected(String quantizationConfig, String messagePart) {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> qc(quantizationConfig));
        assertTrue(e.getMessage(), e.getMessage().contains(messagePart));
    }

    // ------------------------------------------------------------------ sources

    @Test
    public void unquantizedConfigurations() throws IOException {
        for (String members : new String[]{"", "'quantization_config':{}", "'quantization_config':{'bits':4}",
                "'quantization':{'note':'x'}"}) {
            HfQuantConfig q = quant(members);
            assertEquals(members, HfQuantMethod.NONE, q.method());
            assertFalse(q.isQuantized());
            assertTrue(q.rules().isEmpty());
            assertFalse(q.argsFor(Q).isPresent());
            assertEquals("", q.methodName());
        }
        assertEquals(HfQuantMethod.NONE, HfQuantConfig.none().method());
    }

    @Test
    public void onlyTheOnlineShorthandsQuantizeAtLoadByDefault() throws IOException {
        HfQuantConfig channel = quant("'quantization_config':{'quant_method':'fp8_per_channel'}");
        assertEquals(HfQuantMethod.ONLINE, channel.method());
        assertFalse(channel.isQuantized());
        assertEquals(ModelSourceOptions.OnlineQuant.FP8_PER_CHANNEL, channel.onlineTarget());
        assertEquals(ModelSourceOptions.OnlineQuant.NVFP4_PER_TOKEN,
                quant("'quantization_config':{'quant_method':'nvfp4_per_token'}").onlineTarget());
        assertNull("bare online needs linear / moe specs",
                quant("'quantization_config':{'quant_method':'online'}").onlineTarget());
        assertNull("a serialized FP8 checkpoint keeps its float modules", quant("'quantization_config':"
                + "{'quant_method':'fp8','activation_scheme':'dynamic','weight_block_size':[128,128]}")
                .onlineTarget());
        assertNull(quant("").onlineTarget());
    }

    @Test
    public void bareOnlineCarriesItsLayerSelection() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'online','moe':'mxfp4','ignore':['re:.*\\\\.gate$']}");
        assertEquals(HfQuantMethod.ONLINE, q.method());
        assertFalse(q.isQuantized());
        assertNotNull(q.onlineConfig());
        assertTrue(q.onlineConfig().quantizes(OnlineQuantConfig.LayerKind.MOE));
        assertFalse(q.onlineConfig().quantizes(OnlineQuantConfig.LayerKind.LINEAR));
        assertEquals(List.of("re:.*\\.gate$"), q.onlineConfig().ignore());
        assertTrue(q.toString(), q.toString().contains("online=OnlineQuantConfig{"));
        assertNull("a shorthand carries no layer selection", qc("{'quant_method':'mxfp8'}").onlineConfig());
        assertNull(qc("{'quant_method':'fp8_per_block'}").onlineConfig());
        assertNull(quant("").onlineConfig());
        assertRejected("{'quant_method':'online','linear':{'weight':'fp8_per_block_static','activation':"
                + "'fp8_per_token'}}", "config.json: quantization_config: linear.activation=fp8_per_token");
        assertRejected("{'quant_method':'online','weight_block_size':[128,128]}", "unknown member"
                + " 'weight_block_size'");
    }

    @Test
    public void sourcesAreSearchedInOrder() throws IOException {
        assertEquals(HfQuantMethod.AWQ, quant("'compression_config':{'quant_method':'awq','bits':4,'group_size':128}")
                .method());
        assertEquals(HfQuantMethod.FP8, quant("'text_config':{'quantization_config':{'quant_method':'fp8',"
                + "'activation_scheme':'dynamic'}}").method());
        assertEquals(HfQuantMethod.FP8, quant("'quantization_config':{'quant_method':'fp8','activation_scheme':"
                + "'dynamic'},'compression_config':{'quant_method':'awq'}").method());
        HfConfigLoaderTest.write(tmp.getRoot().toPath().resolve("hf_quant_config.json"),
                "{'producer':{'name':'modelopt','version':'0.19.0'},'quantization':{'quant_algo':'FP8',"
                        + "'kv_cache_quant_algo':null,'exclude_modules':['lm_head','*.mlp.gate']}}");
        HfQuantConfig modelOpt = quant("");
        assertEquals(HfQuantMethod.MODELOPT, modelOpt.method());
        assertEquals("FP8", modelOpt.modelOptAlgo().get());
        assertEquals("hf_quant_config.json", modelOpt.view().source());
        assertEquals(ActQuantScheme.FP8_STATIC, args(modelOpt, Q).actScheme());
        assertEquals(List.of("lm_head", "*.mlp.gate"), modelOpt.ignoreList());
        assertTrue(modelOpt.isIgnored("lm_head"));
        assertTrue(modelOpt.isIgnored("model.layers.0.mlp.gate"));
        assertFalse(modelOpt.isIgnored(Q));
        assertEquals(HfQuantMethod.FP8, quant("'quantization_config':{'quant_method':'fp8','activation_scheme':"
                + "'static'}").method());
        HfConfigLoaderTest.write(tmp.getRoot().toPath().resolve("hf_quant_config.json"),
                "{'producer':{'name':'other'},'quantization':{'quant_algo':'FP8'}}");
        assertEquals(HfQuantMethod.NONE, quant("").method());
    }

    @Test
    public void mistralNativeQuantization() throws IOException {
        Path d = tmp.newFolder().toPath();
        HfConfigLoaderTest.write(d.resolve("params.json"), "{'dim':8,'quantization':{'qformat_weight':'fp8_e4m3',"
                + "'qscheme_act':'TENSOR'}}");
        HfQuantConfig q = HfQuantConfig.of(HfConfigLoader.load(d, (Map<String, Object>) null));
        assertEquals(HfQuantMethod.FP8, q.method());
        assertEquals(ActQuantScheme.FP8_STATIC, args(q, Q).actScheme());
        HfConfigLoaderTest.write(d.resolve("params.json"), "{'dim':8,'quantization':{'qformat_weight':'fp8_e4m3',"
                + "'qscheme_act':'NO_SCALES'}}");
        assertEquals(ActQuantScheme.FP8_TOKEN,
                args(HfQuantConfig.of(HfConfigLoader.load(d, (Map<String, Object>) null)), Q).actScheme());
        HfConfigLoaderTest.write(d.resolve("params.json"), "{'dim':8,'quantization':{'qformat_weight':'int4'}}");
        HfConfig cfg = HfConfigLoader.load(d, (Map<String, Object>) null);
        assertThrows(GgufFormatException.class, () -> HfQuantConfig.of(cfg));
    }

    @Test
    public void mistralBlockQuantizationKeepsTheBlockShape() throws IOException {
        Path d = tmp.newFolder().toPath();
        HfConfigLoaderTest.write(d.resolve("params.json"), "{'dim':8,'quantization':{'qformat_weight':"
                + "'fp8_e4m3','qscheme_weight':'BLOCK','block_size_out':128,'block_size_in':128}}");
        QuantArgs a = args(HfQuantConfig.of(HfConfigLoader.load(d, (Map<String, Object>) null)), Q);
        assertEquals(128, a.blockN());
        assertEquals(128, a.blockK());
        assertEquals(ActQuantScheme.FP8_GROUP128, a.actScheme());
    }

    @Test
    public void mistralBlockQuantizationNeedsBothBlockSizes() throws IOException {
        Path d = tmp.newFolder().toPath();
        HfConfigLoaderTest.write(d.resolve("params.json"), "{'dim':8,'quantization':{'qformat_weight':"
                + "'fp8_e4m3','qscheme_weight':'BLOCK','block_size_out':128}}");
        HfConfig cfg = HfConfigLoader.load(d, (Map<String, Object>) null);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfQuantConfig.of(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains("block_size_in"));
    }

    // ------------------------------------------------------------------ names

    @Test
    public void unknownAndRejectedMethodsGiveTheirNamedErrors() {
        assertRejected("{'quant_method':'foo'}", "unsupported quant_method 'foo'");
        assertRejected("{'quant_method':'torchao'}", "quant_type is missing");
        assertRejected("{'quant_method':'torchao','quant_type':{'default':{'_type':'MysteryConfig'}}}",
                "quant_type 'MysteryConfig'");
        assertRejected("{'quant_method':'modelslim'}", "unsupported quant_method 'modelslim': Ascend");
        for (String removed : new String[]{"aqlm", "eetq", "vptq", "spqr", "higgs", "quanto"}) {
            assertRejected("{'quant_method':'" + removed + "'}", "unsupported quant_method '" + removed + "'");
        }
    }

    @Test
    public void torchaoConfigurationsMapOntoTheCheckpointLayouts() throws IOException {
        QuantArgs int4 = args(qc(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'group_size':64}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, int4.format());
        assertEquals(64, int4.groupSize());
        assertEquals(4, int4.bits());

        QuantArgs int4Tile = args(qc(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'group_size':128,"
                + "'int4_packing_format':{'_type':'Int4PackingFormat','_data':'TILE_PACKED_TO_4D'}}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, int4Tile.format());

        QuantArgs int4Preshuffled = args(qc(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'group_size':64,"
                + "'int4_packing_format':{'_type':'Int4PackingFormat','_data':'PRESHUFFLED'}}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, int4Preshuffled.format());
        assertEquals(64, int4Preshuffled.groupSize());

        QuantArgs int4PlainInt32 = args(qc(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'group_size':32,"
                + "'int4_packing_format':{'_type':'Int4PackingFormat','_data':'PLAIN_INT32'}}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, int4PlainInt32.format());
        assertEquals(32, int4PlainInt32.groupSize());

        QuantArgs int4Ntile16 = args(qc(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'group_size':128,"
                + "'int4_packing_format':{'_type':'Int4PackingFormat','_data':'TILE_PACKED_TO_4D'},"
                + "'int4_tile_packed_ntile':16}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, int4Ntile16.format());

        QuantArgs fp8Int4 = args(qc(torchao("{'_type':'Float8DynamicActivationInt4WeightConfig','_data':{"
                + "'int4_packing_format':{'_type':'Int4PackingFormat','_data':'PRESHUFFLED'}}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, fp8Int4.format());
        assertEquals(128, fp8Int4.groupSize());
        assertEquals(ActQuantScheme.FP8_TOKEN, fp8Int4.actScheme());

        QuantArgs int8Static = args(qc(torchao("{'_type':'Int8StaticActivationInt8WeightConfig','_data':{"
                + "'granularity':{'_type':'PerRow','_data':{'dim':-1}},"
                + "'act_mapping_type':{'_type':'MappingType','_data':'ASYMMETRIC'}}}")), Q);
        assertEquals(QuantFormat.INT8_CHANNEL, int8Static.format());
        assertEquals(ActQuantScheme.INT8_STATIC_ASYM, int8Static.actScheme());

        QuantArgs int8StaticSym = args(qc(torchao("{'_type':'Int8StaticActivationInt8WeightConfig',"
                + "'_data':{}}")), Q);
        assertEquals(ActQuantScheme.INT8_STATIC, int8StaticSym.actScheme());

        QuantArgs int8StaticReduced = args(qc(torchao("{'_type':'Int8StaticActivationInt8WeightConfig',"
                + "'_data':{'reduce_range':true}}")), Q);
        assertNull(int8StaticReduced.actScheme());
        assertTrue(int8StaticReduced.actUnemulated(),
                int8StaticReduced.actUnemulated().contains("reduced-range"));

        QuantArgs intx4 = args(qc(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'weight_dtype':"
                + "{'_type':'torch.dtype','_data':'int4'},'granularity':"
                + "{'_type':'PerGroup','_data':{'group_size':32}}}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, intx4.format());
        assertEquals(32, intx4.groupSize());

        QuantArgs intx8 = args(qc(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'weight_dtype':"
                + "{'_type':'torch.dtype','_data':'int8'}}}")), Q);
        assertEquals(QuantFormat.INT8_CHANNEL, intx8.format());

        for (int bits : new int[]{1, 2, 3, 5, 6, 7}) {
            QuantArgs narrow = args(qc(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'weight_dtype':"
                    + "{'_type':'torch.dtype','_data':'int" + bits + "'},'granularity':"
                    + "{'_type':'PerGroup','_data':{'group_size':16}}}}")), Q);
            assertEquals("int" + bits, QuantFormat.INT8_CHANNEL, narrow.format());
            assertEquals("int" + bits, 16, narrow.groupSize());
        }

        QuantArgs intxDynamic = args(qc(torchao("{'_type':'Int8DynamicActivationIntxWeightConfig','_data':{"
                + "'weight_dtype':{'_type':'torch.dtype','_data':'int4'},'weight_granularity':"
                + "{'_type':'PerGroup','_data':{'group_size':64}}}}")), Q);
        assertEquals(QuantFormat.INT4_TORCHAO, intxDynamic.format());
        assertEquals(64, intxDynamic.groupSize());
        assertEquals(ActQuantScheme.INT8_TOKEN, intxDynamic.actScheme());

        QuantArgs intxAsym = args(qc(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'weight_dtype':"
                + "{'_type':'torch.dtype','_data':'int8'},'mapping_type':"
                + "{'_type':'MappingType','_data':'ASYMMETRIC'}}}")), Q);
        assertEquals(QuantFormat.INT8_CHANNEL, intxAsym.format());

        QuantArgs int8 = args(qc(torchao("{'_type':'Int8WeightOnlyConfig','_data':{"
                + "'granularity':{'_type':'PerRow','_data':{'dim':-1}}}}")), Q);
        assertEquals(QuantFormat.INT8_CHANNEL, int8.format());
        assertEquals(QuantArgs.CHANNEL, int8.groupSize());

        QuantArgs int8Dynamic = args(qc(torchao(
                "{'_type':'Int8DynamicActivationInt8WeightConfig','_data':{}}")), Q);
        assertEquals(QuantFormat.INT8_CHANNEL, int8Dynamic.format());
        assertEquals(ActQuantScheme.INT8_TOKEN, int8Dynamic.actScheme());
        assertNull(int8Dynamic.actUnemulated());

        QuantArgs int8DynamicRow = args(qc(torchao("{'_type':'Int8DynamicActivationInt8WeightConfig',"
                + "'_data':{'granularity':[{'_type':'PerRow','_data':{'dim':-1}},"
                + "{'_type':'PerRow','_data':{'dim':-1}}]}}")), Q);
        assertEquals(QuantArgs.CHANNEL, int8DynamicRow.groupSize());
        assertEquals(ActQuantScheme.INT8_TOKEN, int8DynamicRow.actScheme());

        QuantArgs int8Tensor = args(qc(torchao("{'_type':'Int8DynamicActivationInt8WeightConfig','_data':{"
                + "'granularity':{'_type':'PerTensor','_data':{}}}}")), Q);
        assertEquals(QuantArgs.UNGROUPED, int8Tensor.groupSize());
        assertNull(int8Tensor.actScheme());
        assertTrue(int8Tensor.actUnemulated(), int8Tensor.actUnemulated().contains("PerTensor"));

        QuantArgs float8 = args(qc(torchao("{'_type':'Float8WeightOnlyConfig','_data':{"
                + "'granularity':{'_type':'PerRow','_data':{'dim':-1}}}}")), Q);
        assertEquals(QuantFormat.FP8, float8.format());
        assertEquals(QuantArgs.CHANNEL, float8.groupSize());

        QuantArgs float8Group = args(qc(torchao("{'_type':'Float8WeightOnlyConfig','_data':{"
                + "'granularity':{'_type':'PerGroup','_data':{'group_size':128}}}}")), Q);
        assertEquals(Strategy.BLOCK, float8Group.strategy());
        assertEquals(128, float8Group.blockK());

        QuantArgs float8Dynamic = args(qc(torchao("{'_type':'Float8DynamicActivationFloat8WeightConfig',"
                + "'_data':{'granularity':[{'_type':'PerRow','_data':{'dim':-1}},"
                + "{'_type':'PerRow','_data':{'dim':-1}}]}}")), Q);
        assertEquals(QuantFormat.FP8, float8Dynamic.format());
        assertEquals(ActQuantScheme.FP8_TOKEN, float8Dynamic.actScheme());

        QuantArgs float8DynamicDefault = args(qc(torchao(
                "{'_type':'Float8DynamicActivationFloat8WeightConfig','_data':{}}")), Q);
        assertEquals(QuantFormat.FP8, float8DynamicDefault.format());
        assertEquals(QuantArgs.UNGROUPED, float8DynamicDefault.groupSize());
        assertNull(float8DynamicDefault.actScheme());
        assertTrue(float8DynamicDefault.actUnemulated(),
                float8DynamicDefault.actUnemulated().contains("per-tensor"));

        QuantArgs mxfp4 = args(qc(torchao("{'_type':'MXDynamicActivationMXWeightConfig','_data':{"
                + "'weight_dtype':{'_type':'torch.dtype','_data':'float4_e2m1fn_x2'}}}")), Q);
        assertEquals(QuantFormat.MXFP4_CT, mxfp4.format());
        assertEquals(ActQuantScheme.MXFP4, mxfp4.actScheme());
        assertEquals(32, mxfp4.groupSize());

        QuantArgs mxfp8 = args(qc(torchao("{'_type':'MXDynamicActivationMXWeightConfig','_data':{"
                + "'weight_dtype':{'_type':'torch.dtype','_data':'float8_e4m3fn'}}}")), Q);
        assertEquals(QuantFormat.MXFP8, mxfp8.format());
        assertEquals(ActQuantScheme.MXFP8, mxfp8.actScheme());

        QuantArgs mxfp8Default = args(qc(torchao(
                "{'_type':'MXDynamicActivationMXWeightConfig','_data':{}}")), Q);
        assertEquals(QuantFormat.MXFP8, mxfp8Default.format());

        QuantArgs nvfp4 = args(qc(torchao(
                "{'_type':'NVFP4DynamicActivationNVFP4WeightConfig','_data':{}}")), Q);
        assertEquals(QuantFormat.NVFP4_MODELOPT, nvfp4.format());
        assertEquals(16, nvfp4.groupSize());
        assertNull(nvfp4.actScheme());
        assertTrue(nvfp4.actUnemulated(), nvfp4.actUnemulated().contains("dynamic per-tensor"));

        QuantArgs nvfp4Static = args(qc(torchao("{'_type':'NVFP4DynamicActivationNVFP4WeightConfig',"
                + "'_data':{'use_dynamic_per_tensor_scale':false}}")), Q);
        assertEquals(ActQuantScheme.NVFP4, nvfp4Static.actScheme());
    }

    @Test
    public void torchaoUnsupportedConfigurationsAreRefusedByName() {
        assertRejected(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'int4_packing_format':"
                + "{'_type':'Int4PackingFormat','_data':'MYSTERY'}}}"), "MYSTERY");
        assertRejected(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'int4_packing_format':"
                + "{'_type':'Int4PackingFormat','_data':'TILE_PACKED_TO_4D'},"
                + "'int4_tile_packed_ntile':4}}"), "4");
        assertRejected(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'weight_dtype':"
                + "{'_type':'torch.dtype','_data':'int9'}}}"), "int9");
        assertRejected(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'weight_dtype':"
                + "{'_type':'torch.dtype','_data':'int0'}}}"), "int0");
        assertRejected(torchao("{'_type':'IntxWeightOnlyConfig','_data':{'intx_packing_format':"
                + "{'_type':'IntxPackingFormat','_data':'OPAQUE_TORCHAO_AUTO'}}}"), "OPAQUE_TORCHAO_AUTO");
        assertRejected(torchao("{'_type':'MXDynamicActivationMXWeightConfig','_data':{'weight_dtype':"
                + "{'_type':'torch.dtype','_data':'float8_e5m2'}}}"), "float8_e5m2");
        assertRejected(torchao("{'_type':'Float8DynamicActivationFloat8WeightConfig','_data':{"
                + "'packing_format':{'_type':'Float8PackingFormat','_data':'SPARSE'}}}"), "SPARSE");
        assertRejected(torchao("{'_type':'UnknownConfig','_data':{}}"), "has no NeuralFree mapping");
    }

    @Test
    public void torchaoPerModuleConfigurationsBecomeRules() throws IOException {
        HfQuantConfig q = qc(torchao("{'_type':'Int4WeightOnlyConfig','_data':{'module_fqn_to_config':{"
                + "'" + Q + "':{'_type':'Int8WeightOnlyConfig','_data':{}},"
                + "'_default':{'_type':'Float8WeightOnlyConfig','_data':{}},"
                + "'lm_head':null}}}"));
        assertEquals(QuantFormat.INT8_CHANNEL, args(q, Q).format());
        assertEquals(QuantFormat.FP8, args(q, DOWN).format());
        assertFalse(q.argsFor("lm_head").isPresent());
    }

    /** A {@code quant_method: torchao} config whose {@code default} is {@code type}. */
    private static String torchao(String type) {
        return "{'quant_method':'torchao','quant_type':{'default':" + type + "}}";
    }

    @Test
    public void onlineMethodsLoadAsStored() throws IOException {
        for (String name : new String[]{"experts_int8", "quark_int4fp8_moe", "mxfp_w4a8", "ptpc_fp8"}) {
            HfQuantConfig q = qc("{'quant_method':'" + name + "'}");
            assertEquals(name, HfQuantMethod.ONLINE, q.method());
            assertFalse(q.isQuantized());
            assertTrue(q.rules().isEmpty());
        }
        assertEquals(HfQuantMethod.ONLINE, qc("{'quant_method':'bitnet','quantization_mode':'online'}").method());
    }

    @Test
    public void everyAliasRowResolves() throws IOException {
        for (Map.Entry<String, HfQuantMethod> e : HfQuantAliases.table().entrySet()) {
            HfQuantAliases.Resolution r = HfQuantAliases.resolve(e.getKey().toUpperCase(Locale.ROOT));
            assertEquals(e.getKey(), e.getValue(), r.method());
            assertEquals(e.getKey(), r.name());
        }
        assertEquals(HfQuantMethod.GPTQ, qc("{'quant_method':'gptq_marlin','bits':4,'group_size':128}").method());
        assertEquals("gptq_marlin", qc("{'quant_method':'gptq_marlin','bits':4,'group_size':128}").methodName());
        assertEquals(HfQuantMethod.AWQ, qc("{'quant_method':'awq_marlin','bits':4,'group_size':128}").method());
        assertEquals(HfQuantMethod.MODELOPT, qc("{'quant_method':'modelopt_fp4','quant_algo':'NVFP4'}").method());
        assertEquals(HfQuantMethod.MXFP4, qc("{'quant_method':'gpt_oss_mxfp4'}").method());
        assertEquals(HfQuantMethod.COMPRESSED_TENSORS, qc("{'quant_method':'compressed_tensors'}").method());
        assertEquals(HfQuantMethod.W8A8_INT8, qc("{'quant_method':'auto-round-int8'}").method());
    }

    // ------------------------------------------------------------------ fp8 family

    @Test
    public void fp8TensorAndBlock() throws IOException {
        HfQuantConfig dynamic = qc("{'quant_method':'fp8','activation_scheme':'dynamic',"
                + "'ignored_layers':['lm_head','re:.*vision.*']}");
        QuantArgs a = args(dynamic, Q);
        assertEquals(QuantFormat.FP8, a.format());
        assertEquals(Strategy.TENSOR, a.strategy());
        assertEquals(ActQuantScheme.FP8_TOKEN, a.actScheme());
        assertFalse(a.actStatic());
        assertTrue(dynamic.isIgnored("lm_head"));
        assertTrue(dynamic.isIgnored("model.vision_tower.blocks.0.attn"));
        assertFalse(dynamic.argsFor("lm_head").isPresent());
        assertTrue(args(qc("{'quant_method':'fp8','activation_scheme':'static'}"), Q).actStatic());

        QuantArgs block = args(qc("{'quant_method':'fp8','activation_scheme':'dynamic',"
                + "'weight_block_size':[128,128],'modules_to_not_convert':['lm_head']}"), Q);
        assertEquals(Strategy.BLOCK, block.strategy());
        assertEquals(128, block.blockN());
        assertEquals(128, block.blockK());
        assertEquals(ActQuantScheme.FP8_GROUP128, block.actScheme());
        QuantArgs odd = args(qc("{'quant_method':'fp8','activation_scheme':'dynamic','weight_block_size':[64,64]}"),
                Q);
        assertNull(odd.actScheme());
        assertNotNull(odd.actUnemulated());
        assertRejected("{'quant_method':'fp8','activation_scheme':'static','weight_block_size':[128,128]}",
                "requires activation_scheme 'dynamic'");
        assertRejected("{'quant_method':'fp8'}", "activation_scheme");
        HfQuantConfig fallback = qc("{'quant_method':'fp8','activation_scheme':'dynamic','ignored_layers':[],"
                + "'modules_to_not_convert':['lm_head']}");
        assertTrue(fallback.isIgnored("lm_head"));
        assertFalse(fallback.isIgnored("model.lm_head"));
    }

    @Test
    public void deepSeekV4OverrideAndExperts() throws IOException {
        HfQuantConfig q = quant("'model_type':'deepseek_v4','quantization_config':{'quant_method':'fp8',"
                + "'activation_scheme':'dynamic','weight_block_size':[128,128]}");
        assertEquals(HfQuantMethod.DEEPSEEK_V4_FP8, q.method());
        assertEquals(QuantFormat.MXFP4_DSV4, args(q, EXPERT).format());
        assertEquals(QuantFormat.FP8, args(q, Q).format());
        assertEquals(ActQuantScheme.FP8_GROUP128, args(q, Q).actScheme());
        HfQuantConfig fp8Experts = quant("'model_type':'deepseek_v4','expert_dtype':'fp8',"
                + "'quantization_config':{'quant_method':'deepseek_v4_fp8','activation_scheme':'dynamic'}");
        assertEquals(Strategy.BLOCK, args(fp8Experts, EXPERT).strategy());
        HfQuantConfig nvfp4 = qc("{'quant_method':'deepseek_v4_fp8','moe_quant_algo':'NVFP4'}");
        assertEquals(QuantFormat.NVFP4_MODELOPT, args(nvfp4, EXPERT).format());
        HfQuantConfig fp8OverNvfp4 = quant("'expert_dtype':'fp8','quantization_config':"
                + "{'quant_method':'deepseek_v4_fp8','moe_quant_algo':'NVFP4','modules_to_not_convert':['lm_head']}");
        assertEquals(QuantFormat.FP8, args(fp8OverNvfp4, EXPERT).format());
        assertTrue(fp8OverNvfp4.isIgnored("lm_head"));
        assertThrows(GgufFormatException.class, () -> quant("'expert_dtype':'int4','quantization_config':"
                + "{'quant_method':'deepseek_v4_fp8'}"));
    }

    @Test
    public void channelAndBlockMethods() throws IOException {
        QuantArgs fbgemm = args(qc("{'quant_method':'fbgemm_fp8','modules_to_not_convert':['lm_head']}"), Q);
        assertEquals(Strategy.CHANNEL, fbgemm.strategy());
        assertEquals(ActQuantScheme.FP8_TOKEN, fbgemm.actScheme());
        QuantArgs int8 = args(qc("{'quant_method':'w8a8_int8','ignore':['lm_head']}"), Q);
        assertEquals(QuantFormat.INT8_CHANNEL, int8.format());
        assertEquals(ActQuantScheme.INT8_TOKEN, int8.actScheme());
        assertEquals(QuantFormat.FP8, args(qc("{'quant_method':'w8a8_fp8'}"), Q).format());
        QuantArgs blockInt8 = args(qc("{'quant_method':'blockwise_int8','activation_scheme':'dynamic',"
                + "'weight_block_size':[128,128]}"), Q);
        assertEquals(QuantFormat.INT8_BLOCK, blockInt8.format());
        assertEquals(ActQuantScheme.INT8_GROUP128, blockInt8.actScheme());
        assertRejected("{'quant_method':'blockwise_int8','activation_scheme':'static','weight_block_size':[128,128]}",
                "dynamic");
        assertRejected("{'quant_method':'blockwise_int8','activation_scheme':'dynamic'}", "weight_block_size");
        HfQuantConfig mxfp8 = qc("{'quant_method':'mxfp8','ignored_layers':['lm_head','*.mlp.gate']}");
        assertEquals(QuantFormat.MXFP8, args(mxfp8, Q).format());
        assertEquals(ActQuantScheme.MXFP8, args(mxfp8, Q).actScheme());
        assertTrue(mxfp8.isIgnored("model.lm_head"));
        assertTrue(mxfp8.isIgnored("model.layers.0.mlp.gate"));
        HfQuantConfig mxfp8ModelOpt = qc("{'quant_method':'mxfp8','quant_algo':'MXFP8','exclude_modules':['q_proj']}");
        assertEquals("MXFP8", mxfp8ModelOpt.modelOptAlgo().get());
        assertTrue(mxfp8ModelOpt.isIgnored(Q));
        assertEquals(QuantFormat.MXFP8, args(mxfp8ModelOpt, DOWN).format());
        HfQuantConfig w4afp8 = qc("{'quant_method':'w4afp8','ignored_layers':['lm_head']}");
        assertEquals(QuantFormat.INT4_IN_INT8, args(w4afp8, EXPERT).format());
        assertEquals(ActQuantScheme.FP8_STATIC, args(w4afp8, EXPERT).actScheme());
        assertEquals(128, args(w4afp8, EXPERT).groupSize());
        assertEquals(Strategy.BLOCK, args(w4afp8, Q).strategy());
    }

    // ------------------------------------------------------------------ compressed-tensors

    @Test
    public void compressedTensorsGroupsAndTransforms() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'compressed-tensors','format':'pack-quantized','ignore':['lm_head'],"
                + "'config_groups':{"
                + "'attn':{'targets':['LlamaAttention'],'weights':{'num_bits':8,'type':'float'}},"
                + "'g0':{'targets':['Linear'],'weights':{'num_bits':4,'type':'int','strategy':'group',"
                + "'group_size':128,'symmetric':true,'actorder':'static'}},"
                + "'g1':{'targets':['re:.*down_proj'],'format':'int-quantized','weights':{'num_bits':8,'type':'int',"
                + "'strategy':'channel','symmetric':true},'input_activations':{'num_bits':8,'type':'int',"
                + "'dynamic':true,'strategy':'token'}}},"
                + "'transform_config':{'config_groups':{'v':{'type':'hadamard','apply':[{'targets':['Linear'],"
                + "'location':'weight_input'},{'targets':['Linear'],'location':'input','inverse':true}]}}}}");
        assertEquals(HfQuantMethod.COMPRESSED_TENSORS, q.method());
        assertEquals(2, q.rules().size());
        assertEquals(QuantRule.ModuleMatcher.Kind.REGEX, q.rules().get(0).matcher().kind());
        QuantArgs down = args(q, DOWN);
        assertEquals(QuantFormat.INT8_CHANNEL, down.format());
        assertEquals(Strategy.CHANNEL, down.strategy());
        assertEquals(ActQuantScheme.INT8_TOKEN, down.actScheme());
        QuantArgs packed = args(q, Q);
        assertEquals(QuantFormat.INT_PACKED_CT, packed.format());
        assertEquals(4, packed.bits());
        assertEquals(128, packed.groupSize());
        assertTrue(packed.actOrder());
        assertTrue(q.isIgnored("lm_head"));
        HfTransformConfig t = q.transforms().get();
        assertEquals(1, t.schemes().size());
        assertEquals(HfTransformConfig.Location.WEIGHT_INPUT, t.schemes().get(0).applies().get(0).location());
    }

    @Test
    public void compressedTensorsFp4AndRejections() throws IOException {
        QuantArgs nvfp4 = args(qc("{'quant_method':'compressed-tensors','format':'nvfp4-pack-quantized',"
                + "'config_groups':{'g':{'targets':['Linear'],'weights':{'num_bits':4,'type':'float',"
                + "'strategy':'tensor_group','group_size':16},'input_activations':{'num_bits':4,'type':'float',"
                + "'strategy':'tensor_group','group_size':16,'dynamic':'local'}}}}"), Q);
        assertEquals(QuantFormat.NVFP4_CT, nvfp4.format());
        assertEquals(ActQuantScheme.NVFP4, nvfp4.actScheme());
        assertTrue(nvfp4.actStatic());
        assertEquals(HfQuantMethod.COMPRESSED_TENSORS, qc("{'quant_method':'compressed-tensors',"
                + "'sparsity_config':{'targets':[]}}").method());
        assertRejected("{'quant_method':'compressed-tensors','sparsity_config':{'format':'sparse-24-bitmask',"
                + "'targets':['Linear']}}", "unsupported quant_method 'compressed-tensors'");
        assertRejected("{'quant_method':'compressed-tensors','transform_config':{'config_groups':{'v':"
                + "{'type':'fwht','apply':[]}}}}", "unsupported transform type");
    }

    // ------------------------------------------------------------------ ModelOpt

    @Test
    public void everyModelOptAlgorithm() throws IOException {
        QuantArgs fp8 = args(modelOpt("FP8"), Q);
        assertEquals(QuantFormat.FP8, fp8.format());
        assertEquals(ActQuantScheme.FP8_STATIC, fp8.actScheme());
        QuantArgs perChannel = args(modelOpt("FP8_PER_CHANNEL_PER_TOKEN"), Q);
        assertEquals(Strategy.CHANNEL, perChannel.strategy());
        assertEquals(ActQuantScheme.FP8_TOKEN, perChannel.actScheme());
        for (String algo : new String[]{"FP8_PB_WO", "FP8_BLOCK_SCALES"}) {
            QuantArgs block = args(modelOpt(algo), Q);
            assertEquals(Strategy.BLOCK, block.strategy());
            assertEquals(128, block.blockK());
            assertNull(block.actScheme());
        }
        QuantArgs nvfp4 = args(modelOpt("NVFP4"), Q);
        assertEquals(QuantFormat.NVFP4_MODELOPT, nvfp4.format());
        assertEquals(16, nvfp4.groupSize());
        assertEquals(ActQuantScheme.NVFP4, nvfp4.actScheme());
        QuantArgs w4a16 = args(modelOpt("W4A16_NVFP4"), Q);
        assertEquals(QuantFormat.NVFP4_MODELOPT, w4a16.format());
        assertNull(w4a16.actScheme());
        QuantArgs mxfp8 = args(modelOpt("MXFP8"), Q);
        assertEquals(QuantFormat.MXFP8, mxfp8.format());
        assertEquals(ActQuantScheme.MXFP8, mxfp8.actScheme());
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> modelOpt("INT4_AWQ"));
        assertEquals("unsupported ModelOpt quant_algo 'INT4_AWQ'", e.getMessage());
    }

    private HfQuantConfig modelOpt(String algo) throws IOException {
        return qc("{'quant_method':'modelopt','quant_algo':'" + algo + "','group_size':'16','ignore':['lm_head']}");
    }

    @Test
    public void modelOptMixedPrecisionAndLayouts() throws IOException {
        HfQuantConfig mixed = qc("{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','quantized_layers':{"
                + "'model.layers.0.mlp.down_proj':{'quant_algo':'NVFP4','group_size':16},"
                + "'model.layers.1':{'quant_algo':'FP8'}}}");
        assertEquals("MIXED_PRECISION", mixed.modelOptAlgo().get());
        assertEquals(QuantFormat.NVFP4_MODELOPT, args(mixed, DOWN).format());
        assertEquals(QuantFormat.FP8, args(mixed, "model.layers.1.self_attn.q_proj").format());
        assertFalse(mixed.argsFor("model.layers.10.self_attn.q_proj").isPresent());
        assertFalse(mixed.argsFor("model.layers.2.self_attn.q_proj").isPresent());
        assertRejected("{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','quantized_layers':{}}",
                "quantized_layers");
        HfQuantConfig groups = qc("{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','quantized_layers':{"
                + "'model.layers.0.mlp.up_proj':{'quant_algo':'W4A16_NVFP4','group_size':32},"
                + "'model.layers.0.mlp.down_proj':{'quant_algo':'NVFP4'},"
                + "'model.layers.0.mlp.experts':{'quant_algo':'FP8_PB_WO','group_size':64},"
                + "'model.layers.1.mlp.experts':{'quant_algo':'FP8_BLOCK_SCALES','group_size':64}}}");
        assertEquals(32, args(groups, DOWN).groupSize());
        assertEquals(32, args(groups, "model.layers.0.mlp.up_proj").groupSize());
        assertEquals(64, args(groups, "model.layers.1.mlp.experts.3.w1").blockK());
        assertEquals(16, args(qc("{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','group_size':16,"
                + "'quantized_layers':{'" + DOWN + "':{'quant_algo':'NVFP4','group_size':32}}}"), DOWN).groupSize());
        assertRejected("{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','quantized_layers':{"
                + "'a':{'quant_algo':'FP8_PB_WO','group_size':64},'b':{'quant_algo':'FP8_PB_WO'}}}", "group_size");
        HfQuantConfig producer = qc("{'producer':{'name':'modelopt'},'quantization':{'quant_algo':"
                + "'FP8_PER_CHANNEL_PER_TOKEN','exclude_modules':['*vision*']}}");
        assertEquals(HfQuantMethod.MODELOPT, producer.method());
        assertTrue(producer.isIgnored("model.vision_tower.x"));
        assertRejected("{'quant_method':'modelopt'}", "quant_algo");
    }

    @Test
    public void modelOptMixedWithoutLayersReadsTheSideFile() throws IOException {
        HfConfigLoaderTest.write(tmp.getRoot().toPath().resolve("hf_quant_config.json"),
                "{'producer':{'name':'modelopt'},'quantization':{'quant_algo':'MIXED_PRECISION',"
                        + "'quantized_layers':{'" + DOWN + "':{'quant_algo':'FP8'}}}}");
        HfQuantConfig q = qc("{'quant_method':'modelopt_mixed','quant_algo':'MIXED_PRECISION'}");
        assertEquals("hf_quant_config.json", q.view().source());
        assertEquals(QuantFormat.FP8, args(q, DOWN).format());
        assertFalse(q.argsFor(Q).isPresent());
        HfQuantConfig embedded = qc("{'quant_method':'modelopt','quant_algo':'MIXED_PRECISION','quantized_layers':{'"
                + Q + "':{'quant_algo':'FP8'}}}");
        assertEquals("quantization_config", embedded.view().path());
        assertTrue(embedded.argsFor(Q).isPresent());
    }

    // ------------------------------------------------------------------ integer methods

    @Test
    public void gptqDynamicOverridesComeFirst() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'gptq','bits':4,'group_size':128,'desc_act':true,'sym':true,"
                + "'checkpoint_format':'gptq_v2','dynamic':{'-:.*lm_head':{},'+:.*down_proj':{'bits':8,"
                + "'group_size':-1}}}");
        QuantArgs base = args(q, Q);
        assertEquals(QuantFormat.INT_GPTQ, base.format());
        assertEquals(4, base.bits());
        assertEquals(128, base.groupSize());
        assertEquals(2, base.gptqVersion());
        assertTrue(base.actOrder());
        QuantArgs down = args(q, DOWN);
        assertEquals(8, down.bits());
        assertEquals(Strategy.CHANNEL, down.strategy());
        assertFalse(down.actOrder());
        assertFalse(q.argsFor("lm_head").isPresent());
        assertEquals(1, args(qc("{'quant_method':'gptq','bits':4,'group_size':-1}"), Q).gptqVersion());
        assertEquals(8, args(qc("{'quant_method':'moe_wna16','weight_bits':8,'bits':4,'group_size':64}"), Q).bits());
        assertRejected("{'quant_method':'gptq','bits':4,'group_size':128,'checkpoint_format':'marlin'}", "marlin");
        assertRejected("{'quant_method':'gptq','bits':4,'group_size':128,'is_marlin_format':true}", "marlin");
        assertRejected("{'quant_method':'gptq','bits':4}", "group_size");
    }

    @Test
    public void gptqLmHeadAndModulesInBlock() throws IOException {
        assertFalse(qc("{'quant_method':'gptq','bits':4,'group_size':128}").argsFor("lm_head").isPresent());
        assertTrue(qc("{'quant_method':'gptq','bits':4,'group_size':128,'lm_head':true}").argsFor("lm_head")
                .isPresent());
        HfQuantConfig q = qc("{'quant_method':'gptq','bits':4,'group_size':128,'modules_in_block_to_quantize':"
                + "[['self_attn.q_proj','self_attn.k_proj'],['mlp.down_proj']],'dynamic':{'+:.*gate_proj':{'bits':8}}}");
        assertTrue(q.argsFor(Q).isPresent());
        assertTrue(q.argsFor(DOWN).isPresent());
        assertFalse(q.argsFor("model.layers.0.self_attn.o_proj").isPresent());
        assertFalse(q.argsFor("model.layers.0.mlp.gate_proj").isPresent());
        assertFalse(qc("{'quant_method':'gptq','bits':4,'group_size':128,'modules_in_block_to_quantize':['q.proj']}")
                .argsFor("model.layers.0.self_attn.qxproj").isPresent());
        assertRejected("{'quant_method':'gptq','bits':4,'group_size':128,'modules_in_block_to_quantize':'q'}",
                "array");
    }

    @Test
    public void awqReadsBothKeySpellings() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'awq','w_bit':4,'q_group_size':128,'zero_point':true,"
                + "'modules_to_not_convert':['visual']}");
        QuantArgs a = args(q, Q);
        assertEquals(QuantFormat.INT_AWQ, a.format());
        assertEquals(128, a.groupSize());
        assertFalse(a.symmetric());
        assertTrue(q.isIgnored("model.visual.blocks.0.attn.qkv"));
        assertTrue(args(qc("{'quant_method':'awq','bits':4,'group_size':64,'zero_point':false}"), Q).symmetric());
        assertRejected("{'quant_method':'awq','bits':4}", "AWQ needs");
        assertFalse(q.argsFor("lm_head").isPresent());
        assertTrue(qc("{'quant_method':'awq','bits':4,'group_size':128,'lm_head':true}").argsFor("lm_head")
                .isPresent());
    }

    @Test
    public void autoRoundIsReclassifiedByPacking() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'auto-round','bits':4,'group_size':128,'sym':true,"
                + "'extra_config':{'lm_head':{'bits':8},'" + DOWN + "':{'bits':16},'.*gate.*':{'bits':8}}}");
        assertEquals(HfQuantMethod.GPTQ, q.method());
        assertEquals("auto-round", q.methodName());
        assertEquals(8, args(q, "lm_head").bits());
        assertFalse(q.argsFor(DOWN).isPresent());
        assertEquals(8, args(q, "model.layers.1.mlp.gate_proj").bits());
        assertEquals(4, args(q, Q).bits());
        assertFalse(qc("{'quant_method':'auto-round','bits':4,'group_size':128}").argsFor("model.lm_head")
                .isPresent());
        HfQuantConfig awq = qc("{'quant_method':'inc','bits':4,'group_size':128,'packing_format':'auto_round:auto_awq'}");
        assertEquals(HfQuantMethod.AWQ, awq.method());
        assertEquals(QuantFormat.INT_AWQ, args(awq, Q).format());
        assertEquals(HfQuantMethod.GPTQ, qc("{'quant_method':'auto-round','bits':4,'group_size':128,"
                + "'packing_format':'auto_round:auto_awq','backend':'gptq'}").method());
        HfQuantConfig blocks = qc("{'quant_method':'auto-round','bits':4,'group_size':128,"
                + "'to_quant_block_names':['model.layers']}");
        assertTrue(blocks.argsFor(Q).isPresent());
        assertFalse(blocks.argsFor("visual.blocks.0.attn.qkv").isPresent());
        QuantArgs fp8 = args(qc("{'quant_method':'auto-round','data_type':'fp','bits':8,'group_size':[128,128]}"), Q);
        assertEquals(Strategy.BLOCK, fp8.strategy());
        assertEquals(ActQuantScheme.FP8_GROUP128, fp8.actScheme());
        QuantArgs mxfp4 = args(qc("{'quant_method':'auto-round','data_type':'mx_fp','bits':4,'group_size':32}"), Q);
        assertEquals(QuantFormat.MXFP4_CT, mxfp4.format());
        assertEquals(ActQuantScheme.MXFP4, mxfp4.actScheme());
        assertRejected("{'quant_method':'auto-round','data_type':'nv_fp','bits':4,'group_size':16}", "data_type");
    }

    @Test
    public void bitsAndBytes() throws IOException {
        HfQuantConfig nf4 = qc("{'quant_method':'bitsandbytes','load_in_4bit':true,'bnb_4bit_quant_type':'nf4',"
                + "'bnb_4bit_use_double_quant':true,'llm_int8_skip_modules':['lm_head','vision_tower']}");
        assertEquals(QuantFormat.NF4_BNB, args(nf4, Q).format());
        assertEquals(Boolean.TRUE, args(nf4, Q).bnbDoubleQuant());
        assertTrue(nf4.isIgnored("model.vision_tower.encoder.layers.0.q"));
        assertTrue(nf4.isIgnored("lm_head"));
        assertFalse(nf4.isIgnored("lm_head_extra"));
        HfQuantConfig fp4 = qc("{'quant_method':'bitsandbytes','load_in_4bit':true}");
        assertEquals(QuantFormat.FP4_BNB, args(fp4, Q).format());
        assertNull(args(fp4, Q).bnbDoubleQuant());
        assertEquals(QuantFormat.INT8_BNB, args(qc("{'quant_method':'bitsandbytes','load_in_8bit':true}"), Q)
                .format());
        assertRejected("{'quant_method':'bitsandbytes'}", "neither");
        assertRejected("{'quant_method':'bitsandbytes','load_in_4bit':true,'bnb_4bit_quant_type':'int4'}", "int4");
    }

    @Test
    public void quarkRulesInPriorityOrder() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'quark','export':{'pack_method':'reorder'},'exclude':['lm_head'],"
                + "'global_quant_config':{'weight':{'dtype':'int4','qscheme':'per_group','group_size':128,"
                + "'symmetric':true},'input_tensors':null},"
                + "'layer_quant_config':{'*down_proj':{'weight':{'dtype':'fp8_e4m3','qscheme':'per_tensor'},"
                + "'input_tensors':{'dtype':'fp8_e4m3','qscheme':'per_tensor','is_dynamic':false}},"
                + "'*.gate':{'weight':null}}}");
        QuantArgs down = args(q, DOWN);
        assertEquals(QuantFormat.FP8, down.format());
        assertEquals(ActQuantScheme.FP8_STATIC, down.actScheme());
        assertFalse(q.argsFor("model.layers.0.mlp.gate").isPresent());
        QuantArgs base = args(q, Q);
        assertEquals(QuantFormat.INT_QUARK, base.format());
        assertEquals(128, base.groupSize());
        assertTrue(base.quarkPackReorder());
        assertTrue(q.isIgnored("lm_head"));
        QuantArgs mx2d = args(qc("{'quant_method':'quark','global_quant_config':{'weight':{'dtype':'fp8_e4m3',"
                + "'qscheme':'per_block','block_size':[1,32],'scale_type':'float8_e8m0fnu','symmetric':true}}}"), Q);
        assertEquals(QuantFormat.MXFP8, mx2d.format());
        assertEquals(32, mx2d.groupSize());
        QuantArgs mxRows = args(qc("{'quant_method':'quark','global_quant_config':{'weight':{'dtype':'fp8_e4m3',"
                + "'qscheme':'per_block','block_size':[32,32],'scale_type':'float8_e8m0fnu','symmetric':true}}}"), Q);
        assertEquals(QuantFormat.MXFP8, mxRows.format());
        assertEquals(Strategy.BLOCK, mxRows.strategy());
        assertEquals(32, mxRows.blockN());
        assertEquals(QuantArgs.ScaleFormat.E8M0, mxRows.scaleFormat());
        QuantArgs blockE8m0 = args(qc("{'quant_method':'quark','global_quant_config':{'weight':{'dtype':'fp8_e4m3',"
                + "'qscheme':'per_block','block_size':[128,128],'scale_type':'float8_e8m0fnu','symmetric':true}}}"), Q);
        assertEquals(QuantFormat.FP8, blockE8m0.format());
        assertEquals(128, blockE8m0.blockK());
        assertEquals(QuantArgs.ScaleFormat.E8M0, blockE8m0.scaleFormat());
        assertRejected("{'quant_method':'quark','global_quant_config':{'weight':[{'dtype':'int4'},"
                + "{'dtype':'fp8_e4m3'}]}}", "unsupported quant_method 'quark'");
    }

    /** The two-level weight and input specs of V quark.py _is_nvfp4. */
    private static final String NVFP4_WEIGHT = "[{'dtype':'fp4','qscheme':'per_group','group_size':16,"
            + "'is_dynamic':false},{'dtype':'fp8_e4m3','qscheme':'per_tensor','is_dynamic':false}]";
    private static final String NVFP4_INPUT = "[{'dtype':'fp4','qscheme':'per_group','group_size':16,"
            + "'is_dynamic':true},{'dtype':'fp8_e4m3','qscheme':'per_tensor','is_dynamic':false}]";

    @Test
    public void quarkNvfp4TwoLevelSpecs() throws IOException {
        HfQuantConfig q = qc("{'quant_method':'quark','exclude':['lm_head'],'global_quant_config':{'weight':"
                + NVFP4_WEIGHT + ",'input_tensors':" + NVFP4_INPUT + "}}");
        QuantArgs a = args(q, Q);
        assertEquals(QuantFormat.NVFP4_QUARK, a.format());
        assertEquals(4, a.bits());
        assertEquals(16, a.groupSize());
        assertEquals(Strategy.GROUP, a.strategy());
        assertEquals(QuantArgs.ScaleFormat.E4M3, a.scaleFormat());
        assertEquals(ActQuantScheme.NVFP4, a.actScheme());
        assertEquals(HfQuantMethod.QUARK, q.method());
        assertTrue(q.isIgnored("lm_head"));

        QuantArgs single = args(qc("{'quant_method':'quark','global_quant_config':{'weight':[{'dtype':'int4',"
                + "'qscheme':'per_group','group_size':128}]}}"), Q);
        assertEquals(QuantFormat.INT_QUARK, single.format());
        assertEquals(128, single.groupSize());

        assertRejected("{'quant_method':'quark','global_quant_config':{'weight':" + NVFP4_WEIGHT
                + ",'input_tensors':{'dtype':'fp8_e4m3','qscheme':'per_tensor'}}}", "only supported for NVFP4");
        assertRejected("{'quant_method':'quark','global_quant_config':{'weight':[{'dtype':'int4'},"
                + "{'dtype':'fp8_e4m3'}],'input_tensors':" + NVFP4_INPUT + "}}", "only supported for NVFP4");
        assertRejected("{'quant_method':'quark','global_quant_config':{'weight':[{'dtype':'fp4',"
                + "'qscheme':'per_group','group_size':32},{'dtype':'fp8_e4m3','qscheme':'per_tensor'}],"
                + "'input_tensors':" + NVFP4_INPUT + "}}", "only supported for NVFP4");
        assertRejected("{'quant_method':'quark','global_quant_config':{'weight':" + NVFP4_WEIGHT
                + ",'input_tensors':[{'dtype':'fp4','qscheme':'per_group','group_size':16},"
                + "{'dtype':'fp8_e4m3','qscheme':'per_tensor'}]}}", "only supported for NVFP4");
    }

    @Test
    public void mlxBlocks() throws IOException {
        String block = "{'group_size':64,'bits':4,'" + DOWN + "':{'bits':8,'group_size':64},'lm_head':false}";
        for (String members : new String[]{"'quantization':" + block, "'quantization_config':" + block}) {
            HfQuantConfig q = quant(members);
            assertEquals(HfQuantMethod.MLX, q.method());
            assertEquals("mlx_q4", q.methodName());
            QuantArgs a = args(q, Q);
            assertEquals(QuantFormat.INT_MLX, a.format());
            assertEquals(4, a.mlxBits());
            assertEquals(64, a.groupSize());
            assertFalse(a.symmetric());
            assertEquals(8, args(q, DOWN).mlxBits());
            assertFalse(q.argsFor("lm_head").isPresent());
        }
        assertThrows(GgufFormatException.class, () -> quant("'quantization':{'group_size':32,'bits':4,"
                + "'mode':'mxfp4'}"));
        HfQuantConfig q3 = quant("'quantization':{'group_size':64,'bits':3}");
        assertEquals(HfQuantMethod.MLX, q3.method());
        assertEquals("mlx_q3", q3.methodName());
        assertEquals(3, args(q3, Q).mlxBits());
    }

    @Test
    public void bitnetHqqAndFpQuant() throws IOException {
        HfQuantConfig bitnet = qc("{'quant_method':'bitnet','modules_to_not_convert':['lm_head']}");
        assertEquals(QuantFormat.BITNET, args(bitnet, Q).format());
        assertTrue(bitnet.isIgnored("lm_head"));
        QuantArgs hqq = args(qc("{'quant_method':'hqq','quant_config':{'weight_quant_params':{'nbits':4,"
                + "'group_size':64}},'skip_modules':['lm_head']}"), Q);
        assertEquals(QuantFormat.INT_HQQ, hqq.format());
        assertEquals(1, hqq.hqqAxis());
        assertEquals(64, hqq.groupSize());
        assertRejected("{'quant_method':'hqq','quant_config':{}}", "weight_quant_params");

        HfQuantConfig fpq = qc("{'quant_method':'fp_quant','forward_dtype':'mxfp4','hadamard_group_size':32,"
                + "'transform_init':'gsr'}");
        QuantArgs a = args(fpq, Q);
        assertEquals(QuantFormat.FPQUANT_MXFP4, a.format());
        assertEquals(32, a.fpQuantHadamardGroup());
        assertEquals(QuantArgs.FpQuantTransform.GSR, a.fpQuantTransform());
        assertNotNull(a.actUnemulated());
        assertNull(a.actScheme());
        assertTrue(fpq.isIgnored("model.lm_head"));
        QuantArgs nvfp4 = args(qc("{'quant_method':'fp_quant','forward_method':'abs_max'}"), Q);
        assertEquals(QuantFormat.FPQUANT_NVFP4, nvfp4.format());
        assertEquals(16, nvfp4.fpQuantHadamardGroup());
        assertEquals(32, args(qc("{'quant_method':'fp_quant','forward_dtype':'mxfp4','forward_method':'quest'}"), Q)
                .fpQuantHadamardGroup());
        assertRejected("{'quant_method':'fp_quant','forward_dtype':'nvfp4','forward_method':'quest'}", "abs_max");
        assertRejected("{'quant_method':'fp_quant','forward_dtype':'mxfp4','forward_method':'rtn'}", "forward_method");
        assertEquals(QuantFormat.FPQUANT_PSEUDO, args(qc("{'quant_method':'fp_quant','pseudoquantization':true}"),
                Q).format());
        assertRejected("{'quant_method':'fp_quant','forward_dtype':'mxfp4','hadamard_group_size':16}",
                "hadamard_group_size");
        assertRejected("{'quant_method':'fp_quant','forward_dtype':'int4'}", "forward_dtype");
        HfQuantConfig gptOss = qc("{'quant_method':'mxfp4','modules_to_not_convert':['model.layers.*.self_attn',"
                + "'model.layers.1','router']}");
        assertEquals(QuantFormat.MXFP4_GPTOSS, args(gptOss, "model.layers.0.mlp.experts.gate_up_proj").format());
        assertEquals(QuantFormat.MXFP4_GPTOSS, args(gptOss, "model.layers.0.mlp.experts").format());
        assertTrue(gptOss.isIgnored(Q));
        assertTrue(gptOss.isIgnored("model.layers.1.mlp.experts"));
        assertFalse(gptOss.isIgnored("model.layers.11.mlp.experts"));
        assertTrue(gptOss.isIgnored("model.layers.0.mlp.router"));
        assertFalse(gptOss.argsFor(DOWN).isPresent());
        assertFalse(gptOss.argsFor("model.layers.0.mlp.shared_experts.down_proj").isPresent());
    }
}
