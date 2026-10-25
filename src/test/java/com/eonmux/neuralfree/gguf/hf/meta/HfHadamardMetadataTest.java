package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.HadamardPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.PrismHadamardLoader;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HfHadamardMetadata} against {@code add_hadamard_metadata} (LC/conversion/base.py:691-840): schemas 1-3, the
 * key set and order, every rejection path, the class name filter on record names, the {@code prec_a4} flags and the
 * architecture set shared with the runtime loader; and the contract written for an online input transform.
 */
public class HfHadamardMetadataTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static final String Q0 = "model.layers.0.self_attn.q_proj.weight";
    private static final String DOWN1 = "model.layers.1.mlp.down_proj.weight";
    private static final String EMBED = "model.embed_tokens.weight";
    private static final String TRANSFORM_64 = "{'name': 'normalized-signed-sylvester-walsh-hadamard',"
            + " 'block_size': 64, 'sign_mode': 'identity'}";

    /** A checkpoint directory with the configuration, the manifest (when not null) and an index of tensor names. */
    private final class Fixture {
        Arch arch = Arch.LLAMA;
        String hfName = "LlamaForCausalLM";
        String extraConfig = "";
        String manifest;
        List<String> indexNames = new ArrayList<>(Arrays.asList(EMBED, Q0, DOWN1));
        Set<String> planned = new LinkedHashSet<>();
        Map<String, Long> widths = new HashMap<>();
        Map<String, Boolean> precA4 = Collections.emptyMap();
        HfNameFilter nameFilter = HfNameFilter.INHERITED;
        HadamardPlan online;

        HfLoadContext context() throws IOException {
            Path dir = tmp.newFolder().toPath();
            if (manifest != null) {
                Files.write(dir.resolve(HfHadamardMetadata.MANIFEST),
                        manifest.replace('\'', '"').getBytes(StandardCharsets.UTF_8));
            }
            String config = "{'architectures': ['" + hfName + "'], 'num_hidden_layers': 2" + extraConfig + "}";
            HfArchEntry entry = HfArchEntry.of(List.of(hfName), HfArchResolver.of(arch), HfMetadataHook.GENERIC,
                    HfTensorHook.DEFAULT, (ctx, pre) -> {
                        throw new UnsupportedOperationException("vocabulary is not part of this test");
                    }).withNameFilter(nameFilter);
            Map<String, RowSource> index = new LinkedHashMap<>();
            for (String n : indexNames) {
                index.put(n, unreadSource());
            }
            return HfLoadContext.builder(HfConfigLoader.fromJson(dir, config.replace('\'', '"')), entry, arch)
                    .index(HfTensorIndex.of(index)).build().withPrecA4(precA4);
        }

        HfMetadata apply() throws IOException {
            HfMetadataBuilder kv = new HfMetadataBuilder(arch.ggufName());
            HfHadamardMetadata.apply(context(), kv, plan(planned, widths), online);
            return kv.build();
        }

        void expectFailure(String message) throws IOException {
            try {
                apply();
                fail("expected " + message);
            } catch (GgufFormatException e) {
                assertEquals(message, e.getMessage());
            }
        }
    }

    private static RowSource unreadSource() {
        return (RowSource) Proxy.newProxyInstance(RowSource.class.getClassLoader(), new Class<?>[] {RowSource.class},
                (proxy, method, args) -> {
                    throw new UnsupportedOperationException("the Hadamard metadata never reads tensors");
                });
    }

    private static HfPlanView plan(Set<String> names, Map<String, Long> widths) {
        return new HfPlanView() {
            @Override
            public List<String> names() {
                return new ArrayList<>(names);
            }

            @Override
            public boolean has(String ggufName) {
                return names.contains(ggufName);
            }

            @Override
            public long[] ne(String ggufName) {
                Long width = widths.get(ggufName);
                if (width == null) {
                    throw new UnsupportedOperationException(ggufName);
                }
                return new long[] {width, 8};
            }

            @Override
            public GGMLType type(String ggufName) {
                throw new UnsupportedOperationException(ggufName);
            }

            @Override
            public float[] readF32(String ggufName, int maxElements) {
                throw new UnsupportedOperationException(ggufName);
            }

            @Override
            public long[] readI64(String ggufName, int maxElements) {
                throw new UnsupportedOperationException(ggufName);
            }
        };
    }

    private static String manifest(int schema, String transform, String tensors, String extra) {
        return "{'schema_version': " + schema + ", 'kind': 'hadamard-weight-fold',"
                + " 'status': 'requires-matching-runtime', 'transform': " + transform + ", 'tensors': " + tensors
                + extra + "}";
    }

    private static String fold(String... names) {
        StringJoiner j = new StringJoiner(", ", "[", "]");
        for (String n : names) {
            j.add("{'name': '" + n + "', 'axis': -1}");
        }
        return j.toString();
    }

    private static String inverseEmbed() {
        return "{'name': '" + EMBED + "', 'axis': -1, 'role': 'inverse-after-lookup'}";
    }

    private static String signs(int width, int value) {
        StringJoiner j = new StringJoiner(", ", "[", "]");
        for (int i = 0; i < width; i++) {
            j.add(Integer.toString(i % 2 == 0 ? value : -value));
        }
        return j.toString();
    }

    private Fixture schema1(String tensors) {
        Fixture f = new Fixture();
        f.manifest = manifest(1, TRANSFORM_64, tensors, "");
        return f;
    }

    @Test
    public void schema1IdentityWritesTheContract() throws IOException {
        HfMetadata m = schema1(fold(Q0, DOWN1)).apply();
        assertEquals(Arrays.asList("prism.hadamard.version", "prism.hadamard.block_size", "prism.hadamard.transform",
                "prism.hadamard.axis", "prism.hadamard.sign_mode", "prism.hadamard.weight_names"),
                new ArrayList<>(m.keys()));
        assertEquals(1, m.get("prism.hadamard.version"));
        assertEquals(GGUFValueType.UINT32, m.type("prism.hadamard.version"));
        assertEquals(64, m.get("prism.hadamard.block_size"));
        assertEquals("normalized-sylvester-walsh-hadamard", m.get("prism.hadamard.transform"));
        assertEquals("input-last-dimension", m.get("prism.hadamard.axis"));
        assertEquals("identity", m.get("prism.hadamard.sign_mode"));
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "blk.1.ffn_down.weight"),
                m.get("prism.hadamard.weight_names"));
        assertEquals(GGUFValueType.STRING, m.elementType("prism.hadamard.weight_names"));
    }

    @Test
    public void schema2ExplicitSignsAreOrderedByWidth() throws IOException {
        Fixture f = new Fixture();
        f.manifest = manifest(2, "{'name': 'normalized-signed-sylvester-walsh-hadamard', 'block_size': 64,"
                + " 'sign_mode': 'explicit'}", "[" + inverseEmbed() + ", {'name': '" + Q0 + "', 'axis': -1.0,"
                + " 'role': 'fold-before-matmul'}]", ", 'signs': {'128': " + signs(128, -1) + ", '64': "
                + signs(64, 1) + "}");
        HfMetadata m = f.apply();
        assertEquals(Arrays.asList("prism.hadamard.version", "prism.hadamard.block_size", "prism.hadamard.transform",
                "prism.hadamard.axis", "prism.hadamard.sign_mode", "prism.hadamard.weight_names",
                "prism.hadamard.sign_widths", "prism.hadamard.sign_values", "prism.hadamard.inverse_weight_names"),
                new ArrayList<>(m.keys()));
        assertEquals(1, m.get("prism.hadamard.version"));
        assertEquals(Arrays.asList(64, 128), m.get("prism.hadamard.sign_widths"));
        assertEquals(GGUFValueType.INT32, m.elementType("prism.hadamard.sign_values"));
        List<?> values = (List<?>) m.get("prism.hadamard.sign_values");
        assertEquals(192, values.size());
        assertEquals(Arrays.asList(1, -1), values.subList(0, 2));
        assertEquals(Arrays.asList(-1, 1), values.subList(64, 66));
        assertEquals(Collections.singletonList("blk.0.attn_q.weight"), m.get("prism.hadamard.weight_names"));
        assertEquals(Collections.singletonList("token_embd.weight"), m.get("prism.hadamard.inverse_weight_names"));
    }

    @Test
    public void schema3TiedOutputLeadsTheKeys() throws IOException {
        Fixture f = new Fixture();
        f.extraConfig = ", 'tie_word_embeddings': true";
        f.manifest = manifest(3, TRANSFORM_64, "[" + inverseEmbed() + "]", ", 'tied_output': true");
        HfMetadata m = f.apply();
        assertEquals(Arrays.asList("prism.hadamard.tied_output", "prism.hadamard.version", "prism.hadamard.block_size",
                "prism.hadamard.transform", "prism.hadamard.axis", "prism.hadamard.sign_mode",
                "prism.hadamard.inverse_weight_names"), new ArrayList<>(m.keys()));
        assertEquals(true, m.get("prism.hadamard.tied_output"));
        assertEquals(2, m.get("prism.hadamard.version"));
    }

    @Test
    public void manifestHeaderRejections() throws IOException {
        String tensors = fold(Q0);
        String hdr = "hadamard_packing.json: ";
        Fixture f = new Fixture();
        f.manifest = manifest(4, TRANSFORM_64, tensors, "");
        f.expectFailure(hdr + "unsupported Hadamard manifest (schema_version 4, kind hadamard-weight-fold)");
        f.manifest = manifest(1, TRANSFORM_64, tensors, "").replace("hadamard-weight-fold", "other");
        f.expectFailure(hdr + "unsupported Hadamard manifest (schema_version 1, kind other)");
        f.manifest = manifest(1, TRANSFORM_64, tensors, "").replace("requires-matching-runtime", "draft");
        f.expectFailure(hdr + "unexpected Hadamard manifest status: draft");
        f.manifest = manifest(1, "[]", tensors, "");
        f.expectFailure(hdr + "the manifest is missing transform metadata");
        f.manifest = "[]";
        f.expectFailure(hdr + "the manifest is not a JSON object");
    }

    @Test
    public void transformRejections() throws IOException {
        String tensors = fold(Q0);
        String hdr = "hadamard_packing.json: ";
        Fixture f = new Fixture();
        for (String size : Arrays.asList("48", "0", "-64", "2147483648", "64.0", "'64'")) {
            f.manifest = manifest(1, TRANSFORM_64.replace("'block_size': 64", "'block_size': " + size), tensors, "");
            String shown = size.equals("'64'") ? "64" : size;
            f.expectFailure(hdr + "invalid Hadamard block size: " + shown + " (a power of two up to 1073741824)");
        }
        f.manifest = manifest(1, TRANSFORM_64.replace("normalized-signed-", "plain-"), tensors, "");
        f.expectFailure(hdr + "unsupported Hadamard transform: plain-sylvester-walsh-hadamard");
        f.manifest = manifest(1, TRANSFORM_64.replace("identity", "random"), tensors, "");
        f.expectFailure(hdr + "unsupported Hadamard sign mode: random");
    }

    @Test
    public void blockSizesAboveTheKernelLimitAreAccepted() throws IOException {
        for (int size : new int[] {2048, 1 << 30}) {
            Fixture f = new Fixture();
            f.manifest = manifest(1, TRANSFORM_64.replace("'block_size': 64", "'block_size': " + size), fold(Q0), "");
            HfMetadata m = f.apply();
            assertEquals(size, m.get("prism.hadamard.block_size"));
            assertEquals(Collections.singletonList("blk.0.attn_q.weight"), m.get("prism.hadamard.weight_names"));
        }
    }

    @Test
    public void signTableRejections() throws IOException {
        String explicit = TRANSFORM_64.replace("identity", "explicit");
        String tensors = fold(Q0);
        String hdr = "hadamard_packing.json: ";
        Fixture f = new Fixture();
        f.manifest = manifest(1, explicit, tensors, "");
        f.expectFailure(hdr + "explicit sign mode requires a signs table");
        f.manifest = manifest(1, explicit, tensors, ", 'signs': {}");
        f.expectFailure(hdr + "explicit sign mode requires a signs table");
        f.manifest = manifest(1, explicit, tensors, ", 'signs': {'32': " + signs(32, 1) + "}");
        f.expectFailure(hdr + "sign width 32 must be positive and a multiple of block size 64");
        f.manifest = manifest(1, explicit, tensors, ", 'signs': {'64': " + signs(63, 1) + "}");
        f.expectFailure(hdr + "invalid sign vector for width 64");
        f.manifest = manifest(1, explicit, tensors, ", 'signs': {'64': " + signs(64, 2) + "}");
        f.expectFailure(hdr + "invalid sign vector for width 64");
        f.manifest = manifest(1, explicit, tensors, ", 'signs': {'x': " + signs(64, 1) + "}");
        f.expectFailure(hdr + "sign width 'x' is not an integer");
        f.manifest = manifest(1, explicit, tensors, ", 'signs': {'64': " + signs(64, 1) + ", '064': "
                + signs(64, 1) + "}");
        f.expectFailure(hdr + "sign width 64 is listed twice");
    }

    @Test
    public void tensorRecordRejections() throws IOException {
        String hdr = "hadamard_packing.json: ";
        schema1("[]").expectFailure(hdr + "the manifest has no folded tensors");
        schema1("[{'axis': -1}]").expectFailure(hdr + "the manifest has an invalid tensor record");
        schema1("[{'name': '" + Q0 + "', 'axis': 0}]")
                .expectFailure(hdr + "unsupported Hadamard tensor axis for '" + Q0 + "'");
        schema1("[{'name': '" + Q0 + "'}]").expectFailure(hdr + "unsupported Hadamard tensor axis for '" + Q0 + "'");
        schema1("[{'name': '" + Q0 + "', 'axis': -1, 'role': 'rotate'}]")
                .expectFailure(hdr + "unsupported Hadamard tensor role for '" + Q0 + "': rotate");
        schema1(fold("model.visual.blocks.0.attn.qkv.weight"))
                .expectFailure(hdr + "Hadamard tensor is filtered out: 'model.visual.blocks.0.attn.qkv.weight'");
        schema1(fold("model.layers.0.self_attn.foo.weight"))
                .expectFailure(hdr + "Can not map tensor 'model.layers.0.self_attn.foo.weight'");
        schema1("[{'name': '" + Q0 + "', 'axis': -1, 'role': 'inverse-after-lookup'}]")
                .expectFailure(hdr + "Hadamard tensor '" + Q0 + "' maps to 'blk.0.attn_q.weight', which is not a"
                        + " verified inverse-after-lookup table");
        schema1(fold("model.layers.0.input_layernorm.weight"))
                .expectFailure(hdr + "Hadamard tensor 'model.layers.0.input_layernorm.weight' maps to"
                        + " 'blk.0.attn_norm.weight', which is not on a verified Hadamard-aware matmul path");
    }

    @Test
    public void recordNamesGoThroughTheClassNameFilter() throws IOException {
        Fixture f = schema1(fold("model.text_model.layers.0.self_attn.q_proj.weight"));
        f.expectFailure("hadamard_packing.json: Can not map tensor 'model.text_model.layers.0.self_attn.q_proj.weight'");
        f.nameFilter = (n, base) -> base.apply(n.replace("text_model.", ""));
        assertEquals(Collections.singletonList("blk.0.attn_q.weight"), f.apply().get("prism.hadamard.weight_names"));
        f.nameFilter = (n, base) -> n.contains("text_model.") ? null : base.apply(n);
        f.expectFailure("hadamard_packing.json: Hadamard tensor is filtered out: "
                + "'model.text_model.layers.0.self_attn.q_proj.weight'");
    }

    private Fixture online(int block, Map<Integer, float[]> signs, String... weights) {
        Fixture f = new Fixture();
        f.online = HadamardPlan.of(block, signs, Arrays.asList(weights));
        return f;
    }

    @Test
    public void onlineTransformWritesTheContract() throws IOException {
        Fixture f = online(64, Collections.emptyMap(), Q0, DOWN1, "model.visual.blocks.0.attn.qkv.weight");
        f.planned.addAll(Arrays.asList("blk.0.attn_q.weight", "blk.1.ffn_down.weight"));
        f.widths.put("blk.0.attn_q.weight", 128L);
        f.widths.put("blk.1.ffn_down.weight", 256L);
        HfMetadata m = f.apply();
        assertEquals(Arrays.asList("prism.hadamard.version", "prism.hadamard.block_size", "prism.hadamard.transform",
                "prism.hadamard.axis", "prism.hadamard.sign_mode", "prism.hadamard.weight_names"),
                new ArrayList<>(m.keys()));
        assertEquals(1, m.get("prism.hadamard.version"));
        assertEquals(64, m.get("prism.hadamard.block_size"));
        assertEquals("identity", m.get("prism.hadamard.sign_mode"));
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "blk.1.ffn_down.weight"),
                m.get("prism.hadamard.weight_names"));
    }

    @Test
    public void onlineTransformWritesExplicitSigns() throws IOException {
        float[] d = new float[128];
        for (int i = 0; i < d.length; i++) {
            d[i] = i % 3 == 0 ? -1f : 1f;
        }
        Fixture f = online(64, Collections.singletonMap(128, d), Q0);
        f.planned.add("blk.0.attn_q.weight");
        f.widths.put("blk.0.attn_q.weight", 128L);
        HfMetadata m = f.apply();
        assertEquals("explicit", m.get("prism.hadamard.sign_mode"));
        assertEquals(Collections.singletonList(128), m.get("prism.hadamard.sign_widths"));
        List<?> values = (List<?>) m.get("prism.hadamard.sign_values");
        assertEquals(128, values.size());
        assertEquals(Arrays.asList(-1, 1, 1, -1), values.subList(0, 4));
        f.widths.put("blk.0.attn_q.weight", 256L);
        f.expectFailure("Hadamard input transform: 'blk.0.attn_q.weight' has input width 256, which has no sign"
                + " vector or is not a multiple of block 64");
    }

    @Test
    public void onlineTransformNamesStackedExpertsOnce() throws IOException {
        String e0 = "model.layers.0.mlp.experts.0.down_proj.weight";
        String e1 = "model.layers.0.mlp.experts.1.down_proj.weight";
        Fixture f = online(64, Collections.emptyMap(), e0, e1);
        f.arch = Arch.QWEN3MOE;
        f.hfName = "Qwen3MoeForCausalLM";
        f.indexNames = new ArrayList<>(Arrays.asList(EMBED, e0, e1));
        f.planned.add("blk.0.ffn_down_exps.weight");
        f.widths.put("blk.0.ffn_down_exps.weight", 128L);
        assertEquals(Collections.singletonList("blk.0.ffn_down_exps.weight"),
                f.apply().get("prism.hadamard.weight_names"));
        f.online = HadamardPlan.of(64, Collections.emptyMap(), Collections.singletonList(e0));
        f.expectFailure("Hadamard input transform: 'blk.0.ffn_down_exps.weight' is planned from '" + e1
                + "', which has no input transform, and from transformed tensors; PRISM rotates whole tensors");
    }

    @Test
    public void onlineTransformRejections() throws IOException {
        Fixture f = online(64, Collections.emptyMap(), Q0);
        f.expectFailure("Hadamard input transform: '" + Q0 + "' maps to 'blk.0.attn_q.weight', which is not planned");
        f.online = HadamardPlan.of(64, Collections.emptyMap(),
                Collections.singletonList("model.layers.0.input_layernorm.weight"));
        f.indexNames.add("model.layers.0.input_layernorm.weight");
        f.planned.add("blk.0.attn_norm.weight");
        f.expectFailure("Hadamard input transform: 'model.layers.0.input_layernorm.weight' maps to"
                + " 'blk.0.attn_norm.weight', which is not on a verified Hadamard-aware matmul path");
        f.online = HadamardPlan.of(64, Collections.emptyMap(), Collections.singletonList(Q0));
        f.manifest = manifest(1, TRANSFORM_64, fold(Q0), "");
        f.expectFailure("hadamard_packing.json: the checkpoint also has an online Hadamard input transform ("
                + f.online + "); PRISM applies one Hadamard contract per model");
        f.manifest = null;
        f.arch = Arch.QWEN2;
        f.hfName = "Qwen2ForCausalLM";
        f.expectFailure("online Hadamard input transform present but qwen2 has no verified PRISM runtime");
    }

    @Test
    public void unverifiedArchitecturesAreRejected() throws IOException {
        Fixture f = schema1(fold(Q0));
        f.arch = Arch.QWEN2;
        f.hfName = "Qwen2ForCausalLM";
        f.expectFailure("hadamard_packing.json present but qwen2 has no verified PRISM runtime");
    }

    @Test
    public void tiedOutputRejections() throws IOException {
        String hdr = "hadamard_packing.json: ";
        String schemaRule = hdr + "Hadamard schema 3 requires tied_output=true; older schemas forbid it";
        Fixture f = new Fixture();
        f.extraConfig = ", 'tie_word_embeddings': true";
        f.manifest = manifest(3, TRANSFORM_64, "[" + inverseEmbed() + "]", "");
        f.expectFailure(schemaRule);
        f.manifest = manifest(2, TRANSFORM_64, fold(Q0), ", 'tied_output': true");
        f.expectFailure(schemaRule);
        f.manifest = manifest(3, TRANSFORM_64, "[" + inverseEmbed() + "]", ", 'tied_output': 'yes'");
        f.expectFailure(schemaRule);
        f.manifest = manifest(3, TRANSFORM_64, fold(Q0), ", 'tied_output': true");
        f.expectFailure(hdr + "tied Hadamard output requires one latent token embedding");
        f.manifest = manifest(3, TRANSFORM_64, "[" + inverseEmbed() + "]", ", 'tied_output': true");
        f.indexNames.add("lm_head.weight");
        f.expectFailure(hdr + "tied Hadamard output must not carry a separate output head");
        f.indexNames.remove("lm_head.weight");
        f.planned.add("output.weight");
        f.expectFailure(hdr + "tied Hadamard output must not carry a separate output head");
        f.planned.clear();
        f.manifest = manifest(3, TRANSFORM_64, "[" + inverseEmbed() + ", {'name': 'lm_head.weight', 'axis': -1}]",
                ", 'tied_output': true");
        f.expectFailure(hdr + "tied Hadamard output must not carry a separate output head");
        f.extraConfig = "";
        f.manifest = manifest(3, TRANSFORM_64, "[" + inverseEmbed() + "]", ", 'tied_output': true");
        f.expectFailure(hdr + "tied Hadamard output requires tie_word_embeddings=true");
        f.extraConfig = ", 'tie_word_embeddings': true";
        f.manifest = manifest(2, TRANSFORM_64, "[" + inverseEmbed() + "]", "");
        f.expectFailure(hdr + "a tied latent embedding requires Hadamard schema 3 and tied_output=true");
    }

    @Test
    public void precA4FlagsAreWrittenSortedByName() throws IOException {
        Fixture f = new Fixture();
        Map<String, Boolean> flags = new LinkedHashMap<>();
        flags.put("blk.1.ffn_down.weight", false);
        flags.put("blk.0.attn_q.weight", false);
        flags.put("blk.0.attn_k.weight", true);
        f.precA4 = flags;
        HfMetadata m = f.apply();
        assertEquals(Arrays.asList("general.tensor_extra.name", "general.tensor_extra.prec_a4"),
                new ArrayList<>(m.keys()));
        assertEquals(Arrays.asList("blk.0.attn_k.weight", "blk.0.attn_q.weight", "blk.1.ffn_down.weight"),
                m.get("general.tensor_extra.name"));
        assertEquals(Arrays.asList(true, false, false), m.get("general.tensor_extra.prec_a4"));
        assertEquals(GGUFValueType.BOOL, m.elementType("general.tensor_extra.prec_a4"));
    }

    @Test
    public void noManifestAndNoFlagsWriteNothing() throws IOException {
        assertEquals(0, new Fixture().apply().size());
    }

    @Test
    public void foldedNamesListTheRecordsWithoutValidation() throws IOException {
        Fixture f = new Fixture();
        f.manifest = "{'tensors': [{'name': 'a.weight'}, {'axis': -1}, 7, {'name': 'b.weight', 'axis': 3}]}";
        assertEquals(Arrays.asList("a.weight", "b.weight"),
                new ArrayList<>(HfHadamardMetadata.foldedNames(f.context())));
        assertTrue(HfHadamardMetadata.foldedNames(new Fixture().context()).isEmpty());
    }

    @Test
    public void architecturesMatchTheRuntimeLoader() throws ReflectiveOperationException {
        Field f = PrismHadamardLoader.class.getDeclaredField("VERIFIED_ARCHS");
        f.setAccessible(true);
        assertEquals(f.get(null), HfHadamardMetadata.HADAMARD_ARCHS);
    }
}
