package com.eonmux.neuralfree.gguf.hub.slice;

import com.eonmux.neuralfree.io.safetensors.SafetensorsHeader;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SlicePlanner} and {@link ConfigAnalysis}: layer families renumbered and their config counts patched, the
 * dense-first-layers rule, the MTP tail, explicit layer lists, expert families and their stacked and router tensors
 * row-sliced, vocab slicing, per-layer config lists, the config-less fallback and the refusal of an out-of-range
 * {@code --keep-layers}. The shards are real safetensors files the writer produces, so the shapes and offsets the
 * planner reads are the format's.
 */
public class SlicePlannerTest {

    private static final String SHARD = "model.safetensors";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== fixtures

    /** One tensor of a test shard: a name, a dtype, a shape and the bytes {@code F32} or {@code U8} need. */
    private static final class Spec {

        private final String name;
        private final long[] shape;

        private Spec(String name, long... shape) {
            this.name = name;
            this.shape = shape;
        }
    }

    private static Spec f32(String name, long... shape) {
        return new Spec(name, shape);
    }

    private SafetensorsHeader shard(Spec... specs) throws IOException {
        SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        for (Spec spec : specs) {
            long numel = 1L;
            for (long d : spec.shape) {
                numel *= d;
            }
            builder.add(spec.name, StDType.F32, spec.shape, new byte[(int) (numel * 4L)]);
        }
        Path path = tmp.newFile("shard-" + System.nanoTime() + ".safetensors").toPath();
        builder.build().write(path);
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return SafetensorsHeader.read(channel, SHARD);
        }
    }

    private static Object config(String json) {
        return Json.parse(json, Json.Limits.PYTHON);
    }

    private static SlicePlan plan(SafetensorsHeader header, Object config, SliceOptions options) {
        Map<String, SafetensorsHeader> headers = new LinkedHashMap<>();
        headers.put(SHARD, header);
        return SlicePlanner.plan(Collections.singletonList(SHARD), headers, config, options);
    }

    private static SliceOptions.Builder options() {
        return SliceOptions.builder().dryRun(true).numLayers(2);
    }

    private static TensorSlice tensor(SlicePlan plan, String name) {
        for (TensorSlice tensor : plan.tensors()) {
            if (tensor.outputName().equals(name)) {
                return tensor;
            }
        }
        return null;
    }

    private static List<String> outputNames(SlicePlan plan) {
        List<String> names = new ArrayList<>();
        for (TensorSlice tensor : plan.tensors()) {
            names.add(tensor.outputName());
        }
        Collections.sort(names);
        return names;
    }

    private SafetensorsHeader layers(int count) throws IOException {
        List<Spec> specs = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            specs.add(f32("model.layers." + i + ".self_attn.q_proj.weight", 2));
        }
        return shard(specs.toArray(new Spec[0]));
    }

    // ================================================================== layers

    @Test
    public void keepsTheLeadingLayersAndRenumbersThem() throws IOException {
        SafetensorsHeader header = shard(
                f32("model.layers.0.self_attn.q_proj.weight", 2),
                f32("model.layers.1.self_attn.q_proj.weight", 2),
                f32("model.layers.2.self_attn.q_proj.weight", 2),
                f32("model.layers.3.self_attn.q_proj.weight", 2),
                f32("model.embed_tokens.weight", 128, 2),
                f32("model.norm.weight", 2),
                f32("lm_head.weight", 128, 2));
        SlicePlan plan = plan(header, config("{\"num_hidden_layers\": 4, \"vocab_size\": 128}"),
                options().build());
        assertEquals(7, plan.totalTensors());
        assertEquals(5, plan.tensors().size());
        assertNotNull(tensor(plan, "model.layers.0.self_attn.q_proj.weight"));
        assertNotNull(tensor(plan, "model.layers.0.self_attn.q_proj.weight"));
        assertNotNull(tensor(plan, "model.layers.1.self_attn.q_proj.weight"));
        assertNull(tensor(plan, "model.layers.2.self_attn.q_proj.weight"));
        assertNotNull(tensor(plan, "model.embed_tokens.weight"));
        assertNotNull(tensor(plan, "lm_head.weight"));
        assertTrue(plan.hasLayerFamily());

        List<String> patches = plan.patchConfig(null);
        assertTrue(patches.toString(), patches.contains("num_hidden_layers: 4 -> 2"));
    }

    @Test
    public void keepsAnExplicitLayerListAndRenumbersIt() throws IOException {
        SafetensorsHeader header = shard(
                f32("model.layers.0.self_attn.q_proj.weight", 2),
                f32("model.layers.1.self_attn.q_proj.weight", 2),
                f32("model.layers.2.self_attn.q_proj.weight", 2),
                f32("model.layers.3.self_attn.q_proj.weight", 2));
        SlicePlan plan = plan(header, config("{\"num_hidden_layers\": 4}"),
                options().keepLayers(java.util.Arrays.asList(3, 1)).build());
        assertEquals(java.util.Arrays.asList("model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.self_attn.q_proj.weight"), outputNames(plan));
        TensorSlice renumbered = tensor(plan, "model.layers.0.self_attn.q_proj.weight");
        assertEquals("model.layers.1.self_attn.q_proj.weight", renumbered.sourceName());
        assertTrue(plan.patchConfig(null).contains("num_hidden_layers: 4 -> 2"));
    }

    @Test
    public void refusesAKeepLayersIndexOutsideTheRegularLayers() throws IOException {
        SafetensorsHeader header = shard(f32("model.layers.0.self_attn.q_proj.weight", 2),
                f32("model.layers.1.self_attn.q_proj.weight", 2));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> plan(header, config("{\"num_hidden_layers\": 2}"),
                        options().keepLayers(java.util.Arrays.asList(0, 5)).build()));
        assertTrue(e.getMessage(), e.getMessage().contains("keep-layers index 5"));
        assertTrue(e.getMessage(), e.getMessage().contains("2 regular layers"));
    }

    @Test
    public void keepsOneMoeLayerWhenTheFirstLayersAreDense() throws IOException {
        SafetensorsHeader header = layers(8);
        SlicePlan plan = plan(header, config("{\"num_hidden_layers\": 8, \"first_k_dense_replace\": 2}"),
                options().numLayers(2).build());
        assertEquals(java.util.Arrays.asList("model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.self_attn.q_proj.weight"), outputNames(plan));
        TensorSlice second = tensor(plan, "model.layers.1.self_attn.q_proj.weight");
        assertEquals("model.layers.2.self_attn.q_proj.weight", second.sourceName());
        List<String> patches = plan.patchConfig(null);
        assertTrue(patches.toString(), patches.contains("first_k_dense_replace: 2 -> 1"));
        assertTrue(patches.toString(), patches.contains("num_hidden_layers: 8 -> 2"));
    }

    @Test
    public void keepsTheMtpTailAfterTheKeptRegularLayers() throws IOException {
        SafetensorsHeader header = layers(3);
        SlicePlan plan = plan(header,
                config("{\"num_hidden_layers\": 2, \"num_nextn_predict_layers\": 1}"),
                options().numLayers(1).build());
        assertEquals(java.util.Arrays.asList("model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.self_attn.q_proj.weight"), outputNames(plan));
        assertEquals("model.layers.2.self_attn.q_proj.weight",
                tensor(plan, "model.layers.1.self_attn.q_proj.weight").sourceName());
        List<String> patches = plan.patchConfig(null);
        assertTrue(patches.toString(), patches.contains("num_hidden_layers: 2 -> 1"));
        assertFalse(patches.toString(), patches.contains("num_nextn_predict_layers"));
    }

    @Test
    public void findsTheLayerFamilyWithoutAConfig() throws IOException {
        SlicePlan plan = plan(layers(4), null, options().numLayers(2).build());
        assertEquals(2, plan.tensors().size());
        assertTrue(plan.hasLayerFamily());
        assertTrue(plan.patchConfig(null).isEmpty());
    }

    @Test
    public void cutsAPerLayerListToTheKeptLayers() throws IOException {
        SafetensorsHeader header = shard(
                f32("model.layers.0.self_attn.q_proj.weight", 2),
                f32("model.layers.1.self_attn.q_proj.weight", 2),
                f32("model.layers.2.self_attn.q_proj.weight", 2),
                f32("model.layers.3.self_attn.q_proj.weight", 2));
        Object config = config("{\"num_hidden_layers\": 4, \"layer_types\":"
                + "[\"full\",\"sliding\",\"full\",\"sliding\"]}");
        SlicePlan plan = plan(header, config, options().keepLayers(java.util.Arrays.asList(1, 3)).build());
        List<String> patches = plan.patchConfig(null);
        assertTrue(patches.toString(), patches.contains("layer_types: list[4] -> list[2]"));
        assertEquals(java.util.Arrays.asList("sliding", "sliding"),
                ((Map<?, ?>) config).get("layer_types"));
    }

    // ================================================================== experts

    @Test
    public void keepsTheFirstExpertsAndRowSlicesTheirStackedAndRouterTensors() throws IOException {
        List<Spec> specs = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            specs.add(f32("model.layers.0.mlp.experts." + i + ".w1.weight", 2));
        }
        specs.add(f32("model.layers.0.mlp.gate.weight", 8, 2));
        specs.add(f32("model.layers.0.mlp.experts.gate_up_proj", 8, 2));
        specs.add(f32("model.layers.1.mlp.experts.0.w1.weight", 2));
        SafetensorsHeader header = shard(specs.toArray(new Spec[0]));
        Object config = config("{\"num_hidden_layers\": 2, \"num_experts\": 8,"
                + " \"num_experts_per_tok\": 2, \"n_shared_experts\": 1}");
        SlicePlan plan = plan(header, config, options().numLayers(1).numExperts(3).build());

        for (int i = 0; i < 3; i++) {
            assertNotNull("expert " + i, tensor(plan, "model.layers.0.mlp.experts." + i + ".w1.weight"));
        }
        assertNull(tensor(plan, "model.layers.0.mlp.experts.3.w1.weight"));
        assertNull(tensor(plan, "model.layers.1.mlp.experts.0.w1.weight"));

        TensorSlice router = tensor(plan, "model.layers.0.mlp.gate.weight");
        assertNotNull(router);
        assertArrayEquals(new long[]{3, 2}, router.outputShape());
        assertEquals(3L * 2L * 4L, router.byteLength());
        assertTrue(router.rowSliced());

        TensorSlice stacked = tensor(plan, "model.layers.0.mlp.experts.gate_up_proj");
        assertArrayEquals(new long[]{3, 2}, stacked.outputShape());
        assertEquals(24L, stacked.byteLength());

        List<String> patches = plan.patchConfig(null);
        assertTrue(patches.toString(), patches.contains("num_experts: 8 -> 3"));
        assertFalse("top-k 2 stays under 3 experts", patches.toString().contains("num_experts_per_tok"));
        assertFalse("shared experts are not routed experts", patches.toString().contains("n_shared_experts"));
        assertEquals(2L, ((Map<?, ?>) config).get("num_experts_per_tok"));
    }

    @Test
    public void bringsATopKAboveTheKeptExpertsDownToThem() throws IOException {
        List<Spec> specs = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            specs.add(f32("model.layers.0.mlp.experts." + i + ".w1.weight", 2));
        }
        SafetensorsHeader header = shard(specs.toArray(new Spec[0]));
        Object config = config("{\"num_hidden_layers\": 1, \"num_experts\": 8,"
                + " \"num_experts_per_tok\": 6}");
        SlicePlan plan = plan(header, config, options().numLayers(1).numExperts(3).build());
        List<String> patches = plan.patchConfig(null);
        assertTrue(patches.toString(), patches.contains("num_experts_per_tok: 6 -> 3"));
    }

    // ================================================================== vocab

    @Test
    public void rowSlicesVocabSizedTensorsAndPatchesTheVocabCount() throws IOException {
        SafetensorsHeader header = shard(
                f32("model.embed_tokens.weight", 128, 2),
                f32("model.layers.0.self_attn.q_proj.weight", 2),
                f32("lm_head.weight", 110, 2),
                f32("extra.weight", 2000, 2));
        Object config = config("{\"num_hidden_layers\": 1, \"vocab_size\": 100,"
                + " \"padded_vocab_size\": 128}");
        SlicePlan plan = plan(header, config, options().numLayers(1).vocabSize(16).build());

        TensorSlice embedding = tensor(plan, "model.embed_tokens.weight");
        assertArrayEquals(new long[]{16, 2}, embedding.outputShape());
        assertEquals(128L, embedding.byteLength());
        TensorSlice padded = tensor(plan, "lm_head.weight");
        assertArrayEquals(new long[]{16, 2}, padded.outputShape());
        TensorSlice far = tensor(plan, "extra.weight");
        assertArrayEquals("2000 is more than 1024 rows over the declared vocab", new long[]{2000, 2},
                far.outputShape());

        List<String> patches = plan.patchConfig(16);
        assertTrue(patches.toString(), patches.contains("vocab_size: 100 -> 16"));
        assertTrue(patches.toString(), patches.contains("padded_vocab_size: 128 -> 16"));
    }

    @Test
    public void leavesTheVocabWholeWhenTheOptionIsOff() throws IOException {
        SafetensorsHeader header = shard(f32("model.embed_tokens.weight", 128, 2));
        SlicePlan plan = plan(header, config("{\"num_hidden_layers\": 1, \"vocab_size\": 100}"),
                options().numLayers(1).build());
        assertArrayEquals(new long[]{128, 2}, tensor(plan, "model.embed_tokens.weight").outputShape());
        assertTrue(plan.patchConfig(null).isEmpty());
    }

    // ================================================================== source spans

    @Test
    public void sourceSpansAreAbsoluteOffsetsOfTheShard() throws IOException {
        SafetensorsHeader header = shard(f32("model.layers.0.self_attn.q_proj.weight", 2),
                f32("model.layers.1.self_attn.q_proj.weight", 3));
        SlicePlan plan = plan(header, config("{\"num_hidden_layers\": 2}"), options().numLayers(1).build());
        TensorSlice slice = tensor(plan, "model.layers.0.self_attn.q_proj.weight");
        assertEquals(header.dataStart() + header.tensor("model.layers.0.self_attn.q_proj.weight").begin(),
                slice.sourceBegin());
        assertEquals(8L, slice.byteLength());
    }
}
