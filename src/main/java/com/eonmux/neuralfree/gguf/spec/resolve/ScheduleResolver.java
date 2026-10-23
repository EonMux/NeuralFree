package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Resolves the trunk execution order ({@link LayerSchedule}) and separates NextN (multi-token-prediction) blocks
 * from the trunk.
 *
 * <ul>
 *   <li>nanbeige {@code num_loops > 1}: the block count already holds {@code n_phys * num_loops} slots; slot
 *       {@code s} aliases physical weights {@code s % n_phys} (nanbeige.cpp:13-31, 164-171). Per-layer resolvers
 *       must be called with {@link LayerSchedule#weightLayer(int)} of the slot.</li>
 *   <li>hrm_text: {@code layers_per_stack * h_cycles * (l_cycles + 1)} slots over two stacks (hrm-text.cpp:6-32).</li>
 *   <li>otherwise one slot per trunk layer.</li>
 * </ul>
 * NextN blocks {@code [n_layer, n_layer_all)} are never trunk slots. They exist only when
 * {@code blk.<n_layer>.nextn.eh_proj.weight} is present ({@code nextn_flags}, llama-model.cpp:3900-3920).
 * deepseek4 folds absent NextN blocks into the trunk ({@code n_layer_nextn = 0}, deepseek4.cpp:18-25) before the
 * hparams are built ({@code ModelSpecBuilder.withNextnProbe}), so its {@code n_layer} already counts them.
 * gemma4-assistant has no {@code eh_proj}: every block is a NextN block
 * that llama.cpp always builds (models/gemma4-assistant.cpp:47), so its trunk is empty
 * ({@link LayerSchedule#empty()}). The granite-switch router block occupies the NextN slot and is not a NextN
 * block. Immutable.
 */
public final class ScheduleResolver {

    private final ModelHparams hp;
    private final int trunkLayers;
    private final int nextnLayers;
    private final LayerSchedule schedule;

    /**
     * @throws IllegalStateException when the loop count does not divide the block count, loops are combined with
     *                               NextN blocks, or the schedule is otherwise inconsistent with the hparams
     */
    public ScheduleResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        Objects.requireNonNull(tc, "tc");
        int trunk = hp.nLayer();
        int nextn = hp.nLayerAll() - trunk;
        if (hp.routerLayer() >= 0) {
            nextn = 0;
        } else if (hp.arch() == Arch.GEMMA4_ASSISTANT) {
            if (trunk != 0) {
                throw new IllegalStateException(hp.arch().ggufName() + ": every block is an assistant block"
                        + " (nextn_predict_layers = block_count), but " + trunk + " trunk layers remain");
            }
        } else if (nextn > 0 && !tc.has(TensorNames.layerName(trunk, TensorNames.weight(TensorNames.NEXTN_EH_PROJ)))) {
            nextn = 0;
        }
        this.trunkLayers = trunk;
        this.nextnLayers = nextn;
        this.schedule = buildSchedule(trunk, nextn);
    }

    private LayerSchedule buildSchedule(int trunk, int nextn) {
        String arch = hp.arch().ggufName();
        if (trunk == 0 && nextn > 0) {
            return LayerSchedule.empty();
        }
        int loops = hp.numLoops();
        if (loops > 1) {
            if (trunk % loops != 0) {
                throw new IllegalStateException(arch + ": " + trunk + " looped slots are not a multiple of " + loops
                        + " loops");
            }
            if (nextn > 0) {
                throw new IllegalStateException(arch + ": looped schedules with NextN blocks are not built by llama.cpp");
            }
            return LayerSchedule.looped(trunk / loops, loops, hp.skipLoopFinalNorm());
        }
        if (hp.hrmLayersPerStack() > 0) {
            LayerSchedule s = LayerSchedule.hrm(hp.hrmLayersPerStack(), hp.hrmHCycles(), hp.hrmLCycles());
            if (s.slots() != trunk) {
                throw new IllegalStateException(arch + ": HRM schedule has " + s.slots() + " slots but the block count is "
                        + trunk);
            }
            return s;
        }
        return LayerSchedule.linear(trunk);
    }

    /** Trunk execution order. */
    public LayerSchedule schedule() {
        return schedule;
    }

    /** Trunk layers, i.e. slots of {@link #schedule()} (deepseek4 without NextN weights: {@code n_layer_all}). */
    public int trunkLayers() {
        return trunkLayers;
    }

    /** NextN blocks present in the file, starting at layer {@link #trunkLayers()}; 0 when excluded. */
    public int nextnLayers() {
        return nextnLayers;
    }

    /** True when layer {@code il} is a NextN block present in the file. */
    public boolean isNextnLayer(int il) {
        return il >= trunkLayers && il < trunkLayers + nextnLayers;
    }

    /** granite-switch router block index, or -1. */
    public int routerLayer() {
        return hp.routerLayer();
    }

    /** Physical weight layer of trunk slot {@code slot} (loop aliasing). */
    public int weightLayer(int slot) {
        return schedule.weightLayer(slot);
    }
}
