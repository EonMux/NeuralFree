package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * The KV cache spec kind a pool's events name per KV group (vLLM's {@code get_kv_cache_spec_kind}): a
 * sliding-window group by its window rule, a group of MLA latent layers {@code mla_attention}, a full-attention
 * group {@code full_attention}, and no kind for a group that holds no KV layer.
 */
public class PrefixCacheGroupSpecTest {

    private static final int PAGE = 4;

    /** One KV group over {@code layers} layers of one window, the layouts the plan keeps. */
    private static PagedSequences memory(KvWindow window, boolean latent, int layers) {
        int nCtx = 32;
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(nCtx).ubatch(8).parallel(1).kvPageSize(PAGE).build();
        MemoryPlan.Builder plan = MemoryPlan.builder(MemoryKind.KV, nCtx, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(0, nCtx, false, window, PAGE, nCtx / PAGE));
        for (int l = 0; l < layers; l++) {
            KvLayout kv = latent
                    ? KvLayout.latent(nCtx, false, 8, 6, KvDType.F16, window)
                    : KvLayout.of(nCtx, false, 8, 8, KvDType.F16, KvDType.F16, window);
            plan.layer(LayerMemoryPlan.builder(l, LayerMemoryPlan.Kind.KV).kv(kv, 0).build());
        }
        return new PagedSequences(plan.config(cfg).build(), 1);
    }

    @Test
    public void aFullAttentionGroupReadsFullAttention() {
        assertEquals("full_attention", PrefixCache.specKind(memory(KvWindow.FULL_CAUSAL, false, 1), 0));
    }

    @Test
    public void aSlidingWindowGroupReadsItsWindowRule() {
        KvWindow swa = KvWindow.of(SwaType.STANDARD, 8, true);
        assertEquals("sliding_window", PrefixCache.specKind(memory(swa, false, 1), 0));
        KvWindow chunked = KvWindow.of(SwaType.CHUNKED, 16, true);
        assertEquals("chunked_local_attention", PrefixCache.specKind(memory(chunked, false, 1), 0));
        KvWindow symmetric = KvWindow.of(SwaType.SYMMETRIC, 8, true);
        assertEquals("sliding_window", PrefixCache.specKind(memory(symmetric, false, 1), 0));
    }

    @Test
    public void anMlaLatentGroupReadsMlaAttention() {
        assertEquals("mla_attention", PrefixCache.specKind(memory(KvWindow.FULL_CAUSAL, true, 1), 0));
    }

    @Test
    public void aGroupWithoutKvLayersNamesNoKind() {
        MemoryConfig cfg = MemoryConfig.builder().nCtx(32).ubatch(8).parallel(1).kvPageSize(PAGE).build();
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, 32, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(0, 32, false, KvWindow.FULL_CAUSAL, PAGE, 8))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                .config(cfg)
                .build();
        assertNull(PrefixCache.specKind(new PagedSequences(plan, 1), 0));
    }
}
