package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

/**
 * A test-only {@code LlamaForCausalLM} entry: the converter's default tensor mapping (no Q/K permutation), the generic
 * metadata and a GPT-2 vocabulary, plus a draft-head convention modelled on {@code _QwenMtpMixin}
 * (LC/conversion/qwen.py:318-358): checkpoint tensors {@code mtp.layers.<i>.<rest>} are extra decoder layers.
 * <ul>
 *   <li>The block count is indexed: the configuration's layers plus the {@code mtp.layers.<i>} the checkpoint
 *       holds.</li>
 *   <li>The name filter is load-scoped: it renames {@code mtp.layers.<i>.<rest>} to
 *       {@code model.layers.<n + i>.<rest>} with {@code n} the configuration's layer count of that load, and drops
 *       them when the load excludes MTP layers.</li>
 * </ul>
 * Installed through {@link com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry#withExtraForTest}.
 */
final class TestOnlyLlamaEntry {

    /** The checkpoint prefix of the extra layers. */
    static final String MTP_LAYERS = "mtp.layers.";

    private TestOnlyLlamaEntry() {
    }

    /** The entry. */
    static HfArchEntry entry() {
        return HfArchEntry.of(Collections.singletonList("LlamaForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                        HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, TestOnlyLlamaEntry::vocab)
                .withLoadNameFilter(TestOnlyLlamaEntry::nameFilter)
                .withIndexedBlockCount(TestOnlyLlamaEntry::blockCount)
                .withMtpExport(true)
                .withMtpOnlyRootKeep("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight");
    }

    private static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfVocabWriters.Input.of(ctx.config(), ctx.tokenizerFiles(),
                PreTokenizerCanonicalizer::toSpec).withGgufArch(ctx.arch().ggufName());
        return HfVocabWriters.gpt2(in, pre);
    }

    /** The filter of one load: it reads the configuration's layer count and the MTP option of {@code ctx}. */
    static HfNameFilter nameFilter(HfLoadContext ctx) {
        int base = ctx.config().blockCount();
        boolean keep = ctx.options().includeMtp();
        return (name, inherited) -> {
            if (!name.startsWith(MTP_LAYERS)) {
                return inherited.apply(name);
            }
            if (!keep) {
                return null;
            }
            String[] parts = name.substring(MTP_LAYERS.length()).split("\\.", 2);
            if (parts.length != 2 || !parts[0].matches("\\d{1,4}")) {
                return inherited.apply(name);
            }
            return inherited.apply("model.layers." + (base + Integer.parseInt(parts[0])) + "." + parts[1]);
        };
    }

    /** The configuration's layers plus the extra layers named in the load's index. */
    static int blockCount(HfLoadContext ctx) {
        int extra = 0;
        for (String name : ctx.index().names()) {
            if (name.startsWith(MTP_LAYERS)) {
                String[] parts = name.substring(MTP_LAYERS.length()).split("\\.", 2);
                if (parts[0].matches("\\d{1,4}")) {
                    extra = Math.max(extra, Integer.parseInt(parts[0]) + 1);
                }
            }
        }
        return ctx.config().blockCount() + extra;
    }
}
