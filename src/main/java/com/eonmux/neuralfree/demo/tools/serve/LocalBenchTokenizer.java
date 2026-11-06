package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;

import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * The benchmark's local tokenizer: NeuralFree's own loader opens a GGUF file or a Hugging Face
 * checkpoint directory and its tokenizer counts the prompts, exactly as the server tokenizer would,
 * so a text dataset needs no {@code POST /tokenize} round trip. The model's weights are never read:
 * {@link ModelSources#open} parses the metadata and the tokenizer files only.
 */
final class LocalBenchTokenizer implements BenchTokenizer {

    private final GGUFTokenizer tokenizer;

    private LocalBenchTokenizer(GGUFTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    /**
     * The tokenizer of the GGUF file or Hugging Face directory at {@code path}.
     *
     * @throws IllegalArgumentException naming the path when it holds no readable tokenizer
     */
    static LocalBenchTokenizer open(String path) {
        try {
            return new LocalBenchTokenizer(new GGUFTokenizer(ModelSources.open(path)));
        } catch (IOException e) {
            throw new IllegalArgumentException("cannot load the local tokenizer from '" + path + "': "
                    + e.getMessage(), e);
        }
    }

    @Override
    public int[] tokenize(String text) {
        return tokenizer.encode(text);
    }

    @Override
    public String detokenize(int[] tokens) {
        return tokenizer.decode(tokens);
    }

    @Override
    public int vocabSize() {
        return tokenizer.vocabSize();
    }

    @Override
    public int[] specialIds() {
        SpecialTokens special = tokenizer.specialTokens();
        Set<Integer> ids = new LinkedHashSet<>();
        add(ids, special.bos());
        add(ids, special.eos());
        add(ids, special.eot());
        add(ids, special.eom());
        add(ids, special.unk());
        add(ids, special.sep());
        add(ids, special.pad());
        add(ids, special.mask());
        add(ids, special.fimPre());
        add(ids, special.fimSuf());
        add(ids, special.fimMid());
        add(ids, special.fimPad());
        add(ids, special.fimRep());
        add(ids, special.fimSep());
        int[] out = new int[ids.size()];
        int i = 0;
        for (int id : ids) {
            out[i++] = id;
        }
        return out;
    }

    @Override
    public BenchChatTemplate chatTemplate() {
        return (messages, kwargs) -> tokenizer.formatChat(ChatRequest.builder()
                .messages(messages)
                .templateKwargs(kwargs)
                .build());
    }

    private static void add(Set<Integer> ids, int id) {
        if (id >= 0) {
            ids.add(id);
        }
    }
}
