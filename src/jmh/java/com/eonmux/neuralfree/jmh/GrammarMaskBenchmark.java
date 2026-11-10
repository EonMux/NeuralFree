package com.eonmux.neuralfree.jmh;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.TimeUnit;

/**
 * The grammar mask of a JSON-object response format over a {@value SyntheticVocab#SIZE}-token vocabulary
 * ({@link GrammarStage} at the head of a greedy chain, {@link Sampler#apply}): every candidate's piece is matched
 * against the grammar's stacks. {@code object-start} is the state after {@code {}, where the first byte rejects
 * most pieces; {@code in-key} is the state inside a member name, where nearly every piece is walked to its end.
 * {@code vocabFile} names a GGUF model whose tokenizer's vocabulary replaces the synthetic one (for example
 * {@code -p vocabFile=Qwen3-0.6B-Q8_0.gguf}); the pieces the states step through are looked up in it.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class GrammarMaskBenchmark {

    @Param({"object-start", "in-key"})
    public String state;

    /** A GGUF model whose vocabulary the mask runs over; empty for the synthetic vocabulary. */
    @Param({""})
    public String vocabFile;

    private ModelSource source;
    private Sampler sampler;
    private float[] logits;

    @Setup
    public void setUp() throws IOException {
        SamplerVocab vocab;
        if (vocabFile.isEmpty()) {
            vocab = SyntheticVocab.of(3);
        } else {
            source = ModelSources.open(vocabFile);
            vocab = Sampler.vocab(new GGUFTokenizer(source));
        }
        Grammar grammar = JsonSchemaToGbnf.convert("{\"type\": \"object\"}", JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
        sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.singletonList(new GrammarStage(grammar, vocab)));
        sampler.accept(idOf(vocab, "{"));
        if (state.equals("in-key")) {
            sampler.accept(idOf(vocab, "\""));
            sampler.accept(idOf(vocab, "name"));
        } else if (!state.equals("object-start")) {
            throw new IllegalArgumentException("unknown state " + state);
        }
        logits = SamplerChainBenchmark.logits(vocab.vocabSize(), 17);
        sampler.apply(logits);
    }

    @TearDown
    public void tearDown() throws Exception {
        if (source != null) {
            source.close();
        }
    }

    /** The id whose piece is exactly {@code text}. */
    private static int idOf(SamplerVocab vocab, String text) {
        byte[] want = text.getBytes(StandardCharsets.UTF_8);
        for (int id = 0; id < vocab.vocabSize(); id++) {
            if (Arrays.equals(vocab.piece(id), want)) {
                return id;
            }
        }
        throw new IllegalArgumentException("the vocabulary holds no piece '" + text + "'");
    }

    @Benchmark
    public int mask() {
        return sampler.apply(logits).selectedId();
    }
}
