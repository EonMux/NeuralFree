package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ConstraintCompiler}: equal specs share one compilation, failures are never kept, the oldest unused goes at
 * capacity or past the heap budget, and a constraint too large to keep is compiled for its request alone.
 */
public class ConstraintCompilerTest {

    private static final String SCHEMA = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}";

    @Test
    public void anEqualSpecReturnsTheRememberedCompilation() {
        ConstraintCompiler compiler = new ConstraintCompiler(null);
        CompiledConstraint first = compiler.compile(ConstraintSpec.jsonSchema(SCHEMA,
                JsonSchemaToGbnf.WhitespaceMode.COMPACT));
        CompiledConstraint again = compiler.compile(ConstraintSpec.jsonSchema(SCHEMA,
                JsonSchemaToGbnf.WhitespaceMode.COMPACT));
        assertSame(first, again);
        assertEquals(1, compiler.size());
        assertEquals(rules(GbnfBuilder.compile(ConstraintSpec.jsonSchema(SCHEMA,
                JsonSchemaToGbnf.WhitespaceMode.COMPACT)).grammar()), rules(first.grammar()));
    }

    @Test
    public void specsThatDifferInAnyOptionCompileApart() {
        ConstraintCompiler compiler = new ConstraintCompiler(null);
        CompiledConstraint compact = compiler.compile(ConstraintSpec.jsonSchema(SCHEMA,
                JsonSchemaToGbnf.WhitespaceMode.COMPACT));
        CompiledConstraint flexible = compiler.compile(ConstraintSpec.jsonSchema(SCHEMA,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE));
        CompiledConstraint gated = compiler.compile(ConstraintSpec.gbnf("root ::= \"a\"")
                .withTriggers(GrammarTriggers.lazyWords(List.of("go"))));
        CompiledConstraint ungated = compiler.compile(ConstraintSpec.gbnf("root ::= \"a\""));
        assertNotSame(compact, flexible);
        assertNotSame(gated, ungated);
        assertEquals(true, gated.gated());
        assertEquals(false, ungated.gated());
        assertEquals(4, compiler.size());
    }

    @Test
    public void aFailingSpecIsCompiledAndRefusedEveryTime() {
        AtomicInteger lookups = new AtomicInteger();
        ConstraintCompiler compiler = new ConstraintCompiler(text -> {
            lookups.incrementAndGet();
            return new int[]{1, 2};
        });
        ConstraintSpec bad = ConstraintSpec.gbnf("root ::= <not-one-token>");
        assertThrows(IllegalArgumentException.class, () -> compiler.compile(bad));
        assertThrows(IllegalArgumentException.class, () -> compiler.compile(bad));
        assertEquals("the second request compiled again", 2, lookups.get());
        assertEquals(0, compiler.size());
    }

    @Test
    public void tokenReferencesResolveThroughTheCompilersVocabulary() {
        ConstraintCompiler compiler = new ConstraintCompiler(text -> new int[]{7});
        Grammar grammar = compiler.compile(ConstraintSpec.gbnf("root ::= <tool>")).grammar();
        GrammarElement first = grammar.rule(grammar.rootRuleIndex()).get(0);
        assertEquals(GrammarElement.Type.TOKEN, first.type());
        assertEquals(7, first.value());
    }

    private static List<List<GrammarElement>> rules(Grammar grammar) {
        List<List<GrammarElement>> out = new java.util.ArrayList<>();
        for (int i = 0; i < grammar.ruleCount(); i++) {
            out.add(grammar.rule(i));
        }
        return out;
    }

    @Test
    public void theLeastRecentlyUsedSpecIsDroppedAtCapacity() {
        ConstraintCompiler compiler = new ConstraintCompiler(null, 2);
        ConstraintSpec a = ConstraintSpec.gbnf("root ::= \"a\"");
        ConstraintSpec b = ConstraintSpec.gbnf("root ::= \"b\"");
        ConstraintSpec c = ConstraintSpec.gbnf("root ::= \"c\"");
        CompiledConstraint first = compiler.compile(a);
        CompiledConstraint second = compiler.compile(b);
        assertSame(first, compiler.compile(a));
        compiler.compile(c);
        assertEquals(2, compiler.size());
        assertSame("a was used after b, so b went", first, compiler.compile(a));
        assertNotSame(second, compiler.compile(b));
        assertThrows(IllegalArgumentException.class, () -> new ConstraintCompiler(null, 0));
    }

    @Test
    public void aConstraintTooLargeToKeepIsCompiledForItsRequestAlone() {
        ConstraintCompiler compiler = new ConstraintCompiler(null);
        StringBuilder source = new StringBuilder("root ::= \"w0\"");
        for (int i = 1; i < 30_000; i++) {
            source.append(" | \"w").append(i).append('"');
        }
        ConstraintSpec large = ConstraintSpec.gbnf(source.toString());
        CompiledConstraint first = compiler.compile(large);
        assertTrue("the grammar is over a quarter of the budget",
                ConstraintCompiler.estimatedBytes(large, first) > ConstraintCompiler.DEFAULT_BUDGET_BYTES / 4);
        assertNotSame(first, compiler.compile(large));
        assertEquals(0, compiler.size());
        assertEquals(0, compiler.heldBytes());
        ConstraintSpec small = ConstraintSpec.jsonSchema(SCHEMA, JsonSchemaToGbnf.WhitespaceMode.COMPACT);
        assertSame(compiler.compile(small), compiler.compile(small));
        assertEquals(1, compiler.size());
    }

    @Test
    public void theHeapBudgetDropsTheLeastRecentlyUsedSpecs() {
        List<ConstraintSpec> specs = new java.util.ArrayList<>();
        for (char c = 'a'; c <= 'e'; c++) {
            specs.add(ConstraintSpec.gbnf("root ::= \"" + String.valueOf(c).repeat(200) + "\""));
        }
        long each = ConstraintCompiler.estimatedBytes(specs.get(0), GbnfBuilder.compile(specs.get(0), null));
        long budget = 4 * each + each / 2;
        ConstraintCompiler compiler = new ConstraintCompiler(null, ConstraintCompiler.DEFAULT_CAPACITY, budget);
        List<CompiledConstraint> compiled = new java.util.ArrayList<>();
        for (ConstraintSpec spec : specs) {
            compiled.add(compiler.compile(spec));
            assertTrue("held " + compiler.heldBytes() + " of " + budget, compiler.heldBytes() <= budget);
        }
        assertEquals("four specs fit the budget", 4, compiler.size());
        assertEquals(4 * each, compiler.heldBytes());
        assertSame(compiled.get(4), compiler.compile(specs.get(4)));
        assertNotSame("the first spec went", compiled.get(0), compiler.compile(specs.get(0)));
        assertThrows(IllegalArgumentException.class, () -> new ConstraintCompiler(null, 1, 0));
    }
}
