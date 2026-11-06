package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Printing a compiled grammar back to GBNF ({@link GbnfPrinter}) and the gate a constraint waits behind
 * ({@link ConstraintGate}): the printed text parses to the same rules, and a gated constraint compiles into a lazy
 * grammar that matches the gate word, optional whitespace and then the constraint.
 */
public class ConstraintGateTest {

    @Test
    public void aPrintedGrammarParsesToTheSameRules() {
        String source = "root ::= (\"ab\" [c-eX] | [^\\n\\\"]+ item{2,3})? . <[5]> !<[6]>\n"
                + "item ::= \"[\" [0-9]* \"]\" |\n";
        Grammar g = GbnfParser.parse(source);
        String printed = "root ::= " + GbnfPrinter.ruleName(g, g.rootRuleIndex(), "p-") + "\n"
                + GbnfPrinter.print(g, "p-");
        Grammar back = GbnfParser.parse(printed);
        for (int id = 0; id < g.ruleCount(); id++) {
            int copy = back.symbolId(GbnfPrinter.ruleName(g, id, "p-"));
            assertEquals(g.symbolName(id), g.rule(id).size(), back.rule(copy).size());
            for (int k = 0; k < g.rule(id).size(); k++) {
                GrammarElement a = g.rule(id).get(k);
                GrammarElement b = back.rule(copy).get(k);
                assertEquals(a.type(), b.type());
                if (a.type() == GrammarElement.Type.RULE_REF) {
                    assertEquals(GbnfPrinter.ruleName(g, a.value(), "p-"), back.symbolName(b.value()));
                } else {
                    assertEquals(a.value(), b.value());
                }
            }
        }
    }

    @Test
    public void aGatedSchemaWaitsForTheWordThenHoldsTheOutput() {
        ConstraintSpec spec = ConstraintSpec.jsonSchema("{\"type\": \"object\", \"properties\": {\"a\": "
                + "{\"type\": \"integer\"}}, \"required\": [\"a\"]}", null)
                .withGate(ConstraintGate.afterWords(Arrays.asList("</think>")));
        CompiledConstraint c = GbnfBuilder.compile(spec);
        assertTrue(c.gated());
        Grammar g = c.grammar();
        assertTrue(GrammarMatcher.matches(g, "</think>\n\n{\"a\": 1}"));
        assertTrue(GrammarMatcher.matches(g, "</think>{\"a\":2}"));
        assertFalse(GrammarMatcher.matches(g, "</think>\n\nThe answer is 1."));
        assertFalse("the grammar starts at the gate word", GrammarMatcher.matches(g, "{\"a\": 1}"));
    }

    @Test
    public void aRegexAndAChoiceComposeBehindAPrefixRule() {
        ConstraintGate harmony = new ConstraintGate(GrammarTriggers.lazyWords(Collections.singletonList(
                "<|channel|>final")),
                "\"<|channel|>final\" ( [ \\t\\n]* \"<|constrain|>\"? [A-Za-z0-9_-]+ )? \"<|message|>\"");
        Grammar regex = GbnfBuilder.compile(ConstraintSpec.regex("[0-9]{3}", false).withGate(harmony)).grammar();
        assertTrue(GrammarMatcher.matches(regex, "<|channel|>final<|message|>123"));
        assertTrue(GrammarMatcher.matches(regex, "<|channel|>final <|constrain|>json<|message|>456"));
        assertFalse(GrammarMatcher.matches(regex, "<|channel|>final<|message|>12"));
        Grammar choice = GbnfBuilder.compile(ConstraintSpec.choice(Arrays.asList("yes", "no")).withGate(
                ConstraintGate.afterWords(Arrays.asList("</think>")))).grammar();
        assertTrue(GrammarMatcher.matches(choice, "</think> no"));
        assertFalse(GrammarMatcher.matches(choice, "</think> maybe"));
    }

    @Test
    public void aConstraintWithItsOwnTriggersIgnoresTheGate() {
        GrammarTriggers own = GrammarTriggers.lazyWords(Collections.singletonList("<x>"));
        CompiledConstraint c = GbnfBuilder.compile(ConstraintSpec.gbnf("root ::= \"<x>\" \"a\"").withTriggers(own)
                .withGate(ConstraintGate.afterWords(Arrays.asList("</think>"))));
        assertEquals(own, c.triggers());
        assertTrue(GrammarMatcher.matches(c.grammar(), "<x>a"));
        assertNull(GbnfBuilder.compile(ConstraintSpec.gbnf("root ::= \"a\"")).triggers());
    }

    @Test
    public void malformedGatesRefuse() {
        GrammarTriggers words = GrammarTriggers.lazyWords(Collections.singletonList("x"));
        assertThrows(IllegalArgumentException.class, () -> new ConstraintGate(words, " "));
        assertThrows(IllegalArgumentException.class, () -> new ConstraintGate(
                GrammarTriggers.builder(GrammarTriggers.Mode.DISPATCH).word("x").build(), "\"x\""));
    }
}
