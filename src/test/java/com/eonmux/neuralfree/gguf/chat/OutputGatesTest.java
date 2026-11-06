package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintGate;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** Where a chat constraint starts: after open reasoning, after the harmony final header, or from the first token. */
public class OutputGatesTest {

    @Test
    public void aThinkingPromptGatesOnTheEndTagAndAClosedOneDoesNot() {
        ReasoningFilter think = ReasoningFilter.thinkTags();
        ConstraintGate open = OutputGates.forChat(think, "<|im_start|>assistant\n<think>\n", null);
        assertNotNull(open);
        assertTrue(open.triggers().words().containsAll(think.endTags()));
        assertNull(OutputGates.forChat(think, "<|im_start|>assistant\n<think>\n\n</think>\n\n", null));
        assertNull(OutputGates.forChat(think, null, null));
        assertNull(OutputGates.forChat(ReasoningFilter.none(), "<think>", null));
    }

    @Test
    public void harmonyGatesOnTheFinalAnswerHeaderOfItsDialect() {
        ReasoningFilter harmony = ReasoningFilter.harmony();
        ConstraintGate gptOss = OutputGates.forChat(harmony, "<|start|>assistant", "<|channel|>final<|message|>");
        assertEquals("<|channel|>analysis<|message|>hm<|end|><|start|>assistant".length(), gptOss.triggers()
                .firstStart("<|channel|>analysis<|message|>hm<|end|><|start|>assistant<|channel|>final"));
        ConstraintGate muse = OutputGates.forChat(harmony, "<|start|>assistant", "to=user<|message|>x<|eot|><|eom|>");
        assertEquals("a bare final header at the start of the output", 0, muse.triggers().firstStart("<|message|>"));
        assertEquals(" to=self<|message|>".length() + "hm<|eom|>".length(), muse.triggers().firstStart(
                " to=self<|message|>hm<|eom|><|start|>assistant<|message|>"));
        assertEquals("the reasoning message is not the answer", -1,
                muse.triggers().firstStart(" to=self<|message|>hm"));
        for (ConstraintGate gate : new ConstraintGate[]{gptOss, muse}) {
            assertTrue("the prefix rule compiles", GbnfBuilder.compile(ConstraintSpec.jsonObject(null)
                    .withGate(gate)).gated());
        }
    }
}
