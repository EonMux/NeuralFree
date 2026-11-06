package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The pieces of two-level verification ({@code neuralfree.draft.tlv}) that run without a model: the composed
 * proposal the exact target verifies — the Q round's chain as a spine, the view's bonus as their child, the
 * anchored draft hung below it — always a well-formed tree within the fixed width, and the {@code neuralfree.*}
 * keys that switch the round on and size its copy.
 */
public class TwoLevelVerifyTest {

    private static TwoLevelVerify.Chain chain(int root, int[] accepted, int bonus) {
        int[] tokens = new int[accepted.length + 1];
        tokens[0] = root;
        System.arraycopy(accepted, 0, tokens, 1, accepted.length);
        return new TwoLevelVerify.Chain(tokens, bonus);
    }

    @Test
    public void composeBuildsSpineBonusAndSubtree() {
        // the Q round accepted 70, 71 after the pending token 5 and drew bonus 80; the anchored draft proposes
        // 90 below the bonus and 91 below it
        TwoLevelVerify.Chain c = chain(5, new int[] {70, 71}, 80);
        DraftTree sub = new DraftTree(Arrays.asList(
                new DraftTree.Node(90, -1, 1, 0),
                new DraftTree.Node(91, 0, 2, 0)));
        DraftTree composed = TwoLevelVerify.compose(c, 3, sub);

        // nodes: the spine 70 (depth 1), 71 (depth 2), the bonus 80 (depth 3), then the shifted subtree
        assertEquals(Arrays.asList(70, 71, 80, 90, 91), tokenList(composed));
        int[] parents = new int[] {-1, 0, 1, 2, 3};
        for (int i = 0; i < composed.size(); i++) {
            assertEquals("node " + i + "'s parent", parents[i], composed.nodes().get(i).parent());
        }
        assertEquals(Arrays.asList(1, 2, 3, 4, 5), depthList(composed));
        // the anchored draft keeps its branch marks
        assertEquals(0, composed.nodes().get(4).branch());
        // the composed tree is a single chain, which the fixed-width verify handles like any tree
        assertTrue(composed.linear());
    }

    @Test
    public void composeBranchesTheSubtreeUnderTheBonus() {
        TwoLevelVerify.Chain c = chain(5, new int[0], 80);
        DraftTree sub = new DraftTree(Arrays.asList(
                new DraftTree.Node(90, -1, 1, 0),
                new DraftTree.Node(91, -1, 1, 1),
                new DraftTree.Node(92, 1, 2, 1)));
        DraftTree composed = TwoLevelVerify.compose(c, 1, sub);

        // the bonus at index 0, both roots its children at depths 2, the child below the second root at depth 3
        assertEquals(Arrays.asList(80, 90, 91, 92), tokenList(composed));
        assertEquals(-1, composed.nodes().get(0).parent());
        assertEquals(0, composed.nodes().get(1).parent());
        assertEquals(0, composed.nodes().get(2).parent());
        assertEquals(2, composed.nodes().get(3).parent());
        assertEquals(Arrays.asList(1, 2, 2, 3), depthList(composed));
        assertEquals(1, composed.nodes().get(3).branch());
    }

    @Test
    public void composeTruncatesTheChainAtTheCarriedLength() {
        // the Q round accepted three tokens, but the width left room for two: the third token and everything
        // past it leave the proposal
        TwoLevelVerify.Chain c = chain(5, new int[] {70, 71, 72}, 80);
        DraftTree sub = new DraftTree(Collections.singletonList(new DraftTree.Node(90, -1, 1, 0)));
        DraftTree composed = TwoLevelVerify.compose(c, 3, sub);

        assertEquals(Arrays.asList(70, 71, 80, 90), tokenList(composed));
        assertEquals(Arrays.asList(1, 2, 3, 4), depthList(composed));
        assertEquals(2, composed.nodes().get(3).parent());
    }

    @Test
    public void composeWithoutASubtreeCarriesTheChainAlone() {
        TwoLevelVerify.Chain c = chain(5, new int[] {70, 71}, 80);
        DraftTree composed = TwoLevelVerify.compose(c, 3, new DraftTree(new ArrayList<>()));

        assertEquals(Arrays.asList(70, 71, 80), tokenList(composed));
        assertEquals(3, composed.size());
    }

    @Test
    public void composeFitsTheFixedWidthExactly() {
        // width 5: the chain takes at most 4 slots, the bonus the fifth, the anchored draft nothing
        TwoLevelVerify.Chain c = chain(5, new int[] {70, 71, 72, 73, 74}, 80);
        DraftTree sub = new DraftTree(Collections.singletonList(new DraftTree.Node(90, -1, 1, 0)));
        DraftTree composed = TwoLevelVerify.compose(c, 4, sub);
        assertEquals(5, composed.size());
        assertEquals(5, composed.depth());
    }

    @Test
    public void theArgmaxTakesTheFirstOfEqualMaxima() {
        assertEquals(2, TwoLevelVerify.argmax(new float[] {1f, 3f, 7f, 7f, 2f}));
        assertEquals(0, TwoLevelVerify.argmax(new float[] {0f, -1f, -2f}));
        assertEquals(1, TwoLevelVerify.argmax(new float[] {-5f, -1f}));
    }

    @Test
    public void theQuantizedViewIsOffByDefault() {
        SpeculativeOptions d = SpeculativeOptions.defaults();
        assertTrue(!d.twoLevelVerify());
        assertEquals(6, d.tlvBits());
        assertTrue(SpeculativeOptions.fromProperties(new Properties()).isDefault());
    }

    @Test
    public void theQuantizedViewSwitchesOnByName() {
        SpeculativeOptions on = SpeculativeOptions.fromProperties(props("neuralfree.draft.tlv", "true"));
        assertTrue(on.twoLevelVerify());
        assertTrue(!on.isDefault());
        assertTrue(on.toString().contains("tlv=true"));
        assertEquals(6, on.tlvBits());
        // back to the default through the builder
        assertTrue(!on.toBuilder().twoLevelVerify(false).build().twoLevelVerify());
        assertTrue(on.toBuilder().twoLevelVerify(false).build().isDefault());
    }

    @Test
    public void theQuantizedViewBitsAreSixOrEight() {
        assertEquals(8, SpeculativeOptions.fromProperties(
                props("neuralfree.draft.tlv", "true", "neuralfree.draft.tlv.bits", "8")).tlvBits());
        assertEquals(6, SpeculativeOptions.fromProperties(
                props("neuralfree.draft.tlv", "true", "neuralfree.draft.tlv.bits", "6")).tlvBits());
        for (String bad : new String[] {"5", "7", "0", "-1", "4", "16"}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.tlv.bits", bad)));
            assertTrue("bits " + bad + " names the property", e.getMessage()
                    .contains("neuralfree.draft.tlv.bits"));
        }
        // the builder validates the same widths
        assertThrows(IllegalArgumentException.class, () -> SpeculativeOptions.builder().tlvBits(5));
        assertThrows(IllegalArgumentException.class, () -> SpeculativeOptions.builder().tlvBits(9));
        assertEquals(8, SpeculativeOptions.builder().tlvBits(8).build().tlvBits());
    }

    private static Properties props(String... kv) {
        Properties p = new Properties();
        for (int i = 0; i < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return p;
    }

    private static List<Integer> tokenList(DraftTree t) {
        List<Integer> out = new ArrayList<>();
        for (DraftTree.Node n : t.nodes()) {
            out.add(n.token());
        }
        return out;
    }

    private static List<Integer> depthList(DraftTree t) {
        List<Integer> out = new ArrayList<>();
        for (DraftTree.Node n : t.nodes()) {
            out.add(n.depth());
        }
        return out;
    }
}
