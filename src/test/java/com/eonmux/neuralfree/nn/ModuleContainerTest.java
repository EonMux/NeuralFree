package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Child registration of {@link ModuleDict} and {@link ModuleList} stays in step with their contents. */
public class ModuleContainerTest {

    private static final class Leaf extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    @Test
    public void dictPutOnExistingKeyReplacesChildInPlace() {
        Leaf a = new Leaf();
        Leaf b = new Leaf();
        Leaf replacement = new Leaf();
        ModuleDict dict = new ModuleDict();
        assertNull(dict.put("a", a));
        assertNull(dict.put("b", b));

        assertSame(a, dict.put("a", replacement));

        assertSame(replacement, dict.get("a"));
        assertEquals(2, dict.size());
        Map<String, Module> children = dict.namedChildren();
        assertEquals(Arrays.asList("a", "b"), Arrays.asList(children.keySet().toArray()));
        assertSame(replacement, children.get("a"));
        assertSame(b, children.get("b"));
        assertEquals(Arrays.asList("a", "b"), Arrays.asList(dict.keySet().toArray()));
        assertTrue(dict.modules().stream().noneMatch(m -> m == a));
        a.close();
        dict.close();
    }

    @Test
    public void dictPutAllReplacesExistingKeys() {
        Leaf a = new Leaf();
        Leaf replacement = new Leaf();
        Leaf c = new Leaf();
        ModuleDict dict = new ModuleDict();
        dict.put("a", a);

        Map<String, Module> update = new LinkedHashMap<>();
        update.put("a", replacement);
        update.put("c", c);
        dict.putAll(update);

        assertSame(replacement, dict.namedChildren().get("a"));
        assertSame(c, dict.namedChildren().get("c"));
        assertEquals(2, dict.size());
        a.close();
        dict.close();
    }

    @Test
    public void dictRejectsCycleWithoutChangingState() {
        Leaf a = new Leaf();
        ModuleDict dict = new ModuleDict();
        dict.put("a", a);
        try {
            dict.put("a", dict);
            fail("A dict cannot hold itself");
        } catch (IllegalArgumentException expected) {
            assertSame(a, dict.get("a"));
            assertSame(a, dict.namedChildren().get("a"));
        }
        try {
            dict.put("self", dict);
            fail("A dict cannot hold itself");
        } catch (IllegalArgumentException expected) {
            assertEquals(1, dict.size());
            assertEquals(1, dict.namedChildren().size());
        }
        dict.close();
    }

    @Test
    public void dictEntrySetValueIsRejected() {
        Leaf a = new Leaf();
        Leaf other = new Leaf();
        ModuleDict dict = new ModuleDict();
        dict.put("a", a);
        Map.Entry<String, Module> entry = dict.entrySet().iterator().next();
        try {
            entry.setValue(other);
            fail("Entries of a ModuleDict cannot be replaced in place");
        } catch (UnsupportedOperationException expected) {
            assertSame(a, dict.get("a"));
            assertSame(a, dict.namedChildren().get("a"));
        }
        other.close();
        dict.close();
    }

    @Test
    public void dictReplaceAllRegistersNewChildren() {
        Linear a = new Linear(2, 3);
        Linear b = new Linear(3, 4);
        Linear newA = new Linear(2, 3);
        Linear newB = new Linear(3, 4);
        ModuleDict dict = new ModuleDict();
        dict.put("a", a);
        dict.put("b", b);

        dict.replaceAll((key, old) -> "a".equals(key) ? newA : newB);

        assertSame(newA, dict.get("a"));
        assertSame(newB, dict.get("b"));
        assertSame(newA, dict.namedChildren().get("a"));
        assertSame(newB, dict.namedChildren().get("b"));
        assertEquals(Arrays.asList("a", "b"), Arrays.asList(dict.keySet().toArray()));
        List<Parameter> expected = new ArrayList<>(newA.parameters());
        expected.addAll(newB.parameters());
        List<Parameter> actual = dict.parameters();
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            assertSame(expected.get(i), actual.get(i));
        }
        a.close();
        b.close();
        dict.close();
    }

    @Test
    public void dictReplaceAllRejectedReplacementLeavesDictUnchanged() {
        Leaf a = new Leaf();
        Leaf b = new Leaf();
        Leaf newA = new Leaf();
        ModuleDict dict = new ModuleDict();
        dict.put("a", a);
        dict.put("b", b);
        try {
            dict.replaceAll((key, old) -> "a".equals(key) ? newA : dict);
            fail("A dict cannot hold itself");
        } catch (IllegalArgumentException expected) {
            assertSame(a, dict.get("a"));
            assertSame(b, dict.get("b"));
            assertSame(a, dict.namedChildren().get("a"));
            assertSame(b, dict.namedChildren().get("b"));
            assertEquals(2, dict.namedChildren().size());
        }
        try {
            dict.replaceAll((key, old) -> "a".equals(key) ? newA : null);
            fail("A null replacement is rejected");
        } catch (NullPointerException expected) {
            assertSame(a, dict.get("a"));
            assertSame(a, dict.namedChildren().get("a"));
        }
        newA.close();
        dict.close();
    }

    @Test
    public void listInsertShiftsFollowingChildNames() {
        Leaf a = new Leaf();
        Leaf b = new Leaf();
        Leaf c = new Leaf();
        Leaf inserted = new Leaf();
        ModuleList list = new ModuleList(a, b, c);

        list.add(1, inserted);

        assertEquals(4, list.size());
        List<Module> expected = Arrays.asList(a, inserted, b, c);
        for (int i = 0; i < expected.size(); i++) {
            assertSame(expected.get(i), list.get(i));
        }
        Map<String, Module> children = list.namedChildren();
        assertEquals(Arrays.asList("0", "1", "2", "3"), Arrays.asList(children.keySet().toArray()));
        for (int i = 0; i < expected.size(); i++) {
            assertSame(expected.get(i), children.get(String.valueOf(i)));
        }
        list.close();
    }

    @Test
    public void listInsertAtFrontAndEnd() {
        Leaf a = new Leaf();
        Leaf front = new Leaf();
        Leaf end = new Leaf();
        ModuleList list = new ModuleList(a);

        list.add(0, front);
        list.add(2, end);

        List<Module> expected = Arrays.asList(front, a, end);
        Map<String, Module> children = list.namedChildren();
        assertEquals(Arrays.asList("0", "1", "2"), Arrays.asList(children.keySet().toArray()));
        for (int i = 0; i < expected.size(); i++) {
            assertSame(expected.get(i), list.get(i));
            assertSame(expected.get(i), children.get(String.valueOf(i)));
        }
        list.close();
    }

    @Test
    public void listRejectsCycleWithoutChangingState() {
        Leaf a = new Leaf();
        Leaf b = new Leaf();
        ModuleList list = new ModuleList(a, b);
        try {
            list.add(0, list);
            fail("A list cannot hold itself");
        } catch (IllegalArgumentException expected) {
            assertEquals(2, list.size());
            assertSame(a, list.namedChildren().get("0"));
            assertSame(b, list.namedChildren().get("1"));
            assertEquals(2, list.namedChildren().size());
        }
        try {
            list.add(list);
            fail("A list cannot hold itself");
        } catch (IllegalArgumentException expected) {
            assertEquals(2, list.size());
            assertEquals(2, list.namedChildren().size());
        }
        list.close();
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void listInsertOutOfBoundsIsRejected() {
        ModuleList list = new ModuleList(new Leaf());
        try {
            list.add(3, new Leaf());
        } finally {
            assertEquals(1, list.namedChildren().size());
            list.close();
        }
    }
}
