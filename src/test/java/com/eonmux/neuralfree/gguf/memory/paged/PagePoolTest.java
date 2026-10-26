package com.eonmux.neuralfree.gguf.memory.paged;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Reference accounting of {@link PagePool} (PARITY MS-2): every allocation, reference and release keeps the pool
 * self-consistent, the free list is LIFO (the most recently freed page is handed out first), {@code ref} only adds
 * references a holder already owns ({@code refCount > 0}: only {@code alloc} takes a page from the free list, T-4)
 * and every argument is range-checked.
 */
public class PagePoolTest {

    @Test
    public void freshPoolCountsItsPagesFreeWithLifoAllocOrder() {
        PagePool pool = new PagePool(4, 3);
        assertEquals(3, pool.pages());
        assertEquals(4, pool.pageSize());
        assertEquals(12, pool.rows());
        assertEquals(3, pool.free());

        // the free stack starts as [0, 1, 2], so the first allocation takes the last entry
        assertEquals(2, pool.alloc());
        assertEquals(2, pool.free());
        assertEquals(1, pool.alloc());
        assertEquals(0, pool.alloc());
        assertEquals(0, pool.free());
        for (int p = 0; p < 3; p++) {
            assertFalse(pool.isFree(p));
            assertEquals(1, pool.refCount(p));
        }
        assertEquals(-1, pool.alloc());
    }

    @Test
    public void refAndFreeAccounting() {
        PagePool pool = new PagePool(4, 2);
        int a = pool.alloc();
        pool.ref(a);
        assertEquals(2, pool.refCount(a));
        pool.unref(a);
        assertEquals(1, pool.refCount(a));
        assertFalse(pool.isFree(a)); // still held by its owner
        assertEquals(1, pool.free());

        pool.unref(a);
        assertEquals(0, pool.refCount(a));
        assertTrue(pool.isFree(a));
        assertEquals(2, pool.free());

        // a second holder: ref + ref + unref leaves one reference, never below zero
        int b = pool.alloc();
        pool.ref(b);
        pool.ref(b);
        assertEquals(3, pool.refCount(b));
        pool.unref(b);
        pool.unref(b);
        assertEquals(1, pool.refCount(b));
        pool.unref(b);
        assertEquals(2, pool.free());
    }

    @Test
    public void freedPagesAreReusedMostRecentlyFreedFirst() {
        PagePool pool = new PagePool(2, 3);
        int first = pool.alloc();
        int second = pool.alloc();
        int third = pool.alloc();
        // release two pages, the second one later: the LIFO free list hands the second release out first
        pool.unref(second);
        pool.unref(first);
        assertEquals(first, pool.alloc());
        assertEquals(second, pool.alloc());
        assertEquals(0, pool.free());
        assertTrue(pool.refCount(third) == 1 && !pool.isFree(third));
    }

    @Test
    public void peekSeesThePagesAllocWouldTakeWithoutTouchingThePool() {
        PagePool pool = new PagePool(2, 4);
        pool.alloc();
        pool.alloc();
        int[] peek = pool.peek(2);
        assertEquals(2, peek.length);
        assertEquals(1, peek[0]);
        assertEquals(0, peek[1]);
        assertEquals(2, pool.free()); // the pool is untouched
        assertEquals(1, pool.alloc());
        assertEquals(0, pool.alloc());
        assertEquals(2, pool.free());
        assertThrows(IllegalArgumentException.class, () -> pool.peek(-1));
        assertThrows(IllegalArgumentException.class, () -> pool.peek(3));
    }

    @Test
    public void refOfAFreePageThrows() {
        PagePool pool = new PagePool(2, 2);
        int a = pool.alloc();
        pool.unref(a);
        // only alloc moves a page from the free list (T-4): a consumer cannot revive a freed page
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> pool.ref(a));
        assertTrue(e.getMessage(), e.getMessage().contains("use alloc"));
        IllegalStateException u = assertThrows(IllegalStateException.class, () -> pool.unref(a));
        assertTrue(u.getMessage(), u.getMessage().contains("not held"));
        // a page that was never allocated carries no reference either
        IllegalStateException n = assertThrows(IllegalStateException.class, () -> pool.ref(a));
        assertTrue(n.getMessage(), n.getMessage().contains("use alloc"));
    }

    @Test
    public void everyPageIdIsRangeChecked() {
        PagePool pool = new PagePool(2, 2);
        for (String op : new String[]{"ref", "unref", "refCount", "isFree"}) {
            IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class,
                    () -> poolCall(pool, op, -1));
            assertTrue(e.getMessage(), e.getMessage().contains("page -1"));
            assertThrows(IndexOutOfBoundsException.class, () -> poolCall(pool, op, 2));
        }
    }

    private static void poolCall(PagePool pool, String op, int page) {
        if (op.equals("ref")) {
            pool.ref(page);
        } else if (op.equals("unref")) {
            pool.unref(page);
        } else if (op.equals("refCount")) {
            pool.refCount(page);
        } else {
            pool.isFree(page);
        }
    }

    @Test
    public void constructorValidatesItsArguments() {
        assertThrows(IllegalArgumentException.class, () -> new PagePool(0, 2));
        assertThrows(IllegalArgumentException.class, () -> new PagePool(4, 0));
        assertThrows(IllegalArgumentException.class, () -> new PagePool(4, -3));
        PagePool pool = new PagePool(16, 8);
        assertEquals(128, pool.rows());
        assertTrue(pool.toString().contains("pageSize=16"));
    }
}
