package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.sun.jna.Native;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * The D3DKMT structure layouts must match the Windows ABI exactly, because the kernel writes
 * through them; the segment arithmetic is exercised separately.
 */
public class WddmAdapterStatisticsTest {

    @Test
    public void queryStatisticsLayoutMatchesTheWindowsAbi() {
        Assume.assumeTrue("the layout below is the x64 one", Native.POINTER_SIZE == 8);
        WddmAdapterStatistics.QueryStatistics statistics = new WddmAdapterStatistics.QueryStatistics();

        assertEquals(808, statistics.size());
        assertEquals(0, statistics.offsetOf("type"));
        assertEquals(4, statistics.offsetOf("adapterLuid"));
        assertEquals(16, statistics.offsetOf("hProcess"));
        assertEquals(24, statistics.offsetOf("queryResult"));
        assertEquals(800, statistics.offsetOf("querySegmentId"));
    }

    @Test
    public void adapterEnumerationLayoutMatchesTheWindowsAbi() {
        Assume.assumeTrue("the layout below is the x64 one", Native.POINTER_SIZE == 8);

        WddmAdapterStatistics.EnumAdapters2 enumeration = new WddmAdapterStatistics.EnumAdapters2();
        assertEquals(16, enumeration.size());
        assertEquals(0, enumeration.offsetOf("numAdapters"));
        assertEquals(8, enumeration.offsetOf("adapters"));

        WddmAdapterStatistics.AdapterInfo adapter = new WddmAdapterStatistics.AdapterInfo();
        assertEquals(20, adapter.size());
        assertEquals(0, adapter.offsetOf("hAdapter"));
        assertEquals(4, adapter.offsetOf("adapterLuid"));
        assertEquals(12, adapter.offsetOf("numOfSources"));
        assertEquals(16, adapter.offsetOf("precisePresentRegionsPreferred"));

        assertEquals(8, new WddmAdapterStatistics.Luid().size());
    }

    @Test
    public void queryAdapterInfoLayoutMatchesTheWindowsAbi() {
        Assume.assumeTrue("the layout below is the x64 one", Native.POINTER_SIZE == 8);

        WddmAdapterStatistics.QueryAdapterInfo query = new WddmAdapterStatistics.QueryAdapterInfo();
        assertEquals(24, query.size());
        assertEquals(0, query.offsetOf("hAdapter"));
        assertEquals(4, query.offsetOf("type"));
        assertEquals(8, query.offsetOf("privateDriverData"));
        assertEquals(16, query.offsetOf("privateDriverDataSize"));
    }

    @Test
    public void segmentFreeIsCommitLimitMinusResident() {
        byte[] result = new byte[776];
        ByteBuffer buffer = ByteBuffer.wrap(result).order(ByteOrder.nativeOrder());
        buffer.putLong(0, 1000);
        buffer.putLong(16, 400);
        buffer.putInt(40, 0);

        assertEquals(600L, WddmAdapterStatistics.freeFromSegmentResult(result));
    }

    @Test
    public void overResidentSegmentReportsZero() {
        byte[] result = new byte[776];
        ByteBuffer buffer = ByteBuffer.wrap(result).order(ByteOrder.nativeOrder());
        buffer.putLong(0, 1000);
        buffer.putLong(16, 1500);

        assertEquals(0L, WddmAdapterStatistics.freeFromSegmentResult(result));
    }

    @Test
    public void apertureSegmentIsSkipped() {
        byte[] result = new byte[776];
        ByteBuffer buffer = ByteBuffer.wrap(result).order(ByteOrder.nativeOrder());
        buffer.putLong(0, 1000);
        buffer.putLong(16, 0);
        buffer.putInt(40, 1);

        assertEquals(-1L, WddmAdapterStatistics.freeFromSegmentResult(result));
    }

    @Test
    public void parsesNormalisedPciAddresses() {
        assertArrayEquals(new int[]{3, 0, 0}, WddmAdapterStatistics.parsePciAddress("0000:03:00.0"));
        assertArrayEquals(new int[]{0x1f, 0x1f, 7}, WddmAdapterStatistics.parsePciAddress("1f:1f.7"));
        assertArrayEquals(new int[]{0xff, 5, 1}, WddmAdapterStatistics.parsePciAddress("ff:05.1"));
    }

    @Test
    public void rejectsMalformedPciAddresses() {
        assertNull(WddmAdapterStatistics.parsePciAddress(null));
        assertNull(WddmAdapterStatistics.parsePciAddress(""));
        assertNull(WddmAdapterStatistics.parsePciAddress("0000:03:00"));
        assertNull(WddmAdapterStatistics.parsePciAddress("0000:03:00.8"));
        assertNull(WddmAdapterStatistics.parsePciAddress("0000:100:00.0"));
        assertNull(WddmAdapterStatistics.parsePciAddress("0000:03:20.0"));
        assertNull(WddmAdapterStatistics.parsePciAddress("zz:03:00.0"));
    }

    @Test
    public void degradesToUnavailableOffWindows() {
        Assume.assumeFalse(System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win"));

        assertEquals(-1L, WddmAdapterStatistics.freeBytes(DeviceIdentity.ofName("AMD Radeon")));
    }
}
