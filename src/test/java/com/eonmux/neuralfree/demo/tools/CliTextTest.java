package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * The C++ {@code getline} semantics the datafile loaders read with: the HellaSwag loop keeps an unterminated
 * last line, the WinoGrande CSV loop drops it ({@code if (in.fail() || in.eof()) break}), and empty middle
 * lines stay.
 */
public class CliTextTest {

    @Test
    public void getlineAllKeepsAnUnterminatedLastLine() {
        assertEquals(List.of("a", "b"), CliText.getlineAll("a\nb\n"));
        assertEquals(List.of("a", "b"), CliText.getlineAll("a\nb"));
        assertEquals(List.of("a", "", "b"), CliText.getlineAll("a\n\nb"));
        assertEquals(List.of(""), CliText.getlineAll("\n"));
        assertEquals(0, CliText.getlineAll("").size());
    }
    @Test
    public void getlineCompleteDropsAnUnterminatedLastLine() {
        assertEquals(List.of("a", "b"), CliText.getlineComplete("a\nb\n"));
        assertEquals(List.of("a"), CliText.getlineComplete("a\nb"));
        assertEquals(List.of("a", ""), CliText.getlineComplete("a\n\nb"));
        assertEquals(List.of(""), CliText.getlineComplete("\n"));
        assertEquals(0, CliText.getlineComplete("a").size());
        assertEquals(0, CliText.getlineComplete("").size());
    }
}
