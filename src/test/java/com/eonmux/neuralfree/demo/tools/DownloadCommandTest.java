package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The network-free contract of {@link DownloadCommand}: a malformed repository expression is
 * refused as a usage error before any Hub call is made — {@link HubRef#parse} runs first, so no test starts a
 * download, touches the transport or reads the network. The selection logic
 * over the repository listing runs inside the online flow ({@code bestModel}, {@code splitFiles}) and the cache
 * read-back runs over {@link com.eonmux.neuralfree.gguf.hub.HubCache}, whose paths and filename rules
 * {@code HubRefTest} covers; the download itself needs a live Hub and stays out of the tests.
 */
public class DownloadCommandTest {

    @Test
    public void aMalformedRepositoryIsAUsageErrorBeforeAnyHubCall() {
        // Every expression here fails inside HubRef.parse, which downloadModel runs before any Hub call,
        // so no test reaches the transport.
        for (String bad : new String[]{"", "   ", "nomodel", "a/b/c", "user/", "/model", "user/model:",
                "user/model:@main", "user/model:Q4_K/..M", "user//model", "user/model:Q4 M"}) {
            var e = assertThrows(bad, DownloadCommand.UsageException.class,
                    () -> DownloadCommand.downloadModel(bad, null));
            assertTrue(bad, e.getMessage() != null && !e.getMessage().isEmpty());
        }
    }

    @Test
    public void theRefusalNamesTheExpectedExpressionForm() {
        // The parse boundary runs before any repository lookup: an empty quant tag after ':' is refused
        // with the <user>/<model>[:quant][@revision] form, never a Hub error.
        var e = assertThrows(DownloadCommand.UsageException.class,
                () -> DownloadCommand.downloadModel("user/model:", "model.gguf"));
        assertTrue(e.getMessage().contains("<user>/<model>"));
    }
}
