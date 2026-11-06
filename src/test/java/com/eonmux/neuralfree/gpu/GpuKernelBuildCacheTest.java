package com.eonmux.neuralfree.gpu;

import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Guards the build-cache wiring of the GPU kernel artifact tasks in gradle/gpu-kernels.gradle: the project cache
 * is on, and every task that produces an artifact merged into the resources declares relocatable inputs and
 * enables caching with a reason. The end-to-end behavior (a second checkout reports FROM-CACHE) is exercised by
 * the build itself; this test fails fast when a task loses its cache declaration.
 */
public class GpuKernelBuildCacheTest {

    /** Tasks whose outputs are kernel artifacts (or a JNI shim) merged into the application resources. */
    private static final String[] ARTIFACT_TASKS = {
            "compileCudaKernels", "compileHipKernels", "compileShaders", "compileTestShaders", "compileMetalShaders",
            "buildHipShim", "buildMetalShim"
    };

    private File root;
    private String buildScript;

    @Before
    public void readBuildFiles() throws IOException {
        root = projectRoot();
        buildScript = new String(Files.readAllBytes(new File(root, "gradle/gpu-kernels.gradle").toPath()),
                StandardCharsets.UTF_8);
    }

    @Test
    public void theProjectEnablesTheBuildCache() throws IOException {
        String properties = new String(Files.readAllBytes(new File(root, "gradle.properties").toPath()),
                StandardCharsets.UTF_8);
        assertTrue("gradle.properties must set org.gradle.caching=true",
                properties.replace("\r\n", "\n").contains("org.gradle.caching=true"));
    }

    @Test
    public void everyArtifactTaskEnablesCachingWithAReason() {
        for (String task : ARTIFACT_TASKS) {
            String block = taskBlock(task);
            assertTrue(task + " must mark its outputs cacheable", block.contains("outputs.cacheIf("));
        }
    }

    @Test
    public void everyArtifactTaskDeclaresRelocatableInputs() {
        for (String task : ARTIFACT_TASKS) {
            String block = taskBlock(task);
            assertTrue(task + " must declare its file inputs with RELATIVE path sensitivity",
                    block.contains("PathSensitivity.RELATIVE"));
        }
    }

    /** The registration block of the task, up to the next task registration. */
    private String taskBlock(String task) {
        int start = buildScript.indexOf("tasks.register('" + task + "')");
        if (start < 0) {
            fail("no registration of " + task + " in gradle/gpu-kernels.gradle");
        }
        int end = buildScript.indexOf("tasks.register(", start + 1);
        return end < 0 ? buildScript.substring(start) : buildScript.substring(start, end);
    }

    /** Walks up from the test working directory to the checkout that holds gradle/gpu-kernels.gradle. */
    private static File projectRoot() {
        File dir = new File(System.getProperty("user.dir")).getAbsoluteFile();
        while (dir != null) {
            if (new File(dir, "gradle/gpu-kernels.gradle").isFile()) {
                return dir;
            }
            dir = dir.getParentFile();
        }
        throw new IllegalStateException("gradle/gpu-kernels.gradle not found above " + System.getProperty("user.dir"));
    }
}
