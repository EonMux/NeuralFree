package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.SecretProperties;

/**
 * A child-JVM probe of {@link SecretProperties#loadFromEnvironment()}: it loads the settings file the environment
 * names, prints the requested setting and stops. The serve tests run it to prove the child receives the secrets
 * through the file rather than the command line.
 */
public final class SecretPropertiesProbe {

    private SecretPropertiesProbe() {
    }

    public static void main(String[] args) {
        SecretProperties.loadFromEnvironment();
        System.out.println(System.getProperty(args[0], "<absent>"));
        System.out.flush();
    }
}
