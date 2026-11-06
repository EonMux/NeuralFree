package com.eonmux.neuralfree.server;

import io.quarkus.vertx.http.runtime.VertxHttpConfig;
import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.Router;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Singleton;

import java.util.List;

/**
 * llama-server's empty, unauthenticated OPTIONS response, using the Quarkus CORS configuration ({@link CorsPolicy}).
 * Unlike Quarkus's ordinary preflight filter, it also answers OPTIONS without Origin or Access-Control-Request-*
 * headers, and sends the configured header policy verbatim (including '*') rather than reflecting the requested
 * headers. It runs ahead of the {@link ApiPrefixRouter}, so every OPTIONS answers like llama.cpp's pre-routing
 * handler even without the prefix. Quarkus's CORS filter continues to enforce the configured origin and method
 * policy on ordinary requests.
 */
@Singleton
public class CorsPreflight {

    void install(@Observes Router router, VertxHttpConfig http) {
        CorsPolicy cors = CorsPolicy.from(http.cors());
        if (!cors.enabled()) {
            return;
        }
        List<String> methods = http.cors().methods().orElse(List.of("GET", "POST", "DELETE", "OPTIONS"));
        List<String> headers = http.cors().headers().orElse(List.of("*"));
        router.route().order(Integer.MIN_VALUE + 900).handler(context -> {
            if (!context.request().method().equals(HttpMethod.OPTIONS)) {
                context.next();
                return;
            }
            var response = context.response();
            cors.headers(context.request().getHeader("Origin")).forEach(response::putHeader);
            response.putHeader("Access-Control-Allow-Methods", String.join(", ", methods))
                    .putHeader("Access-Control-Allow-Headers", String.join(", ", headers))
                    .putHeader("Content-Type", "text/html")
                    .setStatusCode(200).end();
        });
    }
}
