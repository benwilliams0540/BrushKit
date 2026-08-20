#include "brush_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "BrushKit release identity failure: %s\n", message);
    exit(1);
  }
}

int main(int argc, char **argv) {
  require(argc == 4, "expected revision, version, and Apple feature arguments");

  struct BrushNativeIdentityV2 identity = {
      .struct_size = 0,
      .abi_version = 0,
      .build_revision = NULL,
      .crate_version = NULL,
      .graphics_backend = NULL,
      .adapter_name = NULL,
      .adapter_identity_available = true,
  };
  require(brush_get_abi_version() == 2, "callable ABI is not 2");
  require(brush_get_native_identity_v2(&identity, sizeof(identity)),
          "native identity query failed");
  require(identity.abi_version == 2, "identity ABI is not 2");
  require(strcmp(identity.build_revision, argv[1]) == 0,
          "engine revision mismatch");
  require(strcmp(identity.crate_version, argv[2]) == 0,
          "crate version mismatch");
  require(strcmp(identity.graphics_backend, "Metal") == 0,
          "graphics backend is not Metal");

  const char *provenance = brush_get_build_provenance_v1();
  require(provenance != NULL, "provenance query returned null");
  require(strstr(provenance, "format=brushkit-build-provenance-v1;") != NULL,
          "provenance format missing");
  require(strstr(provenance, argv[1]) != NULL,
          "provenance engine revision mismatch");
  require(strstr(provenance, "callable_abi=2;additive_train_abi=4;") != NULL,
          "provenance ABI record missing");
  require(strstr(provenance, argv[3]) != NULL,
          "provenance Apple feature mismatch");
  require(strstr(provenance, "cubecl_gpu_profile=off;") != NULL,
          "provenance profiler state is not off");
  require(strstr(provenance, "rustc=rustc 1.95.0 ") != NULL,
          "provenance Rust toolchain mismatch");
  require(strstr(provenance, "target=aarch64-apple-darwin") != NULL,
          "provenance target mismatch");

  printf("abi=%u revision=%s version=%s backend=%s provenance=%s\n",
         identity.abi_version, identity.build_revision, identity.crate_version,
         identity.graphics_backend, provenance);
  return 0;
}
