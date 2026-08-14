#include "brush_c.h"
#include <stddef.h>

_Static_assert(sizeof(enum TrainExitCode) == 4, "v0.2.1 exit-code width changed");
_Static_assert(sizeof(enum ProgressMessageKind) == 4, "v0.2.1 event-kind width changed");
_Static_assert(sizeof(struct ProgressMessage) == 16, "v0.2.1 progress layout changed");
_Static_assert(offsetof(struct ProgressMessage, path) == 8, "v0.2.1 progress path moved");
_Static_assert(sizeof(struct TrainOptions) == 40, "v0.2.1 options layout changed");
_Static_assert(offsetof(struct TrainOptions, output_path) == 24, "v0.2.1 output path moved");
_Static_assert(offsetof(struct TrainOptions, export_name) == 32, "v0.2.1 export name moved");

static void legacy_callback(struct ProgressMessage message, void *context) {
  (void)message;
  (void)context;
}

void brush_v021_compile_contract(const char *dataset, const char *output) {
  struct TrainOptions options = {
      .total_train_steps = 300,
      .refine_every = 200,
      .max_resolution = 1080,
      .max_splats = 500000,
      .export_every = 300,
      .output_path = output,
      .export_name = "component-0_{iter}.ply",
  };
  struct BrushJob *job = brush_train_start(dataset, &options, legacy_callback, NULL);
  if (job != NULL) {
    (void)brush_job_cancel(job);
    (void)brush_job_wait(job);
    brush_job_release(job);
  }
  (void)train_and_save(dataset, &options, legacy_callback, NULL);
}
