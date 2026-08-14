import BrushKitFFI
import Darwin
import Foundation

private struct CapturedEvent {
  var kind: Int32
  var iteration: UInt32
  var initializerRoute: Int32
  var errorCode: Int32
  var text: String?
}

private final class EventBox: @unchecked Sendable {
  private let lock = NSLock()
  private var storage: [CapturedEvent] = []

  func append(_ event: BrushEventV2) {
    let captured = CapturedEvent(
      kind: Int32(event.kind.rawValue),
      iteration: event.iteration,
      initializerRoute: Int32(event.initializer_route.rawValue),
      errorCode: Int32(event.error_code.rawValue),
      text: event.text.map(String.init(cString:))
    )
    lock.lock()
    storage.append(captured)
    lock.unlock()
  }

  func snapshot() -> [CapturedEvent] {
    lock.lock()
    defer { lock.unlock() }
    return storage
  }
}

private let eventCallback: @convention(c) (
  BrushEventV2,
  UnsafeMutableRawPointer?
) -> Void = { event, context in
  guard let context else { return }
  Unmanaged<EventBox>.fromOpaque(context).takeUnretainedValue().append(event)
}

private func fail(_ message: String) -> Never {
  fputs("BrushKitV2Smoke: \(message)\n", stderr)
  exit(1)
}

guard CommandLine.arguments.count == 3 else {
  fail("usage: BrushKitV2Smoke <dataset-directory> <output-directory>")
}

let datasetPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]
try FileManager.default.createDirectory(
  atPath: outputPath,
  withIntermediateDirectories: true
)

guard brush_get_abi_version() == 2 else {
  fail("callable ABI version is not 2")
}
var identity = BrushNativeIdentityV2()
guard brush_get_native_identity_v2(
  &identity,
  UInt32(MemoryLayout<BrushNativeIdentityV2>.size)
) else {
  fail("native identity query failed")
}
guard identity.abi_version == 2,
      identity.build_revision != nil,
      identity.graphics_backend != nil
else {
  fail("native identity is incomplete")
}

guard let datasetCString = strdup(datasetPath),
      let outputCString = strdup(outputPath),
      let exportCString = strdup("swift-v2-{iter}.ply"),
      let initializerCString = strdup("strong-init.ply")
else {
  fail("could not allocate C strings")
}
defer {
  free(datasetCString)
  free(outputCString)
  free(exportCString)
  free(initializerCString)
}

var options = TrainOptionsV2()
options.struct_size = UInt32(MemoryLayout<TrainOptionsV2>.size)
options.abi_version = 2
options.seed = 0x0123_4567_89ab_cdef
options.render_mode = 1
options.sh_degree = 3
options.sh_policy = 0
options.total_train_steps = 10
options.refine_every = 200
options.max_resolution = 50
options.max_splats = 1_000
options.export_every = 10
options.output_path = UnsafePointer(outputCString)
options.export_name = UnsafePointer(exportCString)
options.initializer_path = UnsafePointer(initializerCString)
options.initializer_required = 1
options.instrumentation_level = 1

let eventBox = Unmanaged.passRetained(EventBox())
defer { eventBox.release() }
guard let job = brush_train_start_v2(
  UnsafePointer(datasetCString),
  &options,
  eventCallback,
  eventBox.toOpaque()
) else {
  fail("V2 job creation failed")
}
guard let retained = brush_job_retain_v2(job) else {
  brush_job_release_v2(job)
  fail("V2 job retain failed")
}

let retainedStatus = brush_job_wait_v2(retained)
brush_job_release_v2(retained)
let status = brush_job_wait_v2(job)
brush_job_release_v2(job)
guard Int32(status.rawValue) == 0, Int32(retainedStatus.rawValue) == 0 else {
  fail("V2 retained waits did not both succeed")
}

let events = eventBox.takeUnretainedValue().snapshot()
guard events.first?.kind == 0, events.dropFirst().first?.kind == 1,
      events.last?.kind == 11
else {
  fail("unexpected event ordering: \(events.map(\.kind))")
}
guard events.contains(where: { $0.kind == 4 && $0.initializerRoute == 4 }) else {
  fail("missing explicit-strong initializer audit event")
}
guard events.contains(where: { $0.kind == 7 && $0.iteration == 1 }),
      events.contains(where: { $0.kind == 7 && $0.iteration == 10 }),
      events.contains(where: { $0.kind == 9 && $0.iteration == 10 }),
      events.contains(where: { $0.kind == 10 && $0.iteration == 10 })
else {
  fail("missing Phase 0 step/export events")
}
guard events.last?.errorCode == 0 else {
  fail("terminal event reported: \(events.last?.text ?? "unknown error")")
}
let outputs = try FileManager.default.contentsOfDirectory(atPath: outputPath)
guard outputs.contains(where: { $0 == "swift-v2-10.ply" }) else {
  fail("expected export was not written")
}

print("BrushKitV2Smoke: PASS abi=2 events=\(events.count) output=swift-v2-10.ply")
