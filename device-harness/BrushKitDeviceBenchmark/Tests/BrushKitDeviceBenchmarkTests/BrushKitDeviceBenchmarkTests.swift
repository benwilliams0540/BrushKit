import BrushKitFFI
import CryptoKit
import Darwin
import Foundation
import UIKit
import XCTest

private struct ResourceSample: Codable, Sendable {
  var boundary: String
  var iteration: UInt32?
  var elapsedSeconds: Double
  var thermalState: String
  var batteryLevelPercent: Int
  var chargingState: String
  var lowPowerModeEnabled: Bool
  var residentMemoryBytes: UInt64
  var peakResidentMemoryBytes: UInt64
}

private struct StepSample: Codable, Sendable {
  var iteration: UInt32
  var nativeTimestampNanoseconds: UInt64
  var primitiveCount: UInt32
  var forwardNanoseconds: UInt64
  var lossAndSSIMNanoseconds: UInt64
  var backwardNanoseconds: UInt64
  var optimizerNanoseconds: UInt64
  var densificationAndCompactionNanoseconds: UInt64
}

private struct CapturedEvent: Sendable {
  var kind: Int32
  var iteration: UInt32
  var errorCode: Int32
  var capabilityFlags: UInt64
  var initialPrimitiveCount: UInt32
  var finalPrimitiveCount: UInt32
  var primitiveCount: UInt32
  var prunedNonFiniteCount: UInt32
  var text: String?
}

private struct OutputValidation: Codable, Sendable {
  var byteCount: Int
  var vertexCount: Int
  var floatPropertyCount: Int
  var scannedValueCount: Int
  var nonFiniteValueCount: Int
  var sha256: String
}

private struct BenchmarkResult: Codable, Sendable {
  var schemaVersion = 1
  var runLabel: String
  var callableABIVersion: UInt32
  var nativeIdentityABIVersion: UInt32
  var nativeBuildRevision: String
  var crateVersion: String
  var graphicsBackend: String
  var adapterName: String
  var adapterIdentityAvailable: Bool
  var deviceName: String
  var deviceModel: String
  var systemName: String
  var systemVersion: String
  var activeProcessorCount: Int
  var fixtureManifestSHA256: String
  var seed: UInt64
  var renderMode: UInt32
  var shDegree: UInt32
  var totalTrainSteps: UInt32
  var refineEvery: UInt32
  var maxResolution: UInt32
  var maxSplats: UInt32
  var exportEvery: UInt32
  var progressiveResolutionStartPercent: UInt32
  var progressiveResolutionSwitchIteration: UInt32
  var instrumentationLevel: UInt32
  var trainerWallTimeSeconds: Double
  var exitCode: Int32
  var eventKinds: [Int32]
  var capabilityFlags: UInt64
  var initialPrimitiveCount: UInt32
  var terminalFinalPrimitiveCount: UInt32
  var terminalPrunedNonFiniteCount: UInt32
  var terminalText: String?
  var resourceSamples: [ResourceSample]
  var stepSamples: [StepSample]
  var outputValidation: OutputValidation
}

private final class EventBox: @unchecked Sendable {
  private let lock = NSLock()
  private let startedNanoseconds: UInt64
  private let batteryLevelPercent: Int
  private let chargingState: String
  private var events: [CapturedEvent] = []
  private var resources: [ResourceSample] = []
  private var steps: [StepSample] = []

  init(startedNanoseconds: UInt64, batteryLevelPercent: Int, chargingState: String) {
    self.startedNanoseconds = startedNanoseconds
    self.batteryLevelPercent = batteryLevelPercent
    self.chargingState = chargingState
  }

  func append(_ event: BrushEventV2) {
    let kind = Int32(event.kind.rawValue)
    let captured = CapturedEvent(
      kind: kind,
      iteration: event.iteration,
      errorCode: Int32(event.error_code.rawValue),
      capabilityFlags: event.capability_flags,
      initialPrimitiveCount: event.initial_primitive_count,
      finalPrimitiveCount: event.final_primitive_count,
      primitiveCount: event.primitive_count,
      prunedNonFiniteCount: event.pruned_non_finite_count,
      text: event.text.map(String.init(cString:))
    )
    let shouldSample =
      kind == 11 || (kind == 7 && (event.iteration == 1 || event.iteration % 25 == 0))
    let resource =
      shouldSample
      ? makeResourceSample(
        boundary: kind == 11 ? "terminal" : "step",
        iteration: event.iteration,
        startedNanoseconds: startedNanoseconds,
        batteryLevelPercent: batteryLevelPercent,
        chargingState: chargingState
      ) : nil
    let step =
      kind == 7
      ? StepSample(
        iteration: event.iteration,
        nativeTimestampNanoseconds: event.timestamp_ns,
        primitiveCount: event.primitive_count,
        forwardNanoseconds: event.forward_ns,
        lossAndSSIMNanoseconds: event.loss_and_ssim_ns,
        backwardNanoseconds: event.backward_ns,
        optimizerNanoseconds: event.optimizer_ns,
        densificationAndCompactionNanoseconds: event.densification_and_compaction_ns
      ) : nil

    lock.lock()
    events.append(captured)
    if let resource { resources.append(resource) }
    if let step { steps.append(step) }
    lock.unlock()
  }

  func appendResource(_ sample: ResourceSample) {
    lock.lock()
    resources.append(sample)
    lock.unlock()
  }

  func snapshot() -> ([CapturedEvent], [ResourceSample], [StepSample]) {
    lock.lock()
    defer { lock.unlock() }
    return (events, resources, steps)
  }
}

private let eventCallback: @convention(c) (BrushEventV2, UnsafeMutableRawPointer?) -> Void = {
  event, context in
  guard let context else { return }
  Unmanaged<EventBox>.fromOpaque(context).takeUnretainedValue().append(event)
}

private func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
  switch state {
  case .nominal: "nominal"
  case .fair: "fair"
  case .serious: "serious"
  case .critical: "critical"
  @unknown default: "unknown"
  }
}

private func batteryStateName(_ state: UIDevice.BatteryState) -> String {
  switch state {
  case .unknown: "unknown"
  case .unplugged: "unplugged"
  case .charging: "charging"
  case .full: "full"
  @unknown default: "unknown"
  }
}

private func memoryFootprint() -> (resident: UInt64, peak: UInt64) {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(
    MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
  let result = withUnsafeMutablePointer(to: &info) { infoPointer in
    infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
    }
  }
  guard result == KERN_SUCCESS else { return (0, 0) }
  return (UInt64(info.phys_footprint), UInt64(info.resident_size_peak))
}

private func makeResourceSample(
  boundary: String,
  iteration: UInt32?,
  startedNanoseconds: UInt64,
  batteryLevelPercent: Int,
  chargingState: String
) -> ResourceSample {
  let now = DispatchTime.now().uptimeNanoseconds
  let memory = memoryFootprint()
  return ResourceSample(
    boundary: boundary,
    iteration: iteration,
    elapsedSeconds: Double(now - startedNanoseconds) / 1_000_000_000,
    thermalState: thermalStateName(ProcessInfo.processInfo.thermalState),
    batteryLevelPercent: batteryLevelPercent,
    chargingState: chargingState,
    lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled,
    residentMemoryBytes: memory.resident,
    peakResidentMemoryBytes: memory.peak
  )
}

private func sha256(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func fixtureManifestSHA256(_ directory: URL) throws -> String {
  let manager = FileManager.default
  let keys: [URLResourceKey] = [.isRegularFileKey]
  let files = try manager.contentsOfDirectory(
    at: directory,
    includingPropertiesForKeys: keys,
    options: [.skipsHiddenFiles]
  ).flatMap { child -> [URL] in
    let values = try child.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
    if values.isRegularFile == true { return [child] }
    guard values.isDirectory == true else { return [] }
    let enumerator = manager.enumerator(
      at: child,
      includingPropertiesForKeys: keys,
      options: [.skipsHiddenFiles]
    )
    return (enumerator?.allObjects as? [URL] ?? []).filter {
      (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
    }
  }.sorted { $0.path < $1.path }

  var hasher = SHA256()
  for file in files {
    let relative = file.path.replacingOccurrences(of: directory.path + "/", with: "")
    hasher.update(data: Data(relative.utf8))
    hasher.update(data: Data([0]))
    hasher.update(data: Data(SHA256.hash(data: try Data(contentsOf: file))))
  }
  return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func validatePLY(_ url: URL) throws -> OutputValidation {
  let data = try Data(contentsOf: url, options: [.mappedIfSafe])
  let marker = Data("end_header\n".utf8)
  guard let markerRange = data.range(of: marker) else {
    throw NSError(
      domain: "BrushKitDeviceBenchmark", code: 1,
      userInfo: [NSLocalizedDescriptionKey: "PLY header terminator missing"])
  }
  let headerEnd = markerRange.upperBound
  guard let header = String(data: data[..<headerEnd], encoding: .utf8),
    header.contains("format binary_little_endian 1.0")
  else {
    throw NSError(
      domain: "BrushKitDeviceBenchmark", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "PLY is not binary little endian"])
  }
  let lines = header.split(separator: "\n")
  guard let vertexLine = lines.first(where: { $0.hasPrefix("element vertex ") }),
    let vertexCount = Int(vertexLine.split(separator: " ").last ?? "")
  else {
    throw NSError(
      domain: "BrushKitDeviceBenchmark", code: 3,
      userInfo: [NSLocalizedDescriptionKey: "PLY vertex count missing"])
  }
  let propertyCount = lines.filter { $0.hasPrefix("property float ") }.count
  let expectedBodyBytes = vertexCount * propertyCount * MemoryLayout<UInt32>.size
  guard data.count - headerEnd == expectedBodyBytes else {
    throw NSError(
      domain: "BrushKitDeviceBenchmark", code: 4,
      userInfo: [NSLocalizedDescriptionKey: "PLY body size does not match schema"])
  }
  var nonFinite = 0
  data.withUnsafeBytes { bytes in
    for offset in stride(from: headerEnd, to: data.count, by: MemoryLayout<UInt32>.size) {
      let bits = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
      if !Float(bitPattern: bits).isFinite { nonFinite += 1 }
    }
  }
  return OutputValidation(
    byteCount: data.count,
    vertexCount: vertexCount,
    floatPropertyCount: propertyCount,
    scannedValueCount: vertexCount * propertyCount,
    nonFiniteValueCount: nonFinite,
    sha256: sha256(data)
  )
}

final class BrushKitDeviceBenchmarkTests: XCTestCase {
  @MainActor
  func testFrozenRGBTrainerWorkload() throws {
    UIDevice.current.isBatteryMonitoringEnabled = true
    guard
      let datasetURL = Bundle(for: Self.self).url(
        forResource: "pinhole-fullres-component-0",
        withExtension: nil
      )
    else {
      return XCTFail("Frozen trainer leaf is missing from the test bundle")
    }

    let runLabel =
      ProcessInfo.processInfo.environment["BRUSHKIT_BENCHMARK_RUN_LABEL"] ?? "unlabeled"
    let outputURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("BrushKitDeviceBenchmark-(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
    let fixtureChecksum = try fixtureManifestSHA256(datasetURL)

    XCTAssertEqual(brush_get_abi_version(), 2)
    var identity = BrushNativeIdentityV2()
    XCTAssertTrue(
      brush_get_native_identity_v2(
        &identity,
        UInt32(MemoryLayout<BrushNativeIdentityV2>.size)
      ))
    let revision = identity.build_revision.map(String.init(cString:)) ?? "missing"
    let crateVersion = identity.crate_version.map(String.init(cString:)) ?? "missing"
    let graphicsBackend = identity.graphics_backend.map(String.init(cString:)) ?? "missing"
    let adapterName = identity.adapter_name.map(String.init(cString:)) ?? "missing"

    guard let datasetCString = strdup(datasetURL.path),
      let outputCString = strdup(outputURL.path),
      let exportCString = strdup("component-0_{iter}.ply")
    else {
      return XCTFail("Unable to allocate C strings")
    }
    defer {
      free(datasetCString)
      free(outputCString)
      free(exportCString)
    }

    var options = TrainOptionsV3()
    options.struct_size = UInt32(MemoryLayout<TrainOptionsV3>.size)
    options.abi_version = 3
    options.seed = 0
    options.render_mode = 1
    options.sh_degree = 3
    options.sh_policy = 0
    options.total_train_steps = 300
    options.refine_every = 200
    options.max_resolution = 1080
    options.max_splats = 500_000
    options.export_every = 300
    options.output_path = UnsafePointer(outputCString)
    options.export_name = UnsafePointer(exportCString)
    options.initializer_path = nil
    options.initializer_required = 0
    options.instrumentation_level = 1
    options.progressive_resolution_start_percent = 50
    options.progressive_resolution_switch_iteration = 275

    let batteryPercent = max(0, Int((UIDevice.current.batteryLevel * 100).rounded()))
    let chargingState = batteryStateName(UIDevice.current.batteryState)
    let started = DispatchTime.now().uptimeNanoseconds
    let eventBox = Unmanaged.passRetained(
      EventBox(
        startedNanoseconds: started,
        batteryLevelPercent: batteryPercent,
        chargingState: chargingState
      ))
    defer { eventBox.release() }
    eventBox.takeUnretainedValue().appendResource(
      makeResourceSample(
        boundary: "start",
        iteration: nil,
        startedNanoseconds: started,
        batteryLevelPercent: batteryPercent,
        chargingState: chargingState
      ))

    guard
      let job = brush_train_start_v3(
        UnsafePointer(datasetCString),
        &options,
        eventCallback,
        eventBox.toOpaque()
      )
    else {
      return XCTFail("V3 job creation failed")
    }
    let status = brush_job_wait_v2(job)
    brush_job_release_v2(job)
    let ended = DispatchTime.now().uptimeNanoseconds
    let endBatteryPercent = max(0, Int((UIDevice.current.batteryLevel * 100).rounded()))
    let endChargingState = batteryStateName(UIDevice.current.batteryState)
    eventBox.takeUnretainedValue().appendResource(
      makeResourceSample(
        boundary: "end",
        iteration: 300,
        startedNanoseconds: started,
        batteryLevelPercent: endBatteryPercent,
        chargingState: endChargingState
      ))

    let exitCode = Int32(status.rawValue)
    XCTAssertEqual(exitCode, 0)
    let output = outputURL.appendingPathComponent("component-0_300.ply")
    XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
    let validation = try validatePLY(output)
    XCTAssertEqual(validation.vertexCount, 1_038)
    XCTAssertEqual(validation.nonFiniteValueCount, 0)

    let snapshot = eventBox.takeUnretainedValue().snapshot()
    let capabilities = snapshot.0.first(where: { $0.kind == 0 })?.capabilityFlags ?? 0
    let initializer = snapshot.0.first(where: { $0.kind == 4 })
    let terminal = snapshot.0.last(where: { $0.kind == 11 })
    XCTAssertEqual(initializer?.initialPrimitiveCount, 832)
    XCTAssertEqual(terminal?.errorCode, 0)
    XCTAssertTrue(snapshot.0.contains(where: { $0.kind == 12 }))

    let result = BenchmarkResult(
      runLabel: runLabel,
      callableABIVersion: brush_get_abi_version(),
      nativeIdentityABIVersion: identity.abi_version,
      nativeBuildRevision: revision,
      crateVersion: crateVersion,
      graphicsBackend: graphicsBackend,
      adapterName: adapterName,
      adapterIdentityAvailable: identity.adapter_identity_available,
      deviceName: UIDevice.current.name,
      deviceModel: UIDevice.current.model,
      systemName: UIDevice.current.systemName,
      systemVersion: UIDevice.current.systemVersion,
      activeProcessorCount: ProcessInfo.processInfo.activeProcessorCount,
      fixtureManifestSHA256: fixtureChecksum,
      seed: options.seed,
      renderMode: options.render_mode,
      shDegree: options.sh_degree,
      totalTrainSteps: options.total_train_steps,
      refineEvery: options.refine_every,
      maxResolution: options.max_resolution,
      maxSplats: options.max_splats,
      exportEvery: options.export_every,
      progressiveResolutionStartPercent: options.progressive_resolution_start_percent,
      progressiveResolutionSwitchIteration: options.progressive_resolution_switch_iteration,
      instrumentationLevel: options.instrumentation_level,
      trainerWallTimeSeconds: Double(ended - started) / 1_000_000_000,
      exitCode: exitCode,
      eventKinds: snapshot.0.map(\.kind),
      capabilityFlags: capabilities,
      initialPrimitiveCount: initializer?.initialPrimitiveCount ?? 0,
      terminalFinalPrimitiveCount: terminal?.finalPrimitiveCount ?? 0,
      terminalPrunedNonFiniteCount: terminal?.prunedNonFiniteCount ?? 0,
      terminalText: terminal?.text,
      resourceSamples: snapshot.1,
      stepSamples: snapshot.2,
      outputValidation: validation
    )
    let resultData = try JSONEncoder.sorted.encode(result)
    let resultJSON = String(decoding: resultData, as: UTF8.self)
    print("BRUSHKIT_DEVICE_BENCHMARK_RESULT=\(resultJSON)")
    let attachment = XCTAttachment(data: resultData, uniformTypeIdentifier: "public.json")
    attachment.name = "BrushKit-device-benchmark-\(runLabel).json"
    attachment.lifetime = .keepAlways
    add(attachment)
    let outputAttachment = XCTAttachment(
      data: try Data(contentsOf: output), uniformTypeIdentifier: "public.data")
    outputAttachment.name = "component-0_300-\(runLabel).ply"
    outputAttachment.lifetime = .keepAlways
    add(outputAttachment)
  }
}

extension JSONEncoder {
  fileprivate static var sorted: JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return encoder
  }
}
