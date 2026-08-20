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
  var batteryObservation: String
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
  var addedCount: UInt32
  var prunedCount: UInt32
  var prunedNonFiniteCount: UInt32
  var netGrowth: Int64
  var text: String?
}

private struct PopulationEvent: Codable, Sendable {
  var kind: Int32
  var iteration: UInt32
  var primitiveCount: UInt32
  var addedCount: UInt32
  var prunedCount: UInt32
  var prunedNonFiniteCount: UInt32
  var netGrowth: Int64
}

private struct InputValidation: Codable, Sendable {
  var fileCount: Int
  var byteCount: UInt64
  var datasetSHA256: String
  var originalControlCheckpointDatasetSHA256: String
  var camerasSHA256: String
  var imagesSHA256: String
  var points3DSHA256: String
  var trainingInvariantFingerprint: String
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
  var inputValidation: InputValidation
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
  var refinementEvents: [PopulationEvent]
  var terminalCompaction: PopulationEvent
  var accountedFinalPrimitiveCount: Int64
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
  private let startBatteryLevelPercent: Int
  private let startChargingState: String
  private var events: [CapturedEvent] = []
  private var resources: [ResourceSample] = []
  private var steps: [StepSample] = []

  init(
    startedNanoseconds: UInt64,
    startBatteryLevelPercent: Int,
    startChargingState: String
  ) {
    self.startedNanoseconds = startedNanoseconds
    self.startBatteryLevelPercent = startBatteryLevelPercent
    self.startChargingState = startChargingState
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
      addedCount: event.added_count,
      prunedCount: event.pruned_count,
      prunedNonFiniteCount: event.pruned_non_finite_count,
      netGrowth: event.net_growth,
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
        batteryLevelPercent: startBatteryLevelPercent,
        chargingState: startChargingState,
        batteryObservation: "start-snapshot"
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
  chargingState: String,
  batteryObservation: String
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
    batteryObservation: batteryObservation,
    lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled,
    residentMemoryBytes: memory.resident,
    peakResidentMemoryBytes: memory.peak
  )
}

private func sha256(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func datasetIdentity(_ directory: URL) throws -> (String, Int, UInt64, [String: String]) {
  let manager = FileManager.default
  let baseComponentCount = directory.standardizedFileURL.pathComponents.count
  func relativePath(_ url: URL) -> String {
    url.standardizedFileURL.pathComponents.dropFirst(baseComponentCount).joined(separator: "/")
  }
  guard
    let enumerator = manager.enumerator(
      at: directory,
      includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
      options: [.skipsHiddenFiles]
    )
  else { throw CocoaError(.fileReadNoSuchFile) }
  let files = try enumerator.compactMap { value -> URL? in
    guard let url = value as? URL,
      try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
    else { return nil }
    return url
  }.sorted { relativePath($0) < relativePath($1) }

  var hasher = SHA256()
  var totalBytes: UInt64 = 0
  var selectedHashes: [String: String] = [:]
  for file in files {
    let relative = relativePath(file)
    hasher.update(data: Data(relative.utf8))
    hasher.update(data: Data([0]))
    var fileHasher = SHA256()
    let handle = try FileHandle(forReadingFrom: file)
    do {
      while true {
        let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
        if data.isEmpty { break }
        hasher.update(data: data)
        fileHasher.update(data: data)
        totalBytes += UInt64(data.count)
      }
      try handle.close()
    } catch {
      try? handle.close()
      throw error
    }
    hasher.update(data: Data([0]))
    if ["cameras.bin", "images.bin", "points3D.bin"].contains(relative) {
      selectedHashes[relative] = fileHasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
  }
  let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
  return (digest, files.count, totalBytes, selectedHashes)
}

private struct FrozenTrainingInvariant: Codable {
  var componentID = "0"
  var exportNameTemplate = "post-ad-step-{iter}.ply"
  var maxResolution = 1080
  var maxSplats = 500_000
  var renderMode = "mip"
  var refineEvery = 200
  var randomSeed: UInt64 = 0
  var trainingSHDegree = 3
  var initializerRelativePath: String? = nil
  var initializerRequired = false
  var progressiveResolutionStartPercent: Int? = 50
  var progressiveResolutionSwitchIteration: Int? = 200
}

private func trainingInvariantFingerprint() throws -> String {
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  return sha256(try encoder.encode(FrozenTrainingInvariant()))
}

private func populationEvent(_ event: CapturedEvent) -> PopulationEvent {
  PopulationEvent(
    kind: event.kind,
    iteration: event.iteration,
    primitiveCount: event.primitiveCount,
    addedCount: event.addedCount,
    prunedCount: event.prunedCount,
    prunedNonFiniteCount: event.prunedNonFiniteCount,
    netGrowth: event.netGrowth
  )
}

private func hardwareModelName() -> String {
  var systemInfo = utsname()
  uname(&systemInfo)
  let capacity = MemoryLayout.size(ofValue: systemInfo.machine)
  return withUnsafePointer(to: &systemInfo.machine) {
    $0.withMemoryRebound(to: CChar.self, capacity: capacity) {
      String(cString: $0)
    }
  }
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
  func testTrackedPoseFrozen600TrainerLeaf() throws {
    UIDevice.current.isBatteryMonitoringEnabled = true
    guard ProcessInfo.processInfo.thermalState == .nominal else {
      return XCTFail("The physical correctness run requires nominal thermal state at launch")
    }
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
    let identity = try datasetIdentity(datasetURL)
    let invariantFingerprint = try trainingInvariantFingerprint()
    let camerasDigest = identity.3["cameras.bin"] ?? "missing"
    let imagesDigest = identity.3["images.bin"] ?? "missing"
    let points3DDigest = identity.3["points3D.bin"] ?? "missing"
    guard identity.0 == "764504ec6d7fcc1c3b6cb296897d5f27ff6496e320d89f1c5f6efd5c58bf8c2d",
      identity.1 == 110,
      identity.2 == 358_272_040,
      identity.3["cameras.bin"]
        == "fb77970aa4904809074c6367a67c7e54fc524f1894fd7db6eb8fb3d0227e0d10",
      identity.3["images.bin"]
        == "b87a16ffbb523f3eb3663e03d554a2403e6796c6a28953544bfa2defe3e299b1",
      identity.3["points3D.bin"]
        == "14fb2891d9bc6795e45b7f29333963b8d708617412a94935e80c6867bf6ddd7b",
      invariantFingerprint
        == "8f7201b9ff6ed9813da4a3ac1abde5f4b74bbc5e84a4da76de6783345708ae94"
    else {
      return XCTFail(
        "Retained trainer leaf or frozen training invariant changed before launch: "
          + "dataset=\(identity.0) files=\(identity.1) bytes=\(identity.2) "
          + "cameras=\(camerasDigest) images=\(imagesDigest) points3D=\(points3DDigest) "
          + "invariants=\(invariantFingerprint)"
      )
    }

    XCTAssertEqual(brush_get_abi_version(), 2)
    var nativeIdentity = BrushNativeIdentityV2()
    XCTAssertTrue(
      brush_get_native_identity_v2(
        &nativeIdentity,
        UInt32(MemoryLayout<BrushNativeIdentityV2>.size)
      ))
    let revision = nativeIdentity.build_revision.map(String.init(cString:)) ?? "missing"
    let crateVersion = nativeIdentity.crate_version.map(String.init(cString:)) ?? "missing"
    let graphicsBackend = nativeIdentity.graphics_backend.map(String.init(cString:)) ?? "missing"
    let adapterName = nativeIdentity.adapter_name.map(String.init(cString:)) ?? "missing"
    guard nativeIdentity.abi_version == 2,
      revision == "b77dc41172c339ad8792413c22f75360e7a9b0ce",
      crateVersion == "0.3.1",
      graphicsBackend == "Metal"
    else {
      return XCTFail("Linked native identity is not the audited b77dc411 Metal candidate")
    }

    guard let datasetCString = strdup(datasetURL.path),
      let outputCString = strdup(outputURL.path),
      let exportCString = strdup("post-ad-step-{iter}.ply")
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
    options.total_train_steps = 600
    options.refine_every = 200
    options.max_resolution = 1080
    options.max_splats = 500_000
    options.export_every = 600
    options.output_path = UnsafePointer(outputCString)
    options.export_name = UnsafePointer(exportCString)
    options.initializer_path = nil
    options.initializer_required = 0
    options.instrumentation_level = 1
    options.progressive_resolution_start_percent = 50
    options.progressive_resolution_switch_iteration = 200

    let startBatteryLevelPercent = max(
      0, Int((UIDevice.current.batteryLevel * 100).rounded()))
    let startChargingState = batteryStateName(UIDevice.current.batteryState)
    let started = DispatchTime.now().uptimeNanoseconds
    let eventBox = Unmanaged.passRetained(
      EventBox(
        startedNanoseconds: started,
        startBatteryLevelPercent: startBatteryLevelPercent,
        startChargingState: startChargingState
      ))
    defer { eventBox.release() }
    eventBox.takeUnretainedValue().appendResource(
      makeResourceSample(
        boundary: "start",
        iteration: nil,
        startedNanoseconds: started,
        batteryLevelPercent: startBatteryLevelPercent,
        chargingState: startChargingState,
        batteryObservation: "live"
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
    let endBatteryLevelPercent = max(0, Int((UIDevice.current.batteryLevel * 100).rounded()))
    let endChargingState = batteryStateName(UIDevice.current.batteryState)
    eventBox.takeUnretainedValue().appendResource(
      makeResourceSample(
        boundary: "end",
        iteration: 600,
        startedNanoseconds: started,
        batteryLevelPercent: endBatteryLevelPercent,
        chargingState: endChargingState,
        batteryObservation: "live"
      ))

    let exitCode = Int32(status.rawValue)
    XCTAssertEqual(exitCode, 0)
    let output = outputURL.appendingPathComponent("post-ad-step-600.ply")
    XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
    let validation = try validatePLY(output)
    XCTAssertEqual(validation.nonFiniteValueCount, 0)

    let snapshot = eventBox.takeUnretainedValue().snapshot()
    let capabilities = snapshot.0.first(where: { $0.kind == 0 })?.capabilityFlags ?? 0
    let initializer = snapshot.0.first(where: { $0.kind == 4 })
    let refinements = snapshot.0.filter { $0.kind == 8 }
    let terminalCompaction = try XCTUnwrap(snapshot.0.last(where: { $0.kind == 12 }))
    let terminal = try XCTUnwrap(snapshot.0.last(where: { $0.kind == 11 }))
    XCTAssertEqual(initializer?.initialPrimitiveCount, 5_641)
    XCTAssertEqual(refinements.map(\.iteration), [201, 401])
    XCTAssertEqual(terminalCompaction.prunedNonFiniteCount, 0)
    XCTAssertEqual(terminalCompaction.prunedCount, 0)
    XCTAssertEqual(terminal.errorCode, 0)
    XCTAssertEqual(terminalCompaction.primitiveCount, UInt32(validation.vertexCount))
    XCTAssertEqual(terminal.finalPrimitiveCount, UInt32(validation.vertexCount))
    let accountedFinal =
      Int64(initializer?.initialPrimitiveCount ?? 0)
      + refinements.reduce(0) { $0 + $1.netGrowth }
      + terminalCompaction.netGrowth
    XCTAssertEqual(accountedFinal, Int64(validation.vertexCount))
    let compactionIndex = try XCTUnwrap(snapshot.0.lastIndex(where: { $0.kind == 12 }))
    let terminalIndex = try XCTUnwrap(snapshot.0.lastIndex(where: { $0.kind == 11 }))
    XCTAssertTrue(
      snapshot.0.enumerated().contains { index, event in
        index > compactionIndex && index < terminalIndex && event.kind == 7
          && event.iteration == 600
      },
      "The retained failure ordering requires a delayed final Step after terminal compaction"
    )

    let result = BenchmarkResult(
      runLabel: runLabel,
      callableABIVersion: brush_get_abi_version(),
      nativeIdentityABIVersion: nativeIdentity.abi_version,
      nativeBuildRevision: revision,
      crateVersion: crateVersion,
      graphicsBackend: graphicsBackend,
      adapterName: adapterName,
      adapterIdentityAvailable: nativeIdentity.adapter_identity_available,
      deviceName: UIDevice.current.name,
      deviceModel: hardwareModelName(),
      systemName: UIDevice.current.systemName,
      systemVersion: UIDevice.current.systemVersion,
      activeProcessorCount: ProcessInfo.processInfo.activeProcessorCount,
      inputValidation: InputValidation(
        fileCount: identity.1,
        byteCount: identity.2,
        datasetSHA256: identity.0,
        originalControlCheckpointDatasetSHA256:
          "ae9332c46bef2c48ca8c66dea787950095952b5dbb897293a9381ee7956118cf",
        camerasSHA256: identity.3["cameras.bin"] ?? "missing",
        imagesSHA256: identity.3["images.bin"] ?? "missing",
        points3DSHA256: identity.3["points3D.bin"] ?? "missing",
        trainingInvariantFingerprint: invariantFingerprint
      ),
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
      refinementEvents: refinements.map(populationEvent),
      terminalCompaction: populationEvent(terminalCompaction),
      accountedFinalPrimitiveCount: accountedFinal,
      terminalFinalPrimitiveCount: terminal.finalPrimitiveCount,
      terminalPrunedNonFiniteCount: terminalCompaction.prunedNonFiniteCount,
      terminalText: terminal.text,
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
    outputAttachment.name = "post-ad-step-600-\(runLabel).ply"
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
