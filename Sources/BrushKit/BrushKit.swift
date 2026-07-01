import BrushKitFFI
import Darwin
import Foundation

public struct BrushTrainingOptions: Equatable, Sendable {
  public var totalTrainSteps: UInt32
  public var refineEvery: UInt32
  public var maxResolution: UInt32
  public var maxSplats: UInt32
  public var exportEvery: UInt32
  public var exportName: String
  public var outputURL: URL

  public init(
    totalTrainSteps: UInt32,
    refineEvery: UInt32 = 5,
    maxResolution: UInt32,
    maxSplats: UInt32 = 10_000_000,
    exportEvery: UInt32,
    exportName: String = "export_{iter}.ply",
    outputURL: URL
  ) {
    self.totalTrainSteps = totalTrainSteps
    self.refineEvery = refineEvery
    self.maxResolution = maxResolution
    self.maxSplats = maxSplats
    self.exportEvery = exportEvery
    self.exportName = exportName
    self.outputURL = outputURL
  }
}

public enum BrushTrainingEvent: Equatable, Sendable {
  case started
  case training(iteration: UInt32)
  case checkpointExported(iteration: UInt32, path: String?)
  case finished
}

public enum BrushTrainingStatus: Equatable, Sendable {
  case success
  case error
  case cancelled

  init(_ code: TrainExitCode) {
    switch Int32(code.rawValue) {
    case 0:
      self = .success
    case 2:
      self = .cancelled
    default:
      self = .error
    }
  }
}

public enum BrushKitError: Error, Equatable, LocalizedError, Sendable {
  case invalidPath(String)
  case startFailed

  public var errorDescription: String? {
    switch self {
    case let .invalidPath(path):
      "BrushKit could not convert path to a C string: \(path)"
    case .startFailed:
      "BrushKit could not start the training job."
    }
  }
}

public final class BrushTrainingJob: @unchecked Sendable {
  private let lock = NSLock()
  private var handle: UnsafeMutablePointer<BrushJob>?
  private var callbackBox: Unmanaged<BrushProgressCallbackBox>?

  fileprivate init(
    handle: UnsafeMutablePointer<BrushJob>,
    callbackBox: Unmanaged<BrushProgressCallbackBox>
  ) {
    self.handle = handle
    self.callbackBox = callbackBox
  }

  deinit {
    release()
  }

  @discardableResult
  public func cancel() -> Bool {
    lock.lock()
    let currentHandle = handle
    lock.unlock()
    guard let currentHandle else { return false }
    return brush_job_cancel(currentHandle)
  }

  public func wait() -> BrushTrainingStatus {
    lock.lock()
    let currentHandle = handle
    lock.unlock()
    guard let currentHandle else { return .error }
    return BrushTrainingStatus(brush_job_wait(currentHandle))
  }

  public func release() {
    lock.lock()
    let currentHandle = handle
    let currentCallbackBox = callbackBox
    handle = nil
    callbackBox = nil
    lock.unlock()

    if let currentHandle {
      brush_job_release(currentHandle)
    }
    currentCallbackBox?.release()
  }
}

public enum BrushKit {
  public static func startTraining(
    datasetURL: URL,
    options: BrushTrainingOptions,
    events: @escaping @Sendable (BrushTrainingEvent) -> Void
  ) throws -> BrushTrainingJob {
    let datasetPath = datasetURL.path(percentEncoded: false)
    let outputPath = options.outputURL.path(percentEncoded: false)
    let exportName = options.exportName
    guard let datasetCString = strdup(datasetPath) else {
      throw BrushKitError.invalidPath(datasetPath)
    }
    defer { free(datasetCString) }
    guard let outputCString = strdup(outputPath) else {
      throw BrushKitError.invalidPath(outputPath)
    }
    defer { free(outputCString) }
    guard let exportNameCString = strdup(exportName) else {
      throw BrushKitError.invalidPath(exportName)
    }
    defer { free(exportNameCString) }

    var ffiOptions = TrainOptions(
      total_train_steps: options.totalTrainSteps,
      refine_every: options.refineEvery,
      max_resolution: options.maxResolution,
      max_splats: options.maxSplats,
      export_every: options.exportEvery,
      output_path: UnsafePointer(outputCString),
      export_name: UnsafePointer(exportNameCString)
    )
    let callbackBox = Unmanaged.passRetained(BrushProgressCallbackBox(events: events))
    guard let handle = brush_train_start(
      UnsafePointer(datasetCString),
      &ffiOptions,
      brushProgressCallback,
      callbackBox.toOpaque()
    ) else {
      callbackBox.release()
      throw BrushKitError.startFailed
    }
    return BrushTrainingJob(handle: handle, callbackBox: callbackBox)
  }
}

private final class BrushProgressCallbackBox: @unchecked Sendable {
  let events: @Sendable (BrushTrainingEvent) -> Void

  init(events: @escaping @Sendable (BrushTrainingEvent) -> Void) {
    self.events = events
  }
}

private let brushProgressCallback: @convention(c) (
  ProgressMessage,
  UnsafeMutableRawPointer?
) -> Void = { message, userData in
  guard let userData else { return }
  let box = Unmanaged<BrushProgressCallbackBox>
    .fromOpaque(userData)
    .takeUnretainedValue()

  switch Int32(message.kind.rawValue) {
  case 0:
    box.events(.started)
  case 1:
    box.events(.training(iteration: message.iter))
  case 2:
    let path = message.path.map { String(cString: $0) }
    box.events(.checkpointExported(iteration: message.iter, path: path))
  case 3:
    box.events(.finished)
  default:
    break
  }
}
