// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "BrushKit",
  platforms: [
    .macOS(.v13)
  ],
  products: [
    .library(name: "BrushKit", targets: ["BrushKit"])
  ],
  targets: [
    .target(
      name: "BrushKit",
      dependencies: ["BrushKitFFI"]
    ),
    .binaryTarget(
      name: "BrushKitFFI",
      path: "artifacts/BrushKitFFI.xcframework"
    )
  ]
)
