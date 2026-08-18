// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "BrushKit",
  platforms: [
    .macOS(.v13),
    .iOS(.v18)
  ],
  products: [
    .library(name: "BrushKit", targets: ["BrushKit"])
  ],
  targets: [
    .target(
      name: "BrushKit",
      dependencies: ["BrushKitFFI"],
      linkerSettings: [
        .linkedFramework("QuartzCore"),
        .linkedFramework("Metal"),
        .linkedFramework("Foundation"),
        .linkedFramework("CoreFoundation"),
        .linkedLibrary("objc"),
        .linkedLibrary("iconv"),
        .linkedLibrary("c"),
        .linkedLibrary("m"),
      ]
    ),
    .binaryTarget(
      name: "BrushKitFFI",
      url: "https://github.com/benwilliams0540/BrushKit/releases/download/brushkit-ffi-v0.3.1/BrushKitFFI.xcframework.zip",
      checksum: "2654bf3e5f5b12b5685f05b0bdfa36dbda66cc0bd1c37fc223f874724d07dbc1"
    )
  ]
)
