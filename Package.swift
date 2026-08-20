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
      url: "https://github.com/benwilliams0540/BrushKit/releases/download/brushkit-ffi-v0.4.0/BrushKitFFI.xcframework.zip",
      checksum: "288e571bdc7da71dcded5a2ab075c05baf145734e329acf3c93b5de892ca60a7"
    )
  ]
)
