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
      url: "https://github.com/benwilliams0540/BrushKit/releases/download/brushkit-ffi-v0.2.1/BrushKitFFI.xcframework.zip",
      checksum: "e64ad4fc19416d9f8e47a4721fc87e9cd88a53c8ddd0c0ef31633f8ccafea33b"
    )
  ]
)
