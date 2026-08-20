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
      url: "https://github.com/benwilliams0540/BrushKit/releases/download/brushkit-ffi-v0.3.2/BrushKitFFI.xcframework.zip",
      checksum: "66151a697763aad7e8ef2aebac1b883b6517a6500d1dc391ba7f7e03a56fa9b6"
    )
  ]
)
