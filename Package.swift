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
      url: "https://github.com/benwilliams0540/BrushKit/releases/download/brushkit-ffi-v0.2.0/BrushKitFFI.xcframework.zip",
      checksum: "a751e21a360e3a762d38993cacb0342346454d82c9941ceb7ab9c89d27512b1f"
    )
  ]
)
