import ProjectDescription

let developmentTeam = "XKUXN9R3RW"
let baseSettings: SettingsDictionary = [
  "CODE_SIGN_IDENTITY": "-",
  "CODE_SIGN_IDENTITY[sdk=iphoneos*]": "Apple Development",
  "CODE_SIGN_STYLE": "Automatic",
  "DEVELOPMENT_TEAM": .string(developmentTeam),
  "ENABLE_USER_SCRIPT_SANDBOXING": "NO",
  "GENERATE_INFOPLIST_FILE": "YES",
  "SWIFT_VERSION": "6.0",
]
let settings = Settings.settings(base: baseSettings, defaultSettings: .recommended)
let hostedTestSettings = Settings.settings(
  base: baseSettings.merging([
    "BUNDLE_LOADER": "$(TEST_HOST)",
    "TEST_HOST":
      "$(BUILT_PRODUCTS_DIR)/BrushKitDeviceBenchmarkHost.app/BrushKitDeviceBenchmarkHost",
  ]) { _, newValue in newValue },
  defaultSettings: .recommended
)

let project = Project(
  name: "BrushKitDeviceBenchmark",
  targets: [
    .target(
      name: "BrushKitDeviceBenchmarkHost",
      destinations: [.iPhone, .iPad],
      product: .app,
      bundleId: "com.brw.brushkit.device-benchmark-host",
      deploymentTargets: .iOS("18.0"),
      infoPlist: .extendingDefault(with: [
        "UILaunchScreen": [
          "UIColorName": "",
          "UIImageName": "",
        ]
      ]),
      buildableFolders: [
        .folder(.relativeToManifest("Sources/BrushKitDeviceBenchmarkHost"))
      ],
      settings: settings
    ),
    .target(
      name: "BrushKitDeviceBenchmarkTests",
      destinations: [.iPhone, .iPad],
      product: .unitTests,
      bundleId: "com.brw.brushkit.device-benchmark-tests",
      deploymentTargets: .iOS("18.0"),
      infoPlist: nil,
      resources: [
        .folderReference(path: .relativeToManifest("Fixture/pinhole-fullres-component-0"))
      ],
      buildableFolders: [
        .folder(.relativeToManifest("Tests/BrushKitDeviceBenchmarkTests"))
      ],
      dependencies: [
        .target(name: "BrushKitDeviceBenchmarkHost"),
        .xcframework(path: .relativeToManifest("Vendor/BrushKitFFI.xcframework")),
      ],
      settings: hostedTestSettings
    ),
  ]
)
