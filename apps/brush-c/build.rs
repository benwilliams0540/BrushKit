use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=BRUSHKIT_BUILD_REVISION");
    println!("cargo:rerun-if-env-changed=BRUSHKIT_CUBECL_GPU_PROFILE");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_IMAGE_LOSS_BWD_TILE_16");
    println!("cargo:rustc-check-cfg=cfg(brushkit_cubecl_gpu_profile)");

    let cubecl_gpu_profile = std::env::var("BRUSHKIT_CUBECL_GPU_PROFILE").as_deref() == Ok("1");
    if cubecl_gpu_profile {
        println!("cargo:rustc-cfg=brushkit_cubecl_gpu_profile");
    }

    if let Ok(output) = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .output()
        && output.status.success()
        && let Ok(reference) = String::from_utf8(output.stdout)
    {
        println!("cargo:rerun-if-changed=../../.git/{}", reference.trim());
    }

    let revision = std::env::var("BRUSHKIT_BUILD_REVISION").ok().or_else(|| {
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|revision| revision.trim().to_owned())
    });
    let revision = revision.as_deref().unwrap_or("unknown");
    println!("cargo:rustc-env=BRUSHKIT_BUILD_REVISION={revision}");

    let features = if std::env::var_os("CARGO_FEATURE_IMAGE_LOSS_BWD_TILE_16").is_some() {
        "image-loss-bwd-tile-16"
    } else {
        "none"
    };
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |version| version.trim().to_owned());
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());
    let crate_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_owned());
    let provenance = format!(
        "format=brushkit-build-provenance-v1;engine_source_sha={revision};crate_version={crate_version};callable_abi=2;additive_train_abi=3;apple_features={features};cubecl_gpu_profile={};rustc={rustc_version};target={target}",
        if cubecl_gpu_profile { "on" } else { "off" }
    );
    println!("cargo:rustc-env=BRUSHKIT_BUILD_PROVENANCE={provenance}");
}
