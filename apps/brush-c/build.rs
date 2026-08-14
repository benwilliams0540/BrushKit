use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=BRUSHKIT_BUILD_REVISION");

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
    println!(
        "cargo:rustc-env=BRUSHKIT_BUILD_REVISION={}",
        revision.as_deref().unwrap_or("unknown")
    );
}
