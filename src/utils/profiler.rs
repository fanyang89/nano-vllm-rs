use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use burn::tensor::backend::Backend;

#[derive(Clone, Copy, Default)]
struct Stat {
    total: Duration,
    count: u64,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static STATS: OnceLock<Mutex<BTreeMap<&'static str, Stat>>> = OnceLock::new();

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("NANO_VLLM_PROFILE").is_ok())
}

fn stats() -> &'static Mutex<BTreeMap<&'static str, Stat>> {
    STATS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub struct Scope {
    name: &'static str,
    started: Instant,
}

impl Scope {
    pub fn new(name: &'static str) -> Option<Self> {
        if enabled() {
            Some(Self {
                name,
                started: Instant::now(),
            })
        } else {
            None
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        let mut guard = stats().lock().expect("profiler mutex poisoned");
        let entry = guard.entry(self.name).or_default();
        entry.total += elapsed;
        entry.count += 1;
    }
}

pub fn reset() {
    if enabled() {
        stats().lock().expect("profiler mutex poisoned").clear();
    }
}

pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }

    let guard = stats().lock().expect("profiler mutex poisoned");
    if guard.is_empty() {
        return Some("profile: no samples".to_string());
    }

    let mut out = String::from("profile:\n");
    for (name, stat) in guard.iter() {
        let total_ms = stat.total.as_secs_f64() * 1000.0;
        let avg_ms = total_ms / stat.count as f64;
        out.push_str(&format!(
            "  {name:<28} total={total_ms:>9.3} ms  count={:>5}  avg={avg_ms:>8.3} ms\n",
            stat.count
        ));
    }
    Some(out)
}

pub fn sync_backend<B: Backend>(device: &B::Device) -> Result<()> {
    if enabled() {
        B::sync(device).map_err(|err| anyhow::anyhow!(err.to_string()))?;
    }
    Ok(())
}
