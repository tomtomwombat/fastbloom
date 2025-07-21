#[cfg(feature = "atomic")]
use core::sync::atomic::Ordering::Relaxed;

#[cfg(all(feature = "atomic", feature = "loom"))]
pub(crate) use loom::sync::atomic::AtomicU64;

#[cfg(all(feature = "atomic", not(feature = "loom")))]
pub(crate) use core::sync::atomic::AtomicU64;

#[cfg(not(feature = "atomic"))]
pub(crate) type BitStorage = u64;
#[cfg(feature = "atomic")]
pub(crate) type BitStorage = AtomicU64;

#[cfg(not(feature = "atomic"))]
#[inline(always)]
pub(crate) fn fetch(x: &BitStorage) -> u64 {
    *x
}

#[cfg(feature = "atomic")]
#[inline(always)]
pub(crate) fn fetch(x: &BitStorage) -> u64 {
    x.load(Relaxed)
}

#[cfg(not(feature = "atomic"))]
#[inline(always)]
pub(crate) fn new(x: u64) -> BitStorage {
    x
}

#[cfg(feature = "atomic")]
#[inline(always)]
pub(crate) fn new(x: u64) -> BitStorage {
    BitStorage::new(x)
}
