use alloc::{boxed::Box, vec::Vec};
#[cfg(feature = "atomic")]
use core::sync::atomic::Ordering::Relaxed;

/// The number of bits in the bit mask that is used to index a u64's bits.
///
/// u64's are used to store 64 bits, so the index ranges from 0 to 63.
const BIT_MASK_LEN: u32 = u32::ilog2(u64::BITS);

/// Gets 6 last bits from the bit index, which are used to index a u64's bits.
const BIT_MASK: u64 = (1 << BIT_MASK_LEN) - 1;

use crate::number::*;

/// A bit vector partitioned in to `u64` blocks.
#[derive(Debug)]
#[cfg_attr(not(feature = "atomic"), derive(Clone))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct BlockedBitVec {
    bits: Box<[BitStorage]>,
}

impl BlockedBitVec {
    #[inline]
    pub(crate) const fn len(&self) -> usize {
        self.bits.len()
    }

    #[inline]
    pub(crate) const fn num_bits(&self) -> usize {
        self.len() * u64::BITS as usize
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[BitStorage] {
        &self.bits
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.bits.iter().map(fetch)
    }

    #[inline(always)]
    pub(crate) fn check(&self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        fetch(&self.bits[index]) & bit > 0
    }

    #[cfg(not(feature = "atomic"))]
    #[inline(always)]
    pub(crate) fn set(&mut self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        let previously_contained = fetch(&self.bits[index]) & bit > 0;
        self.bits[index] |= bit;
        previously_contained
    }

    #[cfg(feature = "atomic")]
    #[inline(always)]
    pub(crate) fn set(&self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        let previously_contained = fetch(&self.bits[index]) & bit > 0;
        self.bits[index].fetch_or(bit, Relaxed);
        previously_contained
    }

    #[cfg(feature = "atomic")]
    #[inline]
    pub(crate) fn clear(&self) {
        for i in 0..self.len() {
            self.bits[i].store(0, Relaxed);
        }
    }

    #[cfg(not(feature = "atomic"))]
    #[inline]
    pub(crate) fn clear(&mut self) {
        for i in 0..self.len() {
            self.bits[i] = 0;
        }
    }
}

impl FromIterator<u64> for BlockedBitVec {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let mut bits = iter.into_iter().map(new).collect::<Vec<_>>();
        bits.shrink_to_fit();
        Self { bits: bits.into() }
    }
}

impl PartialEq for BlockedBitVec {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        core::iter::zip(self.iter(), other.iter()).all(|(l, r)| l == r)
    }
}
impl Eq for BlockedBitVec {}

#[cfg(feature = "atomic")]
impl Clone for BlockedBitVec {
    fn clone(&self) -> Self {
        self.iter().collect()
    }
}

#[allow(unused_mut)]
#[cfg(test)]
mod tests {
    use super::*;
    use core::iter::repeat;
    use rand::Rng;

    #[test]
    fn test_to_from_vec() {
        let size = 42;
        let b: BlockedBitVec = repeat(0).take(size).collect();
        assert_eq!(b.num_bits(), b.len() * 64);
        assert!(size <= b.len());
        assert!((size + 64) > b.len());
    }

    #[test]
    fn test_only_random_inserts_are_contained() {
        let mut vec: BlockedBitVec = repeat(0).take(80).collect();
        let mut control = Vec::with_capacity(1000);
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let block_index = rng.random_range(0..vec.num_bits() / 64);
            let bit_index = rng.random_range(0..64);

            if !control.contains(&(block_index, bit_index)) {
                assert!(!vec.check(block_index, bit_index));
            }
            control.push((block_index, bit_index));
            vec.set(block_index, bit_index);
            assert!(vec.check(block_index, bit_index));
        }
    }
}
