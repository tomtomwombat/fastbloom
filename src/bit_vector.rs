/// The number of bits in the bit mask that is used to index a u64's bits.
///
/// u64's are used to store 64 bits, so the index ranges from 0 to 63.
const BIT_MASK_LEN: u32 = u32::ilog2(u64::BITS);

/// Gets 6 last bits from the bit index, which are used to index a u64's bits.
const BIT_MASK: u64 = (1 << BIT_MASK_LEN) - 1;

/// A bit vector partitioned in to `u64` blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct BlockedBitVec {
    bits: Box<[u64]>,
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

    #[inline]
    pub(crate) fn set(&mut self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        let previously_contained = self.bits[index] & bit > 0;
        self.bits[index] |= bit;
        previously_contained
    }

    #[inline]
    pub(crate) const fn check(&self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        self.bits[index] & bit > 0
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u64] {
        &self.bits
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        for i in 0..self.bits.len() {
            self.bits[i] = 0;
        }
    }
}

impl From<Vec<u64>> for BlockedBitVec {
    fn from(mut bits: Vec<u64>) -> Self {
        let num_u64s_per_block = 1;
        let r = bits.len() % num_u64s_per_block;
        if r != 0 {
            bits.extend(vec![0; num_u64s_per_block - r]);
        }
        bits.shrink_to_fit();
        Self { bits: bits.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::collections::HashSet;

    #[test]
    fn test_to_from_vec() {
        let size = 42;
        let b: BlockedBitVec = vec![0u64; size].into();
        assert_eq!(b.num_bits(), b.as_slice().len() * 64);
        assert!(size <= b.as_slice().len());
        assert!((size + 64) > b.as_slice().len());
    }

    #[test]
    fn test_only_random_inserts_are_contained() {
        let mut vec = BlockedBitVec::from(vec![0; 80]);
        let mut control = HashSet::new();
        let mut rng = rand::rng();

        for _ in 0..100000 {
            let block_index = rng.random_range(0..vec.num_bits() / 64);
            let bit_index = rng.random_range(0..64);

            if !control.contains(&(block_index, bit_index)) {
                assert!(!vec.check(block_index, bit_index));
            }
            control.insert((block_index, bit_index));
            vec.set(block_index, bit_index);
            assert!(vec.check(block_index, bit_index));
        }
    }
}
