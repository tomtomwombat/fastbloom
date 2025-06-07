/// The number of bits in the bit mask that is used to index a u64's bits.
///
/// u64's are used to store 64 bits, so the index ranges from 0 to 63.
const BIT_MASK_LEN: u32 = u32::ilog2(u64::BITS);

/// Gets 6 last bits from the bit index, which are used to index a u64's bits.
const BIT_MASK: u64 = (1 << BIT_MASK_LEN) - 1;

/// A bit vector partitioned in to blocks.
///
/// Blocks are a power of 2 length array of u64's.
/// The bit size of blocks therefore can be 64, 128, 256, etc.
/// Only `BlockedBitVec`'s with block sizes following this rule can be constructed.
///
/// Loading a block, such as with `get_block`, is cache efficient.
/// Membership checks can be done locally inside a block.
///
/// Indexing a block is also efficient, since it can be done with bit operators because
/// the size of a block is a power of 2.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockedBitVec<const BLOCK_SIZE_BITS: usize> {
    bits: Box<[u64]>,
}

impl<const BLOCK_SIZE_BITS: usize> BlockedBitVec<BLOCK_SIZE_BITS> {
    /// Block size in u64s
    const BLOCK_SIZE: usize = BLOCK_SIZE_BITS / 64;
    /// Used to shift u64 index
    const LOG2_BLOCK_SIZE: u32 = u32::ilog2(Self::BLOCK_SIZE as u32);

    #[inline]
    pub const fn len(&self) -> usize {
        self.bits.len()
    }

    /// The number of blocks in the `BlockedBitVector`
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.bits.len() >> Self::LOG2_BLOCK_SIZE
    }

    #[inline]
    pub const fn set(&mut self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        let previously_contained = self.bits[index] & bit > 0;
        self.bits[index] |= bit;
        previously_contained
    }

    #[inline]
    pub const fn check(&self, index: usize, hash: u64) -> bool {
        let bit = 1u64 << (hash & BIT_MASK);
        self.bits[index] & bit > 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u64] {
        &self.bits
    }

    #[inline]
    pub fn clear(&mut self) {
        for i in 0..self.bits.len() {
            self.bits[i] = 0;
        }
    }
}

impl<const BLOCK_SIZE_BITS: usize> From<Vec<u64>> for BlockedBitVec<BLOCK_SIZE_BITS> {
    fn from(mut bits: Vec<u64>) -> Self {
        let num_u64s_per_block = BLOCK_SIZE_BITS / 64;
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
        fn to_from_<const N: usize>(size: usize) {
            let b: BlockedBitVec<N> = vec![0u64; size].into();
            assert_eq!(b.num_blocks() * N, b.as_slice().len() * 64);
            assert!(size <= b.as_slice().len());
            assert!((size + N) > b.as_slice().len());
        }
        for size in 1..=10009 {
            to_from_::<64>(size);
            to_from_::<128>(size);
            to_from_::<256>(size);
            to_from_::<512>(size);
        }
    }

    #[test]
    fn test_only_random_inserts_are_contained() {
        let mut vec = BlockedBitVec::<64>::from(vec![0; 80]);
        let mut control = HashSet::new();
        let mut rng = rand::rng();

        for _ in 0..100000 {
            let block_index = rng.random_range(0..vec.num_blocks());
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
