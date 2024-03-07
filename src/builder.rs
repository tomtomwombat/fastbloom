use crate::{BlockedBitVec, BloomFilter, BuildHasher, DefaultHasher};
use std::hash::Hash;

use crate::signature;

/// A bloom filter builder.
///
/// This type can be used to construct an instance of [`BloomFilter`]
/// via the builder pattern.
#[derive(Debug, Clone)]
pub struct Builder<const BLOCK_SIZE_BITS: usize = 512, S = DefaultHasher> {
    pub(crate) data: BlockedBitVec<BLOCK_SIZE_BITS>,
    pub(crate) hasher: S,
}

impl<const BLOCK_SIZE_BITS: usize> Builder<BLOCK_SIZE_BITS> {
    /// Sets the seed for this builder. The later constructed [`BloomFilter`]
    /// will use this seed when hashing items.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).seed(&1).hashes(4);
    /// ```
    pub fn seed(mut self, seed: &u128) -> Self {
        self.hasher = DefaultHasher::seeded(&seed.to_be_bytes());
        self
    }
}

impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> Builder<BLOCK_SIZE_BITS, S> {
    /// Sets the hasher for this builder. The later constructed [`BloomFilter`] will use
    /// this hasher when inserting and checking items.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    /// use ahash::RandomState;
    ///
    /// let bloom = BloomFilter::builder(1024).hasher(RandomState::default()).hashes(4);
    /// ```
    pub fn hasher<H: BuildHasher>(self, hasher: H) -> Builder<BLOCK_SIZE_BITS, H> {
        Builder::<BLOCK_SIZE_BITS, H> {
            data: self.data,
            hasher,
        }
    }

    /// "Consumes" this builder, using the provided `num_hashes` to return an
    /// empty [`BloomFilter`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).hashes(4);
    /// ```
    pub fn hashes(self, num_hashes: u32) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        self.hashes_f(num_hashes as f64)
    }

    /// To generate ~`total_num_hashes` we'll use a combination of traditional index derived from hashes and "signatures".
    /// Signature's are per u64 in the block, and for that u64 represent some indexes already set.
    /// "rounds" are the amount of work/iterations we need to do to get a signature.
    /// For more on signatures, see "BloomFilter::signature".
    ///
    /// For example, if our target total hashes 40, and we have a block of two u64s,
    /// we'll require ~40 bits (ignoring probability collisions for simplicity in this example) set across the two u64s.
    /// for each u64 in the block, generate two signatures each with about 16 bits set (2 rounds each).
    /// then calcuate 8 bit indexes from the hash to cover the remaining. 16 + 16 + 8 = 40.
    /// the total work here is 4 rounds + 8 hashes, instead of 40 hashes.
    ///
    /// Note:
    /// - the min number of rounds is 1, generating around ~32 bits, which is the max entropy in the u64.
    /// - the max number of rounds is ~4. That produces a signature of ~4 bits set (1/2^4), at which point we may as well calculate 4 bit indexes normally.
    fn hashes_f(self, total_num_hashes: f64) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        let (num_hashes, num_rounds) =
            signature::optimize_hashing(total_num_hashes, BLOCK_SIZE_BITS);

        BloomFilter {
            bits: self.data,
            target_hashes: total_num_hashes as u64,
            num_hashes,
            num_rounds,
            hasher: self.hasher,
        }
    }

    /// "Consumes" this builder, using the provided `expected_num_items` to return an
    /// empty [`BloomFilter`]. The number of hashes is optimized based on `expected_num_items`
    /// to maximize bloom filter accuracy (minimize false positives chance on [`BloomFilter::contains`]).
    /// More or less than `expected_num_items` may be inserted into [`BloomFilter`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).expected_items(500);
    /// ```
    pub fn expected_items(self, expected_num_items: usize) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        let items_per_block = expected_num_items as f64 / self.data.num_blocks() as f64;
        let num_hashes = BloomFilter::<BLOCK_SIZE_BITS>::optimal_hashes_f(items_per_block);
        self.hashes_f(num_hashes)
    }

    /// "Consumes" this builder and constructs a [`BloomFilter`] containing
    /// all values in `items`. Like [`Builder::expected_items`], the number of hashes per item
    /// is optimized based on `items.len()` to maximize bloom filter accuracy
    /// (minimize false positives chance on [`BloomFilter::contains`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).items([1, 2, 3]);
    /// ```
    pub fn items<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl Hash>>>(
        self,
        items: I,
    ) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        let into_iter = items.into_iter();
        let mut filter = self.expected_items(into_iter.len());
        filter.extend(into_iter);
        filter
    }
}

#[cfg(test)]
mod tests {
    use crate::BloomFilter;
    use ahash::RandomState;

    #[test]
    fn data_size() {
        let size_bits = 512 * 1000;
        let bloom = BloomFilter::<512>::builder_from_bits(size_bits).hashes(4);
        assert_eq!(bloom.as_raw().len() * 64, size_bits);
        let bloom = BloomFilter::<256>::builder_from_bits(size_bits).hashes(4);
        assert_eq!(bloom.as_raw().len() * 64, size_bits);
        let bloom = BloomFilter::<128>::builder_from_bits(size_bits).hashes(4);
        assert_eq!(bloom.as_raw().len() * 64, size_bits);
        let bloom = BloomFilter::<64>::builder_from_bits(size_bits).hashes(4);
        assert_eq!(bloom.as_raw().len() * 64, size_bits);
    }

    #[test]
    fn api() {
        let _bloom = BloomFilter::<64>::builder_from_bits(10)
            .hasher(RandomState::default())
            .hashes(4);
    }

    #[test]
    fn specified_hashes() {
        for num_hashes in 1..1000 {
            let b = BloomFilter::<128>::builder_from_bits(1).hashes(num_hashes);
            assert_eq!(num_hashes, b.num_hashes());
        }
    }
}
