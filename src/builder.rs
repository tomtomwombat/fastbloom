use crate::{BlockedBitVec, BloomFilter, BuildHasher, DefaultHasher};
use std::hash::Hash;

/// A bloom filter builder.
///
/// This type can be used to construct an instance of [`BloomFilter`]
/// via the builder pattern.
#[derive(Debug, Clone)]
pub struct Builder<const BLOCK_SIZE_BITS: usize = 512, S = DefaultHasher> {
    pub(crate) num_blocks: usize,
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
            num_blocks: self.num_blocks,
            hasher,
        }
    }

    /// "Consumes" this builder, using the provided `num_hashes` to return an
    /// empty [`BloomFilter`]. For performance, the actual number of
    /// hashes performed internally will be rounded to down to a power of 2,
    /// depending on `BLOCK_SIZE_BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).hashes(4);
    /// ```
    pub fn hashes(self, num_hashes: u64) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        BloomFilter {
            bits: BlockedBitVec::<BLOCK_SIZE_BITS>::new(self.num_blocks).unwrap(),
            num_hashes,
            hasher: self.hasher,
        }
    }

    /// "Consumes" this builder, using the provided `expected_num_items` to return an
    /// empty [`BloomFilter`]. More or less than `expected_num_items` may be inserted into
    /// [`BloomFilter`], but the number of hashes per item is intially calculated
    /// to minimize false positive rate for exactly `expected_num_items`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).expected_items(500);
    /// ```
    pub fn expected_items(self, expected_num_items: usize) -> BloomFilter<BLOCK_SIZE_BITS, S> {
        let num_hashes =
            BloomFilter::<BLOCK_SIZE_BITS>::optimal_hashes(expected_num_items / self.num_blocks);
        self.hashes(num_hashes)
    }

    /// "Consumes" this builder and constructs a [`BloomFilter`] containing
    /// all values in `items`. The number of hashes per item is calculated
    /// based on `items.len()` to minimize false positive rate.
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
    fn api() {
        let _bloom = BloomFilter::builder128(10)
            .hasher(RandomState::default())
            .hashes(4);
    }
}
