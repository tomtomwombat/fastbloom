use crate::{BloomFilter, BuildHasher, DefaultHasher};
use alloc::vec::Vec;
use core::{f64::consts::LN_2, hash::Hash};

/// A Bloom filter builder with an immutable number of bits.
///
/// This type can be used to construct an instance of [`BloomFilter`] via the builder pattern.
///
/// # Examples
/// ```
/// use fastbloom::BloomFilter;
///
/// let builder = BloomFilter::with_num_bits(1024);
/// let builder = BloomFilter::from_vec(vec![0; 8]);
/// ```
#[derive(Debug, Clone)]
pub struct BuilderWithBits<S = DefaultHasher> {
    pub(crate) data: Vec<u64>,
    pub(crate) hasher: S,
}

impl<S: BuildHasher> PartialEq for BuilderWithBits<S> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl<S: BuildHasher> Eq for BuilderWithBits<S> {}

impl BuilderWithBits {
    /// Sets the seed for this builder. The later constructed [`BloomFilter`]
    /// will use this seed when hashing items.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_num_bits(1024).seed(&1).hashes(4);
    /// ```
    pub fn seed(mut self, seed: &u128) -> Self {
        self.hasher = DefaultHasher::seeded(&seed.to_be_bytes());
        self
    }
}

impl<S: BuildHasher> BuilderWithBits<S> {
    /// Sets the hasher for this builder. The later constructed [`BloomFilter`] will use
    /// this hasher when inserting and checking items.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    /// use ahash::RandomState;
    ///
    /// let bloom = BloomFilter::with_num_bits(1024).hasher(RandomState::default()).hashes(4);
    /// ```
    pub fn hasher<H: BuildHasher>(self, hasher: H) -> BuilderWithBits<H> {
        BuilderWithBits::<H> {
            data: self.data,
            hasher,
        }
    }

    /// "Consumes" this builder, using the provided `num_hashes` to return an
    /// empty [`BloomFilter`].
    ///
    /// # Examples
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_num_bits(1024).hashes(4);
    /// ```
    pub fn hashes(self, num_hashes: u32) -> BloomFilter<S> {
        BloomFilter {
            bits: self.data.into_iter().collect(),
            num_hashes,
            hasher: self.hasher,
        }
    }

    /// "Consumes" this builder, using the provided `expected_num_items` to return an
    /// empty [`BloomFilter`]. The number of hashes is optimized based on `expected_num_items`
    /// to maximize Bloom filter accuracy (minimize false positives chance on [`BloomFilter::contains`]).
    /// More or less than `expected_num_items` may be inserted into [`BloomFilter`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_num_bits(1024).expected_items(500);
    /// ```
    pub fn expected_items(self, expected_num_items: usize) -> BloomFilter<S> {
        let num_hashes = Self::optimal_hashes_f(self.data.len(), expected_num_items);
        self.hashes(num_hashes as u32)
    }

    /// The optimal number of hashes to perform for an item given the expected number of items in the bloom filter.
    /// Proof under "False Positives Analysis": <https://brilliant.org/wiki/bloom-filter/>.
    #[inline]
    fn optimal_hashes_f(len: usize, expected_num_items: usize) -> f64 {
        let load = expected_num_items as f64 / len as f64;
        let x = 64.0f64 / load;
        let hashes_per_u64 = x * f64::ln(2.0f64);

        let max_hashes = Self::hashes_for_bits(32);
        if hashes_per_u64 > max_hashes {
            max_hashes
        } else if hashes_per_u64 < 1.0 {
            1.0
        } else {
            hashes_per_u64
        }
    }

    #[inline]
    pub(crate) fn hashes_for_bits(target_bits_per_u64_per_item: u64) -> f64 {
        f64::ln(-(((target_bits_per_u64_per_item as f64) / 64.0f64) - 1.0f64))
            / f64::ln(63.0f64 / 64.0f64)
    }

    /// "Consumes" this builder and constructs a [`BloomFilter`] containing
    /// all values in `items`. Like [`BuilderWithBits::expected_items`], the number of hashes per item
    /// is optimized based on `items.len()` to maximize Bloom filter accuracy
    /// (minimize false positives chance on [`BloomFilter::contains`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_num_bits(1024).items([1, 2, 3]);
    /// ```
    pub fn items<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl Hash>>>(
        self,
        items: I,
    ) -> BloomFilter<S> {
        let into_iter = items.into_iter();
        let mut filter = self.expected_items(into_iter.len());
        filter.extend(into_iter);
        filter
    }
}

fn optimal_size(items_count: f64, fp_p: f64) -> usize {
    let log2_2 = LN_2 * LN_2;
    let result = 8 * ((items_count) * f64::ln(fp_p) / (-8.0 * log2_2)).ceil() as usize;
    core::cmp::max(result, 512)
}

/// A Bloom filter builder with an immutable false positive rate.
///
/// This type can be used to construct an instance of [`BloomFilter`] via the builder pattern.
///
/// # Examples
///
/// ```
/// use fastbloom::BloomFilter;
///
/// let builder = BloomFilter::with_false_pos(0.01);
/// ```
#[derive(Debug, Clone)]
pub struct BuilderWithFalsePositiveRate<S = DefaultHasher> {
    pub(crate) desired_fp_rate: f64,
    pub(crate) hasher: S,
}

impl<S: BuildHasher> PartialEq for BuilderWithFalsePositiveRate<S> {
    fn eq(&self, other: &Self) -> bool {
        self.desired_fp_rate == other.desired_fp_rate
    }
}
impl<S: BuildHasher> Eq for BuilderWithFalsePositiveRate<S> {}

impl BuilderWithFalsePositiveRate {
    /// Sets the seed for this builder. The later constructed [`BloomFilter`]
    /// will use this seed when hashing items.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_false_pos(0.001).seed(&1).expected_items(100);
    /// ```
    pub fn seed(mut self, seed: &u128) -> Self {
        self.hasher = DefaultHasher::seeded(&seed.to_be_bytes());
        self
    }
}

impl<S: BuildHasher> BuilderWithFalsePositiveRate<S> {
    /// Sets the hasher for this builder. The later constructed [`BloomFilter`] will use
    /// this hasher when inserting and checking items.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    /// use ahash::RandomState;
    ///
    /// let bloom = BloomFilter::with_false_pos(0.001).hasher(RandomState::default()).expected_items(100);
    /// ```
    pub fn hasher<H: BuildHasher>(self, hasher: H) -> BuilderWithFalsePositiveRate<H> {
        BuilderWithFalsePositiveRate::<H> {
            desired_fp_rate: self.desired_fp_rate,
            hasher,
        }
    }

    /// "Consumes" this builder, using the provided `expected_num_items` to return an
    /// empty [`BloomFilter`]. The number of hashes and underlying memory is optimized based on `expected_num_items`
    /// to meet the desired false positive rate.
    /// More or less than `expected_num_items` may be inserted into [`BloomFilter`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_false_pos(0.001).expected_items(500);
    /// ```
    pub fn expected_items(self, expected_num_items: usize) -> BloomFilter<S> {
        let num_bits = optimal_size(expected_num_items as f64, self.desired_fp_rate);
        BloomFilter::new_builder(num_bits)
            .hasher(self.hasher)
            .expected_items(expected_num_items)
    }

    /// "Consumes" this builder and constructs a [`BloomFilter`] containing
    /// all values in `items`. Like [`BuilderWithFalsePositiveRate::expected_items`], the number of hashes per item
    /// and underlying memory is optimized based on `items.len()` to meet the desired false positive rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::with_false_pos(0.001).items([1, 2, 3]);
    /// ```
    pub fn items<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl Hash>>>(
        self,
        items: I,
    ) -> BloomFilter<S> {
        let into_iter = items.into_iter();
        let mut filter = self.expected_items(into_iter.len());
        filter.extend(into_iter);
        filter
    }
}

#[cfg(test)]
mod for_accuracy_tests {
    use crate::BloomFilter;

    #[test]
    fn data_size() {
        let size_bits = 512 * 1000;
        let bloom = BloomFilter::with_num_bits(size_bits).hashes(4);
        assert_eq!(bloom.num_bits(), size_bits);
    }

    #[test]
    fn specified_hashes() {
        for num_hashes in 1..1000 {
            assert_eq!(
                num_hashes,
                BloomFilter::with_num_bits(1)
                    .hashes(num_hashes)
                    .num_hashes()
            );
            assert_eq!(
                num_hashes,
                BloomFilter::with_num_bits(1)
                    .hashes(num_hashes)
                    .num_hashes()
            );
        }
    }
}

#[cfg(test)]
mod for_size_tests {
    use crate::BloomFilter;

    #[test]
    fn test_size() {
        let _: BloomFilter = BloomFilter::new_with_false_pos(0.0001).expected_items(10000);
    }
}
