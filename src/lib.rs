#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]

use std::hash::{BuildHasher, Hash, Hasher};
mod hasher;
use hasher::DefaultHasher;
mod builder;
pub use builder::Builder;
mod bit_vector;
use bit_vector::BlockedBitVec;

/// Produces a new hash efficiently from two orignal hashes and a seed.
///
/// Modified from <https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf>.
#[inline]
fn seeded_hash_from_hashes(h1: &mut u64, h2: &mut u64, seed: u64) -> u64 {
    *h1 = h1.wrapping_add(*h2).rotate_left(5);
    *h2 = h2.wrapping_add(seed);
    *h1
}

/// A space efficient approximate membership set data structure.
/// False positives from [`contains`](Self::contains) are possible, but false negatives
/// are not, i.e. [`contains`](Self::contains) for all items in the set is guaranteed to return
/// true, while [`contains`](Self::contains) for all items not in the set probably return false.
///
/// [`BloomFilter`] is supported by an underlying bit vector, chunked into
/// [`512`](Self::builder), [`256`](Self::builder256), [`128`](Self::builder128), or [`64`](Self::builder64) bit "blocks", to track item membership.
/// To insert, a number of bits, based on the item's hash, are set in the underlying bit vector.
/// To check membership, a number of bits, based on the item's hash, are checked in the underlying bit vector.
///
/// Once constructed, neither the bloom filter's underlying memory usage nor number of bits per item change.
///
/// # Examples
/// Basic usage:
/// ```
/// use fastbloom::BloomFilter;
///
/// let num_bits = 1024;
///
/// let filter = BloomFilter::builder(num_bits).items(["42", "🦀"]);
/// assert!(filter.contains("42"));
/// assert!(filter.contains("🦀"));
/// ```
/// Use any hasher:
/// ```
/// use fastbloom::BloomFilter;
/// use ahash::RandomState;
///
/// let num_bits = 1024;
///
/// let filter = BloomFilter::builder(num_bits)
///     .hasher(RandomState::default())
///     .items(["42", "🦀"]);
/// ```
#[derive(Debug, Clone)]
pub struct BloomFilter<const BLOCK_SIZE_BITS: usize = 512, S = DefaultHasher> {
    bits: BlockedBitVec<BLOCK_SIZE_BITS>,
    num_hashes: u64,
    hasher: S,
}

impl BloomFilter {
    pub(crate) fn new_builder<const BLOCK_SIZE_BITS: usize>(
        num_bits: usize,
    ) -> Builder<BLOCK_SIZE_BITS> {
        assert!(num_bits > 0);
        let num_blocks = num_bits.div_ceil(BLOCK_SIZE_BITS);
        Builder::<BLOCK_SIZE_BITS> {
            num_blocks,
            hasher: Default::default(),
        }
    }

    /// Creates a new instance of [`Builder`] to construct a [`BloomFilter`]
    /// with `num_bits` number of bits for tracking item membership.
    ///
    /// The returned [`BloomFilter`] has a block size of 512 bits.
    ///
    /// Use [`builder256`](Self::builder256), [`builder128`](Self::builder128), or [`builder64`](Self::builder64) for more speed
    /// but slightly higher false positive rates.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).hashes(4);
    /// ```
    pub fn builder(num_bits: usize) -> Builder<512> {
        Self::builder512(num_bits)
    }

    /// Creates a new instance of [`Builder`] to construct a [`BloomFilter`]
    /// with `num_bits` number of bits for tracking item membership.
    ///
    /// The returned [`BloomFilter`] has a block size of 512 bits.
    ///
    /// Use [`builder256`](Self::builder256), [`builder128`](Self::builder128), or [`builder64`](Self::builder64) for more speed
    /// but slightly higher false positive rates.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder512(1024).hashes(4);
    /// ```
    pub fn builder512(num_bits: usize) -> Builder<512> {
        Self::new_builder::<512>(num_bits)
    }

    /// Creates a new instance of [`Builder`] to construct a [`BloomFilter`]
    /// with `num_bits` number of bits for tracking item membership.
    ///
    /// The returned [`BloomFilter`] has a block size of 256 bits.
    ///
    /// [`Builder<256>`] is faster but less accurate than [`Builder<512>`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder256(1024).hashes(4);
    /// ```
    pub fn builder256(num_bits: usize) -> Builder<256> {
        Self::new_builder::<256>(num_bits)
    }

    /// Creates a new instance of [`Builder`] to construct a [`BloomFilter`]
    /// with `num_bits` number of bits for tracking item membership.
    ///
    /// The returned [`BloomFilter`] has a block size of 128 bits.
    ///
    /// [`Builder<128>`] is faster but less accurate than [`Builder<256>`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder128(1024).hashes(8);
    /// ```
    pub fn builder128(num_bits: usize) -> Builder<128> {
        Self::new_builder::<128>(num_bits)
    }

    /// Creates a new instance of [`Builder`] to construct a [`BloomFilter`]
    /// with `num_bits` number of bits for tracking item membership.
    ///
    /// The returned [`BloomFilter`] has a block size of 64 bits.
    ///
    /// [`Builder<64>`] is faster but less accurate than [`Builder<128>`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder64(1024).hashes(8);
    /// ```
    pub fn builder64(num_bits: usize) -> Builder<64> {
        Self::new_builder::<64>(num_bits)
    }
}

impl<const BLOCK_SIZE_BITS: usize, S: BuildHasher> BloomFilter<BLOCK_SIZE_BITS, S> {
    const BIT_INDEX_MASK_LEN: u32 = u32::ilog2(BLOCK_SIZE_BITS as u32);

    /// Used to grab the last N bits from a hash.
    const BIT_INDEX_MASK: u64 = (BLOCK_SIZE_BITS - 1) as u64;

    /// Number of bit indexes that can be derived by one hash.
    ///
    /// Many bit indexes are derived by taking BIT_INDEX_MASK_LEN size slices of the original hash.
    /// Slicing the hash in this way is way more performant than `seeded_hash_from_hashes`.
    ///
    /// From experiments, powers of 2 coordinates from the hash provides the best performance
    /// for `contains` for existing and non-existing values.
    const NUM_COORDS_PER_HASH: u32 = 2u32.pow(u32::ilog2(64 / Self::BIT_INDEX_MASK_LEN));

    #[inline]
    fn floor_round(x: f64) -> u64 {
        let floored = x.floor() as u64;
        let thresh = Self::NUM_COORDS_PER_HASH as u64;
        if floored < thresh {
            thresh
        } else {
            floored - (floored % thresh)
        }
    }

    /// The optimal number of hashes to perform for an item given the expected number of items to be contained in one block.
    /// Proof under "False Positives Analysis": <https://brilliant.org/wiki/bloom-filter/>
    #[inline]
    fn optimal_hashes(items_per_block: usize) -> u64 {
        let block_size = BLOCK_SIZE_BITS as f64;
        let items_per_block = std::cmp::max(items_per_block, 1) as f64;
        let num_hashes = block_size / items_per_block * f64::ln(2.0f64);
        Self::floor_round(num_hashes)
    }

    /// Returns a the block index for an item's hash.
    /// The block index must be in the range `0..self.bits.num_blocks()`.
    /// This implementation is a more performant alternative to `hash % self.bits.num_blocks()`:
    /// <https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/>
    #[inline]
    fn block_index(&self, hash: u64) -> usize {
        (((hash >> 32) as usize * self.bits.num_blocks()) >> 32) as usize
    }

    /// Return the bit indexes within a block for an item's two orginal hashes.
    ///
    /// First, a seeded hash is derived from two orginal hashes, `hash1` and `hash2`.
    /// Second, `Self::NUM_COORDS_PER_HASH` bit indexes are returned, each `Self::BIT_INDEX_MASK_LEN`
    /// consecutive sections of the hash bits.
    #[inline]
    fn bit_indexes(hash1: &mut u64, hash2: &mut u64, seed: u64) -> impl Iterator<Item = usize> {
        let h = seeded_hash_from_hashes(hash1, hash2, seed);
        (0..Self::NUM_COORDS_PER_HASH).map(move |j| {
            let mut bit_index = h.wrapping_shr(j * Self::BIT_INDEX_MASK_LEN); // remove right bits from previous bit index (j - 1)
            bit_index &= Self::BIT_INDEX_MASK; // remove left bits to keep bit index in range of a block's bit size
            bit_index as usize
        })
    }

    /// Returns all seeds that should be used by the hasher
    #[inline]
    fn hash_seeds(size: u64) -> impl Iterator<Item = u64> {
        (0..size).step_by(Self::NUM_COORDS_PER_HASH as usize)
    }

    /// Adds a value to the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::builder(1024).hashes(4);
    /// bloom.insert(&2);
    /// assert!(bloom.contains(&2));
    /// ```
    #[inline]
    pub fn insert(&mut self, val: &(impl Hash + ?Sized)) {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block = &mut self.bits.get_block_mut(self.block_index(h1));
        for i in Self::hash_seeds(self.num_hashes) {
            BlockedBitVec::<BLOCK_SIZE_BITS>::set_all_for_block(
                block,
                Self::bit_indexes(&mut h1, &mut h2, i),
            );
        }
    }

    /// Returns `false` if the bloom filter definitely does not contain a value.
    /// Returns `true` if the bloom filter may contain a value, with a degree of certainty.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastbloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::builder(1024).items([1, 2, 3]);
    /// assert!(bloom.contains(&1));
    /// ```
    #[inline]
    pub fn contains(&self, val: &(impl Hash + ?Sized)) -> bool {
        let [mut h1, mut h2] = self.get_orginal_hashes(val);
        let block = &self.bits.get_block(self.block_index(h1));
        Self::hash_seeds(self.num_hashes).into_iter().all(|i| {
            BlockedBitVec::<BLOCK_SIZE_BITS>::check_all_for_block(
                block,
                Self::bit_indexes(&mut h1, &mut h2, i),
            )
        })
    }

    /// Returns the effective number of hashes per item. In other words,
    /// the number of bits derived per item.
    ///
    /// For performance reasons, the number of bits is rounded to down to a power of 2, depending on `BLOCK_SIZE_BITS`.
    #[inline]
    pub fn num_hashes(&self) -> u64 {
        self.num_hashes
    }

    /// The first two hashes of the value, h1 and h2.
    ///
    /// Subsequent hashes, h, are efficiently derived from these two using `seeded_hash_from_hashes`.
    ///
    /// This strategy is adapted from <https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf>,
    /// in which a keyed hash function is used to generate two real hashes, h1 and h2, which are then used to produce
    /// many more "fake hahes" h, using h = h1 + i * h2.
    ///
    /// However, here we only use 1 real hash, for performance, and derive h1 and h2:
    /// First, we'll think of the real 64 bit real hash as two seperate 32 bit hashes, h1 and h2.
    ///     - Using h = h1 + i * h2 generates entropy in at least the lower 32 bits
    /// Second, for more entropy in the upper 32 bits, we'll populate the upper 32 bits for both h1 and h2:
    /// For h1, we'll use the orginal upper bits 32 of the real hash.
    ///     - h1 is the same as the real hash
    /// For h2 we'll use lower 32 bits of h, and multiply by a large prime
    ///     - h2 is basically a "weak hash" of h1
    #[inline]
    fn get_orginal_hashes(&self, val: &(impl Hash + ?Sized)) -> [u64; 2] {
        let mut state = self.hasher.build_hasher();
        val.hash(&mut state);
        let h1 = state.finish();
        let h2 = h1.wrapping_shr(32).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95); // 0xffff_ffff_ffff_ffff / 0x517c_c1b7_2722_0a95 = π
        [h1, h2]
    }
}

impl<T, const BLOCK_SIZE_BITS: usize, S: BuildHasher> Extend<T> for BloomFilter<BLOCK_SIZE_BITS, S>
where
    T: Hash,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for val in iter {
            self.insert(&val);
        }
    }
}

impl PartialEq for BloomFilter {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits && self.num_hashes == other.num_hashes
    }
}
impl Eq for BloomFilter {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::HashSet;
    use std::iter::repeat;

    fn random_strings(num: usize, min_repeat: u32, max_repeat: u32, seed: u64) -> Vec<String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
        (&mut rng)
            .sample_iter(&gen)
            .filter(|s: &String| s.len() >= min_repeat as usize)
            .take(num)
            .collect()
    }

    #[test]
    fn random_inserts_always_contained() {
        fn random_inserts_always_contained_<const N: usize>() {
            for mag in 1..6 {
                let size = 10usize.pow(mag);
                for bloom_size_mag in 6..10 {
                    let num_blocks_bytes = 1 << bloom_size_mag;
                    let sample_vals = random_strings(size, 16, 32, 52323);
                    let num_bits = num_blocks_bytes * 8;
                    let filter = BloomFilter::new_builder::<N>(num_bits).items(sample_vals.iter());
                    for x in &sample_vals {
                        assert!(filter.contains(x));
                    }
                }
            }
        }
        random_inserts_always_contained_::<512>();
        random_inserts_always_contained_::<256>();
        random_inserts_always_contained_::<128>();
        random_inserts_always_contained_::<64>();
    }

    #[test]
    fn seeded_is_same() {
        let mag = 3;
        let size = 10usize.pow(mag);
        let bloom_size_bytes = 1 << 10;
        let sample_vals = random_strings(size, 16, 32, 53226);
        let block_size = bloom_size_bytes * 8;
        for x in 0u8..4 {
            let seed = [x; 16];
            let filter1 = BloomFilter::builder(block_size)
                .seed(&seed)
                .items(sample_vals.iter());
            let filter2 = BloomFilter::builder(block_size)
                .seed(&seed)
                .items(sample_vals.iter());
            assert_eq!(filter1, filter2);
        }
    }

    fn false_pos_rate<const N: usize>(filter: &BloomFilter<N>, control: &HashSet<String>) -> f64 {
        let sample_anti_vals = random_strings(1000, 16, 32, 11);
        let mut total = 0;
        let mut false_positives = 0;
        for x in &sample_anti_vals {
            if !control.contains(x) {
                total += 1;
                if filter.contains(x) {
                    false_positives += 1;
                }
            }
        }
        (false_positives as f64) / (total as f64)
    }

    #[test]
    fn false_pos_decrease_with_size() {
        for mag in 1..5 {
            let size = 10usize.pow(mag);
            let mut prev_fp = 1.0;
            let mut prev_prev_fp = 1.0;
            for bloom_size_mag in 6..18 {
                let bloom_size_bytes = 1 << bloom_size_mag;
                let num_bits = bloom_size_bytes * 8;
                let sample_vals = random_strings(size, 16, 32, 5234);
                let filter = BloomFilter::builder512(num_bits)
                    .seed(&[1u8; 16])
                    .items(sample_vals.iter());
                let control: HashSet<String> = sample_vals.into_iter().collect();

                let fp = false_pos_rate(&filter, &control);

                println!(
                    "{:?}, {:?}, {:.6}, {:?}",
                    size,
                    bloom_size_bytes,
                    fp,
                    filter.num_hashes(),
                );
                assert!(fp <= prev_fp || prev_fp <= prev_prev_fp); // allows 1 data point to be higher
                prev_prev_fp = prev_fp;
                prev_fp = fp;
            }
        }
    }

    #[test]
    fn test_floor_round() {
        fn assert_floor_round<const N: usize>() {
            let hashes = BloomFilter::<N>::NUM_COORDS_PER_HASH;
            for i in 0..hashes {
                assert_eq!(hashes as u64, BloomFilter::<N>::floor_round(i as f64));
            }
            for i in (hashes as u64..100).step_by(hashes as usize) {
                for j in 0..(hashes as u64) {
                    let x = (i + j) as f64;
                    assert_eq!(i, BloomFilter::<N>::floor_round(x));
                    assert_eq!(i, BloomFilter::<N>::floor_round(x + 0.9999));
                    assert_eq!(i, BloomFilter::<N>::floor_round(x + 0.0001));
                }
            }
        }
        assert_floor_round::<512>();
        assert_floor_round::<256>();
        assert_floor_round::<128>();
        assert_floor_round::<64>();
    }

    fn assert_even_distribution(distr: &[u64], err: f64) {
        assert!(err > 0.0 && err < 1.0);
        let expected: i64 = (distr.iter().sum::<u64>() / (distr.len() as u64)) as i64;
        let thresh = (expected as f64 * err) as i64;
        for x in distr {
            let diff = (*x as i64 - expected).abs();
            assert!(diff <= thresh, "{x:?} deviates from {expected:?}");
        }
    }

    #[test]
    fn block_hash_distribution() {
        fn block_hash_distribution_<const N: usize>(filter: BloomFilter<N>) {
            let mut rng = StdRng::seed_from_u64(1);
            let iterations = 1000000;
            let mut buckets = vec![0; filter.bits.num_blocks()];
            for _ in 0..iterations {
                let h1 = (&mut rng).gen_range(0..u64::MAX);
                buckets[filter.block_index(h1)] += 1;
            }
            assert_even_distribution(&buckets, 0.05);
        }
        let num_bits = 10000;
        let seed = [0; 16];
        block_hash_distribution_::<512>(BloomFilter::builder512(num_bits).seed(&seed).hashes(1));
        block_hash_distribution_::<256>(BloomFilter::builder256(num_bits).seed(&seed).hashes(1));
        block_hash_distribution_::<128>(BloomFilter::builder128(num_bits).seed(&seed).hashes(1));
        block_hash_distribution_::<64>(BloomFilter::builder64(num_bits).seed(&seed).hashes(1));
    }

    #[test]
    fn test_seeded_hash_from_hashes() {
        let mut rng = StdRng::seed_from_u64(524323);
        let mut h1 = (&mut rng).gen_range(0..u64::MAX);
        let mut h2 = (&mut rng).gen_range(0..u64::MAX);
        let size = 1000;
        let mut seeded_hash_counts = vec![0; size];
        let iterations = 10000000;
        for i in 0..iterations {
            let hi = seeded_hash_from_hashes(&mut h1, &mut h2, i);
            seeded_hash_counts[(hi as usize) % size] += 1;
        }
        assert_even_distribution(&seeded_hash_counts, 0.05);
    }

    #[test]
    fn index_hash_distribution() {
        fn index_hash_distribution_<const N: usize>(filter: BloomFilter<N>, thresh_pct: f64) {
            let [mut h1, mut h2] = filter.get_orginal_hashes("qwerty");
            let mut counts = vec![0; N];
            let iterations = 10000 * N as u64;
            for i in 0..iterations {
                for bit_index in BloomFilter::<N>::bit_indexes(&mut h1, &mut h2, i) {
                    let index = bit_index as usize % N;
                    counts[index] += 1;
                }
            }
            assert_even_distribution(&counts, thresh_pct);
        }
        let seed = [0; 16];
        index_hash_distribution_::<512>(BloomFilter::builder512(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<256>(BloomFilter::builder256(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<128>(BloomFilter::builder128(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<64>(BloomFilter::builder64(1).seed(&seed).hashes(1), 0.05);
    }

    #[test]
    fn test_hash_integration() {
        fn test_hash_integration_<const N: usize>(clone_me: BloomFilter<N>, thresh_pct: f64) {
            let mut rng = StdRng::seed_from_u64(524323);
            let num = 2000 * N;
            for distr in [
                repeat(())
                    .map(|_| (&mut rng).gen_range(0..usize::MAX))
                    .take(num)
                    .collect::<Vec<usize>>(),
                (0..num).into_iter().map(|x| x * 2).collect::<Vec<usize>>(),
                (0..num).into_iter().map(|x| x * 3).collect::<Vec<usize>>(),
                (0..num)
                    .into_iter()
                    .map(|x| x * clone_me.num_hashes() as usize)
                    .collect::<Vec<usize>>(),
                (0..num)
                    .into_iter()
                    .map(|x| x * clone_me.bits.num_blocks() as usize)
                    .collect::<Vec<usize>>(),
                (0..num).into_iter().map(|x| x * N).collect::<Vec<usize>>(),
            ] {
                let filter = clone_me.clone();
                let mut counts = vec![0; N * filter.bits.num_blocks()];

                for val in distr.iter() {
                    let [mut h1, mut h2] = filter.get_orginal_hashes(val);
                    let block_index = filter.block_index(h1);
                    for i in BloomFilter::<N>::hash_seeds(filter.num_hashes) {
                        for j in BloomFilter::<N>::bit_indexes(&mut h1, &mut h2, i) {
                            let global = block_index * N + j;
                            counts[global] += 1;
                        }
                    }
                }
                assert_even_distribution(&counts, thresh_pct);
            }
        }
        let seed = [0; 16];
        test_hash_integration_::<512>(BloomFilter::builder512(1).seed(&seed).hashes(1), 0.05);
        test_hash_integration_::<256>(BloomFilter::builder256(1).seed(&seed).hashes(1), 0.05);
        test_hash_integration_::<128>(BloomFilter::builder128(1).seed(&seed).hashes(1), 0.05);
        test_hash_integration_::<64>(BloomFilter::builder64(1).seed(&seed).hashes(1), 0.05);
    }

    #[test]
    fn test_debug() {
        let filter = BloomFilter::builder64(1).hashes(1);
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_clone() {
        let filter = BloomFilter::builder(4).hashes(4);
        assert_eq!(filter, filter.clone());
    }
}
