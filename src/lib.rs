#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]

use std::hash::{BuildHasher, Hash, Hasher};
mod hasher;
pub use hasher::DefaultHasher;
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
/// For h1, we'll use the original upper bits 32 of the real hash.
///     - h1 is the same as the real hash
/// For h2 we'll use lower 32 bits of h, and multiply by a large prime
///     - h2 is basically a "weak hash" of h1
#[inline]
pub(crate) fn get_orginal_hashes(
    hasher: &impl BuildHasher,
    val: &(impl Hash + ?Sized),
) -> [u64; 2] {
    let mut state = hasher.build_hasher();
    val.hash(&mut state);
    let h1 = state.finish();
    let h2 = h1.wrapping_shr(32).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95); // 0xffff_ffff_ffff_ffff / 0x517c_c1b7_2722_0a95 = π
    [h1, h2]
}

/// Returns a the block index for an item's hash.
/// The block index must be in the range `0..self.bits.num_blocks()`.
/// This implementation is a more performant alternative to `hash % self.bits.num_blocks()`:
/// <https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/>
#[inline]
pub(crate) fn block_index(num_blocks: usize, hash: u64) -> usize {
    (((hash >> 32) as usize * num_blocks) >> 32) as usize
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
    target_hashes: u64,
    num_hashes: u64,
    num_rounds: Option<u64>,
    hasher: S,
}

impl BloomFilter {
    fn new_builder<const BLOCK_SIZE_BITS: usize>(num_bits: usize) -> Builder<BLOCK_SIZE_BITS> {
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
    /// `2u32.pow(u32::ilog2(64 / Self::BIT_INDEX_MASK_LEN))` for less accuracy and more performance.
    const NUM_COORDS_PER_HASH: u32 = 1; // ;

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
    fn optimal_hashes_f(items_per_block: usize) -> f64 {
        let m = BLOCK_SIZE_BITS as f64;
        let n = std::cmp::max(items_per_block, 1) as f64;
        let num_hashes = m / n * f64::ln(2.0f64);
        num_hashes
    }

    /// Returns a `u64` with the approx 64 * 1/2^num_rounds set
    ///
    /// Using the raw hash is very fast way of setting ~32 random bits
    /// Using the intersection of two raw hashes is very fast way of setting ~16 random bits.
    /// etc
    ///
    /// If the bloom filter has a higher number of hashes to be performed per item,
    /// we can use "signatures" to quickly get many index bits, and use traditional
    /// index setting for the remainder of hashes.
    #[inline]
    fn signature(h1: &mut u64, h2: &mut u64, num_round: u64) -> u64 {
        let mut data = seeded_hash_from_hashes(h1, h2, 0);

        // Bit thinning round. Each round approx halves the number of 1 bits in `data`.
        for j in 1..num_round {
            let h = seeded_hash_from_hashes(h1, h2, j);
            data &= h;
        }
        data
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
        let [mut h1, mut h2] = get_orginal_hashes(&self.hasher, val);
        let block_index = block_index(self.num_blocks(), h1);
        if let Some(num_rounds) = self.num_rounds {
            for i in 0..self.bits.get_block(block_index).len() {
                let data = Self::signature(&mut h1, &mut h2, num_rounds);
                let block = &mut self.bits.get_block_mut(block_index);
                block[i] |= data;
            }
        }
        let block = &mut self.bits.get_block_mut(block_index);
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
        let [mut h1, mut h2] = get_orginal_hashes(&self.hasher, val);
        let block_index = block_index(self.num_blocks(), h1);
        let block = &self.bits.get_block(block_index);
        (if let Some(num_rounds) = self.num_rounds {
            (0..block.len()).all(|i| {
                let data = Self::signature(&mut h1, &mut h2, num_rounds);
                (block[i] & data) == data
            })
        } else {
            true
        }) && Self::hash_seeds(self.num_hashes).into_iter().all(|i| {
            BlockedBitVec::<BLOCK_SIZE_BITS>::check_all_for_block(
                block,
                Self::bit_indexes(&mut h1, &mut h2, i),
            )
        })
    }

    /// Returns the effective number of hashes per item.
    #[inline]
    pub fn num_hashes(&self) -> u64 {
        self.target_hashes
    }

    /// Returns the total number of in memory bits supporting the bloom filter.
    pub fn num_bits(&self) -> usize {
        self.num_blocks() * BLOCK_SIZE_BITS
    }

    /// Returns the total number of in memory blocks supporting the bloom filter.
    /// Each block is `BLOCK_SIZE_BITS` bits.
    pub fn num_blocks(&self) -> usize {
        self.bits.num_blocks()
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
    use std::{collections::HashSet, iter::repeat};

    fn random_strings(num: usize, min_repeat: u32, max_repeat: u32, seed: u64) -> Vec<String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
        (&mut rng)
            .sample_iter(&gen)
            .filter(|s: &String| s.len() >= min_repeat as usize)
            .take(num)
            .collect()
    }

    fn random_numbers(num: usize, seed: u64) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(seed);
        repeat(()).take(num).map(|_| rng.gen()).collect()
    }

    trait Container {
        fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl Hash>>>(
            num_bits: usize,
            items: I,
        ) -> Self;
        fn check<X: Hash>(&self, s: X) -> bool;
        fn num_hashes(&self) -> usize;
        fn block_counts(&self) -> Vec<u64>;
    }
    impl<const N: usize, H: BuildHasher + Default> Container for BloomFilter<N, H> {
        fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl Hash>>>(
            num_bits: usize,
            items: I,
        ) -> Self {
            BloomFilter::new_builder::<N>(num_bits)
                .seed(&42)
                .hasher(H::default())
                .items(items)
        }
        fn check<X: Hash>(&self, s: X) -> bool {
            self.contains(&s)
        }
        fn num_hashes(&self) -> usize {
            self.num_hashes() as usize
        }
        fn block_counts(&self) -> Vec<u64> {
            (0..self.num_blocks())
                .map(|i| {
                    self.bits
                        .get_block(i)
                        .iter()
                        .map(|x| x.count_ones() as u64)
                        .sum()
                })
                .collect()
        }
    }

    #[test]
    fn random_inserts_always_contained() {
        fn random_inserts_always_contained_<T: Container>() {
            for mag in 1..6 {
                let size = 10usize.pow(mag);
                for bloom_size_mag in 6..10 {
                    let num_blocks_bytes = 1 << bloom_size_mag;
                    let sample_vals = random_numbers(size, 42);
                    let num_bits = num_blocks_bytes * 8;
                    let filter: T = Container::new(num_bits, sample_vals.iter());
                    assert!(sample_vals.into_iter().all(|x| filter.check(x)));
                }
            }
        }
        random_inserts_always_contained_::<BloomFilter<512>>();
        // random_inserts_always_contained_::<BloomFilter<256>>();
        // random_inserts_always_contained_::<BloomFilter<128>>();
        // random_inserts_always_contained_::<BloomFilter<64>>();
        // random_inserts_always_contained_::<BloomFilter<512, ahash::RandomState>>();
        // random_inserts_always_contained_::<BloomFilter<256, ahash::RandomState>>();
        // random_inserts_always_contained_::<BloomFilter<128, ahash::RandomState>>();
        // random_inserts_always_contained_::<BloomFilter<64, ahash::RandomState>>();
    }

    #[test]
    fn seeded_is_same() {
        let num_bits = 1 << 13;
        let sample_vals = random_strings(1000, 16, 32, 53226);
        for x in 0u8..10 {
            let seed = x as u128;
            assert_eq!(
                BloomFilter::builder(num_bits)
                    .seed(&seed)
                    .items(sample_vals.iter()),
                BloomFilter::builder(num_bits)
                    .seed(&seed)
                    .items(sample_vals.iter())
            );
            assert!(
                !(BloomFilter::builder(num_bits)
                    .seed(&(seed + 1))
                    .items(sample_vals.iter())
                    == BloomFilter::builder(num_bits)
                        .seed(&seed)
                        .items(sample_vals.iter()))
            );
        }
    }

    fn false_pos_rate_with_vals<X: Hash + Eq + PartialEq>(
        filter: &impl Container,
        control: &HashSet<X>,
        anti_vals: impl IntoIterator<Item = X>,
    ) -> f64 {
        let mut total = 0;
        let mut false_positives = 0;
        for x in anti_vals.into_iter() {
            if !control.contains(&x) {
                total += 1;
                false_positives += filter.check(&x) as usize;
            }
        }
        (false_positives as f64) / (total as f64)
    }

    #[test]
    fn false_pos_decrease_with_size() {
        fn false_pos_decrease_with_size_<T: Container>() {
            for mag in 5..6 {
                let size = 10usize.pow(mag);
                let mut prev_fp = 1.0;
                let mut prev_prev_fp = 1.0;
                for num_bits_mag in 9..22 {
                    let num_bits = 1 << num_bits_mag;
                    let sample_vals = random_numbers(size, 1);
                    let filter: T = Container::new(num_bits, sample_vals.iter());
                    let control: HashSet<u64> = sample_vals.into_iter().collect();
                    let fp = false_pos_rate_with_vals(&filter, &control, random_numbers(1000, 2));

                    println!(
                        "size: {size:}, num_bits: {num_bits:}, {:.6}, {:?}",
                        fp,
                        filter.num_hashes(),
                    );
                    assert!(fp <= prev_fp || prev_fp <= prev_prev_fp || fp < 0.01); // allows 1 data point to be higher
                    prev_prev_fp = prev_fp;
                    prev_fp = fp;
                }
            }
        }
        false_pos_decrease_with_size_::<BloomFilter<512>>();
        false_pos_decrease_with_size_::<BloomFilter<256>>();
        false_pos_decrease_with_size_::<BloomFilter<128>>();
        false_pos_decrease_with_size_::<BloomFilter<64>>();
        false_pos_decrease_with_size_::<BloomFilter<512, ahash::RandomState>>();
        false_pos_decrease_with_size_::<BloomFilter<256, ahash::RandomState>>();
        false_pos_decrease_with_size_::<BloomFilter<128, ahash::RandomState>>();
        false_pos_decrease_with_size_::<BloomFilter<64, ahash::RandomState>>();
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
    fn block_distribution() {
        fn block_distribution_<T: Container>() {
            let filter: T = Container::new(1000, random_numbers(1000, 1));
            assert_even_distribution(&filter.block_counts(), 0.4);
        }
        block_distribution_::<BloomFilter<512>>();
        block_distribution_::<BloomFilter<256>>();
        block_distribution_::<BloomFilter<128>>();
        block_distribution_::<BloomFilter<64>>();
        block_distribution_::<BloomFilter<512, ahash::RandomState>>();
        block_distribution_::<BloomFilter<256, ahash::RandomState>>();
        block_distribution_::<BloomFilter<128, ahash::RandomState>>();
        block_distribution_::<BloomFilter<64, ahash::RandomState>>();
    }
    #[test]
    fn block_hash_distribution() {
        fn block_hash_distribution_<H: BuildHasher + Default>(num_blocks: usize) {
            let mut buckets = vec![0; num_blocks];
            let hasher = H::default();
            for x in random_numbers(num_blocks * 10000, 42) {
                let [h1, _] = get_orginal_hashes(&hasher, &x);
                buckets[block_index(num_blocks, h1)] += 1;
            }
            assert_even_distribution(&buckets, 0.05);
        }
        for size in [2, 7, 10, 100] {
            block_hash_distribution_::<DefaultHasher>(size);
            block_hash_distribution_::<ahash::RandomState>(size);
        }
    }

    #[test]
    fn test_seeded_hash_from_hashes_depth() {
        for size in [1, 10, 100, 1000] {
            let mut rng = StdRng::seed_from_u64(524323);
            let mut h1 = (&mut rng).gen_range(0..u64::MAX);
            let mut h2 = (&mut rng).gen_range(0..u64::MAX);
            let mut seeded_hash_counts = vec![0; size];
            for i in 0..(size * 10_000) {
                let hi = seeded_hash_from_hashes(&mut h1, &mut h2, i as u64);
                seeded_hash_counts[(hi as usize) % size] += 1;
            }
            assert_even_distribution(&seeded_hash_counts, 0.05);
        }
    }

    #[test]
    fn index_hash_distribution() {
        fn index_hash_distribution_<const N: usize>(filter: BloomFilter<N>, thresh_pct: f64) {
            let [mut h1, mut h2] = get_orginal_hashes(&filter.hasher, "qwerty");
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
        let seed = 0;
        index_hash_distribution_::<512>(BloomFilter::builder512(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<256>(BloomFilter::builder256(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<128>(BloomFilter::builder128(1).seed(&seed).hashes(1), 0.05);
        index_hash_distribution_::<64>(BloomFilter::builder64(1).seed(&seed).hashes(1), 0.05);
    }

    #[test]
    fn test_hash_integration() {
        fn test_hash_integration_<const N: usize, H: BuildHasher + Default>(thresh_pct: f64) {
            fn test_with_distr_fn<
                const N: usize,
                H: BuildHasher + Default,
                F: FnMut(usize) -> usize,
            >(
                mut f: F,
                filter: &BloomFilter<N, H>,
                thresh_pct: f64,
            ) {
                let num = 2000 * N;
                let mut counts = vec![0; N * filter.num_blocks()];
                for val in (0..num).map(|i| f(i)) {
                    let [mut h1, mut h2] = get_orginal_hashes(&filter.hasher, &val);
                    let block_index = block_index(filter.num_blocks(), h1);
                    for i in BloomFilter::<N>::hash_seeds(filter.num_hashes()) {
                        for j in BloomFilter::<N>::bit_indexes(&mut h1, &mut h2, i) {
                            let global = block_index * N + j;
                            counts[global] += 1;
                        }
                    }
                }
                assert_even_distribution(&counts, thresh_pct);
            }
            for num_hashes in [1, 4, 8] {
                let clone_me = BloomFilter::new_builder::<N>(4)
                    .hasher(H::default())
                    .hashes(num_hashes);
                let mut rng = StdRng::seed_from_u64(42);
                test_with_distr_fn(
                    |_| (&mut rng).gen_range(0..usize::MAX),
                    &clone_me,
                    thresh_pct,
                );
                test_with_distr_fn(|x| x * 2, &clone_me, thresh_pct);
                test_with_distr_fn(|x| x * 3, &clone_me, thresh_pct);
                test_with_distr_fn(
                    |x| x * clone_me.num_hashes() as usize,
                    &clone_me,
                    thresh_pct,
                );
                test_with_distr_fn(
                    |x| x * clone_me.num_blocks() as usize,
                    &clone_me,
                    thresh_pct,
                );
                test_with_distr_fn(|x| x * N, &clone_me, thresh_pct);
            }
        }
        let pct = 0.1;
        test_hash_integration_::<512, DefaultHasher>(pct);
        test_hash_integration_::<256, DefaultHasher>(pct);
        test_hash_integration_::<128, DefaultHasher>(pct);
        test_hash_integration_::<64, DefaultHasher>(pct);
        test_hash_integration_::<512, ahash::RandomState>(pct);
        test_hash_integration_::<256, ahash::RandomState>(pct);
        test_hash_integration_::<128, ahash::RandomState>(pct);
        test_hash_integration_::<64, ahash::RandomState>(pct);
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
