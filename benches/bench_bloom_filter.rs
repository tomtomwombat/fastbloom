use ahash;
use bloom::ASMS;
use bloomfilter::Bloom;
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration,
};
use fastbloom::BloomFilter;
use fastbloom_rs;
use fastbloom_rs::Hashes;
use fastbloom_rs::Membership;
use probabilistic_collections::bloom::BloomFilter as ProbBloomFilter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::iter::repeat;

fn run_bench_for<T: Container<String>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_items: usize,
    seed: u64,
) {
    let sample_seed = 1234;
    let num_bytes = 262144;
    let num_bits = num_bytes * 8;
    let bloom: T = Container::new(num_bits, random_strings(num_items, 6, 12, seed));
    let sample_vals = random_strings(1000, 6, 12, sample_seed);
    group.bench_with_input(
        BenchmarkId::new(T::name(), num_items),
        &num_items,
        |b, _| {
            b.iter(|| {
                for val in sample_vals.iter() {
                    black_box(bloom.check(val));
                }
            })
        },
    );
}
fn bench(c: &mut Criterion) {
    // list_fp::<fastbloom::BloomFilter<512>>();
    let sample_seed = 1234;
    let num_bytes = 262144;
    for seed in [1234, 9876] {
        let item_type = if seed == sample_seed {
            "Member"
        } else {
            "Non-Member"
        };
        let mut group = c.benchmark_group(&format!(
            "{} Check Speed vs Number of Items in BloomFilter ({}Kb Allocated, SipHash)",
            item_type,
            num_bytes / 1000
        ));
        group.plot_config(PlotConfiguration::default());
        for num_items in [
            5000, 7500, 10_000, 15_000, 20_000, 25_000, 50_000, 75_000, 100_000,
        ] {
            run_bench_for::<fastbloom::BloomFilter<512>>(&mut group, num_items, seed);
            run_bench_for::<bloom::BloomFilter>(&mut group, num_items, seed);
            run_bench_for::<Bloom<String>>(&mut group, num_items, seed);
            run_bench_for::<ProbBloomFilter<String>>(&mut group, num_items, seed);
        }
        group.finish();
        let mut g2 = c.benchmark_group(&format!(
            "{} Check Speed vs Number of Items in BloomFilter ({}Kb Allocated)",
            item_type,
            num_bytes / 1000
        ));
        g2.plot_config(PlotConfiguration::default());
        for num_items in [
            // 5000, 7500, 10_000, 15_000, 20_000, 25_000, 50_000, 75_000, 100_000,
        ] {
            run_bench_for::<fastbloom::BloomFilter<512, ahash::RandomState>>(
                &mut g2, num_items, seed,
            );
            run_bench_for::<fastbloom_rs::BloomFilter>(&mut g2, num_items, seed);
        }
        g2.finish();
    }
}

#[allow(dead_code)]
fn false_pos_rate_with_vals<X: Hash + Eq + PartialEq>(
    filter: &impl Container<X>,
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

#[allow(dead_code)]
fn list_fp<T: Container<u64>>() {
    let thresh = 0.1;
    let amount = 100_000;
    for bloom_size_bytes in [65536, 262144] {
        let mut fp = 0.0;
        for num_items_base in (8..23).map(|x| 1 << x) {
            let all_num_items: Vec<usize> = if fp > 0.0 && fp < thresh {
                let step = num_items_base >> 8;
                ((num_items_base >> 1 + step)..(num_items_base << 1))
                    .step_by(step)
                    .collect()
            } else {
                std::iter::once(num_items_base).collect()
            };
            for num_items in all_num_items {
                if num_items == 0 {
                    continue;
                }
                let sample_vals = random_numbers(num_items, 42);

                let num_bits = bloom_size_bytes * 8;
                let filter = T::new(num_bits, sample_vals.clone().into_iter()); //BloomFilter::builder512(num_bits).items(sample_vals.iter());
                let control: HashSet<u64> = sample_vals.into_iter().collect();
                fp = false_pos_rate_with_vals(&filter, &control, random_numbers(amount, 43));
                print!("{:?}, ", num_items);
                print!("{:?}, ", bloom_size_bytes);
                print!("{:?}, ", filter.num_hashes());
                print!("{:.8}", fp);
                println!("");
                if fp > thresh {
                    break;
                }
            }
            if fp > thresh {
                break;
            }
        }
    }
}
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench
);
criterion_main!(benches);

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

trait Container<X: Hash> {
    fn check(&self, s: &X) -> bool;
    fn num_hashes(&self) -> usize;
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = X>>>(
        num_bits: usize,
        items: I,
    ) -> Self;
    fn name() -> &'static str;
}

impl<X: Hash, H: BuildHasher + Default> Container<X> for BloomFilter<512, H> {
    #[inline]
    fn check(&self, s: &X) -> bool {
        self.contains(s)
    }
    fn num_hashes(&self) -> usize {
        self.num_hashes() as usize
    }
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = X>>>(
        num_bits: usize,
        items: I,
    ) -> Self {
        BloomFilter::builder(num_bits)
            .hasher(H::default())
            .items(items)
    }
    fn name() -> &'static str {
        "fastbloom"
    }
}

impl<X: Hash> Container<X> for Bloom<X> {
    #[inline]
    fn check(&self, s: &X) -> bool {
        self.check(s)
    }
    fn num_hashes(&self) -> usize {
        self.number_of_hash_functions() as usize
    }
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = X>>>(
        num_bits: usize,
        items: I,
    ) -> Self {
        let items = items.into_iter();
        let mut filter = Bloom::<X>::new(num_bits / 8, items.len());
        for x in items {
            filter.set(&x);
        }
        filter
    }
    fn name() -> &'static str {
        "bloomfilter"
    }
}

impl<X: Hash> Container<X> for bloom::BloomFilter {
    #[inline]
    fn check(&self, s: &X) -> bool {
        self.contains(s)
    }
    fn num_hashes(&self) -> usize {
        self.num_hashes() as usize
    }
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = X>>>(
        num_bits: usize,
        items: I,
    ) -> Self {
        let items = items.into_iter();
        let hashes = bloom::bloom::optimal_num_hashes(num_bits, items.len() as u32);
        let mut filter = bloom::BloomFilter::with_size(num_bits, hashes);
        for x in items {
            filter.insert(&x);
        }
        filter
    }
    fn name() -> &'static str {
        "bloom"
    }
}

impl<X: Hash> Container<X> for ProbBloomFilter<X> {
    #[inline]
    fn check(&self, s: &X) -> bool {
        self.contains(s)
    }
    fn num_hashes(&self) -> usize {
        self.hasher_count() as usize
    }
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = X>>>(
        num_bits: usize,
        items: I,
    ) -> Self {
        let items = items.into_iter();
        let mut filter = ProbBloomFilter::<X>::from_item_count(num_bits, items.len());
        for x in items {
            filter.insert(&x);
        }
        filter
    }
    fn name() -> &'static str {
        "probabilistic-collections"
    }
}

impl Container<u64> for fastbloom_rs::BloomFilter {
    #[inline]
    fn check(&self, s: &u64) -> bool {
        self.contains(&s.to_be_bytes())
    }
    fn num_hashes(&self) -> usize {
        self.hashes() as usize
    }
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = u64>>>(
        num_bits: usize,
        items: I,
    ) -> Self {
        let items = items.into_iter();
        let hashes = bloom::bloom::optimal_num_hashes(num_bits, items.len() as u32);
        let mut filter = fastbloom_rs::FilterBuilder::from_size_and_hashes(num_bits as u64, hashes)
            .build_bloom_filter();
        for x in items {
            filter.add(&x.to_be_bytes());
        }
        filter
    }
    fn name() -> &'static str {
        "fastbloom-rs"
    }
}
impl Container<String> for fastbloom_rs::BloomFilter {
    #[inline]
    fn check(&self, s: &String) -> bool {
        self.contains(&s.as_bytes())
    }
    fn num_hashes(&self) -> usize {
        self.hashes() as usize
    }
    fn new<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = String>>>(
        num_bits: usize,
        items: I,
    ) -> Self {
        let items = items.into_iter();
        let hashes = bloom::bloom::optimal_num_hashes(num_bits, items.len() as u32);
        let mut filter = fastbloom_rs::FilterBuilder::from_size_and_hashes(num_bits as u64, hashes)
            .build_bloom_filter();
        for x in items {
            filter.add(&x.as_bytes());
        }
        filter
    }
    fn name() -> &'static str {
        "fastbloom-rs"
    }
}
