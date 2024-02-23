use criterion::{black_box, criterion_group, Criterion};
use fastbloom::BloomFilter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

fn random_strings(num: usize, min_repeat: u32, max_repeat: u32, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let gen = rand_regex::Regex::compile(r"[a-zA-Z]+", max_repeat).unwrap();
    (&mut rng)
        .sample_iter(&gen)
        .filter(|s: &String| s.len() >= min_repeat as usize)
        .take(num)
        .collect()
}

trait Container {
    fn check<X: std::hash::Hash>(&self, s: X) -> bool;
}

impl<const N: usize> Container for BloomFilter<N> {
    fn check<X: std::hash::Hash>(&self, s: X) -> bool {
        self.contains(&s)
    }
}

fn false_pos_rate<C: Container>(filter: &C, control: &HashSet<String>) -> f64 {
    let sample_anti_vals = random_strings(5000, 8, 8, 11);
    false_pos_rate_with_distr(filter, control, &sample_anti_vals)
}

fn false_pos_rate_with_distr<C: Container, X: std::hash::Hash + Eq>(
    filter: &C,
    control: &HashSet<X>,
    sample_anti_vals: &[X],
) -> f64 {
    let mut total = 0;
    let mut false_positives = 0;
    for x in sample_anti_vals {
        if !control.contains(x) {
            total += 1;
            false_positives += filter.check(x) as usize;
        }
    }
    (false_positives as f64) / (total as f64)
}

fn bench_get(c: &mut Criterion) {
    list_fp();

    for vals in [
        random_strings(1000, 1, 12, 1234),
        random_strings(1000, 1, 16, 9876),
    ] {
        for (num_items, bloom_size_bytes) in [(1000, 1 << 16), (1000, 2097152)] {
            let sample_vals = random_strings(num_items, 1, 12, 1234);
            let num_bits = 8 * bloom_size_bytes;
            let bloom = BloomFilter::builder512(num_bits).items(sample_vals.iter());
            let control: HashSet<String> = sample_vals.clone().into_iter().collect();

            let fp = false_pos_rate(&bloom, &control);

            println!("Number of hashes: {:?}", bloom.num_hashes());
            println!("Sampled False Postive Rate: {:.6}%", 100.0 * fp);
            let name = if vals[0] == sample_vals[0] {
                "existing"
            } else {
                "non-existing"
            };
            let bench_title = format!(
                "BloomFilter ({num_items:} items, {bloom_size_bytes:} bytes): get {name:} {num_items:}",
            );
            c.bench_function(&bench_title, |b| {
                b.iter(|| {
                    for val in vals.iter() {
                        let _ = black_box(bloom.contains(val));
                    }
                })
            });
        }
    }
}
fn list_fp() {
    let bloom_size_bytes = 1 << 16;
    for num_items in (1..25).map(|i| 5_000 * i) {
        let sample_vals = random_strings(num_items, 8, 8, 1234);

        let num_bits = bloom_size_bytes * 8;
        let filter = BloomFilter::builder512(num_bits).items(sample_vals.iter());
        let control: HashSet<String> = sample_vals.into_iter().collect();
        print!("{:?}, ", num_items);
        print!("{:?}, ", filter.num_hashes());
        print!("{:.6}", false_pos_rate(&filter, &control));
        println!("");
    }
}
criterion_group!(bench_bloom_filter, bench_get,);
