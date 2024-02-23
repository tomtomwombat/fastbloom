use criterion::criterion_main;

mod bench_bloom_filter;
use crate::bench_bloom_filter::bench_bloom_filter;

criterion_main!(bench_bloom_filter);
