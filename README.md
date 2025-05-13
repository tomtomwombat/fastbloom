# fastbloom
[![Crates.io](https://img.shields.io/crates/v/fastbloom.svg)](https://crates.io/crates/fastbloom)
[![docs.rs](https://docs.rs/fastbloom/badge.svg)](https://docs.rs/fastbloom)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tomtomwombat/fastbloom/blob/main/LICENSE-MIT)
[![License: APACHE](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/tomtomwombat/fastbloom/blob/main/LICENSE-APACHE)
![Downloads](https://img.shields.io/crates/d/fastbloom)
<a href="https://codecov.io/gh/tomtomwombat/fastbloom">
    <img src="https://codecov.io/gh/tomtomwombat/fastbloom/branch/main/graph/badge.svg">
</a>

The fastest Bloom filter in Rust. No accuracy compromises. Compatible with any hasher.

## Overview

fastbloom is a SIMD accelerated Bloom filter implemented in Rust. fastbloom's default hasher is SipHash-1-3 using randomized keys but can be seeded or configured to use any hasher. fastbloom is 2-400 times faster than existing Bloom filter implementations.

## Usage

Due to a different (improved!) algorithm in 0.11.x, `BloomFilter`s have incompatible serialization/deserialization with 0.10.x!

```toml
# Cargo.toml
[dependencies]
fastbloom = "0.11.0"
```
Basic usage:
```rust
use fastbloom::BloomFilter;

let mut filter = BloomFilter::with_num_bits(1024).expected_items(2);
filter.insert("42");
filter.insert("🦀");
```
Instantiate with a target false positive rate:
```rust
use fastbloom::BloomFilter;

let filter = BloomFilter::with_false_pos(0.001).items(["42", "🦀"]);
assert!(filter.contains("42"));
assert!(filter.contains("🦀"));
```
Use any hasher:
```rust
use fastbloom::BloomFilter;
use ahash::RandomState;

let filter = BloomFilter::with_num_bits(1024)
    .hasher(RandomState::default())
    .items(["42", "🦀"]);
```

## Background
Bloom filters are space-efficient approximate membership set data structures supported by an underlying bit array to track item membership. To insert/check membership, a number of bits are set/checked at positions based on the item's hash. False positives from a membership check are possible, but false negatives are not. Once constructed, neither the Bloom filter's underlying memory usage nor number of bits per item change. [See more.](https://en.wikipedia.org/wiki/Bloom_filter)

```text
hash(4) ──────┬─────┬───────────────┐
              ↓     ↓               ↓
0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0
  ↑           ↑           ↑
  └───────────┴───────────┴──── hash(3) (not in the set)
```

## Implementation

fastbloom is blazingly fast because it uses L1 cache friendly blocks, efficiently derives many index bits from **only one real hash per item**, employs SIMD acceleration, and leverages other research findings on Bloom filters.

fastbloom is a partial blocked Bloom filter. Blocked Bloom filters partition their underlying bit array into sub-array "blocks". Bits set and checked from the item's hash are constrained to a single block instead of the entire bit array. This allows for better cache-efficiency and the opportunity to leverage SIMD and [SWAR](https://en.wikipedia.org/wiki/SWAR) operations when generating bits from an item’s hash. [See more on blocked bloom filters.](https://web.archive.org/web/20070623102632/http://algo2.iti.uni-karlsruhe.de/singler/publications/cacheefficientbloomfilters-wea2007.pdf) Some of fastbloom's hash indexes span the entire bit array while others are confined to a single block.

## Speed
Below is a comparison of runtime speed with other Bloom filter crates:
- "fastbloom" with default block size of 512 bits and using ahash
- "fastbloom - 64" with block size of 64 bits and using ahash

![speed_non_member](https://github.com/user-attachments/assets/c0932c58-9b4f-4305-9115-99692448b6f6)
![speed_member](https://github.com/user-attachments/assets/d7e6f69c-9b5f-44c2-9bb3-486e778e451e)
> Hashers used:
> - xxhash: sbbf, fastbloom-rs
> - Sip1-3: bloom, bloomfilter, probabilistic-collections
> - ahash: fastbloom
> 
> [Benchmark source](https://github.com/tomtomwombat/bench-bloom-filters)

## Accuracy

fastbloom does not compromise accuracy. Below is a comparison of false positive rates with other Bloom filter crates:

![fp_micro](https://github.com/user-attachments/assets/c5dc2801-87ed-4d25-a970-90544fcd39d5)
![fp_macro](https://github.com/user-attachments/assets/8e933289-3ba9-49c5-884e-a97d2efc978c)

[Benchmark source](https://github.com/tomtomwombat/bench-bloom-filters)

## Comparing Block Sizes

fastbloom offers 4 different block sizes: 64, 128, 256, and 512 bits.

```rust
use fastbloom::BloomFilter;

let filter = BloomFilter::with_num_bits(1024).block_size_128().expected_items(2);
```

512 bits is the default. Larger block sizes generally have slower performance but are more accurate, e.g. a Bloom filter with 64 bit blocks is very fast but slightly less accurate.

![b_non_member](https://github.com/user-attachments/assets/bc20120f-84ad-43dc-b244-30bf990029e2)
![b_member](https://github.com/user-attachments/assets/9753dcf4-38a7-4af3-b2a7-b4627eaf07ff)
![b_fp_micro](https://github.com/user-attachments/assets/5a91e435-1a2b-4466-a6c7-d9b2df9c3463)
![b_fp_macro](https://github.com/user-attachments/assets/b5d618d2-d243-4823-bd91-4694456dbaca)
> Bloom filters configured using ahash

## How it Works

fastbloom attributes its performance to two insights:
1. Only one real hash per item is needed, subsequent hashes can be cheaply derived from the real hash using "hash composition"
2. Many bit positions can be derived from a few subsequent hashes through SIMD and SWAR-like operations

#### One Real Hash Per Item

fastbloom employs "hash composition" on two 32-bit halves of an original 64-bit hash. Each subsequent hash is derived by combining the original hash value with a different constant using modular arithmetic and bitwise operations. This results in a set of hash functions that are effectively independent and uniformly distributed, even though they are derived from the same original hash function. Computing the composition of two original hashes is faster than re-computing the hash with a different seed. This technique is [explained in depth in this paper.](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)

#### Many Bit Positions Derived from Subsequent Hashes

Instead of deriving a single bit position per hash, a hash with ~N 1 bits set can be formed by chaining bitwise AND and OR operations of the subsequent hashes.

##### Example

For a Bloom filter with a bit vector of size 64 and desired hashes 24, 24 (potentially overlapping) positions in the bit vector are set or checked for each item on insertion or membership check respectively.

Other traditional Bloom filters derive 24 positions based on 24 hashes of the item:
- `hash0(item) % 64`
- `hash1(item) % 64`
- ...
- `hash23(item) % 64`

Instead, `fastbloom` uses a "sparse hash", a composed hash with less than 32 expected number of bits set. In this case, a ~20 bit set sparse hash is derived from the item and added to the bit vector with a bitwise OR:
- `hash0(item) & hash1(item) | hash2(item) & hash3(item)`

That's 4 hashes versus 24!

Note:
- Given 64 bits, and 24 hashes, a bit has probability (63/64)^24 to NOT be set, i.e. 0, after 24 hashes. The expected number of bits to be set for an item is 64 - (64 * (63/64)^24) ~= 20.
- A 64 bit `hash0(item)` provides us with roughly 32 set bits with a binomial distribution. `hash0(item) & hash1(item)` gives us ~16 set bits, `hash0(item) | hash1(item)` gives us ~48 set bits, etc.

In reality, the Bloom filter may have more than 64 bits of storage. In that case, many underlying `u64`s in the block are operated on using SIMD intrinsics. The number of hashes is adjusted to be the number of hashes per `u64` in the block. Additionally, some bits may be set in the traditional way, across the entire bit vector, to account for any truncating errors from the sparse hash. This also reduces the false positive rate and boosts non-member check speed.

## Available Features

- **`rand`** - Enabled by default, this has the `DefaultHasher` source its random state using `thread_rng()` instead of hardware sources. Getting entropy from a user-space source is considerably faster, but requires additional dependencies to achieve this. Disabling this feature by using `default-features = false` makes `DefaultHasher` source its entropy using `getrandom`, which will have a much simpler code footprint at the expense of speed.

- **`serde`** - `BloomFilter`s implement `Serialize` and `Deserialize` when possible.

## References
- [Bloom filter - Wikipedia](https://en.wikipedia.org/wiki/Bloom_filter)
- [Bloom Filter - Brilliant](https://brilliant.org/wiki/bloom-filter/)
- [Bloom Filter Interactive Demonstration](https://www.jasondavies.com/bloomfilter/)
- [Cache-, Hash- and Space-Efficient Bloom Filters](https://web.archive.org/web/20070623102632/http://algo2.iti.uni-karlsruhe.de/singler/publications/cacheefficientbloomfilters-wea2007.pdf)
- [Less hashing, same performance: Building a better Bloom filter](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)
- [A fast alternative to the modulo reduction](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/)

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
