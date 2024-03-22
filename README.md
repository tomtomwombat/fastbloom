# fastbloom
[![Crates.io](https://img.shields.io/crates/v/fastbloom.svg)](https://crates.io/crates/fastbloom)
[![docs.rs](https://docs.rs/bloomfilter/badge.svg)](https://docs.rs/fastbloom)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tomtomwombat/fastbloom/blob/main/LICENSE-MIT)
[![License: APACHE](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/tomtomwombat/fastbloom/blob/main/LICENSE-APACHE)
![Downloads](https://img.shields.io/crates/d/fastbloom)
<a href="https://codecov.io/gh/tomtomwombat/fastbloom">
    <img src="https://codecov.io/gh/tomtomwombat/fastbloom/branch/main/graph/badge.svg">
</a>

The fastest Bloom filter in Rust. Compatible with any hasher.


## Usage

```toml
# Cargo.toml
[dependencies]
fastbloom = "0.5.1"
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

`fastbloom`'s default hasher is SipHash-1-3 using randomized keys but can be seeded or configured to use any hasher.

## Implementation

`fastbloom` is **several times faster** than existing Bloom filters and scales very well with the number of hashes per item. In all cases, `fastbloom` maintains competitive false positive rates. `fastbloom` is blazingly fast because it uses L1 cache friendly blocks, efficiently derives many index bits from **only one real hash per item**, and leverages other research findings on Bloom filters.


`fastbloom` is implemented as a blocked Bloom filter. Blocked Bloom filters partition their underlying bit array into sub-array “blocks”. Bits set and checked from the item’s hash are constrained to a single block instead of the entire bit array. This allows for better cache-efficiency and the opportunity to leverage SIMD and [SWAR](https://en.wikipedia.org/wiki/SWAR) operations when generating bits from an item’s hash. [See more on blocked bloom filters.](https://web.archive.org/web/20070623102632/http://algo2.iti.uni-karlsruhe.de/singler/publications/cacheefficientbloomfilters-wea2007.pdf)


## Runtime Performance

`fastbloom` is 50-1000% faster than existing Bloom filters implemented in Rust.

#### SipHash
Runtime comparison to other Bloom filter crates (all using SipHash).
Note:
- The number hashes for all Bloom filters is derived to optimize accuracy, meaning fewer items in the Bloom filters result in more hashes per item and generally slower performance.
- As number of items (input) increases, the accuracy of the Bloom filter decreases. 1000 random strings were used to test membership.

![member](https://github.com/tomtomwombat/fastbloom/assets/45644087/c74ea802-a7a2-4df7-943c-92b3bcec982e)
![non-member](https://github.com/tomtomwombat/fastbloom/assets/45644087/326c2558-6f86-4675-99cb-c95aed73e90d)


#### Any Hash Goes
The fastbloom-rs crate (similarily named) uses xxhash, which is faster than SipHash, so it is not fair to compare above. However, we can configure `fastbloom` to use a similarly fast hash, ahash, and compare. 1000 random strings were used to test membership.

![member-fb](https://github.com/tomtomwombat/fastbloom/assets/45644087/9bf303fd-897d-412b-9f42-c57e6460ead0)
![non-member-fb](https://github.com/tomtomwombat/fastbloom/assets/45644087/060e739b-7fb2-4c18-8086-7034f6fb92c0)


[Benchmark source](https://github.com/tomtomwombat/bench-bloom-filters)

## False Positive Performance

`fastbloom` does not compromise accuracy. Below is a comparison of false positive rates with other Bloom filter crates:

![bloom-fp](https://github.com/tomtomwombat/fastbloom/assets/45644087/07e22ab3-f777-4e4e-8910-4f1c764e4134)
> The Bloom filters and a control hash set were populated with a varying number of random 64 bit integers ("Number of Items"). Then 100,000 random 64 bit integers were checked: false positives are numbers that do NOT exist in the control hash set but do report as existing in the Bloom filter.

[Benchmark source](https://github.com/tomtomwombat/bench-bloom-filters)

## Comparing Block Sizes

`fastbloom` offers 4 different block sizes: 64, 128, 256, and 512 bits. 512 bits is the default. Larger block sizes generally have slower performance but are more accurate.

#### Runtime Performance
Times are for 1000 random strings. The Bloom filters used ahash.

![member-fastbloom-blocks](https://github.com/tomtomwombat/fastbloom/assets/45644087/44073965-cc2d-4e70-9151-7e821b30b208)
![non-member-fastbloom-blocks](https://github.com/tomtomwombat/fastbloom/assets/45644087/6e5ee0e0-f460-46b9-95d6-f4b91d9fa424)


#### Accuracy
![fastbloom-block-fp](https://github.com/tomtomwombat/fastbloom/assets/45644087/c8e88ddb-3617-4d85-8f76-b606b4e98e13)

## How it Works

`fastbloom` attributes its performance to two insights:
1. Only one real hash per item is needed, subsequent hashes can be cheaply derived from the real hash using "hash composition"
2. Many bit positions can be derived from a few subsequent hashes through SWAR-like operations

#### One Real Hash Per Item

`fastbloom` employs "hash composition" on two 32-bit halves of an original 64-bit hash. Each subsequent hash is derived by combining the original hash value with a different constant using modular arithmetic and bitwise operations. This results in a set of hash functions that are effectively independent and uniformly distributed, even though they are derived from the same original hash function. Computing the composition of two original hashes is faster than re-computing the hash with a different seed. This technique is [explained in depth in this paper.](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)

#### Many Bit Positions Derived from Subsequent Hashes

Instead of deriving a single bit position per hash, a hash with ~N 1 bits set can be formed by chaining bitwise AND and OR operations of the subsequent hashes.

##### Example

For a Bloom filter with a bit vector of size 64 and desired hashes 24, 24 (potentially overlapping) positions in the bit vector are set or checked for each item on insertion or membership check respectively.

Other Bloom filters derive 24 positions based on 24 hashes of the item:
- `hash0(item) % 64`
- `hash1(item) % 64`
- ...
- `hash23(item) % 64`

Instead, `fastbloom` derives a hash of the item with ~20 bits set and then adds it to the bit vector with a bitwise OR:
- `hash0(item) & hash1(item) | hash2(item) & hash3(item)`

That's 4 hashes versus 24!

Note:
- Given 64 bits, and 24 hashes, a bit has probability (63/64)^24 to NOT be set, i.e. 0, after 24 hashes. The expected number of bits to be set for an item is 64 - (64 * (63/64)^24) ~= 20.
- A 64 bit `hash0(item)` provides us with roughly 32 set bits with a binomial distribution. `hash0(item) & hash1(item)` gives us ~16 set bits, `hash0(item) | hash1(item)` gives us ~48 set bits, etc.

In reality, the Bloom filter may have more than 64 bits of storage. In that case, many underlying `u64`s in the block are operated on, and the number of hashes is adjusted to be the number of hashes per `u64` in the block. Additionally, some bits may be set in the usual way to account for any rounding errors.

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
