# fastbloom
[![Crates.io](https://img.shields.io/crates/v/fastbloom.svg)](https://crates.io/crates/fastbloom)
[![docs.rs](https://docs.rs/bloomfilter/badge.svg)](https://docs.rs/fastbloom)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tomtomwombat/fastbloom/blob/main/LICENSE-MIT)
[![License: APACHE](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/tomtomwombat/fastbloom/blob/main/LICENSE-Apache)
<a href="https://codecov.io/gh/tomtomwombat/fastbloom">
    <img src="https://codecov.io/gh/tomtomwombat/fastbloom/branch/main/graph/badge.svg">
</a>

The fastest bloom filter in Rust. No accuracy compromises. Compatible with any hasher.


### Usage

```toml
# Cargo.toml
[dependencies]
fastbloom = "0.2.0"
```
Basic usage:
```rust
use fastbloom::BloomFilter;

let num_bits = 1024;

let mut filter = BloomFilter::builder(num_bits).expected_items(2);
filter.insert("42");
filter.insert("🦀");
```
Instantiate from a collection of items:
```rust
use fastbloom::BloomFilter;

let num_bits = 1024;

let filter = BloomFilter::builder(num_bits).items(["42", "🦀"]);
assert!(filter.contains("42"));
assert!(filter.contains("🦀"));
```
Use any hasher:
```rust
use fastbloom::BloomFilter;
use ahash::RandomState;

let num_bits = 1024;

let filter = BloomFilter::builder(num_bits)
    .hasher(RandomState::default())
    .items(["42", "🦀"]);
```

### Background
Bloom filters are space efficient approximate membership set data structures. False positives from a membership check are possible, but false negatives are not. [See more.](https://en.wikipedia.org/wiki/Bloom_filter)

Blocked bloom filters are supported by an underlying bit vector, chunked into 512, 256, 128, or 64 bit "blocks", to track item membership. To insert, a number of bits are set at positions based on the item's hash in one of the underlying bit vector's blocks. To check membership, a number of bits are checked at positions based on the item's hash in one of the underlying bit vector's blocks. 
```ignore
hash(4) ──────┰─────┰───────────────┐
              ↓     ↓               ↓
0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0
  ↑           ↑           ↑
  └───────────┴───────────┴──── hash(3) (not in the set)

```
[See more on blocked bloom filters.](https://web.archive.org/web/20070623102632/http://algo2.iti.uni-karlsruhe.de/singler/publications/cacheefficientbloomfilters-wea2007.pdf)

Once constructed, neither the bloom filter's underlying memory usage nor number of bits per item change.


### Implementation

`fastbloom` is **several times faster** than existing bloom filters and scales very well with the number of hashes per item. In all cases, `fastbloom` maintains competitive false positive rates. `fastbloom` is blazingly fast because it uses L1 cache friendly blocks, efficiently derives many index bits from **only one real hash per item**, and leverages other research findings on bloom filters.

`fastbloom`'s default hasher is SipHash-1-3 using randomized keys but can be seeded or configured to use any hasher.

### Runtime Performance

#### SipHash
Runtime comparison to other bloom filter crates (all using SipHash).
Note:
- The number hashes for all bloom filters is derived to optimized accuracy, meaning less items in the bloom filters results in more hashes per item and generally slower performance on membership checks.
- As number of items (input) increases, the accuracy of the bloom filter decreases. 1000 random strings were used to test membership.
> ![member](https://github.com/tomtomwombat/fastbloom/assets/45644087/c74ea802-a7a2-4df7-943c-92b3bcec982e)
> ![non-member](https://github.com/tomtomwombat/fastbloom/assets/45644087/326c2558-6f86-4675-99cb-c95aed73e90d)


#### Any Hash Goes
The fastbloom-rs crate (similarily named) uses xxhash, which is faster than SipHash, so it is not fair to compare above. However, we can configure `fastbloom` to use a similarly fast hash, ahash, and compare. 1000 random strings were used to test membership.
> ![member-fb](https://github.com/tomtomwombat/fastbloom/assets/45644087/9bf303fd-897d-412b-9f42-c57e6460ead0)
> ![non-member-fb](https://github.com/tomtomwombat/fastbloom/assets/45644087/060e739b-7fb2-4c18-8086-7034f6fb92c0)



### False Positive Performance

`fastbloom` does not compromise accuracy. Below is a comparison of false positive rates with other bloom filter crates:
> ![bloom-fp](https://github.com/tomtomwombat/fastbloom/assets/45644087/07e22ab3-f777-4e4e-8910-4f1c764e4134)
> The bloom filters and a control hash set were populated with a varying number of random 64 bit integers ("Number of Items"). Then 100,000 random 64 bit integers were checked: false positives are numbers that do NOT exist in the control hash set but do report as existing in the bloom filter.

### How it Works

For a bloom filter with a bit vector of size 64 and desired hashes 24, 24 (potentially overlapping) positions in the bit vector are set or checked for each item on insertion or membership check respectively.

Other bloom filters derive 24 positions based on 24 hashes of the item:
- `hash0(item) % 64`
- `hash1(item) % 64`
- ...
- `hash23(item) % 64`

Instead, `fastbloom` derives a hash of the item with ~20 bits set and then adds it to the bit vector with a bitwise OR:
- `hash0(item) & hash1(item) | hash2(item) & hash3(item)`

That's 4 hashes versus 24!

Note:
- Given 64 bits, and 24 hashes, a bit has probability ${\frac {63} {64}}^{24}$ to NOT be set, i.e. 0, after 24 hashes. The expected number of bits to be set for an item is $64 - (64 * {\frac {63} {64}}^{24}) ~= 20$.
- A 64 bit `hash0(item)` provides us with roughly 32 set bits with a binomial distribution. `hash0(item) & hash1(item)` gives us ~16 set bits, `hash0(item) | hash1(item)` gives us ~48 set bits, etc.

In reality, the bloom filter may have more than 64 bits of storage. In that case, many underlying `u64`s in the block are operated on, and the number of hashes is adjusted to be the number of hashes per `u64` in the block. Additionally, some bits may be set in the usual way to account for any rounding errors.

## References
- [Bloom filter - Wikipedia](https://en.wikipedia.org/wiki/Bloom_filter)
- [Bloom Filter - Brilliant](https://brilliant.org/wiki/bloom-filter/)
- [Bloom Filter Interactive Demonstration](https://www.jasondavies.com/bloomfilter/)
- [Cache-, Hash- and Space-Efficient Bloom Filters](https://web.archive.org/web/20070623102632/http://algo2.iti.uni-karlsruhe.de/singler/publications/cacheefficientbloomfilters-wea2007.pdf)
- [Less hashing, same performance: Building a better Bloom filter](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)
- [A fast alternative to the modulo reduction](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/)

## License


 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
