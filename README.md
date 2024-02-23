# fastbloom
[![Crates.io](https://img.shields.io/crates/v/fastbloom.svg)](https://crates.io/crates/fastbloom)
[![docs.rs](https://docs.rs/bloomfilter/badge.svg)](https://docs.rs/fastbloom)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tomtomwombat/bloom-filter/blob/main/LICENSE-MIT)
[![License: APACHE](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/tomtomwombat/bloom-filter/blob/main/LICENSE-Apache)
<a href="https://codecov.io/gh/tomtomwombat/fastbloom">
    <img src="https://codecov.io/gh/tomtomwombat/fastbloom/branch/main/graph/badge.svg">
</a>

The fastest bloom filter in Rust. No accuracy compromises. Compatible with any hasher.


### Usage

```toml
# Cargo.toml
[dependencies]
fastbloom = "0.1.0"
```

```rust
use fastbloom::BloomFilter;

let num_bits = 1024;

let filter = BloomFilter::builder(num_bits).items(["42", "🦀"]);
assert!(filter.contains("42"));
assert!(filter.contains("🦀"));
```

```rust
use fastbloom::BloomFilter;
use ahash::RandomState;

let num_bits = 1024;

let filter = BloomFilter::builder(num_bits)
    .hasher(RandomState::default())
    .items(["42", "🦀"]);
```

### Background
Bloom filters are a space efficient approximate membership set data structure. False positives from `contains` are possible, but false negatives are not, i.e. `contains` for all items in the set is guaranteed to return true, while `contains` for all items not in the set probably return false.

Blocked bloom filters are supported by an underlying bit vector, chunked into 512, 256, 128, or 64 bit "blocks", to track item membership. To insert, a number of bits, based on the item's hash, are set in the underlying bit vector. To check membership, a number of bits, based on the item's hash, are checked in the underlying bit vector.

Once constructed, neither the bloom filter's underlying memory usage nor number of bits per item change.


### Implementation

`fastbloom` is blazingly fast because it uses L1 cache friendly blocks and efficiently derives many index bits from only one hash per value. Compared to traditional implementations, `fastbloom` is 2-5 times faster for small sets of items, and hundreds of times faster for larger item sets. In all cases, `fastbloom` maintains competitive false positive rates.

### Runtime Performance

Runtime comparison to other bloom filter crates:
- Bloom memory size = 16Kb
- 1000 contained items
- 364 hashes per item
  
|  | Check Non-Existing (ns) | Check Existing (ns) |
| --- | --- | --- |
| fastbloom | 16.900 | 139.62 |
| *fastbloom-rs | 35.358 | 485.81 |
| bloom | 66.136 | 749.27 |
| bloomfilter | 68.912 | 996.56 |
| probabilistic-collections | 83.385 | 974.67 |

*fastbloom-rs uses XXHash, which is faster than SipHash (the default hasher for all other bloom filters listed).

### False Positive Performance

`fastbloom` does not compromise accuracy. Below is a comparison false positive rate with other bloom filter crates:
> ![bloom-fp](https://github.com/tomtomwombat/fastbloom/assets/45644087/6d3bd507-604a-4ba6-90e0-15e024178bba)



### Scaling

`fastbloom` scales very well.

As the number of bits and set size increase, traditional bloom filters need to perform more hashes per item to keep false positive rates low. However, `fastbloom`'s optimal number of hashes is bounded while keeping near zero rates even for many items:
> ![bloom-scaling](https://github.com/tomtomwombat/fastbloom/assets/45644087/f00607d6-1313-4296-aef2-9b86815eeba7)
>
> Bloom filter speed is directly proportional to number of hashes.

## References
- [Bloom Filter](https://brilliant.org/wiki/bloom-filter/)
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
