use super::seeded_hash_from_hashes;

#[inline]
pub(crate) fn hashes_for_bits(target_bits_per_u64_per_item: u64) -> f64 {
    f64::ln(-(((target_bits_per_u64_per_item as f64) / 64.0f64) - 1.0f64))
        / f64::ln(63.0f64 / 64.0f64)
}

/// The number of bitwise operations needed to compute [`signature`] for `bits`
#[inline]
pub(crate) fn work(bits: u64) -> u64 {
    match bits {
        1 => 6,
        2 => 5,
        3 => 6,
        4 => 4,
        5 => 6,
        6 => 5,
        7 => 6,
        8 => 3,
        9 => 6,
        10 => 5,
        11 => 6,
        12 => 4,
        13 => 6,
        14 => 5,
        15 => 6,
        16 => 2,
        17 => 6,
        18 => 5,
        19 => 6,
        20 => 4,
        21 => 6,
        22 => 5,
        23 => 6,
        24 => 3,
        25 => 6,
        26 => 5,
        27 => 6,
        28 => 4,
        29 => 6,
        30 => 5,
        31 => 6,
        32 => 1,
        _ => 100000,
    }
}

/// Returns a `u64` hash with approx `num_bits` number of bits set.
///
/// Using the raw hash is very fast way of setting ~32 random bits
/// For example, using the intersection of two raw hashes is very fast way of setting ~16 random bits.
/// etc
///
/// If the bloom filter has a higher number of hashes to be performed per item,
/// we can use "signatures" to quickly get many index bits, and use traditional
/// index setting for the remainder of hashes.
#[inline]
pub(crate) fn signature(h1: &mut u64, h2: &mut u64, num_bits: u64) -> u64 {
    let mut d = seeded_hash_from_hashes(h1, h2, 0);
    match num_bits {
        1 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        2 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        3 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        4 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
        }
        5 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        6 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        7 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        8 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
        }
        9 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        10 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        11 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        12 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
        }
        13 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        14 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        15 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        16 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
        }
        17 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        18 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        19 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        20 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
        }
        21 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        22 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        23 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        24 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
        }
        25 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        26 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        27 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d &= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        28 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d &= seeded_hash_from_hashes(h1, h2, 3);
        }
        29 => {
            d &= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        30 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d &= seeded_hash_from_hashes(h1, h2, 4);
        }
        31 => {
            d |= seeded_hash_from_hashes(h1, h2, 1);
            d |= seeded_hash_from_hashes(h1, h2, 2);
            d |= seeded_hash_from_hashes(h1, h2, 3);
            d |= seeded_hash_from_hashes(h1, h2, 4);
            d &= seeded_hash_from_hashes(h1, h2, 5);
        }
        _ => {}
    }
    d
}

pub(crate) fn optimize_hashing(total_num_hashes: f64, block_size: usize) -> (u64, Option<u64>) {
    let num_u64s_per_block = (block_size as u64 / 64) as f64;
    let mut num_hashes = total_num_hashes.round() as u64;
    let mut num_rounds = None;

    for target_bits_per_u64_per_item in 1..=32 {
        let hashes_covered = hashes_for_bits(target_bits_per_u64_per_item);
        let remaining = (total_num_hashes - (hashes_covered * num_u64s_per_block)).round();
        if remaining < 0.0 {
            continue; // signature has too many bits
        }
        let hashing_work = remaining as u64;
        let work_for_target_bits = num_u64s_per_block as u64 * work(target_bits_per_u64_per_item);
        let cur_work = num_hashes + num_u64s_per_block as u64 * num_rounds.unwrap_or(0);
        if (hashing_work + work_for_target_bits) < cur_work {
            num_rounds = Some(target_bits_per_u64_per_item);
            num_hashes = hashing_work;
        }
    }
    (num_hashes, num_rounds)
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_num_bits() {
        let mut rng = StdRng::seed_from_u64(42);
        for target_bits in 1..=32 {
            let trials = 10_000;
            let mut total_bits = 0;
            for _ in 0..trials {
                let mut h1 = rng.gen();
                let mut h2 = rng.gen();
                let h = signature(&mut h1, &mut h2, target_bits);
                total_bits += h.count_ones();
            }
            assert_eq!(
                ((total_bits as f64) / (trials as f64)).round() as u64,
                target_bits
            )
        }
    }

    #[test]
    fn hash_creation() {
        for block_size in [64, 128, 256, 512] {
            for num_hashes in 0..5000 {
                let (hashes, num_rounds) = optimize_hashing(num_hashes as f64, block_size);
                assert!(num_rounds.unwrap_or(0) <= 64);
                match num_rounds {
                    None => assert_eq!(num_hashes, hashes, "None"),
                    Some(x) => {
                        let hashes_for_rounds =
                            (hashes_for_bits(x) * (block_size / 64) as f64).round() as u64;
                        assert_eq!(hashes_for_rounds + hashes, num_hashes,
                        "\ntarget hashes: {num_hashes:}\nhashes for rounds {hashes_for_rounds:}\nrounds {x:}\nhashes: {hashes:}")
                    }
                }
            }
        }
    }
}
