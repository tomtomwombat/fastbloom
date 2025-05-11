use ahash::{AHasher, RandomState};
use std::hash::BuildHasher;

/// The default hasher for `BloomFilter`.
#[derive(Clone, Debug)]
pub struct DefaultHasher {
    state: RandomState,
    #[allow(dead_code)]
    seed: [u64; 2],
}

impl BuildHasher for DefaultHasher {
    type Hasher = AHasher;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.state.build_hasher()
    }
}

impl DefaultHasher {
    pub fn seeded(seed: &[u8; 16]) -> Self {
        let mut b0 = [0; 8];
        b0.copy_from_slice(&seed[..8]);
        let k0 = u64::from_be_bytes(b0);

        let mut b1 = [0; 8];
        b1.copy_from_slice(&seed[8..]);
        let k1 = u64::from_be_bytes(b1);

        Self {
            state: RandomState::with_seeds(k0, k1, 0, 0),
            seed: [k0, k1],
        }
    }
}

impl Default for DefaultHasher {
    #[inline]
    fn default() -> Self {
        let mut seed = [0u8; 16];

        #[cfg(not(feature = "rand"))]
        {
            getrandom::fill(&mut seed).expect("Unable to obtain entropy from OS/Hardware sources");
        }
        #[cfg(feature = "rand")]
        {
            use rand::RngCore;
            rand::rng().fill_bytes(&mut seed);
        }

        Self::seeded(&seed)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for DefaultHasher {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.seed.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DefaultHasher {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let seed = <[u64; 2]>::deserialize(deserializer)?;
        Ok(Self {
            state: RandomState::with_seeds(seed[0], seed[1], 0, 0),
            seed,
        })
    }
}
