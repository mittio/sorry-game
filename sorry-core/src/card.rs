//! Cards used in the standard Sorry! deck.
//!
//! Standard Hasbro deck: 45 cards total.
//!   - 5× One
//!   - 4× each of Two, Three, Four, Five, Seven, Eight, Ten, Eleven, Twelve
//!   - 4× Sorry
//!
//! Note there is no 6 or 9 in the standard deck.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Card {
    One,
    Two,
    Three,
    Four,
    Five,
    Seven,
    Eight,
    Ten,
    Eleven,
    Twelve,
    Sorry,
}

impl Card {
    /// Face value of the card for normal forward movement. Sorry has no rank.
    pub fn rank(self) -> Option<u8> {
        match self {
            Self::One => Some(1),
            Self::Two => Some(2),
            Self::Three => Some(3),
            Self::Four => Some(4),
            Self::Five => Some(5),
            Self::Seven => Some(7),
            Self::Eight => Some(8),
            Self::Ten => Some(10),
            Self::Eleven => Some(11),
            Self::Twelve => Some(12),
            Self::Sorry => None,
        }
    }

    pub fn is_sorry(self) -> bool {
        matches!(self, Self::Sorry)
    }
}

/// Build a fresh, unshuffled Hasbro 45-card Sorry! deck.
pub fn standard_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(45);
    for _ in 0..5 {
        deck.push(Card::One);
    }
    for c in [
        Card::Two,
        Card::Three,
        Card::Four,
        Card::Five,
        Card::Seven,
        Card::Eight,
        Card::Ten,
        Card::Eleven,
        Card::Twelve,
        Card::Sorry,
    ] {
        for _ in 0..4 {
            deck.push(c);
        }
    }
    deck
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_deck_has_45_cards() {
        assert_eq!(standard_deck().len(), 45);
    }

    #[test]
    fn standard_deck_distribution_matches_hasbro() {
        let deck = standard_deck();
        let count = |c: Card| deck.iter().filter(|x| **x == c).count();
        assert_eq!(count(Card::One), 5);
        for c in [
            Card::Two,
            Card::Three,
            Card::Four,
            Card::Five,
            Card::Seven,
            Card::Eight,
            Card::Ten,
            Card::Eleven,
            Card::Twelve,
            Card::Sorry,
        ] {
            assert_eq!(count(c), 4, "expected 4 of {c:?}");
        }
    }

    #[test]
    fn no_six_or_nine_exists_in_enum() {
        // Sanity check: ranks never return 6 or 9.
        for c in [
            Card::One,
            Card::Two,
            Card::Three,
            Card::Four,
            Card::Five,
            Card::Seven,
            Card::Eight,
            Card::Ten,
            Card::Eleven,
            Card::Twelve,
            Card::Sorry,
        ] {
            if let Some(r) = c.rank() {
                assert_ne!(r, 6);
                assert_ne!(r, 9);
            }
        }
    }

    #[test]
    fn sorry_card_has_no_rank() {
        assert!(Card::Sorry.is_sorry());
        assert!(Card::Sorry.rank().is_none());
    }
}
