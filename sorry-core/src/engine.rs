//! Shared draw/commit primitives used by both `Game` (simulator) and
//! `InteractiveGame`. Factored out so the two state machines produce
//! byte-identical `GameHistory` for the same seed.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::board::{BoardState, PlayerId};
use crate::card::Card;
use crate::error::{Result, SorryError};
use crate::history::{Action, TurnRecord};
use crate::moves::{Move, apply_move, legal_moves};
use crate::rules::Rules;

/// Independent RNG streams for one game. The deck stream is only used for
/// shuffles and the starting player, so card order never depends on which
/// strategies are playing; each seat's strategy gets its own stream so one
/// bot's sampling can't shift another's.
pub(crate) struct GameRngs {
    pub deck: StdRng,
    pub seats: Vec<StdRng>,
}

impl GameRngs {
    pub fn new(seed: u64, num_players: usize) -> Self {
        Self {
            // Seeded directly so existing seeds keep their initial deck order.
            deck: StdRng::seed_from_u64(seed),
            seats: (0..num_players as u64)
                .map(|seat| StdRng::seed_from_u64(seat_seed(seed, seat)))
                .collect(),
        }
    }
}

/// splitmix64 finalizer. Mixing `seed` and `seat` through it twice keeps
/// neighbouring seeds and neighbouring seats from sharing a stream.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn seat_seed(seed: u64, seat: u64) -> u64 {
    splitmix64(splitmix64(seed) ^ splitmix64(seat.wrapping_add(1)))
}

pub(crate) struct Engine<'a> {
    pub rules: &'a dyn Rules,
    pub board: &'a mut BoardState,
    /// Deck stream; used only for reshuffles.
    pub rng: &'a mut StdRng,
    pub deck: &'a mut Vec<Card>,
    pub discard: &'a mut Vec<Card>,
}

impl<'a> Engine<'a> {
    pub fn draw_card(&mut self, turn: &mut TurnRecord) -> Result<Card> {
        if self.deck.is_empty() {
            if !self.rules.reshuffle_on_empty_deck() {
                return Err(SorryError::EmptyDeck);
            }
            if self.discard.is_empty() {
                return Err(SorryError::EmptyDeck);
            }
            self.deck.append(self.discard);
            // Sort first so the reshuffle doesn't depend on discard order.
            self.deck.sort();
            self.deck.shuffle(self.rng);
            turn.actions.push(Action::Reshuffle);
        }
        self.deck.pop().ok_or(SorryError::EmptyDeck)
    }

    /// Validate `mv` is a legal move for `player` playing `card`, apply it to
    /// the board, discard the card, and append `Action::Play` to `turn`.
    pub fn commit_play(
        &mut self,
        player: PlayerId,
        card: Card,
        mv: &Move,
        turn: &mut TurnRecord,
    ) -> Result<()> {
        let legal = legal_moves(self.rules, self.board, player, card);
        if !legal.contains(mv) {
            return Err(SorryError::IllegalMove);
        }
        let (bumps, slides) = apply_move(self.rules, self.board, player, mv);
        self.discard.push(card);
        turn.actions.push(Action::Play {
            card,
            mv: mv.clone(),
            bumps,
            slides,
        });
        Ok(())
    }
}
