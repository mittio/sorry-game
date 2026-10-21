//! The deck stream and each seat's strategy stream are independent, so a
//! bot's randomness can't change the cards or an opponent's random choices.

use std::sync::{Arc, Mutex};

use rand::RngCore;
use sorry_core::*;

/// Plays the first legal move, but burns `extra` values from its RNG each call
/// so that streams are consumed at different rates without altering the game.
struct Recorder {
    extra: usize,
    first_values: Arc<Mutex<Vec<u64>>>,
}

impl Strategy for Recorder {
    fn name(&self) -> &str {
        "Recorder"
    }

    fn choose_move(
        &self,
        _view: &StrategyView,
        _rules: &dyn Rules,
        _card: Card,
        legal: &[Move],
        rng: &mut dyn RngCore,
    ) -> Move {
        self.first_values.lock().unwrap().push(rng.next_u64());
        for _ in 0..self.extra {
            rng.next_u64();
        }
        legal[0].clone()
    }
}

fn recorder(extra: usize) -> (Box<dyn Strategy>, Arc<Mutex<Vec<u64>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let s = Box::new(Recorder {
        extra,
        first_values: Arc::clone(&log),
    });
    (s, log)
}

fn draws(h: &GameHistory) -> Vec<Card> {
    h.turns
        .iter()
        .flat_map(|t| t.actions.iter())
        .filter_map(|a| match a {
            Action::Draw { card } => Some(*card),
            _ => None,
        })
        .collect()
}

fn play(strategies: Vec<Box<dyn Strategy>>, seed: u64) -> GameHistory {
    Game::new(Box::new(StandardRules::new()), strategies, seed)
        .expect("game")
        .play()
        .expect("play")
}

#[test]
fn strategies_do_not_change_deck_order_or_starting_player() {
    let mut saw_reshuffle = false;
    for seed in 0..20u64 {
        let a = play(vec![Box::new(RandomStrategy), Box::new(RandomStrategy)], seed);
        let b = play(vec![Box::new(GreedyStrategy), Box::new(SurvivorStrategy)], seed);
        let c = play(vec![Box::new(LoserStrategy), Box::new(SidekickStrategy)], seed);

        assert_eq!(a.initial_deck_order, b.initial_deck_order);
        assert_eq!(a.initial_deck_order, c.initial_deck_order);
        assert_eq!(a.starting_player, b.starting_player);
        assert_eq!(a.starting_player, c.starting_player);

        // Draw order, including every post-reshuffle deck, must match for as
        // long as both games are still running.
        let (da, db, dc) = (draws(&a), draws(&b), draws(&c));
        let n = da.len().min(db.len()).min(dc.len());
        assert_eq!(da[..n], db[..n], "seed {seed}");
        assert_eq!(da[..n], dc[..n], "seed {seed}");
        saw_reshuffle |= n > 45;
    }
    assert!(saw_reshuffle, "no seed ran past the first reshuffle");
}

#[test]
fn seat_rng_is_unaffected_by_other_seats_consumption() {
    for seed in 0..10u64 {
        let (s0_a, log0_a) = recorder(0);
        let (s1_a, _) = recorder(0);
        let (s0_b, log0_b) = recorder(0);
        let (s1_b, _) = recorder(7);

        play(vec![s0_a, s1_a], seed);
        play(vec![s0_b, s1_b], seed);

        let (a, b) = (log0_a.lock().unwrap(), log0_b.lock().unwrap());
        assert!(!a.is_empty());
        assert_eq!(*a, *b, "seed {seed}");
    }
}

#[test]
fn seats_get_distinct_streams() {
    let (s0, log0) = recorder(0);
    let (s1, log1) = recorder(0);
    play(vec![s0, s1], 42);
    assert_ne!(log0.lock().unwrap()[0], log1.lock().unwrap()[0]);
}
