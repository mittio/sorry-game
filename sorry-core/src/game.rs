// ChaCha12 via rand's StdRng is used for platform-agnostic determinism —
// the same seed must produce the same game history across native and WASM
// builds. Do not switch to `rand::thread_rng` or `rand::rngs::OsRng` for
// shuffles.

use rand::seq::SliceRandom;

use crate::board::{BoardState, PlayerId};
use crate::card::Card;
use crate::engine::{Engine, GameRngs};
use crate::error::{Result, SorryError};
use crate::history::{Action, GameHistory, TurnRecord};
use crate::moves::legal_moves;
use crate::rules::Rules;
use crate::strategy::{Strategy, StrategyView};

pub struct Game {
    rules: Box<dyn Rules>,
    strategies: Vec<Box<dyn Strategy>>,
    board: BoardState,
    rngs: GameRngs,
    deck: Vec<Card>,
    discard: Vec<Card>,
    hands: Vec<Vec<Card>>,
    history: GameHistory,
    current: PlayerId,
    over: bool,
    max_turns: usize,
    /// Player most recently knocked back to Start by any bump (landing,
    /// slide, or Sorry). `None` until the first bump of the game.
    /// Surfaced to strategies via `StrategyView::last_bump_victim`.
    last_bump_victim: Option<PlayerId>,
}

const DEFAULT_MAX_TURNS: usize = 5_000;

impl Game {
    pub fn new(
        rules: Box<dyn Rules>,
        strategies: Vec<Box<dyn Strategy>>,
        seed: u64,
    ) -> Result<Self> {
        let num_players = strategies.len();
        if num_players < rules.min_players() {
            return Err(SorryError::NotEnoughPlayers {
                min: rules.min_players(),
                got: num_players,
            });
        }
        if num_players > rules.max_players() {
            return Err(SorryError::TooManyPlayers {
                max: rules.max_players(),
                got: num_players,
            });
        }

        let mut rngs = GameRngs::new(seed, num_players);
        let mut deck = rules.build_deck();
        deck.shuffle(&mut rngs.deck);
        let initial_deck_order = deck.clone();

        let start_areas: Vec<_> = (0..num_players as u8)
            .map(|p| rules.start_area(PlayerId(p)))
            .collect();
        let board = BoardState::fresh(num_players, rules.pawns_per_player(), &start_areas);

        let mut hands = vec![Vec::new(); num_players];
        let hand_size = rules.hand_size();
        for hand in hands.iter_mut() {
            for _ in 0..hand_size {
                let card = deck.pop().ok_or(SorryError::EmptyDeck)?;
                hand.push(card);
            }
        }

        let starting_player = rules.starting_player(num_players, &mut rngs.deck);
        let strategy_names: Vec<String> = strategies.iter().map(|s| s.name().to_string()).collect();
        let rules_name = rules.name().to_string();

        let history = GameHistory {
            seed,
            num_players,
            strategy_names,
            rules_name,
            initial_deck_order,
            starting_player,
            turns: Vec::new(),
            winners: Vec::new(),
            truncated: false,
        };

        Ok(Self {
            rules,
            strategies,
            board,
            rngs,
            deck,
            discard: Vec::new(),
            hands,
            history,
            current: starting_player,
            over: false,
            max_turns: DEFAULT_MAX_TURNS,
            last_bump_victim: None,
        })
    }

    pub fn set_max_turns(&mut self, limit: usize) {
        self.max_turns = limit;
    }

    pub fn play(mut self) -> Result<GameHistory> {
        while !self.over {
            if self.history.turns.len() >= self.max_turns {
                self.history.truncated = true;
                break;
            }
            self.play_turn()?;
            self.record_any_new_finisher();
            self.check_game_over();
            if !self.over {
                self.advance_current();
            }
        }
        if self.over && !self.history.truncated {
            let n = self.history.num_players;
            self.rules.finalize_winners(&mut self.history.winners, n);
        }
        Ok(self.history)
    }

    fn play_turn(&mut self) -> Result<()> {
        let mut turn = TurnRecord {
            player: self.current,
            actions: Vec::new(),
        };
        // Loop allows extra turns granted by card 2.
        loop {
            let card = {
                let mut engine = Engine {
                    rules: &*self.rules,
                    board: &mut self.board,
                    rng: &mut self.rngs.deck,
                    deck: &mut self.deck,
                    discard: &mut self.discard,
                };
                engine.draw_card(&mut turn)?
            };
            turn.actions.push(Action::Draw { card });

            // Resolve which card to play.
            let chosen = if self.rules.hand_size() == 0 {
                card
            } else {
                self.hands[self.current.0 as usize].push(card);
                let view = self.view_for_current(None);
                let idx = self.strategies[self.current.0 as usize].choose_card(
                    &view,
                    &*self.rules,
                    &mut self.rngs.seats[self.current.0 as usize],
                );
                let idx = idx.min(self.hands[self.current.0 as usize].len() - 1);
                let chosen = self.hands[self.current.0 as usize].remove(idx);
                turn.actions.push(Action::ChooseCard {
                    hand_index: idx,
                    card: chosen,
                });
                chosen
            };

            // Generate legal moves and let the strategy pick one.
            let legal = legal_moves(&*self.rules, &self.board, self.current, chosen);
            let view = self.view_for_current(Some(chosen));
            let chosen_move = self.strategies[self.current.0 as usize].choose_move(
                &view,
                &*self.rules,
                chosen,
                &legal,
                &mut self.rngs.seats[self.current.0 as usize],
            );
            if !legal.contains(&chosen_move) {
                return Err(SorryError::InvalidMove(format!(
                    "strategy {} returned a move not in legal set for {:?}",
                    self.strategies[self.current.0 as usize].name(),
                    chosen
                )));
            }

            {
                let mut engine = Engine {
                    rules: &*self.rules,
                    board: &mut self.board,
                    rng: &mut self.rngs.deck,
                    deck: &mut self.deck,
                    discard: &mut self.discard,
                };
                engine.commit_play(self.current, chosen, &chosen_move, &mut turn)?;
            }
            // Any bumps emitted by the play update the most-recent-victim
            // marker used by Sidekick. Read the last Play action (just
            // pushed by commit_play) and take the final bump's victim — a
            // slide-through-multiple-pawns leaves the furthest-along bump
            // last, which is a reasonable "freshest victim" definition.
            if let Some(Action::Play { bumps, .. }) = turn.actions.last()
                && let Some(last) = bumps.last()
            {
                self.last_bump_victim = Some(last.player);
            }

            // Check for winner in the middle of the turn — if we just won, don't
            // take an extra turn.
            if self.rules.is_winner(&self.board, self.current) {
                break;
            }

            if self.rules.grants_extra_turn(chosen) {
                turn.actions.push(Action::ExtraTurnGranted);
                continue;
            }
            break;
        }
        self.history.turns.push(turn);
        Ok(())
    }

    fn advance_current(&mut self) {
        let n = self.history.num_players as u8;
        let mut next = (self.current.0 + 1) % n;
        // Skip anyone already recorded as finished — `play_out` rulesets
        // exclude them from subsequent turns so the remaining players
        // can establish a full placement order.
        for _ in 0..n {
            if !self.history.winners.iter().any(|w| w.0 == next) {
                break;
            }
            next = (next + 1) % n;
        }
        self.current = PlayerId(next);
    }

    /// Append `self.current` to `history.winners` iff they just finished
    /// and aren't already recorded. This is how Play Out builds its
    /// ordered 1st→Nth placement list turn by turn.
    fn record_any_new_finisher(&mut self) {
        if self.rules.is_winner(&self.board, self.current)
            && !self.history.winners.contains(&self.current)
        {
            self.history.winners.push(self.current);
        }
    }

    fn check_game_over(&mut self) {
        let n = self.history.num_players;
        if self.rules.game_over(&self.history.winners, n) {
            self.over = true;
        }
    }

    fn view_for_current(&self, drawn: Option<Card>) -> StrategyView {
        StrategyView {
            my_player: self.current,
            num_players: self.history.num_players,
            hand: self.hands[self.current.0 as usize].clone(),
            drawn_card: drawn,
            pawn_positions: self.board.all_positions().to_vec(),
            discard: self.discard.clone(),
            deck_remaining: self.deck.len(),
            current_player_turn: self.current,
            last_bump_victim: self.last_bump_victim,
        }
    }
}
