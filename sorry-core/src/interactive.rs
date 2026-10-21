//! Interactive-play state machine — drives a `Game` one action at a time so
//! a human, a networked client, or the WASM bridge can pump each decision.
//!
//! Shares `Engine::draw_card` and `Engine::commit_play` with the simulator
//! (`game::Game`), so a fully-bot-driven `InteractiveGame` produces a
//! `GameHistory` byte-identical to `Game::play()` for the same seed and
//! strategies. That invariant is the primary integrity guarantee of the
//! interactive layer and is locked in by
//! `tests/interactive.rs::interactive_history_matches_simulator_history_byte_for_byte`.

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::board::{BoardState, PlayerId, SpaceId};
use crate::card::Card;
use crate::engine::{Engine, GameRngs};
use crate::error::{Result, SorryError};
use crate::history::{Action, GameHistory, TurnRecord};
use crate::moves::{Move, legal_moves};
use crate::rules::Rules;
use crate::strategy::{Strategy, StrategyView};

const DEFAULT_MAX_TURNS: usize = 5_000;

/// State-machine enum describing what decision the game is waiting on.
///
/// Standard Sorry! (`Rules::hand_size() == 0`) never surfaces a draw as a
/// decision — draws are auto-resolved and the next pending action is always
/// `ChooseMove` (or `GameOver`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionNeeded {
    /// Variant rules only (`Rules::hand_size() > 0`). The player's hand has
    /// been topped up with the freshly drawn card; they pick which card to
    /// play.
    ChooseCard {
        player: PlayerId,
        hand: Vec<Card>,
        legal_card_indices: Vec<usize>,
    },
    /// Player must pick a move for `card`.
    ChooseMove {
        player: PlayerId,
        card: Card,
        legal_moves: Vec<Move>,
    },
    GameOver {
        winners: Vec<PlayerId>,
        truncated: bool,
    },
}

/// Caller-supplied action that advances the state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PlayerAction {
    ChooseCard { hand_index: usize },
    PlayMove { mv: Move },
}

/// Full public snapshot — all information the game holds. Intended for
/// local-hotseat UIs and debug output. For networked play use
/// `get_player_view`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveGameState {
    pub num_players: usize,
    pub current_player: PlayerId,
    pub starting_player: PlayerId,
    pub pawn_positions: Vec<Vec<SpaceId>>,
    pub discard: Vec<Card>,
    pub deck_remaining: usize,
    pub hands: Vec<Vec<Card>>,
    pub drawn_card: Option<Card>,
    pub rules_name: String,
    pub strategy_names: Vec<String>,
    pub action_needed: ActionNeeded,
    pub winners: Vec<PlayerId>,
    pub truncated: bool,
    pub turn_count: usize,
    /// Turn currently being assembled — actions accumulate here and only
    /// flush into `history.turns` when the turn ends (which is *after*
    /// an extra-turn chain completes). Exposing it lets clients see the
    /// most recent `Action::Play` that hasn't been finalized yet.
    pub current_turn: Option<TurnRecord>,
    /// `seat_sides[p]` is the board side (0=Red, 1=Blue, 2=Yellow,
    /// 3=Green) for engine player `p`. Default is identity for games
    /// that use all four colors; skipped-color games have non-contiguous
    /// values.
    pub seat_sides: Vec<u8>,
}

/// Per-player snapshot — hides other players' persistent hands from
/// spectators. In variant rules (`Rules::hand_size() > 0`) the drawn card
/// enters the current player's hand and is only visible via their own
/// `hands[viewer]` slot. In standard Sorry! (`hand_size() == 0`) the drawn
/// card is face-up and public, so `drawn_card` is exposed to every viewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerView {
    pub viewer: PlayerId,
    pub num_players: usize,
    pub current_player: PlayerId,
    pub starting_player: PlayerId,
    pub pawn_positions: Vec<Vec<SpaceId>>,
    pub discard: Vec<Card>,
    pub deck_remaining: usize,
    /// `hands[p]` is `Some` only for `p == viewer`.
    pub hands: Vec<Option<Vec<Card>>>,
    /// `Some` when a card has been drawn face-up but not yet played in a
    /// `hand_size == 0` game. Public information — visible to all viewers,
    /// including spectators. Always `None` for variant rules.
    pub drawn_card: Option<Card>,
    pub rules_name: String,
    pub strategy_names: Vec<String>,
    /// `ChooseCard::hand` is redacted (to an empty `Vec`) when
    /// `viewer != player` so spectators don't see another player's hand.
    pub action_needed: ActionNeeded,
    pub winners: Vec<PlayerId>,
    pub truncated: bool,
    pub turn_count: usize,
    /// `seat_sides[p]` — engine-player to board-side mapping. See
    /// `InteractiveGameState::seat_sides` for details.
    pub seat_sides: Vec<u8>,
}

pub struct InteractiveGame {
    rules: Box<dyn Rules>,
    board: BoardState,
    rngs: GameRngs,
    deck: Vec<Card>,
    discard: Vec<Card>,
    hands: Vec<Vec<Card>>,
    history: GameHistory,
    current: PlayerId,
    max_turns: usize,
    action_needed: ActionNeeded,
    /// Accumulates this player's turn. `Some` while the game is mid-turn;
    /// `None` once `action_needed` is `GameOver`.
    current_turn: Option<TurnRecord>,
    /// Most recent player bumped to Start. Mirrors `Game::last_bump_victim`.
    last_bump_victim: Option<PlayerId>,
}

impl InteractiveGame {
    pub fn new(rules: Box<dyn Rules>, num_players: usize, seed: u64) -> Result<Self> {
        let names = vec!["Interactive".to_string(); num_players];
        Self::new_with_strategy_names(rules, names, seed)
    }

    pub fn new_with_strategy_names(
        rules: Box<dyn Rules>,
        strategy_names: Vec<String>,
        seed: u64,
    ) -> Result<Self> {
        let num_players = strategy_names.len();
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

        let mut game = Self {
            rules,
            board,
            rngs,
            deck,
            discard: Vec::new(),
            hands,
            history,
            current: starting_player,
            max_turns: DEFAULT_MAX_TURNS,
            action_needed: ActionNeeded::GameOver {
                winners: Vec::new(),
                truncated: false,
            },
            current_turn: None,
            last_bump_victim: None,
        };
        game.begin_turn_or_game_over()?;
        Ok(game)
    }

    pub fn set_max_turns(&mut self, limit: usize) {
        self.max_turns = limit;
    }

    pub fn history(&self) -> &GameHistory {
        &self.history
    }

    pub fn action_needed(&self) -> &ActionNeeded {
        &self.action_needed
    }

    pub fn current_player(&self) -> PlayerId {
        self.current
    }

    pub fn is_over(&self) -> bool {
        matches!(self.action_needed, ActionNeeded::GameOver { .. })
    }

    /// Validate and apply `action` against the currently pending
    /// `ActionNeeded`. Returns a reference to the new pending state.
    pub fn apply_action(&mut self, action: PlayerAction) -> Result<&ActionNeeded> {
        match (&self.action_needed, action) {
            (ActionNeeded::GameOver { .. }, _) => Err(SorryError::GameAlreadyOver),

            (
                ActionNeeded::ChooseCard {
                    player,
                    hand,
                    legal_card_indices,
                },
                PlayerAction::ChooseCard { hand_index },
            ) => {
                let player = *player;
                let hand_size = hand.len();
                if hand_index >= hand_size {
                    return Err(SorryError::InvalidHandIndex {
                        got: hand_index,
                        hand_size,
                    });
                }
                if !legal_card_indices.contains(&hand_index) {
                    return Err(SorryError::IllegalCardChoice);
                }
                let chosen = self.hands[player.0 as usize].remove(hand_index);
                let turn = self
                    .current_turn
                    .as_mut()
                    .expect("turn in progress during ChooseCard");
                turn.actions.push(Action::ChooseCard {
                    hand_index,
                    card: chosen,
                });
                let legal = legal_moves(&*self.rules, &self.board, player, chosen);
                self.action_needed = ActionNeeded::ChooseMove {
                    player,
                    card: chosen,
                    legal_moves: legal,
                };
                Ok(&self.action_needed)
            }

            (ActionNeeded::ChooseMove { player, card, .. }, PlayerAction::PlayMove { mv }) => {
                let player = *player;
                let card = *card;
                {
                    let turn = self
                        .current_turn
                        .as_mut()
                        .expect("turn in progress during PlayMove");
                    let mut engine = Engine {
                        rules: &*self.rules,
                        board: &mut self.board,
                        rng: &mut self.rngs.deck,
                        deck: &mut self.deck,
                        discard: &mut self.discard,
                    };
                    engine.commit_play(player, card, &mv, turn)?;
                    if let Some(Action::Play { bumps, .. }) = turn.actions.last()
                        && let Some(last) = bumps.last()
                    {
                        self.last_bump_victim = Some(last.player);
                    }
                }
                self.on_play_committed(player, card)?;
                Ok(&self.action_needed)
            }

            (ActionNeeded::ChooseCard { .. }, PlayerAction::PlayMove { .. }) => {
                Err(SorryError::UnexpectedAction {
                    expected: "ChooseCard",
                    got: "PlayMove",
                })
            }
            (ActionNeeded::ChooseMove { .. }, PlayerAction::ChooseCard { .. }) => {
                Err(SorryError::UnexpectedAction {
                    expected: "PlayMove",
                    got: "ChooseCard",
                })
            }
        }
    }

    /// Compute (but do NOT apply) a bot's response to the currently pending
    /// decision. The server/WASM layer typically broadcasts the "bot is
    /// about to act" state and then calls `apply_action` in a second step.
    pub fn get_bot_action(&mut self, strategy: &dyn Strategy) -> Result<PlayerAction> {
        let needed = self.action_needed.clone();
        match needed {
            ActionNeeded::GameOver { .. } => Err(SorryError::GameAlreadyOver),
            ActionNeeded::ChooseCard { player, hand, .. } => {
                let view = self.build_strategy_view(player, None);
                let rng = &mut self.rngs.seats[player.0 as usize];
                let idx = strategy.choose_card(&view, &*self.rules, rng);
                let idx = idx.min(hand.len().saturating_sub(1));
                Ok(PlayerAction::ChooseCard { hand_index: idx })
            }
            ActionNeeded::ChooseMove {
                player,
                card,
                legal_moves: legal,
            } => {
                let view = self.build_strategy_view(player, Some(card));
                let rng = &mut self.rngs.seats[player.0 as usize];
                let mv = strategy.choose_move(&view, &*self.rules, card, &legal, rng);
                Ok(PlayerAction::PlayMove { mv })
            }
        }
    }

    pub fn get_full_state(&self) -> InteractiveGameState {
        let drawn_card = self.current_drawn_card_if_faceup();
        InteractiveGameState {
            num_players: self.history.num_players,
            current_player: self.current,
            starting_player: self.history.starting_player,
            pawn_positions: self.board.all_positions().to_vec(),
            discard: self.discard.clone(),
            deck_remaining: self.deck.len(),
            hands: self.hands.clone(),
            drawn_card,
            rules_name: self.history.rules_name.clone(),
            strategy_names: self.history.strategy_names.clone(),
            action_needed: self.action_needed.clone(),
            winners: self.history.winners.clone(),
            truncated: self.history.truncated,
            turn_count: self.history.turns.len(),
            current_turn: self.current_turn.clone(),
            seat_sides: self.rules.seat_sides(self.history.num_players),
        }
    }

    pub fn get_player_view(&self, viewer: PlayerId) -> PlayerView {
        self.redacted_view(Some(viewer))
    }

    /// Spectator view — all hands redacted, drawn card hidden. The embedded
    /// `viewer` field is set to `PlayerId(u8::MAX)` as a sentinel; callers
    /// driving this view track observer state out-of-band.
    pub fn get_observer_view(&self) -> PlayerView {
        self.redacted_view(None)
    }

    fn redacted_view(&self, viewer: Option<PlayerId>) -> PlayerView {
        // The face-up drawn card is public information in standard Sorry!
        // (`hand_size == 0`); every viewer sees it. In variant rules the
        // drawn card is already absorbed into the current player's hand and
        // `current_drawn_card_if_faceup` returns `None`, so redaction there
        // happens via the per-player `hands` slot below.
        let drawn_card = self.current_drawn_card_if_faceup();
        let hands: Vec<Option<Vec<Card>>> = self
            .hands
            .iter()
            .enumerate()
            .map(|(p, h)| match viewer {
                Some(v) if PlayerId(p as u8) == v => Some(h.clone()),
                _ => None,
            })
            .collect();
        let action_needed = match &self.action_needed {
            ActionNeeded::ChooseCard {
                player,
                hand: _,
                legal_card_indices,
            } if Some(*player) != viewer => ActionNeeded::ChooseCard {
                player: *player,
                hand: Vec::new(),
                legal_card_indices: legal_card_indices.clone(),
            },
            other => other.clone(),
        };
        PlayerView {
            viewer: viewer.unwrap_or(PlayerId(u8::MAX)),
            num_players: self.history.num_players,
            current_player: self.current,
            starting_player: self.history.starting_player,
            pawn_positions: self.board.all_positions().to_vec(),
            discard: self.discard.clone(),
            deck_remaining: self.deck.len(),
            hands,
            drawn_card,
            rules_name: self.history.rules_name.clone(),
            strategy_names: self.history.strategy_names.clone(),
            action_needed,
            winners: self.history.winners.clone(),
            truncated: self.history.truncated,
            turn_count: self.history.turns.len(),
            seat_sides: self.rules.seat_sides(self.history.num_players),
        }
    }

    // ─── Internal helpers ────────────────────────────────────────────

    fn on_play_committed(&mut self, player: PlayerId, card: Card) -> Result<()> {
        let won_mid_turn = self.rules.is_winner(&self.board, player);
        if !won_mid_turn && self.rules.grants_extra_turn(card) {
            let turn = self
                .current_turn
                .as_mut()
                .expect("turn in progress during extra-turn handoff");
            turn.actions.push(Action::ExtraTurnGranted);
            return self.draw_and_compute_decision();
        }
        // Finalize the turn.
        let turn = self.current_turn.take().expect("turn in progress at finalize");
        self.history.turns.push(turn);

        // Append the current player to the finish order if they just
        // finished (and aren't already recorded). `history.winners` now
        // acts as the ordered placement list: winners[0] finished first,
        // winners[1] second, etc.
        if won_mid_turn && !self.history.winners.contains(&player) {
            self.history.winners.push(player);
        }

        let num_players = self.history.num_players;
        if self.rules.game_over(&self.history.winners, num_players) {
            self.rules
                .finalize_winners(&mut self.history.winners, num_players);
            self.action_needed = ActionNeeded::GameOver {
                winners: self.history.winners.clone(),
                truncated: false,
            };
            return Ok(());
        }

        // Advance to the next unfinished player, skipping anyone already
        // in `winners` — Play Out treats finished players as if they
        // weren't at the table anymore (no draws, no turns).
        let n = num_players as u8;
        let mut next = (self.current.0 + 1) % n;
        for _ in 0..n {
            if !self
                .history
                .winners
                .iter()
                .any(|w| w.0 == next)
            {
                break;
            }
            next = (next + 1) % n;
        }
        self.current = PlayerId(next);
        self.begin_turn_or_game_over()
    }

    /// If max_turns has been reached, set truncated + GameOver. Otherwise
    /// open a fresh TurnRecord and auto-resolve the draw into a pending
    /// decision.
    fn begin_turn_or_game_over(&mut self) -> Result<()> {
        if self.history.turns.len() >= self.max_turns {
            self.history.truncated = true;
            self.action_needed = ActionNeeded::GameOver {
                winners: Vec::new(),
                truncated: true,
            };
            return Ok(());
        }
        self.current_turn = Some(TurnRecord {
            player: self.current,
            actions: Vec::new(),
        });
        self.draw_and_compute_decision()
    }

    /// Draw a card into `current_turn` (recording `Action::Draw` and any
    /// implicit `Action::Reshuffle`), then set `action_needed` to the next
    /// pending decision.
    fn draw_and_compute_decision(&mut self) -> Result<()> {
        let card = {
            let turn = self
                .current_turn
                .as_mut()
                .expect("turn in progress during draw");
            let mut engine = Engine {
                rules: &*self.rules,
                board: &mut self.board,
                rng: &mut self.rngs.deck,
                deck: &mut self.deck,
                discard: &mut self.discard,
            };
            engine.draw_card(turn)?
        };
        {
            let turn = self
                .current_turn
                .as_mut()
                .expect("turn in progress during draw post-record");
            turn.actions.push(Action::Draw { card });
        }
        let current = self.current;
        if self.rules.hand_size() == 0 {
            let legal = legal_moves(&*self.rules, &self.board, current, card);
            self.action_needed = ActionNeeded::ChooseMove {
                player: current,
                card,
                legal_moves: legal,
            };
        } else {
            self.hands[current.0 as usize].push(card);
            let hand = self.hands[current.0 as usize].clone();
            let legal_card_indices: Vec<usize> = (0..hand.len()).collect();
            self.action_needed = ActionNeeded::ChooseCard {
                player: current,
                hand,
                legal_card_indices,
            };
        }
        Ok(())
    }

    fn build_strategy_view(&self, player: PlayerId, drawn_card: Option<Card>) -> StrategyView {
        StrategyView {
            my_player: player,
            num_players: self.history.num_players,
            hand: self.hands[player.0 as usize].clone(),
            drawn_card,
            pawn_positions: self.board.all_positions().to_vec(),
            discard: self.discard.clone(),
            deck_remaining: self.deck.len(),
            current_player_turn: self.current,
            last_bump_victim: self.last_bump_victim,
        }
    }

    /// The face-up drawn card for standard rules, exposed on
    /// `InteractiveGameState::drawn_card` / `PlayerView::drawn_card` as a
    /// convenience separate from `action_needed`. In variant rules the
    /// drawn card is already in the hand and this returns `None`.
    fn current_drawn_card_if_faceup(&self) -> Option<Card> {
        if self.rules.hand_size() != 0 {
            return None;
        }
        match &self.action_needed {
            ActionNeeded::ChooseMove { card, .. } => Some(*card),
            _ => None,
        }
    }
}
