# Brag Plan: atomic_az

## What is this app?
A training pipeline that teaches an AlphaZero-style neural net to play **Atomic Chess** — a variant where every capture detonates the adjacent pieces — on one H200 GPU, aiming for parity with Fairy-Stockfish, the reference atomic engine.

## The angle
This is a research project, not a product, but the *material* is unusually cinematic: pieces literally explode. Standard Stockfish cannot play this variant at all. The plan targets an ~8-rung strength ladder against a state-of-the-art opponent. That is a trailer, not a landing page.

The video treats the training run like a mission film: quiet gravitas → the constraint → the machine we built → the ladder we intend to climb → the name. Everything specific to *this* repo — measured baselines, exact hyperparameters, the exact opponent — appears on screen. No generic ML flavor language.

## Hook (first 2-3 seconds)
Black frame. One line in large serif: **"Standard Stockfish cannot play atomic chess."** Comes in fast, holds. That line is copy-pasted from `README.md:24`. It reads as a straight-faced constraint and earns the next 20 seconds.

## Key moments (the middle)
1. **The explosion** — a stylized atomic-chess capture on a compact board: one piece is taken, the 8-neighbor pieces vanish in a warm particle burst. Caption underneath in mono: `capture → explode → no undo`.
2. **The rig, stated as a spec card** — one card, one column, monospace: `AlphaZero · ResNet 128×10 · 300 sims/move · one H200 · scavenger · 24h · --requeue`. All values are real (from `train_atomic_az.sh` / `atomic_az.slurm` / PLAN.md §1.5).
3. **The ladder** — the six-rung strength ladder from PLAN.md §Phase 4 reveals one rung at a time, ending on `Rung 5 — Fairy-SF full, Use NNUE=false — parity — the milestone`. This is the arc of the whole project on one slide.

## Outro / punchline
Board dissolves. Title slams in on the strongest music cue: **`atomic_az`**. Under it, one line: *training an AlphaZero agent to reach parity with the reference atomic engine.*

## User flow worth showing
None — this is a training pipeline, not an interactive app. The centerpiece "flow" is the **strength-ladder ascent** from PLAN.md Phase 4, treated as sequential reveal in Scene 4. That is the closest thing this project has to a happy path — random init → beat skill-0 → climb rungs → parity.

## Tone
- Preset: **cinematic**
- Creative direction: research trailer — dark board, warm-orange explosion light, dramatic serif headlines over monospace data. The gravitas is real; the drama is the science, not a joke.
- Interpretation: 4–5 scenes, 3–5s each. Big type, long-ish holds, hard-cut or wipe transitions on the strongest music beats. Restraint over noise — one explosion, one ladder, one slam.

## Format: landscape — 1920x1080
## Duration: ~22 seconds (target 20-22)

## Visual identity (from the project)
No CSS in the repo (SLURM scripts, C++ evals, markdown). Palette and typography are chosen to match the *concept* — atomic capture on a chess board:

- Background: `#0A0A0F` (near-black, deep space; slight blue undertone)
- Board dark squares: `#1C1D22`
- Board light squares: `#2A2C33`
- Accent (explosion / fission): `#FF7A2B` (warm amber-orange)
- Accent secondary (GPU / tech): `#4EC8FF` (cool cyan, sparingly for data cards)
- Text primary: `#F5F1E8` (warm off-white, slight paper tint for the serif)
- Text muted (mono data): `#9AA0A6`
- Rung ladder background: subtle gradient `#0A0A0F` → `#12111A`

Display font: **Instrument Serif** (dramatic Italian trailer feel — appropriate for cinematic tone; loadable from Google Fonts).
Body / data font: **JetBrains Mono** (research/spec-card feel; loadable from Google Fonts).

Strongest visual element: a stylized atomic-chess capture — a compact board tile with a piece exploding into warm-orange particles that briefly bloom out to the 8 neighboring squares. Grounded in the actual game rule.

## Share copy (draft)
> Training AlphaZero to play atomic chess — the variant standard Stockfish literally cannot play — on one H200. Chasing parity with Fairy-Stockfish (the reference engine).

## Audio direction
- Role: **cinematic support** — one steady bed with a low swell, restrained motion-matched accents on the two big beats (the explosion, the outro slam).
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (bundled). Steady, clean, ~110 BPM — matches cinematic per `audio.md`.
- Music treatment: start at 0.0s at volume 0.32, gentle fade-in over 0.6s. Hold across the video. Fade out over 0.8s during outro so the slam and the title land cleanly.
- Audio-reactive treatment: **subtle**. Use music RMS to make the board's ambient glow breathe, and let the accent color of the outro title get a soft treble halo. No waveform bars, no equalizer, no strobing.
- SFX posture: **sparse, motion-matched**. Three moments carry SFX: (a) the explosion in Scene 2, (b) the sequential ladder rungs in Scene 4, (c) the outro title slam in Scene 5. Everything else lets the music carry.
- Audio-coupled moments: the atomic explosion (impact + slight glass shatter feel), each ladder rung arriving one by one (soft card/drop), the final title slam (deep bell).
- Restraint rule: no SFX under Scene 1's hook line — the silence before "cannot play atomic chess" is part of the punch. No stacked SFX under narration text. Music never above 0.35.

## Music cue guidance
- Track: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (109.96 BPM, cinematic-friendly per audio.md).
- Preset available (`assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.md`).
- Strong-cue locks to target (~±0.15s):
  - **~8.74s** — end of the explosion moment / start of the spec-card reveal.
  - **~17.47s or 18.56s** — the final ladder rung (Rung 5, parity) lands.
  - **~22.93s** — the outro title slam.
- Beat-grid windows for sequential reveals in Scene 4 (the ladder): the 6 rungs should arrive on beats between ~13.11s and ~18.56s (roughly every second beat: 13.11, 14.20, 15.29, 16.38, 17.47, 18.56 — six rung entries, one per).
- Restraint note: cinematic tone, not chaotic — target only the three strong cues above and the ladder beat window; do not sync every element to a beat.

## Storyboard

### Scene 1 — The constraint — 4.0s (0.0–4.0)
Full-bleed black. Instrument Serif, ~110pt, `#F5F1E8`, centered.
Line 1 (0.2–0.4s in, holds through end of scene): **"Standard Stockfish cannot play atomic chess."**
Slight ambient warm glow behind the text pulses gently with music RMS. No product material yet — this is the setup.
Sequential/interaction: none.
Audio intent: quiet, mysterious. Music has just started and is climbing. Let the line breathe. No SFX at all.
Audio-coupled idea: none — the line's power comes from silence around it.
Music: vol-12, fade-in from 0.0s, volume 0.32.
Transition mood: **dramatic wipe** (dark-to-board reveal) → Scene 2. Wipe starts ~3.7s.

### Scene 2 — Atomic — 4.5s (4.0–8.5)
A compact chess-board fragment (5x5 tiles) fills center screen, dark squares `#1C1D22`, light `#2A2C33`. Two pieces visible: a black knight moves into a white pawn's square. On capture (~5.2s), a warm-orange particle burst detonates outward — the capturing piece AND all 8 neighbors dissolve into embers in ~0.4s. Board is left with a scorched empty middle.
Under the board, a mono line fades in at 6.0s and holds: `capture → explode → no undo` (JetBrains Mono ~24pt, `#9AA0A6`).
Sequential/interaction: **yes** — knight glides into pawn (~0.5s), explosion (~0.4s), embers linger (~1.5s), caption fades (~1.2s hold + fade).
Audio intent: build tension into a satisfying detonation, then quiet. Land the explosion tail on the strong cue at **~8.74s** so the music beat reinforces the aftermath.
Audio-coupled idea: **capture explosion** — one impact SFX at ~5.2s (heavy soft impact or a bell shatter feel — Hyperframes to select from `impact/impactBell_heavy_*` or `impact/impactGlass_heavy_002` per sfx-analysis).
Music: vol-12, holding at 0.32.
Transition mood: **hard cut** → Scene 3 at ~8.5s (aligns with the strong cue).

### Scene 3 — The rig — 5.0s (8.5–13.5)
Dark background. Centered, a single spec card: a thin border in `#4EC8FF` at ~15% opacity, JetBrains Mono lines stacked. Above the card, small mono label in muted grey: `THE STACK`.
Card contents (each line reveals with a 0.2s drop-in stagger, all done by 10.5s):
```
AlphaZero · self-play RL
ResNet 128 × 10  ≈ 6M params
300 sims / move
replay 2^18 · reuse 4
one H200 · scavenger · 24h
--requeue · resume-safe
```
Hold the full card static from 10.5s to 13.3s so it can be read (six lines × ~0.3s ≈ 1.8s minimum read; we give 2.8s of settled time).
Sequential/interaction: **yes** — six card lines arrive one by one on beats between ~8.74s and ~10.5s, then hold as a full card.
Audio intent: professional and confident — the rig is real, the numbers are real. No music change; a very soft drop on each line, then quiet during the hold so the music carries.
Audio-coupled idea: **staggered line arrivals** — one very quiet `interface/drop_*` per line, only the first and last accented at slightly higher volume (Hyperframes to select restrained set from sfx-analysis).
Music: vol-12, unchanged at 0.32.
Transition mood: **soft crossfade** → Scene 4 at ~13.3s (crossing the ~13.11s strong beat).

### Scene 4 — The ladder — 5.5s (13.3–18.8)
Dark background, subtle vertical gradient. Left-aligned in the center-left third of the frame, small mono label at top-left of the ladder area: `STRENGTH LADDER`. To the right (or below) the label, six rungs stack in vertically as horizontal rows, arriving one at a time on beats. Each rung: a small mono index (`R0`, `R1`, ...) in cyan, a serif opponent name, and a target result in mono, on one line:

```
R0   rollout MCTS 300 sims          target > 80%
R1   rollout MCTS 3000 sims         target > 60%
R2   Fairy-SF · UCI_Elo 1400        target > 50%
R3   Fairy-SF · UCI_Elo 1800        target > 50%
R4   Fairy-SF · UCI_Elo 2200        target > 40%
R5   Fairy-SF full · Use NNUE=false PARITY — MILESTONE
```

Rung arrival times (aligned to the vol-12 beat grid, ~0.5-0.6s apart with a slightly longer hold on the last):
- R0: 13.11s
- R1: 14.20s
- R2: 15.29s
- R3: 16.38s
- R4: 17.47s (strong beat — subtle accent)
- R5: 18.56s (strong beat — largest visual emphasis: R5 row is 1.3x scale, `#FF7A2B` accent color for the word `PARITY`, brief soft-orange glow behind the row lasting ~0.4s)

Sequential/interaction: **yes** — six rungs arriving one by one on the beat grid, culminating on the strong beat at 18.56s.
Audio intent: build tension across the ladder, then land the final rung on a strong beat with a mild swell. This is the arc of the entire project on one slide — the pacing has to feel earned, not rushed.
Audio-coupled idea: **rung-by-rung reveal** — one very soft `interface/drop_*` or `casino/card-place-*` per rung R0-R4 (Hyperframes to pick a coherent set from sfx-analysis). R5 gets a bigger cue — a `impact/impactBell_heavy_*` at 18.56s reinforced by a brief RMS-driven glow bloom on the row.
Music: vol-12, unchanged at 0.32; consider a tiny RMS-driven brightening on the ladder background as the rungs stack.
Transition mood: **crossfade** → Scene 5 at ~18.7s (holds the strong beat visual for ~0.3s before fading).

### Scene 5 — The name — 3.5s (18.8–22.3)
Dark background. At 19.2s, a small mono line fades in in the top center: `training run · 2026`. At ~22.3s (targeting the **~22.93s** strong cue with a small ~0.6s early bias to feel deliberate rather than late — but the visible slam should be **on** the strong beat, so we let the title enter at 22.3s and land its full weight ON 22.93s), the title slams in:

```
atomic_az
```

Instrument Serif italic, ~180pt, `#F5F1E8` with a soft warm halo behind it that briefly blooms with the strong beat. Below the title, one serif line at ~28pt, `#9AA0A6`: *training an AlphaZero agent to reach parity with the reference atomic engine.*

Sequential/interaction: none — a single decisive title reveal.
Audio intent: cinematic payoff. Music fades from 0.32 → 0.0 across 0.8s starting at ~22.3s so the slam sits in the air. A single deep bell cue reinforces the title landing on the beat.
Audio-coupled idea: **title slam** — `impact/impactBell_heavy_000` at 22.9s (Hyperframes to select from `impact/impactBell_heavy_*` per sfx-analysis).
Music: vol-12 fading out during the slam.
Transition mood: **hold**, then fade to black at ~22.3s end.

**Music mood for this video:** cinematic — steady bed, gentle audio-reactive breathing, three punctuation cues.
**Audio summary:** vol-12 fades in under a silent hook, holds under the atomic capture and the spec card, subtly reinforces each strength-ladder rung, and fades out under a bell-slammed title on the strongest beat in the 25s window.
