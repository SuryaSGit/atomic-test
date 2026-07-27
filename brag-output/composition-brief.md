# Hyperframes Composition Brief: atomic_az

## Objective
Create a short, cinematic research-trailer brag video for `atomic_az` — a training pipeline that teaches an AlphaZero-style agent to play Atomic Chess against Fairy-Stockfish, the reference atomic engine.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: **landscape — 1920x1080**
- Duration: **~22 seconds** (target 20-22)

## Source Material
- Project root: `/Users/surya/atomic_az`
- Primary files read: `README.md`, `PLAN.md`, `train_atomic_az.sh`, `atomic_az.slurm`
- Product name: `atomic_az`
- Tagline / strongest claim: **"Standard Stockfish cannot play atomic chess."** (verbatim from `README.md:24`)
- Key visual moments to recreate:
  - A stylized **atomic-chess capture** (one piece capturing another; capturing piece + 8 neighbors explode in warm particles). This is the defining rule of the variant — it must appear on screen.
  - A monospace **spec card** with the real hyperparameters/rig.
  - The **strength ladder** (six rungs, R0 → R5) revealed one rung at a time, culminating in "Fairy-SF full · Use NNUE=false · PARITY — MILESTONE".
  - The **`atomic_az` title slam** on the strongest music cue.
- Copy that must appear verbatim:
  - `Standard Stockfish cannot play atomic chess.`
  - `capture → explode → no undo`
  - Spec card lines (all six):
    - `AlphaZero · self-play RL`
    - `ResNet 128 × 10  ≈ 6M params`
    - `300 sims / move`
    - `replay 2^18 · reuse 4`
    - `one H200 · scavenger · 24h`
    - `--requeue · resume-safe`
  - Ladder rungs (all six):
    - `R0   rollout MCTS 300 sims          target > 80%`
    - `R1   rollout MCTS 3000 sims         target > 60%`
    - `R2   Fairy-SF · UCI_Elo 1400        target > 50%`
    - `R3   Fairy-SF · UCI_Elo 1800        target > 50%`
    - `R4   Fairy-SF · UCI_Elo 2200        target > 40%`
    - `R5   Fairy-SF full · Use NNUE=false PARITY — MILESTONE`
  - Outro title: `atomic_az`
  - Outro subtitle: `training an AlphaZero agent to reach parity with the reference atomic engine.`

## Creative Direction
- Tone preset: **cinematic**
- Creative direction: research trailer — dark board, warm-orange explosion light, dramatic serif headlines over monospace data. The gravitas is real; the drama is the science, not a joke.
- Interpretation: 5 scenes, ~3-5s each. Big type, comfortable holds, hard-cut or dramatic wipe transitions aligned to the strongest music beats. Restraint over noise — one explosion, one ladder, one slam.
- Angle: This project has unusually cinematic material — pieces literally explode, and the goal is to challenge the state-of-the-art atomic engine on one H200. Treat it like a mission trailer: quiet gravitas → the constraint → the machine → the ladder → the name. No jokes, no winking, no generic ML flavor text.
- Hook: Black screen, giant serif line "Standard Stockfish cannot play atomic chess." Holds ~3 seconds.
- Outro / punchline: `atomic_az` slams in on the strongest beat (~22.93s), music fades under it.
- Avoid:
  - Generic SaaS / ML marketing language ("streamline your workflow", "next-gen AI")
  - Abstract particle-noise or generic motion-graphic filler unrelated to atomic chess
  - Any "loss curves going down" cliché — we have not trained yet, and the claim is honesty about the target, not fabricated progress
  - Waveform bars, equalizers, generic visualizer graphics
  - Overusing beat sync — the tone is cinematic, not chaotic

## Visual Identity
No CSS existed in the source project (SLURM/C++/markdown). Palette chosen to match the concept:

- Background: `#0A0A0F` (near-black, faint blue undertone)
- Board dark squares: `#1C1D22`
- Board light squares: `#2A2C33`
- Accent (explosion / fission): `#FF7A2B` (warm amber-orange)
- Accent secondary (data / tech): `#4EC8FF` (cool cyan, sparingly)
- Text primary: `#F5F1E8` (warm off-white, paper tint)
- Text muted (mono data): `#9AA0A6`
- Ladder area background: subtle vertical gradient `#0A0A0F` → `#12111A`

- Display font: **Instrument Serif** (Google Fonts) — headlines and title. Italic variant for the outro title.
- Body / data font: **JetBrains Mono** (Google Fonts) — spec card, ladder, all data lines.

Visual references from the project:
- Atomic-chess capture rule (README.md's core constraint — "pieces explode")
- The strength ladder from PLAN.md §Phase 4 (verbatim rungs)
- Hyperparameters from `train_atomic_az.sh` and PLAN.md §1.5
- The "Fairy-Stockfish, Use NNUE=false" milestone framing from PLAN.md

## Storyboard
Use `brag-output/brag-plan.md` as the creative contract — full scene detail lives there. Summary:

1. **The constraint — 4.0s (0.0–4.0)** — Full-bleed black, serif line "Standard Stockfish cannot play atomic chess." holds for the scene. Subtle RMS-driven ambient warm glow behind the text.
2. **Atomic — 4.5s (4.0–8.5)** — Compact 5x5 chessboard fragment. Knight captures pawn (~5.2s); capturing piece + 8 neighbors detonate in warm-orange particles. Caption `capture → explode → no undo` fades in at 6.0s. Land the explosion tail on the strong cue at ~8.74s. Impact SFX on capture.
3. **The rig — 5.0s (8.5–13.5)** — Centered spec card with cyan-tinted thin border, JetBrains Mono. Six lines stagger in on beats between ~8.74s and ~10.5s; then the full card holds ~2.8s. Soft drop SFX per line (Hyperframes to select coherent set); no music change.
4. **The ladder — 5.5s (13.3–18.8)** — Six ladder rungs stack in vertically, one per beat, on the beat grid 13.11 / 14.20 / 15.29 / 16.38 / 17.47 / 18.56s. R5 arrives on the strong beat at ~18.56s at 1.3x scale with a warm-orange glow behind it. Card/drop SFX for R0-R4; a bell for R5.
5. **The name — 3.5s (18.8–22.3)** — Small mono line `training run · 2026` fades in top center at 19.2s. Title `atomic_az` slams in on the strong beat at ~22.93s (Instrument Serif italic, ~180pt, warm halo bloom). Subtitle *training an AlphaZero agent to reach parity with the reference atomic engine.* below. Music fades to 0 over 0.8s; one deep bell reinforces the slam.

## Audio
- Audio role: **cinematic support** — one steady bed with a low swell, restrained motion-matched accents on the two big beats (the explosion, the outro slam) and a soft rung-by-rung layer.
- Audio arc:
  - Silence at 0.0s.
  - Music fade-in over 0.6s to volume 0.32 starting at 0.0s (under Scene 1 hook line — quiet, no SFX).
  - Sustains at 0.32 through Scenes 2-4 with the explosion SFX and quiet ladder drops on top.
  - Fades from 0.32 → 0.0 over 0.8s during Scene 5, starting at ~22.3s, so the title slam sits in the air.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (bundled — steady/clean, ~110 BPM, cinematic-friendly per `audio.md`).
- Music treatment: fade in 0.6s from 0.0s at 0.32, hold, fade out 0.8s over Scene 5. Never above 0.35.
- Music cue guidance: **preset available** at `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.md` (also .json). Rich schema with `beats` and `strongCues`. Target three strong-cue locks: **~8.74s** (Scene 2 → Scene 3 hard cut, end of explosion), **~17.47s or 18.56s** (Scene 4 R5 landing), **~22.93s** (Scene 5 title slam). Beat-grid window 13.11-18.56s for the six ladder rungs (one rung per beat, ~1.1s apart — comfortably above the ~0.6s/word reading floor for one-line rows).
- Audio-reactive treatment: **subtle**. Use music RMS to make the ambient warm glow in Scene 1, the board glow in Scene 2, and the halo behind the outro title in Scene 5 breathe. Use a bass band to gently emphasize the spec card border in Scene 3 and the ladder background wash in Scene 4. **Absolutely no** waveform bars, equalizer graphics, musical-note icons, strobing, or heavy pulsing.
- Audio-coupled moments:
  - Scene 2 explosion (~5.2s) — one motion-matched impact hit on the capture.
  - Scene 3 spec-card lines (~8.74s–10.5s) — one very soft drop per line, only first and last slightly accented, most quiet.
  - Scene 4 ladder rungs (13.11 / 14.20 / 15.29 / 16.38 / 17.47 / 18.56s) — one card-place-style drop per rung, with R5 upgraded to a bell.
  - Scene 5 title slam (~22.93s) — one deep bell.
- SFX selection guidance:
  - Explosion: prefer `impact/impactBell_heavy_004` or `impact/impactGlass_heavy_002` layered with a soft `impact/impactSoft_heavy_*` for weight (Hyperframes to choose the best-timed variants per `sfx-analysis.md`). Volume 0.65-0.75.
  - Spec-card drops: `interface/drop_001`/`drop_002` or `interface/select_008`. Volume 0.35-0.45 (very restrained).
  - Ladder rungs (R0-R4): `casino/card-place-1` through `-4` or `interface/drop_*`. Volume 0.45-0.55.
  - Ladder R5 rung: `impact/impactBell_heavy_003` or `_004`. Volume 0.7.
  - Title slam: `impact/impactBell_heavy_000` or `_004`. Volume 0.75.
- SFX analysis guidance: `~/.claude/plugins/cache/brag/brag/0.2.2/skills/brag/assets/sfx/sfx-analysis.md` and `.json`. Prefer low/medium HF-risk files for the repeated ladder rungs and spec-card drops. High-risk files acceptable only for the singular explosion and title-slam moments.
- Exact SFX choice: Hyperframes selects filenames, exact timestamps, density, and volume based on the implemented animation. This brief provides intent, not a fixed cue sheet.
- Audio files to copy into `brag-output/composition/assets/`:
  - **Music (required):** `~/.claude/plugins/cache/brag/brag/0.2.2/skills/brag/assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` → `assets/music/`
  - **Music cues (recommended, for beat sync):** `~/.claude/plugins/cache/brag/brag/0.2.2/skills/brag/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.{json,md}` → `assets/music/cues/`
  - **SFX (Hyperframes selects):** from `~/.claude/plugins/cache/brag/brag/0.2.2/skills/brag/assets/sfx/{impact,interface,casino}/` → `assets/sfx/<family>/`

## Hyperframes Instructions

Load the composition-building Hyperframes domain skills — `hyperframes-core` (composition contract + `data-*` timing), `hyperframes-animation` (motion), `hyperframes-creative` (design spec, beats, audio-reactive), `hyperframes-keyframes` (seek-safe keyframes), and `hyperframes-cli` (lint/check/render). `/brag` is its own workflow: **do not enter the `hyperframes` entry-point intent interview and do not route into its generic promo / launch-video workflow.** Prefer native Hyperframes conventions over anything in `/brag`.

Requirements:
- Show the atomic-chess capture visually (Scene 2) — this is the one visual element from the project's *concept* that must appear.
- Show at least four verbatim copy elements from the source: the hook line, the "capture → explode → no undo" caption, the spec-card lines, the six ladder rungs, and the `atomic_az` title.
- Keep all text readable in the final render — reading-time floor: short label ~0.8s settled, sentence ~0.3s/word (the hook headline needs the longest hold). Scene 3's spec card holds ~2.8s after entry, and Scene 4's ladder rungs arrive one per beat (~1.1s apart at this tempo) — comfortably above the floor for one-line rows.
- Total duration 20-22 seconds. Do not exceed 25.
- Include the planned music/SFX layer. Music at ≤0.35 volume; fade out under the outro.
- Beat lock the three strong-cue moments (Scene 2 → 3 cut at ~8.74s; Scene 4 R5 at ~17.47 or 18.56s; Scene 5 title slam at ~22.93s), each within ±0.15s. Snap the six ladder rungs to consecutive beat-grid timestamps (±0.10s each). Mark them with `// beat-locked` and `// beat-grid` comments.
- Use SFX to support motion: impact on the explosion, soft drops on the sequential card/rung reveals, a bell for the R5 rung and the title slam.
- Honor the music fade-out at the outro so the final bell rings clearly.
- Extract audio data and wire at least one visual element to it per the `hyperframes-creative` audio-reactive workflow. Good targets: Scene 1 ambient glow, Scene 2 board glow, Scene 4 ladder background wash, Scene 5 outro title halo. Subtle only.
- All audio and font/media assets local to the composition directory. No absolute paths.
- Run `npx hyperframes check` — this is `/brag`'s single gate before render.
