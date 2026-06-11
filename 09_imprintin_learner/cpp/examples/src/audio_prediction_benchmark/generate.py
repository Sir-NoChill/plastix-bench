"""Generate audio prediction benchmark dataset.

Creates a continuous audio stream of instrument chords followed by delayed
scalar rewards, as described in Section 9.3 of Khurram Javed's thesis.
"""

import argparse
import json
import os
import random

import numpy as np
import pretty_midi
import soundfile as sf


INSTRUMENT_MAP = {
    "piano": 0,           # Acoustic Grand Piano
    "guitar": 25,         # Acoustic Guitar (steel)
    "electric_guitar": 27,# Electric Guitar (clean)
    "violin": 40,
    "trumpet": 56,
    "flute": 73,
    "organ": 19,
}

CHORD_MAP = {
    # Major chords
    "C_major": [60, 64, 67],
    "D_major": [62, 66, 69],
    "E_major": [64, 68, 71],
    "F_major": [65, 69, 72],
    "G_major": [67, 71, 74],
    "A_major": [69, 73, 76],
    "B_major": [71, 75, 78],
    # Minor chords
    "C_minor": [60, 63, 67],
    "D_minor": [62, 65, 69],
    "E_minor": [64, 67, 71],
    "A_minor": [69, 72, 76],
    # Seventh chords
    "C_major7": [60, 64, 67, 71],
    "D_minor7": [62, 65, 69, 72],
    "G7": [67, 71, 74, 77],
    # Single notes
    "C4": [60],
    "D4": [62],
    "E4": [64],
    "F4": [65],
    "G4": [67],
    "A4": [69],
    "B4": [71],
}


def parse_combinations(combo_strings):
    """Parse combination strings like 'guitar:C_major:1' into structured data."""
    combinations = []
    for s in combo_strings:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid combination format '{s}'. Expected 'instrument:chord:reward'"
            )
        instrument_name, chord_name, reward_str = parts
        if instrument_name not in INSTRUMENT_MAP:
            raise ValueError(
                f"Unknown instrument '{instrument_name}'. "
                f"Available: {list(INSTRUMENT_MAP.keys())}"
            )
        if chord_name not in CHORD_MAP:
            raise ValueError(
                f"Unknown chord '{chord_name}'. Available: {list(CHORD_MAP.keys())}"
            )
        combinations.append({
            "instrument": instrument_name,
            "program": INSTRUMENT_MAP[instrument_name],
            "chord_name": chord_name,
            "notes": CHORD_MAP[chord_name],
            "reward": float(reward_str),
        })
    return combinations


def build_event_timeline(duration, combinations, inter_sound_range, rng):
    """Build a timeline of chord events (reward times assigned after rendering)."""
    events = []
    current_time = inter_sound_range[0]  # Start after initial gap

    while current_time < duration:
        combo = combinations[rng.randint(0, len(combinations) - 1)]
        chord_duration = rng.uniform(0.8, 1.5)

        events.append({
            "chord_time": current_time,
            "chord_duration": chord_duration,
            "instrument": combo["instrument"],
            "program": combo["program"],
            "chord_name": combo["chord_name"],
            "notes": combo["notes"],
            "reward": combo["reward"],
        })

        gap = rng.uniform(*inter_sound_range)
        current_time += gap

    return events


def measure_sound_end(audio, chord_start, sample_rate, max_search=10.0,
                      window_sec=0.05, threshold_ratio=0.01):
    """Measure when a chord's audio actually goes silent.

    Computes short-time RMS energy from chord_start forward and finds the last
    point where energy exceeds a fraction of the peak.

    Args:
        audio: Full rendered audio array.
        chord_start: Chord onset time in seconds.
        sample_rate: Audio sample rate.
        max_search: Maximum seconds after chord_start to search.
        window_sec: RMS window size in seconds.
        threshold_ratio: Fraction of peak RMS below which audio is "silent".

    Returns:
        Time in seconds when the chord audio ends.
    """
    start_sample = int(chord_start * sample_rate)
    end_sample = min(len(audio), start_sample + int(max_search * sample_rate))
    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        return chord_start

    window_size = max(1, int(window_sec * sample_rate))
    n_windows = len(segment) // window_size
    if n_windows == 0:
        return chord_start

    # Compute RMS per window
    trimmed = segment[:n_windows * window_size].reshape(n_windows, window_size)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))

    peak_rms = rms.max()
    if peak_rms == 0:
        return chord_start

    threshold = peak_rms * threshold_ratio

    # Find last window above threshold
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return chord_start

    last_above = above[-1]
    return chord_start + (last_above + 1) * window_sec


def assign_reward_times(events, audio, sample_rate, reward_delay_range, duration, rng):
    """Assign reward times based on measured audio end for each chord.

    Events that would have reward_time >= duration are dropped.
    """
    result = []
    for event in events:
        sound_end = measure_sound_end(audio, event["chord_time"], sample_rate)
        reward_delay = rng.uniform(*reward_delay_range)
        reward_time = sound_end + reward_delay

        if reward_time >= duration:
            continue

        event["sound_end"] = sound_end
        event["reward_time"] = reward_time
        result.append(event)

    return result


def render_audio(events, duration, sample_rate, soundfont, rng):
    """Render all chord events to audio using pretty_midi + fluidsynth."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    # Group events by instrument program to create one MIDI instrument per type
    instruments_by_program = {}
    for event in events:
        prog = event["program"]
        if prog not in instruments_by_program:
            inst = pretty_midi.Instrument(program=prog)
            instruments_by_program[prog] = inst
            pm.instruments.append(inst)

    for event in events:
        inst = instruments_by_program[event["program"]]
        chord_duration = event["chord_duration"]
        is_guitar = "guitar" in event["instrument"]

        cumulative_strum = 0.0
        for i, note_pitch in enumerate(event["notes"]):
            velocity = rng.randint(70, 120)

            if is_guitar:
                # Guitar strum: cumulative delay per note
                jitter = rng.uniform(0.02, 0.06) if i > 0 else 0.0
                cumulative_strum += jitter
                note_offset = cumulative_strum
            else:
                # Piano: small random jitter per note
                note_offset = rng.uniform(0, 0.03)

            start = event["chord_time"] + note_offset
            end = start + chord_duration

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=note_pitch,
                start=start,
                end=end,
            )
            inst.notes.append(note)

    # Render to audio
    audio = pm.fluidsynth(fs=sample_rate, synthesizer=soundfont)

    # Pad or trim to exact duration
    target_samples = int(duration * sample_rate)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]

    return audio


def build_reward_signal(events, duration, sample_rate, step_size):
    """Build per-step reward signal array."""
    n_steps = int(duration * sample_rate) // step_size
    rewards = np.zeros(n_steps, dtype=np.float32)

    for event in events:
        step_idx = int(event["reward_time"] * sample_rate) // step_size
        if 0 <= step_idx < n_steps:
            rewards[step_idx] = event["reward"]

    return rewards


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio prediction benchmark dataset"
    )
    parser.add_argument(
        "--duration", type=int, default=3600,
        help="Total audio duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--inter-sound-time-range", type=float, nargs=2, default=[15, 30],
        metavar=("MIN", "MAX"),
        help="Min and max seconds between sounds (default: 15 30)"
    )
    parser.add_argument(
        "--reward-delay-range", type=float, nargs=2, default=[0.5, 2.0],
        metavar=("MIN", "MAX"),
        help="Min and max seconds between sound end and reward (default: 0.5 2.0)"
    )
    parser.add_argument(
        "--combinations", type=str, nargs="+",
        default=["guitar:C_major:1", "piano:C_major:-1", "piano:D_major:0"],
        help="Sound combinations as instrument:chord:reward (default: guitar:C_major:1 piano:C_major:-1 piano:D_major:0)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16384,
        help="Audio sample rate in Hz (default: 16384)"
    )
    parser.add_argument(
        "--step-size", type=int, default=640,
        help="Samples per time step (default: 640)"
    )
    parser.add_argument(
        "--soundfont", type=str,
        default=os.path.join(os.getcwd(), "FluidR3_GM.sf2"),
        help="Path to SoundFont file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Generating {args.duration}s audio benchmark dataset...")
    print(f"Sample rate: {args.sample_rate} Hz, step size: {args.step_size}")
    print(f"Combinations: {args.combinations}")

    # Parse combinations
    combinations = parse_combinations(args.combinations)

    # Build event timeline (chord times only, no reward times yet)
    events = build_event_timeline(
        duration=args.duration,
        combinations=combinations,
        inter_sound_range=args.inter_sound_time_range,
        rng=rng,
    )
    print(f"Generated {len(events)} chord events")

    # Render audio
    print("Rendering audio via fluidsynth (this may take a while)...")
    audio = render_audio(
        events=events,
        duration=args.duration,
        sample_rate=args.sample_rate,
        soundfont=args.soundfont,
        rng=rng,
    )
    print(f"Audio shape: {audio.shape} ({audio.shape[0] / args.sample_rate:.1f}s)")

    # Measure actual sound durations and assign reward times
    print("Measuring sound durations and assigning reward times...")
    events = assign_reward_times(
        events=events,
        audio=audio,
        sample_rate=args.sample_rate,
        reward_delay_range=args.reward_delay_range,
        duration=args.duration,
        rng=rng,
    )
    print(f"Assigned reward times to {len(events)} events")

    # Build reward signal
    rewards = build_reward_signal(
        events=events,
        duration=args.duration,
        sample_rate=args.sample_rate,
        step_size=args.step_size,
    )
    n_nonzero = np.count_nonzero(rewards)
    print(f"Reward signal: {len(rewards)} steps, {n_nonzero} non-zero rewards")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    audio_path = os.path.join(args.output_dir, "audio.wav")
    sf.write(audio_path, audio, args.sample_rate)
    print(f"Saved audio to {audio_path}")

    rewards_path = os.path.join(args.output_dir, "rewards.npy")
    np.save(rewards_path, rewards)
    print(f"Saved rewards to {rewards_path}")

    # Build metadata
    metadata = {
        "duration": args.duration,
        "sample_rate": args.sample_rate,
        "step_size": args.step_size,
        "inter_sound_time_range": args.inter_sound_time_range,
        "reward_delay_range": args.reward_delay_range,
        "combinations": args.combinations,
        "seed": args.seed,
        "soundfont": args.soundfont,
        "n_events": len(events),
        "n_steps": len(rewards),
        "events": [
            {
                "chord_time": e["chord_time"],
                "chord_duration": e["chord_duration"],
                "sound_end": e["sound_end"],
                "reward_time": e["reward_time"],
                "instrument": e["instrument"],
                "chord_name": e["chord_name"],
                "reward": e["reward"],
            }
            for e in events
        ],
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("Done!")


if __name__ == "__main__":
    main()
