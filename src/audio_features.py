import numpy as np
import librosa


def extract_segment_features(
    audio,
    sr=22050,
    segment_duration=5,
    n_mels=128
):
    """Extract mean log-Mel features from complete audio segments."""
    samples_per_segment = int(sr * segment_duration)

    num_segments = len(audio) // samples_per_segment

    if num_segments == 0:
        return None

    features = []

    for segment_idx in range(num_segments):
        start = segment_idx * samples_per_segment
        end = start + samples_per_segment

        segment = audio[start:end]

        if len(segment) != samples_per_segment:
            continue

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_mels=n_mels,
            fmin=20,
            fmax=sr // 2,
            power=2.0
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        feature_vector = log_mel.mean(axis=1)
        features.append(feature_vector)

    if not features:
        return None

    return np.asarray(features, dtype=np.float32)


def load_audio(
    audio_path,
    sr=22050
):
    """Load an audio file as mono audio."""
    try:
        audio, loaded_sr = librosa.load(
            audio_path,
            sr=sr,
            mono=True
        )

        return audio, loaded_sr

    except Exception as exc:
        print(f"Failed to load audio: {audio_path}")
        print(f"Error: {exc}")
        return None, None


def extract_audio_segment_features(
    audio_path,
    sr=22050,
    segment_duration=5,
    n_mels=128
):
    """Load audio and extract segment-level log-Mel features."""
    audio, loaded_sr = load_audio(
        audio_path,
        sr=sr
    )

    if audio is None:
        return None

    return extract_segment_features(
        audio,
        sr=loaded_sr,
        segment_duration=segment_duration,
        n_mels=n_mels
    )
