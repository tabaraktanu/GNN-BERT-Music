import os
import pandas as pd


GENRE_TO_LABEL = {
    "Electronic": 0,
    "Experimental": 1,
    "Folk": 2,
    "Hip-Hop": 3,
    "Instrumental": 4,
    "International": 5,
    "Pop": 6,
    "Rock": 7,
}

LABEL_TO_GENRE = {
    label: genre
    for genre, label in GENRE_TO_LABEL.items()
}


def load_fma_metadata(metadata_path):
    """Load the FMA metadata CSV file."""
    return pd.read_csv(
        metadata_path,
        header=[0, 1],
        index_col=0
    )


def create_split_metadata(tracks, audio_root):
    """Create split metadata for tracks that have available audio files."""
    audio_files = []

    for root, _, files in os.walk(audio_root):
        for filename in files:
            if filename.lower().endswith(".mp3"):
                track_id = int(os.path.splitext(filename)[0])
                audio_files.append(track_id)

    audio_track_ids = set(audio_files)

    available_ids = tracks.index.intersection(audio_track_ids)

    split_metadata = tracks.loc[
        available_ids,
        [
            ("set", "split"),
            ("set", "subset"),
            ("track", "genre_top"),
            ("track", "title"),
        ]
    ].copy()

    split_metadata.columns = [
        "split",
        "subset",
        "genre",
        "title",
    ]

    return split_metadata


def get_official_split_ids(split_metadata):
    """Return official train, validation, and test track IDs."""
    train_ids = split_metadata.index[
        split_metadata["split"] == "training"
    ].tolist()

    validation_ids = split_metadata.index[
        split_metadata["split"] == "validation"
    ].tolist()

    test_ids = split_metadata.index[
        split_metadata["split"] == "test"
    ].tolist()

    return train_ids, validation_ids, test_ids
