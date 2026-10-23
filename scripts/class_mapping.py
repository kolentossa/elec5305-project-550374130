"""Mapping between the 10 URBAN-SED target classes and YAMNet's 521 AudioSet classes."""

URBAN_SED_MAPPING = {
    "air_conditioner": {
        "primary_index": 407,
        "primary_name": "Air conditioning",
        "indices": [407],
        "names": ["Air conditioning"],
        "alignment": "exact",
        "note": "Direct 1-to-1 match with AudioSet 'Air conditioning' (index 407).",
    },
    "car_horn": {
        "primary_index": 302,
        "primary_name": "Vehicle horn, car horn, honking",
        "indices": [302, 312],
        "names": ["Vehicle horn, car horn, honking", "Air horn, truck horn"],
        "alignment": "exact",
        "note": "Primary match at index 302; index 312 covers heavy-vehicle air horns.",
    },
    "children_playing": {
        "primary_index": 66,
        "primary_name": "Children playing",
        "indices": [66, 10],
        "names": ["Children playing", "Children shouting"],
        "alignment": "exact",
        "note": "Direct match at index 66; outdoor playground clips also activate index 10.",
    },
    "dog_bark": {
        "primary_index": 70,
        "primary_name": "Bark",
        "indices": [70, 69],
        "names": ["Bark", "Dog"],
        "alignment": "exact",
        "note": "Direct match with 'Bark' (index 70) and parent class 'Dog' (index 69).",
    },
    "drilling": {
        "primary_index": 419,
        "primary_name": "Drill",
        "indices": [419],
        "names": ["Drill"],
        "alignment": "exact",
        "note": "Direct 1-to-1 match with AudioSet 'Drill' (index 419).",
    },
    "engine_idling": {
        "primary_index": 346,
        "primary_name": "Idling",
        "indices": [346, 337, 342, 343],
        "names": [
            "Idling",
            "Engine",
            "Medium engine (mid frequency)",
            "Heavy engine (low frequency)",
        ],
        "alignment": "hierarchical",
        "note": (
            "AudioSet includes 'Idling' (346), but stationary vehicle engines frequently "
            "distribute probability across parent 'Engine' (337) and low/mid-frequency "
            "engine nodes (342, 343)."
        ),
    },
    "gun_shot": {
        "primary_index": 421,
        "primary_name": "Gunshot, gunfire",
        "indices": [421, 422],
        "names": ["Gunshot, gunfire", "Machine gun"],
        "alignment": "exact",
        "note": "Direct match with 'Gunshot, gunfire' (index 421).",
    },
    "jackhammer": {
        "primary_index": 414,
        "primary_name": "Jackhammer",
        "indices": [414],
        "names": ["Jackhammer"],
        "alignment": "exact",
        "note": "Direct 1-to-1 match with AudioSet 'Jackhammer' (index 414).",
    },
    "siren": {
        "primary_index": 390,
        "primary_name": "Siren",
        "indices": [390, 317, 318, 319, 391],
        "names": [
            "Siren",
            "Police car (siren)",
            "Ambulance (siren)",
            "Fire engine, fire truck (siren)",
            "Civil defense siren",
        ],
        "alignment": "exact",
        "note": "Direct match with 'Siren' (390) and emergency vehicle siren subclasses.",
    },
    "street_music": {
        "primary_index": 132,
        "primary_name": "Music",
        "indices": [132, 133, 249],
        "names": ["Music", "Musical instrument", "Vocal music"],
        "alignment": "broad_proxy",
        "note": (
            "YAMNet's 521-class AudioSet ontology has no dedicated 'street_music' node; "
            "it is mapped via the general 'Music' node (132) and 'Musical instrument' (133)."
        ),
    },
}


def extract_class_scores(scores_521, classes=None, aggregation="primary"):
    """Extract target urban class probabilities from a (N, 521) YAMNet score matrix.

    aggregation:
      - 'primary': use only the single primary AudioSet index per class.
      - 'max': take the elementwise maximum across the mapped AudioSet subclass indices.
    """
    import numpy as np

    if classes is None:
        classes = list(URBAN_SED_MAPPING.keys())
    cols = []
    for label in classes:
        spec = URBAN_SED_MAPPING[label]
        if aggregation == "primary":
            cols.append(scores_521[:, spec["primary_index"]])
        elif aggregation == "max":
            cols.append(scores_521[:, spec["indices"]].max(axis=1))
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation}")
    return np.column_stack(cols)
