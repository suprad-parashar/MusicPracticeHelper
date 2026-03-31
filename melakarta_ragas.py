"""
All 72 Melakarta ragas with names, aliases, and algorithmic swara computation.

The Melakarta system is a framework of 72 parent ragas in Carnatic music.
Swaras are computed from the melakarta number using the Katapayadi sankhya system.
"""

SWARA_NAMES = {
    "S": "Shadjam",
    "R1": "Shuddha Rishabham",
    "R2": "Chatushruti Rishabham",
    "R3": "Shatshruti Rishabham",
    "G1": "Shuddha Gandharam",
    "G2": "Sadharana Gandharam",
    "G3": "Antara Gandharam",
    "M1": "Shuddha Madhyamam",
    "M2": "Prati Madhyamam",
    "P": "Panchamam",
    "D1": "Shuddha Dhaivatam",
    "D2": "Chatushruti Dhaivatam",
    "D3": "Shatshruti Dhaivatam",
    "N1": "Shuddha Nishadam",
    "N2": "Kaisiki Nishadam",
    "N3": "Kakali Nishadam",
}

RI_GA_COMBINATIONS = [
    ("R1", "G1"),
    ("R1", "G2"),
    ("R1", "G3"),
    ("R2", "G2"),
    ("R2", "G3"),
    ("R3", "G3"),
]

DA_NI_COMBINATIONS = [
    ("D1", "N1"),
    ("D1", "N2"),
    ("D1", "N3"),
    ("D2", "N2"),
    ("D2", "N3"),
    ("D3", "N3"),
]


def compute_swaras(melakarta_number: int) -> dict:
    """Compute the swaras for a melakarta raga from its number (1-72)."""
    if not 1 <= melakarta_number <= 72:
        raise ValueError(f"Melakarta number must be 1-72, got {melakarta_number}")

    idx = melakarta_number - 1
    ma = "M1" if idx < 36 else "M2"
    group_idx = idx % 36
    ri_ga_idx = group_idx // 6
    da_ni_idx = group_idx % 6

    ri, ga = RI_GA_COMBINATIONS[ri_ga_idx]
    da, ni = DA_NI_COMBINATIONS[da_ni_idx]

    arohana = ["S", ri, ga, ma, "P", da, ni, ">S"]
    avrohana = [">S", ni, da, "P", ma, ga, ri, "S"]

    return {
        "arohana": arohana,
        "avrohana": avrohana,
        "arohana_str": " ".join(arohana),
        "avrohana_str": " ".join(avrohana),
        "ma": ma,
        "ri": ri,
        "ga": ga,
        "da": da,
        "ni": ni,
    }


MELAKARTA_RAGAS = {
    1: {"name": "Kanakangi", "aliases": ["Kanakāngi"]},
    2: {"name": "Ratnangi", "aliases": ["Ratnāngi"]},
    3: {"name": "Ganamurthi", "aliases": ["Gānamūrti", "Ganamurti"]},
    4: {"name": "Vanaspati", "aliases": ["Vanaspathi"]},
    5: {"name": "Manavati", "aliases": ["Mānavati", "Manavathi"]},
    6: {"name": "Tanarupi", "aliases": ["Tānarūpi"]},
    7: {"name": "Senavati", "aliases": ["Senāvati", "Senavathi"]},
    8: {"name": "Hanumatodi", "aliases": ["Hanumattodi", "Hanumatodi"]},
    9: {"name": "Dhenuka", "aliases": ["Dhēnuka"]},
    10: {"name": "Natakapriya", "aliases": ["Nātakapriya"]},
    11: {"name": "Kokilapriya", "aliases": ["Kōkilapriya"]},
    12: {"name": "Rupavati", "aliases": ["Rūpavati", "Rupavathi"]},
    13: {"name": "Gayakapriya", "aliases": ["Gāyakapriya"]},
    14: {"name": "Vakulabharanam", "aliases": ["Vakulābharanam"]},
    15: {"name": "Mayamalavagowla", "aliases": ["Māyāmālavagowla", "Mayamalavagaula"]},
    16: {"name": "Chakravakam", "aliases": ["Chakravāka", "Chakravakam"]},
    17: {"name": "Suryakantam", "aliases": ["Sūryakānta", "Suryakantham"]},
    18: {"name": "Hatakambari", "aliases": ["Hatakāmbari", "Hatakambhari"]},
    19: {"name": "Jhankaradhwani", "aliases": ["Jhankāradhvani"]},
    20: {"name": "Natabhairavi", "aliases": ["Natabhairavi"]},
    21: {"name": "Keeravani", "aliases": ["Kīravāni", "Kiravani"]},
    22: {"name": "Kharaharapriya", "aliases": ["Kharaharapriya"]},
    23: {"name": "Gourimanohari", "aliases": ["Gaurimanohari", "Gowrimanohari"]},
    24: {"name": "Varunapriya", "aliases": ["Varunapriya"]},
    25: {"name": "Mararanjani", "aliases": ["Māraranjani"]},
    26: {"name": "Charukesi", "aliases": ["Charukēshi", "Charukeshi"]},
    27: {"name": "Sarasangi", "aliases": ["Sarasāngi"]},
    28: {"name": "Harikambhoji", "aliases": ["Harikāmbhōji"]},
    29: {"name": "Dheerasankarabharanam", "aliases": ["Shankarabharanam", "Dhīrasankarābharanam", "Shankarabharana"]},
    30: {"name": "Naganandini", "aliases": ["Nāganandini", "Naganandhini"]},
    31: {"name": "Yagapriya", "aliases": ["Yāgapriya"]},
    32: {"name": "Ragavardhini", "aliases": ["Rāgavardhini", "Ragavardhani"]},
    33: {"name": "Gangeyabhushani", "aliases": ["Gangēyabhūshani", "Gangeyabhushini"]},
    34: {"name": "Vagadheeswari", "aliases": ["Vāgadhīshvari", "Vagadheeshwari"]},
    35: {"name": "Shulini", "aliases": ["Shūlini", "Sulini"]},
    36: {"name": "Chalanata", "aliases": ["Chalanāta", "Chalanattai"]},
    37: {"name": "Salagam", "aliases": ["Sālagam"]},
    38: {"name": "Jalarnavam", "aliases": ["Jalārnavam", "Jalarnava"]},
    39: {"name": "Jhalavarali", "aliases": ["Jhālavarāli"]},
    40: {"name": "Navaneetam", "aliases": ["Navanītam", "Navanita"]},
    41: {"name": "Pavani", "aliases": ["Pāvani"]},
    42: {"name": "Raghupriya", "aliases": ["Raghupriya"]},
    43: {"name": "Gavambhodi", "aliases": ["Gavāmbōdhi", "Gavambodhi"]},
    44: {"name": "Bhavapriya", "aliases": ["Bhāvapriya"]},
    45: {"name": "Subhapantuvarali", "aliases": ["Shubhapantuvarāli", "Shubhapantuvarali"]},
    46: {"name": "Shadvidamargini", "aliases": ["Shadvidamārgini", "Shadvidhamargini"]},
    47: {"name": "Suvarnangi", "aliases": ["Suvarnāngi"]},
    48: {"name": "Divyamani", "aliases": ["Divyamani"]},
    49: {"name": "Dhavalambari", "aliases": ["Dhavalāmbari"]},
    50: {"name": "Namanarayani", "aliases": ["Nāmanārāyani"]},
    51: {"name": "Kamavardhini", "aliases": ["Kāmavardhini", "Pantuvarali", "Kamavardhani"]},
    52: {"name": "Ramapriya", "aliases": ["Rāmapriya"]},
    53: {"name": "Gamanashrama", "aliases": ["Gamanāshrama", "Gamanasrama"]},
    54: {"name": "Vishwambari", "aliases": ["Vishvāmbari", "Vishvambhari"]},
    55: {"name": "Shamalangi", "aliases": ["Shyāmālangi", "Syamalangi"]},
    56: {"name": "Shanmukhapriya", "aliases": ["Shanmukhpriya"]},
    57: {"name": "Simhendramadhyamam", "aliases": ["Simhēndramadhyamam", "Simhendramadhyama"]},
    58: {"name": "Hemavati", "aliases": ["Hēmavati", "Hemavathi"]},
    59: {"name": "Dharmavati", "aliases": ["Dharmāvati", "Dharmavathi"]},
    60: {"name": "Neetimati", "aliases": ["Nītimati", "Neethimathi"]},
    61: {"name": "Kanthimani", "aliases": ["Kāntāmani", "Kanthamani"]},
    62: {"name": "Rishabhapriya", "aliases": ["Rishabhapriya"]},
    63: {"name": "Latangi", "aliases": ["Latāngi"]},
    64: {"name": "Vachaspati", "aliases": ["Vāchaspati", "Vachaspathi"]},
    65: {"name": "Mechakalyani", "aliases": ["Kalyani", "Mēchakalyāni"]},
    66: {"name": "Chitrambari", "aliases": ["Chitrāmbari", "Chithrambari"]},
    67: {"name": "Sucharitra", "aliases": ["Sucharitra"]},
    68: {"name": "Jyotiswarupini", "aliases": ["Jyōtisvarūpini", "Jyothiswaroopini"]},
    69: {"name": "Dhatuvardhini", "aliases": ["Dhātuvardhini", "Dhatuvardhani"]},
    70: {"name": "Nasikabhushani", "aliases": ["Nāsikābhūshani", "Nasikabhushini"]},
    71: {"name": "Kosalam", "aliases": ["Kōsalam", "Kosala"]},
    72: {"name": "Rasikapriya", "aliases": ["Rasikapriya"]},
}


def get_raga_info(melakarta_number: int) -> dict:
    """Get full info for a melakarta raga including computed swaras."""
    raga = MELAKARTA_RAGAS[melakarta_number].copy()
    raga["melakarta_number"] = melakarta_number
    raga.update(compute_swaras(melakarta_number))
    return raga


def find_raga_by_name(name: str) -> dict | None:
    """Find a melakarta raga by name or alias (case-insensitive)."""
    name_lower = name.lower().strip()
    for num, raga in MELAKARTA_RAGAS.items():
        if raga["name"].lower() == name_lower:
            return get_raga_info(num)
        for alias in raga.get("aliases", []):
            if alias.lower() == name_lower:
                return get_raga_info(num)
    return None


def get_all_ragas() -> list[dict]:
    """Get info for all 72 melakarta ragas."""
    return [get_raga_info(n) for n in range(1, 73)]


def resolve_raga_input(value: str) -> list[dict]:
    """
    Resolve a raga input which can be:
    - "all" -> all 72 ragas
    - a number (1-72) -> that melakarta raga
    - a range "start-end" (e.g. "34-45") -> melakarta ragas in that range
    - a name -> search by name/alias; if not found, treat as janya raga
    """
    value = value.strip()

    if value.lower() == "all":
        return get_all_ragas()

    import re
    range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", value)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start > end:
            start, end = end, start
        if not (1 <= start <= 72 and 1 <= end <= 72):
            raise ValueError(f"Range must be within 1-72, got {start}-{end}")
        return [get_raga_info(n) for n in range(start, end + 1)]

    try:
        num = int(value)
        if 1 <= num <= 72:
            return [get_raga_info(num)]
        raise ValueError(f"Melakarta number must be 1-72, got {num}")
    except ValueError:
        if not value.isdigit():
            raga = find_raga_by_name(value)
            if raga:
                return [raga]
            return [make_janya_raga_dict(value)]
        raise


def make_janya_raga_dict(name: str) -> dict:
    """Create a lightweight raga dict for a janya (non-melakarta) raga."""
    return {
        "name": name.strip(),
        "aliases": [],
        "melakarta_number": None,
        "arohana_str": "",
        "avrohana_str": "",
        "ma": "",
    }
