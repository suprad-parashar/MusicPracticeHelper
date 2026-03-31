"""Normalize tara shadja as >S in arohana/avrohana (not plain S)."""


def normalize_swara_scale_notation(arohana: list[str], avrohana: list[str]) -> tuple[list[str], list[str]]:
    """
    Use >S for upper-octave Shadja:
    - arohana: any S after the first position (ascent to tara sthayi)
    - avrohana: first S only when it begins the descent from upper octave (not the final S, which is madhya)
    """
    a = list(arohana)
    v = list(avrohana)
    for i in range(1, len(a)):
        if a[i] == "S":
            a[i] = ">S"
    if len(v) >= 2 and v[0] == "S":
        v[0] = ">S"
    return a, v


def normalize_raga_dict_swara_fields(data: dict) -> dict:
    """Mutate a loaded raga JSON dict in place for arohana/avrohana keys."""
    ao = data.get("arohana")
    av = data.get("avrohana")
    if not isinstance(ao, list) or not isinstance(av, list):
        return data
    na, nv = normalize_swara_scale_notation(ao, av)
    data["arohana"] = na
    data["avrohana"] = nv
    return data
