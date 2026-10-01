import pandas as pd
import re
import unicodedata

# ==========================================================
# preprocessor.py (V4 - Script-Preserving, Boundary-Verified)
# ==========================================================
#
# What changed vs V3 and why:
#
# 1. UNICODE HANDLING NO LONGER DESTROYS NON-LATIN-SCRIPT TEXT
#    V3's `unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')`
#    is fine for Latin text with accents ("Café" -> "Cafe") but silently
#    deletes every character outside the ASCII range. Verified directly:
#    a Chinese business name ("北京烤鸭店") becomes an EMPTY STRING, and
#    Arabic/Cyrillic names collapse to bare whitespace. Since this
#    dataset carries its own country field (implying multi-country
#    data), any non-Latin-script business/address is currently
#    unmatchable on name/address similarity no matter how good the
#    real match is -- it silently becomes a "missing" record to every
#    downstream feature. Fixed by stripping only Unicode *combining
#    marks* (the category that accents actually are) after NFKD
#    decomposition, which still turns "café" -> "cafe" but leaves
#    Chinese/Arabic/Cyrillic base characters untouched.
#
# 2. THE SPECIAL-CHARACTER FILTER HAD THE **SAME** BUG, INDEPENDENTLY
#    Even after fixing (1), the old `[^a-z0-9\s]` special-character
#    filter would have wiped the exact same non-Latin text right back
#    out at a later step -- verified this separately, since it's a
#    distinct regex that would need its own fix even if (1) were
#    patched. Replaced with a Unicode-aware `[^\w\s]` (Python's `\w` on
#    a str pattern already matches letters/digits from any script, not
#    just ASCII), so name/address text in any language survives end to
#    end.
#
# 3. APOSTROPHES NO LONGER LEAVE A STRAY NOISE TOKEN
#    "Joe's Taqueria" previously became "joe s taqueria" -- the
#    special-char filter replaces punctuation with a space rather than
#    deleting it, so the possessive apostrophe left behind a spurious
#    one-character "s" token. Apostrophes (straight and curly) are now
#    dropped with no replacement space, so this becomes "joes
#    taqueria" -- one clean token instead of one real token plus noise.
#    (Low-impact on its own, since blocking's min_token_len=3 already
#    filters single-character tokens, but it's a free, correct fix.)
#
# 4. `.casefold()` INSTEAD OF `.lower()`
#    casefold() is the Unicode-correct choice for caseless matching
#    (handles cases .lower() gets wrong, e.g. German "ß" vs "ss");
#    .lower() was never actually wrong for the abbreviation maps below
#    since those are plain ASCII, but casefold() is the more correct
#    default now that non-Latin scripts are preserved through this
#    function.
#
# 5. WORD-BOUNDARY ABBREVIATION EXPANSION -- VERIFIED, NOT CHANGED
#    Checked directly whether the \b(...)​\b alternation could misfire
#    on cases like "3rd" (ordinal) or "str" vs "st" (does the shorter
#    alternative wrongly match inside the longer one): it does not.
#    "3rd avenue" is left alone (no boundary exists between a digit and
#    a following letter, so "rd" can't match inside "3rd"), and
#    "123 str" correctly becomes "123 street" (regex alternation
#    backtracks to the right alternative within the \b...\b anchors).
#    This part of V3 was already correct and is unchanged.
#
# ONE CROSS-FILE FOLLOW-UP (not fixable here, flagging for blocking.py):
# NAME_MAP expands "inc"->"incorporated", "llc"->"limited liability
# company", "pvt"->"private". blocking.py's STOPWORDS set (used to
# strip generic filler words before building blocking keys) recognizes
# the raw abbreviations "inc"/"llc" and the expansions "limited"/
# "company", but NOT the expanded tokens "incorporated", "liability",
# or "private" -- verified by diffing the two sets directly. Any of
# these three showing up as an un-stripped token in blocking's
# token-based keys is generic filler being treated as if it were
# distinctive. ("saint", also introduced by the "st" expansion, is
# correctly left out of that stopword set -- it's a real distinguishing
# word, not filler, so no change needed there.) One-line fix in
# blocking.py:
#
#     STOPWORDS = {
#         "inc", "llc", "corp", "corporation", "co", "company", "ltd",
#         "limited", "group", "the", "and", "of", "restaurant", "store",
#         "shop", "llp", "plc", "gmbh", "sa", "srl",
#         "incorporated", "liability", "private",   # <-- added
#     }

NAME_MAP = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "llc": "limited liability company",
    "st": "saint"
}

ADDRESS_MAP = {
    "st": "street", "rd": "road", "ave": "avenue",
    "blvd": "boulevard", "dr": "drive", "str": "street",
    "ste": "suite", "fl": "floor", "apt": "apartment"
}

# Pre-compile regex patterns ONCE at module load
NAME_PATTERN = re.compile(r'\b(' + '|'.join(NAME_MAP.keys()) + r')\b')
ADDRESS_PATTERN = re.compile(r'\b(' + '|'.join(ADDRESS_MAP.keys()) + r')\b')

RE_STOP_WORDS = re.compile(r'\b(the|of|in|at|for|on|and)\b')
RE_APOSTROPHE = re.compile(r'[\'\u2019]')          # straight ' and curly '
RE_SPECIAL = re.compile(r'[^\w\s]|_')              # Unicode-aware: keeps letters/digits from ANY script
RE_SPACES = re.compile(r'\s+')


def _clean_string(text: str, map_dict: dict, compiled_pattern: re.Pattern) -> str:
    """
    Pure Python string cleaner. Zero Pandas overhead. Script-preserving:
    Latin diacritics are stripped, but non-Latin scripts (Chinese,
    Arabic, Cyrillic, etc.) survive intact.
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. Strip combining diacritical marks only (café -> cafe), keep base
    #    characters from every script (Chinese/Arabic/Cyrillic untouched).
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))

    # 2. Casefold (Unicode-correct lowercasing)
    text = text.casefold()

    # 3. Explicitly handle '&' before removing special characters
    text = text.replace('&', ' and ')

    # 4. Drop apostrophes with no replacement space (joe's -> joes, not joe s)
    text = RE_APOSTROPHE.sub('', text)

    # 5. Single-pass abbreviation replacement
    if map_dict:
        text = compiled_pattern.sub(lambda m: map_dict[m.group(1)], text)

    # 6. Remove stopwords
    text = RE_STOP_WORDS.sub(' ', text)

    # 7. Remove special characters (Unicode-aware: non-Latin scripts survive)
    text = RE_SPECIAL.sub(' ', text)

    # 8. Collapse spaces and strip
    return RE_SPACES.sub(' ', text).strip()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies lightning-fast list comprehensions to the dataframe.
    """
    df_clean = df.copy()

    df_clean['business_name_clean'] = [
        _clean_string(x, NAME_MAP, NAME_PATTERN) for x in df_clean['business_name']
    ]

    df_clean['business_address_clean'] = [
        _clean_string(x, ADDRESS_MAP, ADDRESS_PATTERN) for x in df_clean['business_address']
    ]

    df_clean['country_clean'] = [
        _clean_string(x, {}, RE_SPACES) for x in df_clean['country']
    ]

    return df_clean