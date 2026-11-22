"""English number-to-words helpers (standard library only).

Used by textclean to turn digits into something the phonemizer reads naturally.
"""

ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]

_ORD_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _below_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(ONES[n // 100] + " hundred")
        n %= 100
        if n == 0:
            return parts[0]
    if n < 20:
        parts.append(ONES[n])
    else:
        t, o = divmod(n, 10)
        parts.append(TENS[t] + (("-" + ONES[o]) if o else ""))
    return " ".join(parts)


def int_to_words(n: int) -> str:
    """1234 -> 'one thousand two hundred thirty-four'. Very long numbers are read digit by digit."""
    if n < 0:
        return "minus " + int_to_words(-n)
    if n == 0:
        return "zero"
    if n >= 10 ** (3 * len(SCALES)):
        return digits_to_words(str(n))
    chunks = []
    i = 0
    while n > 0:
        n, rem = divmod(n, 1000)
        if rem:
            words = _below_1000(rem)
            if SCALES[i]:
                words += " " + SCALES[i]
            chunks.append(words)
        i += 1
    return " ".join(reversed(chunks))


def digits_to_words(digits: str) -> str:
    """'3141' -> 'three one four one' (used for decimals and very long numbers)."""
    return " ".join(ONES[int(d)] for d in digits if d.isdigit())


def ordinal_words(n: int) -> str:
    """21 -> 'twenty-first'."""
    words = int_to_words(n)
    head, sep, last = words.rpartition(" ")
    last_part = last.rsplit("-", 1)
    base = last_part[-1]
    if base in _ORD_IRREGULAR:
        ordinal = _ORD_IRREGULAR[base]
    elif base.endswith("y"):
        ordinal = base[:-1] + "ieth"
    else:
        ordinal = base + "th"
    last_new = "-".join(last_part[:-1] + [ordinal])
    return head + sep + last_new


def year_words(n: int) -> str:
    """1984 -> 'nineteen eighty-four', 1905 -> 'nineteen oh five', 2005 -> 'two thousand five'."""
    if 2000 <= n <= 2009:
        return "two thousand" + ((" " + ONES[n - 2000]) if n > 2000 else "")
    hi, lo = divmod(n, 100)
    if lo == 0:
        return int_to_words(hi) + " hundred"
    if lo < 10:
        return int_to_words(hi) + " oh " + ONES[lo]
    return int_to_words(hi) + " " + int_to_words(lo)


def pluralize_year_words(words: str) -> str:
    """'nineteen eighty' -> 'nineteen eighties', 'nineteen hundred' -> 'nineteen hundreds'."""
    head, sep, last = words.rpartition(" ")
    if last.endswith("y"):
        last = last[:-1] + "ies"
    else:
        last += "s"
    return head + sep + last


def roman_to_int(s: str) -> int:
    """'XIV' -> 14. Returns 0 if s is not a plain roman numeral."""
    s = s.upper()
    if not s or any(c not in _ROMAN for c in s):
        return 0
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN[c]
        if i + 1 < len(s) and _ROMAN[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total
