"""Unit tests for the audiobook text pipeline (no model needed).

Run from the wpiper folder:  python -m unittest tests.test_audiobook_text -v
"""

import unittest

from wpiper.audiobook.config import BookConfig, Chapter
from wpiper.audiobook.ingest import split_chapters
from wpiper.audiobook.numbers import int_to_words, ordinal_words, roman_to_int, year_words
from wpiper.audiobook.plan import build_units
from wpiper.audiobook.splitter import detect_wrap, split_long, split_paragraphs, split_sentences
from wpiper.audiobook.srt import Cue, build_cues, fmt_time, wrap_lines
from wpiper.audiobook.config import Placed, Unit
from wpiper.audiobook.textclean import clean_for_speech, clean_heading, compile_pronunciations


class Numbers(unittest.TestCase):
    def test_int(self):
        self.assertEqual(int_to_words(0), "zero")
        self.assertEqual(int_to_words(1234), "one thousand two hundred thirty-four")
        self.assertEqual(int_to_words(1000000), "one million")
        self.assertEqual(int_to_words(101), "one hundred one")

    def test_ordinal(self):
        self.assertEqual(ordinal_words(1), "first")
        self.assertEqual(ordinal_words(21), "twenty-first")
        self.assertEqual(ordinal_words(40), "fortieth")
        self.assertEqual(ordinal_words(100), "one hundredth")

    def test_year(self):
        self.assertEqual(year_words(1984), "nineteen eighty-four")
        self.assertEqual(year_words(1905), "nineteen oh five")
        self.assertEqual(year_words(1900), "nineteen hundred")
        self.assertEqual(year_words(2005), "two thousand five")
        self.assertEqual(year_words(2024), "twenty twenty-four")

    def test_roman(self):
        self.assertEqual(roman_to_int("XIV"), 14)
        self.assertEqual(roman_to_int("hello"), 0)


class Clean(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(clean_for_speech("In 1984 he paid $5.50."), "In nineteen eighty-four he paid five dollars fifty cents.")
        self.assertEqual(clean_for_speech("She was 21st in line."), "She was twenty-first in line.")
        self.assertEqual(clean_for_speech("It cost 1,234 coins."), "It cost one thousand two hundred thirty-four coins.")
        self.assertEqual(clean_for_speech("A 50% chance."), "A fifty percent chance.")
        self.assertEqual(clean_for_speech("Pi is 3.14."), "Pi is three point one four.")
        self.assertEqual(clean_for_speech("Meet at 10:30."), "Meet at ten thirty.")
        self.assertEqual(clean_for_speech("The 1980s."), "The nineteen eighties.")

    def test_abbreviations(self):
        self.assertEqual(clean_for_speech("Mr. Smith and Dr. Jones."), "Mister Smith and Doctor Jones.")
        self.assertEqual(clean_for_speech("Apples, pears, etc."), "Apples, pears, et cetera.")
        self.assertEqual(clean_for_speech("See No. 5."), "See Number five.")

    def test_punctuation(self):
        self.assertEqual(clean_for_speech("“Hello,” she said — quietly."), "Hello, she said, quietly.")
        self.assertEqual(clean_for_speech("It was fine (really)."), "It was fine, really.")
        self.assertEqual(clean_for_speech("A **bold** move[3]."), "A bold move.")
        self.assertEqual(clean_for_speech("Visit https://example.com now."), "Visit link now.")

    def test_heading(self):
        self.assertEqual(clean_heading("Chapter XII"), "Chapter twelve")
        self.assertEqual(clean_heading("Chapter 3: The Dark"), "Chapter three: The Dark")

    def test_pronunciations(self):
        pron = compile_pronunciations({"Hermione": "Her my oh knee"})
        self.assertEqual(clean_for_speech("Hermione ran.", pron), "Her my oh knee ran.")


class Sentences(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(split_sentences("One. Two! Three? Four."), ["One.", "Two!", "Three?", "Four."])

    def test_abbreviations_not_split(self):
        self.assertEqual(split_sentences("Mr. Smith met Dr. Jones. They left."),
                         ["Mr. Smith met Dr. Jones.", "They left."])
        self.assertEqual(split_sentences("J. R. R. Tolkien wrote it. Fine."), ["J. R. R. Tolkien wrote it.", "Fine."])
        self.assertEqual(split_sentences("Use tea, e.g. green. Then wait."), ["Use tea, e.g. green.", "Then wait."])

    def test_etc_and_numbers(self):
        self.assertEqual(split_sentences("Apples, etc. and pears. Done."), ["Apples, etc. and pears.", "Done."])
        self.assertEqual(split_sentences("See Fig. 3 for more. Next."), ["See Fig. 3 for more.", "Next."])
        self.assertEqual(split_sentences("It is 3.14 exactly. Yes."), ["It is 3.14 exactly.", "Yes."])

    def test_quotes(self):
        self.assertEqual(split_sentences('"Stop!" he cried. "Now."'), ['"Stop!" he cried.', '"Now."'])
        self.assertEqual(split_sentences('He said, "Go." Then he left.'), ['He said, "Go."', "Then he left."])
        self.assertEqual(split_sentences("“No.” She sighed."), ["“No.”", "She sighed."])

    def test_ellipsis(self):
        self.assertEqual(split_sentences("Well... maybe so. Okay."), ["Well... maybe so.", "Okay."])
        self.assertEqual(split_sentences("He paused... The room fell silent."), ["He paused...", "The room fell silent."])

    def test_long_split(self):
        s = "one two three four five six; seven eight nine ten eleven twelve"
        self.assertEqual(split_long(s, 8, 3), ["one two three four five six;", "seven eight nine ten eleven twelve"])
        long = " ".join(["word"] * 100)
        parts = split_long(long, 30, 4)
        self.assertTrue(all(len(p.split()) <= 30 for p in parts))
        self.assertEqual(" ".join(parts), long)

    def test_conjunction_split(self):
        s = "the quick brown fox jumped over the fence and ran into the forest beyond the hills"
        parts = split_long(s, 10, 3)
        self.assertEqual(" ".join(parts), s)
        self.assertTrue(all(len(p.split()) <= 10 for p in parts))


class Lines(unittest.TestCase):
    def test_hard_wrap_detected(self):
        import textwrap
        para = ("This is a long paragraph from a fixed width ebook file that was wrapped at about seventy "
                "characters per line, so single newlines fall in the middle of sentences and carry no meaning "
                "at all, and the whole thing ends with a full stop right here.")
        text = "\n\n".join(textwrap.fill(para, 70) for _ in range(5))
        self.assertEqual(detect_wrap(split_paragraphs(text)), "hard")

    def test_lines_mode_for_poem(self):
        text = "\n\n".join("Roses are red\nViolets are blue\nSugar is sweet" for _ in range(5))
        self.assertEqual(detect_wrap(split_paragraphs(text)), "lines")

    def test_build_units_flags(self):
        ch = [Chapter(0, "Chapter 1", "First sentence. Second one.\n\nNew paragraph here.", has_heading=True)]
        units = build_units(ch, BookConfig())
        self.assertEqual([u.kind for u in units], ["heading", "sentence", "sentence", "sentence"])
        self.assertTrue(units[0].spoken.endswith("."))
        self.assertTrue(units[2].ends_para)
        self.assertTrue(units[-1].ends_chapter)
        self.assertFalse(units[1].ends_para)

    def test_scene_break_is_silent(self):
        ch = [Chapter(0, "x", "Before.\n\n* * *\n\nAfter.", has_heading=False)]
        units = build_units(ch, BookConfig())
        self.assertEqual([u.kind for u in units], ["sentence", "scene_break", "sentence"])
        self.assertEqual(units[1].spoken, "")

    def test_long_sentence_pieces_share_sentence_id(self):
        long = ", ".join(["alpha beta gamma delta"] * 6) + "."
        units = build_units([Chapter(0, "x", long)], BookConfig(max_words=12))
        self.assertGreater(len(units), 1)
        self.assertEqual(len({u.sent for u in units}), 1)
        self.assertEqual([u.ends_sentence for u in units], [False] * (len(units) - 1) + [True])
        self.assertTrue(all(u.spoken.endswith(",") for u in units[:-1]))


class Chapters(unittest.TestCase):
    def test_headings(self):
        text = "Preface text here.\n\nChapter 1\n\nBody one.\n\nCHAPTER II\n\nBody two.\n"
        chs = split_chapters(text, "Book")
        self.assertEqual([c.title for c in chs], ["Front matter", "Chapter 1", "CHAPTER II"])
        self.assertEqual(chs[1].text, "Body one.")

    def test_no_headings(self):
        chs = split_chapters("Just text.\n\nMore text.", "My Book")
        self.assertEqual(len(chs), 1)
        self.assertFalse(chs[0].has_heading)


class HeadingRules(unittest.TestCase):
    def test_sequential_numbers_become_chapters(self):
        text = "1\n\nFirst body.\n\n2\n\nSecond body.\n\n3\n\nThird body.\n"
        chs = split_chapters(text, "Book")
        self.assertEqual([c.title for c in chs], ["Chapter 1", "Chapter 2", "Chapter 3"])
        self.assertEqual(chs[1].text, "Second body.")
        self.assertTrue(all(c.has_heading for c in chs))

    def test_number_label_can_be_empty(self):
        text = "1\n\nAlpha body.\n\n2\n\nBeta body.\n\n3\n\nGamma body.\n"
        chs = split_chapters(text, "Book", BookConfig(number_label=""))
        self.assertEqual([c.title for c in chs], ["1", "2", "3"])

    def test_roman_sequence_becomes_chapters(self):
        text = "I\n\nOne body.\n\nII\n\nTwo body.\n\nIII\n\nThree body.\n"
        self.assertEqual([c.title for c in split_chapters(text, "B")], ["Chapter 1", "Chapter 2", "Chapter 3"])

    def test_lone_pronoun_I_is_not_a_chapter(self):
        text = "Some text.\n\nI\n\nMore text.\n"
        chs = split_chapters(text, "Book")
        self.assertEqual(len(chs), 1)

    def test_page_numbers_dropped(self):
        text = "Body one.\n\n19\n\nBody two.\n\n47\n\nBody three.\n"
        chs = split_chapters(text, "Book")
        self.assertEqual(len(chs), 1)
        self.assertNotIn("19", chs[0].text)
        self.assertNotIn("47", chs[0].text)
        kept = split_chapters(text, "Book", BookConfig(drop_page_numbers=False))
        self.assertIn("19", kept[0].text)

    def test_many_numbers_are_page_numbers(self):
        text = "\n\n".join(f"{n}\n\nBody {n}." for n in range(1, 200))
        chs = split_chapters(text, "Book")
        self.assertEqual(len(chs), 1)

    def test_caps_become_chapters_when_no_other_headings(self):
        text = "THIS WRECKAGE\n\nBody one.\n\nTHIS DYING MACHINE\n\nBody two.\n"
        chs = split_chapters(text, "Book")
        self.assertEqual([c.title for c in chs], ["THIS WRECKAGE", "THIS DYING MACHINE"])

    def test_caps_stay_inline_when_chapters_exist(self):
        text = "Chapter 1\n\nSome body text.\n\nKAUAI, HAWAII\n\nMore body.\n\nChapter 2\n\nThe end now.\n"
        chs = split_chapters(text, "Book")
        self.assertEqual([c.title for c in chs], ["Chapter 1", "Chapter 2"])
        self.assertIn("KAUAI, HAWAII", chs[0].text)

    def test_caps_modes(self):
        text = "ONE TITLE\n\nSome body.\n\nTWO TITLE\n\nOther body.\n"
        self.assertEqual(len(split_chapters(text, "B", BookConfig(caps_headings="off"))), 1)
        self.assertEqual(len(split_chapters(text, "B", BookConfig(caps_headings="inline"))), 1)
        self.assertEqual(len(split_chapters(text, "B", BookConfig(caps_headings="chapters"))), 2)

    def test_contents_list_numbers_are_not_chapters(self):
        toc = "\n".join(f"{n}\n" for n in range(1, 6))               # 1..5 on neighbouring lines
        body = "\n\n".join(f"{n}\n\nSome words in section {n} here." for n in range(1, 6))
        chs = split_chapters(toc + "\n\n" + body, "Book")
        self.assertEqual([c.title for c in chs], [f"Chapter {n}" for n in range(1, 6)])
        self.assertEqual(len(chs), 5)

    def test_empty_heading_is_dropped(self):
        text = ("Contents\n\nChapter 1\n\nChapter 2\n\n"
                "Chapter 1\n\nA real chapter body here.\n\nChapter 2\n\nAnother real body here.\n")
        chs = split_chapters(text, "Book")
        self.assertEqual([c.title for c in chs], ["Front matter", "Chapter 1", "Chapter 2"])
        self.assertEqual(chs[0].text, "Contents")

    def test_single_caps_line_is_not_promoted(self):
        self.assertEqual(len(split_chapters("A STORY\n\nBody text here.\n", "Book")), 1)

    def test_shouted_line_with_punctuation_is_not_a_title(self):
        text = "Body.\n\nGET OUT!\n\nMore.\n\nRUN NOW.\n\nEnd.\n"
        self.assertEqual(len(split_chapters(text, "Book")), 1)

    def test_caps_spoken_as_words(self):
        self.assertEqual(clean_heading("THIS WRECKAGE"), "This Wreckage.".rstrip("."))
        self.assertEqual(clean_heading("PART II"), "Part two")
        self.assertEqual(clean_heading("THE END OF PART ONE"), "The End Of Part One")
        units = build_units([Chapter(0, "x", "Intro text.\n\n12 AUGUST 2113\n\nBody.", False)], BookConfig())
        dateline = [u for u in units if u.text == "12 AUGUST 2113"][0]
        self.assertEqual(dateline.kind, "heading")
        self.assertEqual(dateline.spoken, "twelve August twenty-one thirteen.")


class Subtitles(unittest.TestCase):
    def test_time(self):
        self.assertEqual(fmt_time(3661.5), "01:01:01,500")
        self.assertEqual(fmt_time(0), "00:00:00,000")

    def test_wrap(self):
        self.assertEqual(wrap_lines("short", 42, 2), "short")
        out = wrap_lines("this is a rather long subtitle line that needs wrapping", 42, 2)
        self.assertEqual(len(out.split("\n")), 2)
        self.assertTrue(all(len(l) <= 42 for l in out.split("\n")))

    def test_cues_follow_placements(self):
        cfg = BookConfig()
        sr = 1000
        mk = lambda i, text, a, b, para=0: Placed(Unit(i, 0, para, i, "sentence", text, text), a, b)
        placed = [mk(0, "First sentence here.", 0, 2000), mk(1, "Second sentence here.", 2500, 5000)]
        cues = build_cues(placed, sr, cfg)
        self.assertEqual([c.text for c in cues], ["First sentence here.", "Second sentence here."])
        self.assertAlmostEqual(cues[0].start, 0.0)
        self.assertAlmostEqual(cues[1].start, 2.5)
        self.assertLessEqual(cues[0].end, cues[1].start)

    def test_long_sentence_splits_evenly_and_wraps(self):
        cfg = BookConfig()
        text = ("Mr. Harlow, its keeper, climbed the 112 steps every evening at 6:30, whatever the weather.")
        cues = build_cues([Placed(Unit(0, 0, 0, 0, "sentence", text, text), 0, 9000)], 1000, cfg)
        self.assertGreaterEqual(len(cues), 2)
        self.assertTrue(all(len(c.text.split()) >= 3 for c in cues), [c.text for c in cues])
        self.assertAlmostEqual(cues[0].start, 0.0)
        for c in cues:
            self.assertTrue(all(len(l) <= cfg.srt_max_chars for l in wrap_lines(c.text, 42, 2).split("\n")), c.text)

    def test_last_cue_not_past_audio_end(self):
        cfg = BookConfig()
        placed = [Placed(Unit(0, 0, 0, 0, "sentence", "The end of it all.", "The end of it all."), 0, 2000)]
        cues = build_cues(placed, 1000, cfg, total_samples=2050)
        self.assertLessEqual(cues[-1].end, 2.05 + 1e-9)

    def test_short_cue_merges_with_neighbour(self):
        cfg = BookConfig()
        mk = lambda i, text, a, b: Placed(Unit(i, 0, 0, i, "sentence", text, text), a, b)
        cues = build_cues([mk(0, "Yes.", 0, 400), mk(1, "I will go there now.", 700, 2500)], 1000, cfg)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Yes. I will go there now.")


if __name__ == "__main__":
    unittest.main()
