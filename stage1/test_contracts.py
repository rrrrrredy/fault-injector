import unittest

from corpus import assess_pair, check_ticket


class CorpusContracts(unittest.TestCase):
    def test_historyless_renamed_copy(self):
        common = {("name", str(i)) for i in range(120)}
        a = dict(roots=["a"], core_sha256="x", package="first")
        b = dict(roots=["b"], core_sha256="y", package="renamed")
        reasons, _ = assess_pair(a, b, common, common | {("name", "extra")})
        self.assertIn("near_duplicate", reasons)
        self.assertFalse(assess_pair(a, b, common, {("other", "code")})[0])

    def test_ticket_rejects_provenance_and_repair_instructions(self):
        for leak in ("See PR #6", "https://github.com/a/b", "commit deadbeef",
                     "replace the regex", "line 24", "```diff\n+something\n```"):
            with self.subTest(leak=leak), self.assertRaises(ValueError):
                check_ticket(dict(symptom=leak, expected="A plural.", reproduce="Enter a word."))
