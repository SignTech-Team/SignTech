import unittest

from signtech_nlp import SignTechNLP


class SignTechNLPTests(unittest.TestCase):
    def setUp(self):
        self.engine = SignTechNLP(stable_windows=3, threshold=0.7, release_windows=2)

    def test_vocabulary_and_ids(self):
        self.assertEqual(len(self.engine.id_to_word), 226)
        self.assertEqual(self.engine.add_confirmed(175)["raw_text"], "Hava")

    def test_rule_and_raw_separate(self):
        self.engine.add_confirmed("Hava")
        result = self.engine.add_confirmed("Güzel")
        self.assertEqual(result["raw_text"], "Hava Güzel")
        self.assertEqual(result["suggested_sentence"], "Hava güzel.")

    def test_no_invented_sentence(self):
        self.engine.add_confirmed("Hava")
        result = self.engine.add_confirmed("Kapı")
        self.assertIsNone(result["suggested_sentence"])
        self.assertEqual(result["raw_text"], "Hava Kapı")

    def test_threshold_and_stability(self):
        for _ in range(8):
            self.assertNotEqual(self.engine.update("Hava", 0.50)["status"], "accepted")
        for _ in range(2):
            self.engine.update("Hava", 0.85)
        self.assertEqual(self.engine.state()["raw_words"], [])
        self.assertEqual(self.engine.update("Hava", 0.85)["raw_words"], ["Hava"])
        self.assertEqual(self.engine.update("Hava", 0.85)["raw_words"], ["Hava"])

    def test_rearm_after_gap_and_repeat(self):
        for _ in range(3):
            self.engine.update("Hava", 0.9)
        for _ in range(2):
            self.engine.update(None, 0, hands_visible=False)
        for _ in range(3):
            self.engine.update("Hava", 0.9)
        self.assertEqual(self.engine.state()["raw_words"], ["Hava", "Hava"])

    def test_unknown_ignored(self):
        self.assertEqual(self.engine.add_confirmed("Ben")["status"], "unknown_word")
        self.assertEqual(self.engine.state()["raw_text"], "")

    def test_reset_undo(self):
        self.engine.add_confirmed("Anne")
        self.engine.add_confirmed("Hasta")
        self.assertEqual(self.engine.undo()["raw_text"], "Anne")
        self.engine.reset()
        self.assertEqual(self.engine.state()["raw_text"], "")

    def test_invalid_percent_input(self):
        with self.assertRaises(ValueError):
            self.engine.update("Hava", 85)




class SignTechV2RulesTests(unittest.TestCase):
    def setUp(self):
        self.engine = SignTechNLP()

    def feed(self, *words):
        self.engine.reset()
        state = None
        for word in words:
            state = self.engine.add_confirmed(word)
        return state

    def test_three_signs(self):
        result = self.feed("Hava", "Çok", "Güzel")
        self.assertEqual(result["suggested_sentence"], "Hava çok güzel.")
        self.assertFalse(result["inference_used"])

    def test_person_slot(self):
        result = self.feed("Kardeş", "Hasta")
        self.assertEqual(result["suggested_sentence"], "Kardeş hasta.")
        self.assertEqual(result["applied_rules"], ["person_sick"])

    def test_two_sentences(self):
        result = self.feed("Hava", "Güzel", "Baba", "Hasta")
        self.assertEqual(result["suggested_sentence"], "Hava güzel. Baba hasta.")

    def test_unmatched_means_none(self):
        result = self.feed("Hava", "Güzel", "Bilgisayar")
        self.assertIsNone(result["suggested_sentence"])
        self.assertEqual(result["raw_text"], "Hava Güzel Bilgisayar")

    def test_inference_default_disabled(self):
        result = self.feed("Anne", "Ev", "Gitmek")
        self.assertIsNone(result["suggested_sentence"])

    def test_inference_opt_in(self):
        self.engine = SignTechNLP(allow_inference=True)
        result = self.feed("Anne", "Ev", "Gitmek")
        self.assertEqual(result["suggested_sentence"], "Anne eve gidiyor.")
        self.assertTrue(result["inference_used"])
        self.assertIn("zaman", " ".join(result["suggestion_notes"]))

    def test_metadata_after_undo(self):
        self.feed("Hava", "Çok", "Güzel")
        result = self.engine.undo()
        self.assertIsNone(result["suggested_sentence"])

    def test_226_words_only(self):
        result = self.feed("Ben")
        self.assertEqual(result["status"], "unknown_word")
        self.assertIsNone(result["suggested_sentence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
