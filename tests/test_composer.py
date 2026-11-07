import unittest
from text_composer import TextComposerState


class TestTextComposer(unittest.TestCase):
    def setUp(self):
        # Fake clock starting at t = 0.0
        self.simulated_time = 0.0

        def fake_clock():
            return self.simulated_time

        self.clock = fake_clock
        self.composer = TextComposerState(hold_duration=0.8, min_confidence=0.50, clock_func=self.clock)

    def advance_time(self, seconds: float):
        self.simulated_time += seconds

    def test_stable_prediction_adds_exactly_one_letter(self):
        """Holding a valid sign for hold_duration (0.8s) appends exactly one letter."""
        # Frame at t = 0.0
        confirmed = self.composer.update("A", is_valid=True, confidence=0.92)
        self.assertFalse(confirmed)
        self.assertEqual(self.composer.get_text(), "")
        self.assertAlmostEqual(self.composer.get_hold_progress(), 0.0)

        # Advance by 0.4s (halfway)
        self.advance_time(0.4)
        confirmed = self.composer.update("A", is_valid=True, confidence=0.92)
        self.assertFalse(confirmed)
        self.assertEqual(self.composer.get_text(), "")
        self.assertAlmostEqual(self.composer.get_hold_progress(), 0.5)

        # Advance to 0.8s (threshold reached)
        self.advance_time(0.4)
        confirmed = self.composer.update("A", is_valid=True, confidence=0.92)
        self.assertTrue(confirmed)
        self.assertEqual(self.composer.get_text(), "A")

    def test_holding_same_sign_does_not_append_repeated_letters(self):
        """Continuing to hold the same sign does not append repeated duplicate letters."""
        # Hold until confirmed
        self.composer.update("B", is_valid=True, confidence=0.85)
        self.advance_time(0.8)
        confirmed = self.composer.update("B", is_valid=True, confidence=0.85)
        self.assertTrue(confirmed)
        self.assertEqual(self.composer.get_text(), "B")

        # Keep holding for another 2.5 seconds
        for _ in range(25):
            self.advance_time(0.1)
            subsequent = self.composer.update("B", is_valid=True, confidence=0.85)
            self.assertFalse(subsequent)

        self.assertEqual(self.composer.get_text(), "B")

    def test_releasing_and_signing_same_letter_appends_again(self):
        """Releasing the sign re-arms the composer, allowing the same letter to be typed again."""
        # First "A"
        self.composer.update("A", is_valid=True, confidence=0.90)
        self.advance_time(0.8)
        self.composer.update("A", is_valid=True, confidence=0.90)
        self.assertEqual(self.composer.get_text(), "A")

        # User releases hands (no hand / invalid frame)
        self.advance_time(0.2)
        self.composer.update(None, is_valid=False, confidence=0.0)
        self.assertTrue(self.composer.is_armed)

        # User signs "A" again
        self.advance_time(0.1)
        self.composer.update("A", is_valid=True, confidence=0.90)
        self.advance_time(0.8)
        confirmed_second = self.composer.update("A", is_valid=True, confidence=0.90)

        self.assertTrue(confirmed_second)
        self.assertEqual(self.composer.get_text(), "AA")

    def test_unstable_predictions_do_not_append_letters(self):
        """Rapidly alternating or unstable predictions never reach confirmation threshold."""
        # Alternate between 'C' and 'D' every 0.3s (hold_duration is 0.8s)
        for _ in range(10):
            self.composer.update("C", is_valid=True, confidence=0.80)
            self.advance_time(0.3)
            self.composer.update("D", is_valid=True, confidence=0.80)
            self.advance_time(0.3)

        self.assertEqual(self.composer.get_text(), "")
        self.assertLess(self.composer.get_hold_progress(), 1.0)

        # Once released, hold progress drops back to 0.0
        self.composer.update(None, is_valid=False, confidence=0.0)
        self.assertEqual(self.composer.get_hold_progress(), 0.0)

    def test_invalid_predictions_reset_confirmation_appropriately(self):
        """An invalid or uncertain prediction immediately resets the stability timer."""
        # Hold 'H' for 0.6s (75% progress)
        self.composer.update("H", is_valid=True, confidence=0.85)
        self.advance_time(0.6)
        self.composer.update("H", is_valid=True, confidence=0.85)
        self.assertAlmostEqual(self.composer.get_hold_progress(), 0.75)

        # Single hand dropped or uncertain prediction (is_valid=False)
        self.advance_time(0.1)
        self.composer.update("H", is_valid=False, confidence=0.85)
        self.assertAlmostEqual(self.composer.get_hold_progress(), 0.0)

        # Resume valid 'H': timer must start from 0.0, requiring another full 0.8s
        self.advance_time(0.1)
        self.composer.update("H", is_valid=True, confidence=0.85)
        self.advance_time(0.5)  # only 0.5s since resumption
        confirmed = self.composer.update("H", is_valid=True, confidence=0.85)
        self.assertFalse(confirmed)
        self.assertEqual(self.composer.get_text(), "")

        # Complete the remaining 0.3s
        self.advance_time(0.3)
        confirmed = self.composer.update("H", is_valid=True, confidence=0.85)
        self.assertTrue(confirmed)
        self.assertEqual(self.composer.get_text(), "H")

    def test_switching_to_different_letter_without_release(self):
        """Switching from one held sign to another confirms the second sign once held steady."""
        # Confirm 'M'
        self.composer.update("M", is_valid=True, confidence=0.95)
        self.advance_time(0.8)
        self.composer.update("M", is_valid=True, confidence=0.95)
        self.assertEqual(self.composer.get_text(), "M")

        # Immediately transition to 'N' without empty frame
        self.advance_time(0.1)
        self.composer.update("N", is_valid=True, confidence=0.92)
        self.advance_time(0.8)
        confirmed_n = self.composer.update("N", is_valid=True, confidence=0.92)
        self.assertTrue(confirmed_n)
        self.assertEqual(self.composer.get_text(), "MN")

    def test_low_confidence_below_threshold_ignored(self):
        """Predictions with confidence below min_confidence are treated as invalid."""
        for _ in range(15):
            self.composer.update("A", is_valid=True, confidence=0.35)  # threshold is 0.50
            self.advance_time(0.1)

        self.assertEqual(self.composer.get_text(), "")
        self.assertAlmostEqual(self.composer.get_hold_progress(), 0.0)

    def test_space_undo_clear_actions(self):
        """SPACE, UNDO, and CLEAR behave correctly and preserve text integrity."""
        # Type "HI"
        self.composer.update("H", is_valid=True, confidence=0.90)
        self.advance_time(0.8)
        self.composer.update("H", is_valid=True, confidence=0.90)

        self.advance_time(0.2)
        self.composer.update("I", is_valid=True, confidence=0.90)
        self.advance_time(0.8)
        self.composer.update("I", is_valid=True, confidence=0.90)
        self.assertEqual(self.composer.get_text(), "HI")

        # Space
        self.composer.add_space()
        self.assertEqual(self.composer.get_text(), "HI ")

        # Undo space
        self.composer.undo()
        self.assertEqual(self.composer.get_text(), "HI")

        # Undo 'I'
        self.composer.undo()
        self.assertEqual(self.composer.get_text(), "H")

        # Clear
        self.composer.clear()
        self.assertEqual(self.composer.get_text(), "")

        # Undo on empty text does not crash
        self.composer.undo()
        self.assertEqual(self.composer.get_text(), "")


if __name__ == "__main__":
    unittest.main()
