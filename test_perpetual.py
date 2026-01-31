#!/usr/bin/env python3
"""
Basic test suite for perpetual.py

Run with: python3 test_perpetual.py
"""

import unittest
from pathlib import Path
from perpetual import (
    clean_filename,
    parse_se,
    dynamic_episode_pattern,
    SPECIAL_CASES,
)


class TestFilenameParser(unittest.TestCase):
    """Test TV show filename parsing."""

    def setUp(self):
        self.garbage_words = {"1080p", "x264", "hdtv", "web"}
        self.filetypes = ["mkv", "mp4", "avi"]
        self.special_cases = SPECIAL_CASES

    def test_basic_parse(self):
        """Test basic filename parsing."""
        result = clean_filename(
            "The.Show.s01e05.Episode.Title.mkv",
            self.garbage_words,
            self.filetypes,
            self.special_cases
        )
        self.assertIsNotNone(result)
        folder, filename = result
        self.assertEqual(folder, "The.Show")
        self.assertTrue("S01E05" in filename)
        self.assertTrue("Episode.Title" in filename)

    def test_with_garbage_words(self):
        """Test parsing with garbage words."""
        result = clean_filename(
            "The.Show.s01e05.Title.1080p.x264.mkv",
            self.garbage_words,
            self.filetypes,
            self.special_cases
        )
        self.assertIsNotNone(result)
        folder, filename = result
        self.assertNotIn("1080p", filename)
        self.assertNotIn("x264", filename)

    def test_special_cases(self):
        """Test special case handling (e.g., FBI, USA)."""
        result = clean_filename(
            "FBI.s01e05.The.Case.mkv",
            self.garbage_words,
            self.filetypes,
            self.special_cases
        )
        self.assertIsNotNone(result)
        folder, filename = result
        self.assertEqual(folder, "FBI")  # Should remain uppercase
        self.assertTrue("FBI.S01E05" in filename)

    def test_invalid_filename(self):
        """Test handling of invalid filenames."""
        result = clean_filename(
            "not_a_tv_show.mkv",
            self.garbage_words,
            self.filetypes,
            self.special_cases
        )
        self.assertIsNone(result)

    def test_season_episode_parsing(self):
        """Test season/episode number extraction."""
        test_path = Path("The.Show.S02E15.Title.mkv")
        se = parse_se(test_path)
        self.assertIsNotNone(se)
        season, episode = se
        self.assertEqual(season, 2)
        self.assertEqual(episode, 15)

    def test_pattern_matching(self):
        """Test episode pattern regex."""
        pattern = dynamic_episode_pattern(["mkv", "mp4"])

        # Should match
        self.assertTrue(pattern.match("show.s01e01.title.mkv"))
        self.assertTrue(pattern.match("show_s01e01.mkv"))
        self.assertTrue(pattern.match("SHOW.S01E01.TITLE.MKV"))

        # Should not match
        self.assertFalse(pattern.match("not_a_show.mkv"))
        self.assertFalse(pattern.match("show.mkv"))


class TestMetrics(unittest.TestCase):
    """Test metrics tracking."""

    def test_metrics_initialization(self):
        """Test metrics class initialization."""
        from perpetual import Metrics
        m = Metrics()
        self.assertEqual(m.ipc_reconnections, 0)
        self.assertEqual(m.ipc_failures, 0)
        self.assertEqual(m.playlist_syncs, 0)
        self.assertGreater(m.start_time, 0)

    def test_metrics_uptime(self):
        """Test uptime calculation."""
        from perpetual import Metrics
        import time
        m = Metrics()
        time.sleep(0.1)
        uptime = m.uptime()
        self.assertGreater(uptime, 0.09)


class TestConfig(unittest.TestCase):
    """Test configuration management."""

    def test_config_dataclass(self):
        """Test Config dataclass structure."""
        from perpetual import Config
        from dataclasses import fields

        field_names = {f.name for f in fields(Config)}
        required_fields = {
            'base_folder', 'days', 'filetypes', 'socket_path',
            'log_level', 'auto_launch_mpv', 'auto_restart_mpv'
        }
        self.assertTrue(required_fields.issubset(field_names))


class TestSignalHandling(unittest.TestCase):
    """Test signal handling."""

    def test_shutdown_flag(self):
        """Test shutdown flag mechanism."""
        from perpetual import should_shutdown, _shutdown_requested
        import perpetual

        # Initially should be False
        perpetual._shutdown_requested = False
        self.assertFalse(should_shutdown())

        # After setting, should be True
        perpetual._shutdown_requested = True
        self.assertTrue(should_shutdown())

        # Reset for other tests
        perpetual._shutdown_requested = False


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
