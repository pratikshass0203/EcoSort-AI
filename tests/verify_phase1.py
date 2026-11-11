"""
Zero-dependency Phase 1 Verification Script for EcoSort AI
Uses strictly Python 3.13 standard library (no third-party dependencies required).
"""

import os
import sys
import unittest
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestPhase1Foundation(unittest.TestCase):
    def test_python_version(self):
        """Verify the virtual environment interpreter is Python 3.13.x."""
        v = sys.version_info
        self.assertEqual(v.major, 3, "Major Python version must be 3")
        self.assertEqual(v.minor, 13, f"Expected Python 3.13, found {v.major}.{v.minor}.{v.micro}")

    def test_directory_hierarchy(self):
        """Verify all planned directories exist."""
        required_dirs = [
            "configs",
            os.path.join("data", "raw"),
            os.path.join("data", "processed", "train", "biodegradable"),
            os.path.join("data", "processed", "train", "recyclable"),
            os.path.join("data", "processed", "train", "electronic"),
            os.path.join("data", "processed", "train", "hazardous"),
            os.path.join("data", "samples"),
            "docs",
            os.path.join("models", "checkpoints"),
            os.path.join("models", "exported"),
            os.path.join("src", "vision"),
            os.path.join("src", "knowledge"),
            os.path.join("src", "llm"),
            os.path.join("src", "edge"),
            os.path.join("src", "utils"),
            "app",
            "tests",
        ]
        for d in required_dirs:
            p = os.path.join(PROJECT_ROOT, d)
            self.assertTrue(os.path.isdir(p), f"Directory missing: {d}")

    def test_critical_files_exist(self):
        """Verify all essential foundational files exist."""
        required_files = [
            "LICENSE",
            "README.md",
            ".gitignore",
            "requirements.txt",
            "pyproject.toml",
            os.path.join("configs", "model_config.yaml"),
            os.path.join("configs", "disposal_rules.yaml"),
            os.path.join("docs", "architecture.md"),
            os.path.join("docs", "hardware_specs.md"),
            os.path.join(".github", "workflows", "ci.yml"),
        ]
        for f in required_files:
            p = os.path.join(PROJECT_ROOT, f)
            self.assertTrue(os.path.isfile(p), f"File missing: {f}")

    def test_gitkeep_files_preserved(self):
        """Verify .gitkeep files exist in empty tracking directories."""
        gitkeeps = [
            os.path.join("data", "raw", ".gitkeep"),
            os.path.join("data", "samples", ".gitkeep"),
            os.path.join("models", "checkpoints", ".gitkeep"),
            os.path.join("models", "exported", ".gitkeep"),
        ]
        for gk in gitkeeps:
            p = os.path.join(PROJECT_ROOT, gk)
            self.assertTrue(os.path.isfile(p), f".gitkeep missing in: {gk}")

    def test_model_config_content(self):
        """Verify model_config.yaml has required model parameters and thresholds."""
        cfg_path = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("mobilenet_v2", content)
        self.assertIn("confidence_threshold_tau: 0.70", content)
        self.assertIn("ambiguity_margin: 0.15", content)
        self.assertIn("biodegradable", content)
        self.assertIn("recyclable", content)
        self.assertIn("electronic", content)
        self.assertIn("hazardous", content)

    def test_disposal_rules_safety(self):
        """Verify disposal_rules.yaml contains strict safety alerts for hazardous waste."""
        rules_path = os.path.join(PROJECT_ROOT, "configs", "disposal_rules.yaml")
        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("MANDATORY SAFETY WARNING", content)
        self.assertIn("Green", content)
        self.assertIn("Blue", content)
        self.assertIn("Red / Black", content)

    def test_requirements_pins(self):
        """Verify exact version pins in requirements.txt (no unpinned minimums)."""
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Must include exact pins
        self.assertIn("torch==2.6.0+cpu", lines)
        self.assertIn("torchvision==0.21.0+cpu", lines)
        self.assertIn("numpy==2.1.3", lines)
        # Torchaudio must NOT be present
        for line in lines:
            self.assertFalse(line.startswith("torchaudio"), "torchaudio must be removed")


if __name__ == "__main__":
    unittest.main()

