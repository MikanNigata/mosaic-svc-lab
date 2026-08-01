import tempfile
import unittest
from pathlib import Path

from mosaic_lab.experiment import plan_jobs


class ExperimentTests(unittest.TestCase):
    def test_plan_expands_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            config = {
                "experiment_id": "P1",
                "output_root": str(root / "out"),
                "seeds": [11],
                "sources": {"E1": "source.wav"},
                "references": {"P05": "prompt.wav"},
                "conditions": [
                    {
                        "id": "S0",
                        "backend": "seed-vc",
                        "reference": "P05",
                        "command": ["python", "runner.py", "{source}", "{reference}", "{output_file}", "{seed}"],
                    }
                ],
            }
            jobs = plan_jobs(config, config_path=config_path)
            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual(job.condition_id, "S0")
            self.assertEqual(job.seed, 11)
            self.assertEqual(job.command[-1], "11")
            self.assertTrue(str(job.output_file).endswith("output.wav"))
            self.assertTrue(job.output_file.is_relative_to(root / "out"))

    def test_disabled_condition_is_skipped(self) -> None:
        config = {
            "experiment_id": "P1",
            "output_root": "out",
            "sources": {"E1": "source.wav"},
            "references": {"P05": "prompt.wav"},
            "conditions": [
                {
                    "id": "disabled",
                    "backend": "hq-svc",
                    "reference": "P05",
                    "enabled": False,
                    "command": ["false"],
                },
                {
                    "id": "enabled",
                    "backend": "seed-vc",
                    "reference": "P05",
                    "command": ["true"],
                },
            ],
        }
        jobs = plan_jobs(config)
        self.assertEqual([job.condition_id for job in jobs], ["enabled"])


if __name__ == "__main__":
    unittest.main()
