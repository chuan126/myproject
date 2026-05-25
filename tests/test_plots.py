import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.plots import save_soc_curves_by_sequence


class SocCurvePlotTests(unittest.TestCase):
    def test_sequence_identifiers_create_portable_output_names(self) -> None:
        predictions = pd.DataFrame(
            {
                "sequence_id": ["folder/a:b?", "CON"],
                "time": [0.0, 0.0],
                "actual_soc": [0.4, 0.5],
                "predicted_soc": [0.45, 0.55],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_soc_curves_by_sequence(predictions, output_dir)
            names = sorted(path.name for path in output_dir.glob("*.png"))

        self.assertEqual(len(names), 2)
        self.assertIn("_CON_soc_over_time.png", names)
        self.assertTrue(all(not set('<>:"/\\|?*').intersection(name) for name in names))


if __name__ == "__main__":
    unittest.main()
