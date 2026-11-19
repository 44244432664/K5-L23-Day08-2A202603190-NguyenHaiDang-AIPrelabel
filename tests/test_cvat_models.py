"""Adapter/registration tests; no real images, models, CVAT tasks, or CUDA required."""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"day08_cvat_{name}", ROOT / "serverless" / "day08" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("main")
deployment = load_module("deploy")


class Coordinates:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.coordinates


def box(class_id=0, coordinates=None):
    return SimpleNamespace(
        cls=np.array(class_id), conf=np.array(0.8),
        xyxy=[Coordinates(coordinates if coordinates is not None else [1, 2, 20, 30])],
    )


class CvatModelAdapterTests(unittest.TestCase):
    def test_dashboard_accepts_empty_202_response(self):
        with patch.object(deployment.subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stdout="\n", stderr="",
        )):
            self.assertIsNone(deployment.dashboard("POST", body={}))

    def test_smoke_test_refuses_gold_and_batch_before_inference(self):
        for filename in ("img_001.jpg", "img_004.jpg"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "serverless/day08/smoke_test.py"), "--image", filename],
                capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("gold and batch are protected", result.stderr)

    def test_threshold_validation(self):
        for value in (0, 0.25, 1, "0.5"):
            self.assertEqual(adapter.threshold_value(value), float(value))
        for value in (-1, 2, math.inf, math.nan, "not a number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.threshold_value(value)

    def test_sam_single_object_prompt_batch(self):
        args = adapter.prompt_arguments({
            "pos_points": [[10, 20], [20, 30]], "neg_points": [[40, 50]],
            "obj_bbox": [[0, 0], [80, 90]],
        })
        self.assertEqual(args["points"], [[[10, 20], [20, 30], [40, 50]]])
        self.assertEqual(args["labels"], [[1, 1, 0]])
        self.assertEqual(args["bboxes"], [[0, 0, 80, 90]])

    def test_sam_flat_box_without_click(self):
        self.assertEqual(
            adapter.prompt_arguments({"obj_bbox": [0, 1, 20, 30]}),
            {"bboxes": [[0, 1, 20, 30]]},
        )

    def test_sam_rejects_invalid_prompts(self):
        for data in (
            {}, {"pos_points": [[10]]}, {"pos_points": [[math.nan, 20]]},
            {"obj_bbox": [[0, 0], [0, 5]]}, {"obj_bbox": [1, 2, 3]},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                adapter.prompt_arguments(data)

    def test_coco_motorcycle_mapping_and_clipping(self):
        prediction = SimpleNamespace(
            names={0: "motorcycle", 1: "banana"},
            boxes=[box(coordinates=[-2, -3, 200, 300]), box(class_id=1)], masks=None,
        )
        shapes = adapter.detector_shapes(prediction, "yolo", 100, 80)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["label"], "motorbike")
        self.assertEqual(shapes[0]["points"], [0, 0, 100, 80])
        self.assertEqual(shapes[0]["type"], "rectangle")

    def test_three_wheeler_synonyms_and_polygons(self):
        for prompt in ("tuk-tuk", "cyclo", "cargo tricycle"):
            prediction = SimpleNamespace(
                names={0: prompt}, boxes=[box()],
                masks=SimpleNamespace(xy=[np.array([[1, 2], [20, 3], [10, 30]])]),
            )
            shapes = adapter.detector_shapes(prediction, "yoloe-polygon", 100, 80)
            self.assertEqual(shapes[0]["label"], "three_wheeler")
            self.assertEqual(shapes[0]["type"], "polygon")
            self.assertEqual(len(shapes[0]["points"]), 6)

    def test_polygon_does_not_silently_fall_back_to_box(self):
        prediction = SimpleNamespace(names={0: "car"}, boxes=[box()], masks=None)
        self.assertEqual(adapter.detector_shapes(prediction, "yoloe-polygon", 100, 80), [])

    def test_functions_request_one_gpu_and_do_not_collide_with_existing_ids(self):
        for entry in deployment.FUNCTIONS:
            config = deployment.configuration(*entry)
            self.assertTrue(config["metadata"]["name"].startswith("day08-"))
            self.assertEqual(config["metadata"]["annotations"]["skip-build"], "true")
            self.assertNotIn("image", config["spec"]["build"])
            self.assertEqual(config["metadata"]["labels"]["nuclio.io/project-name"], "cvat")
            self.assertEqual(config["spec"]["resources"]["limits"]["nvidia.com/gpu"], "1")
            self.assertEqual(config["spec"]["triggers"]["http"]["numWorkers"], 1)
            if entry[2] == "sam":
                self.assertEqual(config["metadata"]["annotations"]["version"], "1")
                self.assertEqual(config["metadata"]["annotations"]["type"], "interactor")


if __name__ == "__main__":
    unittest.main()
