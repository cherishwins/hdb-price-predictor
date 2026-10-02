"""End-to-end checks that model.bst, scaler.joblib and FEATURE_NAMES agree.

xgboost accepts a feature vector of the wrong length or order without
complaint, so a layout mismatch never raises: it just prices every flat
wrong. These tests drive the real app and check the price it shows for a
transaction whose sale price is public record.

Run from the repo root with the app's own dependencies (no extra packages):
    python -m unittest discover -s tests -v
"""
import os
import re
import shutil
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILES = ["model.bst", "scaler.joblib", "postal_data.json"]

# Blk 163 Simei Road, Tampines (postal 520163): 4 ROOM, Model A, 104 sqm,
# storey 07 TO 09, lease from 1989, resold June 2025 for S$645,000
# (data.gov.sg, "Resale flat prices based on registration date from
# Jan-2017 onwards"). The model was trained on data up to 30 Jun 2025.
KNOWN_PRICE = 645_000
TOLERANCE = 0.15


def run_app(app_dir):
    cwd = os.getcwd()
    os.chdir(app_dir)  # app.py loads its model and data by relative path
    try:
        at = AppTest.from_file(os.path.join(app_dir, "app.py"), default_timeout=120)
        return at, at.run()
    finally:
        os.chdir(cwd)


def rerun(at, app_dir):
    cwd = os.getcwd()
    os.chdir(app_dir)
    try:
        return at.run()
    finally:
        os.chdir(cwd)


def by_label(elements, label):
    return next(e for e in elements if e.label == label)


class KnownTransactionTest(unittest.TestCase):
    def test_app_starts_without_layout_error(self):
        _, at = run_app(REPO)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([e.value for e in at.error], [])
        self.assertEqual(by_label(at.selectbox, "Flat Model").value, "IMPROVED")

    def test_known_transaction_is_priced_sanely(self):
        at, _ = run_app(REPO)
        at.toggle(key="postal_toggle").set_value(True)
        rerun(at, REPO)
        at.text_input(key="postal_code_input").input("520163")
        rerun(at, REPO)
        self.assertEqual(by_label(at.selectbox, "Town").value, "TAMPINES")
        self.assertEqual(at.number_input(key="lease_year_input").value, 1989)

        at.selectbox(key="sale_year_select").set_value(2025)
        at.selectbox(key="sale_month_select").set_value("June")
        by_label(at.number_input, "Floor Area (sqm)").set_value(104.0)
        by_label(at.number_input, "Storey (Average)").set_value(8.0)
        by_label(at.selectbox, "Flat Type").set_value("4 ROOM")
        by_label(at.selectbox, "Flat Model").set_value("MODEL A")
        by_label(at.button, "🔮 Predict Resale Price").click()
        rerun(at, REPO)

        self.assertEqual([e.value for e in at.error], [])
        prices = [m.value for m in at.markdown if "Predicted Resale Price" in m.value]
        self.assertEqual(len(prices), 1, "no prediction card rendered")
        price = float(re.search(r"S\$ ([\d,]+\.\d+)", prices[0]).group(1).replace(",", ""))
        low, high = KNOWN_PRICE * (1 - TOLERANCE), KNOWN_PRICE * (1 + TOLERANCE)
        self.assertTrue(low <= price <= high, f"predicted S$ {price:,.0f}, sold for S$ {KNOWN_PRICE:,}")


class LayoutMismatchTest(unittest.TestCase):
    def test_feature_layout_mismatch_is_fatal(self):
        # The same app with one feature dropped from FEATURE_NAMES: the
        # model still expects 60, so the app must refuse to serve.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        for name in DATA_FILES:
            os.symlink(os.path.join(REPO, name), os.path.join(tmp, name))
        with open(os.path.join(REPO, "app.py")) as f:
            src = f.read()
        layout = "FEATURE_NAMES = ['floor_area_sqm', 'postal', "
        self.assertTrue(layout in src, "FEATURE_NAMES no longer starts as expected")
        src = src.replace(layout, "FEATURE_NAMES = ['floor_area_sqm', ", 1)
        with open(os.path.join(tmp, "app.py"), "w") as f:
            f.write(src)

        _, at = run_app(tmp)
        errors = [e.value for e in at.error]
        self.assertTrue(any("Model expects 60 features but the app builds 59" in e for e in errors), errors)
        self.assertEqual(len(at.button), 0, "app kept serving after a layout mismatch")


if __name__ == "__main__":
    unittest.main()
