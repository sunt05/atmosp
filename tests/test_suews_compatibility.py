# -*- coding: utf-8 -*-
"""
Tests for SUEWS compatibility - these calculations must work WITHOUT cfunits.
These mirror the actual usage patterns in SUEWS/SuPy.
"""
import unittest
import numpy as np


class TestSUEWSCompatibility(unittest.TestCase):
    """
    Tests for SUEWS compatibility - these calculations must work WITHOUT cfunits.
    These mirror the actual usage patterns in SUEWS/SuPy.
    """

    def test_cfunits_optional_import(self):
        """Verify atmosp imports successfully regardless of cfunits availability."""
        from atmosp import calculate
        from atmosp.solve import _HAS_CFUNITS
        self.assertTrue(callable(calculate))
        # This test should pass whether cfunits is available or not
        self.assertIn(_HAS_CFUNITS, [True, False])

    def test_saturation_vapor_pressure(self):
        """Test es calculation (used in cal_des_dta)."""
        from atmosp import calculate
        ta = 300  # K
        pa = 101325  # Pa
        es = calculate('es', p=pa, T=ta)
        # es at 300K should be around 3500 Pa
        self.assertGreater(es, 3000)
        self.assertLess(es, 4000)

    def test_specific_humidity_from_rh(self):
        """Test qv calculation from RH (used in cal_qa)."""
        from atmosp import calculate
        rh_pct = 50
        pres_hPa = 1013.25
        theta_K = 300
        qa = calculate('qv', RH=rh_pct, p=pres_hPa * 100, theta=theta_K)
        # Specific humidity should be positive and reasonable
        self.assertGreater(qa, 0)
        self.assertLess(qa, 0.05)  # Less than 50 g/kg

    def test_relative_humidity_from_qv(self):
        """Test RH calculation from qv (used in cal_rh)."""
        from atmosp import calculate
        qa = 0.01  # kg/kg
        theta_K = 300
        pres_hPa = 1013.25
        RH = calculate('RH', qv=qa, p=pres_hPa * 100, theta=theta_K)
        self.assertGreater(RH, 0)
        self.assertLess(RH, 100)

    def test_humidity_round_trip(self):
        """Test that qv -> RH -> qv gives consistent results."""
        from atmosp import calculate
        qa_orig = 0.012
        theta_K = 295
        pres_hPa = 1000
        # qv to RH
        RH = calculate('RH', qv=qa_orig, p=pres_hPa * 100, theta=theta_K)
        # RH back to qv
        qa_back = calculate('qv', RH=RH, p=pres_hPa * 100, theta=theta_K)
        self.assertAlmostEqual(qa_orig, qa_back, places=6)

    def test_wet_bulb_temperature(self):
        """Test Tw calculation (used in cal_lat_vap)."""
        from atmosp import calculate
        qa = 0.01
        theta_K = 300
        pres_hPa = 1013.25
        tw = calculate(
            'Tw',
            qv=qa,
            p=pres_hPa * 100,
            theta=theta_K,
            remove_assumptions=('constant Lv',)
        )
        # Wet bulb should be less than or equal to dry bulb
        self.assertLess(tw, theta_K)
        self.assertGreater(tw, 250)  # Reasonable range

    def test_saturation_specific_humidity(self):
        """Test qvs calculation (used in cal_dq)."""
        from atmosp import calculate
        ta_k = 298
        pa = 101325
        qvs = calculate('qvs', T=ta_k, p=pa)
        self.assertGreater(qvs, 0)
        self.assertLess(qvs, 0.05)

    def test_des_dta_pattern(self):
        """Test the derivative pattern used in cal_des_dta."""
        from atmosp import calculate
        ta = 300
        pa = 101325
        dta = 1.0
        es_plus = calculate('es', p=pa, T=ta + dta / 2)
        es_minus = calculate('es', p=pa, T=ta - dta / 2)
        des_dta = (es_plus - es_minus) / dta
        # Slope should be positive (es increases with T)
        self.assertGreater(des_dta, 0)

    def test_array_input(self):
        """Test that array inputs work correctly."""
        from atmosp import calculate
        T_array = np.array([290, 295, 300, 305])
        p = 101325
        es_array = calculate('es', T=T_array, p=p)
        self.assertEqual(es_array.shape, T_array.shape)
        # es should increase with temperature
        self.assertTrue(np.all(np.diff(es_array) > 0))

    def test_scalar_input(self):
        """Test that scalar inputs return scalars."""
        from atmosp import calculate
        es = calculate('es', T=300.0, p=101325.0)
        self.assertIsInstance(es, float)

    def test_unit_conversion_raises_helpful_error(self):
        """Test that unit conversion without cfunits gives helpful error."""
        from atmosp import calculate
        from atmosp.solve import _HAS_CFUNITS
        if not _HAS_CFUNITS:
            with self.assertRaises(ImportError) as context:
                calculate('es', T=25, T_unit='degC', p=101325)
            self.assertIn('cfunits', str(context.exception))
            self.assertIn('pip install', str(context.exception))


if __name__ == '__main__':
    unittest.main()
