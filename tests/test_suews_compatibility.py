# -*- coding: utf-8 -*-
"""
Tests for SUEWS compatibility - these calculations must work WITHOUT cfunits.
These mirror the actual usage patterns in SUEWS/SuPy utility functions:
- supy.util._atm: cal_des_dta, cal_qa, cal_dq, cal_rh, cal_lat_vap, cal_cp
- supy.util._gs: cal_rs_obs (uses atmosp via _atm functions)
- supy.util._era5: gen_df_diag_era5_csv (uses atmosp for humidity calcs)
"""
import unittest
import numpy as np
import pandas as pd


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

    def test_saturation_vapor_pressure_es(self):
        """Test es calculation - used in cal_des_dta.

        Pattern from supy.util._atm.cal_des_dta:
            ac("es", p=pa, T=ta + dta / 2)
        """
        from atmosp import calculate as ac
        ta = 300  # K
        pa = 101325  # Pa
        es = ac('es', p=pa, T=ta)
        # es at 300K should be around 3500 Pa
        self.assertGreater(es, 3000)
        self.assertLess(es, 4000)

    def test_specific_humidity_from_rh_theta(self):
        """Test qv from RH with theta - used in cal_qa.

        Pattern from supy.util._atm.cal_qa:
            ac("qv", RH=rh_pct, p=pres_hPa * 100, theta=theta_K)
        """
        from atmosp import calculate as ac
        rh_pct = 60.0
        pres_hPa = 1013.25
        theta_K = 293.15  # ~20°C
        qa = ac('qv', RH=rh_pct, p=pres_hPa * 100, theta=theta_K)
        self.assertGreater(qa, 0)
        self.assertLess(qa, 0.05)

    def test_saturation_specific_humidity_qvs(self):
        """Test qvs calculation - used in cal_dq.

        Pattern from supy.util._atm.cal_dq:
            ac("qvs", T=ta_k, p=pa)
        """
        from atmosp import calculate as ac
        ta_k = 293.31  # 20°C + 273.16
        pa = 101325
        qvs = ac('qvs', T=ta_k, p=pa)
        self.assertGreater(qvs, 0)
        self.assertLess(qvs, 0.05)

    def test_specific_humidity_from_rh_T(self):
        """Test qv from RH with T - used in cal_dq.

        Pattern from supy.util._atm.cal_dq:
            ac("qv", T=ta_k, p=pa, RH=rh_pct)
        """
        from atmosp import calculate as ac
        ta_k = 293.31
        pa = 101325
        rh_pct = 60.0
        qv = ac('qv', T=ta_k, p=pa, RH=rh_pct)
        self.assertGreater(qv, 0)
        self.assertLess(qv, 0.05)

    def test_relative_humidity_from_qv_theta(self):
        """Test RH from qv with theta - used in cal_rh.

        Pattern from supy.util._atm.cal_rh:
            ac("RH", qv=qa_kgkg, p=pres_hPa * 100, theta=theta_K)
        """
        from atmosp import calculate as ac
        qa_kgkg = 0.01
        theta_K = 293.15
        pres_hPa = 1013.25
        RH = ac('RH', qv=qa_kgkg, p=pres_hPa * 100, theta=theta_K)
        self.assertGreater(RH, 0)
        self.assertLess(RH, 100)

    def test_relative_humidity_from_qv_T(self):
        """Test RH from qv with T - used in cal_cp.

        Pattern from supy.util._atm.cal_cp:
            ac("RH", qv=qa_kgkg, T=ta_K, p=pres_hPa * 100)
        """
        from atmosp import calculate as ac
        qa_kgkg = 0.01
        ta_K = 293.15
        pres_hPa = 1013.25
        RH = ac('RH', qv=qa_kgkg, T=ta_K, p=pres_hPa * 100)
        self.assertGreater(RH, 0)
        self.assertLess(RH, 100)

    def test_wet_bulb_temperature(self):
        """Test Tw calculation - used in cal_lat_vap.

        Pattern from supy.util._atm.cal_lat_vap:
            ac("Tw", qv=qa_kgkg, p=pres_hPa * 100, theta=theta_K,
               remove_assumptions=("constant Lv"))
        """
        from atmosp import calculate as ac
        qa_kgkg = 0.01
        theta_K = 298.15
        pres_hPa = 1013.25
        tw = ac(
            'Tw',
            qv=qa_kgkg,
            p=pres_hPa * 100,
            theta=theta_K,
            remove_assumptions=('constant Lv',)
        )
        self.assertLess(tw, theta_K)
        self.assertGreater(tw, 250)

    def test_humidity_round_trip(self):
        """Test that qv -> RH -> qv gives consistent results.

        Mirrors test_util_atm.py::test_humidity_round_trip
        """
        from atmosp import calculate as ac

        # Standard conditions from SUEWS tests
        TA_K = 20.0 + 273.15
        RH_orig = 60.0
        PRES_HPA = 1013.25

        # qa -> RH -> qa round trip
        qa = ac('qv', RH=RH_orig, p=PRES_HPA * 100, theta=TA_K)
        rh_calc = ac('RH', qv=qa, p=PRES_HPA * 100, theta=TA_K)

        self.assertAlmostEqual(RH_orig, rh_calc, delta=1.0)

    def test_humidity_budget_closure(self):
        """Test that qa + dq ≈ qa_sat (humidity budget closes).

        Mirrors test_util_atm.py::test_humidity_consistency
        """
        from atmosp import calculate as ac

        TA_C = 20.0
        TA_K = TA_C + 273.15
        RH = 60.0
        PRES_HPA = 1013.25

        # Replicate cal_qa
        qa = ac('qv', RH=RH, p=PRES_HPA * 100, theta=TA_K)

        # Replicate cal_dq
        ta_k = TA_C + 273.16
        pa = PRES_HPA * 100
        dq = ac('qvs', T=ta_k, p=pa) - ac('qv', T=ta_k, p=pa, RH=RH)

        # Replicate qa_sat
        qa_sat = ac('qv', RH=100.0, p=PRES_HPA * 100, theta=TA_K)

        # Budget should close
        self.assertAlmostEqual(qa + dq, qa_sat, delta=1e-3)

    def test_des_dta_pattern(self):
        """Test the derivative pattern used in cal_des_dta.

        Pattern from supy.util._atm.cal_des_dta:
            des = ac("es", p=pa, T=ta + dta / 2) - ac("es", p=pa, T=ta - dta / 2)
        """
        from atmosp import calculate as ac
        ta = 300
        pa = 101325
        dta = 1.0
        es_plus = ac('es', p=pa, T=ta + dta / 2)
        es_minus = ac('es', p=pa, T=ta - dta / 2)
        des_dta = (es_plus - es_minus) / dta
        self.assertGreater(des_dta, 0)

    def test_clausius_clapeyron(self):
        """Test that saturation vapour pressure slope increases with temperature.

        Mirrors test_util_atm.py::test_clausius_clapeyron
        """
        from atmosp import calculate as ac

        temps_k = np.array([273.15, 283.15, 293.15, 303.15, 313.15])
        pa = 101325
        dta = 1.0

        des_dta_values = []
        for ta in temps_k:
            es_plus = ac('es', p=pa, T=ta + dta / 2)
            es_minus = ac('es', p=pa, T=ta - dta / 2)
            des_dta_values.append((es_plus - es_minus) / dta)

        des_dta = np.array(des_dta_values)
        # Clausius-Clapeyron: slope should increase with temperature
        self.assertTrue(np.all(np.diff(des_dta) > 0))

    def test_latent_heat_range(self):
        """Test latent heat of vaporisation is in expected range.

        Mirrors test_util_atm.py::test_latent_heat_range
        Replicates supy.util._atm.cal_lat_vap
        """
        from atmosp import calculate as ac

        TA_K = 20.0 + 273.15
        RH = 60.0
        PRES_HPA = 1013.25

        qa = ac('qv', RH=RH, p=PRES_HPA * 100, theta=TA_K)
        tw = ac(
            'Tw',
            qv=qa,
            p=PRES_HPA * 100,
            theta=TA_K,
            remove_assumptions=('constant Lv',)
        )
        Lv = 2.501e6 - 2370.0 * (tw - 273.15)

        # Typical range: 2.4-2.6 MJ/kg
        self.assertGreater(Lv, 2.4e6)
        self.assertLess(Lv, 2.6e6)

    def test_humidity_extremes_zero(self):
        """Test calculations at 0% humidity.

        Mirrors test_util_atm.py::test_humidity_extremes[0.0-zero_humidity]
        """
        from atmosp import calculate as ac

        TA_K = 20.0 + 273.15
        TA_C = 20.0
        PRES_HPA = 1013.25

        qa = ac('qv', RH=0.0, p=PRES_HPA * 100, theta=TA_K)

        ta_k = TA_C + 273.16
        pa = PRES_HPA * 100
        dq = ac('qvs', T=ta_k, p=pa) - ac('qv', T=ta_k, p=pa, RH=0.0)

        self.assertLess(abs(qa), 1e-10)
        self.assertGreater(dq, 0)

    def test_humidity_extremes_saturated(self):
        """Test calculations at 100% humidity.

        Mirrors test_util_atm.py::test_humidity_extremes[100.0-saturated]
        """
        from atmosp import calculate as ac

        TA_C = 20.0
        PRES_HPA = 1013.25

        ta_k = TA_C + 273.16
        pa = PRES_HPA * 100
        dq = ac('qvs', T=ta_k, p=pa) - ac('qv', T=ta_k, p=pa, RH=100.0)

        self.assertLess(abs(dq), 1e-6)

    def test_pressure_effect_sea_level(self):
        """Test calculations at sea level pressure.

        Mirrors test_util_atm.py::test_pressure_effect[1013.25-sea_level]
        """
        from atmosp import calculate as ac

        TA_K = 20.0 + 273.15
        RH = 60.0

        qa = ac('qv', RH=RH, p=1013.25 * 100, theta=TA_K)
        self.assertGreater(qa, 0)
        self.assertLess(qa, 0.05)

    def test_pressure_effect_high_altitude(self):
        """Test calculations at high altitude pressure (3000m).

        Mirrors test_util_atm.py::test_pressure_effect[700.0-high_altitude_3000m]
        """
        from atmosp import calculate as ac

        TA_K = 20.0 + 273.15
        RH = 60.0

        qa = ac('qv', RH=RH, p=700.0 * 100, theta=TA_K)
        self.assertGreater(qa, 0)
        self.assertLess(qa, 0.05)

    def test_pandas_series_input(self):
        """Test that pandas Series inputs work correctly.

        SUEWS uses pandas Series in cal_des_dta with time index.
        """
        from atmosp import calculate as ac

        temps_k = np.array([273.15, 283.15, 293.15, 303.15, 313.15])
        idx = pd.date_range('2023-01-01', periods=len(temps_k), freq='h')
        ta_series = pd.Series(temps_k, index=idx)

        pa = 101325
        dta = 1.0

        es_plus = ac('es', p=pa, T=ta_series + dta / 2)
        es_minus = ac('es', p=pa, T=ta_series - dta / 2)
        des_dta = (es_plus - es_minus) / dta

        # Result should be array-like with same length
        self.assertEqual(len(des_dta), len(temps_k))
        # All values should be positive (Clausius-Clapeyron)
        self.assertTrue(np.all(des_dta > 0))

    def test_numpy_array_input(self):
        """Test that numpy array inputs work correctly."""
        from atmosp import calculate as ac

        T_array = np.array([290, 295, 300, 305])
        p = 101325
        es_array = ac('es', T=T_array, p=p)

        self.assertEqual(es_array.shape, T_array.shape)
        # es should increase with temperature
        self.assertTrue(np.all(np.diff(es_array) > 0))

    def test_scalar_input(self):
        """Test that scalar inputs return scalars."""
        from atmosp import calculate as ac

        es = ac('es', T=300.0, p=101325.0)
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
