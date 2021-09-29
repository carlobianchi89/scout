#!/usr/bin/env python3

""" Tests for running the measure preparation routine """

# Import code to be tested
import ecm_prep
# Import needed packages
import unittest
import numpy
import os
from collections import OrderedDict
import warnings
import copy
import itertools


class CommonMethods(object):
    """Define common methods for use in all tests below."""

    def dict_check(self, dict1, dict2):
        """Check the equality of two dicts.

        Args:
            dict1 (dict): First dictionary to be compared
            dict2 (dict): Second dictionary to be compared

        Raises:
            AssertionError: If dictionaries are not equal.
        """
        # zip() and zip_longest() produce tuples for the items
        # identified, where in the case of a dict, the first item
        # in the tuple is the key and the second item is the value;
        # in the case where the dicts are not of identical size,
        # zip_longest() will use the fill value created below as a
        # substitute in the dict that has missing content; this
        # value is given as a tuple to be of comparable structure
        # to the normal output from zip_longest()
        fill_val = ('substituted entry', 5.2)

        # In this structure, k and k2 are the keys that correspond to
        # the dicts or unitary values that are found in i and i2,
        # respectively, at the current level of the recursive
        # exploration of dict1 and dict2, respectively
        for (k, i), (k2, i2) in itertools.zip_longest(sorted(dict1.items()),
                                                      sorted(dict2.items()),
                                                      fillvalue=fill_val):

            # Confirm that at the current location in the dict structure,
            # the keys are equal; this should fail if one of the dicts
            # is empty, is missing section(s), or has different key names
            self.assertEqual(k, k2)

            # If the recursion has not yet reached the terminal/leaf node
            if isinstance(i, dict):
                # Test that the dicts from the current keys are equal
                self.assertCountEqual(i, i2)
                # Continue to recursively traverse the dict
                self.dict_check(i, i2)
            # At the terminal/leaf node, formatted as a numpy array or list
            # (for time sensitive valuation test cases)
            elif isinstance(i, numpy.ndarray) or isinstance(i, list):
                self.assertTrue(type(i) == type(i2) and len(i) == len(i2))
                for x in range(0, len(i)):
                    # Handle lists of strings
                    if isinstance(i[x], str):
                        self.assertEqual(i[x], i2[x])
                    # Ensure that very small numbers do not reduce to 0
                    elif round(i[x], 5) != 0:
                        self.assertAlmostEqual(i[x], i2[x], places=5)
                    else:
                        self.assertAlmostEqual(i[x], i2[x], places=10)
            elif isinstance(i, str):
                self.assertEqual(i, i2)
            # At the terminal/leaf node, formatted as a point value
            else:
                # Compare the values, allowing for floating point inaccuracy.
                # Ensure that very small numbers do not reduce to 0
                if round(i, 2) != 0:
                    self.assertAlmostEqual(i, i2, places=2)
                else:
                    self.assertAlmostEqual(i, i2, places=10)



class UserOptions(object):
    """Generate sample user-specified execution options."""
    def __init__(self, site, capt, regions, tsv_metrics,
                 sect_shapes, rp_persist, health_costs, warnings):
        # Options include site energy outputs, captured energy site-source
        # calculation method, alternate regions, time sensitive output metrics,
        # sector-level load shapes, and verbose mode that prints all warnings
        self.site_energy = site
        self.captured_energy = capt
        self.alt_regions = regions
        self.rp_persist = rp_persist
        self.verbose = warnings
        self.tsv_metrics = tsv_metrics
        self.health_costs = health_costs
        self.sect_shapes = sect_shapes


class PartitionMicrosegmentTest(unittest.TestCase, CommonMethods):
    """Test the operation of the 'partition_microsegment' function.

    Ensure that the function properly partitions an input microsegment
    to yield the required total, competed, and efficient stock, energy,
    carbon and cost information.

    Attributes:
        opts (object): Stores user-specified execution options.
        time_horizons (list): A series of modeling time horizons to use
            in the various test functions of the class.
        handyfiles (object): Global input file data to use for the test.
        handyvars (object): Global variables to use for the test.
        sample_measure_in (dict): Sample measure attributes.
        ok_diffuse_params_in (NoneType): Placeholder for eventual technology
            diffusion parameters to be used in 'adjusted adoption' scenario.
        ok_mskeys_in (list): Sample key chains associated with the market
            microsegment being partitioned by the function.
        ok_mkt_scale_frac_in (float): Sample market microsegment scaling
            factor.
        ok_tsv_scale_fracs_in (dict): Sample time sensitive valuation scaling
            fractions.
        ok_newbldg_frac_in (list): Sample fraction of the total stock that
            is new construction, by year.
        ok_stock_in (list): Sample baseline microsegment stock data, by year.
        ok_energy_in (list): Sample baseline microsegment energy data, by year.
        ok_carb_in (list): Sample baseline microsegment carbon data, by year.
        ok_base_cost_in (list): Sample baseline technology unit costs, by year.
        ok_cost_meas_in (list): Sample measure unit costs.
        ok_cost_energy_base_in (numpy.ndarray): Sample baseline fuel costs.
        ok_cost_energy_meas_in (numpy.ndarray): Sample measure fuel costs.
        ok_relperf_in (list): Sample measure relative performance values.
        ok_life_base_in (dict): Sample baseline technology lifetimes, by year.
        ok_life_meas_in (int): Sample measure lifetime.
        ok_ssconv_base_in (numpy.ndarray): Sample baseline fuel site-source
            conversions, by year.
        ok_ssconv_meas_in (numpy.ndarray): Sample measure fuel site-source
            conversions, by year.
        ok_carbint_base_in (numpy.ndarray): Sample baseline fuel carbon
            intensities, by year.
        ok_carbint_meas_in (numpy.ndarray): Sample measure fuel carbon
            intensities, by year.
        ok_tsv_scale_fracs_in (dict): Annual scaling factors for time sensitive
            valuation adjustments to energy, cost, and carbon data
        ok_tsv_shapes_in (boolean): Hourly scaling factors used to generate
            sector-level baseline and efficient energy load profiles
        ok_out (list): Outputs that should be yielded by the function given
            valid inputs.
    """

    @classmethod
    def setUpClass(cls):
        """Define variables and objects for use across all class functions."""
        cls.opts = UserOptions(
            site=None, capt=None, regions=None, tsv_metrics=None,
            sect_shapes=None, rp_persist=None, health_costs=None,
            warnings=None)
        cls.time_horizons = ["2009", "2010", "2011"]
        # Base directory
        base_dir = os.getcwd()
        cls.handyfiles = ecm_prep.UsefulInputFiles(capt_energy=None,
                                                   regions="AIA")
        cls.handyvars = ecm_prep.UsefulVars(base_dir, cls.handyfiles,
                                            regions="AIA", tsv_metrics=None,
                                            health_costs=None)
        cls.handyvars.ccosts = numpy.array(
            (b'Test', 1, 4, 1), dtype=[
                ('Category', 'S11'), ('2009', '<f8'),
                ('2010', '<f8'), ('2011', '<f8')])
        sample_measure_base = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficients":1,
            "retro_rate": 0.02}
        sample_measure_single_coeff = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficients":0.5,
            "retro_rate": 0.02}
        sample_measure_multiple_coeffs = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficients": {
                "2009": 0.9,
                "2010": 0.7,
                "2011": 0.5,
                "2012": 0.3
              },
            "retro_rate": 0.02}
        sample_measure_negative_coeff = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficients":-0.5,
            "retro_rate": 0.02}
        sample_measure_bad_string = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficients":'a',
            "retro_rate": 0.02}
        sample_measure_valid_string = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficients":'0.5',
            "retro_rate": 0.02}
        sample_measure_wrong_name = {
            "name": "sample measure 1 partition",
            "active": 1,
            "market_entry_year": None,
            "market_exit_year": None,
            "market_scaling_fractions": None,
            "market_scaling_fractions_source": None,
            "measure_type": "full service",
            "structure_type": ["new", "existing"],
            "climate_zone": ["AIA_CZ1", "AIA_CZ2"],
            "bldg_type": ["single family home"],
            "fuel_type": {
                "primary": ["electricity"],
                "secondary": None},
            "fuel_switch_to": None,
            "end_use": {
                "primary": ["heating", "cooling"],
                "secondary": None},
            "technology": {
                "primary": ["resistance heat", "ASHP", "GSHP", "room AC"],
                "secondary": None},
            "diffusion_coefficient":0.5,
            "retro_rate": 0.02}

        cls.measure_instance_base = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_base)
        cls.measure_instance_single_coeff = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_single_coeff)
        cls.measure_instance_multiple_coeffs = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_multiple_coeffs)
        cls.measure_instance_negative_coeff = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_negative_coeff)
        cls.measure_instance_bad_string = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_bad_string)
        cls.measure_instance_valid_string = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_valid_string)
        cls.measure_instance_wrong_name = ecm_prep.Measure(
            base_dir, cls.handyvars, cls.handyfiles, site_energy=None,
            capt_energy=None, regions="AIA", tsv_metrics=None,
            health_costs=None,
            **sample_measure_wrong_name)
        cls.ok_diffuse_params_in = None
        cls.ok_mskeys_in = [
            ('primary', 'AIA_CZ1', 'single family home',
             'electricity', 'heating', 'supply', 'resistance heat',
             'new'),
            ('primary', 'AIA_CZ1', 'single family home',
             'electricity', 'heating', 'supply', 'resistance heat',
             'existing')]
        cls.ok_mkt_scale_frac_in = 1
        cls.ok_new_bldg_constr = {
            "annual new": {"2009": 10, "2010": 5, "2011": 10},
            "total new": {"2009": 10, "2010": 15, "2011": 25}}
        cls.ok_stock_in = {"2009": 100, "2010": 200, "2011": 300}
        cls.ok_energy_scnd_in = {"2009": 10, "2010": 20, "2011": 30}
        cls.ok_energy_in = {"2009": 10, "2010": 20, "2011": 30}
        cls.ok_carb_in = {"2009": 30, "2010": 60, "2011": 90}
        cls.ok_base_cost_in = {"2009": 10, "2010": 10, "2011": 10}
        cls.ok_cost_meas_in = 20
        cls.ok_cost_energy_base_in, cls.ok_cost_energy_meas_in = \
            (numpy.array((b'Test', 1, 2, 2),
                         dtype=[('Category', 'S11'), ('2009', '<f8'),
                                ('2010', '<f8'), ('2011', '<f8')])
             for n in range(2))
        cls.ok_relperf_in = {"2009": 0.30, "2010": 0.30, "2011": 0.30}
        cls.ok_life_base_in = {"2009": 10, "2010": 10, "2011": 10}
        cls.ok_life_meas_in = 10
        cls.ok_ssconv_base_in, cls.ok_ssconv_meas_in = \
            (numpy.array((b'Test', 1, 1, 1),
                         dtype=[('Category', 'S11'), ('2009', '<f8'),
                                ('2010', '<f8'), ('2011', '<f8')])
             for n in range(2))
        cls.ok_carbint_base_in, cls.ok_carbint_meas_in = \
            (numpy.array((b'Test', 1, 1, 1),
                         dtype=[('Category', 'S11'), ('2009', '<f8'),
                                ('2010', '<f8'), ('2011', '<f8')])
             for n in range(2))
        cls.ok_tsv_scale_fracs_in = {
          "energy": {"baseline": 1, "efficient": 1},
          "cost": {"baseline": 1, "efficient": 1},
          "carbon": {"baseline": 1, "efficient": 1}}
        cls.ok_tsv_shapes_in = None
        cls.ok_out_base = \
                [[[{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
                {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
                {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 2000, '2010': 4000, '2011': 6000}, 
                {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
                {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 2000, '2010': 4000, '2011': 6000}, 
                {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
                {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}],
                [{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
                {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
                {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 2000, '2010': 4000, '2011': 6000}, 
                {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
                {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 2000, '2010': 4000, '2011': 6000}, 
                {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
                {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}]],
                [[{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 100, '2010': 200.0, '2011': 300.0}, 
                {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
                {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000007}, 
                {'2009': 100, '2010': 100.0, '2011': 100.0}, 
                {'2009': 10, '2010': 10.0, '2011': 10.0}, 
                {'2009': 30, '2010': 30.0, '2011': 30.0}, 
                {'2009': 100, '2010': 100.0, '2011': 100.0}, 
                {'2009': 3.0, '2010': 3.0000000000000004, '2011': 3.0000000000000004}, 
                {'2009': 9.0, '2010': 9.000000000000002, '2011': 9.000000000000002}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 2000, '2010': 4000.0, '2011': 6000.0}, 
                {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
                {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000007}, 
                {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
                {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
                {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
                {'2009': 2000, '2010': 2000.0, '2011': 2000.0}, 
                {'2009': 3.0, '2010': 6.000000000000001, '2011': 6.000000000000001}, 
                {'2009': 9.0, '2010': 36.00000000000001, '2011': 9.000000000000002}],
                [{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 12.000000000000002, '2010': 48.00000000000001, '2011': 108.00000000000001}, 
                {'2009': 9.16, '2010': 16.841600000000003, '2011': 23.044800000000002}, 
                {'2009': 27.479999999999997, '2010': 50.5248, '2011': 69.1344}, 
                {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
                {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
                {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
                {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
                {'2009': 0.36000000000000004, '2010': 0.7200000000000002, '2011': 1.0800000000000003}, 
                {'2009': 1.08, '2010': 2.1600000000000006, '2011': 3.2400000000000007}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1120.0, '2010': 2480.0, '2011': 4080.0000000000005}, 
                {'2009': 9.16, '2010': 33.68320000000001, '2011': 46.089600000000004}, 
                {'2009': 27.479999999999997, '2010': 202.0992, '2011': 69.1344}, 
                {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
                {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
                {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
                {'2009': 240.00000000000003, '2010': 480.00000000000006, '2011': 720.0}, 
                {'2009': 0.36000000000000004, '2010': 1.4400000000000004, '2011': 2.1600000000000006}, 
                {'2009': 1.08, '2010': 8.640000000000002, '2011': 3.2400000000000007}]]]
        cls.ok_out_single_coeff = \
                [[[{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 50.0, '2010': 200, '2011': 300}, 
                {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
                {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 50.0, '2010': 100.0, '2011': 150.0}, 
                {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
                {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1500.0, '2010': 4000, '2011': 6000}, 
                {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
                {'2009': 19.5, '2010': 156.0, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1500.0, '2010': 3000.0, '2011': 4500.0}, 
                {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
                {'2009': 19.5, '2010': 156.0, '2011': 58.5}],
                [{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 50.0, '2010': 200, '2011': 300}, 
                {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
                {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 50.0, '2010': 100.0, '2011': 150.0}, 
                {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
                {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1500.0, '2010': 4000, '2011': 6000}, 
                {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
                {'2009': 19.5, '2010': 156.0, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1500.0, '2010': 3000.0, '2011': 4500.0}, 
                {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
                {'2009': 19.5, '2010': 156.0, '2011': 58.5}]], 
                [[{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 50.0, '2010': 100.0, '2011': 150.0}, 
                {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
                {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
                {'2009': 100, '2010': 100.0, '2011': 100.0}, 
                {'2009': 10, '2010': 10.0, '2011': 10.0}, 
                {'2009': 30, '2010': 30.0, '2011': 30.0}, 
                {'2009': 50.0, '2010': 50.0, '2011': 50.0}, 
                {'2009': 6.5, '2010': 6.5, '2011': 6.5}, 
                {'2009': 19.5, '2010': 19.5, '2011': 19.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1500.0, '2010': 3000.0, '2011': 4500.0}, 
                {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
                {'2009': 19.5, '2010': 156.0, '2011': 58.5}, 
                {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
                {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
                {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
                {'2009': 1500.0, '2010': 1500.0, '2011': 1500.0}, 
                {'2009': 6.5, '2010': 13.0, '2011': 13.0}, 
                {'2009': 19.5, '2010': 78.0, '2011': 19.5}],
                [{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 6.000000000000001, '2010': 24.000000000000004, '2011': 54.00000000000001}, 
                {'2009': 9.58, '2010': 18.4208, '2011': 26.5224}, 
                {'2009': 28.74, '2010': 55.2624, '2011': 79.5672}, 
                {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
                {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
                {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
                {'2009': 6.000000000000001, '2010': 12.000000000000002, '2011': 18.0}, 
                {'2009': 0.7800000000000001, '2010': 1.5600000000000003, '2011': 2.3400000000000003}, 
                {'2009': 2.34, '2010': 4.680000000000001, '2011': 7.0200000000000005}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1060.0, '2010': 2240.0, '2011': 3540.0}, 
                {'2009': 9.58, '2010': 36.8416, '2011': 53.0448}, 
                {'2009': 28.74, '2010': 221.0496, '2011': 79.5672}, 
                {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
                {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
                {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
                {'2009': 180.00000000000003, '2010': 360.00000000000006, '2011': 540.0}, 
                {'2009': 0.7800000000000001, '2010': 3.1200000000000006, '2011': 4.680000000000001}, 
                {'2009': 2.34, '2010': 18.720000000000002, '2011': 7.0200000000000005}]]]
        cls.ok_out_multiple_coeffs = \
                [[[{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 90.0, '2010': 200, '2011': 300}, 
                {'2009': 3.6999999999999993, '2010': 10.200000000000003, '2011': 19.5}, 
                {'2009': 11.099999999999998, '2010': 30.600000000000005, '2011': 58.5}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 90.0, '2010': 140.0, '2011': 150.0}, 
                {'2009': 3.6999999999999993, '2010': 10.200000000000003, '2011': 19.5}, 
                {'2009': 11.099999999999998, '2010': 30.600000000000005, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1900.0, '2010': 4000, '2011': 6000}, 
                {'2009': 3.6999999999999993, '2010': 20.400000000000006, '2011': 39.0}, 
                {'2009': 11.099999999999998, '2010': 122.40000000000002, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1900.0, '2010': 3400.0, '2011': 4500.0}, 
                {'2009': 3.6999999999999993, '2010': 20.400000000000006, '2011': 39.0}, 
                {'2009': 11.099999999999998, '2010': 122.40000000000002, '2011': 58.5}],
                [{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 90.0, '2010': 200, '2011': 300}, 
                {'2009': 3.6999999999999993, '2010': 10.200000000000003, '2011': 19.5}, 
                {'2009': 11.099999999999998, '2010': 30.600000000000005, '2011': 58.5}, 
                {'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 90.0, '2010': 140.0, '2011': 150.0}, 
                {'2009': 3.6999999999999993, '2010': 10.200000000000003, '2011': 19.5}, 
                {'2009': 11.099999999999998, '2010': 30.600000000000005, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1900.0, '2010': 4000, '2011': 6000}, 
                {'2009': 3.6999999999999993, '2010': 20.400000000000006, '2011': 39.0}, 
                {'2009': 11.099999999999998, '2010': 122.40000000000002, '2011': 58.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1900.0, '2010': 3400.0, '2011': 4500.0}, 
                {'2009': 3.6999999999999993, '2010': 20.400000000000006, '2011': 39.0}, 
                {'2009': 11.099999999999998, '2010': 122.40000000000002, '2011': 58.5}]], 
                [[{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 90.0, '2010': 160.0, '2011': 210.0}, 
                {'2009': 3.6999999999999993, '2010': 8.8, '2011': 15.3}, 
                {'2009': 11.099999999999998, '2010': 26.400000000000006, '2011': 45.900000000000006}, 
                {'2009': 100, '2010': 100.0, '2011': 100.0}, 
                {'2009': 10, '2010': 10.0, '2011': 10.0}, 
                {'2009': 30, '2010': 30.0, '2011': 30.0}, 
                {'2009': 90.0, '2010': 70.0, '2011': 50.0}, 
                {'2009': 3.6999999999999993, '2010': 5.100000000000001, '2011': 6.5}, 
                {'2009': 11.099999999999998, '2010': 15.300000000000002, '2011': 19.5}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1900.0, '2010': 3600.0, '2011': 5100.0}, 
                {'2009': 3.6999999999999993, '2010': 17.6, '2011': 30.6}, 
                {'2009': 11.099999999999998, '2010': 105.60000000000002, '2011': 45.900000000000006}, 
                {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
                {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
                {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
                {'2009': 1900.0, '2010': 1700.0, '2011': 1500.0}, 
                {'2009': 3.6999999999999993, '2010': 10.200000000000003, '2011': 13.0}, 
                {'2009': 11.099999999999998, '2010': 61.20000000000001, '2011': 19.5}],
                [{'2009': 100, '2010': 200, '2011': 300}, 
                {'2009': 10, '2010': 20, '2011': 30}, 
                {'2009': 30, '2010': 60, '2011': 90}, 
                {'2009': 10.8, '2010': 38.400000000000006, '2011': 75.60000000000001}, 
                {'2009': 9.244, '2010': 17.49344, '2011': 25.19184}, 
                {'2009': 27.732, '2010': 52.48032, '2011': 75.57552}, 
                {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
                {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
                {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
                {'2009': 10.8, '2010': 16.8, '2011': 18.0}, 
                {'2009': 0.44399999999999995, '2010': 1.2240000000000002, '2011': 2.3400000000000003}, 
                {'2009': 1.3319999999999999, '2010': 3.6720000000000006, '2011': 7.0200000000000005}, 
                {'2009': 1000, '2010': 2000, '2011': 3000}, 
                {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
                {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
                {'2009': 1108.0, '2010': 2384.0, '2011': 3756.0}, 
                {'2009': 9.244, '2010': 34.98688, '2011': 50.38368}, 
                {'2009': 27.732, '2010': 209.92128, '2011': 75.57552}, 
                {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
                {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
                {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
                {'2009': 228.0, '2010': 408.0, '2011': 540.0}, 
                {'2009': 0.44399999999999995, '2010': 2.4480000000000004, '2011': 4.680000000000001}, 
                {'2009': 1.3319999999999999, '2010': 14.688000000000002, '2011': 7.0200000000000005}]]]
        cls.ok_out_negative_coeff = \
            [[[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}]],
            [[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200.0, '2011': 300.0}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000007}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 10, '2010': 10.0, '2011': 10.0}, 
            {'2009': 30, '2010': 30.0, '2011': 30.0}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 3.0, '2010': 3.0000000000000004, '2011': 3.0000000000000004}, 
            {'2009': 9.0, '2010': 9.000000000000002, '2011': 9.000000000000002}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000.0, '2011': 6000.0}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000007}, 
            {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
            {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
            {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
            {'2009': 2000, '2010': 2000.0, '2011': 2000.0}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 6.000000000000001}, 
            {'2009': 9.0, '2010': 36.00000000000001, '2011': 9.000000000000002}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 12.000000000000002, '2010': 48.00000000000001, '2011': 108.00000000000001}, 
            {'2009': 9.16, '2010': 16.841600000000003, '2011': 23.044800000000002}, 
            {'2009': 27.479999999999997, '2010': 50.5248, '2011': 69.1344}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
            {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 0.36000000000000004, '2010': 0.7200000000000002, '2011': 1.0800000000000003}, 
            {'2009': 1.08, '2010': 2.1600000000000006, '2011': 3.2400000000000007}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1120.0, '2010': 2480.0, '2011': 4080.0000000000005}, 
            {'2009': 9.16, '2010': 33.68320000000001, '2011': 46.089600000000004}, 
            {'2009': 27.479999999999997, '2010': 202.0992, '2011': 69.1344}, 
            {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
            {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
            {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
            {'2009': 240.00000000000003, '2010': 480.00000000000006, '2011': 720.0}, 
            {'2009': 0.36000000000000004, '2010': 1.4400000000000004, '2011': 2.1600000000000006}, 
            {'2009': 1.08, '2010': 8.640000000000002, '2011': 3.2400000000000007}]]]
        cls.ok_out_bad_string = \
            [[[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}]],
            [[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200.0, '2011': 300.0}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000007}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 10, '2010': 10.0, '2011': 10.0}, 
            {'2009': 30, '2010': 30.0, '2011': 30.0}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 3.0, '2010': 3.0000000000000004, '2011': 3.0000000000000004}, 
            {'2009': 9.0, '2010': 9.000000000000002, '2011': 9.000000000000002}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000.0, '2011': 6000.0}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000007}, 
            {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
            {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
            {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
            {'2009': 2000, '2010': 2000.0, '2011': 2000.0}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 6.000000000000001}, 
            {'2009': 9.0, '2010': 36.00000000000001, '2011': 9.000000000000002}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 12.000000000000002, '2010': 48.00000000000001, '2011': 108.00000000000001}, 
            {'2009': 9.16, '2010': 16.841600000000003, '2011': 23.044800000000002}, 
            {'2009': 27.479999999999997, '2010': 50.5248, '2011': 69.1344}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
            {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 0.36000000000000004, '2010': 0.7200000000000002, '2011': 1.0800000000000003}, 
            {'2009': 1.08, '2010': 2.1600000000000006, '2011': 3.2400000000000007}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1120.0, '2010': 2480.0, '2011': 4080.0000000000005}, 
            {'2009': 9.16, '2010': 33.68320000000001, '2011': 46.089600000000004}, 
            {'2009': 27.479999999999997, '2010': 202.0992, '2011': 69.1344}, 
            {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
            {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
            {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
            {'2009': 240.00000000000003, '2010': 480.00000000000006, '2011': 720.0}, 
            {'2009': 0.36000000000000004, '2010': 1.4400000000000004, '2011': 2.1600000000000006}, 
            {'2009': 1.08, '2010': 8.640000000000002, '2011': 3.2400000000000007}]]]
        cls.ok_out_valid_string = \
            [[[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 50.0, '2010': 200, '2011': 300}, 
            {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
            {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 50.0, '2010': 100.0, '2011': 150.0}, 
            {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
            {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1500.0, '2010': 4000, '2011': 6000}, 
            {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
            {'2009': 19.5, '2010': 156.0, '2011': 58.5}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1500.0, '2010': 3000.0, '2011': 4500.0}, 
            {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
            {'2009': 19.5, '2010': 156.0, '2011': 58.5}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 50.0, '2010': 200, '2011': 300}, 
            {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
            {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 50.0, '2010': 100.0, '2011': 150.0}, 
            {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
            {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1500.0, '2010': 4000, '2011': 6000}, 
            {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
            {'2009': 19.5, '2010': 156.0, '2011': 58.5}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1500.0, '2010': 3000.0, '2011': 4500.0}, 
            {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
            {'2009': 19.5, '2010': 156.0, '2011': 58.5}]],
            [[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 50.0, '2010': 100.0, '2011': 150.0}, 
            {'2009': 6.5, '2010': 13.0, '2011': 19.5}, 
            {'2009': 19.5, '2010': 39.0, '2011': 58.5}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 10, '2010': 10.0, '2011': 10.0}, 
            {'2009': 30, '2010': 30.0, '2011': 30.0}, 
            {'2009': 50.0, '2010': 50.0, '2011': 50.0}, 
            {'2009': 6.5, '2010': 6.5, '2011': 6.5}, 
            {'2009': 19.5, '2010': 19.5, '2011': 19.5}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1500.0, '2010': 3000.0, '2011': 4500.0}, 
            {'2009': 6.5, '2010': 26.0, '2011': 39.0}, 
            {'2009': 19.5, '2010': 156.0, '2011': 58.5}, 
            {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
            {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
            {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
            {'2009': 1500.0, '2010': 1500.0, '2011': 1500.0}, 
            {'2009': 6.5, '2010': 13.0, '2011': 13.0}, 
            {'2009': 19.5, '2010': 78.0, '2011': 19.5}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 6.000000000000001, '2010': 24.000000000000004, '2011': 54.00000000000001}, 
            {'2009': 9.58, '2010': 18.4208, '2011': 26.5224}, 
            {'2009': 28.74, '2010': 55.2624, '2011': 79.5672}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
            {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
            {'2009': 6.000000000000001, '2010': 12.000000000000002, '2011': 18.0}, 
            {'2009': 0.7800000000000001, '2010': 1.5600000000000003, '2011': 2.3400000000000003}, 
            {'2009': 2.34, '2010': 4.680000000000001, '2011': 7.0200000000000005}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1060.0, '2010': 2240.0, '2011': 3540.0}, 
            {'2009': 9.58, '2010': 36.8416, '2011': 53.0448}, 
            {'2009': 28.74, '2010': 221.0496, '2011': 79.5672}, 
            {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
            {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
            {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
            {'2009': 180.00000000000003, '2010': 360.00000000000006, '2011': 540.0}, 
            {'2009': 0.7800000000000001, '2010': 3.1200000000000006, '2011': 4.680000000000001}, 
            {'2009': 2.34, '2010': 18.720000000000002, '2011': 7.0200000000000005}]]]
        cls.ok_out_wrong_name = \
            [[[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000, '2011': 6000}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000004}]],
            [[{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 100, '2010': 200.0, '2011': 300.0}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 9.000000000000002}, 
            {'2009': 9.0, '2010': 18.000000000000004, '2011': 27.000000000000007}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 10, '2010': 10.0, '2011': 10.0}, 
            {'2009': 30, '2010': 30.0, '2011': 30.0}, 
            {'2009': 100, '2010': 100.0, '2011': 100.0}, 
            {'2009': 3.0, '2010': 3.0000000000000004, '2011': 3.0000000000000004}, 
            {'2009': 9.0, '2010': 9.000000000000002, '2011': 9.000000000000002}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 2000, '2010': 4000.0, '2011': 6000.0}, 
            {'2009': 3.0, '2010': 12.000000000000002, '2011': 18.000000000000004}, 
            {'2009': 9.0, '2010': 72.00000000000001, '2011': 27.000000000000007}, 
            {'2009': 1000, '2010': 1000.0, '2011': 1000.0}, 
            {'2009': 10.0, '2010': 20.0, '2011': 20.0}, 
            {'2009': 30.0, '2010': 120.0, '2011': 30.0}, 
            {'2009': 2000, '2010': 2000.0, '2011': 2000.0}, 
            {'2009': 3.0, '2010': 6.000000000000001, '2011': 6.000000000000001}, 
            {'2009': 9.0, '2010': 36.00000000000001, '2011': 9.000000000000002}],
            [{'2009': 100, '2010': 200, '2011': 300}, 
            {'2009': 10, '2010': 20, '2011': 30}, 
            {'2009': 30, '2010': 60, '2011': 90}, 
            {'2009': 12.000000000000002, '2010': 48.00000000000001, '2011': 108.00000000000001}, 
            {'2009': 9.16, '2010': 16.841600000000003, '2011': 23.044800000000002}, 
            {'2009': 27.479999999999997, '2010': 50.5248, '2011': 69.1344}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 1.2000000000000002, '2010': 2.4000000000000004, '2011': 3.6}, 
            {'2009': 3.6, '2010': 7.2, '2011': 10.8}, 
            {'2009': 12.000000000000002, '2010': 24.000000000000004, '2011': 36.0}, 
            {'2009': 0.36000000000000004, '2010': 0.7200000000000002, '2011': 1.0800000000000003}, 
            {'2009': 1.08, '2010': 2.1600000000000006, '2011': 3.2400000000000007}, 
            {'2009': 1000, '2010': 2000, '2011': 3000}, 
            {'2009': 10.0, '2010': 40.0, '2011': 60.0}, 
            {'2009': 30.0, '2010': 240.0, '2011': 90.0}, 
            {'2009': 1120.0, '2010': 2480.0, '2011': 4080.0000000000005}, 
            {'2009': 9.16, '2010': 33.68320000000001, '2011': 46.089600000000004}, 
            {'2009': 27.479999999999997, '2010': 202.0992, '2011': 69.1344}, 
            {'2009': 120.00000000000001, '2010': 240.00000000000003, '2011': 360.0}, 
            {'2009': 1.2000000000000002, '2010': 4.800000000000001, '2011': 7.2}, 
            {'2009': 3.6, '2010': 28.8, '2011': 10.8}, 
            {'2009': 240.00000000000003, '2010': 480.00000000000006, '2011': 720.0}, 
            {'2009': 0.36000000000000004, '2010': 1.4400000000000004, '2011': 2.1600000000000006}, 
            {'2009': 1.08, '2010': 8.640000000000002, '2011': 3.2400000000000007}]]]



    def test_ok(self):
        """Test the 'partition_microsegment' function given valid inputs.

        Raises:
            AssertionError: If function yields unexpected results.
        """
        # Loop through 'ok_out' elements

        # Reset AEO time horizon and market entry/exit years
        self.measure_instance_base.handyvars.aeo_years = \
            self.time_horizons
        self.measure_instance_base.market_entry_year = \
            int(self.time_horizons[0])
        self.measure_instance_base.market_exit_year = \
            int(self.time_horizons[-1]) + 1
        # Reset AEO time horizon and market entry/exit years
        self.measure_instance_single_coeff.handyvars.aeo_years = \
            self.time_horizons
        self.measure_instance_single_coeff.market_entry_year = \
            int(self.time_horizons[0])
        self.measure_instance_single_coeff.market_exit_year = \
            int(self.time_horizons[-1]) + 1
        # Reset AEO time horizon and market entry/exit years
        self.measure_instance_multiple_coeffs.handyvars.aeo_years = \
            self.time_horizons
        self.measure_instance_multiple_coeffs.market_entry_year = \
            int(self.time_horizons[0])
        self.measure_instance_multiple_coeffs.market_exit_year = \
            int(self.time_horizons[-1]) + 1
        # Reset AEO time horizon and market entry/exit years
        self.measure_instance_valid_string.handyvars.aeo_years = \
            self.time_horizons
        self.measure_instance_valid_string.market_entry_year = \
            int(self.time_horizons[0])
        self.measure_instance_valid_string.market_exit_year = \
            int(self.time_horizons[-1]) + 1



        # Loop through two test schemes (Technical potential and Max
        # adoption potential)
        for scn in range(0, len(self.handyvars.adopt_schemes)):
            # Loop through two microsegment key chains (one applying
            # to new structure type, another to existing structure type)
            for k in range(0, len(self.ok_mskeys_in)):
                # List of output dicts generated by the function
                lists_base = self.measure_instance_base.partition_microsegment(
                    self.handyvars.adopt_schemes[scn],
                    self.ok_diffuse_params_in,
                    self.ok_mskeys_in[k],
                    self.ok_mkt_scale_frac_in,
                    self.ok_new_bldg_constr,
                    self.ok_stock_in, self.ok_energy_in,
                    self.ok_carb_in,
                    self.ok_base_cost_in, self.ok_cost_meas_in,
                    self.ok_cost_energy_base_in,
                    self.ok_cost_energy_meas_in,
                    self.ok_relperf_in,
                    self.ok_life_base_in,
                    self.ok_life_meas_in,
                    self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                    self.ok_carbint_base_in, self.ok_carbint_meas_in,
                    self.ok_energy_scnd_in,
                    self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                    self.opts)
                lists_single_coeff = self.measure_instance_single_coeff.partition_microsegment(
                    self.handyvars.adopt_schemes[scn],
                    self.ok_diffuse_params_in,
                    self.ok_mskeys_in[k],
                    self.ok_mkt_scale_frac_in,
                    self.ok_new_bldg_constr,
                    self.ok_stock_in, self.ok_energy_in,
                    self.ok_carb_in,
                    self.ok_base_cost_in, self.ok_cost_meas_in,
                    self.ok_cost_energy_base_in,
                    self.ok_cost_energy_meas_in,
                    self.ok_relperf_in,
                    self.ok_life_base_in,
                    self.ok_life_meas_in,
                    self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                    self.ok_carbint_base_in, self.ok_carbint_meas_in,
                    self.ok_energy_scnd_in,
                    self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                    self.opts)
                lists_multiple_coeffs = self.measure_instance_multiple_coeffs.partition_microsegment(
                    self.handyvars.adopt_schemes[scn],
                    self.ok_diffuse_params_in,
                    self.ok_mskeys_in[k],
                    self.ok_mkt_scale_frac_in,
                    self.ok_new_bldg_constr,
                    self.ok_stock_in, self.ok_energy_in,
                    self.ok_carb_in,
                    self.ok_base_cost_in, self.ok_cost_meas_in,
                    self.ok_cost_energy_base_in,
                    self.ok_cost_energy_meas_in,
                    self.ok_relperf_in,
                    self.ok_life_base_in,
                    self.ok_life_meas_in,
                    self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                    self.ok_carbint_base_in, self.ok_carbint_meas_in,
                    self.ok_energy_scnd_in,
                    self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                    self.opts)
                lists_valid_string = self.measure_instance_valid_string.partition_microsegment(
                    self.handyvars.adopt_schemes[scn],
                    self.ok_diffuse_params_in,
                    self.ok_mskeys_in[k],
                    self.ok_mkt_scale_frac_in,
                    self.ok_new_bldg_constr,
                    self.ok_stock_in, self.ok_energy_in,
                    self.ok_carb_in,
                    self.ok_base_cost_in, self.ok_cost_meas_in,
                    self.ok_cost_energy_base_in,
                    self.ok_cost_energy_meas_in,
                    self.ok_relperf_in,
                    self.ok_life_base_in,
                    self.ok_life_meas_in,
                    self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                    self.ok_carbint_base_in, self.ok_carbint_meas_in,
                    self.ok_energy_scnd_in,
                    self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                    self.opts)

                # Correct list of output dicts
                lists_check_base = self.ok_out_base[scn][k]
                # Compare each element of the lists of output dicts
                for elem2 in range(0, len(lists_check_base)):
                    self.dict_check(lists_check_base[elem2], lists_base[elem2])
                # 
                # Correct list of output dicts
                lists_check_single_coeff = self.ok_out_single_coeff[scn][k]
                # Compare each element of the lists of output dicts
                for elem2 in range(0, len(lists_check_single_coeff)):
                    self.dict_check(lists_check_single_coeff[elem2], lists_single_coeff[elem2])
                # 
                # Correct list of output dicts
                lists_check_multiple_coeffs = self.ok_out_multiple_coeffs[scn][k]
                # Compare each element of the lists of output dicts
                for elem2 in range(0, len(lists_check_multiple_coeffs)):
                    self.dict_check(lists_check_multiple_coeffs[elem2], lists_multiple_coeffs[elem2])
                # 
                # Correct list of output dicts
                lists_check_valid_string = self.ok_out_valid_string[scn][k]
                # Compare each element of the lists of output dicts
                for elem2 in range(0, len(lists_check_valid_string)):
                    self.dict_check(lists_check_valid_string[elem2], lists_valid_string[elem2])



    def test_overrides(self):
            """Test the 'partition_microsegment' function given valid inputs.

            Raises:
                AssertionError: If function yields unexpected results.
            """
            # Loop through 'ok_out' elements

            # Reset AEO time horizon and market entry/exit years
            self.measure_instance_negative_coeff.handyvars.aeo_years = \
                self.time_horizons
            self.measure_instance_negative_coeff.market_entry_year = \
                int(self.time_horizons[0])
            self.measure_instance_negative_coeff.market_exit_year = \
                int(self.time_horizons[-1]) + 1
            # Reset AEO time horizon and market entry/exit years
            self.measure_instance_bad_string.handyvars.aeo_years = \
                self.time_horizons
            self.measure_instance_bad_string.market_entry_year = \
                int(self.time_horizons[0])
            self.measure_instance_bad_string.market_exit_year = \
                int(self.time_horizons[-1]) + 1
            # Reset AEO time horizon and market entry/exit years
            self.measure_instance_wrong_name.handyvars.aeo_years = \
                self.time_horizons
            self.measure_instance_wrong_name.market_entry_year = \
                int(self.time_horizons[0])
            self.measure_instance_wrong_name.market_exit_year = \
                int(self.time_horizons[-1]) + 1


            # Loop through two test schemes (Technical potential and Max
            # adoption potential)
            for scn in range(0, len(self.handyvars.adopt_schemes)):
                # Loop through two microsegment key chains (one applying
                # to new structure type, another to existing structure type)
                for k in range(0, len(self.ok_mskeys_in)):
                    # List of output dicts generated by the function
                    lists_negative_coeff = self.measure_instance_negative_coeff.partition_microsegment(
                        self.handyvars.adopt_schemes[scn],
                        self.ok_diffuse_params_in,
                        self.ok_mskeys_in[k],
                        self.ok_mkt_scale_frac_in,
                        self.ok_new_bldg_constr,
                        self.ok_stock_in, self.ok_energy_in,
                        self.ok_carb_in,
                        self.ok_base_cost_in, self.ok_cost_meas_in,
                        self.ok_cost_energy_base_in,
                        self.ok_cost_energy_meas_in,
                        self.ok_relperf_in,
                        self.ok_life_base_in,
                        self.ok_life_meas_in,
                        self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                        self.ok_carbint_base_in, self.ok_carbint_meas_in,
                        self.ok_energy_scnd_in,
                        self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                        self.opts)
                    lists_bad_string = self.measure_instance_bad_string.partition_microsegment(
                        self.handyvars.adopt_schemes[scn],
                        self.ok_diffuse_params_in,
                        self.ok_mskeys_in[k],
                        self.ok_mkt_scale_frac_in,
                        self.ok_new_bldg_constr,
                        self.ok_stock_in, self.ok_energy_in,
                        self.ok_carb_in,
                        self.ok_base_cost_in, self.ok_cost_meas_in,
                        self.ok_cost_energy_base_in,
                        self.ok_cost_energy_meas_in,
                        self.ok_relperf_in,
                        self.ok_life_base_in,
                        self.ok_life_meas_in,
                        self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                        self.ok_carbint_base_in, self.ok_carbint_meas_in,
                        self.ok_energy_scnd_in,
                        self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                        self.opts)
                    lists_wrong_name = self.measure_instance_wrong_name.partition_microsegment(
                        self.handyvars.adopt_schemes[scn],
                        self.ok_diffuse_params_in,
                        self.ok_mskeys_in[k],
                        self.ok_mkt_scale_frac_in,
                        self.ok_new_bldg_constr,
                        self.ok_stock_in, self.ok_energy_in,
                        self.ok_carb_in,
                        self.ok_base_cost_in, self.ok_cost_meas_in,
                        self.ok_cost_energy_base_in,
                        self.ok_cost_energy_meas_in,
                        self.ok_relperf_in,
                        self.ok_life_base_in,
                        self.ok_life_meas_in,
                        self.ok_ssconv_base_in, self.ok_ssconv_meas_in,
                        self.ok_carbint_base_in, self.ok_carbint_meas_in,
                        self.ok_energy_scnd_in,
                        self.ok_tsv_scale_fracs_in, self.ok_tsv_shapes_in,
                        self.opts)


                    # Correct list of output dicts
                    lists_check_negative_coeff = self.ok_out_negative_coeff[scn][k]
                    # Compare each element of the lists of output dicts
                    for elem2 in range(0, len(lists_check_negative_coeff)):
                        self.dict_check(lists_check_negative_coeff[elem2], lists_negative_coeff[elem2])
                    # 
                    # Correct list of output dicts
                    lists_check_bad_string = self.ok_out_bad_string[scn][k]
                    # Compare each element of the lists of output dicts
                    for elem2 in range(0, len(lists_check_bad_string)):
                        self.dict_check(lists_check_bad_string[elem2], lists_bad_string[elem2])
                    # 
                    # Correct list of output dicts
                    lists_check_wrong_name = self.ok_out_wrong_name[scn][k]
                    # Compare each element of the lists of output dicts
                    for elem2 in range(0, len(lists_check_wrong_name)):
                        self.dict_check(lists_check_wrong_name[elem2], lists_wrong_name[elem2])



# Offer external code execution (include all lines below this point in all
# test files)
def main():
    """Trigger default behavior of running all test fixtures in the file."""
    unittest.main()


if __name__ == "__main__":
    main()
