import logging
import sys
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import cpnet

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger()
logger.level = logging.DEBUG


class TestCalc(unittest.TestCase):
    def setUp(self):
        # BE =========================
        self.G = nx.karate_club_graph()
        self.models = [
            cpnet.BE(),
            cpnet.Lip(),
            cpnet.LowRankCore(),
            cpnet.LapCore(),
            cpnet.LapSgnCore(),
            cpnet.Rombach(),
            cpnet.Rossa(),
            cpnet.KM_ER(),
            cpnet.KM_config(),
            cpnet.Surprise(),
        ]
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    def test_detect(self):

        for model in self.models:

            logger.info("Core-periphery detection %s " % str(model))

            model.detect(self.G)
            pair_id = model.get_pair_id()
            coreness = model.get_coreness()

            self.assertIsInstance(pair_id, dict)
            self.assertIsInstance(coreness, dict)

    def test_significance_test(self):

        for model in self.models:

            logger.info("Significance test: core-periphery detection %s " % str(model))

            model.detect(self.G)
            pair_id = model.get_pair_id()
            coreness = model.get_coreness()

            sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(
                pair_id, coreness, self.G, model
            )


if __name__ == "__main__":
    unittest.main()
