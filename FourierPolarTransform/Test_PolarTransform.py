import unittest
import numpy as np
from scipy.ndimage import imread
from Algorithms.PolarTransform import cart2pol_2d


class TestPolarTransform(unittest.TestCase):
    def setUp(self):
        self.testimage =  imread("../Materials/lena_gray.png", "L").astype('float')
        self.expected = [imread("../Materials/lena_gray_polar.png", "L").astype('float')]
        pass

    def test_cart2pol_2d(self):
        value = np.mean(cart2pol_2d(self.testimage)[0] - self.expected[0])
        self.assertTrue(abs(value) < 1)

    # def test_pol2cart_2d(self):
    #     polar = cart2pol_2d(self.testimage)[0]
    #     value = np.mean(polar - pol2cart2_2d(polar)[0])
    #     self.assertTrue(abs(value) < 1)