# Module which contains the functions used for post-processing
import sys
import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

# PyDenseCRF courtesy of https://github.com/lucasb-eyer/pydensecrf:
# Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
# Philipp Krahenbuhl and Vladlen Koltun
# NIPS 2011

class Postprocessing():

    def __init__(self, config):
    	self.config = config
        
    def crf(self, image, prediction):
        # prepare input for crf
        prediction = prediction.squeeze()
        processed_prediction = np.array([ prediction, 1-prediction ])
        processed_prediction = processed_prediction.reshape(( 2,-1 ))
        
        unary = unary_from_softmax( processed_prediction ) 
        unary = np.ascontiguousarray(unary) # necessary since the library pydensecrf is using a cython wrapper
        
        # create crf
        d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
        d.setUnaryEnergy(unary)
        
        # use location features to refine segmentation (color-independent)
        pairwise_energy = create_pairwise_gaussian(sdims=(self.config.POST_SDIMS_GAUSSIAN_X, self.config.POST_SDIMS_GAUSSIAN_Y), shape=image.shape[:2])
        d.addPairwiseEnergy(pairwise_energy, compat=self.config.POST_COMPAT_GAUSSIAN,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        # use local color features to refine segmentation
        pairwise_energy = create_pairwise_bilateral(sdims=(self.config.POST_SDIMS_BILATERAL_X, self.config.POST_SDIMS_BILATERAL_Y),
                                                    schan=(self.config.POST_SCHAN_BILATERAL_R, self.config.POST_SCHAN_BILATERAL_G, self.config.POST_SCHAN_BILATERAL_B),
                                                    img=image, chdim=2)
        d.addPairwiseEnergy(pairwise_energy, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        # do inference (with 5 iterations)
        Q = d.inference(self.config.POST_NUM_INFERENCE_IT)
        result = np.argmin(Q, axis=0)
        processed_prediction = result.reshape((image.shape[0], image.shape[1]))
        
        return processed_prediction