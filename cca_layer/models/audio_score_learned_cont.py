#!/usr/bin/env python

from audio_score_ccal_tno import *

WEIGHT_TNO = 0.0
USE_CCAL = False
GAMMA = 0.5

build_model = get_build_model(weight_tno=WEIGHT_TNO, alpha=ALPHA, dim_latent=DIM_LATENT, use_ccal=USE_CCAL)


def objectives():
    """ Compile objectives """
    from objectives import get_contrastive_cos_loss
    return get_contrastive_cos_loss(1.0 - WEIGHT_TNO, GAMMA, symmetric=True)
