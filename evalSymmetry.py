# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:30:33 2022

@author: ABALTER
"""
import os
import fancy_symmetry
import json
import numpy as np


def evaluateSymmetry(file, stoneName):
    mesh_file = file
    symmetry = 'two fold'  # one-fold, two-fold

    save_name = 'symmetry_image'

    # Read demo mesh
    with open(mesh_file, 'r') as f:
        mesh_str = f.read()

    # Json to dict
    mesh_dict = json.loads(mesh_str)

    # Attributes required for FancySymmetry Class
    points = np.array(mesh_dict['points'])
    poly = mesh_dict['poly']
    facet_angles = mesh_dict['facet_angles']
    outline_indexes = mesh_dict['outline_indexes']

    # FancySymmetry Instance
    fs = fancy_symmetry.FancySymmetry(points,
                       poly,
                       facet_angles,
                       outline_indexes,
                       symmetry=symmetry)

    ## Private method for returning facet pairs
    ## Usually only called internally by creation of asymmtry image

    flip = 'lr' # lr, ud
    side = 'crown' # crown, pavilion
    pairs_diagnostic_image_name =  save_name + "_Pairs_" + flip + "_" + side + ".png"

    # Dont need to save diagnostic image - just saving for demo
    pairs = fs._pair_facet_indexes(flip=flip,
                                   side=side,
                                   save_diagnostic_plot=True,
                                   diagnostic_plot_file_name=pairs_diagnostic_image_name)
    # Each value in the list of tuples refers to the index of poly (facets)
    # If tuple is not a pair, this indicates a facet that pairs with itself
    # Or a facet that failed to pair (ex: extra facet)
    # These facets are actually treated more or less the same for asymmetry image
    print(pairs)

    # Asymmetry image creation
    fs.save_asymmetry_image_crown(stoneName + '_Asym_Crown.jpg')

    fs.save_asymmetry_image_pav(stoneName + '_Asym_Pav.jpg')

    fs.save_wireframe_image(stoneName + '_Wireframe.jpg')

    fileCrown, filePav = stoneName + '_Asym_Crown.jpg', stoneName + '_Asym_Pav.jpg'
    return fileCrown, filePav, pairs

fileName = r"C:\Users\jressler\Desktop\ObservationTracerSandbox\RD094 cushion bt.json"
thisFile = fileName.split("\\")[-1]
thisStone = thisFile.split('.')[0]

fileCrown, filePav, pairs = evaluateSymmetry(fileName, thisStone)