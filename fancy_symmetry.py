# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:56:49 2022

@author: ABALTER
"""
import numpy as np
import shapely as shp
import json
from shapely.geometry import Polygon, LineString
from shapely.geometry import mapping
from shapely.ops import split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from shapely.validation import make_valid
from shapely.affinity import translate
from shapely.affinity import rotate
# from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

# import plotting as plot


class FancySymmetry():
    # initiate symmetry class --- everything we need
    def __init__(self, points, poly, facet_angles, 
                 outline_indexes,
                 symmetry='one fold',
                 girdle_angle_range=(75, 105),
                 dpi=100,
                 simple_pairs=False):
        
        self.points = points
        self.alignGirdle()
        self.points_90 = self._rotateWireframe2d(points, 90)
        self.poly = poly
        self.facet_angles = facet_angles
        self.outline_indexes = outline_indexes
        self.symmetry = symmetry
        self.girdle_angle_range = girdle_angle_range
        
        self.crown_indexes = [i for i, a in enumerate(self.facet_angles)
                              if a < self.girdle_angle_range[0]]
        self.girdle_indexes = [i for i, a in enumerate(self.facet_angles) if (a >= girdle_angle_range[0] and a <= girdle_angle_range[1])]
        self.pavilion_indexes = [i for i, a in enumerate(self.facet_angles)
                                 if a > girdle_angle_range[1]]
        self.dpi=dpi
        self.length = points[:, 1].max() - points[:, 1].min()
        self.width = points[:, 0].max() - points[:, 0].min()
        self.simple_pairs = simple_pairs
        self.lineOfSymmetry = None

        
    def _make_2d_plot_split(self,
                            side='crown', figsize=(8, 8), 
                            split_angle=105,
                            flip_pav=True):
        
        points = self.points
        poly = self.poly
        angles = self.facet_angles
        
        fig = Figure(figsize=figsize)
        max_val = np.abs(points[:, :2]).max().max() * 1.1
        # DPI = fig.get_dpi()
        # fig.set_size_inches(1280.0/float(DPI),1024.0/float(DPI))
        ax = fig.gca()
        # plt.axis('off')

        ax.set_axis_off()
        ax.set_xlim((-max_val, max_val))
        ax.set_ylim((-max_val, max_val))
        
        new_points = points.copy()
        if side == 'pavilion' and flip_pav == True:
            new_points[:, 0] = new_points[:, 0]  * -1
            
        line_dict = self._make_line_dict_split(new_points, poly, angles, 
                                               half=side, split_angle=split_angle)
        lc = LineCollection(line_dict['line_list'], 
                            color=line_dict['colors'], 
                            linewidth=0.5)
        ax.add_collection(lc)
        return fig

    
    def _make_line_dict_split(self, points, poly, angles,
                              color='black', half='crown', 
                              split_angle=105):
        line_list = []
        color_list = []
        col = color
        for j, facet in enumerate(poly):
            crown_test = half == 'crown' and angles[j] < split_angle
            pavilion_test = half == 'pavilion' and angles[j] > split_angle
            
            if crown_test or pavilion_test:
                for i, f in enumerate(facet):
                    if i == len(facet)-1:
                        line = [
                            (points[f, 0], points[f, 1]),
                            (points[facet[0], 0], points[facet[0], 1])
                            ]
                        if line not in line_list and [line[1], line[0]] not in line_list:
                            color_list.append(col)
                            line_list.append(line)
                    else:
                        line = [
                            (points[f, 0], points[f, 1]),
                            (points[facet[i+1], 0], points[facet[i+1], 1])
                            ]
                        if line not in line_list and [line[1], line[0]] not in line_list:
                            color_list.append(col)
                            line_list.append(line)
                            
        return {'line_list': line_list,
                'colors': color_list}
        
    def _pair_facet_indexes(self, flip='lr', 
                            side='crown',  
                            save_diagnostic_plot=False,
                            diagnostic_plot_file_name=None):
        # points = fs.points
        # indexes = fs.crown_indexes
        # poly = fs.poly
        # flip = 'lr'
        # length = fs.length
        # width = fs.width
        

        poly = self.poly
        if flip == 'lr':
            points = self.points
        else:
            points = self.points_90
            
        if side == 'crown':
            indexes = self.crown_indexes
        else:
            indexes = self.pavilion_indexes
            
        # Simple pairs is centroid only method
        # Alternative pairs uses centroids to find initial possiblities
        # And then does a more complex operation to find pairs on potential candidates
        # Currently only simple method used for pavilion
        if self.simple_pairs == True or side == 'pavilion':
            alternative_pairs_bypass = True
        else:
            alternative_pairs_bypass = False
        
        # Put polygons into Polygon class
        polygons = []
        for idx in indexes:
            facet_points_xy = points[poly[idx], :2]
            polygons.append(Polygon(facet_points_xy))
        
        # Get centroids and associated indexes
        centroids = [p.centroid for p in polygons]
        inds_cents = [(indexes[i], (p.xy[0][0], p.xy[1][0])) 
                      for i, p in enumerate(centroids)]
        
        # Split to left and right sides --- want to split down line of symmetry
        left = np.array([ci for ci in inds_cents if ci[1][0] < 0], dtype=object)
        right = np.array([ci for ci in inds_cents if ci[1][0] >= 0], dtype=object)

        # # Find side with more points
        # if len(left) == len(right):
        #     side_1 = left
        #     side_2 = right
        # else:
        #     side_1 = left if len(left) > len(right) else right
        #     side_2 = right if len(left) > len(right) else left
        side_1 = left
        side_2 = right
        
        
        side_1_point_array = np.array([x[1] for x in side_1])
        side_2_point_array = np.array([x[1] for x in side_2])

        ### CALCULATE LINE OF SYMMETRY
        xmin = self.points[:, 0].min()
        xmax = self.points[:, 0].max()
        ymin = self.points[:, 1].min()
        ymax = self.points[:, 1].max()
        # gets line of symmetry using extremes of model
        xval = max([abs(xmin), abs(xmax)])
        yval = max([abs(ymin), abs(ymax)])

        # Find line of symmetry
        self.lineOfSymmetry = [(0, -yval * 100),(0, yval * 100)]
        #
        # ### HERE IS WHERE WE NEED TO FLIP
        a1 = reflection_of_point(side_1_point_array.copy(), self.lineOfSymmetry)
        a2 = reflection_of_point(side_2_point_array.copy(), self.lineOfSymmetry)
        side_1_point_array_flipped = side_1_point_array.copy()
        side_1_point_array_flipped[:, 0] = side_1_point_array_flipped[:, 0] * -1
        side_2_point_array_flipped = side_2_point_array.copy()
        side_2_point_array_flipped[:, 0] = side_2_point_array_flipped[:, 0] * -1
        
        # Find points where closest centroid is itself (flipped)
        side_1_remove_indexes = []
        for i, ps1 in enumerate(side_1_point_array):
            d = np.sqrt(
                (ps1[0] - side_2_point_array_flipped[:, 0])**2 +
                (ps1[1] - side_2_point_array_flipped[:, 1])**2
                )
            min_d = d.min()
            sd = np.sqrt(
                (ps1[0] - (ps1[0]*-1))**2 + 
                (ps1[1] - (ps1[1]))**2
                )
            if sd < min_d:
                side_1_remove_indexes.append(i)
                
        side_2_remove_indexes = []
        for i, ps2 in enumerate(side_2_point_array):
            d = np.sqrt(
                (ps2[0] - side_1_point_array_flipped[:, 0])**2 +
                (ps2[1] - side_1_point_array_flipped[:, 1])**2
                )
            min_d = d.min()
            sd = np.sqrt(
                (ps2[0] - (ps2[0]*-1))**2 + 
                (ps2[1] - (ps2[1]))**2
                )
            if sd < min_d:
                side_2_remove_indexes.append(i)
        
        ## Begin pairs
        # Start with points that are closest to themselves
        pairs = []
        s1r = side_1[side_1_remove_indexes]
        s2r = side_2[side_2_remove_indexes]
        for x in s1r:
            pairs.append((x[0],))
        for x in s2r:
            pairs.append((x[0],))
        
        # Remove those indexes from arrays
        side_1 = np.delete(side_1, side_1_remove_indexes, axis=0)
        side_2 = np.delete(side_2, side_2_remove_indexes, axis=0)
        side_1_point_array = np.delete(side_1_point_array, side_1_remove_indexes,
                                       axis=0)
        side_2_point_array = np.delete(side_2_point_array, side_2_remove_indexes,
                                       axis=0)
        side_1_point_array_flipped = np.delete(side_1_point_array_flipped, 
                                               side_1_remove_indexes,
                                               axis=0)
        side_2_point_array_flipped = np.delete(side_2_point_array_flipped, 
                                               side_2_remove_indexes,
                                               axis=0)
        
        # Loop through every point and calculate distance to all points on the other side
        dists = []
        for i, ii in enumerate(side_1_point_array):
            for j, jj in enumerate(side_2_point_array_flipped):
                i1 = side_1[i][0]
                i2 = side_2[j][0]
                
                d = np.sqrt(
                    (ii[0] - jj[0])**2 +
                    (ii[1] - jj[1])**2
                    )
                dists.append((i1, i2, d))
                
        dists_final = np.array(dists)
        if dists_final.ndim > 1:
            dists_final = dists_final[np.argsort(dists_final[:, 2])]
        len_df = len(dists_final)
        
        # Loop through distsances and remove pairs that are closest together until nothing remains
        while(len_df > 0):
            df = dists_final[0]
            i1 = int(df[0])

            # Alternative pairs check
            # If bypass is true, return pair with smallest
            # centroid distance without checking
            i2 = self._check_alternative_pairs(i1, 
                                               dists_final, 
                                               points, 
                                               poly, 
                                               dist_pct=0.5,
                                               bypass=alternative_pairs_bypass)
            
            pairs.append( (i1, i2) )
            
            remove_indexes = np.where((dists_final[:, 0] == i1) |
                                      (dists_final[:, 1] == i2) |
                                      (dists_final[:, 0] == i2) |
                                      (dists_final[:, 1] == i1))
            
            dists_final = np.delete(dists_final, remove_indexes[0], axis=0)
            len_df = len(dists_final)
        
        # Find any indexes that exist but did not find pairs
        unique_indexes = set()
        for pair in pairs:
            unique_indexes.update(pair)
            
        # Add remaining unpaired facets to list to treat self-paired facets    
        indexes_set = set(indexes)
        remaining_indexes = indexes_set.difference(unique_indexes)        
        for ri in remaining_indexes:
            pairs.append((ri, ))
            
        if save_diagnostic_plot == True and diagnostic_plot_file_name is not None:
            cents = np.array([x[1] for x in inds_cents])
            inds = indexes
            if flip == 'lr':
                plot_points = points
            else:
                plot_points = self._rotateWireframe2d(points, -90)
                cents = self._rotateWireframe2d(cents, -90)
                
            
            fig = self._make_2d_plot_split(side=side,
                                           split_angle=90,
                                           flip_pav=False)
            fig_ax = fig.gca()
            for i, ii in enumerate(inds):
                pair = [x for x in pairs if ii in x][0]
                c = cents[i]
                fig_ax.text(c[0], c[1], s=str(ii),
                            horizontalalignment='center',
                            verticalalignment='top',
                            fontsize=8)
                fig_ax.text(c[0], c[1], s=str(pair),
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=8)
                
            if flip == 'lr':
                ymax = plot_points[:, 1].max() * 1000
                fig_ax.vlines(0, -ymax, ymax,
                              alpha=0.15)
            else:
                xmax = plot_points[:, 0].max() * 1000
                fig_ax.hlines(0, -xmax, xmax,
                              alpha=0.15)
                
            fig.tight_layout()
            fig.savefig(diagnostic_plot_file_name, dpi=self.dpi)
            plt.cla() 
            plt.clf() 
            plt.close(fig)
            
        return pairs
        
    def _rotateWireframe2d(self, points, degrees):
        
        new_points = points.copy()
        cs = np.cos((degrees * -1) * (np.pi / 180))
        sn = np.sin((degrees * -1) * (np.pi / 180))
        rot_matrix = np.array([[cs, -sn], [sn, cs]])
        new_points[:, :2] = new_points[:, :2] @ rot_matrix

        return new_points
    
    def _get_asymmetry_polygons(self, flip='lr', side='crown',
                                save_diagnostic_plot=False,
                                diagnostic_plot_file_name=None):
        
        poly = self.poly.copy()
        points = self.points.copy()
        pairs = self._pair_facet_indexes(flip=flip, side=side)
        xmin = self.points[:, 0].min()
        xmax = self.points[:, 0].max()
        ymin = self.points[:, 1].min()
        ymax = self.points[:, 1].max()
        # gets line of symmetry using extremes of model
        xval = max([abs(xmin), abs(xmax)])
        yval = max([abs(ymin), abs(ymax)])
        
        # Find line of symmetry
        if flip == 'lr':
            line_of_symmetry = LineString([(0, -yval*100),
                                           (0, yval*100)])
        else:
            line_of_symmetry = LineString([(-xval*100, 0),
                                           (xval*100, 0)])
            
   ### THIS IS WHERE WE ARE GETTING SYMMETRY MAP (flipping facets and comparing)
        # If only one polygon, measure asymmetry on itself split of LOS
        # or itself flipped if it doesnt cross LOS
        # If two polygons, measure asymmetry vs each other
        # If one or both polygons cross LOS, only use side on correct LOS
        asymmetry_polygons = []
        for p in pairs:
            # print(p)
            # If only one polygon, check for split
            # If doesn't cross LOS, poly2 is itself flipped

        ### FIRST CONSIDER UNPAIRED FACET
            # New addition
            # Extra facets which dont intersect with line of symmetry were
            # previously being accounted for by surrounding facets
            # now explicity added
            no_split_extra = False
            if len(p) < 2:
                ### IF FACET UNPAIRED, first check splitting over LoS, if not just flip and add ti asymmetry array
                poly0pts = points[poly[p[0]], :2]
                poly0pg = Polygon(poly0pts)
                split_check = self._split_polygon_on_los(poly0pg,
                                                         line_of_symmetry)
                if split_check is None:
                    poly1pts = poly0pts
                    poly1pg = poly0pg
                    poly2pts = self._flip_array(poly1pts, flip=flip)
                    poly2pg = Polygon(poly2pts)
                    asymmetry_polygons.append(poly2pg)
                    no_split_extra = True
                else:
                    ### extract pair of polys
                    poly1pg = split_check[0][0]
                    poly2pg = split_check[1][0]
                    poly1pts = self._extract_coords_from_Polygon(poly1pg)
                    poly2pts = self._extract_coords_from_Polygon(poly2pg)
            
            # If two polygons, check for LOS split
            # If no split, polygons are as is
            # If one or both cross LOS, take the correct side for each
            else:
                ### If two polygons, first check if LOS split
                poly1pts = points[poly[p[0]], :2]
                poly2pts = points[poly[p[1]], :2]
                poly1pg = Polygon(poly1pts)
                poly2pg = Polygon(poly2pts)
                # Pairs are (left, right) or (top, bottom)
                split_check_1 = self._split_polygon_on_los(poly1pg,
                                                           line_of_symmetry)
                split_check_2 = self._split_polygon_on_los(poly2pg,
                                                           line_of_symmetry)
                
                if split_check_1 is None:
                    poly1pg = Polygon(poly1pts)
                else:
                    if flip == 'lr':
                        split_poly_1 = [x for x in split_check_1 if x[1][0] < 0]
                    else:
                        split_poly_1 = [x for x in split_check_1 if x[1][1] >= 0]
                    poly1pg = split_poly_1[0][0]
                    poly1pts = self._extract_coords_from_Polygon(poly1pg)
                
                if split_check_2 is None:
                    poly2pg = Polygon(poly2pts)
                else:
                    if flip == 'lr':
                        split_poly_2 = [x for x in split_check_2 if x[1][0] >= 0]
                    else:
                        split_poly_2 = [x for x in split_check_2 if x[1][1] < 0]
                    poly2pg = split_poly_2[0][0]
                    poly2pts = self._extract_coords_from_Polygon(poly2pg)
                    

            ### FLIP POLYGONS AND COMPARE
            if no_split_extra == False:
                # Create flipped versions of each polygon        
                poly1pts_flipped = self._flip_array(poly1pts, flip=flip)
                poly2pts_flipped = self._flip_array(poly2pts, flip=flip)
                poly1pg_flipped = Polygon(poly1pts_flipped)
                poly2pg_flipped = Polygon(poly2pts_flipped)

                ### GET DIFFERENCE AND APPEND TO POLY LIST
                # Return polygons of difference between 
                # Poly1 and Poly2flipped
                # Poly2 and Poly1flipped
                poly1_diff = self._get_polygon_diff(poly1pg, poly2pg_flipped)
                poly2_diff = self._get_polygon_diff(poly2pg, poly1pg_flipped)
                for pd1 in poly1_diff:
                    asymmetry_polygons.append(pd1)
                for pd2 in poly2_diff:
                    asymmetry_polygons.append(pd2)
                
        return asymmetry_polygons

    def _extract_coords_from_Polygon(self, polygon):       
        try:
            pmap = mapping(polygon)
            return_array = np.array(pmap['coordinates'][0])
        except:
            return_array = np.array([])
        return return_array
    
    def _flip_array(self, array, flip='lr'):
        
        array_copy = array.copy()
        if flip == 'lr':
            array_copy[:, 0] = array_copy[:, 0] * -1
        else:
            array_copy[:, 1] = array_copy[:, 1] * -1
            
        return array_copy
    
    
    def _split_polygon_on_los(self, polygon, line_of_symmetry):
        crosses_line_check = polygon.crosses(line_of_symmetry)
        if crosses_line_check == False:
            return None
        
        else:
            poly_split = split(polygon, line_of_symmetry)
            poly_geoms = list(poly_split.geoms)
            
            centroids = [x.centroid for x in poly_geoms]
            centroids = [(p.xy[0][0], p.xy[1][0]) for p in centroids]
            
            return list(zip(poly_geoms, centroids))
        
    def _get_polygon_diff(self, poly1, poly2):
        try:
            valid_test = poly1.is_valid or poly2.is_valid
            if not valid_test:
                poly1 = make_valid(poly1)
                poly2 = make_valid(poly2)
                
            diff = poly1.symmetric_difference(poly2)
            # If either poly contains the other, can have donut shaped polygon difference
            if poly1.contains(poly2) or poly2.contains(poly1):
                # Can return a geometry collection?
                # Find the polygon difference
                if diff.geom_type == 'GeometryCollection':
                    for geom in diff.geoms:
                        if geom.geom_type == 'Polygon':
                            diff = geom
                # Get exterior of "donut"
                ext = np.array(diff.exterior.coords)
                
                # X axis LOS is never actually used because up-down symmetry
                # is actually using the stone rotated to 90 degrees
                if len(ext):
                    # minx = ext[:, 0].min()
                    # maxx = ext[:, 0].max()
                    miny = ext[:, 1].min()
                    maxy = ext[:, 1].max()
                    # xval = max([abs(minx), abs(maxx)])
                    yval = max([abs(miny), abs(maxy)])
                    cent = diff.centroid.x, diff.centroid.y
                    line1 = LineString([(cent[0], -yval*100), (cent[0], yval*100)])
                    # line2 = LineString([(-xval*100, cent[1]), (xval*100, cent[1])])
                    # Split donut on line crossing centroid
                    poly_split = split(diff, line1)
                    polys = list(poly_split.geoms)
                else:
                    polys = []
            else:
                # diff = poly1.difference(poly2)
                # If diff has attribute of "geoms", the difference is multiple polygons
                if hasattr(diff, "geoms"):
                    polys = list(diff.geoms)
                else:
                    polys = [diff]
                        
            return polys
        
        except:
            return [None]
    
    def make_2d_plot(self, figsize=(8, 8), colors=('black', 'gray'),
                     color_angle=105, facecolor='white', auto_limit=True):
        fig = Figure(figsize=figsize, facecolor=facecolor)
        max_val = np.abs(self.points[:, :2]).max().max() * 1.1
        
        # DPI = fig.get_dpi()
        # fig.set_size_inches(1280.0/float(DPI),1024.0/float(DPI))
        ax = fig.gca()
        # plt.axis('off')
        ax.set_axis_off()
        if auto_limit == True:           
            ax.set_xlim((-max_val, max_val))
            ax.set_ylim((-max_val, max_val))
        
        line_dict = self._make_line_dict(colors=colors,
                                         color_angle=color_angle)
        lc = LineCollection(line_dict['line_list'], 
                            color=line_dict['colors'], linewidth=0.5)
        ax.add_collection(lc)
        return fig
    
    def _make_line_dict(self, colors=('black', 'gray'), color_angle=105):
        line_list = []
        color_list = []
        for j, facet in enumerate(self.poly):
            if self.facet_angles[j] >= color_angle:
                col = colors[1]
            else:
                col = colors[0]
            for i, f in enumerate(facet):
                if i == len(facet)-1:
                    line = [
                        (self.points[f, 0], self.points[f, 1]),
                        (self.points[facet[0], 0], self.points[facet[0], 1])
                        ]
                    if line not in line_list and [line[1], line[0]] not in line_list:
                        color_list.append(col)
                        line_list.append(line)
                else:
                    line = [
                        (self.points[f, 0], self.points[f, 1]),
                        (self.points[facet[i+1], 0], self.points[facet[i+1], 1])
                        ]
                    if line not in line_list and [line[1], line[0]] not in line_list:
                        color_list.append(col)
                        line_list.append(line)
        return {'line_list': line_list,
                'colors': color_list}
    
    def return_asymmetry_figure(self):
        
        try:
            asym_crown_lr = self._get_asymmetry_polygons(flip='lr',
                                                         side='crown')
        except:
            asym_crown_lr = []
            
        try:
            asym_pavilion_lr = self._get_asymmetry_polygons(flip='lr',
                                                            side='pavilion')
        except:
            asym_pavilion_lr = []
        
        asym_crown_lr_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                   in asym_crown_lr]
        asym_pavilion_lr_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                      in asym_pavilion_lr]

        
        # Encountered instances of polygons having 1 vertex
        cr_lr_pt_arr_filtered = [x for x in asym_crown_lr_pt_arrays if x.ndim >= 2]
        pav_lr_pt_arr_filtered = [x for x in asym_pavilion_lr_pt_arrays if x.ndim >= 2]
        
        if self.symmetry == 'two fold':
            try:
                asym_crown_ud = self._get_asymmetry_polygons(flip='ud',
                                                             side='crown')
            except:
                asym_crown_ud = []
            
            try:
                asym_pavilion_ud = self._get_asymmetry_polygons(flip='ud',
                                                               side='pavilion')
            except:
                asym_pavilion_ud = []
            
            asym_crown_ud_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                       in asym_crown_ud]
            asym_pavilion_ud_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                          in asym_pavilion_ud]
            # Encountered instances of polygons having 1 vertex
            cr_ud_pt_arr_filtered = [x for x in asym_crown_ud_pt_arrays if x.ndim >= 2]
            pav_ud_pt_arr_filtered = [x for x in asym_pavilion_ud_pt_arrays if x.ndim >= 2]
        
        # Plot Colors
        crown_line_color = (0, 1, 0)
        pav_line_color = (0, 0.5, 0)
        outline_fill_color = (0, 0.1, 0)
        lr_crown_color = (1, 0, 0)
        lr_pav_color = (0, 0, 1)
        ud_crown_color = (1, 0, 0)
        ud_pav_color = (0, 0, 1)
        alpha = 0.5
        
        # Make 2d Plot
        fig = self.make_2d_plot(colors=(crown_line_color, pav_line_color),
                                facecolor='black')
        fig_ax = fig.gca()
        
        # Add outline fill
        outline_points = self.points[self.outline_indexes, :2]
        fig_ax.fill(outline_points[:, 0], outline_points[:, 1], color=outline_fill_color)
        
            
        # Add LR Pavilion
        for asplr in pav_lr_pt_arr_filtered:
            fig_ax.fill(asplr[:, 0], asplr[:, 1], color=lr_pav_color, alpha=alpha)
            
        if self.symmetry == 'two fold':
            # Add UD Pavilion
            for aspud in pav_ud_pt_arr_filtered:
                fig_ax.fill(aspud[:, 0], aspud[:, 1], color=ud_pav_color, alpha=alpha)
                
            # Add UD Crown
            for ascud in cr_ud_pt_arr_filtered:
                fig_ax.fill(ascud[:, 0], ascud[:, 1], color=ud_crown_color, alpha=alpha)
                
                
        # Add LR Crown
        for asclr in cr_lr_pt_arr_filtered:
            fig_ax.fill(asclr[:, 0], asclr[:, 1], color=lr_crown_color, alpha=alpha)
        
        fig.tight_layout()
        
        return fig

    def return_asymmetry_crown(self):

        try:
            asym_crown_lr = self._get_asymmetry_polygons(flip='lr',
                                                         side='crown')
        except:
            asym_crown_lr = []


        asym_crown_lr_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                   in asym_crown_lr]

        # Encountered instances of polygons having 1 vertex
        cr_lr_pt_arr_filtered = [x for x in asym_crown_lr_pt_arrays if x.ndim >= 2]

        if self.symmetry == 'two fold':
            try:
                asym_crown_ud = self._get_asymmetry_polygons(flip='ud',
                                                             side='crown')
            except:
                asym_crown_ud = []

            asym_crown_ud_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                       in asym_crown_ud]
            # Encountered instances of polygons having 1 vertex
            cr_ud_pt_arr_filtered = [x for x in asym_crown_ud_pt_arrays if x.ndim >= 2]

        # Plot Colors
        crown_line_color = (0, 1, 0)
        pav_line_color = (0, 0.5, 0)
        outline_fill_color = (1, 1, 1)
        lr_crown_color = (1, 0, 0)
        lr_pav_color = (0, 0, 1)
        ud_crown_color = (1, 0, 0)
        ud_pav_color = (0, 0, 1)
        alpha = 0.5

        # Make 2d Plot
        figCrown = self._make_2d_plot_split('crown')
        fig_ax_crown = figCrown.gca()

        # Add outline fill
        outline_crown = self.points[self.outline_indexes, :2]
        fig_ax_crown.fill(outline_crown[:, 0], outline_crown[:, 1], color=outline_fill_color)

        if self.symmetry == 'two fold':
            # Add UD Pavilion

            # Add UD Crown
            for ascud in cr_ud_pt_arr_filtered:
                fig_ax_crown.fill(ascud[:, 0], ascud[:, 1], color=ud_crown_color, alpha=alpha)

        # Add LR Crown
        for asclr in cr_lr_pt_arr_filtered:
            fig_ax_crown.fill(asclr[:, 0], asclr[:, 1], color=lr_crown_color, alpha=alpha)

        figCrown.tight_layout()

        return figCrown

    def return_asymmetry_pav(self):
        try:
            asym_pavilion_lr = self._get_asymmetry_polygons(flip='lr',
                                                            side='pavilion')
        except:
            asym_pavilion_lr = []

        asym_pavilion_lr_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                      in asym_pavilion_lr]

        # Encountered instances of polygons having 1 vertex
        pav_lr_pt_arr_filtered = [x for x in asym_pavilion_lr_pt_arrays if x.ndim >= 2]

        if self.symmetry == 'two fold':
            try:
                asym_pavilion_ud = self._get_asymmetry_polygons(flip='ud',
                                                                side='pavilion')
            except:
                asym_pavilion_ud = []

            asym_pavilion_ud_pt_arrays = [self._extract_coords_from_Polygon(x) for x
                                          in asym_pavilion_ud]
            # Encountered instances of polygons having 1 vertex
            pav_ud_pt_arr_filtered = [x for x in asym_pavilion_ud_pt_arrays if x.ndim >= 2]

        # Plot Colors
        crown_line_color = (0, 1, 0)
        pav_line_color = (0, 0.5, 0)
        outline_fill_color = (1, 1, 1)
        lr_crown_color = (1, 0, 0)
        lr_pav_color = (0, 0, 1)
        ud_crown_color = (1, 0, 0)
        ud_pav_color = (0, 0, 1)
        alpha = 0.5

        # Make 2d Plot
        figPav = self._make_2d_plot_split('pavilion')
        fig_ax_pav = figPav.gca()

        # Add outline fill
        outline_pav = self.points[self.outline_indexes, :2]
        fig_ax_pav.fill(outline_pav[:, 0], outline_pav[:, 1], color=outline_fill_color)

        # Add LR Pavilion
        for asplr in pav_lr_pt_arr_filtered:
            fig_ax_pav.fill(asplr[:, 0], asplr[:, 1], color=lr_pav_color, alpha=alpha)

        if self.symmetry == 'two fold':
            # Add UD Pavilion
            for aspud in pav_ud_pt_arr_filtered:
                fig_ax_pav.fill(aspud[:, 0], aspud[:, 1], color=ud_pav_color, alpha=alpha)
        figPav.tight_layout()

        return figPav

    def save_asymmetry_image(self, save_file_name, dpi=100):
        
        fig = self.return_asymmetry_figure()
        fig.savefig(save_file_name, dpi=dpi)  
        plt.cla() 
        plt.clf() 
        plt.close(fig)

    def save_asymmetry_image_crown(self, crownName, dpi=100):
        figCrown = self.return_asymmetry_crown()
        figCrown.savefig(crownName, dpi=dpi)
        plt.cla()
        plt.clf()
        plt.close(figCrown)

    def save_asymmetry_image_pav(self, pavName, dpi=100):
        figPav = self.return_asymmetry_pav()
        figPav.savefig(pavName, dpi=dpi)
        plt.cla()
        plt.clf()
        plt.close(figPav)


    def save_wireframe_image(self, save_file_name, dpi=100):

        fig = self.make_2d_plot()
        fig.tight_layout()
        fig.savefig(save_file_name, dpi=dpi)
        plt.cla()
        plt.clf()
        plt.close(fig)

    def _centroid_to_zero_zero(self, polygon):
        c = polygon.centroid.xy[0][0], polygon.centroid.xy[1][0]
        p_new = translate(polygon, xoff=c[0] * -1, yoff=c[1] * -1)
        return p_new


    def _calc_pct_area_of_intersection(self, deg, p0, p1, sign=1.0):
        p1_rotated = rotate(p1, angle=deg)
        union_area = p0.union(p1_rotated).area
        return sign * (p0.intersection(p1_rotated).area / union_area)

    def _check_alternative_pairs(self, index, dists_matrix, 
                                 points, poly, dist_pct=0.5,
                                 bypass=False):
        
        dists_sub = dists_matrix[dists_matrix[:, 0].astype(int) == index]
        
        # If only one choice, return only choice
        # Or if bypass is True
        i1 = int(dists_sub[0, 1])
        if dists_sub.shape[0] < 2 or bypass == True:
            return i1
        
        # Divide first match distance by second match distance
        # if value is >= dist_pct continue, otherwise return first match
        # Namely, if distance to closest point is gte to (ex 50%) of the distance
        # of the second point, disambiguate the pairs
        d1 = dists_sub[0, 2]
        d2 = dists_sub[1, 2]
        if d1 / d2 < dist_pct:
            return i1
        
        # Get top two choice indexes
        i2 = int(dists_sub[1, 1])
        

        # Left side Polygon
        p0 = Polygon(points[poly[index], :2])
        p0_centered = self._centroid_to_zero_zero(p0)
          
        
        # Invert along X for matches side
        p1_points = points[poly[i1], :2]
        p2_points = points[poly[i2], :2]
        p1_points[:, 0] = p1_points[:, 0] * -1
        p2_points[:, 0] = p2_points[:, 0] * -1
        
        
        # Right side polygons
        p1 = Polygon(p1_points)
        p2 = Polygon(p2_points)
        
        # If either polygon is invalid, return shortest distance
        if not p0.is_valid or not p1.is_valid or not p2.is_valid:
            return i1
        
        
        # Center on 0, 0
        p1_centered = self._centroid_to_zero_zero(p1)
        p2_centered = self._centroid_to_zero_zero(p2)
        
        # Find area of maximum overlap
        p0_1_a = minimize(self._calc_pct_area_of_intersection, x0=0, 
                          args=(p0_centered, p1_centered, -1))
        p0_2_a = minimize(self._calc_pct_area_of_intersection, x0=0, 
                          args=(p0_centered, p2_centered, -1))
     
        # Area of overlap expressed as percent of union of polygons    
        p0_1_int_pct = p0_1_a.fun * -1
        p0_2_int_pct = p0_2_a.fun * -1
        
        # If area of maximumzed overlap is larger on index2, return i2
        # Otherwise return smallest distance as originally designed
        if p0_2_int_pct > p0_1_int_pct:
            return i2
        else:
            return i1
            
        
        # x1 = p0_1_a.x[0]
        # x2 = p0_2_a.x[0]
        # p1_r = rotate(p1_centered, x1)
        # p2_r = rotate(p2_centered, x2)
        # fig, axs = plt.subplots(figsize=(8, 8))
        # axs.set_aspect('equal', 'datalim')
        # colors = ['blue', 'red', 'green']
        # for i, p in enumerate([p0_centered, p1_r, p2_r]):
        #     xs, ys = p.exterior.xy
        #     axs.fill(xs, ys, alpha = 0.5, fc=colors[i], ec='black')


    # Function to split down line of symmetry
    def calcLineOfSymmetry(self):
        raise NotImplementedError

    def alignGirdle(self):
        girdleHull = convexHullPoints(self.points)
        diameter, diamPts = diameterPts(girdleHull)
        # rotate stone such that diamter pts on x-axis
        theta = math.atan2(diamPts[0][1], diamPts[0][0])
        # for each vert, want to apply transformation
        self.rotate(-theta)

    def rotate(self, degrees):
        cs = np.cos(degrees)
        sn = np.sin(degrees)
        rot_matrix = np.array([[cs, -sn], [sn, cs]])
        self.points[:, :2] = self.points[:, :2] @ rot_matrix

def diameterPts(hullPts):
    diameter = 0
    diameterPts = []
    for i in range(len(hullPts) - 1):
        for j in range(len(hullPts) - i - 1):
            distance2d = abs(hullPts[i][0] - hullPts[i + j + 1][0])
            if distance2d > diameter:
                diameter = distance2d
                diameterPts = [hullPts[i], hullPts[i + j + 1]]
    return diameter, diameterPts


def reflection_of_point(pointArray, q):
    """Calculates reflection of a point across an edge

    Args:
        pointArray (ndarray): Inner points, (2,n)
        q (ndarray): First vertex of the edge, (2,). Second vertex of the edge, (2,)

    Returns:
        ndarray: Reflected points, (2,n)
    """

    q_i = q[0]
    q_j = q[1]

    a = q_i[1] - q_j[1]
    b = q_j[0] - q_i[0]
    c = - (a * q_i[0] + b * q_i[1])

    newPoints = [((np.array([[b**2 - a**2, -2 * a * b], [-2 * a * b, a**2 - b**2]]) @ pt
                  - 2 * c * np.array([a, b])) / (a**2 + b**2)).tolist() for pt in pointArray]

    return np.array(newPoints)


def convexHullPoints(pointArray):
    # project points to 2d
    pointSet = []
    for i in range(len(pointArray)):
        pointSet.append(pointArray[i][:2].tolist())

    # get hull
    hullPts = convexHull(pointSet)

    return hullPts

def convexHull(pointCloud):
    hull = ConvexHull(pointCloud)
    hull_indices = np.unique(hull.simplices.flat)
    hull_pts = []
    for i in range(len(hull_indices)):
        hull_pts.append(pointCloud[hull_indices[i]])
    return hull_pts
