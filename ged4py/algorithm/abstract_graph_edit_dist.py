# -*- coding: UTF-8 -*-
from __future__ import print_function

from scipy.optimize import linear_sum_assignment
import sys
import numpy as np
from networkx import __version__ as nxv


class AbstractGraphEditDistance(object):
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

    def distance(self, normalized=True, mapping=False):
        """
        Returns the graph edit distance between graph g1 & g2
        The distance is normalized on the size of the two graphs.
        This is done to avoid favorisation towards smaller graphs
        """
        avg_graphlen = (len(self.g1) + len(self.g2)) / 2
        res = self.compute()
        if normalized:
            res["distance"] /= avg_graphlen

        return res if mapping else res["distance"]

    def compute(self):

        cost_matrix = self.create_cost_matrix()

        row_ind,col_ind = linear_sum_assignment(cost_matrix)
        mapping = extract_mapping(row_ind, col_ind, self.g1, self.g2)
        self.result = {
            'mapping': mapping,
            'distance': sum([cost_matrix[row_ind[i]][col_ind[i]] for i in range(len(row_ind))])
        }

        return self.result


    def create_cost_matrix(self):
        """
        Creates a |N+M| X |N+M| cost matrix between all nodes in
        graphs g1 and g2
        Each cost represents the cost of substituting,
        deleting or inserting a node
        The cost matrix consists of four regions:

        substitute 	| insert costs
        -------------------------------
        delete 		| delete -> delete

        The delete -> delete region is filled with zeros
        """
        n = len(self.g1)
        m = len(self.g2)
        cost_matrix = np.zeros((n+m,n+m))
        #cost_matrix = [[0 for i in range(n + m)] for j in range(n + m)]
        nodes1 = self.g1.nodes() if float(nxv) < 2 else list(self.g1.nodes())
        nodes2 = self.g2.nodes() if float(nxv) < 2 else list(self.g2.nodes())

        for i in range(n):
            for j in range(m):
                cost_matrix[i,j] = self.substitute_cost(nodes1[i], nodes2[j])

        for i in range(m):
            for j in range(m):
                cost_matrix[i+n,j] = self.insert_cost(i, j, nodes2)

        for i in range(n):
            for j in range(n):
                cost_matrix[j,i+m] = self.delete_cost(i, j, nodes1)

        self.cost_matrix = cost_matrix
        return cost_matrix

    def insert_cost(self, i, j):
        raise NotImplementedError

    def delete_cost(self, i, j):
        raise NotImplementedError

    def substitute_cost(self, nodes1, nodes2):
        raise NotImplementedError

    def print_matrix(self):
        print("cost matrix:")
        # for column in self.create_cost_matrix():
        #     for row in column:
        #         if row == sys.maxint:
        #             print ("inf\t")
        #         else:
        #             print ("%.2f\t" % float(row))
        #     print("")

        for row in np.transpose(self.create_cost_matrix()):
            for val in row:
                print('{:8.2}'.format(val if val != sys.maxsize else -1.), end='')
            print()

def extract_mapping(row_ind, col_ind, g1, g2):
    mapping = []
    for i in range(len(row_ind)):
        if row_ind[i] < len(g1) or col_ind[i] < len(g2):
            i_row = row_ind[i]
            i_col = col_ind[i]
            if i_row >= len(g1):
                i_row = -1
            if i_col >= len(g2):
                i_col = -1
            mapping.append([i_row, i_col])

    return mapping

