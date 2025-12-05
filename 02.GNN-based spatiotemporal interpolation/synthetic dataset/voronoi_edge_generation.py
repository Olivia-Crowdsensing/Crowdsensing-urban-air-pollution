import scipy as sp
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def generate_plot_voronoi(points):
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)
    plt.show()


def get_neighbors(vertex_id, triang):
    helper = triang.vertex_neighbor_vertices
    index_pointers = helper[0]
    indices = helper[1]
    result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
    return result_ids


def writeRecord(csvfile, writeline):
    from csv import writer
    c = open(csvfile, 'a', newline='')
    writer = writer(c)
    writer.writerow(writeline)
    c.close()


if __name__ == '__main__':
    df = pd.read_excel("./data/01-data-raw/highAuto_highHete.xlsx")
    x, y = df["x"], df["y"]
    points = np.vstack([x.values, y.values]).T  # This is the dataframe -> of original excels
    # plt.plot(*np.transpose(points), marker='o', ls='')  # plot these points
    # To compute the triangulation

    dela = sp.spatial.Delaunay
    triang = dela(points)

    for i in range(0, len(x)):
        tmp_neigh_ids = get_neighbors(i, triang)
        print(tmp_neigh_ids)
        writeRecord("neighbour_output.csv", [i, tmp_neigh_ids])
