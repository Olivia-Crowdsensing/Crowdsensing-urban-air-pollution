import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap, interp
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, MultiPolygon, box
from descartes import PolygonPatch


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    from http://stackoverflow.com/a/20678647/416626
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_voronoi(lon, lat, ax, **kwargs):
    """
    More flexible Voronoi plotting
    kwargs are passed to ax.plot()
    returns computed voronoi object
    This is a bit of a black box tbh
    adapted from http://nbviewer.ipython.org/gist/wiso/6755034
    """
    vor = Voronoi(df[['projected_lon', 'projected_lat']])  # 对应于Voronoi(points)
    xy = np.dstack((lon, lat))[0]

    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], **kwargs)
    center = xy.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]
            t = xy[pointidx[1]] - xy[pointidx[0]]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = xy[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + \
                        np.sign(np.dot(midpoint - center, n)) * n * 10000000
            vor.complete_regions, vor.complete_vertices = voronoi_finite_polygons_2d(vor)
            #  this plots the far points - maybe consider a dashed line
            ax.plot(
                [vor.vertices[i, 0], far_point[0]],
                [vor.vertices[i, 1], far_point[1]],
                **kwargs)
    return vor


if __name__ == '__main__':
    df = pd.read_excel("./data/China/2017-01_geo.xlsx")
    x, y = df["lat"], df["lng"]
    points = np.vstack([x.values, y.values]).T

    # define map extent
    lllat = 17
    lllon = 75
    urlat = 51
    urlon = 132

    # set up Basemap instance
    m = Basemap(
        projection='merc',
        llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat,
        resolution='h')

    # transform lon / lat coordinates to map projection
    df['projected_lon'], df['projected_lat'] = m(*(df["lng"].values,
                                                   df["lat"].values))
    norm = Normalize()
    fig = plt.figure(figsize=(8, 8))
    ax2 = fig.add_subplot(111, facecolor='w', frame_on=False)

    # draw map details
    m.drawmapboundary(fill_color='white', ax=ax2)
    m.fillcontinents(color='#DCDCDC', lake_color='#7093DB', ax=ax2)
    # m.drawcoastlines()
    m.drawcountries(
        linewidth=1, linestyle='solid', color='#000073',
        antialiased=True,
        ax=ax2, zorder=3)
    m.drawparallels(
        np.arange(lllat, urlat, 10.),
        color='black', linewidth=0.5,
        labels=[True, False, False, False], ax=ax2)
    m.drawmeridians(
        np.arange(lllon, urlon, 10.),
        color='0.25', linewidth=0.5,
        labels=[False, False, False, True], ax=ax2)

    # initial params
    subdiv = 5
    init_mask_frac = 0.0
    min_circle_ratio = .01
    random_gen = np.random.mtrand.RandomState(seed=127260)

    # meshing with Delaunay triangulation
    tri = Triangulation(df.projected_lon, df.projected_lat)
    ntri = tri.triangles.shape[0]

    # Some invalid data are masked out
    mask_init = np.zeros(ntri, dtype=np.bool)
    masked_tri = random_gen.randint(0, ntri, int(ntri * init_mask_frac))
    mask_init[masked_tri] = True
    tri.set_mask(mask_init)
    mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
    tri.set_mask(mask)

    m.scatter(
        df["projected_lon"],
        df["projected_lat"],
        color='#545454',
        edgecolor='#ffffff',
        alpha=1.,
        s=5,
        ax=ax2,
    )

    # plot of the initial coarse mesh
    plt.triplot(tri, color='#545454', zorder=1, lw=.7, alpha=.75)
    plt.tight_layout(pad=1.25)
    # plt.show()
    plt.savefig("voronoi_national.png", format="png", bbox_inches='tight', transparent=True, dpi=300)