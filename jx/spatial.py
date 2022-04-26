import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

inla = importr("INLA")

robjects.r('''
# compute scaling factor for a fully connected areal map
# accounts for differences in spatial connectivity
scaling_factor <- function(node1, node2) {
    adj_matrix = sparseMatrix(i=node1,j=node2,x=1,symmetric=TRUE)

    N = dim(adj_matrix)[1]

    # Create ICAR precision matrix  (diag - adjacency): this is singular
    # function Diagonal creates a square matrix with given diagonal
    Q =  Diagonal(N, rowSums(adj_matrix)) - adj_matrix

    # Add a small jitter to the diagonal for numerical stability (optional but recommended)
    Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)

    # Function inla.qinv provides efficient way to calculate the elements of the
    # the inverse corresponding to the non-zero elements of Q
    Q_inv = inla.qinv(Q_pert, constr=list(A = matrix(1,1,N),e=0))

    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    scaling_factor <- exp(mean(log((Matrix::diag(Q_inv)))))
    return(scaling_factor) 
}
'''
)

def geo_to_nb(df: gpd.GeoDataFrame):
    """
    Given a GeoPandas data frame with a geometry column, return a list containing the indices of the neighbours of each of the rows.
    """
    assert 'geometry' in df.columns, "GeoPandas data frame must have a geometry column"

    neighbours_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        nb = df[df['geometry'].touches(row['geometry'])]
        neighbours_list.append(nb.index)

    return neighbours_list


def nb_to_graph(nb_list):
    """
    Given a list of indices of neighbours, construct a graph and return the indices of the nodes connected by each edge.
    """
    node1 = []
    node2 = []
    for i, j in enumerate(nb_list):
        for n in j:
            if n > i:
                node1.append(i)
                node2.append(n)

    node1 = np.array(node1)
    node2 = np.array(node2)

    assert node1.size == node2.size, "Something went wrong with the construction of the graph"
    return node1, node2


def scaling_factor(node1, node2):
    return robjects.globalenv['scaling_factor'](
        robjects.FloatVector(node1 + 1),
        robjects.FloatVector(node2 + 1),
    )[0]


