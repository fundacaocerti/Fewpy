from enum import Enum


class DistanceMetric(Enum):

    EUCLIDEAN = "euclidean"
    COSINE_SIMILARITY = "cosine"
    LEARNABLE_MAHALANOBIS = "learnable_mahalanobis"
    FULL_MATRIX_MAHALANOBIS = "full_matrix_mahalanobis"
    DIAGONAL_MAHALANOBIS = "diag_mahalanobis"
    SOFTMAX_KV = "softmax_kv"
    KV = "kv"