library(reticulate)
use_condaenv("utopia", required = TRUE)
library(Seurat)

.parse_args <- function(argv) {
    out <- list(
        seed = 9L,
        percentage = 0.05,
        npz_path = "SCDRL_data/mouse_human.npz",
        out_dir = "results/seurat",
        write_base = FALSE
    )
    if (length(argv) == 0) return(out)

    i <- 1L
    while (i <= length(argv)) {
        key <- argv[[i]]
        if (!startsWith(key, "--")) {
            i <- i + 1L
            next
        }
        key <- sub("^--", "", key)

        if (key %in% c("no_write_base")) {
            out$write_base <- FALSE
            i <- i + 1L
            next
        }

        if (i == length(argv)) break
        val <- argv[[i + 1L]]

        if (key == "seed") out$seed <- as.integer(val)
        if (key == "percentage") out$percentage <- as.numeric(val)
        if (key == "npz_path") out$npz_path <- as.character(val)
        if (key == "out_dir") out$out_dir <- as.character(val)
        if (key == "write_base") out$write_base <- as.logical(val)

        i <- i + 2L
    }
    out
}

args <- .parse_args(commandArgs(trailingOnly = TRUE))

set.seed(args$seed)

np <- import("numpy")
np$random$seed(args$seed)

npzfile <- np$load(args$npz_path)
print(npzfile$files)

counts <- t(npzfile$f[["counts"]])
factors <- npzfile$f[["factors"]]

pbmc <- CreateSeuratObject(counts = counts, project = "mouse_human")
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc)

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc), npcs = 19)

X <- Embeddings(pbmc, reduction = "pca")
dir.create(args$out_dir, showWarnings = FALSE, recursive = TRUE)

base_pca_path <- file.path(args$out_dir, "Seurat_mouse_human_pca.npy")
tagged_pca_path <- file.path(
    args$out_dir,
    paste0("Seurat_mouse_human_pca_", args$percentage, "_", args$seed, ".npy")
)
if (isTRUE(args$write_base)) {
    np$save(base_pca_path, as.matrix(X))
}
np$save(tagged_pca_path, as.matrix(X))

spearman_cor <- matrix(NA, nrow = ncol(factors), ncol = ncol(X))
for (idx in seq_len(ncol(factors))) {
    spearman_cor[idx, ] <- apply(
        X,
        2,
        function(col) cor(col, factors[, idx], method = "spearman")
    )
}

spearman_cor <- as.data.frame(spearman_cor)
rownames(spearman_cor) <- colnames(factors)
colnames(spearman_cor) <- colnames(X)

base_cor_path <- file.path(args$out_dir, "Seurat_mouse_human_cor.csv")
tagged_cor_path <- file.path(
    args$out_dir,
    paste0("Seurat_mouse_human_cor_", args$percentage, "_", args$seed, ".csv")
)
if (isTRUE(args$write_base)) {
    write.csv(spearman_cor, file = base_cor_path, row.names = FALSE)
}
write.csv(spearman_cor, file = tagged_cor_path, row.names = FALSE)

pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.36) # 17 clusters
print(table(pbmc$seurat_clusters))

seurat_clusters <- pbmc$seurat_clusters

if (
    requireNamespace("SingleCellExperiment", quietly = TRUE) &&
    requireNamespace("scran", quietly = TRUE) &&
    requireNamespace("bluster", quietly = TRUE)
) {
    sce <- as.SingleCellExperiment(pbmc)
    kmeans_clusters <- scran::clusterCells(
        sce,
        use.dimred = "PCA",
        BLUSPARAM = bluster::KmeansParam(centers = 17)
    )
} else {
    warning(
        "Bioconductor packages (SingleCellExperiment/scran/bluster) not available; using base::kmeans on PCA embeddings."
    )
    X_km <- Embeddings(pbmc, reduction = "pca")
    kmeans_clusters <- as.integer(kmeans(X_km, centers = 17, nstart = 10)$cluster)
}

print(table(kmeans_clusters))

df <- data.frame(seurat_clusters, kmeans_clusters)
base_pred_path <- file.path(args$out_dir, "Seurat_mouse_human.csv")
tagged_pred_path <- file.path(
    args$out_dir,
    paste0("Seurat_mouse_human_", args$percentage, "_", args$seed, ".csv")
)
if (isTRUE(args$write_base)) {
    write.csv(df, file = base_pred_path, row.names = FALSE)
}
write.csv(df, file = tagged_pred_path, row.names = FALSE)
