library(reticulate)
use_condaenv("utopia")
library(Seurat)

np <- import("numpy")
npzfile <- np$load('/Users/haoran/Documents/SCDRL/data/SCDRL_data/simulation_data.npz')
print(npzfile$files)

counts <- t(npzfile$f[["counts"]]) # transpose to match Seurat's format
factors <- npzfile$f[["factors"]]

pbmc <- CreateSeuratObject(counts = counts, project = "simulation")

pbmc <- NormalizeData(pbmc)

pbmc <- FindVariableFeatures(pbmc)

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# SingleCellExperiment
sce <- as.SingleCellExperiment(pbmc)

pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.1102) # 16 clusters
print(table(pbmc$seurat_clusters))
# pbmc <- RunUMAP(pbmc, dims = 1:10)
# DimPlot(pbmc, reduction = "umap")

seurat_clusters <- pbmc$seurat_clusters

####################
####################
library(scran)
library(bluster)

# nn_clusters <- clusterCells(sce, use.dimred="PCA", BLUSPARAM=NNGraphParam(k=10))
# print(table(nn_clusters))

kmeans_clusters <- clusterCells(sce, use.dimred="PCA", BLUSPARAM=KmeansParam(centers=16)) # 16 clusters
print(table(kmeans_clusters))

df <- data.frame(seurat_clusters, kmeans_clusters)
write.csv(df, file = "/Users/haoran/Documents/SCDRL/results/Seurat_simulation.csv", row.names = FALSE)
