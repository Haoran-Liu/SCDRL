library(reticulate)
use_python(Sys.which("python"), required = TRUE)
library(Seurat)
library(ggplot2)

np <- import("numpy")

npzfile <- np$load('results/SCDRL/SCDRL_simulation_0.05_9.npz')
print(npzfile$files)
SCDRL_predictions <- npzfile$f[["predictions"]]
random_idx <- npzfile$f[["random_idx"]]
labeled_idx <- npzfile$f[["labeled_idx"]]
test_idx <- npzfile$f[["test_idx"]]

npzfile <- np$load('SCDRL_data/simulation_data.npz')
print(npzfile$files)

counts <- t(npzfile$f[["counts"]][random_idx + 1, ]) # transpose to match Seurat's format
factors <- npzfile$f[["factors"]][random_idx + 1, ] # python is 0-indexed

pbmc <- CreateSeuratObject(counts = counts, project = "simulation")

pbmc <- NormalizeData(pbmc)

pbmc <- FindVariableFeatures(pbmc)

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.1102) # 16 clusters
print(table(pbmc$seurat_clusters))

pbmc <- RunUMAP(pbmc, dims = 1:10)

# ground truth
df <- pbmc[["umap"]]@cell.embeddings
df <- data.frame(df)
df <- df[test_idx + 1, ] # python is 0-indexed
df$batch <- factors[test_idx + 1, 1]
df$condition_1 <- factors[test_idx + 1, 2]
df$condition_2 <- factors[test_idx + 1, 3]
df$cell_type <- factors[test_idx + 1, 4]

pdf("plot/figures/UMAP_simulation_ground_truth_batch.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(batch))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Batch")
dev.off()

pdf("plot/figures/UMAP_simulation_ground_truth_condition_1.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(condition_1))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Condition 1")
dev.off()

pdf("plot/figures/UMAP_simulation_ground_truth_condition_2.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(condition_2))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Condition 2")
dev.off()

pdf("plot/figures/UMAP_simulation_ground_truth_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cell Type")
dev.off()

# SCDRL
df$batch <- SCDRL_predictions[, 1]
df$condition_1 <- SCDRL_predictions[, 2]
df$condition_2 <- SCDRL_predictions[, 3]
df$cell_type <- SCDRL_predictions[, 4]

pdf("plot/figures/UMAP_simulation_SCDRL_batch.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(batch))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Batch")
dev.off()

pdf("plot/figures/UMAP_simulation_SCDRL_condition_1.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(condition_1))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Condition 1")
dev.off()

pdf("plot/figures/UMAP_simulation_SCDRL_condition_2.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(condition_2))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Condition 2")
dev.off()

pdf("plot/figures/UMAP_simulation_SCDRL_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cell Type")
dev.off()

# biolord
npzfile <- np$load('results/biolord/biolord_simulation_0.05_9.npz')
print(npzfile$files)
biolord_predictions <- npzfile$f[["predictions"]]

df$batch <- biolord_predictions[, 1]
df$condition_1 <- biolord_predictions[, 2]
df$condition_2 <- biolord_predictions[, 3]
df$cell_type <- biolord_predictions[, 4]

pdf("plot/figures/UMAP_simulation_biolord_batch.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(batch))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Batch")
dev.off()

pdf("plot/figures/UMAP_simulation_biolord_condition_1.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(condition_1))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Condition 1")
dev.off()

pdf("plot/figures/UMAP_simulation_biolord_condition_2.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(condition_2))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Condition 2")
dev.off()

pdf("plot/figures/UMAP_simulation_biolord_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cell Type")
dev.off()

# scVI
scVI_predictions <- read.csv("results/scVI_simulation.csv")
tmp <- scVI_predictions[random_idx + 1, 1]
tmp <- tmp[test_idx + 1]
df$cell_type <- tmp

pdf("plot/figures/UMAP_simulation_scVI_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cluster")
dev.off()

# Seurat
Seurat_predictions <- read.csv("results/seurat/Seurat_simulation_0.05_9.csv")
tmp <- Seurat_predictions[random_idx + 1, 1]
tmp <- tmp[test_idx + 1]
df$cell_type <- tmp

pdf("plot/figures/UMAP_simulation_Seurat_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cluster")
dev.off()





# ARI
tmp_SCDRL_ARI = 0
tmp_biolord_ARI = 0
seed <- c(9, 19, 29, 39, 49, 59, 69, 79, 89, 99)
for (i_seed in seed) {
    
    file_name = paste0("performance/simulation_0.05_",
                i_seed, ".csv")
    if (!file.exists(file_name)) {
        file_name = paste0("performance/_combined/simulation_0.05_",
                    i_seed, ".csv")
    }
    
    df <- read.csv(file_name, row.names = 1)
    df <- t(df)
    df <- data.frame(df)
    df$method <- rownames(df)
    df <- df[, c("method", "ARI_cell_type")]
    colnames(df) <- c("method", "ARI")
    
    tmp_SCDRL_ARI = tmp_SCDRL_ARI + df[df$method == "SCDRL", "ARI"]
    tmp_biolord_ARI = tmp_biolord_ARI + df[df$method == "biolord", "ARI"]

}

file_name = "performance/simulation_0.05_9.csv"
if (!file.exists(file_name)) {
    file_name = "performance/_combined/simulation_0.05_9.csv"
}
df <- read.csv(file_name, row.names = 1)
df <- t(df)
df <- data.frame(df)
df$method <- rownames(df)
df <- df[, c("method", "ARI_cell_type")]
colnames(df) <- c("method", "ARI")

df[df$method == "SCDRL", "ARI"] = tmp_SCDRL_ARI/length(seed)
df[df$method == "biolord", "ARI"] = tmp_biolord_ARI/length(seed)

df <- df[df$method %in% c("SCDRL", "scVI", "Seurat"), ]

df$method <- factor(df$method, levels = c("SCDRL", "scVI", "Seurat"))

pdf("plot/figures/simulation_ARI.pdf", width = 9, height = 9)
ggplot(df, aes(x = method, y = ARI, fill = method)) +
    geom_bar(stat = "identity") +
    theme_classic() +
    ylab("ARI") +
    ylim(0, 0.6) +
    theme(axis.text = element_text(size = rel(4)),
    axis.title.x = element_blank(), axis.title.y = element_text(size = rel(3), vjust = -0.5),
    legend.position = "none", plot.title = element_text(size = rel(4)))
dev.off()
