library(reticulate)
use_python(Sys.which("python"), required = TRUE)
library(Seurat)
library(ggplot2)

np <- import("numpy")

npzfile <- np$load('results/SCDRL/SCDRL_mouse_human_0.05_9.npz')
print(npzfile$files)
SCDRL_predictions <- npzfile$f[["predictions"]]
random_idx <- npzfile$f[["random_idx"]]
labeled_idx <- npzfile$f[["labeled_idx"]]
test_idx <- npzfile$f[["test_idx"]]

npzfile <- np$load('SCDRL_data/mouse_human.npz')
print(npzfile$files)

counts <- t(npzfile$f[["counts"]][random_idx + 1, ]) # transpose to match Seurat's format
factors <- npzfile$f[["factors"]][random_idx + 1, ] # python is 0-indexed

mapping_index <- npzfile$f[["mapping_index"]]
mapping_cell_type <- npzfile$f[["mapping_cell_type"]]
names(mapping_cell_type) <- mapping_index + 1 # python is 0-indexed

mapping_system <- c("Mouse", "Human")
names(mapping_system) <- c(1, 2)

pbmc <- CreateSeuratObject(counts = counts, project = "mouse_human")

pbmc <- NormalizeData(pbmc)

pbmc <- FindVariableFeatures(pbmc)

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.36) # 17 clusters
print(table(pbmc$seurat_clusters))

pbmc <- RunUMAP(pbmc, dims = 1:10)

# ground truth
df <- pbmc[["umap"]]@cell.embeddings
df <- data.frame(df)
df <- df[test_idx + 1, ] # python is 0-indexed
df$system <- factors[test_idx + 1, 1]
df$cell_type <- factors[test_idx + 1, 2]

df$system <- df$system + 1 # python is 0-indexed
df$cell_type <- df$cell_type + 1

df$system <- mapping_system[df$system]
df$cell_type <- mapping_cell_type[df$cell_type]

pdf("plot/figures/UMAP_mouse_human_ground_truth_system.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(system))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 30)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "System")
dev.off()

pdf("plot/figures/UMAP_mouse_human_ground_truth_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 20)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cell Type")
dev.off()

# SCDRL
df$system <- SCDRL_predictions[, 1] + 1 # python is 0-indexed
df$cell_type <- SCDRL_predictions[, 2] + 1

df$system <- mapping_system[df$system]
df$cell_type <- mapping_cell_type[df$cell_type]

pdf("plot/figures/UMAP_mouse_human_SCDRL_system.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(system))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 30)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "System")
dev.off()

pdf("plot/figures/UMAP_mouse_human_SCDRL_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 20)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cell Type")
dev.off()

# biolord
npzfile <- np$load('results/biolord/biolord_mouse_human_0.05_9.npz')
print(npzfile$files)
biolord_predictions <- npzfile$f[["predictions"]]

df$system <- biolord_predictions[, 1] + 1 # python is 0-indexed
df$cell_type <- biolord_predictions[, 2] + 1

df$system <- mapping_system[df$system]
df$cell_type <- mapping_cell_type[df$cell_type]

pdf("plot/figures/UMAP_mouse_human_biolord_system.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(system))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "System")
dev.off()

pdf("plot/figures/UMAP_mouse_human_biolord_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cell Type")
dev.off()

# scVI
scvi_path <- "results/scVI/scVI_mouse_human_seed9.csv"
if (!file.exists(scvi_path)) {
    scvi_path <- "results/scVI_mouse_human.csv"
}
scVI_predictions <- read.csv(scvi_path)
tmp <- scVI_predictions[random_idx + 1, 1]
tmp <- tmp[test_idx + 1]
df$cell_type <- tmp

pdf("plot/figures/UMAP_mouse_human_scVI_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(umap_1, umap_2)) + geom_point(aes(colour = factor(cell_type))) +
    theme_classic() + theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25),
    legend.title = element_text(size = 25), legend.text = element_text(size = 25)) +
    guides(color = guide_legend(override.aes = list(size = 7))) +
    labs(color = "Cluster")
dev.off()

# Seurat
seurat_path <- "results/seurat/Seurat_mouse_human_0.05_9.csv"
if (!file.exists(seurat_path)) {
    seurat_path <- "results/Seurat_mouse_human.csv"
}
Seurat_predictions <- read.csv(seurat_path)
tmp <- Seurat_predictions[random_idx + 1, 1]
tmp <- tmp[test_idx + 1]
df$cell_type <- tmp

pdf("plot/figures/UMAP_mouse_human_Seurat_cell_type.pdf", width = 12, height = 9)
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
    
    file_name = paste0("performance/mouse_human_0.05_",
                i_seed, ".csv")
    if (!file.exists(file_name)) {
        file_name = paste0("performance/_combined/mouse_human_0.05_",
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

file_name = "performance/mouse_human_0.05_9.csv"
if (!file.exists(file_name)) {
    file_name = "performance/_combined/mouse_human_0.05_9.csv"
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

pdf("plot/figures/mouse_human_ARI.pdf", width = 9, height = 9)
ggplot(df, aes(x = method, y = ARI, fill = method)) +
    geom_bar(stat = "identity") +
    theme_classic() +
    ylab("ARI") +
    theme(axis.text = element_text(size = rel(4)),
    axis.title.x = element_blank(), axis.title.y = element_text(size = rel(3), vjust = -0.5),
    legend.position = "none", plot.title = element_text(size = rel(4)))
dev.off()
