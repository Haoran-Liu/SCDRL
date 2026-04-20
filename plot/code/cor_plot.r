library(ggplot2)

# mouse_human
SCDRL <- read.csv("results/SCDRL_mouse_human_cor.csv", header = FALSE)
biolord <- read.csv("results/biolord_mouse_human_cor.csv", header = FALSE)
scVI <- read.csv("results/scVI_mouse_human_cor.csv", header = FALSE)
Seurat <- read.csv("results/Seurat_mouse_human_cor.csv")

# SCDRL
# batch
df <- data.frame(
        Dimensions = factor(as.character(1:dim(SCDRL)[2]), levels =as.character(1:dim(SCDRL)[2])),
        Correlations = unlist(SCDRL[1, ]) # batch
)

pdf("plot/figures/cor_mouse_human_batch.pdf", width = 12, height = 9)
ggplot(df, aes(x = Dimensions, y = Correlations)) +
    geom_bar(stat = "identity") +
    theme_classic() +
    labs(x = "Dimensions", y = "Correlations") +
    theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(-1, 1)
dev.off()

# condition 1
df <- data.frame(
        Dimensions = factor(as.character(1:dim(SCDRL)[2]), levels =as.character(1:dim(SCDRL)[2])),
        Correlations = unlist(SCDRL[2, ]) # condition 1
)

pdf("plot/figures/cor_mouse_human_c1.pdf", width = 12, height = 9)
ggplot(df, aes(x = Dimensions, y = Correlations)) +
    geom_bar(stat = "identity") +
    theme_classic() +
    labs(x = "Dimensions", y = "Correlations") +
    theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(-1, 1)
dev.off()

# condition 2
df <- data.frame(
        Dimensions = factor(as.character(1:dim(SCDRL)[2]), levels =as.character(1:dim(SCDRL)[2])),
        Correlations = unlist(SCDRL[3, ]) # condition 2
)

pdf("plot/figures/cor_mouse_human_c2.pdf", width = 12, height = 9)
ggplot(df, aes(x = Dimensions, y = Correlations)) +
    geom_bar(stat = "identity") +
    theme_classic() +
    labs(x = "Dimensions", y = "Correlations") +
    theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(-1, 1)
dev.off()

# cell_type
df <- data.frame(
        Dimensions = factor(as.character(1:dim(SCDRL)[2]), levels =as.character(1:dim(SCDRL)[2])),
        Correlations = unlist(SCDRL[4, ]) # cell_type
)

pdf("plot/figures/cor_mouse_human_cell_type.pdf", width = 12, height = 9)
ggplot(df, aes(x = Dimensions, y = Correlations)) +
    geom_bar(stat = "identity") +
    theme_classic() +
    labs(x = "Dimensions", y = "Correlations") +
    theme(axis.text = element_text(size = 25),
    axis.title.x = element_text(size = 25), axis.title.y = element_text(size = 25)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(-1, 1)
dev.off()
