library(reticulate)
use_python(Sys.which("python"), required = TRUE)
library(Seurat)
library(ggplot2)
library(reshape2)

np <- import("numpy")

# SCDRL
SCDRL_f1_batch <- c()
SCDRL_f1_condition_1 <- c()
SCDRL_f1_condition_2 <- c()
SCDRL_f1_cell_type <- c()

SCDRL_ARI_batch <- c()
SCDRL_ARI_condition_1 <- c()
SCDRL_ARI_condition_2 <- c()
SCDRL_ARI_cell_type <- c()

percentage <- c(0.05, 0.15, 0.25)
seed <- c(9, 19, 29, 39, 49, 59, 69, 79, 89, 99)

for (i_percentage in percentage) {
    for (i_seed in seed) {
        file_path <- paste0("results/SCDRL/SCDRL_simulation_",
                        i_percentage, "_",
                        i_seed, '.npz')
        npzfile <- np$load(file_path)
        
        SCDRL_performance <- npzfile$f[["performance"]]
        
        SCDRL_f1_batch <- c(SCDRL_f1_batch, SCDRL_performance[2, 1])
        SCDRL_f1_condition_1 <- c(SCDRL_f1_condition_1, SCDRL_performance[2, 2])
        SCDRL_f1_condition_2 <- c(SCDRL_f1_condition_2, SCDRL_performance[2, 3])
        SCDRL_f1_cell_type <- c(SCDRL_f1_cell_type, SCDRL_performance[2, 4])
        
        SCDRL_ARI_batch <- c(SCDRL_ARI_batch, SCDRL_performance[3, 1])
        SCDRL_ARI_condition_1 <- c(SCDRL_ARI_condition_1, SCDRL_performance[3, 2])
        SCDRL_ARI_condition_2 <- c(SCDRL_ARI_condition_2, SCDRL_performance[3, 3])
        SCDRL_ARI_cell_type <- c(SCDRL_ARI_cell_type, SCDRL_performance[3, 4])
    }
}

#
biolord_f1_batch <- c()
biolord_f1_condition_1 <- c()
biolord_f1_condition_2 <- c()
biolord_f1_cell_type <- c()

biolord_ARI_batch <- c()
biolord_ARI_condition_1 <- c()
biolord_ARI_condition_2 <- c()
biolord_ARI_cell_type <- c()

for (i_percentage in percentage) {
    for (i_seed in seed) {
        file_path <- paste0("results/biolord/biolord_simulation_",
                        i_percentage, "_",
                        i_seed, '.npz')
        npzfile <- np$load(file_path)
        
        biolord_performance <- npzfile$f[["performance"]]
        
        biolord_f1_batch <- c(biolord_f1_batch, biolord_performance[2, 1])
        biolord_f1_condition_1 <- c(biolord_f1_condition_1, biolord_performance[2, 2])
        biolord_f1_condition_2 <- c(biolord_f1_condition_2, biolord_performance[2, 3])
        biolord_f1_cell_type <- c(biolord_f1_cell_type, biolord_performance[2, 4])
        
        biolord_ARI_batch <- c(biolord_ARI_batch, biolord_performance[3, 1])
        biolord_ARI_condition_1 <- c(biolord_ARI_condition_1, biolord_performance[3, 2])
        biolord_ARI_condition_2 <- c(biolord_ARI_condition_2, biolord_performance[3, 3])
        biolord_ARI_cell_type <- c(biolord_ARI_cell_type, biolord_performance[3, 4])
    }
}

#
tmp_df_1 <- data.frame("percentage"=rep(percentage, each=10),
                        "seed"=seed,
                        "batch" = SCDRL_f1_batch,
                        "condition_1" = SCDRL_f1_condition_1,
                        "condition_2" = SCDRL_f1_condition_2,
                        "cell_type" = SCDRL_f1_cell_type
                )

tmp_df_2 <- data.frame("percentage"=rep(percentage, each=10),
                        "seed"=seed,
                        "batch" = SCDRL_ARI_batch,
                        "condition_1" = SCDRL_ARI_condition_1,
                        "condition_2" = SCDRL_ARI_condition_2,
                        "cell_type" = SCDRL_ARI_cell_type
                )

# melt.data.frame
tmp_df_1 <- melt(tmp_df_1, id = c("percentage", "seed"), variable.name = "attribute", value.name = "f1_score")
tmp_df_2 <- melt(tmp_df_2, id = c("percentage", "seed"), variable.name = "attribute", value.name = "ARI")

df_SCDRL <- tmp_df_1
df_SCDRL$ARI = tmp_df_2$ARI
df_SCDRL$method = "SCDRL"

#
tmp_df_1 <- data.frame("percentage"=rep(percentage, each=10),
                        "seed"=seed,
                        "batch" = biolord_f1_batch,
                        "condition_1" = biolord_f1_condition_1,
                        "condition_2" = biolord_f1_condition_2,
                        "cell_type" = biolord_f1_cell_type
                )

tmp_df_2 <- data.frame("percentage"=rep(percentage, each=10),
                        "seed"=seed,
                        "batch" = biolord_ARI_batch,
                        "condition_1" = biolord_ARI_condition_1,
                        "condition_2" = biolord_ARI_condition_2,
                        "cell_type" = biolord_ARI_cell_type
                )

# melt.data.frame
tmp_df_1 <- melt(tmp_df_1, id = c("percentage", "seed"), variable.name = "attribute", value.name = "f1_score")
tmp_df_2 <- melt(tmp_df_2, id = c("percentage", "seed"), variable.name = "attribute", value.name = "ARI")

df_biolord <- tmp_df_1
df_biolord$ARI = tmp_df_2$ARI
df_biolord$method = "biolord"

#
df <- rbind(df_SCDRL, df_biolord)
df <- df[c("method", "percentage", "seed", "attribute", "f1_score", "ARI")]
df$method <- factor(df$method, levels = c("SCDRL", "biolord"))

for (i_attribute in c("batch", "condition_1", "condition_2")) {
    for (i_percentage in percentage) {
        
        tmp_df <- df[df$attribute == i_attribute, ]
        tmp_df <- tmp_df[tmp_df$percentage == i_percentage, ]
        
        file_name = paste0("plot/figures/simulation_f1_",
                    i_attribute, "_",
                    i_percentage, ".pdf"
                    )
        
        p <- ggplot(tmp_df, aes(x = method, y = f1_score, color = method)) +
                    geom_boxplot(outlier.shape = NA) +
                    geom_jitter(size = rel(3.5)) +
                    theme_classic() +
                    ylim(0, NA) +
                    xlab(paste0("percentage = ", i_percentage)) +
                    theme(axis.text = element_text(size = rel(3.5)),
                            axis.title.x = element_text(size = rel(3.5)),
                            axis.title.y = element_blank(),
                            legend.position = "none")
        
        pdf(file_name, width = 6, height = 9)
        print(p)
        dev.off()
    
    }
}

for (i_percentage in percentage) {
    
    tmp_df <- df[df$attribute == "cell_type", ]
    tmp_df <- tmp_df[tmp_df$percentage == i_percentage, ]
    
    file_name = paste0("plot/figures/simulation_f1_",
                "cell_type", "_",
                i_percentage, ".pdf"
                )
    
    p <- ggplot(tmp_df, aes(x = method, y = f1_score, color = method)) +
                geom_boxplot(outlier.shape = NA) +
                geom_jitter(size = rel(3.5)) +
                theme_classic() +
                ylim(0, 0.8) +
                xlab(paste0("percentage = ", i_percentage)) +
                theme(axis.text = element_text(size = rel(3.5)),
                        axis.title.x = element_text(size = rel(3.5)),
                        axis.title.y = element_blank(),
                        legend.position = "none")
    
    pdf(file_name, width = 6, height = 9)
    print(p)
    dev.off()
}
