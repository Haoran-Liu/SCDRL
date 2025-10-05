library(SymSim)

# a tree which defines relationship between populations
phyla1 <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2,E:3);")

# total number of cells from all populations
ncells_total <- 10000
# number of cells in the rarest population
min_popsize <- 100
# number of genes
ngenes <- 500
# controls heterogeneity each population
sigma <- 0.4
# number of batches the cells are sequenced on
nbatch <- 2

# number of perturbed genes
n_diff_genes <- 100
# perturbation parameters
epsilon <- 8

####################
# SymSim
true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, min_popsize=min_popsize, i_minpop=3, ngenes=ngenes, nevf=10, evf_type="discrete", n_de_evf=9, vary="s", Sigma=sigma, phyla=phyla1, randseed=0)

data(gene_len_pool)
gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="UMI", alpha_mean=0.05, alpha_sd=0.02, gene_len=gene_len, depth_mean=5e4, depth_sd=3e3)

observed_counts_batches <- DivideBatches(observed_counts_res = observed_counts, nbatch = nbatch, batch_effect_size = 1)

save(true_counts_res, observed_counts, observed_counts_batches, file = "/ssd/haoran/SCDRL/data/simulation_data/SymSim.RData")

# load("/ssd/haoran/SCDRL/data/simulation_data/SymSim.RData")
####################
# set random seed
set.seed(1)

batch1_idx <- which(observed_counts_batches[[2]]$batch==1)
batch2_idx <- which(observed_counts_batches[[2]]$batch==2)

# cell meta
cell_meta_batch1 <- observed_counts_batches$cell_meta[batch1_idx,c("cellid", "pop")]
cell_meta_batch2 <- observed_counts_batches$cell_meta[batch2_idx,c("cellid", "pop")]
names(cell_meta_batch1) <- c("cell_id", "cell_type")
names(cell_meta_batch2) <- c("cell_id", "cell_type")
cell_meta_batch1$batch <- 0
cell_meta_batch2$batch <- 1

# counts / all
counts_batch1 <- observed_counts_batches[[1]][, batch1_idx]
counts_batch2 <- observed_counts_batches[[1]][, batch2_idx]
colnames(counts_batch1) <- cell_meta_batch1$cell_id
colnames(counts_batch2) <- cell_meta_batch2$cell_id
counts <- list(counts_batch1, counts_batch2)

# condition idx
tmp_idx <- sample(ngenes, 2*n_diff_genes, replace = FALSE)
cond1_idx <- tmp_idx[1:n_diff_genes]
cond2_idx <- tmp_idx[(n_diff_genes+1):(2*n_diff_genes)]

# batch_1 idx
len = ncol(counts_batch1)
tmp_idx <- sample(len)
ctrl_healthy_batch1_idx <- tmp_idx[1:round(len/4)]
ctrl_severe_batch1_idx <- tmp_idx[(round(len/4)+1):round(len/2)]
stim_healthy_batch1_idx <- tmp_idx[(round(len/2)+1):round(len*3/4)]
stim_severe_batch1_idx <- tmp_idx[(round(len*3/4)+1):len]

cell_meta_batch1$condition_1 <- -1
cell_meta_batch1$condition_2 <- -1

cell_meta_batch1$condition_1[ctrl_healthy_batch1_idx] <- 0
cell_meta_batch1$condition_2[ctrl_healthy_batch1_idx] <- 0

cell_meta_batch1$condition_1[ctrl_severe_batch1_idx] <- 0
cell_meta_batch1$condition_2[ctrl_severe_batch1_idx] <- 1

cell_meta_batch1$condition_1[stim_healthy_batch1_idx] <- 1
cell_meta_batch1$condition_2[stim_healthy_batch1_idx] <- 0

cell_meta_batch1$condition_1[stim_severe_batch1_idx] <- 1
cell_meta_batch1$condition_2[stim_severe_batch1_idx] <- 1

# batch_2 idx
len = ncol(counts_batch2)
tmp_idx <- sample(len)
ctrl_healthy_batch2_idx <- tmp_idx[1:round(len/4)]
ctrl_severe_batch2_idx <- tmp_idx[(round(len/4)+1):round(len/2)]
stim_healthy_batch2_idx <- tmp_idx[(round(len/2)+1):round(len*3/4)]
stim_severe_batch2_idx <- tmp_idx[(round(len*3/4)+1):len]

cell_meta_batch2$condition_1 <- -1
cell_meta_batch2$condition_2 <- -1

cell_meta_batch2$condition_1[ctrl_healthy_batch2_idx] <- 0
cell_meta_batch2$condition_2[ctrl_healthy_batch2_idx] <- 0

cell_meta_batch2$condition_1[ctrl_severe_batch2_idx] <- 0
cell_meta_batch2$condition_2[ctrl_severe_batch2_idx] <- 1

cell_meta_batch2$condition_1[stim_healthy_batch2_idx] <- 1
cell_meta_batch2$condition_2[stim_healthy_batch2_idx] <- 0

cell_meta_batch2$condition_1[stim_severe_batch2_idx] <- 1
cell_meta_batch2$condition_2[stim_severe_batch2_idx] <- 1

# control / healthy
counts_ctrl_healthy <- list(counts_batch1[,ctrl_healthy_batch1_idx], counts_batch2[,ctrl_healthy_batch2_idx])

# control / severe
counts_ctrl_severe <- list(counts_batch1[,ctrl_severe_batch1_idx], counts_batch2[,ctrl_severe_batch2_idx])
counts_ctrl_severe <- lapply(counts_ctrl_severe, function(x){
    # uniform distribution
    x[cond2_idx,] <- x[cond2_idx,] + matrix(runif(n_diff_genes * dim(x)[2], min = -1, max = 1) + epsilon, n_diff_genes, dim(x)[2])
    x[cond2_idx,] <- round(x[cond2_idx,])
    #
    return(x)
})

# stimulation / healthy
counts_stim_healthy <- list(counts_batch1[,stim_healthy_batch1_idx], counts_batch2[,stim_healthy_batch2_idx])
counts_stim_healthy <- lapply(counts_stim_healthy, function(x){
    # uniform distribution
    x[cond1_idx,] <- x[cond1_idx,] + matrix(runif(n_diff_genes * dim(x)[2], min = -1, max = 1) + epsilon, n_diff_genes, dim(x)[2])
    x[cond1_idx,] <- round(x[cond1_idx,])
    #
    return(x)
})

# stimulation / severe
counts_stim_severe <- list(counts_batch1[,stim_severe_batch1_idx], counts_batch2[,stim_severe_batch2_idx])
counts_stim_severe <- lapply(counts_stim_severe, function(x){
    # uniform distribution
    x[cond1_idx,] <- x[cond1_idx,] + matrix(runif(n_diff_genes * dim(x)[2], min = -1, max = 1) + epsilon, n_diff_genes, dim(x)[2])
    x[cond1_idx,] <- round(x[cond1_idx,])
    # uniform distribution
    x[cond2_idx,] <- x[cond2_idx,] + matrix(runif(n_diff_genes * dim(x)[2], min = -1, max = 1) + epsilon, n_diff_genes, dim(x)[2])
    x[cond2_idx,] <- round(x[cond2_idx,])
    #
    return(x)
})

# save data
counts_ctrl_healthy <- cbind(counts_ctrl_healthy[[1]], counts_ctrl_healthy[[2]])
counts_ctrl_severe <- cbind(counts_ctrl_severe[[1]], counts_ctrl_severe[[2]])
counts_stim_healthy <- cbind(counts_stim_healthy[[1]], counts_stim_healthy[[2]])
counts_stim_severe <- cbind(counts_stim_severe[[1]], counts_stim_severe[[2]])

metadata_ctrl_healthy <- rbind(cell_meta_batch1[ctrl_healthy_batch1_idx, ], cell_meta_batch2[ctrl_healthy_batch2_idx, ])
metadata_ctrl_severe <- rbind(cell_meta_batch1[ctrl_severe_batch1_idx, ], cell_meta_batch2[ctrl_severe_batch2_idx, ])
metadata_stim_healthy <- rbind(cell_meta_batch1[stim_healthy_batch1_idx, ], cell_meta_batch2[stim_healthy_batch2_idx, ])
metadata_stim_severe <- rbind(cell_meta_batch1[stim_severe_batch1_idx, ], cell_meta_batch2[stim_severe_batch2_idx, ])

counts <- do.call(cbind, list(counts_ctrl_healthy, counts_ctrl_severe, counts_stim_healthy, counts_stim_severe))
metadata <- do.call(rbind, list(metadata_ctrl_healthy, metadata_ctrl_severe, metadata_stim_healthy, metadata_stim_severe))
metadata <- metadata[,c("cell_id", "batch", "condition_1", "condition_2", "cell_type")]
metadata$cell_type <- metadata$cell_type - 1 # 0-based

write.table(counts, "/ssd/haoran/SCDRL/data/simulation_data/counts_gene_by_cell.txt", quote=FALSE, row.names = FALSE, sep = "\t")
write.table(metadata, "/ssd/haoran/SCDRL/data/simulation_data/metadata.txt", quote=FALSE, row.names = FALSE, sep = "\t")
