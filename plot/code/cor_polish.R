suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
})

# Polished correlation heatmaps
# - cor_1.pdf: simulation (Batch / Condition 1 / Condition 2 / Cell type)
# - cor_2.pdf: real datasets (covid + mouse–human; Status/System + Cell type)

METHOD_ORDER <- c("Seurat", "scVI", "biolord", "SCDRL")
METHOD_LABELS <- c("Seurat" = "Seurat", "scVI" = "scVI", "biolord" = "biolord", "SCDRL" = "SCDRL")

read_cor_matrix <- function(dataset, method) {
  path <- file.path("results", sprintf("%s_%s_cor.csv", method, dataset))
  if (!file.exists(path)) {
    stop("Missing file: ", path)
  }

  if (method == "Seurat") {
    df <- read.csv(path, check.names = FALSE)
    mat <- as.matrix(df)
  } else {
    mat <- as.matrix(read.csv(path, header = FALSE))
  }

  storage.mode(mat) <- "numeric"
  mat
}

to_long <- function(mat, method, dataset, dataset_label, factor_labels) {
  stopifnot(nrow(mat) == length(factor_labels))

  df_wide <- as.data.frame(mat)
  df_wide$factor_idx <- seq_len(nrow(df_wide))
  dim_cols <- setdiff(names(df_wide), "factor_idx")

  df <- df_wide %>%
    pivot_longer(
      cols = -factor_idx,
      names_to = "dim_name",
      values_to = "cor"
    ) %>%
    mutate(
      dim = match(dim_name, dim_cols),
      method = factor(method, levels = METHOD_ORDER),
      dataset = dataset,
      dataset_label = dataset_label,
      factor = factor(factor_labels[factor_idx], levels = factor_labels)
    ) %>%
    arrange(factor_idx, dim) %>%
    select(dataset, dataset_label, method, factor, dim, cor)

  df
}

normalize_dims <- function(mat, target_dim) {
  current_dim <- ncol(mat)

  if (current_dim > target_dim) {
    mat <- mat[, seq_len(target_dim), drop = FALSE]
  } else if (current_dim < target_dim) {
    pad <- matrix(NA_real_, nrow = nrow(mat), ncol = target_dim - current_dim)
    mat <- cbind(mat, pad)
  }

  colnames(mat) <- paste0("D", seq_len(target_dim))
  mat
}

make_heatmap <- function(df, facet_formula, title) {
  max_dim <- max(df$dim, na.rm = TRUE)
  x_breaks <- if (max_dim <= 24) seq(1, max_dim, by = 2) else pretty_breaks(n = 10)(1:max_dim)

  ggplot(df, aes(x = dim, y = method, fill = cor)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_gradient2(
      low = "#2C7BB6",
      mid = "#F7F7F7",
      high = "#D7191C",
      midpoint = 0,
      limits = c(-1, 1),
      oob = squish,
      name = expression(paste("Spearman ", rho))
    ) +
    scale_x_continuous(breaks = x_breaks, expand = expansion(mult = c(0.01, 0.01))) +
    scale_y_discrete(labels = METHOD_LABELS, drop = FALSE) +
    facet_grid(
      facet_formula,
      switch = "y",
      labeller = label_wrap_gen(width = 18)
    ) +
    labs(title = title, x = "Latent dimension", y = NULL) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0),
      axis.text.x = element_text(size = 9),
      axis.text.y = element_text(size = 11, face = "bold"),
      panel.grid = element_blank(),
      strip.text = element_text(face = "bold", size = 11),
      strip.text.y.left = element_text(angle = 0, hjust = 0.5, vjust = 0.5, margin = margin(r = 6)),
      strip.placement = "outside",
      strip.background = element_rect(fill = "#F5F5F5", color = "#E0E0E0"),
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      panel.spacing = unit(0.9, "lines"),
      plot.margin = margin(t = 6, r = 8, b = 6, l = 22)
    )
}

# -------- cor_1: simulation --------
make_cor_1 <- function() {
  dataset <- "simulation"
  dataset_label <- "simulation"
  factor_labels <- c("Batch", "Condition 1", "Condition 2", "Cell type")

  mats <- lapply(METHOD_ORDER, function(m) read_cor_matrix(dataset, m))
  ncols <- vapply(mats, ncol, integer(1))
  freq <- table(ncols)
  target_dim <- min(as.integer(names(freq[freq == max(freq)])))

  df_all <- bind_rows(lapply(seq_along(METHOD_ORDER), function(i) {
    m <- METHOD_ORDER[[i]]
    mat <- normalize_dims(mats[[i]], target_dim)
    to_long(mat, m, dataset, dataset_label, factor_labels)
  }))

  p <- make_heatmap(
    df_all,
    facet_formula = vars(factor),
    title = "Correlation of latent dimensions with generative factors (simulation)"
  )

  ggsave("plot/figures/cor_1.pdf", p, width = 12.5, height = 4.8, device = cairo_pdf)
  ggsave("plot/figures/cor_1.png", p, width = 12.5, height = 4.8, dpi = 220)
}

# -------- cor_2: covid + mouse–human --------
make_cor_2 <- function() {
  cfg <- tribble(
    ~dataset, ~dataset_label, ~factor_labels,
    "haniffa", "covid", c("Status", "Cell type"),
    "mouse_human", "mouse–human", c("System", "Cell type")
  )

  df_all <- bind_rows(lapply(seq_len(nrow(cfg)), function(i) {
    dataset <- cfg$dataset[[i]]
    dataset_label <- cfg$dataset_label[[i]]
    factor_labels <- cfg$factor_labels[[i]]

    mats <- lapply(METHOD_ORDER, function(m) read_cor_matrix(dataset, m))
    ncols <- vapply(mats, ncol, integer(1))
    freq <- table(ncols)
    target_dim <- min(as.integer(names(freq[freq == max(freq)])))

    bind_rows(lapply(seq_along(METHOD_ORDER), function(j) {
      m <- METHOD_ORDER[[j]]
      mat <- normalize_dims(mats[[j]], target_dim)
      to_long(mat, m, dataset, dataset_label, factor_labels)
    }))
  })) %>%
    mutate(
      dataset_label = factor(dataset_label, levels = c("covid", "mouse–human")),
      factor = factor(as.character(factor), levels = c("Status", "System", "Cell type"))
    )

  # Stack dataset×factor panels vertically; order is controlled by factor levels above
  p <- make_heatmap(
    df_all,
    facet_formula = vars(dataset_label, factor),
    title = "Correlation of latent dimensions with biological factors"
  )

  ggsave("plot/figures/cor_2.pdf", p, width = 12.5, height = 6.0, device = cairo_pdf)
  ggsave("plot/figures/cor_2.png", p, width = 12.5, height = 6.0, dpi = 220)
}

dir.create("plot/figures", showWarnings = FALSE, recursive = TRUE)

make_cor_1()
make_cor_2()

cat("Wrote: plot/figures/cor_1.pdf\n")
cat("Wrote: plot/figures/cor_2.pdf\n")
