#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

parse_args <- function(args) {
  out <- list(
    pct = "0.05",
    outdir = "plot/figures/paper",
    width = 10,
    height = 4.6,
    dpi = 300
  )

  if (length(args) == 0) return(out)

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    val <- if (i < length(args)) args[[i + 1]] else NA_character_

    if (key %in% c("--pct")) {
      out$pct <- val
      i <- i + 2
      next
    }
    if (key %in% c("--outdir")) {
      out$outdir <- val
      i <- i + 2
      next
    }
    if (key %in% c("--width")) {
      out$width <- as.numeric(val)
      i <- i + 2
      next
    }
    if (key %in% c("--height")) {
      out$height <- as.numeric(val)
      i <- i + 2
      next
    }
    if (key %in% c("--dpi")) {
      out$dpi <- as.integer(val)
      i <- i + 2
      next
    }

    stop(paste0("Unknown argument: ", key))
  }

  out
}

load_dataset_summary <- function(dataset, pct) {
  mean_path <- file.path(
    "performance",
    "summary_tables",
    "disentanglement",
    paste0(dataset, "_", pct, "_mean.csv")
  )
  std_path <- file.path(
    "performance",
    "summary_tables",
    "disentanglement",
    paste0(dataset, "_", pct, "_std.csv")
  )

  if (!file.exists(mean_path)) stop(paste0("Missing file: ", mean_path))
  if (!file.exists(std_path)) stop(paste0("Missing file: ", std_path))

  mean_df <- read_csv(mean_path, show_col_types = FALSE)
  std_df <- read_csv(std_path, show_col_types = FALSE)

  mean_long <- mean_df %>% pivot_longer(-metric, names_to = "method", values_to = "mean")
  std_long <- std_df %>% pivot_longer(-metric, names_to = "method", values_to = "std")

  mean_long %>%
    inner_join(std_long, by = c("metric", "method")) %>%
    mutate(dataset = dataset)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  datasets <- c("haniffa", "mouse_human", "simulation")
  df <- bind_rows(lapply(datasets, function(d) load_dataset_summary(d, args$pct)))

  metrics <- c(
    "MIG_mean",
    "SAP_mean",
    "Hungarian_matched_mean",
    "DCI_informativeness_mean",
    "DCI_disentanglement_mean",
    "DCI_completeness_mean"
  )

  metric_labels <- c(
    MIG_mean = "MIG",
    SAP_mean = "SAP",
    Hungarian_matched_mean = "Hungarian",
    DCI_informativeness_mean = "DCI Info",
    DCI_disentanglement_mean = "DCI Dis",
    DCI_completeness_mean = "DCI Comp"
  )

  method_order <- c("Seurat", "scVI", "biolord", "SCDRL")
  present_methods <- intersect(method_order, unique(df$method))
  if (length(present_methods) == 0) present_methods <- unique(df$method)

  dataset_order <- c("haniffa", "mouse_human", "simulation")
  present_datasets <- intersect(dataset_order, unique(df$dataset))

  df_plot <- df %>%
    filter(metric %in% metrics) %>%
    mutate(
      metric = factor(metric, levels = metrics),
      metric_label = factor(metric_labels[as.character(metric)], levels = metric_labels[metrics]),
      method = factor(method, levels = present_methods),
      dataset = factor(dataset, levels = present_datasets),
      dataset_label = recode(as.character(dataset), haniffa = "covid", .default = as.character(dataset)),
      ymin = pmax(mean - std, 0),
      ymax = mean + std
    )

  # Colorblind-friendly palette (Okabe–Ito + neutral gray)
  method_colors <- c(
    Seurat = "#7A7A7A",
    scVI = "#0072B2",
    biolord = "#E69F00",
    SCDRL = "#009E73"
  )

  title_txt <- paste0("Disentanglement metrics (pct = ", args$pct, ")")

  dodge <- position_dodge(width = 0.78)

  p <- ggplot(df_plot, aes(x = metric_label, y = mean, fill = method)) +
    geom_col(
      position = dodge,
      width = 0.72,
      color = NA
    ) +
    geom_errorbar(
      aes(ymin = ymin, ymax = ymax),
      position = dodge,
      width = 0.18,
      linewidth = 0.35,
      color = "#2B2B2B",
      alpha = 0.7
    ) +
    facet_wrap(~dataset_label, ncol = 3, scales = "free_y") +
    scale_fill_manual(values = method_colors, drop = FALSE) +
    scale_y_continuous(
      breaks = breaks_pretty(n = 4),
      labels = label_number(accuracy = 0.01),
      expand = expansion(mult = c(0.02, 0.08))
    ) +
    labs(title = title_txt, x = NULL, y = "Score", fill = "Method") +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#E6E6E6", linewidth = 0.35),
      strip.background = element_rect(fill = "#F5F5F5", color = "#E0E0E0"),
      strip.text = element_text(face = "bold", size = 11),
      axis.title.y = element_text(face = "bold", size = 11),
      axis.text.x = element_text(angle = 25, hjust = 1, vjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_text(face = "bold"),
      legend.key.width = unit(1.2, "lines"),
      legend.spacing.x = unit(0.6, "lines"),
      panel.spacing = unit(1.0, "lines")
    )

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  out_base <- file.path(args$outdir, paste0("disentanglement_datasets_x_metrics_pct", args$pct))
  ggsave(filename = paste0(out_base, ".pdf"), plot = p, width = args$width, height = args$height, dpi = args$dpi)
  ggsave(filename = paste0(out_base, ".png"), plot = p, width = args$width, height = args$height, dpi = args$dpi)

  message("Wrote: ", out_base, ".pdf")
  message("Wrote: ", out_base, ".png")
}

main()
