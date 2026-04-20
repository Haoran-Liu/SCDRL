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
    height = 6.2,
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

  mean_long <- mean_df %>%
    pivot_longer(-metric, names_to = "method", values_to = "mean")

  std_long <- std_df %>%
    pivot_longer(-metric, names_to = "method", values_to = "std")

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
    DCI_informativeness_mean = "DCI: informativeness",
    DCI_disentanglement_mean = "DCI: disentanglement",
    DCI_completeness_mean = "DCI: completeness"
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

  # Visual encodings for dataset (shape + line type; color helps readability)
  ds_colors <- c(
    haniffa = "#0072B2",
    mouse_human = "#D55E00",
    simulation = "#009E73"
  )
  ds_shapes <- c(haniffa = 16, mouse_human = 17, simulation = 15)
  ds_linetypes <- c(haniffa = "solid", mouse_human = "dashed", simulation = "dotdash")

  title_txt <- paste0("Disentanglement metrics (methods × datasets, pct=", args$pct, ")")

  p <- ggplot(df_plot, aes(x = method, y = mean, group = dataset_label)) +
    geom_line(aes(color = dataset_label, linetype = dataset_label), linewidth = 0.55, alpha = 0.9) +
    geom_point(aes(color = dataset_label, shape = dataset_label), size = 2.2, stroke = 0.2) +
    geom_errorbar(
      aes(ymin = ymin, ymax = ymax, color = dataset_label),
      width = 0.12,
      linewidth = 0.35,
      alpha = 0.65
    ) +
    facet_wrap(~metric_label, ncol = 3, scales = "free_y") +
    scale_color_manual(values = c(covid = unname(ds_colors["haniffa"]), mouse_human = unname(ds_colors["mouse_human"]), simulation = unname(ds_colors["simulation"])), drop = FALSE) +
    scale_shape_manual(values = c(covid = unname(ds_shapes["haniffa"]), mouse_human = unname(ds_shapes["mouse_human"]), simulation = unname(ds_shapes["simulation"])), drop = FALSE) +
    scale_linetype_manual(values = c(covid = unname(ds_linetypes["haniffa"]), mouse_human = unname(ds_linetypes["mouse_human"]), simulation = unname(ds_linetypes["simulation"])), drop = FALSE) +
    scale_y_continuous(breaks = breaks_pretty(n = 4), labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.05, 0.08))) +
    labs(title = title_txt, x = NULL, y = NULL, color = "Dataset", shape = "Dataset", linetype = "Dataset") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 11),
      axis.text.x = element_text(angle = 22, hjust = 1, vjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_text(face = "bold"),
      panel.spacing = unit(1.2, "lines")
    )

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  out_base <- file.path(args$outdir, paste0("disentanglement_2x3_methods_x_datasets_pct", args$pct))
  ggsave(filename = paste0(out_base, ".pdf"), plot = p, width = args$width, height = args$height, dpi = args$dpi)
  ggsave(filename = paste0(out_base, ".png"), plot = p, width = args$width, height = args$height, dpi = args$dpi)

  message("Wrote: ", out_base, ".pdf")
  message("Wrote: ", out_base, ".png")
}

main()
