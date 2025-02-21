library(tidyverse)
library(easystats)
library(circular)
library(glmmTMB)
library(brms)

# Read data ---------------------------------------------------------------
df <- read.csv("../data/data_tap.csv")

dfsub <- read.csv("https://raw.githubusercontent.com/RealityBending/InteroceptionPrimals/main/data/data_participants.csv") |>
  rename(Participant="participant_id") |>
  select(-matches("\\d")) |>
  filter(Participant %in% unique(df$Participant)) |>
  arrange(as.character(Participant))

# Modelling ---------------------------------------------------------------

# options(mc.cores = parallel::detectCores())
#
# f <- bf(Cardiac_Radian ~ Condition + (Condition | Participant),
#         kappa ~ Condition + (Condition | Participant),
#         family = von_mises())
#
# model <- brm(
#   f,
#   data = df,
#   # refresh = 0,
#   iter = 2000,
#   chains = 4,
#   cores = 4,
#   num_paths=10,
#   single_path_draws=2000,
#   algorithm="pathfinder",
#   backend="cmdstanr")
#
# parameters::parameters(model)
# estimate_contrasts(model)
#
# estimate_relation(model)
#
# estimate_means(model, by = c("Condition")) |>
#   ggplot(aes(x=Condition, y=Mean)) +
#   geom_pointrange(aes(ymin=CI_low, ymax=CI_high))
#
#
#
# dffeat <- marginaleffects::predictions(model,
#                                        newdata=insight::get_datagrid(model, include_random=TRUE),
#                                        dpar="kappa") |>
#   as.data.frame() |>
#   select(Participant, Condition, Orientation=estimate) |>
#   pivot_wider(names_from="Condition", values_from="Orientation", names_prefix="Orientation_") |>
#   arrange(as.character(Participant))


df |>
  ggplot(aes(x=Cardiac_Angle)) +
  geom_density(aes(color=Participant)) +
  facet_wrap(~Condition, scales="free")



# Circular ----------------------------------------------------------------


circular_parameters <- function(x){

  # Density
  c <- circular::circular(x, units = "degrees", rotation="clock")

  # plot(density(c, bw = 360, kernel="vonmises"), zero.line = TRUE, points.plot=TRUE)
  # arrows.circular(mean(c))

  # Uniformity (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7373216/)
  test <- circular::rayleigh.test(c)
  params <- data.frame(
    # Rayleigh = test$statistic,  # Same as strength
    Rayleigh_p = test$p.value,
    Kuiper = circular::kuiper.test(c, alpha=0.05)$statistic,
    Watson = circular::watson.test(c, alpha=0.05)$statistic,
    # RaoSpacing = circular::rao.spacing.test(c, alpha=0.05)$statistic,
    # Angular mean and length (strength)
    Orientation = as.numeric(mean(c)),
    Strength = as.numeric(rho.circular(c))
  )

  params
}


dffeat <- data.frame()
for(ppt in unique(df$Participant)) {
  for(cond in unique(df$Condition)) {
    if(nrow(filter(df, Participant==ppt, Condition==cond))==0) next
    angles <- filter(df, Participant==ppt, Condition==cond)$Cardiac_Angle
    params <- circular_parameters(angles)
    params$Participant <- ppt
    params$Condition <- cond
    dffeat <- rbind(dffeat, params)
  }
}





# Correlations ------------------------------------------------------------

dffeatwide <- dffeat |>
  select(-Kuiper, -Watson) |>
  pivot_wider(names_from="Condition",
              values_from=c("Rayleigh_p", "Orientation", "Strength")) |>
  datawizard::data_addprefix("TAP_", exclude="Participant")


dfsub <- merge(dfsub, dffeatwide, by="Participant")

plot_cor <- function(df1, df2=NULL) {
  correlation::correlation(data=df1,
                           data2=df2,
                           p_adjust="none") |>
    summary(redundant=TRUE) |>
    correlation::cor_sort() |>
    plot() +
    theme(axis.text.x=element_text(angle=45, hjust=1))
}

plot_cor(select(dfsub, starts_with("TAP")))
plot_cor(select(dfsub, matches("MAIA|IAS")),
         select(dfsub, starts_with("TAP")))
plot_cor(select(dfsub, matches("HCT_")),
         select(dfsub, starts_with("TAP")))
plot_cor(select(dfsub, matches("HRV")),
         select(dfsub, starts_with("TAP")))


