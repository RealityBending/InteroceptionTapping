library(tidyverse)
library(easystats)
library(circular)
library(brms)



# Convenience functions ---------------------------------------------------

deg2rad <- function(deg) {
  z_deg <- deg %% 360
  z_deg[z_deg > 180] <- z_deg[z_deg > 180] - 360
  z_deg * (pi / 180)
  }

circular_parameters <- function(x){

  x <- x[!is.na(x)]  # Remove na

  # Density
  c <- circular::circular(x, units = "degrees", rotation="clock")
  D <- data.frame(density(c, bw = 100,
                          kernel="vonmises",
                          control.circular=list(units = "degrees"))[c("x", "y")])
  D$y <- datawizard::rescale(D$y, to = c(0, 1), from = c(0, max(D$y)))
  D$x <- D$x + 360

  # Angular mean and length (strength)
  average <- mean(c)
  rho <- rho.circular(c)

  # Uniformity (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7373216/)
  c <- circular::circular(x, units="degrees")
  test <- circular::rayleigh.test(c)
  test <- data.frame(
    Rayleigh = test$statistic,
    Rayleigh_p = test$p.value,
    Kuiper = circular::kuiper.test(c, alpha=0.05)$statistic,
    Watson = circular::watson.test(c, alpha=0.05)$statistic,
    RaoSpacing = circular::rao.spacing.test(c, alpha=0.05)$statistic
  )

  # Histogram
  H <- hist(x, plot = FALSE, breaks = seq(0, 360, by=10))
  H <- data.frame(Count = H$counts / max(H$counts),
                  Angle = H$mids,
                  Width = H$breaks[2] - H$breaks[1])

  list(density = D, hist = H, test = test)
}

theme_custom <- list(
  scale_x_continuous(breaks=seq(0, 360, by=90), expand=c(0,0), lim=c(0, 360)),
  scale_y_continuous(breaks=NULL),
  coord_polar(start = 2*pi),
  theme_bw(),
  theme(
    panel.grid.major.y = element_line(colour = c("black", NA)),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
  )
)


distance_harmonics <- function(x) {
  freqs <- c(0.25, 0.5, 1, 2, 4, 8, 16)
  closest <- freqs[sapply(x, \(v) which.min(abs(freqs - v)))]
  closest - x
}

# Read data ---------------------------------------------------------------
df <- read.csv("../data/data_tap.csv") |>
  rename(Participant = participant_id) |>
  mutate(Cardiac_Angle = abs(Closest_R_Pre) / (abs(Closest_R_Pre) + Closest_R_Post) * 360,
         RSP_Angle = ifelse(RSP_Phase == 1,
                            abs(Closest_RSP_Trough_Pre) / (abs(Closest_RSP_Trough_Pre) + Closest_RSP_Peak_Post) * 180,
                            abs(Closest_RSP_Peak_Pre) / (abs(Closest_RSP_Peak_Pre) + Closest_RSP_Trough_Post) * 180 + 180),
         Rate_Ratio = Tapping_Rate / ECG_Rate,
         Rate_Harmonic_Distance = distance_harmonics(Rate_Ratio),
         Condition = fct_relevel(Condition, "Baseline", "Slower", "Faster", "Random", "Heart"))

summary(df$RSP_Angle)
# head(df)

# Cleaning ---------------------------------------------------------------------

df <- filter(df, !is.na(Cardiac_Angle)) # Remove NA values

ggplot(df, aes(x = Tapping_Rate)) +
  geom_density()

df$Tapping_Rate[df$Tapping_Rate > 600] <- NA


# Rate ====================================================================


df |>
  group_by(Participant, Condition) |>
  standardize() |>
  ungroup() |>
  filter(Condition != "Random") |>
  ggplot(aes(x = Trial_Order,  color=Condition)) +
  geom_line(aes(y = ECG_Rate)) +
  geom_line(aes(y = Tapping_Rate), linetype = "dashed") +
  facet_wrap(~Participant, scales="free")

df |>
  ggplot(aes(x = Trial_Order, y = Rate_Ratio, group=Participant)) +
  geom_line() +
  scale_y_continuous(breaks = c(0.5, 1, 2, 4)) +
  facet_wrap(~Condition, scales="free")

df |>
  ggplot(aes(x = Trial_Order, y = Rate_Harmonic_Distance, group=Participant)) +
  geom_line() +
  facet_wrap(~Condition, scales="free")

# TODO: what about lagged correlation?
df |>
  ggplot(aes(x = ECG_Rate, y = Tapping_Rate, color=Participant)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE) +
  facet_wrap(~Condition, scales="free")
model <- glmmTMB::glmmTMB(Tapping_Rate ~ Condition / ECG_Rate + (1 + Condition / ECG_Rate | Participant), data=df)
summary(model)

dfsub <- df |>
  group_by(Participant, Condition) |>
  summarise(Rate_Ratio = mean(Rate_Ratio),
            Rate_Harmonic_Distance = abs(mean(Rate_Harmonic_Distance)),
            Rate_Correlation_Pearson = cor(ECG_Rate, Tapping_Rate),
            Rate_Correlation_Spearman = cor(ECG_Rate, Tapping_Rate, method="spearman"))

summary(lm(Rate_Harmonic_Distance ~ Condition, data=dfsub))
summary(lm(Rate_Harmonic_Distance ~ Condition / Rate_Correlation_Spearman, data=dfsub))
plot(estimate_density(dfsub$Rate_Correlation_Pearson))

# Cardiac Cycle ===========================================================

# Global ------------------------------------------------------------------


# Convert control data to circular
c <- circular::circular(df$Cardiac_Angle, units = "degrees", template ="none", rotation="clock")
plot(c, stack = TRUE)
arrows.circular(mean(c)) #add mean to plot
rayleigh.test(c)

# Per Condition -----------------------------------------------------------

cond_density <- data.frame()
cond_hist <- data.frame()
per_cond <- data.frame()
for(cond in unique(df$Condition)){
  out <- df |>
    filter(Condition == cond) |>
    pull(Cardiac_Angle) |>
    circular_parameters()

  cond_density <- out$density |>
    mutate(Condition = cond) |>
    rbind(cond_density)

  cond_hist <- out$hist |>
    mutate(Condition = cond) |>
    rbind(cond_hist)

  per_cond <- data.frame(Condition = cond) |>
    cbind(out$test) |>
    rbind(per_cond)
}

ggplot(df, aes(x = Cardiac_Angle)) +
  geom_bar(data=cond_hist, aes(x=Angle, y=Count, fill=Condition), width=5, stat="identity", color="black") +
  geom_line(data=cond_density, aes(x=x, y=y), color="black") +
  facet_wrap(~Condition) +
  theme_custom

ggplot(per_cond, aes(x=Rayleigh_p, fill=Condition)) +
  geom_histogram(bins=20, position="dodge") +
  geom_vline(xintercept = 0.05, linetype="dashed")

# Per Subject -------------------------------------------------------------


sub_density <- data.frame()
per_sub <- data.frame()
for(sub in unique(df$Participant)){
  for(cond in unique(filter(df, Participant == sub)$Condition)){
    out <- df |>
      filter(Condition == cond, Participant == sub) |>
      pull(Cardiac_Angle) |>
      circular_parameters()

    sub_density <- out$density |>
      mutate(Condition = cond,
             Participant=sub) |>
      rbind(sub_density)

    per_sub <- data.frame(Participant = sub,
                       Condition = cond) |>
      cbind(out$test) |>
      rbind(per_sub)
  }
}

ggplot(df, aes(x = Cardiac_Angle)) +
  geom_line(data=sub_density, aes(x=x, y=y, color=Participant)) +
  facet_wrap(~Condition) +
  theme_custom

ggplot(per_sub, aes(x=Rayleigh_p, fill=Condition)) +
  geom_histogram(bins=20, position="dodge") +
  geom_vline(xintercept = 0.05, linetype="dashed") +
  facet_wrap(~Condition)


# Modelling ---------------------------------------------------------------

# df$Radian <- deg2rad(df$Cardiac_Angle)
#
# f <- bf(Radian ~ 0 + Condition,
#         kappa ~ 0 + Condition,
#         family = von_mises())
#
# model <- brm(
#   f,
#   data = df,
#   # refresh = 0,
#   iter = 1000)




# Respiratory Cycle ===========================================================

# Global ------------------------------------------------------------------


# Convert control data to circular
c <- circular::circular(df$RSP_Angle, units = "degrees", template ="none", rotation="clock")
plot(c, stack = TRUE)
arrows.circular(mean(c)) #add mean to plot
rayleigh.test(c)

# Per Condition -----------------------------------------------------------
mean(abs(df$Closest_RSP_Trough_Pre) + abs(df$Closest_RSP_Peak_Post), na.rm=TRUE)
mean(abs(df$Closest_RSP_Peak_Pre) + abs(df$Closest_RSP_Trough_Post), na.rm=TRUE)

cond_density <- data.frame()
cond_hist <- data.frame()
per_cond <- data.frame()
for(cond in unique(df$Condition)){
  out <- df |>
    filter(Condition == cond) |>
    pull(RSP_Angle) |>
    circular_parameters()

  cond_density <- out$density |>
    mutate(Condition = cond) |>
    rbind(cond_density)

  cond_hist <- out$hist |>
    mutate(Condition = cond) |>
    rbind(cond_hist)

  per_cond <- data.frame(Condition = cond) |>
    cbind(out$test) |>
    rbind(per_cond)
}

ggplot(df, aes(x = Angle)) +
  geom_bar(data=cond_hist, aes(x=Angle, y=Count, fill=Condition), width=10, stat="identity", color="black") +
  geom_line(data=cond_density, aes(x=x, y=y), color="black") +
  facet_wrap(~Condition) +
  theme_custom

ggplot(per_cond, aes(x=Rayleigh_p, fill=Condition)) +
  geom_histogram(bins=20, position="dodge") +
  geom_vline(xintercept = 0.05, linetype="dashed")
