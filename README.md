# Reconstructing paleo lightning, fire, and burned area in Alaska

Uden, Beckage, Clemins 2026

Code for final chapter of my dissertation. 

Pipeline: 

drivers -> strike rate -> strike count

drivers -> ignition efficiency | 1 strike

drivers -> burned area | 1 ignition

strike count + ignition efficiency -> fire count

fire count + burned area | 1 ignition -> total burned area

------LIGHTNING STRIKE RATE------

r_strike ~ Gamma(alpha, beta),

alpha = exp(a_alpha + b_alpha * X),

beta = exp(a_beta + b_beta * X),

a_alpha, b_alpha, a_beta, b_beta ~ Normal(0, 1).


r_strike: lightning strike rate / km2 / month
X: temperature, precipitation, relative humidity, shortwave radiation, surface pressure
exp(): ensures alpha and beta stay positive, which the gamma distribution requires. 


------LIGHTNING STRIKE COUNT------

mu = r_strike * area

n_strikes ~ NegBinomial(mu, theta)

theta = exp(gamma_0 + gamma * X),

Gamma_0 ~ Normal(2,2)

Gamma ~ Normal(0, sigma2)

Sigma = 1 


r_strike: strikes/km2/month

area: gridcell area, km2

n_strikes: number of strikes in a gridcell-month

X: monthly climate predictors, temperature, precipitation, wind, relative humidity, shortwave radiation, surface pressure

theta: Precision parameter of the Negative Binomial distribution predicted from climate. Theta > 0.

mu: mean lightning strike rate, strikes/gridcell/month. This is predicted from the Gamma model above and is the same for all summer months (June, July, August) in a year. The summer mean was used to ensure that mu > 0, which is required by the negative binomial distribution. 


------IGNITION EFFICIENCY, P(ignition | 1 strike)------

Y ~ Binomial(n_strikes, p_ignite) —> technically, this is applied to estimate fire count.

logit(p_ignite) = alpha + X * beta

OR

p_ignite = plogis(alpha + X * beta)

alpha ~ normal(mu_alpha, sigma_alpha)

beta  ~ normal(0, sigma_beta)

p_ignite_corrected = plogis(qlogis(pie) + cal_shift)



p_ignite: probability of 1 strike resulting in an ignition in a given gridcell and month. 

X: average precipitation rate, temperature, relative humidity, cumulative precipitation for the previous 1, 2, 3 months, and 5 years, mean temperature for the previous 1, 2, 3 months, and 5 years, and % broadleaf cover. 


------BURNED AREA | 1 strike------

Burned_area ~ Gamma(theta, theta/mu)

log(mu) = alpha + alpha_year + X * beta

alpha ~ Normal(0, sigma^2_alpha)

beta ~ Normal(0, sigma^2_beta)

theta ~ Exponential(1)

alpha_year ~ Normal(0, sigma^2_year)

sigma^2_year ~ Exponential(1)


Burned_area: area of 1 lightning-caused ignition (km2)

X: Relative humidity, temperature, precipitation, wind, cumulative precipitation for the previous 1, 3 months, and 5 years, mean temperature for the previous 2 months and 5 years, proximity to urban area (binary; 1 = within 40 km of an urban area), proximity to road (binary; 1 = within 40 km of any major road)

alpha_year: year random affect


------FIRE COUNT------

n_fires = Binomial(n_strikes, p_ignite)


N_fires: number of fires in a gridcell-month

N_strikes: strikes/gridcell/month

P_ignite: P(ignition | 1 strike)


------TOTAL BURNED AREA | fire count------

total_burned_area = rgamma(n_fires, burned_area_shape, burned_area_rate)


Total_burned_area: burned are in a gridcell | fire count and gamma shape and rate parameters predicted from climate

Burned_area_shape: estimated from burned area | 1 ignition 

burned_area_rate: estimated from burned area | 1 ignition




Note: Configuration files can either be ignored, or set your own file paths for

drivers_root = /../data/paleo_fire/training_data

rds_root = /../data/paleo_fire/rds

table_root = /../data/paleo_fire/model_performance

predictions_root = /../data/paleo_fire/model_predictions

mu_sigma_root = /../data/paleo_fire/mu_sigma

performance_root = /../data/paleo_fire/model_performance

figure_root = /../Paleo_Lightning_Fire/Figures

fire_data_root = /../data/paleo_fire/ABoVE

paleocliamte_data_root = /../data/paleo_fire/TraCE-21ka

era5_data_root = /../data/paleo_fire/ERA5

proximity_data_root = /../data/paleo_fire/proximity_data

Data can be downloaded from Zenodo.
