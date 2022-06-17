library(gstat)
library(sp)
library(readxl)

# Load the data
df <- read_excel('wolfcamp.xlsx')

coordinates(df) = ~X + Y
# max distance between points to consider
cutoff = 40 # I just arbitrary chose 20 km
# width of each interval - I just arbitrarily divided the max length by 15
width = cutoff / 10
# The head valye
df$z =df$`Data`
# Find experimental variograms (this "df$z ~ 1" means use ordinary kriging)
v = variogram(df$z ~ 1, data = df, width=width, cutoff=cutoff)
v
plot(v)
# Variogram
fitted = fit.variogram(v, vgm(295, "Gau", 10000, 5))
plot(v, fitted)
print(fitted)
g <- gstat(NULL, id = "data", formula = z ~ 1, data = df, model = fitted)


# do cross-validation to remove an observation, predict at that location and repeat for each observation
out = gstat.cv(g, remove.all=TRUE)
print("Mean error (m)")
print(mean(out$residual))
print("Mean squared error (m)")
print(mean(out$residual^2))
print("Mean standardised squared error (m) - i.e., mean(squared error / prediction variance)")
print(mean((out$residual^2) / out$data.var))